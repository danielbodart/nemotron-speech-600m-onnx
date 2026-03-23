#!/usr/bin/env python3
"""Quantize Nemotron encoder to INT8 using ONNX Runtime quantization.

Reads the FP32 encoder from models/fp32/, quantizes, and writes to
models/int8-dynamic/ or models/int8-static/ depending on mode.

Two modes are available:

  dynamic — Weights quantized to INT8, activations stay FP32. No calibration
            needed. Best for Intel CPUs (VNNI/AMX). Uses MatMulInteger ops
            which only run on CPU execution provider.

  static  — Weights and activations quantized to INT8 via QDQ nodes with
            calibrated activation scales. Uses warm-cache streaming calibration
            to avoid the activation range errors that break naive static
            quantization of streaming models. QDQ format loads on both CPU
            and CUDA execution providers, saving ~45% VRAM vs FP16 on GPU.

The decoder stays unquantized (copied from FP32 source).
External I/O stays FP32 — no runtime changes needed.

Usage:
    python onnx_int8_quantize.py                    # dynamic (default)
    python onnx_int8_quantize.py --mode static      # static (QDQ, CUDA-compatible)
    python onnx_int8_quantize.py --reduce-range      # for older CPUs without VNNI
"""

import argparse
import shutil
import sys
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic, quantize_static

from onnx_int8_calibration import MelCalibrationReader

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "models" / "fp32"
WAV_DIR = PROJECT_DIR / "test"
FILTERBANK_PATH = SRC_DIR / "filterbank.bin"

ENCODER_MODEL = "encoder-model-streaming.onnx"

# Decoder and support files copied without modification
COPY_MODELS = [
    "decoder_joint-model.onnx",
    "decoder_joint-model-streaming.onnx",
]
COPY_FILES = [
    "filterbank.bin",
    "filterbank.meta",
    "tokens.txt",
    "preprocessor.config",
]


def copy_model_with_external_data(src_path: Path, dst_path: Path) -> None:
    """Copy an ONNX model and all its external data files."""
    # Load with external data to resolve all references
    model = onnx.load(str(src_path), load_external_data=True)
    # Save to destination with external data
    onnx.save(
        model,
        str(dst_path),
        save_as_external_data=True,
        all_tensors_to_one_file=False,
        size_threshold=1024,
    )


def quantize_encoder(dst_dir: Path, mode: str = "dynamic", reduce_range: bool = False, max_samples: int = 500) -> None:
    """Quantize the streaming encoder to INT8."""
    src = SRC_DIR / ENCODER_MODEL
    dst = dst_dir / ENCODER_MODEL

    if not src.exists():
        print(f"ERROR: Source encoder not found: {src}", file=sys.stderr)
        sys.exit(1)

    print(f"Quantizing {ENCODER_MODEL}...")
    print(f"  Source: {src}")
    print(f"  Destination: {dst}")
    print(f"  mode: {mode}")
    print(f"  reduce_range: {reduce_range}")

    if mode == "dynamic":
        print(f"  Running dynamic quantization (S8 weights only, MatMul ops, activations stay FP32)...")
        quantize_dynamic(
            model_input=str(src),
            model_output=str(dst),
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=reduce_range,
            use_external_data_format=True,
            op_types_to_quantize=["MatMul"],
        )
    else:
        # Build calibration reader with warm caches from FP32 model.
        # 6 warmup chunks (~3.4s) ensures caches are fully representative
        # before collecting calibration statistics.
        reader = MelCalibrationReader(
            wav_dir=WAV_DIR,
            filterbank_path=FILTERBANK_PATH,
            fp32_model_path=src,
            max_samples=max_samples,
            warmup_chunks=6,
        )

        print(f"  Running static quantization (U8 activations, S8 weights, per-channel, QDQ)...")
        print(f"  Only quantizing MatMul ops (like dynamic, but with calibrated activation scales)")
        quantize_static(
            model_input=str(src),
            model_output=str(dst),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=reduce_range,
            use_external_data_format=True,
            op_types_to_quantize=["MatMul"],
            extra_options={
                "WeightSymmetric": True,
                "ActivationSymmetric": False,
                "MinimumRealRange": 0.0001,
                "CalibMovingAverage": True,
                "CalibMovingAverageConstant": 0.01,
            },
        )

    # Report quantized model info
    quantized_model = onnx.load(str(dst), load_external_data=False)
    qdq_count = sum(
        1 for node in quantized_model.graph.node
        if node.op_type in ("QuantizeLinear", "DequantizeLinear")
    )
    total_nodes = len(quantized_model.graph.node)
    print(f"  Quantized: {qdq_count} Q/DQ nodes out of {total_nodes} total nodes")
    print(f"  Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize Nemotron encoder to INT8")
    parser.add_argument(
        "--mode",
        choices=["dynamic", "static"],
        default="dynamic",
        help="Quantization mode: 'dynamic' (weights only, CPU) or 'static' (weights+activations QDQ, CPU+CUDA)",
    )
    parser.add_argument(
        "--reduce-range",
        action="store_true",
        help="Use reduced INT8 range [-64,63] for older CPUs without VNNI (pre-Cascade Lake)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum number of calibration samples for static mode (default: 500)",
    )
    args = parser.parse_args()

    if not SRC_DIR.exists():
        print(f"ERROR: Source directory not found: {SRC_DIR}", file=sys.stderr)
        sys.exit(1)

    dst_dir = PROJECT_DIR / "models" / f"int8-{args.mode}"
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Quantize encoder
    quantize_encoder(dst_dir, mode=args.mode, reduce_range=args.reduce_range, max_samples=args.max_samples)

    # Copy decoder models with their external data
    for name in COPY_MODELS:
        src = SRC_DIR / name
        dst = dst_dir / name
        if not src.exists():
            print(f"SKIP: {name} not found in source")
            continue
        print(f"Copying {name} (with external data)...")
        copy_model_with_external_data(src, dst)

    # Copy support files
    for name in COPY_FILES:
        src = SRC_DIR / name
        dst = dst_dir / name
        if src.exists():
            shutil.copy2(str(src), str(dst))
            print(f"Copied {name}")

    # Size comparison
    src_size = sum(f.stat().st_size for f in SRC_DIR.rglob("*") if f.is_file())
    dst_size = sum(f.stat().st_size for f in dst_dir.rglob("*") if f.is_file())
    print(f"\nFP32 total:  {src_size / 1024**3:.2f} GB")
    print(f"INT8 total:  {dst_size / 1024**3:.2f} GB")
    print(f"Reduction:   {(1 - dst_size / src_size) * 100:.1f}%")


if __name__ == "__main__":
    main()
