#!/usr/bin/env python3
"""Validate FP16 conversion by comparing outputs against FP32 originals.

Runs both models on identical synthetic input (CPU for determinism)
and checks numerical closeness.
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FP32_DIR = PROJECT_DIR / "dist" / "models" / "nemotron-600m-onnx"
FP16_DIR = PROJECT_DIR / "dist" / "models" / "nemotron-600m-onnx-fp16"

# Thresholds — FP16 has ~3 decimal digits of precision.
# Logit outputs can have higher absolute diff but what matters is argmax agreement.
MAX_ABS_DIFF = 0.3    # logits are large values, absolute diff can be high
MEAN_ABS_DIFF = 0.05


def create_session(model_path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # suppress warnings
    return ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def make_encoder_inputs(batch: int = 1) -> dict:
    """Create synthetic inputs matching encoder-model-streaming.onnx.

    Shapes from nemotron_pipeline.zig:
    - MEL_SHIFT=56, PRE_ENCODE_CACHE=9 → total 65 mel frames per chunk
    - CACHE_CH_DIM=70, CACHE_TIME_DIM=8, ENC_LAYERS=24, ENC_DIM=1024
    """
    mel_frames = 56 + 9  # MEL_SHIFT + PRE_ENCODE_CACHE
    return {
        "audio_signal": np.random.randn(batch, 128, mel_frames).astype(np.float32),
        "length": np.array([mel_frames], dtype=np.int64),
        "cache_last_channel": np.zeros((batch, 24, 70, 1024), dtype=np.float32),
        "cache_last_time": np.zeros((batch, 24, 1024, 8), dtype=np.float32),
        "cache_last_channel_len": np.zeros((batch,), dtype=np.int64),
    }


def make_decoder_inputs(batch: int = 1, enc_frames: int = 8) -> dict:
    """Create synthetic inputs matching decoder_joint-model-streaming.onnx."""
    return {
        "encoder_outputs": np.random.randn(batch, 1024, enc_frames).astype(np.float32),
        "targets": np.array([[0]], dtype=np.int32),
        "target_length": np.array([1], dtype=np.int32),
        "input_states_1": np.zeros((2, batch, 640), dtype=np.float32),
        "input_states_2": np.zeros((2, batch, 640), dtype=np.float32),
    }


def compare_outputs(name: str, fp32_outs: list, fp16_outs: list) -> bool:
    passed = True
    for i, (o32, o16) in enumerate(zip(fp32_outs, fp16_outs)):
        if o32.dtype != o16.dtype:
            # Cast both to float32 for comparison
            o32 = o32.astype(np.float32)
            o16 = o16.astype(np.float32)

        abs_diff = np.abs(o32 - o16)
        max_diff = abs_diff.max()
        mean_diff = abs_diff.mean()

        # Relative error (avoid div by zero)
        denom = np.maximum(np.abs(o32), 1e-7)
        rel_err = (abs_diff / denom).mean()

        status = "PASS" if (max_diff < MAX_ABS_DIFF and mean_diff < MEAN_ABS_DIFF) else "FAIL"
        if status == "FAIL":
            passed = False

        print(f"  output[{i}] shape={o32.shape} dtype={o32.dtype}")
        print(f"    max_abs_diff:  {max_diff:.6f}  (threshold: {MAX_ABS_DIFF})")
        print(f"    mean_abs_diff: {mean_diff:.6f}  (threshold: {MEAN_ABS_DIFF})")
        print(f"    mean_rel_err:  {rel_err:.6f}")
        print(f"    [{status}]")

    return passed


def validate_model(name: str, fp32_path: Path, fp16_path: Path, inputs: dict) -> bool:
    print(f"\n=== {name} ===")

    if not fp32_path.exists():
        print(f"  SKIP: {fp32_path} not found")
        return True
    if not fp16_path.exists():
        print(f"  FAIL: {fp16_path} not found")
        return False

    print(f"  Loading FP32...")
    sess32 = create_session(fp32_path)
    print(f"  Loading FP16...")
    sess16 = create_session(fp16_path)

    print(f"  Running inference...")
    out32 = sess32.run(None, inputs)
    out16 = sess16.run(None, inputs)

    return compare_outputs(name, out32, out16)


def main() -> None:
    np.random.seed(42)

    all_passed = True

    # Encoder streaming
    enc_inputs = make_encoder_inputs()
    ok = validate_model(
        "encoder-model-streaming",
        FP32_DIR / "encoder-model-streaming.onnx",
        FP16_DIR / "encoder-model-streaming.onnx",
        enc_inputs,
    )
    all_passed &= ok

    # Decoder streaming
    dec_inputs = make_decoder_inputs()
    ok = validate_model(
        "decoder_joint-model-streaming",
        FP32_DIR / "decoder_joint-model-streaming.onnx",
        FP16_DIR / "decoder_joint-model-streaming.onnx",
        dec_inputs,
    )
    all_passed &= ok

    print(f"\n{'=' * 40}")
    if all_passed:
        print("ALL VALIDATIONS PASSED")
    else:
        print("SOME VALIDATIONS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
