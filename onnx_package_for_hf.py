#!/usr/bin/env python3
"""Package Nemotron ONNX models for Hugging Face Hub upload.

Consolidates per-tensor external data files into single .onnx.data files,
renames to HF conventions, and organizes into hf-upload/ directory.

Output: hf-upload/ ready for `hf upload`
"""

import shutil
import sys
from pathlib import Path

import onnx

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "hf-upload"

VARIANTS = {
    "fp32": PROJECT_DIR / "models" / "fp32",
    "fp16": PROJECT_DIR / "models" / "fp16",
    "int8": PROJECT_DIR / "models" / "int8",
}

# Maps source filenames to HF-convention names
MODEL_RENAMES = {
    "encoder-model-streaming.onnx": "encoder_model.onnx",
    "decoder_joint-model-streaming.onnx": "decoder_model.onnx",
}

SHARED_FILES = [
    "filterbank.bin",
    "filterbank.meta",
    "tokens.txt",
    "preprocessor.config",
]


def consolidate_model(src_path: Path, dst_dir: Path, dst_name: str) -> None:
    """Load model with external data, save as consolidated .onnx + .onnx.data pair."""
    print(f"  Consolidating {src_path.name} → {dst_name}...")
    model = onnx.load(str(src_path), load_external_data=True)

    dst_path = dst_dir / dst_name
    data_name = dst_name + ".data"

    onnx.save(
        model,
        str(dst_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=0,  # externalize all tensors
    )

    # Report sizes
    graph_size = dst_path.stat().st_size
    data_path = dst_dir / data_name
    data_size = data_path.stat().st_size
    print(f"    {dst_name}: {graph_size / 1024:.0f} KB")
    print(f"    {data_name}: {data_size / 1024 / 1024:.1f} MB")


def main() -> None:
    # Clean output
    if OUTPUT_DIR.exists():
        shutil.rmtree(str(OUTPUT_DIR))

    # Process each variant
    for variant_name, src_dir in VARIANTS.items():
        if not src_dir.exists():
            print(f"SKIP: {variant_name} not found at {src_dir}")
            continue

        print(f"\n=== {variant_name} ===")
        dst_dir = OUTPUT_DIR / variant_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        for src_name, dst_name in MODEL_RENAMES.items():
            src_path = src_dir / src_name
            if not src_path.exists():
                print(f"  SKIP: {src_name} not found")
                continue
            consolidate_model(src_path, dst_dir, dst_name)

    # Copy shared files (from fp32 source, they're identical)
    print(f"\n=== shared ===")
    shared_dir = OUTPUT_DIR / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    src_dir = VARIANTS["fp32"]
    for name in SHARED_FILES:
        src = src_dir / name
        if src.exists():
            shutil.copy2(str(src), str(shared_dir / name))
            print(f"  Copied {name} ({src.stat().st_size / 1024:.0f} KB)")

    # Total size summary
    print(f"\n=== Summary ===")
    for subdir in sorted(OUTPUT_DIR.iterdir()):
        if subdir.is_dir():
            total = sum(f.stat().st_size for f in subdir.rglob("*") if f.is_file())
            print(f"  {subdir.name}/: {total / 1024 / 1024:.1f} MB")

    total = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())
    print(f"  TOTAL: {total / 1024 / 1024 / 1024:.2f} GB")


if __name__ == "__main__":
    main()
