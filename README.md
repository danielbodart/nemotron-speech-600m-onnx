# nemotron-speech-600m-onnx

Scripts to export, convert, quantize, and validate the [NVIDIA Nemotron Speech Streaming 600M](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) model for ONNX Runtime inference.

Pre-built model files are available on Hugging Face Hub: **[danielbodart/nemotron-speech-600m-onnx](https://huggingface.co/danielbodart/nemotron-speech-600m-onnx)**

## Why this exists

The original NeMo model requires the full NeMo toolkit to run. Third-party ONNX conversions we found had subtle issues — wrong cache shapes, incorrect mel preprocessing, or quantization that broke streaming accuracy. This repo provides a verified pipeline that exports directly from the original NeMo checkpoint and validates each conversion step numerically.

## Available Precisions

| Variant | Size | Target Hardware | Execution Provider | Notes |
|---------|------|-----------------|--------------------|-------|
| FP32 | 2.4 GB | Any | CPU, CUDA | Original precision, exported directly from NeMo |
| FP16 | 1.2 GB | NVIDIA GPU, Apple Silicon | CPU, CUDA | Recommended for GPU inference |
| INT8 Dynamic | 876 MB | Intel CPU (VNNI/AMX) | CPU only | Dynamic quantization, MatMul weights only |
| INT8 Static | 876 MB | NVIDIA GPU, Intel CPU | CPU, CUDA | QDQ format with warm-cache calibration. ~45% less VRAM than FP16 on GPU |

### Choosing a variant

- **NVIDIA GPU (memory constrained):** INT8 Static — loads on CUDA EP, uses ~1.3 GB VRAM vs ~2.4 GB for FP16
- **NVIDIA GPU (quality first):** FP16 — marginally better on edge cases with repetitive content
- **Intel CPU:** INT8 Dynamic — best CPU throughput via VNNI/AMX integer instructions
- **Apple Silicon:** FP16 — optimized for Neural Engine

## Project Layout

```
nemotron-speech-600m-onnx/
├── config.json                  # Machine-readable runtime parameters
├── nemo_export_onnx.py          # Export from NeMo checkpoint → models/fp32/
├── onnx_fp16_convert.py         # models/fp32/ → models/fp16/
├── onnx_fp16_validate.py        # Validate FP16 against FP32
├── onnx_int8_quantize.py        # models/fp32/ → models/int8-{dynamic,static}/
├── onnx_int8_calibration.py     # Warm-cache mel calibration data reader
├── onnx_int8_validate.py        # Validate INT8 against FP32
├── onnx_package_for_hf.py       # Consolidate and package for HF Hub upload
├── models/                      # Local model files (gitignored)
│   ├── fp32/                    # NeMo export output
│   ├── fp16/                    # FP16 conversion output
│   ├── int8-dynamic/            # Dynamic INT8 quantization output
│   └── int8-static/             # Static INT8 quantization output
└── test/                        # WAV files for calibration/validation (gitignored)
```

## Scripts

### 1. Export from NeMo

```bash
uv run --with "nemo_toolkit[asr]" python3 nemo_export_onnx.py [output-dir]
```

Downloads the pretrained model from HuggingFace, exports encoder + decoder to ONNX, and saves the filterbank weights, vocabulary, and preprocessor config. Requires a CUDA GPU. Default output: `models/fp32/`.

### 2. FP16 Conversion

```bash
uv run --with "onnx>=1.20,numpy" python3 onnx_fp16_convert.py
```

Converts FP32 models to FP16. All weights, constants, and Cast nodes are converted. I/O is wrapped with Cast nodes so the external interface stays FP32 (no calling code changes needed). Reads from `models/fp32/`, writes to `models/fp16/`.

### 3. FP16 Validation

```bash
uv run --with "onnx>=1.20,onnxruntime>=1.24,numpy" python3 onnx_fp16_validate.py
```

Runs both FP32 and FP16 models on identical synthetic input (CPU, deterministic) and compares outputs numerically. Encoder max abs diff < 0.001, decoder argmax agreement > 99%.

### 4. INT8 Quantization

```bash
# Dynamic (CPU-only, no calibration needed)
uv run --with "onnx>=1.20,onnxruntime>=1.24,numpy" python3 onnx_int8_quantize.py --mode dynamic

# Static (CUDA-compatible, requires WAV files in test/ for calibration)
uv run --with "onnx>=1.20,onnxruntime>=1.24,numpy" python3 onnx_int8_quantize.py --mode static
```

Both modes quantize encoder MatMul weights to INT8. The decoder stays FP32 (too small to benefit).

**Dynamic** uses `quantize_dynamic` — weights are INT8, activations computed at runtime in FP32. Output uses `MatMulInteger` ops which only run on the CPU execution provider.

**Static** uses `quantize_static` with warm-cache streaming calibration — both weights and activations are INT8 with pre-computed scale factors. Output uses QDQ (`QuantizeLinear`/`DequantizeLinear`) nodes which load on both CPU and CUDA execution providers. The warm-cache calibration runs audio chunks sequentially through the FP32 model, carrying encoder cache state forward, to capture realistic activation ranges for streaming inference.

### 5. INT8 Validation

```bash
uv run --with "onnx>=1.20,onnxruntime>=1.24,numpy" python3 onnx_int8_validate.py --mode static
uv run --with "onnx>=1.20,onnxruntime>=1.24,numpy" python3 onnx_int8_validate.py --mode dynamic
```

Tests both single-chunk numerical agreement and multi-chunk streaming behavior with warm caches. The streaming test catches quantization errors that compound across chunks.

### 6. Package for Hugging Face Hub

```bash
uv run --with "onnx>=1.20,numpy" python3 onnx_package_for_hf.py
```

Consolidates per-tensor external data files into single `.onnx.data` files, renames to HF conventions, and organizes into `hf-upload/` ready for `hf upload`.

## Runtime Configuration

All parameters needed to run the model are in [`config.json`](config.json). See the [model card](https://huggingface.co/danielbodart/nemotron-speech-600m-onnx) for full documentation of encoder cache shapes, decoder parameters, and the streaming protocol.

## License

The scripts in this repo are MIT licensed. The model weights (on Hugging Face Hub) are [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) per NVIDIA's original license.
