# nemotron-speech-600m-onnx

Scripts to export, convert, quantize, and validate the [NVIDIA Nemotron Speech Streaming 600M](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) model for ONNX Runtime inference.

Pre-built model files are available on Hugging Face Hub: **[danielbodart/nemotron-speech-600m-onnx](https://huggingface.co/danielbodart/nemotron-speech-600m-onnx)**

## Why this exists

The original NeMo model requires the full NeMo toolkit to run. Third-party ONNX conversions we found had subtle issues — wrong cache shapes, incorrect mel preprocessing, or quantization that broke streaming accuracy. This repo provides a verified pipeline that exports directly from the original NeMo checkpoint and validates each conversion step numerically.

## Available Precisions

| Variant | Size | Target Hardware | Notes |
|---------|------|-----------------|-------|
| FP32 | 2.4 GB | Any | Original precision, exported directly from NeMo |
| FP16 | 1.2 GB | NVIDIA GPU (tensor cores), Apple Silicon | Recommended for GPU inference |
| INT8 | 876 MB | Intel CPU (VNNI/AMX) | Dynamic quantization, encoder MatMul weights only |

## Project Layout

```
nemotron-speech-600m-onnx/
├── config.json                  # Machine-readable runtime parameters
├── nemo_export_onnx.py          # Export from NeMo checkpoint → models/fp32/
├── onnx_fp16_convert.py         # models/fp32/ → models/fp16/
├── onnx_fp16_validate.py        # Validate FP16 against FP32
├── onnx_int8_quantize.py        # models/fp32/ → models/int8/
├── onnx_int8_calibration.py     # Mel calibration data reader for static quantization
├── onnx_int8_validate.py        # Validate INT8 against FP32
├── onnx_package_for_hf.py       # Consolidate and package for HF Hub upload
├── models/                      # Local model files (gitignored)
│   ├── fp32/                    # NeMo export output
│   ├── fp16/                    # FP16 conversion output
│   └── int8/                    # INT8 quantization output
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
uv run --with "onnx>=1.20,onnxruntime>=1.24,numpy" python3 onnx_int8_quantize.py
```

Applies dynamic INT8 quantization to encoder MatMul weights via `onnxruntime.quantization.quantize_dynamic`. Decoder stays FP32 (too small to benefit). Reads from `models/fp32/`, writes to `models/int8/`. Note: static quantization breaks streaming — produces all BLANK tokens.

### 5. INT8 Validation

```bash
uv run --with "onnx>=1.20,onnxruntime>=1.24,numpy" python3 onnx_int8_validate.py
```

### 6. Package for Hugging Face Hub

```bash
uv run --with "onnx>=1.20,numpy" python3 onnx_package_for_hf.py
```

Consolidates per-tensor external data files into single `.onnx.data` files, renames to HF conventions, and organizes into `hf-upload/` ready for `hf upload`.

## Runtime Configuration

All parameters needed to run the model are in [`config.json`](config.json). See the [model card](https://huggingface.co/danielbodart/nemotron-speech-600m-onnx) for full documentation of encoder cache shapes, decoder parameters, and the streaming protocol.

## License

The scripts in this repo are MIT licensed. The model weights (on Hugging Face Hub) are [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) per NVIDIA's original license.
