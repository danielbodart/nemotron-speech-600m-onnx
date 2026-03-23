#!/usr/bin/env python3
"""Validate INT8 quantized encoder by comparing outputs against FP32 original.

Tests both single-chunk numerical agreement and multi-chunk streaming behavior
with warm caches. The streaming test is critical because static quantization
can produce models that pass single-chunk tests but fail when caches accumulate
quantization error across chunks.
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from onnx_int8_calibration import (
    CACHE_CH_DIM,
    CACHE_TIME_DIM,
    ENC_DIM,
    ENC_LAYERS,
    MEL_SHIFT,
    N_MELS,
    PRE_ENCODE_CACHE,
    TOTAL_CHUNK_FRAMES,
    compute_mel_bandmajor,
    load_filterbank,
    load_wav_as_f32,
)

PROJECT_DIR = Path(__file__).resolve().parent
FP32_DIR = PROJECT_DIR / "models" / "fp32"
WAV_DIR = PROJECT_DIR / "test"

# Single-chunk numerical thresholds
MAX_ABS_DIFF = 50.0
MEAN_ABS_DIFF = 1.0

# Streaming: minimum fraction of non-blank tokens that must agree
STREAMING_TOKEN_AGREEMENT = 0.80


def create_session(model_path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def make_encoder_inputs(batch: int = 1) -> dict:
    mel_frames = MEL_SHIFT + PRE_ENCODE_CACHE
    return {
        "audio_signal": np.random.randn(batch, N_MELS, mel_frames).astype(np.float32),
        "length": np.array([mel_frames], dtype=np.int64),
        "cache_last_channel": np.zeros((batch, ENC_LAYERS, CACHE_CH_DIM, ENC_DIM), dtype=np.float32),
        "cache_last_time": np.zeros((batch, ENC_LAYERS, ENC_DIM, CACHE_TIME_DIM), dtype=np.float32),
        "cache_last_channel_len": np.zeros((batch,), dtype=np.int64),
    }


def compare_outputs(name: str, fp32_outs: list, int8_outs: list) -> bool:
    passed = True
    for i, (o32, o8) in enumerate(zip(fp32_outs, int8_outs)):
        if o32.dtype != o8.dtype:
            o32 = o32.astype(np.float32)
            o8 = o8.astype(np.float32)

        abs_diff = np.abs(o32 - o8)
        max_diff = abs_diff.max()
        mean_diff = abs_diff.mean()

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


def validate_encoder(fp32_path: Path, int8_path: Path, inputs: dict) -> bool:
    print(f"\n=== Encoder: single-chunk numerical comparison ===")

    if not fp32_path.exists():
        print(f"  SKIP: {fp32_path} not found")
        return True
    if not int8_path.exists():
        print(f"  FAIL: {int8_path} not found")
        return False

    print(f"  Loading FP32...")
    sess32 = create_session(fp32_path)
    print(f"  Loading INT8...")
    sess8 = create_session(int8_path)

    print(f"  Running inference...")
    out32 = sess32.run(None, inputs)
    out8 = sess8.run(None, inputs)

    return compare_outputs("encoder", out32, out8)


def validate_streaming(int8_dir: Path, wav_name: str = "jfk.wav") -> bool:
    """Streaming validation: feed real audio chunk-by-chunk with warm caches.

    Runs both FP32 and INT8 encoders sequentially, feeding output caches back
    as input to the next chunk. Then runs the decoder on each encoder frame
    and compares token output.

    This catches quantization errors that compound across chunks — the failure
    mode that broke static quantization.
    """
    print(f"\n=== Streaming validation ({wav_name}) ===")

    wav_path = WAV_DIR / wav_name
    if not wav_path.exists():
        print(f"  SKIP: {wav_path} not found")
        return True

    filterbank_path = FP32_DIR / "filterbank.bin"
    if not filterbank_path.exists():
        print(f"  SKIP: {filterbank_path} not found")
        return True

    # Load models
    fp32_enc = create_session(FP32_DIR / "encoder-model-streaming.onnx")
    int8_enc = create_session(int8_dir / "encoder-model-streaming.onnx")
    decoder = create_session(FP32_DIR / "decoder_joint-model-streaming.onnx")

    # Compute mel features from real audio
    audio = load_wav_as_f32(wav_path)
    filterbank = load_filterbank(filterbank_path)
    mel = compute_mel_bandmajor(audio, filterbank)  # [128, n_frames]
    n_frames = mel.shape[1]

    print(f"  Audio: {len(audio)/16000:.1f}s, {n_frames} mel frames")

    # Initialize caches
    fp32_cache_ch = np.zeros((1, ENC_LAYERS, CACHE_CH_DIM, ENC_DIM), dtype=np.float32)
    fp32_cache_time = np.zeros((1, ENC_LAYERS, ENC_DIM, CACHE_TIME_DIM), dtype=np.float32)
    fp32_cache_ch_len = np.zeros((1,), dtype=np.int64)

    int8_cache_ch = np.zeros_like(fp32_cache_ch)
    int8_cache_time = np.zeros_like(fp32_cache_time)
    int8_cache_ch_len = np.zeros_like(fp32_cache_ch_len)

    # Pre-encode cache (first PRE_ENCODE_CACHE frames, initially zeros)
    fp32_pre_cache = np.zeros((N_MELS, PRE_ENCODE_CACHE), dtype=np.float32)
    int8_pre_cache = np.zeros_like(fp32_pre_cache)

    BLANK_ID = 1024
    VOCAB_SIZE = 1024

    fp32_tokens = []
    int8_tokens = []
    n_chunks = 0

    # Process chunks
    cursor = 0
    while cursor + MEL_SHIFT <= n_frames:
        # Build chunk: pre_cache + new frames
        fp32_chunk = np.concatenate([fp32_pre_cache, mel[:, cursor:cursor+MEL_SHIFT]], axis=1)
        int8_chunk = np.concatenate([int8_pre_cache, mel[:, cursor:cursor+MEL_SHIFT]], axis=1)

        chunk_frames = fp32_chunk.shape[1]

        # Run encoder
        fp32_inputs = {
            "audio_signal": fp32_chunk[np.newaxis, :, :],
            "length": np.array([chunk_frames], dtype=np.int64),
            "cache_last_channel": fp32_cache_ch,
            "cache_last_time": fp32_cache_time,
            "cache_last_channel_len": fp32_cache_ch_len,
        }
        int8_inputs = {
            "audio_signal": int8_chunk[np.newaxis, :, :],
            "length": np.array([chunk_frames], dtype=np.int64),
            "cache_last_channel": int8_cache_ch,
            "cache_last_time": int8_cache_time,
            "cache_last_channel_len": int8_cache_ch_len,
        }

        fp32_out = fp32_enc.run(None, fp32_inputs)
        int8_out = int8_enc.run(None, int8_inputs)

        # Update caches
        fp32_cache_ch = fp32_out[2]
        fp32_cache_time = fp32_out[3]
        fp32_cache_ch_len = fp32_out[4]

        int8_cache_ch = int8_out[2]
        int8_cache_time = int8_out[3]
        int8_cache_ch_len = int8_out[4]

        # Update pre-encode cache (last PRE_ENCODE_CACHE frames of this chunk's new audio)
        new_mel = mel[:, cursor:cursor+MEL_SHIFT]
        if new_mel.shape[1] >= PRE_ENCODE_CACHE:
            fp32_pre_cache = new_mel[:, -PRE_ENCODE_CACHE:]
            int8_pre_cache = new_mel[:, -PRE_ENCODE_CACHE:]
        else:
            fp32_pre_cache = new_mel
            int8_pre_cache = new_mel

        # Decode tokens from encoder output
        fp32_encoded = fp32_out[0]  # [1, ENC_DIM, T_out]
        int8_encoded = int8_out[0]
        enc_len = int(fp32_out[1][0])

        for t in range(enc_len):
            # Run decoder on each encoder frame
            dec_input_fp32 = {
                "encoder_outputs": fp32_encoded[:, :, t:t+1].astype(np.float32),
                "targets": np.array([[0]], dtype=np.int32),
                "target_length": np.array([1], dtype=np.int32),
                "input_states_1": np.zeros((2, 1, 640), dtype=np.float32),
                "input_states_2": np.zeros((2, 1, 640), dtype=np.float32),
            }
            dec_input_int8 = {
                "encoder_outputs": int8_encoded[:, :, t:t+1].astype(np.float32),
                "targets": np.array([[0]], dtype=np.int32),
                "target_length": np.array([1], dtype=np.int32),
                "input_states_1": np.zeros((2, 1, 640), dtype=np.float32),
                "input_states_2": np.zeros((2, 1, 640), dtype=np.float32),
            }

            fp32_logits = decoder.run(None, dec_input_fp32)[0].flatten()
            int8_logits = decoder.run(None, dec_input_int8)[0].flatten()

            fp32_tok = int(np.argmax(fp32_logits[:VOCAB_SIZE + 1]))
            int8_tok = int(np.argmax(int8_logits[:VOCAB_SIZE + 1]))

            fp32_tokens.append(fp32_tok)
            int8_tokens.append(int8_tok)

        cursor += MEL_SHIFT
        n_chunks += 1

    # Analyze results
    fp32_nonblank = sum(1 for t in fp32_tokens if t != BLANK_ID)
    int8_nonblank = sum(1 for t in int8_tokens if t != BLANK_ID)
    agree = sum(1 for a, b in zip(fp32_tokens, int8_tokens) if a == b)
    total = len(fp32_tokens)

    # Check that INT8 produces non-blank tokens (the static quant failure mode)
    int8_has_speech = int8_nonblank > 0

    # Check token agreement rate
    agree_rate = agree / total if total else 0

    print(f"  Chunks: {n_chunks}")
    print(f"  Total decoder frames: {total}")
    print(f"  FP32 non-blank tokens: {fp32_nonblank}")
    print(f"  INT8 non-blank tokens: {int8_nonblank}")
    print(f"  Token agreement: {agree}/{total} ({agree_rate:.1%})")

    speech_ok = int8_has_speech
    agree_ok = agree_rate >= STREAMING_TOKEN_AGREEMENT

    print(f"  INT8 produces speech: [{'PASS' if speech_ok else 'FAIL'}]")
    print(f"  Token agreement: [{'PASS' if agree_ok else 'FAIL'}] (threshold: {STREAMING_TOKEN_AGREEMENT:.0%})")

    return speech_ok and agree_ok


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Validate INT8 quantized encoder against FP32")
    parser.add_argument(
        "--mode",
        choices=["dynamic", "static"],
        default="static",
        help="Which INT8 variant to validate (default: static)",
    )
    args = parser.parse_args()

    int8_dir = PROJECT_DIR / "models" / f"int8-{args.mode}"

    np.random.seed(42)

    all_passed = True

    # Single-chunk numerical comparison
    enc_inputs = make_encoder_inputs()
    ok = validate_encoder(
        FP32_DIR / "encoder-model-streaming.onnx",
        int8_dir / "encoder-model-streaming.onnx",
        enc_inputs,
    )
    all_passed &= ok

    # Streaming validation with real audio and warm caches
    ok = validate_streaming(int8_dir, "jfk.wav")
    all_passed &= ok

    print(f"\n{'=' * 40}")
    if all_passed:
        print("ALL VALIDATIONS PASSED")
    else:
        print("SOME VALIDATIONS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
