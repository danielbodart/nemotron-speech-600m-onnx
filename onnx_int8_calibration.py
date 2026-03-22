#!/usr/bin/env python3
"""Calibration data reader for INT8 quantization of Nemotron encoder.

Extracts mel spectrograms from test WAV files using the exact same pipeline
as nemo_mel.zig / nemo_mel_state.zig:
  pre-emphasis(0.97) → STFT(n_fft=512, hop=160, win=400, center=True, Hann)
  → power spectrum → mel filterbank → ln(x + 2^-24)

Output: band-major [1, 128, 65] chunks matching encoder-model-streaming.onnx input.
"""

import struct
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader

# NeMo mel parameters — must match nemo_mel.zig exactly
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
N_FFT_BINS = 1 + N_FFT // 2  # 257
N_MELS = 128
SAMPLE_RATE = 16000
PREEMPH = 0.97
LOG_GUARD = 2**-24  # 5.960464477539063e-08

# Encoder streaming constants — from nemotron_pipeline.zig
MEL_SHIFT = 56
PRE_ENCODE_CACHE = 9
TOTAL_CHUNK_FRAMES = MEL_SHIFT + PRE_ENCODE_CACHE  # 65
ENC_LAYERS = 24
CACHE_CH_DIM = 70
ENC_DIM = 1024
CACHE_TIME_DIM = 8


def load_wav_as_f32(path: Path) -> np.ndarray:
    """Load a 16kHz mono WAV file and return f32 samples in [-1, 1]."""
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1, f"Expected mono, got {wf.getnchannels()} channels"
        assert wf.getframerate() == SAMPLE_RATE, f"Expected {SAMPLE_RATE}Hz, got {wf.getframerate()}Hz"
        assert wf.getsampwidth() == 2, f"Expected 16-bit, got {wf.getsampwidth() * 8}-bit"
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


def apply_preemphasis(samples: np.ndarray, coeff: float = PREEMPH) -> np.ndarray:
    """Apply pre-emphasis filter: y[n] = x[n] - coeff * x[n-1], y[0] = x[0].

    Matches nemo_mel.zig line 33: preemph[0] = samples[0] (i.e. x[0] - 0.97*0).
    """
    preemph = np.empty_like(samples)
    preemph[0] = samples[0]
    preemph[1:] = samples[1:] - coeff * samples[:-1]
    return preemph


def load_filterbank(path: Path) -> np.ndarray:
    """Load filterbank.bin as [N_MELS, N_FFT_BINS] f32 array.

    File format: [1, 128, 257] raw f32 (leading dim=1 is a no-op).
    """
    data = path.read_bytes()
    expected = 1 * N_MELS * N_FFT_BINS * 4
    assert len(data) == expected, f"filterbank.bin: expected {expected} bytes, got {len(data)}"
    fb = np.frombuffer(data, dtype=np.float32).reshape(N_MELS, N_FFT_BINS)
    return fb


def compute_mel_bandmajor(samples: np.ndarray, filterbank: np.ndarray) -> np.ndarray:
    """Compute mel spectrogram in band-major layout [N_MELS, n_frames].

    Replicates nemo_mel.zig compute() exactly:
    - Pre-emphasis with coeff=0.97
    - STFT with center=True reflect padding, symmetric Hann window, n_fft=512
    - Power spectrum |FFT|^2
    - Mel filterbank dot product in f64
    - ln(energy + 2^-24)
    """
    preemph = apply_preemphasis(samples)

    # Symmetric Hann window: w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
    # Cast to f32 to match nemo_mel.zig's compile-time f32 Hann table.
    hann = np.zeros(N_FFT, dtype=np.float64)
    win_offset = (N_FFT - WIN_LENGTH) // 2  # 56
    for i in range(WIN_LENGTH):
        hann[win_offset + i] = 0.5 * (1.0 - np.cos(2.0 * np.pi * i / (WIN_LENGTH - 1)))
    hann = hann.astype(np.float32)

    # Frame count with center=True padding
    pad = N_FFT // 2  # 256
    padded_len = len(preemph) + 2 * pad
    n_frames = (padded_len - N_FFT) // HOP_LENGTH + 1

    result = np.empty((N_MELS, n_frames), dtype=np.float32)

    for frame in range(n_frames):
        frame_start = frame * HOP_LENGTH

        # Extract N_FFT samples from padded signal with reflect padding
        windowed = np.zeros(N_FFT, dtype=np.float64)
        for j in range(N_FFT):
            padded_idx = frame_start + j
            if padded_idx < pad:
                # Left reflect padding
                reflect_idx = pad - padded_idx
                sample = float(preemph[reflect_idx]) if reflect_idx < len(preemph) else 0.0
            elif padded_idx - pad < len(preemph):
                sample = float(preemph[padded_idx - pad])
            else:
                # Right reflect padding
                over = padded_idx - pad - len(preemph)
                reflect_idx = len(preemph) - 2 - over
                sample = float(preemph[reflect_idx]) if 0 <= reflect_idx < len(preemph) else 0.0

            windowed[j] = float(hann[j]) * sample

        # FFT → power spectrum
        fft_out = np.fft.rfft(windowed, n=N_FFT)
        power = np.abs(fft_out) ** 2

        # Mel filterbank dot product (in f64 to match Zig's f64 accumulator)
        for band in range(N_MELS):
            energy = np.float64(filterbank[band]).dot(np.float64(power))
            result[band, frame] = np.float32(np.log(energy + LOG_GUARD))

    return result


class MelCalibrationReader(CalibrationDataReader):
    """Yields encoder input dicts from real audio mel spectrograms with warm caches.

    Runs each WAV file's mel chunks sequentially through the FP32 encoder,
    carrying cache state forward between chunks. The first `warmup_chunks`
    per file are discarded (cold-start caches produce unrepresentative
    activation ranges). Only warm-cache samples are used for calibration.

    This is critical for streaming models: zero-cache calibration produces
    scale factors that don't cover real warm-cache activation ranges, causing
    output collapse after a few streaming chunks.
    """

    def __init__(
        self,
        wav_dir: Path,
        filterbank_path: Path,
        fp32_model_path: Path,
        max_samples: int = 300,
        warmup_chunks: int = 3,
    ):
        self.filterbank = load_filterbank(filterbank_path)
        self.samples: list[dict] = []
        self._build_samples(wav_dir, fp32_model_path, max_samples, warmup_chunks)
        self._idx = 0

    def _build_samples(
        self, wav_dir: Path, fp32_model_path: Path, max_samples: int, warmup_chunks: int
    ) -> None:
        """Run chunks through FP32 encoder sequentially, collecting warm-cache inputs."""
        wav_files = sorted(wav_dir.glob("*.wav"))
        if not wav_files:
            raise FileNotFoundError(f"No WAV files found in {wav_dir}")

        # Load FP32 encoder to generate realistic cache states
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        fp32_session = ort.InferenceSession(
            str(fp32_model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )

        print(f"Building warm-cache calibration data from {len(wav_files)} WAV files...")
        print(f"  Warmup chunks per file: {warmup_chunks} (discarded)")

        for wav_path in wav_files:
            audio = load_wav_as_f32(wav_path)
            mel = compute_mel_bandmajor(audio, self.filterbank)
            n_frames = mel.shape[1]

            # Reset caches for each file (fresh stream)
            cache_ch = np.zeros((1, ENC_LAYERS, CACHE_CH_DIM, ENC_DIM), dtype=np.float32)
            cache_time = np.zeros((1, ENC_LAYERS, ENC_DIM, CACHE_TIME_DIM), dtype=np.float32)
            cache_ch_len = np.zeros((1,), dtype=np.int64)
            pre_cache = np.zeros((N_MELS, PRE_ENCODE_CACHE), dtype=np.float32)

            cursor = 0
            chunk_idx = 0
            while cursor + MEL_SHIFT <= n_frames:
                # Build chunk: pre_cache + new frames (same as validate/runtime)
                new_mel = mel[:, cursor:cursor + MEL_SHIFT]
                chunk = np.concatenate([pre_cache, new_mel], axis=1)  # [128, 65]

                feed = {
                    "audio_signal": chunk[np.newaxis, :, :].astype(np.float32),
                    "length": np.array([chunk.shape[1]], dtype=np.int64),
                    "cache_last_channel": cache_ch.copy(),
                    "cache_last_time": cache_time.copy(),
                    "cache_last_channel_len": cache_ch_len.copy(),
                }

                # Run FP32 encoder to get real cache outputs
                outputs = fp32_session.run(None, feed)
                cache_ch = outputs[2]
                cache_time = outputs[3]
                cache_ch_len = outputs[4]

                # Update pre-encode cache
                if new_mel.shape[1] >= PRE_ENCODE_CACHE:
                    pre_cache = new_mel[:, -PRE_ENCODE_CACHE:]
                else:
                    pre_cache = new_mel

                # Only collect samples after warmup (caches are now realistic)
                if chunk_idx >= warmup_chunks:
                    self.samples.append(feed)

                cursor += MEL_SHIFT
                chunk_idx += 1

                if len(self.samples) >= max_samples:
                    break

            if len(self.samples) >= max_samples:
                break

        print(f"  Collected {len(self.samples)} warm-cache calibration samples")

    def get_next(self) -> dict | None:
        if self._idx >= len(self.samples):
            return None

        sample = self.samples[self._idx]
        self._idx += 1
        return sample

    def rewind(self):
        """Reset iterator to the beginning (required by some ORT quantization flows)."""
        self._idx = 0
