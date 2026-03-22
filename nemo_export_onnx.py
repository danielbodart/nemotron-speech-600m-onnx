#!/usr/bin/env python3
"""
Export Nemotron Speech Streaming model from .nemo to ONNX format.

Produces three ONNX files (encoder, decoder, joiner) suitable for
direct onnxruntime inference with cache-aware streaming.

Usage:
    uv run --with "nemo_toolkit[asr]" python3 scripts/nemo_export_onnx.py [output-dir]
"""
import sys
import os
import torch
import numpy as np

def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "dist/models/nemotron-600m-onnx"
    os.makedirs(out_dir, exist_ok=True)

    import nemo.collections.asr as nemo_asr

    print("Loading model from HuggingFace...")
    model = nemo_asr.models.ASRModel.from_pretrained('nvidia/nemotron-speech-streaming-en-0.6b')
    model.eval()
    model = model.cuda()

    print(f"Model type: {type(model).__name__}")
    print(f"Encoder layers: {model.encoder._cfg.n_layers}")
    print(f"d_model: {model.encoder._cfg.d_model}")
    print(f"Vocab size: {model.decoder._cfg.vocab_size}")

    # Use NeMo's built-in ONNX export
    # This handles the cache tensor interface correctly for streaming
    print(f"\nExporting to {out_dir}...")
    model.export(
        output=os.path.join(out_dir, "model.onnx"),
        check_trace=False,  # Skip trace validation (can be slow)
    )

    # List what was produced
    print(f"\nExported files:")
    for f in sorted(os.listdir(out_dir)):
        size = os.path.getsize(os.path.join(out_dir, f))
        print(f"  {f}: {size / 1024 / 1024:.1f} MB")

    # Also copy tokens.txt from the model
    tokenizer = model.tokenizer
    tokens_path = os.path.join(out_dir, "tokens.txt")
    with open(tokens_path, 'w') as f:
        for i in range(tokenizer.vocab_size):
            token = tokenizer.ids_to_tokens([i])[0]
            f.write(f"{token} {i}\n")
    print(f"  tokens.txt: {tokenizer.vocab_size} tokens")

    # Save preprocessor config for reference
    pp = model.preprocessor._cfg
    config_path = os.path.join(out_dir, "preprocessor.config")
    with open(config_path, 'w') as f:
        f.write(f"sample_rate={pp.sample_rate}\n")
        f.write(f"n_mels={pp.features}\n")
        f.write(f"window_size={pp.window_size}\n")
        f.write(f"window_stride={pp.window_stride}\n")
        f.write(f"n_fft={pp.n_fft}\n")
        f.write(f"window={pp.window}\n")
        f.write(f"normalize={pp.normalize}\n")
        f.write(f"dither={pp.dither}\n")
        f.write(f"preemph=0.97\n")
        f.write(f"pad_to={pp.pad_to}\n")
    print(f"  preprocessor.config saved")

    # Save the mel filterbank weights (slaney norm, needed by Zig)
    fb = model.preprocessor.featurizer.fb.cpu().numpy()
    fb_path = os.path.join(out_dir, "filterbank.bin")
    np.array(fb, dtype=np.float32).tofile(fb_path)
    fb_meta_path = os.path.join(out_dir, "filterbank.meta")
    with open(fb_meta_path, 'w') as f:
        f.write(f"shape={fb.shape[0]}x{fb.shape[1]}x{fb.shape[2]}\n")
        f.write(f"# [1, n_mels, n_fft/2+1] = [1, 128, 257]\n")
    print(f"  filterbank.bin: {fb.shape}")

    print(f"\nDone. Model exported to {out_dir}")
    print(f"Use with: --asr nemotron --model {out_dir}")


if __name__ == '__main__':
    main()
