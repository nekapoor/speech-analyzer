#!/usr/bin/env python3
"""Local audio enhancement — runs INSIDE .venv-denoise.

Aimed at far-field, reverberant room recordings (OSCE CCTV-style mics):
  1) WPE dereverberation  (nara_wpe) — pulls apart the smeared echo tails that
     make one speaker bleed into the next (the thing that wrecks diarization).
  2) Spectral denoise     (noisereduce) — knocks down steady room/HVAC noise.

Reads a 16k mono wav, writes an enhanced 16k mono wav to --out. Pure Python, no
build tools, fully offline. Every step degrades gracefully: if enhancement fails
we copy the input through untouched, so it can never break the pipeline.
"""

import argparse, shutil, sys, time
import numpy as np
import soundfile as sf


def log(m): print(f"[denoise] {m}", file=sys.stderr, flush=True)


def _dereverb(y: np.ndarray, sr: int) -> np.ndarray:
    """Single-channel WPE dereverberation in the STFT domain."""
    from nara_wpe.wpe import wpe
    from nara_wpe.utils import stft, istft
    opts = dict(size=512, shift=128)
    y2 = y[None, :]                       # (channels=1, samples)
    Y = stft(y2, **opts).transpose(1, 0, 2)   # (F, D, T)
    Z = wpe(Y, taps=10, delay=3, iterations=5, statistics_mode="full")
    z = istft(Z.transpose(1, 0, 2), **opts)   # (D, samples)
    return z[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")           # 16k mono wav from the parent
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-dereverb", action="store_true")
    ap.add_argument("--no-denoise", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    try:
        y, sr = sf.read(args.audio)
        if y.ndim > 1:                 # force mono
            y = y.mean(axis=1)
        y = y.astype(np.float64)

        if not args.no_dereverb:
            try:
                log("dereverberating (WPE) ...")
                y = _dereverb(y, sr)
            except Exception as e:
                log(f"dereverb skipped ({e})")

        if not args.no_denoise:
            try:
                log("denoising ...")
                import noisereduce as nr
                y = nr.reduce_noise(y=y, sr=sr, stationary=False)
            except Exception as e:
                log(f"denoise skipped ({e})")

        # normalize to avoid clipping, write 16-bit pcm
        peak = float(np.max(np.abs(y))) or 1.0
        y = (y / peak * 0.95).astype(np.float32)
        sf.write(args.out, y, sr, subtype="PCM_16")
        log(f"done in {round(time.time()-t0,1)}s -> {args.out}")
    except Exception as e:
        log(f"enhancement failed, passing audio through untouched: {e}")
        shutil.copyfile(args.audio, args.out)


if __name__ == "__main__":
    main()
