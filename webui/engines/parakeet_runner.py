#!/usr/bin/env python3
"""Parakeet transcription — runs INSIDE .venv-parakeet (Apple MLX / GPU).

NVIDIA Parakeet-TDT via parakeet-mlx: tops the open English ASR leaderboard and
runs on the M-series GPU. Transcription only (no speakers). Writes segments with
timestamps to --out so a diarizer can be layered on top.
"""

import argparse, json, sys, time


def log(m): print(f"[parakeet] {m}", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")  # a 16k mono wav prepared by the parent
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="mlx-community/parakeet-tdt-0.6b-v2")
    args = ap.parse_args()

    from parakeet_mlx import from_pretrained

    t0 = time.time()
    log(f"loading {args.model} ...")
    model = from_pretrained(args.model)
    log("transcribing ...")
    r = model.transcribe(args.audio)

    segments = [
        {"start": round(float(s.start), 2), "end": round(float(s.end), 2),
         "text": (s.text or "").strip(), "speaker": None}
        for s in r.sentences
    ]
    out = {"segments": segments, "language": "en",
           "elapsed_sec": round(time.time() - t0, 1), "text": r.text}
    with open(args.out, "w") as f:
        json.dump(out, f)
    log(f"done in {out['elapsed_sec']}s, {len(segments)} segments")


if __name__ == "__main__":
    main()
