#!/usr/bin/env python3
"""NeMo Sortformer diarization — runs INSIDE .venv-nemo.

nvidia/diar_sortformer_4spk-v1: end-to-end, overlap-aware speaker diarization, up
to 4 speakers, ungated (no HF token). Reads a 16k mono wav, writes speaker turns
[{start, end, speaker}] to --out. Transcription is done separately; the parent
assigns words to these turns by time overlap.
"""

import argparse, json, sys, time


def log(m): print(f"[sortformer] {m}", file=sys.stderr, flush=True)


def _parse_pred(entry):
    """Normalize one Sortformer prediction into {start,end,speaker}.

    NeMo returns per-file lists of 'start end speaker' strings (RTTM-ish). Be
    defensive about the exact shape.
    """
    if isinstance(entry, str):
        parts = entry.split()
        if len(parts) >= 3:
            try:
                return {"start": round(float(parts[0]), 2),
                        "end": round(float(parts[1]), 2),
                        "speaker": parts[2]}
            except ValueError:
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")  # 16k mono wav prepared by the parent
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="nvidia/diar_sortformer_4spk-v1")
    args = ap.parse_args()

    from nemo.collections.asr.models import SortformerEncLabelModel

    t0 = time.time()
    log(f"loading {args.model} ...")
    model = SortformerEncLabelModel.from_pretrained(args.model)
    model.eval()

    log("diarizing ...")
    preds = model.diarize(audio=args.audio, batch_size=1, verbose=False)
    # preds is List[List[str]] (one inner list per audio file)
    first = preds[0] if preds and isinstance(preds[0], (list, tuple)) else preds
    turns = [t for t in (_parse_pred(e) for e in first) if t]

    with open(args.out, "w") as f:
        json.dump({"turns": turns, "elapsed_sec": round(time.time() - t0, 1)}, f)
    n_spk = len({t["speaker"] for t in turns})
    log(f"done in {round(time.time()-t0,1)}s, {len(turns)} turns, {n_spk} speakers")


if __name__ == "__main__":
    main()
