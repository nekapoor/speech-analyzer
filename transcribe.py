#!/usr/bin/env python3
"""Transcribe a recording with the original OpenAI Whisper library — locally.

Uses the `openai-whisper` package (github.com/openai/whisper). It downloads the
model weights ONCE from a CDN and then runs entirely on this machine — it does NOT
call the OpenAI API, and no audio ever leaves the computer.

Note: this is the same Whisper model as the rest of the tool (faster-whisper runs
the identical weights, just faster). Use this when you specifically want the
reference implementation's output to compare against.

Outputs, next to the recording:
    <name>.whisper.txt    readable, one line per segment with [mm:ss-mm:ss] times
    <name>.whisper.srt    subtitles (good for playing alongside the video)
    <name>.whisper.json   full data incl. word-level timestamps

Usage:
    .venv-openai/bin/python transcribe.py "OSCE encounter 3.mp4"
    .venv-openai/bin/python transcribe.py "OSCE encounter 3.mp4" --model large-v3

Models (accuracy vs speed): tiny.en, base.en, small.en, medium.en, large-v3.
Bigger = more accurate + slower. On an M2, large-v3 on CPU is SLOW (many minutes
for a 10-min file); small.en or medium.en are the practical sweet spot.
"""

import argparse, json, sys
from pathlib import Path


def mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def srt_time(t: float) -> str:
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="recording (mp4, mov, m4a, mp3, wav, ...)")
    ap.add_argument("--model", default="small.en",
                    help="tiny.en | base.en | small.en | medium.en | large-v3 (default: small.en)")
    ap.add_argument("--language", default="en")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: next to audio)")
    args = ap.parse_args()

    audio = Path(args.audio)
    if not audio.exists():
        sys.exit(f"Not found: {audio}")
    out_dir = args.out or audio.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    import whisper

    print(f"Loading OpenAI Whisper '{args.model}' (downloads once, then fully local) ...")
    model = whisper.load_model(args.model)
    print("Transcribing ... (first run also downloads the model)")
    result = model.transcribe(str(audio), language=args.language,
                              word_timestamps=True, verbose=False)

    segs = result.get("segments", [])
    stem = audio.stem

    # readable txt
    lines = [f"# {audio.name} — OpenAI Whisper ({args.model})", ""]
    for s in segs:
        lines.append(f"[{mmss(s['start'])}-{mmss(s['end'])}] {s['text'].strip()}")
    (out_dir / f"{stem}.whisper.txt").write_text("\n".join(lines))

    # srt
    srt = []
    for i, s in enumerate(segs, 1):
        srt += [str(i), f"{srt_time(s['start'])} --> {srt_time(s['end'])}",
                s["text"].strip(), ""]
    (out_dir / f"{stem}.whisper.srt").write_text("\n".join(srt))

    # full json incl. word timestamps
    out = {"source": str(audio), "language": result.get("language"),
           "model": args.model,
           "segments": [
               {"start": round(s["start"], 3), "end": round(s["end"], 3),
                "text": s["text"].strip(),
                "words": [{"word": w["word"], "start": round(w["start"], 3),
                           "end": round(w["end"], 3)} for w in s.get("words", [])]}
               for s in segs]}
    (out_dir / f"{stem}.whisper.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"\nDone ({len(segs)} segments):")
    for f in ("txt", "srt", "json"):
        print(f"  {out_dir / (stem + '.whisper.' + f)}")


if __name__ == "__main__":
    main()
