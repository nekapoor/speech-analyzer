#!/usr/bin/env python3
"""Clean one recording so you can LISTEN to the before/after — nothing else.

This is the "let me hear what the cleanup does to my real audio" tool. It does the
exact dereverb + denoise the web UI's "Clean audio first" toggle uses, but instead
of feeding it to a diarizer, it just writes two wav files next to your recording:

    <name>.original.wav   <- the audio as-is (extracted to 16k mono)
    <name>.cleaned.wav    <- after dereverb + denoise

Play them back-to-back and decide for yourself whether the cleanup helps.

100% local. Nothing is uploaded, nothing is deleted, nothing leaves your computer.

Run it with the denoise environment's Python:

    .venv-denoise/bin/python clean_audio.py "OSCE encounter 3.mp4"

Optional flags:
    --out DIR          write the two wavs into DIR instead of next to the input
    --no-dereverb      skip the dereverb step (denoise only)
    --no-denoise       skip the denoise step (dereverb only)
"""

import argparse, subprocess, sys
from pathlib import Path

# reuse the exact enhancement the web app runs
sys.path.insert(0, str(Path(__file__).resolve().parent / "webui" / "engines"))
from denoise_runner import enhance_wav  # noqa: E402


def extract_16k_mono(src: Path, dst: Path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000",
         "-vn", "-loglevel", "error", str(dst)],
        check=True,
    )


def main():
    ap = argparse.ArgumentParser(description="Write original + cleaned wavs so you can compare them by ear.")
    ap.add_argument("input", help="an audio or video recording (mp4, mov, m4a, mp3, wav, ...)")
    ap.add_argument("--out", type=Path, default=None, help="folder to write the two wavs into")
    ap.add_argument("--no-dereverb", action="store_true")
    ap.add_argument("--no-denoise", action="store_true")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"File not found: {src}")

    out_dir = args.out or src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    original = out_dir / f"{src.stem}.original.wav"
    cleaned = out_dir / f"{src.stem}.cleaned.wav"

    print(f"Extracting audio from {src.name} ...")
    extract_16k_mono(src, original)

    print("Cleaning (dereverb + denoise) ...")
    enhance_wav(str(original), str(cleaned),
                dereverb=not args.no_dereverb, denoise=not args.no_denoise)

    print("\nDone. Listen to these two and compare:")
    print(f"  BEFORE : {original}")
    print(f"  AFTER  : {cleaned}")
    print("\n(open them in QuickTime / double-click in Finder. Nothing left this computer.)")


if __name__ == "__main__":
    main()
