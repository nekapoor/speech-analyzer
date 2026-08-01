#!/usr/bin/env python3
"""
analyze_speech_nodiar.py — offline transcription + per-turn prosody (NO diarization).

Same output shape as the full pipeline, but speaker labels are omitted
(diarization needs a Hugging Face token). Uses faster-whisper for word-level
timestamps + Parselmouth for acoustic metrics.

Usage:
  .venv/bin/python analyze_speech_nodiar.py <file-or-folder> [--out OUTDIR]
                   [--model small.en] [--language en]
"""

import argparse, json, math, re, subprocess, sys, warnings
from pathlib import Path

# Hide harmless numpy RuntimeWarnings from faster-whisper on silent audio.
warnings.filterwarnings("ignore", category=RuntimeWarning)

MEDIA_EXTS = {".mp3", ".mp4", ".wav", ".m4a", ".mov", ".aac", ".flac", ".mkv"}
F0_FLOOR, F0_CEIL = 75.0, 500.0


def find_inputs(path: Path):
    if path.is_file():
        return [path]
    files = sorted(p for p in path.rglob("*") if p.suffix.lower() in MEDIA_EXTS)
    if not files:
        sys.exit(f"No media files found under {path}")
    return files


def extract_wav(src: Path, dst: Path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000",
         "-vn", "-loglevel", "error", str(dst)],
        check=True,
    )


def _num(x, n=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), n)


def mmss(t):
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def prosody_for_turn(sound, start, end, words, text):
    from parselmouth.praat import call
    m = {k: None for k in (
        "pitch_mean_hz", "pitch_std_hz", "pitch_range_hz",
        "intensity_mean_db", "intensity_std_db",
        "speaking_rate_wps", "articulation_rate_wps", "pause_ratio",
        "jitter_pct", "shimmer_pct", "hnr_db")}
    dur = end - start

    timed = [w for w in words if w.get("start") is not None and w.get("end") is not None]
    n = len(timed)
    if n and dur > 0:
        phon = sum(w["end"] - w["start"] for w in timed)
        m["speaking_rate_wps"] = _num(n / dur, 2)
        if phon > 0:
            m["articulation_rate_wps"] = _num(n / phon, 2)
        m["pause_ratio"] = _num(max(0.0, 1 - phon / dur), 3)

    if dur < 0.12:
        return m
    try:
        part = sound.extract_part(from_time=start, to_time=end, preserve_times=True)
        pitch = part.to_pitch(pitch_floor=F0_FLOOR, pitch_ceiling=F0_CEIL)
        m["pitch_mean_hz"] = _num(call(pitch, "Get mean", 0, 0, "Hertz"), 1)
        m["pitch_std_hz"] = _num(call(pitch, "Get standard deviation", 0, 0, "Hertz"), 1)
        lo = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
        hi = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
        if not (math.isnan(lo) or math.isnan(hi)):
            m["pitch_range_hz"] = _num(hi - lo, 1)
        inten = part.to_intensity(minimum_pitch=F0_FLOOR)
        m["intensity_mean_db"] = _num(call(inten, "Get mean", 0, 0, "dB"), 1)
        m["intensity_std_db"] = _num(call(inten, "Get standard deviation", 0, 0), 1)
        pp = call(part, "To PointProcess (periodic, cc)", F0_FLOOR, F0_CEIL)
        m["jitter_pct"] = _num(call(pp, "Get jitter (local)", 0, 0, 1e-4, 0.02, 1.3) * 100, 3)
        m["shimmer_pct"] = _num(call([part, pp], "Get shimmer (local)", 0, 0, 1e-4, 0.02, 1.3, 1.6) * 100, 3)
        hnr = part.to_harmonicity_cc(minimum_pitch=F0_FLOOR)
        m["hnr_db"] = _num(call(hnr, "Get mean", 0, 0), 1)
    except Exception:
        pass
    return m


def process(src, outdir, model, args):
    import parselmouth
    stem = src.stem
    wav_path = outdir / f"{stem}.16k.wav"
    print(f"\n=== {src.name} ===")
    extract_wav(src, wav_path)

    print("  transcribing...")
    segments, info = model.transcribe(
        str(wav_path), language=args.language, word_timestamps=True,
        vad_filter=True)
    sound = parselmouth.Sound(str(wav_path))

    turns = []
    for i, seg in enumerate(segments):
        words = [{"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)}
                 for w in (seg.words or [])]
        start, end = round(seg.start, 3), round(seg.end, 3)
        text = seg.text.strip()
        turns.append({
            "id": i, "start": start, "end": end, "duration": round(end - start, 3),
            "text": text,
            "prosody": prosody_for_turn(sound, start, end, words, text),
        })
        print(f"    [{mmss(start)}] {text[:70]}")

    out = {"source_file": str(src), "language": info.language,
           "duration_sec": round(info.duration, 2),
           "note": "no speaker labels in this version; add diarization (HF_TOKEN) to distinguish student vs patient",
           "segments": turns}
    (outdir / f"{stem}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    write_compact(out, outdir / f"{stem}.txt")
    print(f"  -> {stem}.json / {stem}.txt  ({len(turns)} turns)")


def write_compact(out, path):
    lines = [f"# {Path(out['source_file']).name}  ({out['duration_sec']}s)",
             f"# {out['note']}",
             "# fmt: [start-end] | pitch mean±std Hz | int dB | rate w/s "
             "| pause% | jit% shim% hnr | \"text\"", ""]
    for t in out["segments"]:
        p = t["prosody"]
        def g(k): return "?" if p.get(k) is None else p[k]
        lines.append(
            f"[{mmss(t['start'])}-{mmss(t['end'])}] | "
            f"pitch {g('pitch_mean_hz')}±{g('pitch_std_hz')}Hz | "
            f"{g('intensity_mean_db')}dB | {g('speaking_rate_wps')}w/s | "
            f"pause {g('pause_ratio')} | "
            f"jit {g('jitter_pct')}% shim {g('shimmer_pct')}% hnr {g('hnr_db')} | "
            f"\"{t['text']}\"")
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--model", default="large-v3")  # most accurate; cached on first run
    ap.add_argument("--language", default="en")
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Loading faster-whisper {args.model} (cpu/int8)...")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    for src in find_inputs(args.input):
        try:
            process(src, args.out, model, args)
        except Exception as e:
            print(f"  !! failed on {src.name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
