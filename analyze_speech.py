#!/usr/bin/env python3
"""
analyze_speech.py — offline transcription + 2-speaker diarization + per-turn prosody.

Same output as analyze_speech_nodiar.py, but every turn is labelled with WHO spoke:
"student" or "patient". Diarization is done fully locally and token-free — it does
NOT use pyannote or any gated Hugging Face model, so no access token is needed.

Pipeline (all on-device, no data ever leaves the machine):
  faster-whisper                 -> transcript + word timestamps
  speechbrain ECAPA (public)     -> a voice "fingerprint" per turn
  KMeans (2 clusters)            -> split turns into two speakers
  parselmouth (Praat)            -> pitch / loudness / rate / pauses / voice quality

Who is "student" vs "patient": by default the person who speaks FIRST is the
student (the intern pharmacist usually opens the call). Pass --patient-first to
flip it, or --swap after the fact.

Usage:
  .venv/bin/python analyze_speech.py <file-or-folder> [--out OUTDIR]
                   [--model model] [--language en] [--patient-first]
"""

import argparse, json, math, subprocess, sys, warnings
from pathlib import Path

import numpy as np

# faster-whisper's feature extractor emits harmless numpy RuntimeWarnings on
# audio with silent stretches; hide them so the output stays clean.
warnings.filterwarnings("ignore", category=RuntimeWarning)

MEDIA_EXTS = {".mp3", ".mp4", ".wav", ".m4a", ".mov", ".aac", ".flac", ".mkv"}
F0_FLOOR, F0_CEIL = 75.0, 500.0
SAMPLE_RATE = 16_000

# Diarization windowing (only used for the embedding of each turn).
_MIN_TURN_SEC = 0.4        # turns shorter than this are labelled by their neighbours
_SILENCE_RMS = 0.005       # skip near-silent audio so silence isn't its own "speaker"


def find_inputs(path: Path):
    if path.is_file():
        return [path]
    files = sorted(p for p in path.rglob("*") if p.suffix.lower() in MEDIA_EXTS)
    if not files:
        sys.exit(f"No media files found under {path}")
    return files


def extract_wav(src: Path, dst: Path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-vn", "-loglevel", "error", str(dst)],
        check=True,
    )


def _num(x, n=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), n)


def mmss(t):
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


# --------------------------------------------------------------------------- #
# Diarization: label each turn as speaker 0 or 1, then map to student/patient. #
# --------------------------------------------------------------------------- #

_ENCODER = None


def _get_encoder():
    """Load the ECAPA speaker-embedding model once. Public, non-gated: no token."""
    global _ENCODER
    if _ENCODER is None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # older speechbrain
            from speechbrain.pretrained import EncoderClassifier
        _ENCODER = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=".cache/ecapa",
            run_opts={"device": "cpu"},
        )
    return _ENCODER


def _embed(enc, wave: np.ndarray) -> np.ndarray | None:
    """Voice fingerprint for one chunk of audio, or None if too short/silent."""
    import torch
    if len(wave) < int(_MIN_TURN_SEC * SAMPLE_RATE):
        return None
    if float(np.sqrt(np.mean(wave ** 2))) < _SILENCE_RMS:
        return None
    with torch.no_grad():
        e = enc.encode_batch(torch.tensor(wave).unsqueeze(0)).squeeze().cpu().numpy()
    return e


def _cluster_two(embs: np.ndarray) -> np.ndarray:
    """Split turn fingerprints into two speakers.

    ECAPA embeddings share a big common component (recording channel, "average
    voice"), so raw distances mostly reflect that shared part, not identity. We
    remove it by centering the embeddings, length-normalize onto the unit sphere
    (where KMeans == cosine clustering), then KMeans into two balanced clusters.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize
    x = normalize(embs - embs.mean(axis=0, keepdims=True))
    return KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(x)


def diarize(wave: np.ndarray, turns: list[dict]) -> None:
    """Attach turn['speaker_id'] (0/1) to every turn, in place."""
    enc = _get_encoder()

    embs, idx = [], []
    for i, t in enumerate(turns):
        seg = wave[int(t["start"] * SAMPLE_RATE):int(t["end"] * SAMPLE_RATE)]
        e = _embed(enc, seg)
        if e is not None:
            embs.append(e)
            idx.append(i)

    if len(embs) < 2:
        for t in turns:                       # not enough signal to tell them apart
            t["speaker_id"] = 0
        return

    labels = _cluster_two(np.array(embs))
    for j, i in enumerate(idx):
        turns[i]["speaker_id"] = int(labels[j])

    # Turns too short/silent to embed: borrow the nearest labelled turn's speaker.
    labelled = [i for i in idx]
    for i, t in enumerate(turns):
        if "speaker_id" in t:
            continue
        nearest = min(labelled, key=lambda k: abs(turns[k]["start"] - t["start"]))
        t["speaker_id"] = turns[nearest]["speaker_id"]

    _smooth(turns)


def _smooth(turns: list[dict], max_flip_sec: float = 1.2) -> None:
    """Flip a lone short turn back when both neighbours are the other speaker."""
    for i in range(1, len(turns) - 1):
        prev, cur, nxt = turns[i - 1], turns[i], turns[i + 1]
        if (cur["duration"] <= max_flip_sec
                and prev["speaker_id"] == nxt["speaker_id"]
                and cur["speaker_id"] != prev["speaker_id"]):
            cur["speaker_id"] = prev["speaker_id"]


def assign_roles(turns: list[dict], patient_first: bool) -> dict:
    """Map speaker_id 0/1 -> 'student'/'patient'. First speaker = student by default."""
    if not turns:
        return {}
    first_id = turns[0]["speaker_id"]
    role_of = {first_id: "patient" if patient_first else "student"}
    role_of[1 - first_id] = "student" if patient_first else "patient"
    for t in turns:
        t["speaker"] = role_of[t["speaker_id"]]
        del t["speaker_id"]
    return role_of


# --------------------------------------------------------------------------- #
# Prosody (unchanged from the no-diarization version).                         #
# --------------------------------------------------------------------------- #

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
    wave = np.asarray(sound.values[0], dtype=np.float32)  # mono 16 kHz for ECAPA

    turns = []
    for i, seg in enumerate(segments):
        words = [{"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)}
                 for w in (seg.words or [])]
        start, end = round(seg.start, 3), round(seg.end, 3)
        turns.append({
            "id": i, "start": start, "end": end, "duration": round(end - start, 3),
            "text": seg.text.strip(), "_words": words,
        })

    print("  diarizing (student vs patient)...")
    diarize(wave, turns)
    roles = assign_roles(turns, args.patient_first)

    for t in turns:
        t["prosody"] = prosody_for_turn(sound, t["start"], t["end"], t.pop("_words"), t["text"])
        print(f"    [{mmss(t['start'])}] {t['speaker']:>7}: {t['text'][:60]}")

    who_first = turns[0]["speaker"] if turns else "student"
    out = {"source_file": str(src), "language": info.language,
           "duration_sec": round(info.duration, 2),
           "speaker_note": f"2 speakers auto-detected; first speaker labelled '{who_first}' "
                           f"(use --patient-first to swap)",
           "segments": turns}
    (outdir / f"{stem}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    write_compact(out, outdir / f"{stem}.txt")
    print(f"  -> {stem}.json / {stem}.txt  ({len(turns)} turns)")


def write_compact(out, path):
    lines = [f"# {Path(out['source_file']).name}  ({out['duration_sec']}s)",
             f"# {out['speaker_note']}",
             "# fmt: [start-end] SPEAKER | pitch mean±std Hz | int dB | rate w/s "
             "| pause% | jit% shim% hnr | \"text\"", ""]
    for t in out["segments"]:
        p = t["prosody"]
        def g(k): return "?" if p.get(k) is None else p[k]
        lines.append(
            f"[{mmss(t['start'])}-{mmss(t['end'])}] {t['speaker']:>7} | "
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
    ap.add_argument("--model", default="large-v3")  # most accurate; downloaded & cached on first run
    ap.add_argument("--language", default="en")
    ap.add_argument("--patient-first", action="store_true",
                    help="the patient speaks first (default assumes the student does)")
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
