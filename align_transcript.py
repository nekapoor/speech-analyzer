#!/usr/bin/env python3
"""Retro-timestamp a speaker-labeled transcript against the audio.

You already have a transcript that a person split by speaker but WITHOUT reliable
times:

    Pharmacist: Hi, I'm the intern pharmacist, how are you today?
    Patient: I've been a bit worried about the new medication.
    Pharmacist: That's completely understandable...

This tool lays down a timing track with faster-whisper (word-level timestamps),
aligns YOUR words onto it by text-matching, and carries your speaker labels
through. Out comes the same transcript, now with real start/end times per turn —
which is exactly what the rest of the analyzer needs.

Why this way: diarization on far-field reverberant audio is unreliable, but a human
can tell the speakers apart effortlessly. So let the human diarize; let the machine
do the timestamps. No cloud, no new installs, runs on the main .venv.

Usage:
    .venv/bin/python align_transcript.py "encounter 3.mp4" "encounter 3.txt"
    .venv/bin/python align_transcript.py "encounter 3.mp4" "encounter 3.txt" --model medium.en

Transcript format: one turn per block, each starting with "Speaker:".
Lines without a "Speaker:" prefix continue the current turn. Blank lines are fine.
"""

import argparse, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_speech as A  # extract_wav, mmss, prosody helpers  # noqa: E402

_SPEAKER_RE = re.compile(r"^\s*([A-Za-z][\w .'&/-]{0,30}?)\s*[:–-]\s+(\S.*)$")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def parse_labeled_transcript(text: str):
    """-> list of turns [{speaker, text}], preserving order. A line that starts
    with 'Name:' opens a new turn; other lines append to the current one."""
    turns = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        m = _SPEAKER_RE.match(line)
        if m:
            turns.append({"speaker": m.group(1).strip(), "text": m.group(2).strip()})
        elif turns:
            turns[-1]["text"] += " " + line.strip()
    return turns


def machine_words(wav_path, model_name, language):
    """faster-whisper transcription -> flat list of {w, start, end} (word level)."""
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(wav_path), language=language,
                                   word_timestamps=True, vad_filter=True)
    words = []
    for seg in segments:
        for w in (seg.words or []):
            words.append({"w": w.word.strip(), "start": round(w.start, 3),
                          "end": round(w.end, 3)})
    return words


def _norm(s):
    return _WORD_RE.findall(s.lower())


def align(human_turns, mwords, audio_end):
    """Snap human words onto machine word timings by text-matching, then collapse
    back into per-turn start/end. Returns turns with start/end/speaker/text."""
    from difflib import SequenceMatcher

    # flatten human words, remembering which turn each belongs to
    h_words, h_turn = [], []
    for ti, t in enumerate(human_turns):
        for tok in _norm(t["text"]):
            h_words.append(tok)
            h_turn.append(ti)

    m_norm = [ (_WORD_RE.findall(x["w"].lower()) or [""])[0] for x in mwords ]
    m_time = [ (x["start"], x["end"]) for x in mwords ]

    # anchor: for each human-word index, a timestamp (or None) from matched machine word
    anchor = [None] * len(h_words)
    sm = SequenceMatcher(a=h_words, b=m_norm, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                anchor[i1 + k] = m_time[j1 + k]

    # fill gaps by interpolating between surrounding anchors
    times = _interpolate(anchor, audio_end)

    # collapse per turn
    out = []
    for ti, t in enumerate(human_turns):
        idxs = [i for i, tt in enumerate(h_turn) if tt == ti]
        if not idxs:  # a turn with no alignable words (e.g. "[laughs]")
            out.append({"start": None, "end": None, "speaker": t["speaker"],
                        "text": t["text"]})
            continue
        start = min(times[i][0] for i in idxs)
        end = max(times[i][1] for i in idxs)
        out.append({"start": round(start, 2), "end": round(end, 2),
                    "speaker": t["speaker"], "text": t["text"]})
    _monotonic(out)
    return out


def _interpolate(anchor, audio_end):
    """Give every human word a (start,end); unanchored runs are spread evenly
    between the nearest anchored words on each side."""
    n = len(anchor)
    times = [None] * n
    # first pass: keep anchors
    for i, a in enumerate(anchor):
        if a is not None:
            times[i] = a
    # find anchored indices
    known = [i for i in range(n) if times[i] is not None]
    if not known:  # nothing matched at all -> spread uniformly across the audio
        for i in range(n):
            s = audio_end * i / max(1, n)
            e = audio_end * (i + 1) / max(1, n)
            times[i] = (s, e)
        return times
    # leading gap
    for i in range(0, known[0]):
        times[i] = (max(0.0, times[known[0]][0] - (known[0] - i) * 0.3), times[known[0]][0])
    # trailing gap
    for i in range(known[-1] + 1, n):
        times[i] = (times[known[-1]][1], min(audio_end, times[known[-1]][1] + (i - known[-1]) * 0.3))
    # interior gaps
    for a, b in zip(known, known[1:]):
        if b - a > 1:
            t0, t1 = times[a][1], times[b][0]
            span = max(0.0, t1 - t0)
            gap = b - a
            for k in range(1, gap):
                s = t0 + span * (k - 1) / gap
                e = t0 + span * k / gap
                times[a + k] = (s, e)
    return times


def _monotonic(turns):
    """Nudge any out-of-order boundaries so turns don't overlap backwards."""
    last = 0.0
    for t in turns:
        if t["start"] is None:
            continue
        if t["start"] < last:
            t["start"] = last
        if t["end"] < t["start"]:
            t["end"] = t["start"]
        last = t["end"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="the recording (mp4, mov, m4a, mp3, wav, ...)")
    ap.add_argument("transcript", help="speaker-labeled transcript (.txt)")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: next to audio)")
    ap.add_argument("--model", default="small.en", help="faster-whisper model for the timing track")
    ap.add_argument("--language", default="en")
    args = ap.parse_args()

    audio = Path(args.audio)
    tpath = Path(args.transcript)
    for p in (audio, tpath):
        if not p.exists():
            sys.exit(f"Not found: {p}")

    turns = parse_labeled_transcript(tpath.read_text())
    if not turns:
        sys.exit("No 'Speaker:' turns found. Each turn should start with a name and a colon.")
    speakers = sorted({t["speaker"] for t in turns})
    print(f"Parsed {len(turns)} turns, speakers: {', '.join(speakers)}")

    out_dir = args.out or audio.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / f"{audio.stem}.16k.wav"
    print("Extracting audio + building timing track (faster-whisper) ...")
    A.extract_wav(audio, wav)
    mwords = machine_words(wav, args.model, args.language)
    try:
        import soundfile as sf
        audio_end = len(sf.SoundFile(str(wav))) / 16000.0
    except Exception:
        audio_end = mwords[-1]["end"] if mwords else 0.0
    print(f"  timing track: {len(mwords)} machine words over {audio_end:.1f}s")

    print("Aligning your labeled transcript onto the timeline ...")
    aligned = align(turns, mwords, audio_end)

    stem = audio.stem
    (out_dir / f"{stem}.aligned.json").write_text(json.dumps(
        {"source": str(audio), "speakers": speakers, "segments": aligned}, indent=2))
    lines = [f"# {audio.name} — speaker labels from your transcript, times aligned to audio", ""]
    for t in aligned:
        ts = f"[{A.mmss(t['start'])}-{A.mmss(t['end'])}]" if t["start"] is not None else "[  ?  ]"
        lines.append(f"{ts} {t['speaker']}: {t['text']}")
    (out_dir / f"{stem}.aligned.txt").write_text("\n".join(lines))

    try:
        wav.unlink()
    except OSError:
        pass
    print(f"\nDone:\n  {out_dir / (stem + '.aligned.txt')}\n  {out_dir / (stem + '.aligned.json')}")


if __name__ == "__main__":
    main()
