"""Shared result schema + helpers for every engine.

Every engine, no matter how different internally, returns a dict shaped like:

    {
      "engine":       "faster-whisper + ECAPA",   # display name
      "model":        "large-v3",
      "language":     "en",
      "elapsed_sec":  12.3,
      "num_speakers": 2,
      "diarization":  "ECAPA + KMeans (token-free, forced 2)",  # how speakers were found
      "segments": [ {"start": 0.0, "end": 3.2, "speaker": "SPEAKER_0", "text": "..."}, ... ],
    }

Keeping the shape identical is the whole point: the UI renders columns side by
side, so any difference you see is a real difference between the engines, not an
artifact of formatting.
"""

from __future__ import annotations


def mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def normalize_speaker_labels(segments: list[dict]) -> int:
    """Rewrite whatever a diarizer called its speakers into stable SPEAKER_0/1/...

    Different engines invent their own labels (0/1, 'SPEAKER_00', 'student').
    We map them to SPEAKER_<n> in order of first appearance so the colors in the
    UI are consistent within a column. Returns the speaker count.
    """
    order: dict = {}
    for s in segments:
        spk = s.get("speaker")
        if spk is None:
            continue
        if spk not in order:
            order[spk] = f"SPEAKER_{len(order)}"
    for s in segments:
        if s.get("speaker") is not None:
            s["speaker"] = order[s["speaker"]]
    return len(order)


def assign_speakers(segments: list[dict], turns: list[dict]) -> None:
    """Label each transcript segment with the diarization speaker it overlaps most.

    This is the "right" way (what whisperX/pyannote do): diarize independently, then
    map words/segments onto speaker turns by time — instead of clustering the
    transcriber's own segments. `turns` is [{start, end, speaker}, ...]. Mutates
    `segments` in place, setting seg['speaker'].
    """
    for s in segments:
        ss, se = s["start"], s["end"]
        best, best_ov = None, 0.0
        for t in turns:
            ov = max(0.0, min(se, t["end"]) - max(ss, t["start"]))
            if ov > best_ov:
                best_ov, best = ov, t["speaker"]
        # if a segment overlaps no turn (silence gap), fall back to nearest turn
        if best is None and turns:
            best = min(turns, key=lambda t: abs((t["start"] + t["end"]) / 2 - (ss + se) / 2))["speaker"]
        s["speaker"] = best


def result(engine, model, language, elapsed_sec, segments, diarization):
    """Build the common result dict, normalizing speaker labels + counting them."""
    n = normalize_speaker_labels(segments)
    return {
        "engine": engine,
        "model": model,
        "language": language,
        "elapsed_sec": round(elapsed_sec, 1),
        "num_speakers": n,
        "diarization": diarization,
        "segments": [
            {
                "start": round(float(s["start"]), 2),
                "end": round(float(s["end"]), 2),
                "speaker": s.get("speaker"),
                "text": (s.get("text") or "").strip(),
            }
            for s in segments
        ],
    }
