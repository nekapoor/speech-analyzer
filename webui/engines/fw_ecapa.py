"""faster-whisper + ECAPA/KMeans engine — runs in the main .venv.

This is the token-free baseline. It reuses the exact diarization code from the
project's analyze_speech.py (ECAPA voice embeddings -> KMeans into 2 speakers),
so the UI compares the *real* thing, not a reimplementation.

Speaker labels are kept generic (SPEAKER_0/SPEAKER_1) rather than student/patient
so the column lines up honestly against whisperX's SPEAKER_00/01. (The CLI script
analyze_speech.py still gives you student/patient + prosody.)
"""

from __future__ import annotations

import sys, time, tempfile
from pathlib import Path

import numpy as np

# import the project's own diarization + audio helpers (project root on sys.path)
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import analyze_speech as A  # noqa: E402  (extract_wav, diarize, SAMPLE_RATE)
from .common import result  # noqa: E402

_MODEL_CACHE: dict = {}


def _get_model(model_name: str):
    """Load + cache a WhisperModel so repeated runs don't reload from disk."""
    if model_name not in _MODEL_CACHE:
        from faster_whisper import WhisperModel
        _MODEL_CACHE[model_name] = WhisperModel(
            model_name, device="cpu", compute_type="int8"
        )
    return _MODEL_CACHE[model_name]


def transcribe_segments(wav_path, model_name: str = "large-v3", language: str = "en"):
    """faster-whisper transcription from a prepared 16k wav (no diarization).

    Used when pairing faster-whisper transcription with an external diarizer
    (e.g. Sortformer). Returns (segments, detected_language).
    """
    model = _get_model(model_name)
    segments, info = model.transcribe(
        str(wav_path), language=language, vad_filter=True
    )
    segs = [{"start": round(seg.start, 2), "end": round(seg.end, 2),
             "text": seg.text.strip()} for seg in segments]
    return segs, info.language


def run(audio_path: str, model_name: str = "large-v3", language: str = "en") -> dict:
    import parselmouth

    t0 = time.time()
    model = _get_model(model_name)

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "audio.16k.wav"
        A.extract_wav(Path(audio_path), wav)

        segments, info = model.transcribe(
            str(wav), language=language, word_timestamps=True, vad_filter=True
        )
        sound = parselmouth.Sound(str(wav))
        wave = np.asarray(sound.values[0], dtype=np.float32)

        turns = []
        for i, seg in enumerate(segments):
            start, end = round(seg.start, 3), round(seg.end, 3)
            turns.append({
                "id": i, "start": start, "end": end,
                "duration": round(end - start, 3), "text": seg.text.strip(),
            })

        A.diarize(wave, turns)  # attaches speaker_id (0/1) in place

    for t in turns:
        t["speaker"] = f"SPEAKER_{t.get('speaker_id', 0)}"

    return result(
        engine="faster-whisper + ECAPA",
        model=model_name,
        language=info.language,
        elapsed_sec=time.time() - t0,
        segments=turns,
        diarization="ECAPA + KMeans (token-free, forced 2 speakers)",
    )
