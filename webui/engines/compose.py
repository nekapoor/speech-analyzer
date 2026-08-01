"""Composed engines: pair a transcriber with a diarizer.

Parakeet (transcription) and Sortformer (diarization) each live in their own venv
and run as subprocesses. This module prepares a single 16k wav, fans it out to the
right child processes, and stitches transcript + speaker turns together with
common.assign_speakers (overlap-based word→speaker mapping — the "right" way).
"""

from __future__ import annotations

import json, subprocess, sys, tempfile, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_ENG = _ROOT / "webui" / "engines"
_PARA_PY = _ROOT / ".venv-parakeet" / "bin" / "python"
_NEMO_PY = _ROOT / ".venv-nemo" / "bin" / "python"
_DENOISE_PY = _ROOT / ".venv-denoise" / "bin" / "python"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import analyze_speech as A  # noqa: E402  (extract_wav)
from . import fw_ecapa  # noqa: E402
from .common import result, assign_speakers  # noqa: E402


def _extract_wav(audio: str, td: str) -> Path:
    wav = Path(td) / "audio.16k.wav"
    A.extract_wav(Path(audio), wav)
    return wav


def _subprocess_json(py: Path, script: str, wav: Path, extra: list[str], label: str) -> dict:
    if not py.exists():
        raise RuntimeError(f"{py.parent.parent.name} not found — is that engine installed?")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "result.json"
        cmd = [str(py), str(_ENG / script), str(wav), "--out", str(out), *extra]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if not out.exists():
            tail = "\n".join((proc.stderr or "").strip().splitlines()[-8:])
            raise RuntimeError(f"{label} failed (exit {proc.returncode}):\n{tail}")
        return json.loads(out.read_text())


def enhance_audio(src_audio: str, out_wav: str) -> str:
    """Dereverb + denoise the source into out_wav (a 16k mono wav). Local, ~1s.

    Runs in .venv-denoise. On any failure the runner passes the audio through
    untouched, so this never blocks a comparison — worst case you get the raw audio.
    """
    if not _DENOISE_PY.exists():
        raise RuntimeError(".venv-denoise not installed — run setup.sh")
    with tempfile.TemporaryDirectory() as td:
        raw = _extract_wav(src_audio, td)
        cmd = [str(_DENOISE_PY), str(_ENG / "denoise_runner.py"), str(raw),
               "--out", str(out_wav)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if not Path(out_wav).exists():
            tail = "\n".join((proc.stderr or "").strip().splitlines()[-6:])
            raise RuntimeError(f"enhancement failed:\n{tail}")
    return out_wav


def _parakeet_transcribe(wav: Path) -> dict:
    return _subprocess_json(_PARA_PY, "parakeet_runner.py", wav, [], "Parakeet")


def _sortformer_diarize(wav: Path) -> dict:
    return _subprocess_json(_NEMO_PY, "sortformer_runner.py", wav, [], "Sortformer")


# --------------------------------------------------------------------------- #
# Public engine entry points (called from app.py::_run_engine).               #
# All accept (**opts) and ignore what they don't use.                         #
# --------------------------------------------------------------------------- #

def run_parakeet(audio_path, **opts) -> dict:
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        wav = _extract_wav(audio_path, td)
        r = _parakeet_transcribe(wav)
    return result(
        engine="Parakeet-TDT (MLX)", model="parakeet-tdt-0.6b-v2",
        language=r.get("language", "en"), elapsed_sec=time.time() - t0,
        segments=r["segments"], diarization="none (transcription only)")


def run_fw_sortformer(audio_path, **opts) -> dict:
    model = opts.get("model") or "large-v3"
    language = opts.get("language") or "en"
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        wav = _extract_wav(audio_path, td)
        segs, lang = fw_ecapa.transcribe_segments(wav, model, language)
        diar = _sortformer_diarize(wav)
    assign_speakers(segs, diar["turns"])
    return result(
        engine="faster-whisper + Sortformer", model=model, language=lang,
        elapsed_sec=time.time() - t0, segments=segs,
        diarization="NeMo Sortformer (ungated, overlap-aware)")


def run_parakeet_sortformer(audio_path, **opts) -> dict:
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        wav = _extract_wav(audio_path, td)
        r = _parakeet_transcribe(wav)
        diar = _sortformer_diarize(wav)
    segs = r["segments"]
    assign_speakers(segs, diar["turns"])
    return result(
        engine="Parakeet + Sortformer", model="parakeet-tdt-0.6b-v2",
        language=r.get("language", "en"), elapsed_sec=time.time() - t0,
        segments=segs, diarization="NeMo Sortformer (ungated, overlap-aware)")
