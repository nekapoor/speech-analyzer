"""Parent-side wrapper that runs whisperx_runner.py in the isolated venv.

We never import whisperx here (wrong venv). We shell out, pass the HF token via
env (never on the command line, so it can't leak into `ps`), and read the JSON
the runner writes.
"""

from __future__ import annotations

import json, os, subprocess, tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_WX_PY = _ROOT / ".venv-whisperx" / "bin" / "python"
_RUNNER = _ROOT / "webui" / "engines" / "whisperx_runner.py"


def run(audio_path, model_name="large-v3", language="en",
        min_speakers=None, max_speakers=None, hf_token=None, diar_model=None) -> dict:
    if not _WX_PY.exists():
        raise RuntimeError(".venv-whisperx not found — whisperX isn't installed")

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "result.json"
        cmd = [str(_WX_PY), str(_RUNNER), str(audio_path), "--out", str(out),
               "--model", model_name, "--language", language]
        if min_speakers is not None:
            cmd += ["--min-speakers", str(min_speakers)]
        if max_speakers is not None:
            cmd += ["--max-speakers", str(max_speakers)]
        if diar_model:
            cmd += ["--diar-model", diar_model]

        env = dict(os.environ)
        if hf_token:
            env["HF_TOKEN"] = hf_token
        # NOTE: no torchcodec/ffmpeg@7 shim needed. whisperX decodes audio with the
        # ffmpeg CLI and hands pyannote an in-memory waveform, so pyannote never
        # invokes torchcodec's (ffmpeg<=7) file decoder. The harmless import-time
        # "torchcodec not installed correctly" warning can be ignored.

        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if not out.exists():
            # runner died before writing — surface the tail of its stderr
            tail = "\n".join((proc.stderr or "").strip().splitlines()[-8:])
            raise RuntimeError(f"whisperX runner failed (exit {proc.returncode}):\n{tail}")
        return json.loads(out.read_text())
