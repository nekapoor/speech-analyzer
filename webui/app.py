#!/usr/bin/env python3
"""Local web UI to compare transcription + diarization engines side by side.

Upload one recording, pick which engines to run, and see their transcripts +
speaker labels in parallel columns so you can judge which does the better job.

Runs entirely on localhost. Nothing is uploaded anywhere. The only network use is
model downloads on first run (and, for whisperX, fetching the gated pyannote model
with your HF token).

Run:  .venv/bin/python webui/app.py   ->   http://127.0.0.1:5001
"""

from __future__ import annotations

import os, threading, time, uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from engines import REGISTRY

ROOT = Path(__file__).resolve().parent
UPLOADS = ROOT / "uploads"
SECRETS = ROOT / ".secrets"
TOKEN_FILE = SECRETS / "hf_token"
UPLOADS.mkdir(exist_ok=True)
SECRETS.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB uploads

# job_id -> {"file": name, "created": ts, "engines": {key: {status, result?, error?, elapsed?}}}
JOBS: dict = {}
JOBS_LOCK = threading.Lock()


def _load_env_token() -> str:
    """Pull an HF token from the project-root .env (or the environment).

    Accepts the common key spellings so users don't have to guess. Never logged.
    """
    for var in ("HF_TOKEN", "HF_ACCESS_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var].strip()
    env_path = ROOT.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() in ("HF_TOKEN", "HF_ACCESS_TOKEN", "HUGGINGFACE_TOKEN",
                             "HUGGING_FACE_HUB_TOKEN"):
                return v.strip().strip('"').strip("'")
    return ""


def _load_token() -> str:
    """Token precedence: saved secret file, then .env / environment."""
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()
    return _load_env_token()


def _save_token(tok: str):
    if tok:
        TOKEN_FILE.write_text(tok.strip())
        TOKEN_FILE.chmod(0o600)


def _run_engine(job_id, key, audio_path, opts):
    """Run one engine in a background thread and stash its result on the job."""
    started = time.time()
    try:
        if key == "fw_ecapa":
            from engines import fw_ecapa
            res = fw_ecapa.run(audio_path, model_name=opts["model"],
                               language=opts["language"])
        elif key.startswith("whisperx"):
            from engines import whisperx_engine
            res = whisperx_engine.run(
                audio_path, model_name=opts["model"], language=opts["language"],
                min_speakers=opts["min_speakers"], max_speakers=opts["max_speakers"],
                hf_token=opts["hf_token"], diar_model=REGISTRY[key].get("diar_model"))
        elif key == "parakeet":
            from engines import compose
            res = compose.run_parakeet(audio_path, **opts)
        elif key == "fw_sortformer":
            from engines import compose
            res = compose.run_fw_sortformer(audio_path, **opts)
        elif key == "parakeet_sortformer":
            from engines import compose
            res = compose.run_parakeet_sortformer(audio_path, **opts)
        else:
            raise ValueError(f"unknown engine {key}")
        with JOBS_LOCK:
            JOBS[job_id]["engines"][key] = {"status": "done", "result": res}
    except Exception as e:  # keep other engines alive; report this one's failure
        with JOBS_LOCK:
            JOBS[job_id]["engines"][key] = {
                "status": "error", "error": str(e),
                "elapsed": round(time.time() - started, 1)}


def _dispatch(job_id, keys, raw_path, opts):
    """Coordinate one job: optionally enhance the audio ONCE, run every engine on
    it, then delete all audio from disk (FERPA). Runs in its own thread so /run
    returns immediately."""
    audio = raw_path
    paths = {raw_path}
    if opts.get("enhance"):
        try:
            from engines import compose
            enhanced = str(UPLOADS / f"{job_id}_enhanced.wav")
            compose.enhance_audio(raw_path, enhanced)
            audio, _ = enhanced, paths.add(enhanced)
        except Exception as e:
            with JOBS_LOCK:
                JOBS[job_id]["enhance_error"] = str(e)  # fall back to raw audio

    threads = [threading.Thread(target=_run_engine, args=(job_id, k, audio, opts),
                                daemon=True) for k in keys]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for p in paths:  # every engine done -> nothing is retained on disk
        try:
            os.remove(p)
        except OSError:
            pass


@app.route("/")
def index():
    return render_template(
        "index.html",
        engines=REGISTRY,
        has_token=bool(_load_token()),
        whisperx_ready=(ROOT.parent / ".venv-whisperx" / "bin" / "python").exists(),
    )


@app.route("/run", methods=["POST"])
def run():
    f = request.files.get("audio")
    if not f or not f.filename:
        return jsonify({"error": "no file uploaded"}), 400

    keys = [k for k in request.form.getlist("engines") if k in REGISTRY]
    if not keys:
        return jsonify({"error": "select at least one engine"}), 400

    # token: use the newly-typed one if present, else the saved one
    token = (request.form.get("hf_token") or "").strip() or _load_token()
    if request.form.get("hf_token"):
        _save_token(token)

    def _int(name):
        v = (request.form.get(name) or "").strip()
        return int(v) if v.isdigit() else None

    opts = {
        "model": request.form.get("model") or "large-v3",
        "language": request.form.get("language") or "en",
        "min_speakers": _int("min_speakers"),
        "max_speakers": _int("max_speakers"),
        "hf_token": token,
        "enhance": request.form.get("enhance") == "1",
    }

    job_id = uuid.uuid4().hex[:12]
    safe = secure_filename(f.filename)
    dest = UPLOADS / f"{job_id}_{safe}"
    f.save(dest)

    with JOBS_LOCK:
        JOBS[job_id] = {
            "file": safe, "created": time.time(),
            "engines": {k: {"status": "running"} for k in keys},
        }

    threading.Thread(target=_dispatch, args=(job_id, keys, str(dest), opts),
                     daemon=True).start()

    return jsonify({"job_id": job_id, "engines": keys, "file": safe})


@app.route("/status/<job_id>")
def status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "unknown job"}), 404
        # deep-ish copy is unnecessary; results are only ever appended, not mutated
        return jsonify({
            "file": job["file"],
            "engines": job["engines"],
            "done": all(e["status"] != "running" for e in job["engines"].values()),
        })


@app.route("/uploads/<path:name>")
def uploaded(name):
    return send_from_directory(UPLOADS, name)


if __name__ == "__main__":
    print("Speech engine comparison UI  ->  http://127.0.0.1:5001")
    app.run(host="127.0.0.1", port=5001, threaded=True, debug=False)
