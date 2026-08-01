#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-time setup for the Speech Engine Shootout web UI.
#
# Creates the four Python virtual environments (main + three isolated engine
# venvs) and installs each one's dependencies. Safe to re-run.
#
# Requires: macOS on Apple Silicon, Homebrew, and `uv`.
#   - uv:      https://docs.astral.sh/uv/  (curl -LsSf https://astral.sh/uv/install.sh | sh)
#   - ffmpeg:  brew install ffmpeg
#
# Usage:  ./setup.sh
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")"

say()  { printf "\n\033[1;34m==>\033[0m %s\n" "$*"; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
warn() { printf "  \033[33m!\033[0m %s\n" "$*"; }

# --- prerequisites ---------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' is not installed. Install it, then re-run:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
ok "uv found"

if ! command -v ffmpeg >/dev/null 2>&1; then
  warn "ffmpeg not found — installing via Homebrew"
  brew install ffmpeg || { echo "ERROR: install ffmpeg manually (brew install ffmpeg)"; exit 1; }
fi
ok "ffmpeg found"

[ "$(uname -m)" = "arm64" ] || warn "Not Apple Silicon — the Parakeet engine (MLX) will not work; the others will."

# --- helper: build one venv, tolerate failure on the optional ones ---------
build () {  # build <venv-dir> <requirements-file> <required|optional>
  local venv="$1" reqs="$2" mode="$3"
  say "Setting up $venv  ($reqs)"
  uv venv --python 3.12 "$venv" >/dev/null 2>&1 || { warn "could not create $venv"; [ "$mode" = required ] && exit 1; return; }
  if uv pip install --python "$venv/bin/python" -r "$reqs"; then
    ok "$venv ready"
  else
    if [ "$mode" = required ]; then
      echo "ERROR: failed to install $reqs into $venv"; exit 1
    else
      warn "$venv failed — that engine will be unavailable, the rest still work"
    fi
  fi
}

# --- main venv (required): faster-whisper, ECAPA diarization, prosody, Flask -
build .venv            requirements.txt          required

# --- isolated engine venvs (optional; each adds engines to the comparison) --
build .venv-parakeet   requirements-parakeet.txt optional   # Parakeet transcription (Apple GPU)
build .venv-nemo       requirements-nemo.txt     optional   # Sortformer diarization (ungated)
build .venv-denoise    requirements-denoise.txt  optional   # audio cleanup (dereverb + denoise)
build .venv-whisperx   requirements-whisperx.txt optional   # whisperX + pyannote (needs HF token)

# --- token file ------------------------------------------------------------
if [ ! -f .env ]; then
  cp .env.example .env
  ok "created .env  (add a HuggingFace token only if you want the whisperX engines)"
fi

say "Done."
echo "Start the app with:"
echo "    .venv/bin/python webui/app.py"
echo "Then open  http://127.0.0.1:5001"
echo
echo "First run downloads the speech models (a few GB) — after that it works offline."
