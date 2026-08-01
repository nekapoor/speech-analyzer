# Speech Analyzer — transcription + diarization, compared

A local tool for reviewing counseling encounters. It transcribes a recording,
labels **who spoke** (student vs. patient), and lets you compare several
transcription + diarization engines side by side to see which does the best job on
*your* audio.

There are two ways to use it:
- **Web UI** (recommended) — upload a recording, pick engines, compare results in
  your browser. See [`webui/README.md`](webui/README.md).
- **Command line** — `analyze_speech.py` for a single file or a folder, with prosody
  metrics (pitch, loudness, rate, pauses). See [`README_INSTALL.md`](README_INSTALL.md).

---

## 🔒 Privacy / FERPA — read this first

**Everything runs on your own computer. No audio ever leaves the machine.**

- The web app listens on `127.0.0.1` (localhost) — it is not reachable from the
  network or the internet. "Upload" just means your browser handing the file to the
  program running on the *same computer*.
- Uploaded audio is **auto-deleted** from disk the moment analysis finishes.
- The only network use is a one-time download of the speech models. After that you
  can **turn off the internet** and it still works — the strongest proof nothing is
  being sent anywhere.
- This repository contains **code only**. No recordings, transcripts, or tokens are
  in it (see `.gitignore`). Keep it that way: never commit anything from `out/`,
  `webui/uploads/`, or `.env`.

Each person runs their own private copy. Recordings are never shared between
machines.

---

## Requirements

- **A Mac with Apple Silicon** (M1/M2/M3/M4). The Parakeet engine needs it; the rest
  are cross-platform but this is only tested on macOS.
- [Homebrew](https://brew.sh) and **ffmpeg** (`brew install ffmpeg`)
- [`uv`](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

## Setup (one time)

```bash
git clone https://github.com/nekapoor/<REPO>.git speech-analyzer
cd speech-analyzer
./setup.sh
```

`setup.sh` builds four Python environments and installs everything. First run of the
app downloads the speech models (a few GB) — after that it works offline.

## Run the web UI

```bash
.venv/bin/python webui/app.py
```

Open **http://127.0.0.1:5001**, choose a recording, pick which engines to run, and
compare. The four token-free engines are pre-selected.

---

## Hear what the cleanup does (before trusting it)

The "Clean audio first" toggle dereverbs + denoises your audio before diarization,
but the web app deletes that cleaned audio when it's done — so you can't hear it. To
actually *listen* to the before/after on a real recording, run:

```bash
.venv-denoise/bin/python clean_audio.py "OSCE encounter 3.mp4"
```

It writes two files next to your recording — `...original.wav` and `...cleaned.wav` —
that you open in QuickTime and compare by ear. No diarization, nothing deleted, nothing
leaves your computer. Use it to decide whether the cleanup is worth turning on.

## The engines

| Engine | Transcription | Diarization | Needs a token? |
|--------|---------------|-------------|----------------|
| faster-whisper + ECAPA | Whisper | ECAPA→KMeans (forced 2 speakers) | no |
| **Parakeet-TDT** | NVIDIA Parakeet (Apple GPU) | — (transcription only) | no |
| **faster-whisper + Sortformer** | Whisper | NeMo Sortformer (overlap-aware) | no |
| **Parakeet + Sortformer** | NVIDIA Parakeet | NeMo Sortformer | no |
| whisperX + community-1 | Whisper + alignment | pyannote community-1 | yes (HuggingFace) |
| whisperX + pyannote 3.1 | Whisper + alignment | pyannote 3.1 | yes (HuggingFace) |

**Start with Parakeet + Sortformer** — it's the most accurate stack that needs no
account, no token, and no signup. The whisperX/pyannote engines are optional and
need a free HuggingFace token (see [`webui/README.md`](webui/README.md)).

---

## Something break?

Copy this into ChatGPT or Claude, then paste your error:

> I have a local Python project (a Flask web app in `webui/` plus CLI scripts) that
> transcribes audio and labels speakers. It uses four `uv`-created virtualenvs
> (`.venv`, `.venv-parakeet`, `.venv-nemo`, `.venv-whisperx`), needs `ffmpeg`, and
> runs on an Apple Silicon Mac. I set it up with `./setup.sh` and run it with
> `.venv/bin/python webui/app.py`. Everything is local. Here's my problem: [paste].
