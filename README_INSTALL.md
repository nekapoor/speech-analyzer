# Speech Analyzer — install & run

Local transcription + speaker labeling (student vs. patient) + prosody analysis
of counseling recordings. Outputs a `.json` and a compact `.txt` per file.

---

## Privacy: nothing goes to the cloud

- **The analysis is 100% local.** No cloud APIs, no LLM, no accounts, no
  telemetry. Nothing you feed it — audio, transcripts, metrics — ever leaves the
  computer. It runs on the CPU.
- **Token-free.** Speaker labeling uses a public speaker model, so there is no
  Hugging Face token or login to set up.
- **The only network use is downloads, once:** the first run fetches the speech
  model (`large-v3`, ~3 GB) and the speaker model (~80 MB) from Hugging Face and
  caches them. That's software coming *in* — never recording data going *out*.
  After that first run it works offline.

---

## Requirements

- **Python 3.11 or 3.12** (not 3.13+ — the ML packages don't have wheels yet).
- **ffmpeg** (an audio/video tool, installed separately from Python).

### Install Python + ffmpeg

**macOS** (install [Homebrew](https://brew.sh) first if needed):
```bash
brew install python@3.12 ffmpeg
```

**Windows:**
1. Install Python 3.12 from https://www.python.org/downloads/ — **check "Add
   Python to PATH"** during setup.
2. Install ffmpeg: easiest is `winget install Gyan.FFmpeg` in PowerShell, or
   download from https://www.gyan.dev/ffmpeg/builds/ and add its `bin` folder to
   PATH.

---

## Setup (one time)

From inside this `speech-analyzer` folder:

**macOS / Linux:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The install pulls in ~1.5 GB of packages (PyTorch is the big one) and takes a
few minutes.

---

## Run

Point it at a single file **or** a whole folder of recordings:

```bash
# one file
python analyze_speech.py "recording.mp3"

# a folder (processes every audio/video file inside)
python analyze_speech.py "path/to/recordings" --out results

# if the patient speaks first, flip the labels
python analyze_speech.py "recording.mp3" --patient-first
```

Results land in the `out/` folder (or whatever you pass to `--out`):
- `<name>.json` — full structured metrics
- `<name>.txt` — one readable line per turn (speaker, pitch, loudness, rate,
  pauses, etc.)

Supported inputs: mp3, mp4, wav, m4a, mov, aac, flac, mkv.

**Speed:** with the `large-v3` model, analysis runs a bit slower than real time
on a laptop CPU — roughly 5–8 minutes for a 7-minute recording. The first run is
slower still because it downloads the models.

---

## The two scripts

- **`analyze_speech.py`** — transcription **+** student/patient labels. Use this.
- **`analyze_speech_nodiar.py`** — same transcription, no speaker labels. Needs
  fewer packages (no PyTorch), so it's lighter to install and a touch faster.
  If you only ever use this one, you can delete the four "Diarization" lines
  from `requirements.txt` before installing.

## Choosing a smaller/faster model (optional)

`large-v3` is the most accurate. If a machine is slow, you can trade some
accuracy for speed with `--model`:
```bash
python analyze_speech.py "recording.mp3" --model medium.en   # faster, ~1.5 GB
python analyze_speech.py "recording.mp3" --model small.en    # fastest, ~465 MB
```
