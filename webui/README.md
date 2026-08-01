# Speech Engine Shootout — web UI

A local web app to compare transcription **and** diarization engines side by side.
Upload one recording, pick which engines to run, and eyeball which does the better
job on *your* audio.

Everything runs on `localhost`. Nothing is uploaded anywhere. The only network use
is one-time model downloads.

## Engines

| Engine | Transcription | Diarization | Token? | Venv |
|--------|---------------|-------------|--------|------|
| **faster-whisper + ECAPA** | faster-whisper | ECAPA → KMeans, **forced 2 speakers** | none | `.venv` |
| **whisperX + community-1** | faster-whisper + alignment | pyannote community-1, auto count | HF (gated) | `.venv-whisperx` |
| **whisperX + pyannote 3.1** | faster-whisper + alignment | pyannote 3.1 (+ segmentation-3.0) | HF (gated) | `.venv-whisperx` |
| **Parakeet-TDT (MLX)** | NVIDIA Parakeet on Apple GPU | none (transcription only) | none | `.venv-parakeet` |
| **faster-whisper + Sortformer** | faster-whisper | NeMo Sortformer, overlap-aware, auto count | none | `.venv` + `.venv-nemo` |
| **Parakeet + Sortformer** | NVIDIA Parakeet | NeMo Sortformer | none | `.venv-parakeet` + `.venv-nemo` |

Two axes to compare:

- **Transcription:** Whisper vs. **Parakeet-TDT** (tops the open English ASR
  leaderboard; runs on the M-series GPU via MLX).
- **Diarization:** token-free-but-forced-2 (ECAPA) vs. state-of-the-art-gated
  (pyannote) vs. **ungated + overlap-aware** (NeMo Sortformer).

**Parakeet + Sortformer** is the fully-local, no-token, near-SOTA stack — nothing
gated, nothing to sign up for. The UI pre-checks the four ungated engines; the
whisperX/pyannote ones are opt-in (they need the HF setup below).

Diarizers are paired with transcribers by **time-overlap assignment**: diarize the
audio independently into speaker turns, then label each transcript segment with the
speaker it overlaps most. That's why any transcriber can pair with any diarizer.

## Run it

```bash
cd /Users/neerajkapoor/Projects/speech-analyzer
.venv/bin/python webui/app.py
# open http://127.0.0.1:5001
```

Pick `small.en` for quick tests; switch to `large-v3` for real accuracy (first
use downloads ~3 GB).

## whisperX diarization: one-time token setup

The UI reads a token automatically from the project-root **`.env`**
(`HF_ACCESS_TOKEN=hf_...`), from the environment, or from what you paste in the field
(saved to `webui/.secrets/hf_token`, `chmod 600`).

**Use a classic *read* token.** Fine-grained tokens 403 on gated repos unless you
explicitly enable the "Read access to contents of all public gated repos" permission —
even after you accept the model terms. Create one at
<https://huggingface.co/settings/tokens> (type: **Read**).

Then accept the terms ("Agree and access") for whichever engine you want — approval is
usually instant:

| Engine | Accept terms on |
|--------|-----------------|
| **whisperX + community-1** | <https://huggingface.co/pyannote/speaker-diarization-community-1> |
| **whisperX + pyannote 3.1** | <https://huggingface.co/pyannote/speaker-diarization-3.1> **and** <https://huggingface.co/pyannote/segmentation-3.0> |

> The 3.1 pipeline depends on the separate **`segmentation-3.0`** model — accepting
> only the 3.1 page still 403s on segmentation-3.0. Accept **both**.

Without access, whisperX still transcribes + aligns; its diarization cell shows the
`FAILED: ...GatedRepoError` reason so you know exactly which page to visit.

Without a token, whisperX still runs — it just transcribes + aligns and its
diarization column says "SKIPPED: no HuggingFace token".

## Min / max speakers

Only whisperX uses these (pyannote can auto-detect count, or you can bound it).
The ECAPA engine ignores them — it's hard-wired to split into exactly 2 speakers.

## How it's wired

- `app.py` — Flask server + in-memory job queue; each engine runs in its own thread.
- `engines/common.py` — the shared result schema every engine normalizes to.
- `engines/fw_ecapa.py` — reuses the diarization code from `../analyze_speech.py`
  (runs in the main `.venv`).
- `engines/whisperx_runner.py` — runs **inside** `.venv-whisperx` (isolated deps);
  `engines/whisperx_engine.py` shells out to it and reads back JSON.
