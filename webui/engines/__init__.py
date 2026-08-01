"""Engine registry.

Each engine is a callable run(audio_path, **opts) -> common result dict.
faster-whisper runs in-process; whisperX/Parakeet/Sortformer run in isolated venvs
via subprocess. `default_on` controls which cards are pre-checked in the UI (the
ungated, no-setup ones); gated whisperX variants are opt-in.
"""

REGISTRY = {
    "fw_ecapa": {
        "label": "faster-whisper + ECAPA",
        "diar": "ECAPA + KMeans (token-free, forced 2 speakers)",
        "needs_token": False,
        "diar_model": None,
        "default_on": True,
    },
    "whisperx": {
        "label": "whisperX + community-1",
        "diar": "pyannote community-1 (auto count; needs HF access to community-1)",
        "needs_token": True,
        "diar_model": None,  # None -> whisperX default = community-1
        "default_on": False,
    },
    "whisperx_31": {
        "label": "whisperX + pyannote 3.1",
        "diar": "pyannote 3.1 (needs HF access to speaker-diarization-3.1 AND segmentation-3.0)",
        "needs_token": True,
        "diar_model": "pyannote/speaker-diarization-3.1",
        "default_on": False,
    },
    "parakeet": {
        "label": "Parakeet-TDT (MLX)",
        "diar": "none — transcription only (SOTA English ASR, Apple GPU, ungated)",
        "needs_token": False,
        "diar_model": None,
        "default_on": True,
    },
    "fw_sortformer": {
        "label": "faster-whisper + Sortformer",
        "diar": "NeMo Sortformer (ungated, no token, overlap-aware)",
        "needs_token": False,
        "diar_model": None,
        "default_on": True,
    },
    "parakeet_sortformer": {
        "label": "Parakeet + Sortformer",
        "diar": "NeMo Sortformer (ungated, no token, overlap-aware)",
        "needs_token": False,
        "diar_model": None,
        "default_on": True,
    },
}
