#!/usr/bin/env python3
"""whisperX engine — runs INSIDE .venv-whisperx as a subprocess.

Isolated on purpose: whisperX pins its own torch/ctranslate2, which would fight
the main .venv. The web server never imports this; it shells out to
`.venv-whisperx/bin/python whisperx_runner.py ...` and reads the JSON we write to
--out.

Pipeline: transcribe (faster-whisper under the hood) -> forced alignment
(tighter word timestamps) -> diarization via pyannote (needs HF token).

All whisperX/pyannote console noise goes to stderr; the ONLY thing we write to
--out is the normalized result dict, so the parent's parse can't be corrupted.
"""

import argparse, json, os, sys, time

# common.py lives one dir up in the import path we set from the parent; but since
# we run standalone, re-implement the tiny normalizer here to avoid cross-venv
# import games.
def _normalize(segments):
    order = {}
    for s in segments:
        spk = s.get("speaker")
        if spk is not None and spk not in order:
            order[spk] = f"SPEAKER_{len(order)}"
    for s in segments:
        if s.get("speaker") is not None:
            s["speaker"] = order[s["speaker"]]
    return len(order)


def log(msg):
    print(f"[whisperx] {msg}", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--out", required=True, help="where to write the result JSON")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--language", default="en")
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)
    ap.add_argument("--diar-model", default=None,
                    help="pyannote pipeline to use (default: whisperX's community-1)")
    ap.add_argument("--no-diarize", action="store_true")
    args = ap.parse_args()

    import whisperx

    device = "cpu"
    compute_type = "int8"  # CPU-friendly; ctranslate2 has no MPS backend
    t0 = time.time()

    log(f"loading model {args.model} ...")
    model = whisperx.load_model(
        args.model, device=device, compute_type=compute_type, language=args.language
    )
    log("loading audio ...")
    audio = whisperx.load_audio(args.audio)

    log("transcribing ...")
    tr = model.transcribe(audio, batch_size=8, language=args.language)
    language = tr.get("language", args.language)

    log("aligning (forced phoneme alignment) ...")
    try:
        model_a, meta = whisperx.load_align_model(language_code=language, device=device)
        aligned = whisperx.align(
            tr["segments"], model_a, meta, audio, device,
            return_char_alignments=False,
        )
        segments = aligned["segments"]
    except Exception as e:  # alignment is a nicety, not required for comparison
        log(f"alignment skipped ({e})")
        segments = tr["segments"]

    diar_desc = "none (transcription only)"
    if not args.no_diarize:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not token:
            log("no HF token in env -> skipping diarization")
            diar_desc = "SKIPPED: no HuggingFace token provided"
        else:
            log("diarizing (pyannote) ...")
            try:
                # class location moved across versions; try the known spots
                try:
                    from whisperx.diarize import DiarizationPipeline
                except Exception:
                    from whisperx import DiarizationPipeline  # older layout
                dp_kw = {"token": token, "device": device}
                if args.diar_model:
                    dp_kw["model_name"] = args.diar_model
                diar = DiarizationPipeline(**dp_kw)
                kw = {}
                if args.min_speakers is not None:
                    kw["min_speakers"] = args.min_speakers
                if args.max_speakers is not None:
                    kw["max_speakers"] = args.max_speakers
                diar_segments = diar(audio, **kw)
                res = whisperx.assign_word_speakers(diar_segments, {"segments": segments})
                segments = res["segments"]
                model_short = (args.diar_model or "community-1").split("/")[-1]
                diar_desc = f"pyannote {model_short}"
            except Exception as e:
                log(f"diarization FAILED: {e}")
                diar_desc = f"FAILED: {e}"

    out_segments = [
        {
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "speaker": s.get("speaker"),
            "text": (s.get("text") or "").strip(),
        }
        for s in segments
        if s.get("text")
    ]
    n = _normalize(out_segments)

    _diar_short = (args.diar_model or "community-1").split("/")[-1]
    result = {
        "engine": f"whisperX + {_diar_short}",
        "model": args.model,
        "language": language,
        "elapsed_sec": round(time.time() - t0, 1),
        "num_speakers": n,
        "diarization": diar_desc,
        "segments": [
            {"start": round(s["start"], 2), "end": round(s["end"], 2),
             "speaker": s["speaker"], "text": s["text"]}
            for s in out_segments
        ],
    }
    with open(args.out, "w") as f:
        json.dump(result, f)
    log(f"done in {result['elapsed_sec']}s, {n} speaker(s), {len(out_segments)} segments")


if __name__ == "__main__":
    main()
