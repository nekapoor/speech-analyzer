#!/usr/bin/env python3
"""Analyze a diarized, timestamped transcript together with its audio.

Accepted caption examples (one segment per line):

    [00:02.940 --> 00:05.700] STUDENT: Hi, my name is Tiffany.
    00:05.840 --> 00:07.460 | STUDENT | Can you confirm your name?
    PATIENT [00:07.880 - 00:11.680]: My name is Mary Smith.

SRT/VTT-like blocks are also accepted:

    00:02.940 --> 00:05.700
    STUDENT: Hi, my name is Tiffany.

The program produces JSON and CSV evidence files plus an overview chart.  Its
measurements are descriptive; they should not be used to infer emotion,
personality, gender, health status, or clinical correctness.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import librosa
    import matplotlib.pyplot as plt
    import numpy as np
    import parselmouth
    from parselmouth.praat import call as praat_call
except ImportError as exc:  # pragma: no cover - exercised by missing dependency
    raise SystemExit(
        "Missing dependency. Install with:\n"
        "  python -m pip install librosa numpy matplotlib soundfile praat-parselmouth\n"
        f"Original error: {exc}"
    ) from exc


EPSILON = 1e-12
TIMESTAMP = r"(?:(?:\d{1,2}:)?\d{1,2}:\d{2}(?:[.,]\d{1,3})?)"
RANGE_RE = re.compile(
    rf"(?P<start>{TIMESTAMP})\s*(?:-->|-|–|—)\s*(?P<end>{TIMESTAMP})"
)
WORD_RE = re.compile(r"\b(?:[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?)\b")

HESITATION_RE = re.compile(r"\b(?:um+|uh+|erm+|er+|hmm+)\b", re.IGNORECASE)
DISCOURSE_PATTERNS = {
    "like": re.compile(r"\blike\b", re.IGNORECASE),
    "you know": re.compile(r"\byou\s+know\b", re.IGNORECASE),
    "i mean": re.compile(r"\bi\s+mean\b", re.IGNORECASE),
    "so": re.compile(r"\bso\b", re.IGNORECASE),
    "well": re.compile(r"\bwell\b", re.IGNORECASE),
    "okay": re.compile(r"\bok(?:ay)?\b", re.IGNORECASE),
    "yeah": re.compile(r"\byeah\b", re.IGNORECASE),
    "actually": re.compile(r"\bactually\b", re.IGNORECASE),
    "basically": re.compile(r"\bbasically\b", re.IGNORECASE),
}
OPEN_QUESTION_RE = re.compile(
    r"^\s*(?:what|how|why|tell me|describe|explain|walk me through)\b",
    re.IGNORECASE,
)
CLOSED_QUESTION_RE = re.compile(
    r"^\s*(?:do|does|did|are|is|was|were|can|could|would|will|have|has)\b",
    re.IGNORECASE,
)
TEACH_BACK_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bin your own words\b",
        r"\btell me how\b",
        r"\bshow me how\b",
        r"\bwalk me through\b",
        r"\bhow will you\b",
        r"\bwhat will you do\b",
    )
]
EMPATHY_PHRASE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bi(?:'m| am) sorry\b",
        r"\bthat sounds\b",
        r"\bi understand\b",
        r"\bthat must\b",
        r"\bi can see\b",
        r"\bthank you for (?:telling|sharing)\b",
        r"\bit makes sense\b",
    )
]

GLOBAL_RUBRIC_TEXT = """GLOBAL RUBRIC SCORING CRITERIA — give ONE score for each category
Excellent (3 points): Student did not perform poorly on ANY listed item in that category.
Satisfactory (2 points): Student performed poorly on ONE listed item in that category.
Needs Improvement (1 point): Student performed poorly on TWO OR MORE, BUT NOT ALL, listed items in that category.
Failure (0 points): Student performed poorly on ALL listed items in that category.

Critical grading instructions:
- Grade a second-year (P2) pharmacy student at that level of experience.
- Grade the Global Rubric independently of clinical or analytical correctness.
- Do not lower communication scores for clinical errors or missed clinical items.
- Do not penalize normal conversational filler; count filler only when genuinely distracting.

Category 1 — Verbal Expression: Mechanics (HOW)
- Speaks with proper grammar and fluency.
- Uses filler words only at a normal conversational level, not a distracting one.
- Speaks with an appropriate rate of speech.
- Uses appropriate volume for the context.
- Speaks with appropriate modulation to convey the message effectively.

Category 2 — Verbal Expression: Content (WHAT)
- Selects and uses vocabulary appropriate for the context.
- Uses vocabulary appropriate for the audience.
- Uses lay language with patients; avoids medical terms and abbreviations.
- Uses open-ended questions.

Category 3 — Non-Verbal Expression
- Maintains appropriate eye contact, with brief necessary breaks.
- Sits or stands upright with professional posture.
- Does not engage in distracting gestures.
- Does not create awkward silences; silence at the end does not count.
- Maintains a comfortable physical distance for the context.

Category 4 — Interaction with Patient
- Displays active listening.
- Displays empathy and sensitivity appropriate for the context.
- Conducts the interaction in a non-formulaic manner.
- Demonstrates perceptiveness by responding to cues and situations appropriately.
- Shows respect; avoids a hostile or condescending manner.
- Displays professional confidence without arrogance.
- Maintains professional mannerism.

Category 5 — Organization & Logic
- Presents information in a logical order.
- Information flows smoothly with good transitions.
- Shows flexibility to reorganize when unexpected information appears.
- Maintains control and returns to the topic when the patient distracts.

Category 6 — Professional Appearance and Rapport
- Provides an appropriate introductory greeting, including name and position for new encounters.
- Attire and overall appearance are professional.
- Confirms understanding using pertinent, specific teach-back questions.
- Ends the session appropriately with proper closure.

Final global question:
Choose Excellent, Satisfactory, or Needs Improvement based on the six category scores. Choose the level assigned most often; if two levels tie, choose the lower one.
"""


@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class FrameFeatures:
    times: np.ndarray
    rms_dbfs: np.ndarray
    f0_hz: np.ndarray
    voiced_probability: np.ndarray
    spectral_centroid_hz: np.ndarray
    spectral_bandwidth_hz: np.ndarray
    spectral_rolloff_hz: np.ndarray
    spectral_flatness: np.ndarray
    zero_crossing_rate: np.ndarray
    activity_threshold_dbfs: float
    hop_length: int
    frame_length: int


def timestamp_to_seconds(value: str) -> float:
    parts = value.strip().replace(",", ".").split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"Unsupported timestamp: {value!r}")


def clean_speaker(value: str) -> str:
    value = value.strip().strip("[]()|:- \t")
    return re.sub(r"\s+", " ", value) or "UNKNOWN"


def split_speaker_text(value: str) -> tuple[str, str]:
    value = value.strip().strip("[] ")

    # Pipe-separated form: | SPEAKER | words here
    pipe_parts = [part.strip() for part in value.strip("|").split("|")]
    if len(pipe_parts) >= 2 and pipe_parts[0] and pipe_parts[1]:
        return clean_speaker(pipe_parts[0]), " | ".join(pipe_parts[1:]).strip()

    # SPEAKER: words, [SPEAKER]: words, or SPEAKER<TAB>words.
    match = re.match(
        r"^\[?(?P<speaker>[A-Za-z][A-Za-z0-9_. -]{0,60})\]?\s*(?::|\t)\s*(?P<text>.+)$",
        value,
    )
    if match:
        return clean_speaker(match.group("speaker")), match.group("text").strip()

    return "UNKNOWN", value.strip(" :-|\t")


def parse_captions(path: Path) -> list[Segment]:
    raw_lines = path.read_text(encoding="utf-8-sig").splitlines()
    segments: list[Segment] = []
    pending_range: tuple[float, float] | None = None

    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line or line.upper() == "WEBVTT" or re.fullmatch(r"\d+", line):
            continue

        timing = RANGE_RE.search(line)
        if timing:
            start = timestamp_to_seconds(timing.group("start"))
            end = timestamp_to_seconds(timing.group("end"))
            if end < start:
                raise ValueError(f"Caption ends before it begins: {raw_line}")

            remainder = (line[: timing.start()] + " " + line[timing.end() :]).strip()
            remainder = remainder.strip("[]()|:- \t")
            if remainder:
                speaker, text = split_speaker_text(remainder)
                segments.append(Segment(start, end, speaker, text))
                pending_range = None
            else:
                pending_range = (start, end)
            continue

        if pending_range is not None:
            speaker, text = split_speaker_text(line)
            segments.append(Segment(*pending_range, speaker, text))
            pending_range = None
            continue

        # Permit wrapped/multiline captions by adding text to the prior segment.
        if segments:
            segments[-1].text = f"{segments[-1].text} {line}".strip()

    if pending_range is not None:
        raise ValueError("The final caption timestamp has no speaker/text line.")
    if not segments:
        raise ValueError(
            "No captions were parsed. See the accepted formats in this script's docstring."
        )

    segments.sort(key=lambda item: (item.start, item.end, item.speaker))
    return segments


def merge_same_speaker_segments(
    segments: Sequence[Segment], max_gap_seconds: float
) -> list[Segment]:
    turns: list[Segment] = []
    for segment in segments:
        if (
            turns
            and turns[-1].speaker == segment.speaker
            and segment.start - turns[-1].end <= max_gap_seconds
        ):
            turns[-1].end = max(turns[-1].end, segment.end)
            turns[-1].text = f"{turns[-1].text} {segment.text}".strip()
        else:
            turns.append(Segment(**asdict(segment)))
    return turns


def safe_log10(value: float) -> float:
    return 20.0 * math.log10(max(abs(value), EPSILON))


def finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def percentile_or_none(values: np.ndarray, percentile: float) -> float | None:
    values = finite(np.asarray(values, dtype=float))
    if values.size == 0:
        return None
    return float(np.percentile(values, percentile))


def round_or_none(value: Any, digits: int = 3) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return round(numeric, digits)


def compute_frame_features(
    audio: np.ndarray,
    sample_rate: int,
    fmin_hz: float,
    fmax_hz: float,
) -> FrameFeatures:
    # Approximately 64 ms frames and 16 ms hops at the default 16 kHz rate.
    frame_length = 2048 if sample_rate >= 32000 else 1024
    hop_length = max(128, int(round(sample_rate * 0.016)))

    rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length, center=True
    )[0]
    rms_dbfs = librosa.amplitude_to_db(np.maximum(rms, EPSILON), ref=1.0)

    f0_hz, _voiced_flag, voiced_probability = librosa.pyin(
        audio,
        fmin=fmin_hz,
        fmax=fmax_hz,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )
    if voiced_probability is None:
        voiced_probability = np.full_like(f0_hz, np.nan)

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sample_rate, n_fft=frame_length, hop_length=hop_length
    )[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio, sr=sample_rate, n_fft=frame_length, hop_length=hop_length
    )[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sample_rate,
        n_fft=frame_length,
        hop_length=hop_length,
        roll_percent=0.85,
    )[0]
    spectral_flatness = librosa.feature.spectral_flatness(
        y=audio, n_fft=frame_length, hop_length=hop_length
    )[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        audio, frame_length=frame_length, hop_length=hop_length
    )[0]

    arrays = [
        rms_dbfs,
        f0_hz,
        voiced_probability,
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff,
        spectral_flatness,
        zero_crossing_rate,
    ]
    common_length = min(map(len, arrays))
    arrays = [array[:common_length] for array in arrays]
    times = librosa.frames_to_time(
        np.arange(common_length), sr=sample_rate, hop_length=hop_length
    )

    noise_floor = float(np.percentile(arrays[0], 10))
    speech_level = float(np.percentile(arrays[0], 90))
    separation = max(0.0, speech_level - noise_floor)
    activity_threshold = noise_floor + min(8.0, max(3.0, separation * 0.40))

    return FrameFeatures(
        times=times,
        rms_dbfs=arrays[0],
        f0_hz=arrays[1],
        voiced_probability=arrays[2],
        spectral_centroid_hz=arrays[3],
        spectral_bandwidth_hz=arrays[4],
        spectral_rolloff_hz=arrays[5],
        spectral_flatness=arrays[6],
        zero_crossing_rate=arrays[7],
        activity_threshold_dbfs=activity_threshold,
        hop_length=hop_length,
        frame_length=frame_length,
    )


def interval_mask(times: np.ndarray, intervals: Iterable[tuple[float, float]]) -> np.ndarray:
    mask = np.zeros(times.shape, dtype=bool)
    for start, end in intervals:
        mask |= (times >= start) & (times < end)
    return mask


def summarize_acoustics(
    features: FrameFeatures,
    mask: np.ndarray,
) -> dict[str, Any]:
    if not np.any(mask):
        return {
            "active_audio_fraction": None,
            "pause_fraction_estimate": None,
            "median_level_dbfs": None,
            "level_iqr_db": None,
            "level_5_95_range_db": None,
            "median_f0_hz": None,
            "pitch_iqr_semitones": None,
            "pitch_5_95_range_semitones": None,
            "voiced_frame_fraction": None,
            "median_spectral_centroid_hz": None,
            "median_spectral_bandwidth_hz": None,
            "median_spectral_rolloff_hz": None,
            "median_spectral_flatness": None,
            "median_zero_crossing_rate": None,
        }

    rms_db = features.rms_dbfs[mask]
    active = rms_db >= features.activity_threshold_dbfs
    active_fraction = float(np.mean(active)) if active.size else None

    f0 = features.f0_hz[mask]
    valid_f0 = finite(f0)
    median_f0 = float(np.median(valid_f0)) if valid_f0.size else None
    if median_f0 and median_f0 > 0:
        semitone_offsets = 12.0 * np.log2(valid_f0 / median_f0)
        pitch_iqr = float(np.percentile(semitone_offsets, 75) - np.percentile(semitone_offsets, 25))
        pitch_range = float(np.percentile(semitone_offsets, 95) - np.percentile(semitone_offsets, 5))
    else:
        pitch_iqr = None
        pitch_range = None

    def active_or_all(array: np.ndarray) -> np.ndarray:
        selected = array[mask]
        return selected[active] if np.any(active) else selected

    return {
        "active_audio_fraction": round_or_none(active_fraction),
        "pause_fraction_estimate": round_or_none(
            None if active_fraction is None else 1.0 - active_fraction
        ),
        "median_level_dbfs": round_or_none(np.median(rms_db)),
        "level_iqr_db": round_or_none(np.percentile(rms_db, 75) - np.percentile(rms_db, 25)),
        "level_5_95_range_db": round_or_none(
            np.percentile(rms_db, 95) - np.percentile(rms_db, 5)
        ),
        "median_f0_hz": round_or_none(median_f0),
        "pitch_iqr_semitones": round_or_none(pitch_iqr),
        "pitch_5_95_range_semitones": round_or_none(pitch_range),
        "voiced_frame_fraction": round_or_none(np.mean(np.isfinite(f0))),
        "median_spectral_centroid_hz": round_or_none(
            np.median(active_or_all(features.spectral_centroid_hz))
        ),
        "median_spectral_bandwidth_hz": round_or_none(
            np.median(active_or_all(features.spectral_bandwidth_hz))
        ),
        "median_spectral_rolloff_hz": round_or_none(
            np.median(active_or_all(features.spectral_rolloff_hz))
        ),
        "median_spectral_flatness": round_or_none(
            np.median(active_or_all(features.spectral_flatness))
        ),
        "median_zero_crossing_rate": round_or_none(
            np.median(active_or_all(features.zero_crossing_rate))
        ),
    }


def duration_weighted_mean(values: Sequence[tuple[float, float]]) -> float | None:
    valid = [
        (float(value), float(weight))
        for value, weight in values
        if math.isfinite(float(value)) and float(weight) > 0
    ]
    if not valid:
        return None
    total_weight = sum(weight for _, weight in valid)
    return sum(value * weight for value, weight in valid) / total_weight


def measure_praat_prosody(
    audio: np.ndarray,
    sample_rate: int,
    turns: Sequence[Segment],
    fmin_hz: float,
    fmax_hz: float,
) -> dict[str, Any]:
    """Measure speaker prosody with Praat on each continuous speaker turn.

    Turn-level processing avoids creating artificial pitch periods at the joins
    that would result from concatenating non-contiguous audio. Jitter, shimmer,
    and HNR are duration-weighted across turns long enough for voice-quality
    analysis.
    """

    pitch_values: list[float] = []
    intensity_values: list[float] = []
    jitter_values: list[tuple[float, float]] = []
    shimmer_values: list[tuple[float, float]] = []
    hnr_values: list[tuple[float, float]] = []
    voice_quality_turns_used = 0

    period_floor = 0.8 / fmax_hz
    period_ceiling = 1.25 / fmin_hz

    for turn in turns:
        start_sample = max(0, int(round(turn.start * sample_rate)))
        end_sample = min(len(audio), int(round(turn.end * sample_rate)))
        turn_audio = audio[start_sample:end_sample]
        if len(turn_audio) < int(0.10 * sample_rate):
            continue

        sound = parselmouth.Sound(
            np.asarray(turn_audio, dtype=np.float64), sampling_frequency=sample_rate
        )

        try:
            pitch = sound.to_pitch(
                time_step=0.01,
                pitch_floor=fmin_hz,
                pitch_ceiling=fmax_hz,
            )
            values = np.asarray(pitch.selected_array["frequency"], dtype=float)
            pitch_values.extend(values[np.isfinite(values) & (values > 0)].tolist())
        except Exception:
            pass

        try:
            intensity = sound.to_intensity(
                minimum_pitch=fmin_hz,
                time_step=0.01,
                subtract_mean=True,
            )
            values = np.asarray(intensity.values[0], dtype=float)
            intensity_values.extend(values[np.isfinite(values)].tolist())
        except Exception:
            pass

        # Very short turns do not contain enough periods for stable measures.
        if turn.duration < 0.50:
            continue

        try:
            point_process = praat_call(
                sound, "To PointProcess (periodic, cc)", fmin_hz, fmax_hz
            )
            jitter = praat_call(
                point_process,
                "Get jitter (local)",
                0,
                0,
                period_floor,
                period_ceiling,
                1.3,
            )
            shimmer = praat_call(
                [sound, point_process],
                "Get shimmer (local)",
                0,
                0,
                period_floor,
                period_ceiling,
                1.3,
                1.6,
            )
            harmonicity = sound.to_harmonicity_cc(
                time_step=0.01,
                minimum_pitch=fmin_hz,
                silence_threshold=0.1,
                periods_per_window=4.5,
            )
            hnr = praat_call(harmonicity, "Get mean", 0, 0)

            added = False
            if math.isfinite(float(jitter)):
                jitter_values.append((float(jitter) * 100.0, turn.duration))
                added = True
            if math.isfinite(float(shimmer)):
                shimmer_values.append((float(shimmer) * 100.0, turn.duration))
                added = True
            if math.isfinite(float(hnr)) and float(hnr) > -100:
                hnr_values.append((float(hnr), turn.duration))
                added = True
            if added:
                voice_quality_turns_used += 1
        except Exception:
            # Praat returns undefined values for some unvoiced/noisy turns.
            continue

    pitch_array = np.asarray(pitch_values, dtype=float)
    intensity_array = np.asarray(intensity_values, dtype=float)
    return {
        "pitch_mean_hz": round_or_none(np.mean(pitch_array) if pitch_array.size else None),
        "pitch_std_hz": round_or_none(np.std(pitch_array) if pitch_array.size else None),
        "pitch_range_hz": round_or_none(
            np.max(pitch_array) - np.min(pitch_array) if pitch_array.size else None
        ),
        # Praat intensity is referenced to its acoustic convention but is not
        # calibrated sound-pressure level when the source is an ordinary MP3.
        "intensity_mean_db": round_or_none(
            np.mean(intensity_array) if intensity_array.size else None
        ),
        "intensity_std_db": round_or_none(
            np.std(intensity_array) if intensity_array.size else None
        ),
        "jitter_pct": round_or_none(duration_weighted_mean(jitter_values)),
        "shimmer_pct": round_or_none(duration_weighted_mean(shimmer_values)),
        "hnr_db": round_or_none(duration_weighted_mean(hnr_values)),
        "voice_quality_turns_used": voice_quality_turns_used,
    }


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text)


def repeated_word_count(words: Sequence[str]) -> int:
    lowered = [word.lower() for word in words]
    return sum(first == second for first, second in zip(lowered, lowered[1:]))


def text_measurements(text: str) -> dict[str, Any]:
    words = tokenize(text)
    hesitation_count = len(HESITATION_RE.findall(text))
    discourse_counts = {
        name: len(pattern.findall(text)) for name, pattern in DISCOURSE_PATTERNS.items()
    }
    sentences = [
        piece.strip() for piece in re.split(r"(?<=[.!?])\s+", text) if piece.strip()
    ]
    question_sentences = [piece for piece in sentences if piece.endswith("?")]
    open_questions = sum(bool(OPEN_QUESTION_RE.search(item)) for item in question_sentences)
    closed_questions = sum(bool(CLOSED_QUESTION_RE.search(item)) for item in question_sentences)

    return {
        "word_count": len(words),
        "hesitation_count": hesitation_count,
        "discourse_marker_count": sum(discourse_counts.values()),
        "discourse_marker_breakdown": discourse_counts,
        "adjacent_repeated_word_count": repeated_word_count(words),
        "question_count": len(question_sentences),
        "open_question_candidate_count": open_questions,
        "closed_question_candidate_count": closed_questions,
        "teach_back_phrase_candidate_count": sum(
            len(pattern.findall(text)) for pattern in TEACH_BACK_PATTERNS
        ),
        "empathy_phrase_candidate_count": sum(
            len(pattern.findall(text)) for pattern in EMPATHY_PHRASE_PATTERNS
        ),
    }


def question_candidates(text: str) -> list[dict[str, str]]:
    sentences = [
        piece.strip() for piece in re.split(r"(?<=[.!?])\s+", text) if piece.strip()
    ]
    questions: list[dict[str, str]] = []
    for sentence in sentences:
        if not sentence.endswith("?"):
            continue
        if OPEN_QUESTION_RE.search(sentence):
            classification = "open_candidate"
        elif CLOSED_QUESTION_RE.search(sentence):
            classification = "closed_candidate"
        else:
            classification = "unclassified"
        questions.append({"text": sentence, "classification": classification})
    return questions


def matched_phrases(text: str, patterns: Sequence[re.Pattern[str]]) -> list[str]:
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(match.group(0) for match in pattern.finditer(text))
    return matches


def discourse_marker_matches(text: str) -> dict[str, int]:
    return {
        name: len(pattern.findall(text))
        for name, pattern in DISCOURSE_PATTERNS.items()
        if pattern.search(text)
    }


def detect_low_energy_pauses(
    features: FrameFeatures,
    turn: Segment,
    minimum_duration_seconds: float,
) -> list[dict[str, Any]]:
    frame_indices = np.flatnonzero(
        (features.times >= turn.start) & (features.times < turn.end)
    )
    if frame_indices.size == 0:
        return []

    inactive = (
        features.rms_dbfs[frame_indices] < features.activity_threshold_dbfs
    )
    hop_seconds = features.hop_length / 16_000.0
    # Correct the hop duration below if analysis used a non-default sample rate;
    # the frame-time differences are authoritative when available.
    if len(features.times) > 1:
        hop_seconds = float(np.median(np.diff(features.times)))

    pauses: list[dict[str, Any]] = []
    run_start: int | None = None
    for position, is_inactive in enumerate(inactive):
        if is_inactive and run_start is None:
            run_start = position
        is_last = position == len(inactive) - 1
        if run_start is not None and (not is_inactive or is_last):
            run_end = position if is_inactive and is_last else position - 1
            duration = (run_end - run_start + 1) * hop_seconds
            if duration >= minimum_duration_seconds:
                first_index = frame_indices[run_start]
                last_index = frame_indices[run_end]
                pause_start = max(turn.start, features.times[first_index] - hop_seconds / 2)
                pause_end = min(turn.end, features.times[last_index] + hop_seconds / 2)
                if pause_start <= turn.start + hop_seconds:
                    location = "turn_initial"
                elif pause_end >= turn.end - hop_seconds:
                    location = "turn_final"
                else:
                    location = "turn_internal"
                pauses.append(
                    {
                        "start_seconds": round(pause_start, 3),
                        "end_seconds": round(pause_end, 3),
                        "duration_seconds": round(max(0.0, pause_end - pause_start), 3),
                        "location": location,
                    }
                )
            run_start = None
    return pauses


def segment_row(segment: Segment, features: FrameFeatures) -> dict[str, Any]:
    text_stats = text_measurements(segment.text)
    mask = interval_mask(features.times, [(segment.start, segment.end)])
    acoustics = summarize_acoustics(features, mask)
    duration = segment.duration
    word_count = text_stats["word_count"]
    active_seconds = duration * (acoustics["active_audio_fraction"] or 0.0)

    return {
        "start_seconds": round(segment.start, 3),
        "end_seconds": round(segment.end, 3),
        "duration_seconds": round(duration, 3),
        "speaker": segment.speaker,
        "text": segment.text,
        **text_stats,
        "caption_speaking_rate_wpm": round_or_none(
            word_count * 60.0 / duration if duration > 0 else None
        ),
        "estimated_articulation_rate_wpm": round_or_none(
            word_count * 60.0 / active_seconds if active_seconds > 0 else None
        ),
        **acoustics,
    }


def union_duration(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return 0.0
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def overlap_duration(intervals: Iterable[tuple[float, float]]) -> float:
    events: list[tuple[float, int]] = []
    for start, end in intervals:
        if end > start:
            events.extend(((start, 1), (end, -1)))
    events.sort(key=lambda event: (event[0], event[1]))
    active = 0
    last_time: float | None = None
    overlap = 0.0
    for time, change in events:
        if last_time is not None and active >= 2:
            overlap += time - last_time
        active += change
        last_time = time
    return overlap


def speaker_summaries(
    segments: Sequence[Segment],
    turns: Sequence[Segment],
    features: FrameFeatures,
    audio: np.ndarray,
    sample_rate: int,
    audio_duration: float,
    fmin_hz: float,
    fmax_hz: float,
) -> list[dict[str, Any]]:
    by_speaker_segments: dict[str, list[Segment]] = defaultdict(list)
    by_speaker_turns: dict[str, list[Segment]] = defaultdict(list)
    response_latencies: dict[str, list[float]] = defaultdict(list)
    overlap_events: dict[str, int] = defaultdict(int)

    for segment in segments:
        by_speaker_segments[segment.speaker].append(segment)
    for turn in turns:
        by_speaker_turns[turn.speaker].append(turn)
    for previous, current in zip(turns, turns[1:]):
        if previous.speaker == current.speaker:
            continue
        latency = current.start - previous.end
        response_latencies[current.speaker].append(latency)
        if latency < -0.15:
            overlap_events[current.speaker] += 1

    summaries: list[dict[str, Any]] = []
    for speaker, speaker_segments in sorted(by_speaker_segments.items()):
        speaker_turns = by_speaker_turns[speaker]
        intervals = [(item.start, item.end) for item in speaker_turns]
        talk_seconds = union_duration(intervals)
        combined_text = " ".join(item.text for item in speaker_segments)
        text_stats = text_measurements(combined_text)
        mask = interval_mask(features.times, intervals)
        acoustics = summarize_acoustics(features, mask)
        active_seconds = talk_seconds * (acoustics["active_audio_fraction"] or 0.0)
        latencies = response_latencies[speaker]
        positive_latencies = [value for value in latencies if value >= 0]
        praat_prosody = measure_praat_prosody(
            audio,
            sample_rate,
            speaker_turns,
            fmin_hz,
            fmax_hz,
        )
        prosody = {
            "pitch_mean_hz": praat_prosody["pitch_mean_hz"],
            "pitch_std_hz": praat_prosody["pitch_std_hz"],
            "pitch_range_hz": praat_prosody["pitch_range_hz"],
            "intensity_mean_db": praat_prosody["intensity_mean_db"],
            "intensity_std_db": praat_prosody["intensity_std_db"],
            "speaking_rate_wps": round_or_none(
                text_stats["word_count"] / talk_seconds if talk_seconds > 0 else None
            ),
            "articulation_rate_wps": round_or_none(
                text_stats["word_count"] / active_seconds if active_seconds > 0 else None
            ),
            "pause_ratio": acoustics["pause_fraction_estimate"],
            "jitter_pct": praat_prosody["jitter_pct"],
            "shimmer_pct": praat_prosody["shimmer_pct"],
            "hnr_db": praat_prosody["hnr_db"],
        }

        summaries.append(
            {
                "speaker": speaker,
                "prosody": prosody,
                "prosody_quality": {
                    "voice_quality_turns_used": praat_prosody[
                        "voice_quality_turns_used"
                    ],
                    "intensity_is_calibrated_spl": False,
                },
                "turn_count": len(speaker_turns),
                "caption_segment_count": len(speaker_segments),
                "talk_time_seconds": round(talk_seconds, 3),
                "talk_time_fraction_of_recording": round_or_none(
                    talk_seconds / audio_duration if audio_duration else None
                ),
                "median_turn_duration_seconds": round_or_none(
                    np.median([item.duration for item in speaker_turns])
                ),
                "mean_turn_duration_seconds": round_or_none(
                    np.mean([item.duration for item in speaker_turns])
                ),
                **text_stats,
                "caption_speaking_rate_wpm": round_or_none(
                    text_stats["word_count"] * 60.0 / talk_seconds
                    if talk_seconds > 0
                    else None
                ),
                "estimated_articulation_rate_wpm": round_or_none(
                    text_stats["word_count"] * 60.0 / active_seconds
                    if active_seconds > 0
                    else None
                ),
                "hesitations_per_100_words": round_or_none(
                    text_stats["hesitation_count"] * 100.0 / text_stats["word_count"]
                    if text_stats["word_count"]
                    else None
                ),
                "discourse_markers_per_100_words": round_or_none(
                    text_stats["discourse_marker_count"]
                    * 100.0
                    / text_stats["word_count"]
                    if text_stats["word_count"]
                    else None
                ),
                "median_nonoverlap_response_latency_seconds": round_or_none(
                    np.median(positive_latencies) if positive_latencies else None
                ),
                "response_overlap_event_count": overlap_events[speaker],
                **acoustics,
            }
        )
    return summaries


def conversation_summary(
    segments: Sequence[Segment], turns: Sequence[Segment], audio_duration: float
) -> dict[str, Any]:
    intervals = [(item.start, item.end) for item in segments]
    transition_gaps = [
        current.start - previous.end
        for previous, current in zip(turns, turns[1:])
        if previous.speaker != current.speaker
    ]
    nonoverlap_gaps = [gap for gap in transition_gaps if gap >= 0]
    return {
        "speaker_count": len({item.speaker for item in segments}),
        "caption_segment_count": len(segments),
        "merged_turn_count": len(turns),
        "caption_coverage_seconds": round(union_duration(intervals), 3),
        "caption_coverage_fraction_of_audio": round_or_none(
            union_duration(intervals) / audio_duration if audio_duration else None
        ),
        "caption_overlap_seconds": round(overlap_duration(intervals), 3),
        "speaker_change_count": sum(
            previous.speaker != current.speaker
            for previous, current in zip(turns, turns[1:])
        ),
        "median_between_speaker_gap_seconds": round_or_none(
            np.median(nonoverlap_gaps) if nonoverlap_gaps else None
        ),
        "long_between_speaker_gap_count_over_2s": sum(gap > 2.0 for gap in nonoverlap_gaps),
        "overlapping_speaker_change_count": sum(gap < -0.15 for gap in transition_gaps),
    }


def audio_quality_summary(
    audio: np.ndarray, sample_rate: int, features: FrameFeatures
) -> dict[str, Any]:
    absolute = np.abs(audio)
    rms = math.sqrt(float(np.mean(np.square(audio))))
    noise_floor = float(np.percentile(features.rms_dbfs, 10))
    speech_level = float(np.percentile(features.rms_dbfs, 90))
    return {
        "duration_seconds": round(len(audio) / sample_rate, 3),
        "analysis_sample_rate_hz": sample_rate,
        "peak_level_dbfs": round_or_none(safe_log10(float(np.max(absolute)))),
        "overall_rms_level_dbfs": round_or_none(safe_log10(rms)),
        "frame_noise_floor_p10_dbfs": round_or_none(noise_floor),
        "frame_speech_level_p90_dbfs": round_or_none(speech_level),
        "estimated_speech_to_floor_separation_db": round_or_none(speech_level - noise_floor),
        "activity_threshold_dbfs": round_or_none(features.activity_threshold_dbfs),
        "clipped_sample_fraction": round_or_none(np.mean(absolute >= 0.999)),
        "near_clipped_sample_fraction": round_or_none(np.mean(absolute >= 0.98)),
        "dc_offset": round_or_none(np.mean(audio), 6),
    }


def resolve_speaker_name(
    requested: str | None, speaker_rows: Sequence[dict[str, Any]]
) -> str | None:
    if requested is None:
        return None
    for item in speaker_rows:
        if item["speaker"].casefold() == requested.casefold():
            return str(item["speaker"])
    return None


def build_llm_rubric_payload(
    report: dict[str, Any],
    turns: Sequence[Segment],
    features: FrameFeatures,
    requested_student_speaker: str | None,
    minimum_pause_seconds: float,
) -> dict[str, Any]:
    speaker_rows = report["speakers"]
    student_speaker = resolve_speaker_name(requested_student_speaker, speaker_rows)
    levels = {
        str(item["speaker"]): item.get("median_level_dbfs") for item in speaker_rows
    }

    compact_speakers: list[dict[str, Any]] = []
    for item in speaker_rows:
        other_levels = [
            float(level)
            for name, level in levels.items()
            if name != item["speaker"] and level is not None
        ]
        relative_level = (
            float(item["median_level_dbfs"]) - float(np.mean(other_levels))
            if item.get("median_level_dbfs") is not None and other_levels
            else None
        )
        compact_speakers.append(
            {
                "speaker": item["speaker"],
                "is_target_student": item["speaker"] == student_speaker,
                "turn_count": item["turn_count"],
                "word_count": item["word_count"],
                "talk_time_seconds": item["talk_time_seconds"],
                "talk_time_fraction_of_recording": item[
                    "talk_time_fraction_of_recording"
                ],
                "median_turn_duration_seconds": item[
                    "median_turn_duration_seconds"
                ],
                "hesitation_count": item["hesitation_count"],
                "hesitations_per_100_words": item[
                    "hesitations_per_100_words"
                ],
                "discourse_marker_count": item["discourse_marker_count"],
                "discourse_markers_per_100_words": item[
                    "discourse_markers_per_100_words"
                ],
                "adjacent_repeated_word_count": item[
                    "adjacent_repeated_word_count"
                ],
                "question_count": item["question_count"],
                "open_question_candidate_count": item[
                    "open_question_candidate_count"
                ],
                "closed_question_candidate_count": item[
                    "closed_question_candidate_count"
                ],
                "teach_back_phrase_candidate_count": item[
                    "teach_back_phrase_candidate_count"
                ],
                "empathy_phrase_candidate_count": item[
                    "empathy_phrase_candidate_count"
                ],
                "median_nonoverlap_response_latency_seconds": item[
                    "median_nonoverlap_response_latency_seconds"
                ],
                "response_overlap_event_count": item[
                    "response_overlap_event_count"
                ],
                "median_level_dbfs": item["median_level_dbfs"],
                "level_difference_vs_other_speakers_db": round_or_none(
                    relative_level
                ),
                "pitch_iqr_semitones": item["pitch_iqr_semitones"],
                "pitch_5_95_range_semitones": item[
                    "pitch_5_95_range_semitones"
                ],
                "prosody": item["prosody"],
            }
        )

    compact_turns: list[dict[str, Any]] = []
    candidate_events: list[dict[str, Any]] = []
    pause_events: list[dict[str, Any]] = []
    for turn_index, turn in enumerate(turns, start=1):
        measurements = segment_row(turn, features)
        hesitations = HESITATION_RE.findall(turn.text)
        questions = question_candidates(turn.text)
        teach_back_matches = matched_phrases(turn.text, TEACH_BACK_PATTERNS)
        empathy_matches = matched_phrases(turn.text, EMPATHY_PHRASE_PATTERNS)
        turn_pauses = detect_low_energy_pauses(
            features, turn, minimum_pause_seconds
        )
        for pause in turn_pauses:
            pause_events.append(
                {
                    "turn_id": turn_index,
                    "speaker": turn.speaker,
                    **pause,
                }
            )

        turn_evidence = {
            "turn_id": turn_index,
            "speaker": turn.speaker,
            "is_target_student": turn.speaker == student_speaker,
            "start_seconds": round(turn.start, 3),
            "end_seconds": round(turn.end, 3),
            "duration_seconds": round(turn.duration, 3),
            "text": turn.text,
            "word_count": measurements["word_count"],
            "caption_speaking_rate_wps": round_or_none(
                (measurements["caption_speaking_rate_wpm"] or 0.0) / 60.0
                if measurements["caption_speaking_rate_wpm"] is not None
                else None
            ),
            "estimated_articulation_rate_wps": round_or_none(
                (measurements["estimated_articulation_rate_wpm"] or 0.0) / 60.0
                if measurements["estimated_articulation_rate_wpm"] is not None
                else None
            ),
            "pause_ratio_estimate": measurements["pause_fraction_estimate"],
            "estimated_pause_count": len(turn_pauses),
            "hesitation_tokens": hesitations,
            "discourse_marker_candidates": discourse_marker_matches(turn.text),
            "question_candidates": questions,
            "teach_back_phrase_candidates": teach_back_matches,
            "empathy_phrase_candidates": empathy_matches,
            "median_level_dbfs": measurements["median_level_dbfs"],
            "level_variation_iqr_db": measurements["level_iqr_db"],
            "median_f0_hz": measurements["median_f0_hz"],
            "pitch_variation_iqr_semitones": measurements[
                "pitch_iqr_semitones"
            ],
        }
        compact_turns.append(turn_evidence)

        if hesitations or questions or teach_back_matches or empathy_matches:
            candidate_events.append(
                {
                    "turn_id": turn_index,
                    "speaker": turn.speaker,
                    "start_seconds": round(turn.start, 3),
                    "end_seconds": round(turn.end, 3),
                    "hesitation_tokens": hesitations,
                    "questions": questions,
                    "teach_back_phrases": teach_back_matches,
                    "empathy_phrases": empathy_matches,
                }
            )

    transitions: list[dict[str, Any]] = []
    for previous_index, (previous, current) in enumerate(
        zip(turns, turns[1:]), start=1
    ):
        gap = current.start - previous.end
        transitions.append(
            {
                "from_turn_id": previous_index,
                "to_turn_id": previous_index + 1,
                "from_speaker": previous.speaker,
                "to_speaker": current.speaker,
                "speaker_changed": previous.speaker != current.speaker,
                "gap_seconds": round(max(0.0, gap), 3),
                "overlap_seconds": round(max(0.0, -gap), 3),
                "previous_turn_end_seconds": round(previous.end, 3),
                "current_turn_start_seconds": round(current.start, 3),
                "previous_text": previous.text,
                "response_text": current.text,
            }
        )

    warnings: list[str] = []
    quality = report["audio_quality"]
    conversation = report["conversation"]
    if requested_student_speaker and student_speaker is None:
        warnings.append(
            f"Requested student speaker {requested_student_speaker!r} was not found."
        )
    if student_speaker is None:
        warnings.append(
            "Target student speaker is unset; the grading model must not guess the target."
        )
    if conversation["caption_coverage_fraction_of_audio"] is not None and conversation[
        "caption_coverage_fraction_of_audio"
    ] < 0.60:
        warnings.append(
            "Captions cover less than 60% of the audio; verify that silence or omitted speech explains the gap."
        )
    if quality["estimated_speech_to_floor_separation_db"] is not None and quality[
        "estimated_speech_to_floor_separation_db"
    ] < 10:
        warnings.append(
            "Speech-to-noise-floor separation is low; pause, volume, and pitch measurements may be unreliable."
        )
    if quality["near_clipped_sample_fraction"] and quality[
        "near_clipped_sample_fraction"
    ] > 0.001:
        warnings.append(
            "The recording contains substantial near-clipping; intensity and voice-quality measures may be distorted."
        )

    return {
        "schema_version": "1.0",
        "purpose": "Evidence packet for provisional grading of the pharmacy Global Communication Rubric.",
        "target_student_speaker": student_speaker,
        "requested_student_speaker": requested_student_speaker,
        "grading_scope": {
            "directly_supported": {
                "category_1": [
                    "grammar/fluency from transcript",
                    "filler context and frequency",
                    "speech-rate evidence",
                    "relative recorded level",
                    "pitch and level variability as limited modulation evidence",
                ],
                "category_2": [
                    "vocabulary and lay-language analysis from transcript",
                    "open-ended question evidence",
                ],
                "category_4": [
                    "turn-taking, cue-response, empathy language, respect, and responsiveness from transcript",
                ],
                "category_5": [
                    "sequence, transitions, flexibility, and topic control from transcript",
                ],
                "category_6": [
                    "introduction, teach-back, and closure from transcript"
                ],
            },
            "partially_supported": {
                "category_3": ["awkward silence evidence from audio/timestamps"],
                "category_4": [
                    "vocal prosody is supporting evidence only and cannot prove empathy, confidence, hostility, or arrogance"
                ],
            },
            "not_observable_from_audio_or_transcript": {
                "category_3": [
                    "eye contact",
                    "posture",
                    "distracting visible gestures",
                    "physical distance",
                ],
                "category_6": ["attire and overall visual appearance"],
            },
        },
        "grading_guardrails": [
            "Grade communication independently of clinical correctness or missing clinical content.",
            "Do not mark an item poor solely because a clinical answer is wrong or incomplete.",
            "Do not treat candidate phrase counts as conclusions; inspect the timestamped context.",
            "Do not penalize filler unless it is frequent enough to be genuinely distracting.",
            "Do not infer empathy, confidence, hostility, arrogance, emotion, personality, gender, health, or deception from pitch, intensity, jitter, shimmer, or HNR.",
            "Recorded intensity is not calibrated sound-pressure level; use only relative within-recording comparisons.",
            "Tightly cropped captions can inflate speaking and articulation rates.",
            "Mark unavailable visual criteria not_observable; never count them as poor.",
        ],
        "data_quality": {
            "audio_quality": quality,
            "conversation_coverage": conversation,
            "warnings": warnings,
        },
        "speaker_summaries": compact_speakers,
        "turn_timeline": compact_turns,
        "speaker_transition_evidence": transitions,
        "estimated_within_turn_pauses": pause_events,
        "candidate_language_events": candidate_events,
        "measurement_notes": [
            "All timestamp evidence is in seconds from the beginning of the audio.",
            "Low-energy pauses are threshold estimates and can be affected by microphone level and background noise.",
            "Question, empathy, filler, and teach-back fields are lexical candidates for model review, not final classifications.",
            "Jitter, shimmer, and HNR are exploratory on connected speech and are not direct Global Rubric criteria.",
            "This payload contains the transcript and may contain sensitive patient information; the script does not de-identify it.",
        ],
    }


def write_gemini_packet(path: Path, payload: dict[str, Any]) -> None:
    grading_instructions = """You are grading a second-year (P2) pharmacy student's communication. Use the rubric exactly.

Requirements:
1. Assess every listed rubric item as performed_well, performed_poorly, not_observable, or insufficient_evidence.
2. Give timestamped transcript/audio evidence for every poor assessment and important positive assessment.
3. Keep clinical correctness completely separate from communication. A clinically flawed plan may still communicate well.
4. Treat the automated measurements and phrase detections as leads, not conclusions. Resolve them by reading the surrounding turns.
5. Do not infer interpersonal traits or emotion from pitch, intensity, jitter, shimmer, or HNR.
6. Do not penalize normal conversational fillers.
7. Never count not_observable items as poor. If a category contains not_observable items, label its original-rubric score provisional and provide the possible score range under best- and worst-case assumptions for those items.
8. Category 3 visual items and Category 6 attire are unavailable in this packet. Do not invent observations.
9. For the final global rating, clearly state whether it is complete or provisional. Apply the modal-level rule only to category scores you can defend, and explain exclusions or assumptions.

Return a structured result with, for each category: item assessments, poor-item count among observable items, not-observable count, score or score range, level, confidence, and timestamped rationale. End with the final global rating and the three most useful coaching priorities.
"""
    rendered_payload = json.dumps(payload, indent=2, ensure_ascii=False)
    content = (
        "# Gemini Global Rubric Grading Packet\n\n"
        "## Grading instructions\n\n"
        f"{grading_instructions}\n"
        "## Global rubric\n\n"
        f"{GLOBAL_RUBRIC_TEXT}\n"
        "## Transcript and audio-evidence payload\n\n"
        "```json\n"
        f"{rendered_payload}\n"
        "```\n"
    )
    path.write_text(content, encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flattened = {
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
            writer.writerow(flattened)


def create_overview_plot(
    output_path: Path,
    audio: np.ndarray,
    sample_rate: int,
    turns: Sequence[Segment],
    speakers: Sequence[dict[str, Any]],
) -> None:
    names = [item["speaker"] for item in speakers]
    palette = plt.get_cmap("tab10")
    colors = {name: palette(index % 10) for index, name in enumerate(names)}

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(14, 9),
        gridspec_kw={"height_ratios": [2.2, 1.1, 1.5]},
        constrained_layout=True,
    )

    max_points = 25_000
    stride = max(1, len(audio) // max_points)
    sample_times = np.arange(0, len(audio), stride) / sample_rate
    axes[0].plot(sample_times, audio[::stride], color="#334155", linewidth=0.55)
    axes[0].set(title="Waveform and diarized turns", ylabel="Amplitude")
    axes[0].set_xlim(0, len(audio) / sample_rate)
    for turn in turns:
        axes[0].axvspan(
            turn.start,
            turn.end,
            color=colors.get(turn.speaker, "gray"),
            alpha=0.16,
            linewidth=0,
        )

    y_positions = {name: index for index, name in enumerate(names)}
    for turn in turns:
        axes[1].barh(
            y_positions[turn.speaker],
            turn.duration,
            left=turn.start,
            height=0.6,
            color=colors[turn.speaker],
        )
    axes[1].set(
        title="Speaker-turn timeline",
        xlabel="Seconds",
        yticks=range(len(names)),
        yticklabels=names,
        xlim=(0, len(audio) / sample_rate),
    )

    talk_times = [item["talk_time_seconds"] for item in speakers]
    bars = axes[2].bar(names, talk_times, color=[colors[name] for name in names])
    axes[2].set(title="Captioned talk time by speaker", ylabel="Seconds")
    axes[2].bar_label(bars, fmt="%.1f")
    axes[2].tick_params(axis="x", rotation=20)

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def build_report(
    audio_path: Path,
    captions_path: Path,
    segments: Sequence[Segment],
    turns: Sequence[Segment],
    audio: np.ndarray,
    sample_rate: int,
    features: FrameFeatures,
    same_speaker_gap_seconds: float,
    fmin_hz: float,
    fmax_hz: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    duration = len(audio) / sample_rate
    segment_rows = [segment_row(item, features) for item in segments]
    speaker_rows = speaker_summaries(
        segments,
        turns,
        features,
        audio,
        sample_rate,
        duration,
        fmin_hz,
        fmax_hz,
    )

    report = {
        "inputs": {
            "audio_file": str(audio_path.resolve()),
            "captions_file": str(captions_path.resolve()),
        },
        "audio_quality": audio_quality_summary(audio, sample_rate, features),
        "conversation": conversation_summary(segments, turns, duration),
        "speakers": speaker_rows,
        "settings": {
            "same_speaker_merge_gap_seconds": same_speaker_gap_seconds,
            "pitch_search_min_hz": fmin_hz,
            "pitch_search_max_hz": fmax_hz,
            "analysis_note": "Pitch and activity are estimated from compressed audio and timestamps.",
        },
        "interpretation_notes": [
            "Speaking rate uses caption words divided by captioned time; articulation rate uses estimated active-audio time.",
            "Pause/activity estimates depend on an automatically derived recording-level threshold.",
            "Pitch is reported in hertz plus within-speaker semitone variability; compare variability more readily than raw hertz across people.",
            "Jitter, shimmer, and HNR are measured by Praat within continuous turns of at least 0.5 seconds and duration-weighted by speaker.",
            "Praat intensity values from an MP3 are not calibrated physical sound-pressure levels; compare them only within recordings made under the same conditions.",
            "Lexical filler and empathy/teach-back phrase counts are candidate detectors and require contextual review.",
            "Overlap indicates timestamp overlap, not necessarily an interruption.",
            "Spectral measures are strongly affected by microphone, room, codec, and distance.",
            "Do not infer emotion, confidence, personality, gender, health, deception, or clinical competence from these acoustic features.",
        ],
    }
    return report, segment_rows, speaker_rows


def print_summary(report: dict[str, Any], selected_speaker: str | None) -> None:
    print(json.dumps({"audio_quality": report["audio_quality"], "conversation": report["conversation"]}, indent=2))
    speakers = report["speakers"]
    if selected_speaker:
        matches = [
            item for item in speakers if item["speaker"].casefold() == selected_speaker.casefold()
        ]
        if not matches:
            available = ", ".join(item["speaker"] for item in speakers)
            print(
                f"Warning: speaker {selected_speaker!r} not found. Available: {available}",
                file=sys.stderr,
            )
        speakers = matches
    print("\nSpeaker summary:")
    for item in speakers:
        print(json.dumps({"speaker": item["speaker"], "prosody": item["prosody"]}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze diarized timestamped captions and the corresponding audio."
    )
    parser.add_argument("--audio", required=True, type=Path, help="MP3/WAV/M4A audio file")
    parser.add_argument("--captions", required=True, type=Path, help="Timestamped diarized TXT file")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for JSON/CSV/PNG outputs")
    parser.add_argument(
        "--student-speaker",
        "--speaker",
        dest="student_speaker",
        help="Diarization label for the student being graded, for example STUDENT or SPEAKER_00",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16_000,
        help="Analysis sample rate (default: 16000)",
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=65.0,
        help="Minimum pitch searched by pYIN in Hz (default: 65)",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=500.0,
        help="Maximum pitch searched by pYIN in Hz (default: 500)",
    )
    parser.add_argument(
        "--same-speaker-gap",
        type=float,
        default=1.0,
        help="Merge adjacent same-speaker captions into one turn up to this gap in seconds",
    )
    parser.add_argument(
        "--minimum-pause-seconds",
        type=float,
        default=0.50,
        help="Minimum low-energy run reported as a possible within-turn pause (default: 0.50)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.audio.is_file():
        raise SystemExit(f"Audio file not found: {args.audio}")
    if not args.captions.is_file():
        raise SystemExit(f"Caption file not found: {args.captions}")
    if args.fmin <= 0 or args.fmax <= args.fmin or args.fmax >= args.sample_rate / 2:
        raise SystemExit("Require 0 < --fmin < --fmax < half of --sample-rate")
    if args.minimum_pause_seconds <= 0:
        raise SystemExit("--minimum-pause-seconds must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    segments = parse_captions(args.captions)
    turns = merge_same_speaker_segments(segments, args.same_speaker_gap)

    audio, sample_rate = librosa.load(
        args.audio, sr=args.sample_rate, mono=True, dtype=np.float32
    )
    if audio.size == 0:
        raise SystemExit("The decoded audio is empty.")
    duration = len(audio) / sample_rate
    if segments[-1].end > duration + 1.0:
        print(
            f"Warning: final caption ends at {segments[-1].end:.2f}s, "
            f"but audio duration is {duration:.2f}s.",
            file=sys.stderr,
        )

    features = compute_frame_features(audio, sample_rate, args.fmin, args.fmax)
    report, segment_rows, speaker_rows = build_report(
        args.audio,
        args.captions,
        segments,
        turns,
        audio,
        sample_rate,
        features,
        args.same_speaker_gap,
        args.fmin,
        args.fmax,
    )
    llm_payload = build_llm_rubric_payload(
        report,
        turns,
        features,
        args.student_speaker,
        args.minimum_pause_seconds,
    )

    (args.output_dir / "analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "llm_rubric_payload.json").write_text(
        json.dumps(llm_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_gemini_packet(
        args.output_dir / "gemini_grading_packet.md", llm_payload
    )
    write_csv(args.output_dir / "segments.csv", segment_rows)
    write_csv(args.output_dir / "speakers.csv", speaker_rows)
    write_csv(
        args.output_dir / "turns.csv",
        [
            {
                "start_seconds": round(item.start, 3),
                "end_seconds": round(item.end, 3),
                "duration_seconds": round(item.duration, 3),
                "speaker": item.speaker,
                "text": item.text,
            }
            for item in turns
        ],
    )
    create_overview_plot(
        args.output_dir / "overview.png", audio, sample_rate, turns, speaker_rows
    )
    print_summary(report, args.student_speaker)
    print(f"\nWrote results to: {args.output_dir.resolve()}")
    print("Gemini-ready packet: gemini_grading_packet.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
