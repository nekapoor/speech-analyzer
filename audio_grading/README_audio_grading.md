# Audio-to-Gemini Global Rubric packet

This tool takes an audio recording and a diarized, timestamped transcript. It creates a ready-to-paste Gemini grading packet containing the transcript, rubric, acoustic evidence, interaction timing, limitations, and grading instructions.

## 1. Install

Place these files in the same folder:

- `audio_communication_analyzer.py`
- `audio_analysis_requirements.txt`

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r audio_analysis_requirements.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r audio_analysis_requirements.txt
```

## 2. Caption format

Use the same diarization label consistently. For example:

```text
[00:02.940 --> 00:05.700] STUDENT: Hi, my name is Tiffany.
[00:05.840 --> 00:07.460] STUDENT: Can you confirm your name?
[00:07.880 --> 00:11.680] PATIENT: My name is Mary Smith.
```

SRT/VTT-style blocks and pipe-delimited lines are also accepted.

## 3. Run

```bash
python audio_communication_analyzer.py \
  --audio encounter.mp3 \
  --captions captions.txt \
  --output-dir grading-results \
  --student-speaker STUDENT
```

Replace `STUDENT` with the exact diarization label assigned to the student, such as `SPEAKER_00`.

## 4. Give Gemini the packet

Open `grading-results/gemini_grading_packet.md` and paste or upload it to Gemini. It already includes:

- The complete Global Rubric and P2 grading instructions
- Timestamped diarized turns
- Questions, filler, teach-back, and empathy phrase candidates
- Speaking rate, pause, pitch-variation, and relative-level evidence
- Response gaps, overlaps, and cue-response context
- Data-quality warnings and measurement limitations
- Instructions not to confuse clinical correctness with communication quality
- Instructions to mark unavailable visual behavior as `not_observable`

`llm_rubric_payload.json` contains the same evidence without the surrounding Gemini prompt.

## Other outputs

- `analysis.json`: full technical analysis
- `segments.csv`: caption-segment evidence
- `turns.csv`: merged conversational turns
- `speakers.csv`: per-speaker measurements
- `overview.png`: waveform and turn timeline

## Important limitations

Audio and transcripts cannot establish eye contact, posture, visible gestures, physical distance, or attire. Those items must not be counted as poor. Scores involving unavailable items are provisional.

Pitch, intensity, jitter, shimmer, and HNR do not prove empathy, confidence, hostility, emotion, personality, gender, health, or deception. They are supporting or diagnostic metadata only.

The generated packet contains the transcript and may contain patient information. The script does not de-identify it. Use only an institutionally approved environment and follow the applicable privacy requirements before uploading real conversations.
