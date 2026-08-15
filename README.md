<div align="center">

# 🎙️ Voice Dataset Forge

**Turn messy stream recordings into a clean, speaker-specific voice dataset with a few clicks.**

A desktop tool for extracting one target speaker from long recordings using speaker diarization, voice matching, optional music separation, quality filtering, and transcription.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Primary%20platform-Windows-0078D4?logo=windows&logoColor=white)
![GPU](https://img.shields.io/badge/NVIDIA%20CUDA-optional%20but%20recommended-76B900?logo=nvidia&logoColor=white)
![Status](https://img.shields.io/badge/status-usable-brightgreen)

</div>

---

## Why Voice Dataset Forge?

Preparing a voice-conversion dataset from streams is usually annoying:

- background music is mixed with speech;
- multiple people may be talking;
- recordings can be hours long;
- the target speaker must be identified reliably;
- bad, silent, clipped, or ambiguous segments should be removed;
- everything still needs to end up in one clean dataset.

**Voice Dataset Forge is built around one simple workflow:**

> Select your stream folder → select reference voice samples → choose a quality preset → build the dataset.

No manual cutting of multi-hour streams is required.

---

## ✨ Features

- 🎯 **Target-speaker extraction** from recordings containing multiple speakers
- 🎵 **Optional music/background separation** with Demucs
- 🗣️ **Speaker diarization** with `pyannote/speaker-diarization-community-1`
- 🧬 **Reference voice matching** with WeSpeaker embeddings
- ✂️ **Automatic clip generation** into one continuous dataset
- 🔍 **Best-speaker margin filtering** to reduce accidental inclusion of another speaker
- 🧹 **Audio quality filtering** for short, quiet, clipped, or mostly-silent segments
- 📝 **Optional transcription** with faster-whisper
- ⏯️ **Incremental dataset writing** — accepted clips are saved while processing
- 💾 **Caching and resume support** for expensive stages
- 🩹 **Self-healing project state** when cache or output files are manually deleted
- 🕒 **Multi-hour stream support** without manually splitting source recordings
- 🧠 **Memory-aware processing** designed to avoid keeping every heavy model in VRAM at once
- 📊 **Detailed processing log** with RAM/VRAM telemetry
- 🖥️ Simple **PySide6 desktop GUI**

---

## 🧭 What this project is — and is not

Voice Dataset Forge is a **dataset preparation tool**.

It does **not** train a voice-conversion model itself. Its job is to turn raw recordings into a clean master dataset that can later be used with RVC, Seed-VC, or other voice-conversion / speech-training pipelines.

---

## 🔄 Pipeline

```text
Stream recordings
        │
        ▼
   FFmpeg extraction
        │
        ├─────────────── Fast ───────────────┐
        │                                     │
        ▼                                     │
Demucs voice separation                       │
(Balanced / Maximum)                          │
        │                                     │
        └─────────────────────────────────────┘
        │
        ▼
pyannote speaker diarization
        │
        ▼
WeSpeaker reference matching
        │
        ▼
Best-speaker / ambiguity filter
        │
        ▼
Overlap + audio-quality filtering
        │
        ▼
48 kHz mono PCM16 clips
        │
        ├──────── Optional ────────► faster-whisper
        │
        ▼
One unified dataset
```

---

## 🚀 Quick Start — Windows

### 1. Requirements

Install:

- **Windows 10/11**
- **Python 3.11 x64**
- **FFmpeg** and make sure `ffmpeg` + `ffprobe` are available in `PATH`
- An NVIDIA GPU is strongly recommended for Balanced / Maximum

The app can use CPU fallbacks for some stages, but processing will be much slower.

### 2. Clone the repository

```bash
git clone https://github.com/<your-username>/voice-dataset-forge.git
cd voice-dataset-forge
```

Or download the repository as a ZIP and extract it.

### 3. Install the environment

Run:

```bat
install_windows.bat
```

The installer creates a local `.venv` so the project does not pollute your system Python installation.

### 4. Enable pyannote access

You need a Hugging Face account.

1. Open the `pyannote/speaker-diarization-community-1` model page.
2. Accept the model access conditions.
3. Create a Hugging Face access token.
4. Paste that token into Voice Dataset Forge.

### 5. Check your environment

```bat
.venv\Scripts\activate.bat
python doctor.py
```

The doctor checks the main runtime requirements such as Python, FFmpeg, PyTorch/CUDA, GPU visibility, and ML dependencies.

### 6. Launch

```bat
run_windows.bat
```

---

## 🖥️ Basic Usage

In the GUI, choose:

1. **Streams folder**  
   Put your stream recordings or other source media here.

2. **Reference folder**  
   Put clean recordings of the target speaker here.

3. **Dataset output folder**  
   All accepted clips from every source file are written into one dataset project.

4. **Quality preset**  
   `Fast`, `Balanced`, or `Maximum`.

5. Optional: enable **transcription**.

Then click **Build Dataset**.

You do not need to manually split a two-hour stream. Long recordings are chunked internally.

---

## 🎚️ Quality Presets

### ⚡ Fast

Best for smoke tests and quick extraction.

- no Demucs source separation;
- speaker diarization + reference matching;
- softer filtering;
- diarization windows up to ~5 minutes;
- faster-whisper `small` when transcription is enabled;
- lowest processing cost.

Use this first when checking a new installation.

### ⚖️ Balanced

Recommended general-purpose preset.

- Demucs `htdemucs`;
- reference audio is also cleaned before building the voice embedding;
- stricter speaker filtering;
- long recordings are separated in bounded outer jobs;
- faster-whisper `turbo` when transcription is enabled;
- good balance between speed and dataset cleanliness.

### 💎 Maximum

Prioritizes dataset purity over speed.

- Demucs `htdemucs_ft`;
- strictest speaker filtering;
- extra verification of accepted clips;
- long streams use smaller separation jobs to bound host RAM;
- faster-whisper `large-v3` when transcription is enabled;
- designed for final high-quality dataset generation.

---

## 🎤 Reference Voice Guidelines

The reference folder may contain one or multiple files.

For best results:

- use only the **target speaker**;
- provide at least **30–60 seconds** of speech in total;
- include different speaking styles / intonations if possible;
- avoid other people speaking in the reference;
- avoid heavy clipping and strong echo.

Clean speech is ideal.

In **Balanced** and **Maximum**, reference files are automatically passed through Demucs before their speaker embeddings are calculated, so moderate background music is acceptable. Music containing another vocalist is still not ideal.

---

## 🕒 Long Streams

Long recordings are handled automatically.

For Demucs, the application uses an additional **outer chunking layer** so the separator never needs to hold the complete multi-hour result tensor in RAM at once.

Typical behavior:

```text
2-hour recording
    ↓
Demucs job 1
Demucs job 2
Demucs job 3
...
    ↓
merged vocal stem
    ↓
diarization chunks
    ↓
speaker verification
    ↓
dataset clips
```

Completed expensive stages are cached when possible, so a failed late-stage run does not necessarily require starting everything from zero.

---

## 🧠 Memory-Safe Design

The pipeline avoids loading every heavy model onto the GPU simultaneously.

The general execution strategy is:

```text
Reference preparation
        ↓
WeSpeaker on CPU
        ↓
Demucs on GPU
        ↓
pyannote on GPU
        ↓
unload pyannote
        ↓
reuse WeSpeaker on CPU
        ↓
Whisper on GPU (optional)
```

WeSpeaker is intentionally kept on CPU for stability.

The application also logs RAM and VRAM usage around heavy stages to make crashes easier to diagnose.

---

## 📁 Dataset Output

Example:

```text
MyVoice_Dataset/
├── audio/
│   ├── 000001.wav
│   ├── 000002.wav
│   ├── 000003.wav
│   └── ...
├── metadata.csv
├── metadata_extended.csv
├── metadata.jsonl
├── statistics.json
├── dataset_report.txt
├── settings.json
├── processing.log
├── .vdf_state.json
└── .cache/
```

Final clips are exported as:

```text
48 kHz
mono
PCM16 WAV
```

### `metadata.csv`

The primary metadata file uses a simple pipe-separated format:

```text
audio/000001.wav|Hello, this is the first accepted segment.
audio/000002.wav|And this is the next one.
```

There is no header.

When transcription is disabled, the text field may be empty:

```text
audio/000001.wav|
```

This keeps the master dataset useful for voice-conversion workflows where transcripts are not required.

`metadata_extended.csv` contains additional information such as source file, timestamps, duration, speaker similarity, and technical quality metrics.

---

## 📝 Transcription

Transcription is optional and powered by **faster-whisper**.

- **Fast:** Whisper `small`
- **Balanced:** Whisper `turbo`
- **Maximum:** Whisper `large-v3`

For voice-conversion training, transcripts are often not required, so transcription can be disabled to save time and GPU resources.

---

## 💾 Cache, Resume & Self-Healing

Voice Dataset Forge treats the final dataset and intermediate cache as separate layers.

It can rebuild missing or invalid intermediate artifacts such as:

- reference voice centroids;
- diarization results;
- speaker candidate decisions;
- Demucs long-stream chunks.

If dataset WAV files are manually removed while metadata still references them, stale metadata is cleaned and completed-source state can be reset automatically.

This means missing cache files are treated as **work to recompute**, not as a fatal project error.

---

## 🔧 Updating an Existing Installation

If you already have a working `.venv`:

```bat
update_windows.bat
```

If the CUDA build of PyTorch needs repair:

```bat
repair_cuda.bat
```

---

## 🧪 Tests

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Run:

```bash
pytest -q
```

---

## ⚠️ Limitations

- Speaker diarization does not physically separate two people speaking at exactly the same time. Ambiguous / overlapping speech is preferably discarded.
- Demucs can leave artifacts, especially with difficult music or overlapping vocals.
- Speaker similarity scores are cosine similarities, **not calibrated biometric probabilities**.
- CPU fallback is useful for stability, but can be very slow.
- Very poor source audio can still produce poor dataset clips.
- Windows is currently the primary tested platform.
- The `rejected/` workflow is not considered a core/stable feature yet.

---

## 🔐 Responsible Use

Voice cloning and voice conversion can be powerful technologies.

Please use this project only with audio you have the right to process, and do not use generated or converted voices to impersonate people deceptively, bypass authentication, or commit fraud.

---

## 🤖 Development Note

Voice Dataset Forge was built with **AI-assisted development** and iteratively tested against real-world, multi-hour stream recordings.

The goal of the project is not to hide that fact — it is to show how quickly a practical tool can be created, tested, debugged, and improved when AI is used as part of the development workflow.

Bug reports, fixes, testing feedback, and pull requests are welcome.

---

## 🛣️ Possible Next Steps

Some ideas for future releases:

- better rejected-segment review;
- export presets for specific VC/TTS training frameworks;
- configurable speaker thresholds in the GUI;
- richer dataset statistics;
- Linux installation helpers;
- packaged standalone Windows builds.

---

## ❤️ Contributing

Issues and pull requests are welcome.

If you find a bug, including `processing.log` and your basic environment information (GPU, RAM, Python version, and selected quality preset) will make debugging much easier.

---

<div align="center">

**Raw streams in. Clean voice dataset out.**

</div>
