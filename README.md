![Demo](assets/git.gif)

# Live Speech Transcription

Real-time browser-to-FastAPI speech transcription with modular noise suppression, voice activity detection, and multiple STT backends.

```
Browser (Microphone) → WebSocket → FastAPI Backend
                                      ↓
                                 NS → VAD → STT
                                      ↓
                                 Live Transcript
```

## Features

- **Noise Suppression** — DeepFilterNet3 removes background noise in real time
- **Voice Activity Detection** — Silero VAD or WebRTC VAD to gate silence
- **Speech-to-Text** — Faster Whisper, OpenAI Whisper, Wav2Vec2, MMS Multilingual
- **Runtime Configuration** — Switch models, languages, and pipeline stages from the UI
- **Lazy Model Loading** — STT models load on first use, cached for the session
- **Audio Visualizer** — Real-time frequency spectrum display

## Dependency Baseline

| Component | Version |
|---|---|
| Python | `3.10` |
| PyTorch | `2.5.1` |
| torchaudio | `2.5.1` |
| CUDA runtime | `12.1` via `pytorch-cuda=12.1` |
| `openai-whisper` | `20250625` |

## Quick Start

```bash
# Create conda environment (includes PyTorch + CUDA 12.1)
conda env create -f environment.yml
conda activate vadstt_live

# Install Python dependencies
pip install -r requirements.txt

# Start the server
python -m uvicorn backend.server:app --reload
```

Open `http://localhost:8000`.

### GPU Verification

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Expected: `2.5.1 12.1 True`

> **Note:** Avoid `pip install --force-reinstall` for `requirements.txt` on Windows — it can replace the conda CUDA PyTorch build with CPU-only wheels.

## Project Structure

```
├── backend/
│   ├── server.py                  # FastAPI app (REST + WebSocket)
│   ├── config.py                  # Central configuration & model registry
│   ├── model_manager.py           # Eager (VAD, NS) + lazy (STT) model loading
│   ├── pipeline/
│   │   ├── base_task.py           # Abstract task interface
│   │   ├── pipeline_manager.py    # Orchestrates NS → VAD → STT
│   │   ├── audio_input_task.py    # Bytes → Float32 numpy array
│   │   ├── noise_suppression_task.py
│   │   ├── vad_task.py            # Silero / WebRTC dispatch
│   │   └── stt_task.py            # Multi-engine transcription
│   └── models/
│       ├── deepfilternet_model.py # DeepFilterNet3 wrapper
│       ├── silero_vad_model.py
│       ├── webrtc_vad_model.py
│       ├── faster_whisper_model.py
│       ├── whisper_model.py       # OpenAI Whisper wrapper
│       └── wav2vec_model.py       # Wav2Vec2 + MMS + SpeechBrain ASR
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── models/
│   └── DeepFilterNet3/            # Local model checkpoint
├── environment.yml
├── requirements.txt
└── README.md
```

## Pipeline Architecture

The pipeline follows a Luigi-style modular design. Each stage inherits from `BaseTask`:

| # | Stage | Description |
|---|-------|-------------|
| 1 | **AudioInputTask** | Converts raw binary WebSocket data to float32 numpy arrays |
| 2 | **NoiseSuppressionTask** | Optional DeepFilterNet3 enhancement (16 kHz ↔ 48 kHz transparent resampling) |
| 3 | **VADTask** | Speech/silence gate — Silero (neural) or WebRTC (lightweight CPU) |
| 4 | **STTTask** | Multi-engine transcription with hallucination filtering |

Each task can return `None` to stop the pipeline (e.g., VAD detects silence).

## WebSocket Protocol

```
Client → Server  :  JSON config  {"ns_enabled", "vad_model", "stt_model", "language"}
Server → Client  :  {"type": "config_ack", "status": {...}}
Client → Server  :  [binary float32 audio chunks, 16 kHz mono]
Server → Client  :  {"type": "transcription", "text": "...", "latency": N, "is_speech": bool}
Client → Server  :  "STOP" (text)
```

## STT Model Registry

<details>
<summary>Available models</summary>

| Key | Engine | Model | Languages |
|---|---|---|---|
| `fasterwhisper-tiny` | faster-whisper | `tiny` | all |
| `fasterwhisper-base` | faster-whisper | `base` | all |
| `fasterwhisper-small` | faster-whisper | `small` | all |
| `fasterwhisper-medium` | faster-whisper | `medium` | all |
| `fasterwhisper-large` | faster-whisper | `large-v3` | all |
| `whisper-large-v3` | faster-whisper | `large-v3` | all |
| `whisper-large-v3-turbo` | faster-whisper | `large-v3-turbo` | all |
| `whisper-distil-large-v3` | faster-whisper | `distil-large-v3` | all |
| `openai-whisper-base` | openai-whisper | `base` | all |
| `openai-whisper-large` | openai-whisper | `large-v3` | all |
| `wav2vec-fast` | wav2vec | `facebook/wav2vec2-base-960h` | en |
| `wav2vec-fast` | wav2vec-sb | `speechbrain/asr-wav2vec2-commonvoice-de` | de |
| `wav2vec-accurate` | wav2vec | `facebook/wav2vec2-large-960h-lv60-self` | en |
| `wav2vec-accurate` | wav2vec | `jonatasgrosman/wav2vec2-large-xlsr-53-german` | de |
| `wav2vec-multilingual` | wav2vec | `facebook/mms-1b-all` | all |

</details>

## Runtime Notes

- `DeepFilterNet3` expects local assets under `models/DeepFilterNet3/`. If missing, noise suppression is unavailable.
- First STT chunk may trigger a model download — the client shows a loading indicator.
- Audio chunks are configurable from 250 ms to 5000 ms (default: 1000 ms).
- All models expect 16 kHz mono float32 audio.

## Branches

| Branch | Description |
|---|---|
| **`main`** | Core STT pipeline (NS + VAD + STT) |
| **`diarization`** | Adds real-time speaker diarization (pyannote, NeMo, SpeechBrain) |
