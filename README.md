# Live Speech Transcription — Speaker Diarization

> **Branch:** `diarization`
>
> Extends the [`main`](../../tree/main) branch with real-time speaker identification.

```
Browser (Mic) → WebSocket → [NS → VAD → STT] → Live Transcript
                                    ↓
                           Background Diarization
                                    ↓
                           speaker_1, speaker_2, ...
```

---

## Branches

| Branch | Pipeline | Install |
|---|---|---|
| [`main`](../../tree/main) | NS + VAD + STT | `pip install -r requirements.txt` |
| **`diarization`** *(this)* | NS + VAD + STT + Speaker Diarization | `pip install -r requirements.txt` then `pip install -r requirements_diarization.txt` |

Both branches share the same `environment.yml` and `requirements.txt`. The diarization branch adds `requirements_diarization.txt` for pyannote, NeMo, and related utilities.

---

## Quick Start

### 1. Environment (both branches)

```bash
conda env create -f environment.yml
conda activate vadstt_live
pip install -r requirements.txt
```

### 2a. Main branch only — done

```bash
python -m uvicorn backend.server:app --reload
```

### 2b. Diarization branch — extra steps

```bash
# Diarization dependencies (pyannote, utilities)
pip install -r requirements_diarization.txt

# (Optional) NVIDIA NeMo diarization
pip install "nemo_toolkit[asr]>=2.0.0"
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

Set your Hugging Face token (required for pyannote):

```bash
cp .env.example .env
# Edit .env → PYANNOTE_HF_TOKEN=hf_your_token_here
```

Then start:

```bash
python -m uvicorn backend.server:app --reload
```

Open `http://localhost:8000`.

### GPU Verification

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Expected: `2.5.1 12.1 True`

> **Note:** Avoid `pip install --force-reinstall` on Windows — it can replace the conda CUDA PyTorch build with CPU-only wheels.

---

## What This Branch Adds

| Feature | Description |
|---|---|
| **Speaker Diarization** | Identify and label different speakers in real time |
| **Three backends** | pyannote, NVIDIA NeMo (TitaNet), SpeechBrain (ECAPA-TDNN) |
| **Late label patching** | Transcript text appears immediately; speaker labels arrive in the background |
| **Session speaker bank** | Stable speaker identities via embedding matching |
| **Structured transcript** | Transcript items with speaker chips instead of plain text |

---

## Speaker Diarization Backends

### pyannote

Requires a Hugging Face token with access to gated models:

```powershell
$env:PYANNOTE_HF_TOKEN="your_token_here"
```

Accept access on Hugging Face for:
- `pyannote/speaker-diarization-3.1`
- `pyannote/segmentation-3.0`

If pyannote assets fail to load, diarization is disabled for that session.

### NVIDIA NeMo (TitaNet)

Install NeMo, then reinstall CUDA PyTorch (NeMo pulls CPU-only `torch>=2.8`):

```bash
pip install "nemo_toolkit[asr]>=2.0.0"
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

Default model: `nvidia/speakerverification_en_titanet_large`

Override:
```powershell
$env:NEMO_EMBEDDING_MODEL_ID="nvidia/speakerverification_en_titanet_large"
```

### SpeechBrain (ECAPA-TDNN)

Included in `requirements.txt` — no extra install.

Default model: `speechbrain/spkrec-ecapa-voxceleb`

Override:
```powershell
$env:SPEECHBRAIN_EMBEDDING_MODEL_ID="speechbrain/spkrec-ecapa-voxceleb"
```

### Common Behavior

- All backends produce `speaker_1`, `speaker_2`, ... labels via a session speaker bank.
- Text appears immediately; speaker labels are patched via `speaker_update` WebSocket messages.
- When diarization is active, only diarization-supported STT models are enabled in the UI.
- If a backend fails to load, diarization is disabled for that session and STT continues without labels.

---

## Diarization Flow

```
STT (hot path)  →  Immediate transcript with "loading..."
                   ↓
          Background diarization window
                   ↓
          Embedding extraction + clustering
                   ↓
          Speaker bank matching
                   ↓
          speaker_update messages → UI patches labels
```

---

## Dependency Baseline

| Component | Version |
|---|---|
| Python | `3.10` |
| PyTorch | `2.5.1` |
| torchaudio | `2.5.1` |
| CUDA runtime | `12.1` via `pytorch-cuda=12.1` |
| `pyannote.audio` | `3.4.0` |
| `openai-whisper` | `20250625` |
| `huggingface_hub` | `<1.0` |
| `speechbrain` | `>=1.0.0` |
| `nemo_toolkit[asr]` | `>=2.0.0` *(optional)* |

---

## Project Structure

```
├── backend/
│   ├── server.py                        # FastAPI app (REST + WebSocket)
│   ├── config.py                        # Configuration & model registry
│   ├── model_manager.py                 # Model loading (eager NS/VAD, lazy STT + diarization)
│   ├── realtime_diarization.py          # Session-level speaker tracking
│   ├── pipeline/
│   │   ├── base_task.py                 # Abstract task interface
│   │   ├── pipeline_manager.py          # Orchestrates pipeline + diarization flow
│   │   ├── audio_input_task.py
│   │   ├── noise_suppression_task.py
│   │   ├── vad_task.py
│   │   └── stt_task.py                  # Speaker chips + structured transcript items
│   └── models/
│       ├── deepfilternet_model.py
│       ├── silero_vad_model.py
│       ├── webrtc_vad_model.py
│       ├── faster_whisper_model.py
│       ├── whisper_model.py
│       ├── wav2vec_model.py
│       ├── pyannote_diarization_model.py
│       ├── nemo_diarization_model.py
│       └── speechbrain_diarization_model.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── models/
│   └── DeepFilterNet3/
├── .env.example
├── environment.yml
├── requirements.txt
├── requirements_diarization.txt
└── README.md
```

---

## WebSocket Protocol

```
Client → Server  :  JSON config  {"ns_enabled", "vad_model", "stt_model", "language", "diarization"}
Server → Client  :  {"type": "config_ack", "status": {...}}
Client → Server  :  [binary float32 audio chunks, 16 kHz mono]
Server → Client  :  {"type": "transcription", "text": "...", "items": [...]}
Server → Client  :  {"type": "speaker_update", "updates": [...]}
Client → Server  :  "STOP" (text)
```

---

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

---

## Runtime Notes

- Transcript text appears immediately; speaker labels are patched later in the background.
- If embedding-based speaker matching is inconclusive, the backend falls back to raw diarization tracks for `speaker_n` labels.
- `DeepFilterNet3` expects local assets under `models/DeepFilterNet3/`. If missing, noise suppression is unavailable.
- Audio chunks are configurable from 250 ms to 5000 ms (default: 1000 ms).
- All models expect 16 kHz mono float32 audio.

