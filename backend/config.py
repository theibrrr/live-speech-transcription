"""
config.py - Central configuration for the speech-to-text application.

Provides a single source of truth for all configurable parameters including
audio settings, model paths, hardware device selection, STT model registry,
and speaker diarization settings.
"""

import importlib.util
import os

import torch

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


class Config:
    """Application-wide configuration constants."""

    # Audio settings
    SAMPLE_RATE: int = 16_000
    DEEPFILTER_SR: int = 48_000
    CHUNK_DURATION_MS: int = 1000
    CHUNK_SAMPLES: int = SAMPLE_RATE * CHUNK_DURATION_MS // 1000

    # Device detection
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Model defaults
    DEFAULT_NS_ENABLED: bool = False
    DEFAULT_VAD_MODEL: str = "silero"
    DEFAULT_STT_MODEL: str = "fasterwhisper-base"
    DEFAULT_LANGUAGE: str = "en"
    DEFAULT_DIARIZATION_MODEL: str = "none"

    # DeepFilterNet local model path
    DEEPFILTER_MODEL_DIR: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "DeepFilterNet3",
    )

    # VAD settings
    WEBRTC_VAD_AGGRESSIVENESS: int = 2
    SILERO_VAD_THRESHOLD: float = 0.65
    MIN_AUDIO_ENERGY: float = 0.005

    # Speaker diarization settings
    PYANNOTE_DIARIZATION_MODEL_ID: str = os.getenv(
        "PYANNOTE_DIARIZATION_MODEL_ID",
        "pyannote/speaker-diarization-3.1",
    )
    PYANNOTE_EMBEDDING_MODEL_ID: str = os.getenv(
        "PYANNOTE_EMBEDDING_MODEL_ID",
        "pyannote/embedding",
    )
    PYANNOTE_MODEL_DIR: str = os.getenv("PYANNOTE_MODEL_DIR", "")
    PYANNOTE_EMBEDDING_DIR: str = os.getenv("PYANNOTE_EMBEDDING_DIR", "")
    HUGGINGFACE_TOKEN: str = (
        os.getenv("PYANNOTE_HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HF_TOKEN")
        or ""
    )

    # NeMo diarization settings
    NEMO_EMBEDDING_MODEL_ID: str = os.getenv(
        "NEMO_EMBEDDING_MODEL_ID",
        "nvidia/speakerverification_en_titanet_large",
    )

    # SpeechBrain diarization settings
    SPEECHBRAIN_EMBEDDING_MODEL_ID: str = os.getenv(
        "SPEECHBRAIN_EMBEDDING_MODEL_ID",
        "speechbrain/spkrec-ecapa-voxceleb",
    )

    # Windowed clustering parameters (NeMo & SpeechBrain)
    DIARIZATION_CLUSTER_WINDOW_S: float = 1.5
    DIARIZATION_CLUSTER_HOP_S: float = 0.75
    DIARIZATION_CLUSTER_THRESHOLD: float = 0.5

    DIARIZATION_WINDOW_MS: int = 2_000
    DIARIZATION_MIN_WINDOW_MS: int = 2_000
    DIARIZATION_UPDATE_INTERVAL_MS: int = 3_000
    DIARIZATION_AUDIO_RETENTION_MS: int = 45_000
    DIARIZATION_EMBEDDING_MIN_MS: int = 2000
    DIARIZATION_EMBEDDING_MAX_MS: int = 12_000
    DIARIZATION_SPEAKER_MATCH_THRESHOLD: float = 0.62
    DIARIZATION_PENDING_SPEAKER_THRESHOLD: float = 0.58
    DIARIZATION_NEW_SPEAKER_CONFIRMATIONS: int = 2
    DIARIZATION_PENDING_SPEAKER_MAX_AGE_MS: int = 20_000
    DIARIZATION_FINALIZATION_DELAY_MS: int = 6_000
    DIARIZATION_MIN_LABEL_OBSERVATIONS: int = 2
    DIARIZATION_MIN_LABEL_WIN_RATIO: float = 0.68
    DIARIZATION_MIN_LABEL_MARGIN_S: float = 0.40
    DIARIZATION_FORCE_ASSIGNMENT_MS: int = 12_000
    DIARIZATION_FORCE_ASSIGNMENT_RATIO: float = 0.55

    DIARIZATION_MODELS: dict = {
        "none": {
            "label": "None",
            "available": True,
        },
        "pyannote": {
            "label": "pyannote",
            "available": True,
        },
        "nemo": {
            "label": "NVIDIA NeMo",
            "available": True,
        },
        "speechbrain": {
            "label": "SpeechBrain",
            "available": True,
        },
    }

    # Supported languages
    ALL_LANGUAGES = {
        "en": "English",
        "tr": "Turkce",
        "de": "Deutsch",
        "fr": "Francais",
        "es": "Espanol",
        "it": "Italiano",
        "pt": "Portugues",
        "nl": "Nederlands",
        "ru": "Russkiy",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "ar": "Arabic",
        "hi": "Hindi",
        "pl": "Polski",
        "sv": "Svenska",
        "uk": "Ukrainian",
    }

    # STT model registry
    STT_MODELS: dict = {
        "fasterwhisper-tiny": {
            "engine": "faster-whisper",
            "model_id": "tiny",
            "label": "Faster Whisper Tiny",
            "group": "Faster Whisper",
            "languages": "all",
            "diarization_supported": True,
        },
        "fasterwhisper-base": {
            "engine": "faster-whisper",
            "model_id": "base",
            "label": "Faster Whisper Base",
            "group": "Faster Whisper",
            "languages": "all",
            "diarization_supported": True,
        },
        "fasterwhisper-small": {
            "engine": "faster-whisper",
            "model_id": "small",
            "label": "Faster Whisper Small",
            "group": "Faster Whisper",
            "languages": "all",
            "diarization_supported": True,
        },
        "fasterwhisper-medium": {
            "engine": "faster-whisper",
            "model_id": "medium",
            "label": "Faster Whisper Medium",
            "group": "Faster Whisper",
            "languages": "all",
            "diarization_supported": True,
        },
        "fasterwhisper-large": {
            "engine": "faster-whisper",
            "model_id": "large-v3",
            "label": "Faster Whisper Large",
            "group": "Faster Whisper",
            "languages": "all",
            "diarization_supported": True,
        },
        "whisper-large-v3": {
            "engine": "faster-whisper",
            "model_id": "large-v3",
            "label": "Whisper Large v3",
            "group": "Whisper",
            "languages": "all",
            "diarization_supported": False,
        },
        "whisper-large-v3-turbo": {
            "engine": "faster-whisper",
            "model_id": "large-v3-turbo",
            "label": "Whisper Large v3 Turbo",
            "group": "Whisper",
            "languages": "all",
            "diarization_supported": False,
        },
        "whisper-distil-large-v3": {
            "engine": "faster-whisper",
            "model_id": "distil-large-v3",
            "label": "Whisper Distil Large v3",
            "group": "Whisper",
            "languages": "all",
            "diarization_supported": False,
        },
        "openai-whisper-base": {
            "engine": "openai-whisper",
            "model_id": "base",
            "label": "OpenAI Whisper Base",
            "group": "OpenAI Whisper",
            "languages": "all",
            "diarization_supported": False,
        },
        "openai-whisper-large": {
            "engine": "openai-whisper",
            "model_id": "large-v3",
            "label": "OpenAI Whisper Large",
            "group": "OpenAI Whisper",
            "languages": "all",
            "diarization_supported": False,
        },
        "wav2vec-fast": {
            "label": "wav2vec (Fast)",
            "group": "Wav2Vec2",
            "languages": ["en", "de"],
            "diarization_supported": False,
            "variants": {
                "en": {"engine": "wav2vec", "model_id": "facebook/wav2vec2-base-960h"},
                "de": {"engine": "wav2vec-sb", "model_id": "speechbrain/asr-wav2vec2-commonvoice-de"},
            },
        },
        "wav2vec-accurate": {
            "label": "wav2vec (Accurate)",
            "group": "Wav2Vec2",
            "languages": ["en", "de"],
            "diarization_supported": False,
            "variants": {
                "en": {"engine": "wav2vec", "model_id": "facebook/wav2vec2-large-960h-lv60-self"},
                "de": {"engine": "wav2vec", "model_id": "jonatasgrosman/wav2vec2-large-xlsr-53-german"},
            },
        },
        "wav2vec-multilingual": {
            "engine": "wav2vec",
            "model_id": "facebook/mms-1b-all",
            "label": "MMS Multilingual",
            "group": "Wav2Vec2",
            "languages": "all",
            "diarization_supported": False,
        },
    }

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    @classmethod
    def pyannote_available(cls) -> bool:
        try:
            has_package = importlib.util.find_spec("pyannote.audio") is not None
        except ModuleNotFoundError:
            has_package = False
        has_local_assets = bool(
            cls.PYANNOTE_MODEL_DIR
            and os.path.isdir(cls.PYANNOTE_MODEL_DIR)
            and cls.PYANNOTE_EMBEDDING_DIR
            and os.path.isdir(cls.PYANNOTE_EMBEDDING_DIR)
        )
        has_token = bool(cls.HUGGINGFACE_TOKEN)
        return has_package and (has_local_assets or has_token)

    @classmethod
    def nemo_available(cls) -> bool:
        try:
            return importlib.util.find_spec("nemo.collections.asr") is not None
        except (ModuleNotFoundError, ValueError):
            return False

    @classmethod
    def speechbrain_diarization_available(cls) -> bool:
        try:
            return importlib.util.find_spec("speechbrain") is not None
        except (ModuleNotFoundError, ValueError):
            return False

    @classmethod
    def diarization_available(cls, key: str) -> bool:
        if key == "none":
            return True
        if key == "pyannote":
            return cls.pyannote_available()
        if key == "nemo":
            return cls.nemo_available()
        if key == "speechbrain":
            return cls.speechbrain_diarization_available()
        return False

    @classmethod
    def supports_diarization(cls, model_key: str) -> bool:
        model = cls.STT_MODELS.get(model_key) or {}
        return bool(model.get("diarization_supported", False))

    @classmethod
    def summary(cls) -> dict:
        """Return config summary for API responses."""
        diarization_models = {
            key: {
                "label": value["label"],
                "available": cls.diarization_available(key),
            }
            for key, value in cls.DIARIZATION_MODELS.items()
        }
        return {
            "device": cls.DEVICE,
            "sample_rate": cls.SAMPLE_RATE,
            "chunk_duration_ms": cls.CHUNK_DURATION_MS,
            "default_stt_model": cls.DEFAULT_STT_MODEL,
            "default_language": cls.DEFAULT_LANGUAGE,
            "default_diarization_model": cls.DEFAULT_DIARIZATION_MODEL,
            "vad_models": ["silero", "webrtc", "none"],
            "deepfilter_available": os.path.isdir(cls.DEEPFILTER_MODEL_DIR),
            "diarization_models": diarization_models,
            "all_languages": cls.ALL_LANGUAGES,
            "stt_models": {
                key: {
                    "label": model["label"],
                    "group": model["group"],
                    "languages": model["languages"],
                    "diarization_supported": model.get("diarization_supported", False),
                }
                for key, model in cls.STT_MODELS.items()
            },
        }
