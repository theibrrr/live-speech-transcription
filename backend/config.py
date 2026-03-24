"""
config.py — Central configuration for the speech-to-text application.

Provides a single source of truth for all configurable parameters including
audio settings, model paths, hardware device selection, and the full
STT model registry.
"""

import os
import torch


class Config:
    """Application-wide configuration constants."""

    # ── Audio Settings ──────────────────────────────────────────────────
    SAMPLE_RATE: int = 16_000           # Whisper / wav2vec expects 16 kHz
    DEEPFILTER_SR: int = 48_000         # DeepFilterNet operates at 48 kHz
    CHUNK_DURATION_MS: int = 1000       # Duration of each audio chunk (ms)
    CHUNK_SAMPLES: int = SAMPLE_RATE * CHUNK_DURATION_MS // 1000

    # ── Device Detection ────────────────────────────────────────────────
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Model Defaults ──────────────────────────────────────────────────
    DEFAULT_NS_ENABLED: bool = False
    DEFAULT_VAD_MODEL: str = "silero"
    DEFAULT_STT_MODEL: str = "fasterwhisper-base"
    DEFAULT_LANGUAGE: str = "en"

    # ── DeepFilterNet Local Model Path ──────────────────────────────────
    DEEPFILTER_MODEL_DIR: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "DeepFilterNet3"
    )

    # ── VAD Settings ────────────────────────────────────────────────────
    WEBRTC_VAD_AGGRESSIVENESS: int = 2
    SILERO_VAD_THRESHOLD: float = 0.65
    MIN_AUDIO_ENERGY: float = 0.005

    # ── Supported Languages ─────────────────────────────────────────────
    ALL_LANGUAGES = {
        "en": "English",
        "tr": "Türkçe",
        "de": "Deutsch",
        "fr": "Français",
        "es": "Español",
        "it": "Italiano",
        "pt": "Português",
        "nl": "Nederlands",
        "ru": "Русский",
        "zh": "中文",
        "ja": "日本語",
        "ko": "한국어",
        "ar": "العربية",
        "hi": "हिन्दी",
        "pl": "Polski",
        "sv": "Svenska",
        "uk": "Українська",
    }

    # ══════════════════════════════════════════════════════════════════════
    # STT MODEL REGISTRY
    #
    # Each entry defines:
    #   engine     — "faster-whisper" | "wav2vec" | "wav2vec-sb"
    #   model_id   — HuggingFace model ID or faster-whisper size string
    #   label      — Human-readable name for the UI
    #   group      — UI grouping (optgroup in <select>)
    #   languages  — "all" or list of ISO 639-1 codes
    # ══════════════════════════════════════════════════════════════════════

    STT_MODELS: dict = {
        # ── Faster Whisper ──────────────────────────────────────────────
        "fasterwhisper-tiny": {
            "engine": "faster-whisper",
            "model_id": "tiny",
            "label": "Faster Whisper Tiny",
            "group": "Faster Whisper",
            "languages": "all",
        },
        "fasterwhisper-base": {
            "engine": "faster-whisper",
            "model_id": "base",
            "label": "Faster Whisper Base",
            "group": "Faster Whisper",
            "languages": "all",
        },
        "fasterwhisper-small": {
            "engine": "faster-whisper",
            "model_id": "small",
            "label": "Faster Whisper Small",
            "group": "Faster Whisper",
            "languages": "all",
        },
        "fasterwhisper-medium": {
            "engine": "faster-whisper",
            "model_id": "medium",
            "label": "Faster Whisper Medium",
            "group": "Faster Whisper",
            "languages": "all",
        },
        "fasterwhisper-large": {
            "engine": "faster-whisper",
            "model_id": "large-v3",
            "label": "Faster Whisper Large",
            "group": "Faster Whisper",
            "languages": "all",
        },

        # ── Whisper (via faster-whisper engine) ─────────────────────────
        "whisper-large-v3": {
            "engine": "faster-whisper",
            "model_id": "large-v3",
            "label": "Whisper Large v3",
            "group": "Whisper",
            "languages": "all",
        },
        "whisper-large-v3-turbo": {
            "engine": "faster-whisper",
            "model_id": "large-v3-turbo",
            "label": "Whisper Large v3 Turbo",
            "group": "Whisper",
            "languages": "all",
        },
        "whisper-distil-large-v3": {
            "engine": "faster-whisper",
            "model_id": "distil-large-v3",
            "label": "Whisper Distil Large v3",
            "group": "Whisper",
            "languages": "all",
        },

        # ── OpenAI Whisper ──────────────────────────────────────────────
        "openai-whisper-base": {
            "engine": "openai-whisper",
            "model_id": "base",
            "label": "OpenAI Whisper Base",
            "group": "OpenAI Whisper",
            "languages": "all",
        },
        "openai-whisper-large": {
            "engine": "openai-whisper",
            "model_id": "large-v3",
            "label": "OpenAI Whisper Large",
            "group": "OpenAI Whisper",
            "languages": "all",
        },

        # ── Wav2Vec2 ───────────────────────────────────────────────────
        # "Fast" and "Accurate" resolve to different models per language.
        # The actual engine + model_id is looked up from "variants" at runtime.
        "wav2vec-fast": {
            "label": "wav2vec (Fast)",
            "group": "Wav2Vec2",
            "languages": ["en", "de"],
            "variants": {
                "en": {"engine": "wav2vec", "model_id": "facebook/wav2vec2-base-960h"},
                "de": {"engine": "wav2vec-sb", "model_id": "speechbrain/asr-wav2vec2-commonvoice-de"},
            },
        },
        "wav2vec-accurate": {
            "label": "wav2vec (Accurate)",
            "group": "Wav2Vec2",
            "languages": ["en", "de"],
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
        },
    }

    # ── Server Settings ─────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    @classmethod
    def summary(cls) -> dict:
        """Return config summary for API responses (includes model registry)."""
        return {
            "device": cls.DEVICE,
            "sample_rate": cls.SAMPLE_RATE,
            "chunk_duration_ms": cls.CHUNK_DURATION_MS,
            "default_stt_model": cls.DEFAULT_STT_MODEL,
            "default_language": cls.DEFAULT_LANGUAGE,
            "vad_models": ["silero", "webrtc"],
            "deepfilter_available": os.path.isdir(cls.DEEPFILTER_MODEL_DIR),
            "all_languages": cls.ALL_LANGUAGES,
            "stt_models": {
                key: {
                    "label": m["label"],
                    "group": m["group"],
                    "languages": m["languages"],
                }
                for key, m in cls.STT_MODELS.items()
            },
        }
