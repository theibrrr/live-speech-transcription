"""
model_manager.py — Centralized model loading and lifecycle management.

Loads VAD and DeepFilterNet models eagerly at startup.
STT models are loaded LAZILY on first use and cached — this avoids
loading all STT models into memory at once.

Thread-safe: uses a Lock for the STT cache since pipeline.process()
runs in a background thread via asyncio.to_thread().
"""

import logging
from typing import Optional, Any
from threading import Lock

from backend.config import Config
from backend.models.deepfilternet_model import DeepFilterNetModel
from backend.models.silero_vad_model import SileroVADModel
from backend.models.webrtc_vad_model import WebRTCVADModel

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages all ML models: eager loading for VAD/NS, lazy loading for STT.

    STT models are loaded on first request via get_stt_model() and cached
    by (engine, model_id) so that duplicate entries share the same instance.
    """

    def __init__(self):
        self._deepfilter: Optional[DeepFilterNetModel] = None
        self._silero_vad: Optional[SileroVADModel] = None
        self._webrtc_vad: Optional[WebRTCVADModel] = None

        # STT cache: keyed by "engine:model_id" to share identical models
        self._stt_cache: dict[str, Any] = {}
        self._stt_lock = Lock()  # Thread-safe access to cache
        self._loaded = False

    # ── Startup Loading (eager for DeepFilter + VAD) ────────────────────

    def load_all(self):
        """Load non-STT models at startup. STT models load lazily on first use."""
        if self._loaded:
            logger.warning("Models already loaded, skipping.")
            return

        device = Config.DEVICE
        logger.info(f"=== Loading pipeline models on {device} ===")

        self._load_deepfilter(device)
        self._load_silero_vad(device)
        self._load_webrtc_vad()

        self._loaded = True
        logger.info("=== Pipeline models loaded. STT loads on first use. ===")

    def _load_deepfilter(self, device: str):
        try:
            self._deepfilter = DeepFilterNetModel(
                model_dir=Config.DEEPFILTER_MODEL_DIR, device=device,
            )
        except Exception as e:
            logger.warning(f"DeepFilterNet3 unavailable: {e}")

    def _load_silero_vad(self, device: str):
        try:
            self._silero_vad = SileroVADModel(
                device=device, threshold=Config.SILERO_VAD_THRESHOLD,
            )
        except Exception as e:
            logger.warning(f"Silero VAD unavailable: {e}")

    def _load_webrtc_vad(self):
        try:
            self._webrtc_vad = WebRTCVADModel(
                aggressiveness=Config.WEBRTC_VAD_AGGRESSIVENESS,
            )
        except Exception as e:
            logger.warning(f"WebRTC VAD unavailable: {e}")

    # ── STT — Lazy Loading (thread-safe) ────────────────────────────────

    def get_stt_model(self, model_key: str, language: str = "en") -> Optional[Any]:
        """
        Get an STT model by its registry key. Loads lazily on first use.

        For models with "variants" (e.g. wav2vec-fast), the actual model
        is resolved based on the language parameter.

        Thread-safe: uses a lock to prevent duplicate loading.

        Args:
            model_key: Key from Config.STT_MODELS (e.g. "fasterwhisper-base").
            language: ISO 639-1 code, used to resolve variant models.

        Returns:
            Model instance with a transcribe(audio, language) method, or None.
        """
        model_config = Config.STT_MODELS.get(model_key)
        if model_config is None:
            logger.warning(f"Unknown STT model: {model_key}")
            return None

        # Resolve variant models (wav2vec-fast, wav2vec-accurate)
        if "variants" in model_config:
            variant = model_config["variants"].get(language)
            if variant is None:
                logger.warning(
                    f"Model '{model_key}' has no variant for language '{language}'. "
                    f"Available: {list(model_config['variants'].keys())}"
                )
                return None
            engine = variant["engine"]
            model_id = variant["model_id"]
        else:
            engine = model_config["engine"]
            model_id = model_config["model_id"]

        # Cache key: share identical models across entries
        cache_key = f"{engine}:{model_id}"

        # Fast path: check cache without lock
        if cache_key in self._stt_cache:
            return self._stt_cache[cache_key]

        # Slow path: load model under lock
        with self._stt_lock:
            # Double-check after acquiring lock
            if cache_key in self._stt_cache:
                return self._stt_cache[cache_key]

            logger.info(f"Loading STT model: {model_key} → {cache_key}")
            model = self._create_stt_model(engine, model_id)

            if model is not None and model.available:
                self._stt_cache[cache_key] = model
                logger.info(f"STT model '{model_key}' ready.")
                return model
            else:
                # Negative cache: prevent retry loops on every audio chunk
                self._stt_cache[cache_key] = None
                logger.error(
                    f"STT model '{model_key}' ({cache_key}) failed to load. "
                    f"Cached as unavailable — restart server to retry."
                )
                return None

    def _create_stt_model(self, engine: str, model_id: str) -> Optional[Any]:
        """Factory: create the right model wrapper for the engine type."""
        device = Config.DEVICE

        if engine == "openai-whisper":
            from backend.models.whisper_model import WhisperModel
            return WhisperModel(model_name=model_id, device=device)

        elif engine == "faster-whisper":
            from backend.models.faster_whisper_model import FasterWhisperModel
            return FasterWhisperModel(model_id=model_id, device=device)

        elif engine in ("wav2vec", "wav2vec-sb"):
            from backend.models.wav2vec_model import Wav2VecModel
            return Wav2VecModel(model_id=model_id, device=device, engine=engine)

        else:
            logger.error(f"Unknown STT engine: {engine}")
            return None

    # ── Accessors ───────────────────────────────────────────────────────

    @property
    def deepfilter(self) -> Optional[DeepFilterNetModel]:
        return self._deepfilter

    @property
    def silero_vad(self) -> Optional[SileroVADModel]:
        return self._silero_vad

    @property
    def webrtc_vad(self) -> Optional[WebRTCVADModel]:
        return self._webrtc_vad

    @property
    def status(self) -> dict:
        return {
            "deepfilter": self._deepfilter.available if self._deepfilter else False,
            "silero_vad": self._silero_vad.available if self._silero_vad else False,
            "webrtc_vad": self._webrtc_vad.available if self._webrtc_vad else False,
            "stt_cached": list(self._stt_cache.keys()),
            "device": Config.DEVICE,
        }
