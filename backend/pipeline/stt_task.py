"""
stt_task.py — Pipeline task for Speech-to-Text transcription.

Dispatches to the correct STT engine (Faster-Whisper or Wav2Vec2)
based on the model registry in Config.STT_MODELS.
Models are loaded lazily via ModelManager on first use.
"""

import logging
import numpy as np
from typing import Optional

from backend.pipeline.base_task import BaseTask

logger = logging.getLogger(__name__)


class STTTask(BaseTask):
    """
    Speech-to-Text pipeline task — supports multiple STT engines.

    Input: Float32 numpy array (16 kHz).
    Output: Transcribed text string.
    """

    def __init__(self, model_manager, model_key: str = "fasterwhisper-base", language: str = "en"):
        """
        Initialize the STT task.

        Args:
            model_manager: ModelManager instance for lazy model loading.
            model_key: Key from Config.STT_MODELS.
            language: Language code for transcription.
        """
        super().__init__(name="STT", enabled=True)
        self.model_manager = model_manager
        self.model_key = model_key
        self.language = language

    def set_model(self, model_key: str):
        """Switch the STT model at runtime."""
        from backend.config import Config
        if model_key in Config.STT_MODELS:
            self.model_key = model_key
            logger.info(f"STT model set to: {model_key}")
        else:
            logger.warning(f"Unknown STT model: {model_key}")

    def set_language(self, language: str):
        """Switch the transcription language at runtime."""
        self.language = language
        logger.info(f"STT language set to: {language}")

    # ── Known Whisper hallucination phrases ──────────────────────────────
    HALLUCINATION_PHRASES = {
        "you", "thank you", "thanks", "thank you.", "thanks.",
        "bye", "bye.", "goodbye", "the end", "the end.",
        "thanks for watching", "thanks for watching.",
        "thank you for watching", "thank you for watching.",
        "subscribe", "like and subscribe",
        "so", "uh", "um", "hmm", "huh", "oh",
        "mbc 뉴스", "amara.org", "www.mooji.org",
        "подписывайтесь на канал",
    }

    def process(self, data: np.ndarray) -> Optional[str]:
        """
        Transcribe the audio chunk to text.

        Args:
            data: Float32 numpy array of audio samples (16 kHz).

        Returns:
            Transcribed text string, or None on failure / silence.
        """
        if not self.enabled:
            return None

        # Lazy-load the model via ModelManager (language resolves wav2vec variants)
        model = self.model_manager.get_stt_model(self.model_key, self.language)
        if model is None or not model.available:
            logger.warning(f"STT: model '{self.model_key}' not available.")
            return None

        try:
            text = model.transcribe(data, language=self.language)

            # Filter out known Whisper hallucinations
            if text and text.strip().lower() in self.HALLUCINATION_PHRASES:
                logger.debug(f"STT: filtered hallucination: '{text.strip()}'")
                return None

            return text if text else None

        except Exception as e:
            logger.error(f"STT error: {e}")
            return None
