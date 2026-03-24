"""
stt_task.py - Pipeline task for speech-to-text transcription.

Dispatches to the correct STT engine based on the model registry in
Config.STT_MODELS and returns a structured transcription payload.
"""

import logging
from typing import Optional

import numpy as np

from backend.pipeline.base_task import BaseTask

logger = logging.getLogger(__name__)


class STTTask(BaseTask):
    """
    Speech-to-Text pipeline task supporting multiple STT engines.

    Input: Float32 numpy array (16 kHz).
    Output: Dict with `text` and `segments`.
    """

    HALLUCINATION_PHRASES = {
        "you", "thank you", "thanks", "thank you.", "thanks.",
        "bye", "bye.", "goodbye", "the end", "the end.",
        "thanks for watching", "thanks for watching.",
        "thank you for watching", "thank you for watching.",
        "subscribe", "like and subscribe",
        "so", "uh", "um", "hmm", "huh", "oh",
        "mbc news", "amara.org", "www.mooji.org",
        "podpisivaytes na kanal",
    }

    def __init__(self, model_manager, model_key: str = "fasterwhisper-base", language: str = "en"):
        super().__init__(name="STT", enabled=True)
        self.model_manager = model_manager
        self.model_key = model_key
        self.language = language

    def set_model(self, model_key: str):
        from backend.config import Config
        if model_key in Config.STT_MODELS:
            self.model_key = model_key
            logger.info(f"STT model set to: {model_key}")
        else:
            logger.warning(f"Unknown STT model: {model_key}")

    def set_language(self, language: str):
        self.language = language
        logger.info(f"STT language set to: {language}")

    def process(self, data: np.ndarray) -> Optional[dict]:
        """Transcribe the audio chunk and return structured segment data."""
        if not self.enabled:
            return None

        model = self.model_manager.get_stt_model(self.model_key, self.language)
        if model is None or not model.available:
            logger.warning(f"STT: model '{self.model_key}' not available.")
            return None

        try:
            if hasattr(model, "transcribe_detailed"):
                result = model.transcribe_detailed(data, language=self.language)
            else:
                text = model.transcribe(data, language=self.language)
                result = {
                    "text": text or "",
                    "segments": [{
                        "text": text or "",
                        "start_s": 0.0,
                        "end_s": float(len(data) / 16000.0),
                    }] if text else [],
                }

            cleaned_segments = []
            for segment in result.get("segments", []):
                text = (segment.get("text") or "").strip()
                if not text:
                    continue
                if text.lower() in self.HALLUCINATION_PHRASES:
                    logger.debug("STT: filtered hallucination segment '%s'", text)
                    continue
                cleaned_segments.append({
                    "text": text,
                    "start_s": float(segment.get("start_s", 0.0) or 0.0),
                    "end_s": float(segment.get("end_s", 0.0) or 0.0),
                })

            full_text = " ".join(segment["text"] for segment in cleaned_segments).strip()
            if not full_text:
                return None

            return {
                "text": full_text,
                "segments": cleaned_segments,
            }

        except Exception as e:
            logger.error(f"STT error: {e}")
            return None
