"""
faster_whisper_model.py — Wrapper for Faster-Whisper STT models.

Uses CTranslate2-optimized Whisper models via the faster-whisper library
for significantly faster inference than the original openai-whisper.

Supports: tiny, base, small, medium, large-v3, large-v3-turbo, distil-large-v3
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class FasterWhisperModel:
    """
    Wraps faster-whisper for speech-to-text transcription.

    Models are loaded lazily and cached after first load.
    Automatically selects float16 for CUDA and int8 for CPU.
    """

    def __init__(self, model_id: str, device: str = "cpu"):
        """
        Load a Faster-Whisper model.

        Args:
            model_id: Model size or HuggingFace repo ID
                      (e.g. "base", "large-v3", "large-v3-turbo").
            device: "cuda" or "cpu".
        """
        self.model_id = model_id
        self.device = device
        self.model = None
        self._available = False

        try:
            from faster_whisper import WhisperModel

            # Select compute type based on device
            if device == "cuda":
                compute_type = "float16"
            else:
                compute_type = "float32"

            logger.info(f"Loading Faster-Whisper '{model_id}' on {device} ({compute_type})...")
            self.model = WhisperModel(
                model_id,
                device=device,
                compute_type=compute_type,
            )
            self._available = True
            logger.info(f"Faster-Whisper '{model_id}' loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load Faster-Whisper '{model_id}': {e}")

    @property
    def available(self) -> bool:
        return self._available

    def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Float32 numpy array (16 kHz mono).
            language: ISO 639-1 language code.

        Returns:
            Transcribed text string.
        """
        if not self._available or self.model is None:
            return ""

        try:
            segments, info = self.model.transcribe(
                audio,
                language=language,
                beam_size=5,
                vad_filter=False,  # We handle VAD in the pipeline
            )

            # Collect all segment texts
            text = " ".join(segment.text.strip() for segment in segments)
            return text.strip()

        except Exception as e:
            logger.error(f"Faster-Whisper transcription error: {e}")
            return ""
