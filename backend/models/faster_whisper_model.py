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

    def transcribe_detailed(self, audio: np.ndarray, language: str = "en") -> dict:
        """
        Transcribe audio and return timestamped segment metadata.

        Returns:
            Dict with `text` and `segments`.
        """
        if not self._available or self.model is None:
            return {"text": "", "segments": []}

        try:
            segments, _ = self.model.transcribe(
                audio,
                language=language,
                beam_size=5,
                vad_filter=False,
            )

            collected_segments = []
            for segment in segments:
                text = segment.text.strip()
                if not text:
                    continue
                collected_segments.append({
                    "text": text,
                    "start_s": float(getattr(segment, "start", 0.0) or 0.0),
                    "end_s": float(getattr(segment, "end", 0.0) or 0.0),
                })

            full_text = " ".join(segment["text"] for segment in collected_segments).strip()
            return {"text": full_text, "segments": collected_segments}

        except Exception as e:
            logger.error(f"Faster-Whisper transcription error: {e}")
            return {"text": "", "segments": []}

    def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Float32 numpy array (16 kHz mono).
            language: ISO 639-1 language code.

        Returns:
            Transcribed text string.
        """
        return self.transcribe_detailed(audio, language=language)["text"]
