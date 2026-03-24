"""
whisper_model.py — Wrapper for OpenAI Whisper speech-to-text models.

Loads Whisper models (base / large) and provides a simple `transcribe()`
interface for real-time speech recognition.
"""

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


class WhisperModel:
    """
    Wraps OpenAI Whisper for speech-to-text transcription.

    Supports loading multiple model sizes and switching between them.
    Models are cached in memory after first load.
    """

    def __init__(self, model_name: str = "base", device: str = "cpu"):
        """
        Load a Whisper model.

        Args:
            model_name: Whisper model size ("base" or "large").
            device: "cuda" or "cpu".
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._available = False

        try:
            import whisper

            logger.info(f"Loading Whisper '{model_name}' on {device}...")
            self.model = whisper.load_model(model_name, device=device)
            self._available = True
            logger.info(f"Whisper '{model_name}' loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load Whisper '{model_name}': {e}")

    @property
    def available(self) -> bool:
        """Whether the model was loaded successfully."""
        return self._available

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000, language: str = "en") -> str:
        """
        Transcribe an audio chunk to text.

        Args:
            audio: Float32 numpy array of audio samples.
            sample_rate: Sample rate of the audio (default 16 kHz).
            language: Language code (e.g. "en", "tr", "de").

        Returns:
            Transcribed text as a string, or empty string on failure.
        """
        if not self._available or self.model is None:
            return ""

        try:
            # Whisper expects float32 audio at 16 kHz
            # Pad or trim to exactly 30 seconds for optimal processing
            import whisper

            audio_padded = whisper.pad_or_trim(audio.astype(np.float32))

            # Create mel spectrogram (large-v3 uses 128 mels, base uses 80)
            n_mels = self.model.dims.n_mels
            mel = whisper.log_mel_spectrogram(audio_padded, n_mels=n_mels).to(self.device)

            # Decode with suppression of blank/special tokens
            options = whisper.DecodingOptions(
                language=language,
                without_timestamps=True,
                fp16=(self.device == "cuda"),
            )
            result = whisper.decode(self.model, mel, options)

            return result.text.strip()

        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return ""
