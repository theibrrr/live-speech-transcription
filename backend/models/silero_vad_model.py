"""
silero_vad_model.py — Wrapper for the Silero Voice Activity Detection model.

Loads Silero VAD via torch.hub and provides a simple `is_speech()` interface
that returns True if the audio chunk contains speech.
"""

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


class SileroVADModel:
    """
    Wraps Silero VAD for real-time voice activity detection.

    The model is loaded once via torch.hub and reused for all inference calls.
    """

    def __init__(self, device: str = "cpu", threshold: float = 0.5):
        """
        Load the Silero VAD model.

        Args:
            device: "cuda" or "cpu".
            threshold: Speech probability threshold (0.0–1.0).
        """
        self.device = device
        self.threshold = threshold
        self.model = None
        self._available = False

        try:
            logger.info(f"Loading Silero VAD on {device}...")
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            self.model = model.to(device)
            self._get_speech_timestamps = utils[0]
            self._available = True
            logger.info("Silero VAD loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")

    @property
    def available(self) -> bool:
        """Whether the model was loaded successfully."""
        return self._available

    def is_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Detect whether the given audio chunk contains speech.

        Silero VAD requires exactly 512 samples per call at 16kHz.
        We process the chunk in 512-sample windows and return True
        if any window exceeds the speech threshold.

        Args:
            audio: Float32 numpy array of audio samples.
            sample_rate: Sample rate of the audio (default 16 kHz).

        Returns:
            True if speech is detected above the threshold.
        """
        if not self._available:
            return True  # Fail-open: assume speech if model unavailable

        try:
            WINDOW_SIZE = 512  # Silero VAD requires exactly 512 samples at 16kHz

            # Process in 512-sample windows
            max_confidence = 0.0
            for start in range(0, len(audio) - WINDOW_SIZE + 1, WINDOW_SIZE):
                window = audio[start : start + WINDOW_SIZE]
                audio_tensor = torch.from_numpy(window).to(self.device)
                confidence = self.model(audio_tensor, sample_rate).item()
                max_confidence = max(max_confidence, confidence)

                # Early exit if we already found speech
                if max_confidence >= self.threshold:
                    return True

            return max_confidence >= self.threshold

        except Exception as e:
            logger.error(f"Silero VAD error: {e}")
            return True  # Fail-open

    def reset_states(self):
        """Reset the model's internal states between sessions."""
        if self._available and self.model is not None:
            self.model.reset_states()
