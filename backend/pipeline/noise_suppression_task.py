"""
noise_suppression_task.py — Pipeline task for DeepFilterNet noise suppression.

Applies real-time noise suppression to audio chunks using DeepFilterNet3.
This task is optional and can be enabled/disabled by the user at runtime.
"""

import logging
import numpy as np
from typing import Optional

from backend.pipeline.base_task import BaseTask
from backend.models.deepfilternet_model import DeepFilterNetModel

logger = logging.getLogger(__name__)


class NoiseSuppressionTask(BaseTask):
    """
    Applies noise suppression using DeepFilterNet3.

    Input: Float32 numpy array (16 kHz).
    Output: Enhanced float32 numpy array (16 kHz).

    The task handles the 16 kHz ↔ 48 kHz resampling internally via
    the DeepFilterNetModel wrapper.
    """

    def __init__(self, model: Optional[DeepFilterNetModel] = None, enabled: bool = False):
        """
        Initialize the noise suppression task.

        Args:
            model: DeepFilterNetModel instance (from ModelManager).
            enabled: Whether noise suppression is active.
        """
        super().__init__(name="NoiseSuppression", enabled=enabled)
        self.model = model

    def process(self, data: np.ndarray) -> Optional[np.ndarray]:
        """
        Apply noise suppression to the audio chunk.

        Args:
            data: Float32 numpy array of audio samples (16 kHz).

        Returns:
            Enhanced audio, or the original audio if NS is disabled/unavailable.
        """
        # Skip if disabled or model unavailable
        if not self.enabled:
            return data

        if self.model is None or not self.model.available:
            logger.debug("NoiseSuppression: model unavailable, passing through.")
            return data

        try:
            enhanced = self.model.enhance(data, sample_rate=16000)
            return enhanced

        except Exception as e:
            logger.error(f"NoiseSuppression error: {e}")
            return data  # Fail-safe: return original audio
