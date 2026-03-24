"""
audio_input_task.py — First stage of the pipeline: raw bytes → numpy array.

Converts raw binary audio data from the WebSocket into a normalized
float32 numpy array suitable for downstream processing.
"""

import logging
import numpy as np
from typing import Optional

from backend.pipeline.base_task import BaseTask

logger = logging.getLogger(__name__)


class AudioInputTask(BaseTask):
    """
    Converts raw binary audio data into a normalized float32 numpy array.

    Expected input: Raw bytes (16-bit PCM, mono, 16 kHz).
    Output: Float32 numpy array in range [-1.0, 1.0].
    """

    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the audio input task.

        Args:
            sample_rate: Expected sample rate of incoming audio.
        """
        super().__init__(name="AudioInput", enabled=True)
        self.sample_rate = sample_rate

    def process(self, data: bytes) -> Optional[np.ndarray]:
        """
        Convert raw bytes to a normalized float32 numpy array.

        Args:
            data: Raw binary audio data (16-bit PCM or float32 PCM).

        Returns:
            Normalized float32 numpy array, or None if conversion fails.
        """
        if not data or len(data) == 0:
            logger.warning("AudioInputTask received empty data.")
            return None

        try:
            # Try interpreting as float32 first (browser sends float32 PCM)
            if len(data) % 4 == 0:
                audio = np.frombuffer(data, dtype=np.float32).copy()
            elif len(data) % 2 == 0:
                # Fallback to 16-bit PCM
                audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                audio /= 32768.0  # Normalize to [-1.0, 1.0]
            else:
                logger.warning(f"Unexpected audio data size: {len(data)} bytes")
                return None

            # Clip to valid range
            audio = np.clip(audio, -1.0, 1.0)

            return audio

        except Exception as e:
            logger.error(f"AudioInputTask error: {e}")
            return None
