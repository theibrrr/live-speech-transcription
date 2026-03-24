"""
vad_task.py — Pipeline task for Voice Activity Detection.

Dispatches to either Silero VAD or WebRTC VAD depending on user selection.
If voice is detected, audio passes through to the next stage.
If silence is detected, returns None to skip STT processing.
"""

import logging
import numpy as np
from typing import Optional

from backend.pipeline.base_task import BaseTask
from backend.models.silero_vad_model import SileroVADModel
from backend.models.webrtc_vad_model import WebRTCVADModel

logger = logging.getLogger(__name__)


class VADTask(BaseTask):
    """
    Voice Activity Detection pipeline task.

    Supports two VAD backends:
        - "silero": Neural network-based (more accurate, slightly more latency)
        - "webrtc": Rule-based (faster, lighter, CPU-only)

    Input: Float32 numpy array (16 kHz).
    Output: Same array if speech detected, None if silence.
    """

    def __init__(
        self,
        silero_model: Optional[SileroVADModel] = None,
        webrtc_model: Optional[WebRTCVADModel] = None,
        vad_type: str = "silero",
    ):
        """
        Initialize the VAD task.

        Args:
            silero_model: SileroVADModel instance (from ModelManager).
            webrtc_model: WebRTCVADModel instance (from ModelManager).
            vad_type: Which VAD backend to use ("silero" or "webrtc").
        """
        super().__init__(name="VAD", enabled=True)
        self.silero_model = silero_model
        self.webrtc_model = webrtc_model
        self.vad_type = vad_type

    def set_vad_type(self, vad_type: str):
        """Switch the VAD backend at runtime."""
        if vad_type in ("silero", "webrtc", "none"):
            self.vad_type = vad_type
        else:
            logger.warning(f"Unknown VAD type: {vad_type}, keeping {self.vad_type}")

    def process(self, data: np.ndarray) -> Optional[np.ndarray]:
        """
        Check if the audio chunk contains speech.

        Args:
            data: Float32 numpy array of audio samples (16 kHz).

        Returns:
            The audio array if speech is detected, None if silent.
        """
        if not self.enabled:
            return data

        try:
            # ── Energy gate: reject very quiet chunks immediately ────────
            rms_energy = float(np.sqrt(np.mean(data ** 2)))
            if rms_energy < 0.005:
                logger.debug(f"VAD: energy too low ({rms_energy:.4f}), skipping.")
                return None

            is_speech = self._detect_speech(data)

            if is_speech:
                return data
            else:
                logger.debug("VAD: silence detected, skipping STT.")
                return None

        except Exception as e:
            logger.error(f"VAD error: {e}")
            return data  # Fail-open: assume speech on error

    def _detect_speech(self, audio: np.ndarray) -> bool:
        """Dispatch to the selected VAD backend."""
        if self.vad_type == "none":
            return True  # No VAD — pass all audio through
        elif self.vad_type == "silero" and self.silero_model is not None:
            return self.silero_model.is_speech(audio)
        elif self.vad_type == "webrtc" and self.webrtc_model is not None:
            return self.webrtc_model.is_speech(audio)
        else:
            logger.warning(f"VAD model '{self.vad_type}' unavailable, passing through.")
            return True

    def reset(self):
        """Reset VAD internal states between sessions."""
        if self.silero_model is not None:
            self.silero_model.reset_states()
