"""
webrtc_vad_model.py — Wrapper for WebRTC Voice Activity Detection.

Uses the webrtcvad library to perform lightweight, CPU-based voice activity
detection. WebRTC VAD requires 16-bit PCM audio at specific frame durations
(10, 20, or 30 ms).
"""

import logging
import struct
import numpy as np

logger = logging.getLogger(__name__)


class WebRTCVADModel:
    """
    Wraps webrtcvad.Vad for real-time voice activity detection.

    This is a lightweight alternative to Silero VAD that runs entirely
    on CPU with minimal latency.
    """

    # WebRTC VAD only supports these frame durations (ms)
    VALID_FRAME_DURATIONS = (10, 20, 30)

    def __init__(self, aggressiveness: int = 2):
        """
        Initialize WebRTC VAD.

        Args:
            aggressiveness: Filtering aggressiveness (0–3).
                0 = least aggressive (more false positives)
                3 = most aggressive (more false negatives)
        """
        self.aggressiveness = aggressiveness
        self.vad = None
        self._available = False

        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(aggressiveness)
            self._available = True
            logger.info(f"WebRTC VAD initialized (aggressiveness={aggressiveness}).")

        except Exception as e:
            logger.error(f"Failed to initialize WebRTC VAD: {e}")

    @property
    def available(self) -> bool:
        """Whether the VAD was initialized successfully."""
        return self._available

    def is_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Detect whether the given audio chunk contains speech.

        Splits the audio into 30ms frames and returns True if more than
        30% of the frames contain speech.

        Args:
            audio: Float32 numpy array of audio samples.
            sample_rate: Sample rate of the audio (default 16 kHz).

        Returns:
            True if speech is detected in the majority of frames.
        """
        if not self._available:
            return True  # Fail-open

        try:
            # Convert float32 [-1.0, 1.0] to 16-bit PCM
            pcm_data = (audio * 32767).astype(np.int16)
            pcm_bytes = pcm_data.tobytes()

            # Use 30ms frames (480 samples at 16kHz)
            frame_duration_ms = 30
            frame_size = int(sample_rate * frame_duration_ms / 1000)
            frame_bytes = frame_size * 2  # 2 bytes per int16 sample

            speech_frames = 0
            total_frames = 0

            for offset in range(0, len(pcm_bytes) - frame_bytes + 1, frame_bytes):
                frame = pcm_bytes[offset:offset + frame_bytes]
                if len(frame) == frame_bytes:
                    is_speech = self.vad.is_speech(frame, sample_rate)
                    if is_speech:
                        speech_frames += 1
                    total_frames += 1

            if total_frames == 0:
                return True  # Fail-open for very short chunks

            # Speech if more than 30% of frames contain voice
            speech_ratio = speech_frames / total_frames
            return speech_ratio > 0.3

        except Exception as e:
            logger.error(f"WebRTC VAD error: {e}")
            return True  # Fail-open
