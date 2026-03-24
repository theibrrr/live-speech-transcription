"""
pipeline_manager.py — Orchestrates the modular audio processing pipeline.

Builds and executes a configurable pipeline based on user selections:
    AudioInput → NoiseSuppression → VAD → STT

The pipeline is assembled from independent task classes, each of which
can be enabled/disabled or swapped at runtime.
"""

import logging
import time
import numpy as np
from typing import Optional

from backend.model_manager import ModelManager
from backend.pipeline.audio_input_task import AudioInputTask
from backend.pipeline.noise_suppression_task import NoiseSuppressionTask
from backend.pipeline.vad_task import VADTask
from backend.pipeline.stt_task import STTTask
from backend.config import Config

logger = logging.getLogger(__name__)


class PipelineManager:
    """
    Factory and executor for the audio processing pipeline.

    Constructs a chain of tasks based on a configuration dict and
    runs binary audio data through them, producing transcribed text.
    """

    def __init__(self, model_manager: ModelManager):
        """
        Initialize the pipeline manager.

        Args:
            model_manager: Loaded ModelManager with cached models.
        """
        self.model_manager = model_manager

        # Pipeline stages
        self.audio_input = AudioInputTask(sample_rate=Config.SAMPLE_RATE)
        self.noise_suppression = NoiseSuppressionTask(
            model=model_manager.deepfilter,
            enabled=Config.DEFAULT_NS_ENABLED,
        )
        self.vad = VADTask(
            silero_model=model_manager.silero_vad,
            webrtc_model=model_manager.webrtc_vad,
            vad_type=Config.DEFAULT_VAD_MODEL,
        )
        self.stt = STTTask(
            model_manager=model_manager,
            model_key=Config.DEFAULT_STT_MODEL,
            language=Config.DEFAULT_LANGUAGE,
        )

        logger.info("Pipeline assembled:")
        logger.info(f"  {self.audio_input}")
        logger.info(f"  {self.noise_suppression}")
        logger.info(f"  {self.vad}")
        logger.info(f"  {self.stt}")

    def configure(self, config: dict):
        """
        Reconfigure the pipeline based on user selections.

        Args:
            config: Dict with keys:
                - ns_enabled (bool)
                - vad_model (str): "silero" or "webrtc"
                - stt_model (str): key from Config.STT_MODELS
                - language (str): ISO 639-1 code
        """
        if "ns_enabled" in config:
            self.noise_suppression.enabled = bool(config["ns_enabled"])
            logger.info(f"Noise suppression: {'ON' if config['ns_enabled'] else 'OFF'}")

        if "vad_model" in config:
            self.vad.set_vad_type(config["vad_model"])
            logger.info(f"VAD model: {config['vad_model']}")

        if "stt_model" in config:
            self.stt.set_model(config["stt_model"])
            logger.info(f"STT model: {config['stt_model']}")

        if "language" in config:
            self.stt.set_language(config["language"])
            logger.info(f"Language: {config['language']}")

    def process(self, raw_audio: bytes) -> dict:
        """
        Run raw audio bytes through the full pipeline.

        Args:
            raw_audio: Binary audio data from the WebSocket.

        Returns:
            Dict with text, latency (ms), and is_speech flag.
        """
        start_time = time.time()

        # Stage 1: AudioInput
        audio = self.audio_input.process(raw_audio)
        if audio is None:
            return self._result("", start_time, is_speech=False)

        # Stage 2: NoiseSuppression (optional)
        audio = self.noise_suppression.process(audio)
        if audio is None:
            return self._result("", start_time, is_speech=False)

        # Stage 3: VAD
        audio = self.vad.process(audio)
        if audio is None:
            return self._result("", start_time, is_speech=False)

        # Stage 4: STT
        text = self.stt.process(audio)
        return self._result(text or "", start_time, is_speech=True)

    def reset(self):
        """Reset all pipeline tasks for a new streaming session."""
        self.audio_input.reset()
        self.noise_suppression.reset()
        self.vad.reset()
        self.stt.reset()
        logger.info("Pipeline reset for new session.")

    def get_status(self) -> dict:
        """Return current pipeline configuration status."""
        return {
            "ns_enabled": self.noise_suppression.enabled,
            "vad_model": self.vad.vad_type,
            "stt_model": self.stt.model_key,
            "language": self.stt.language,
            "device": Config.DEVICE,
        }

    @staticmethod
    def _result(text: str, start_time: float, is_speech: bool) -> dict:
        latency_ms = int((time.time() - start_time) * 1000)
        return {"text": text, "latency": latency_ms, "is_speech": is_speech}
