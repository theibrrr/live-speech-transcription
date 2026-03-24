"""
pipeline_manager.py - Orchestrates the modular audio processing pipeline.

Builds and executes a configurable pipeline based on user selections:
    AudioInput -> NoiseSuppression -> VAD -> STT
"""

import logging
import time

from backend.config import Config
from backend.model_manager import ModelManager
from backend.pipeline.audio_input_task import AudioInputTask
from backend.pipeline.noise_suppression_task import NoiseSuppressionTask
from backend.pipeline.stt_task import STTTask
from backend.pipeline.vad_task import VADTask

logger = logging.getLogger(__name__)


class PipelineManager:
    """Factory and executor for the audio processing pipeline."""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

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

        self.diarization_model = Config.DEFAULT_DIARIZATION_MODEL
        self.timeline_cursor_ms = 0
        self.next_utterance_id = 1
        self.last_chunk_audio = None

        logger.info("Pipeline assembled:")
        logger.info(f"  {self.audio_input}")
        logger.info(f"  {self.noise_suppression}")
        logger.info(f"  {self.vad}")
        logger.info(f"  {self.stt}")

    def configure(self, config: dict):
        """Reconfigure the pipeline based on user selections."""
        if "ns_enabled" in config:
            self.noise_suppression.enabled = bool(config["ns_enabled"])
            logger.info("Noise suppression: %s", "ON" if config["ns_enabled"] else "OFF")

        if "vad_model" in config:
            self.vad.set_vad_type(config["vad_model"])
            logger.info("VAD model: %s", config["vad_model"])

        if "stt_model" in config:
            self.stt.set_model(config["stt_model"])
            logger.info("STT model: %s", config["stt_model"])

        if "language" in config:
            self.stt.set_language(config["language"])
            logger.info("Language: %s", config["language"])

        if "diarization_model" in config:
            self.diarization_model = config["diarization_model"]
            logger.info("Speaker diarization: %s", self.diarization_model)

        self._normalize_diarization_mode()

    def process(self, raw_audio: bytes) -> dict:
        """Run raw audio bytes through the full pipeline."""
        start_time = time.time()
        self.last_chunk_audio = None

        audio = self.audio_input.process(raw_audio)
        if audio is None:
            return self._result("", start_time, is_speech=False, items=[])

        chunk_start_ms = self.timeline_cursor_ms
        chunk_duration_ms = max(1, int(round(len(audio) * 1000.0 / Config.SAMPLE_RATE)))
        chunk_end_ms = chunk_start_ms + chunk_duration_ms
        self.timeline_cursor_ms = chunk_end_ms
        self.last_chunk_audio = audio.copy()

        processed_audio = self.noise_suppression.process(audio)
        if processed_audio is None:
            return self._result(
                "",
                start_time,
                is_speech=False,
                items=[],
                chunk_start_ms=chunk_start_ms,
                chunk_end_ms=chunk_end_ms,
            )

        processed_audio = self.vad.process(processed_audio)
        if processed_audio is None:
            return self._result(
                "",
                start_time,
                is_speech=False,
                items=[],
                chunk_start_ms=chunk_start_ms,
                chunk_end_ms=chunk_end_ms,
            )

        stt_result = self.stt.process(processed_audio)
        if not stt_result:
            return self._result(
                "",
                start_time,
                is_speech=True,
                items=[],
                chunk_start_ms=chunk_start_ms,
                chunk_end_ms=chunk_end_ms,
            )

        items = self._build_items(
            stt_result=stt_result,
            chunk_start_ms=chunk_start_ms,
            chunk_end_ms=chunk_end_ms,
        )
        text = " ".join(item["text"] for item in items).strip()
        return self._result(
            text,
            start_time,
            is_speech=True,
            items=items,
            chunk_start_ms=chunk_start_ms,
            chunk_end_ms=chunk_end_ms,
        )

    def reset(self):
        """Reset all pipeline tasks for a new streaming session."""
        self.audio_input.reset()
        self.noise_suppression.reset()
        self.vad.reset()
        self.stt.reset()

        self.timeline_cursor_ms = 0
        self.next_utterance_id = 1
        self.last_chunk_audio = None
        logger.info("Pipeline reset for new session.")

    def disable_diarization(self):
        """Disable diarization dynamically if the backend cannot support it."""
        if self.diarization_model != "none":
            logger.warning("Disabling diarization for this session.")
        self.diarization_model = "none"

    def get_status(self) -> dict:
        """Return current pipeline configuration status."""
        return {
            "ns_enabled": self.noise_suppression.enabled,
            "vad_model": self.vad.vad_type,
            "stt_model": self.stt.model_key,
            "language": self.stt.language,
            "diarization_model": self.diarization_model,
            "device": Config.DEVICE,
        }

    def _normalize_diarization_mode(self):
        if self.diarization_model == "none":
            return

        if self.diarization_model not in Config.DIARIZATION_MODELS:
            logger.warning(
                "Unknown diarization model '%s'; falling back to none.",
                self.diarization_model,
            )
            self.diarization_model = "none"
            return

        if not Config.diarization_available(self.diarization_model):
            logger.warning(
                "%s requested but not available; falling back to none.",
                self.diarization_model,
            )
            self.diarization_model = "none"
            return

        if not Config.supports_diarization(self.stt.model_key):
            logger.warning(
                "Model '%s' does not support diarization; falling back to none.",
                self.stt.model_key,
            )
            self.diarization_model = "none"

    def _build_items(self, stt_result: dict, chunk_start_ms: int, chunk_end_ms: int) -> list[dict]:
        items = []
        segments = stt_result.get("segments") or []
        if not segments and stt_result.get("text"):
            segments = [{
                "text": stt_result["text"],
                "start_s": 0.0,
                "end_s": max(0.001, (chunk_end_ms - chunk_start_ms) / 1000.0),
            }]

        for segment in segments:
            text = (segment.get("text") or "").strip()
            if not text:
                continue

            start_s = max(0.0, float(segment.get("start_s", 0.0) or 0.0))
            end_s = max(start_s, float(segment.get("end_s", start_s) or start_s))
            start_ms = chunk_start_ms + int(round(start_s * 1000.0))
            end_ms = chunk_start_ms + int(round(end_s * 1000.0))

            start_ms = max(chunk_start_ms, min(start_ms, chunk_end_ms))
            end_ms = max(start_ms + 1, min(max(end_ms, start_ms + 1), chunk_end_ms))

            items.append({
                "id": f"utt_{self.next_utterance_id}",
                "text": text,
                "speaker": "loading..." if self.diarization_model != "none" else None,
                "speaker_pending": self.diarization_model != "none",
                "start_ms": start_ms,
                "end_ms": end_ms,
            })
            self.next_utterance_id += 1

        return items

    @staticmethod
    def _result(
        text: str,
        start_time: float,
        is_speech: bool,
        items: list[dict],
        chunk_start_ms: int = 0,
        chunk_end_ms: int = 0,
    ) -> dict:
        latency_ms = int((time.time() - start_time) * 1000)
        return {
            "text": text,
            "items": items,
            "latency": latency_ms,
            "is_speech": is_speech,
            "chunk_start_ms": chunk_start_ms,
            "chunk_end_ms": chunk_end_ms,
        }
