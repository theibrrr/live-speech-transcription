"""
pyannote_diarization_model.py - Wrapper for pyannote speaker diarization.

Loads both the diarization pipeline and the speaker embedding model so the
backend can resolve stable session-level speaker labels in real time.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any, Callable, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class PyannoteDiarizationModel:
    """Wrap pyannote diarization + embedding extraction behind a simple API."""

    def __init__(
        self,
        diarization_model_id: str,
        embedding_model_id: str,
        device: str = "cpu",
        token: str = "",
        local_model_dir: str = "",
        local_embedding_dir: str = "",
    ):
        self.diarization_model_id = diarization_model_id
        self.embedding_model_id = embedding_model_id
        self.device = device
        self.token = token or ""
        self.local_model_dir = local_model_dir or ""
        self.local_embedding_dir = local_embedding_dir or ""

        self.pipeline = None
        self.embedding_inference = None
        self._available = False

        try:
            from pyannote.audio import Inference, Model, Pipeline

            diarization_source = self._resolve_source(self.local_model_dir, self.diarization_model_id)
            embedding_source = self._resolve_source(self.local_embedding_dir, self.embedding_model_id)

            logger.info("Loading pyannote diarization pipeline from %s...", diarization_source)
            self.pipeline = self._load_pretrained(Pipeline.from_pretrained, diarization_source)
            if self.pipeline is None:
                raise RuntimeError(
                    "pyannote diarization pipeline could not be loaded. "
                    "Check Hugging Face token access and accept the model conditions."
                )

            logger.info("Loading pyannote embedding model from %s...", embedding_source)
            embedding_model = self._load_pretrained(Model.from_pretrained, embedding_source)
            if embedding_model is None:
                raise RuntimeError("pyannote embedding model could not be loaded.")
            if hasattr(embedding_model, "to"):
                embedding_model.to(torch.device(device))

            self.embedding_inference = Inference(embedding_model, window="whole")
            if hasattr(self.pipeline, "to"):
                self.pipeline.to(torch.device(device))

            self._available = True
            logger.info("pyannote diarization loaded successfully.")

        except Exception as exc:
            logger.error("Failed to load pyannote diarization: %s", exc)

    @staticmethod
    def _resolve_source(local_dir: str, remote_id: str) -> str:
        return local_dir if local_dir and os.path.isdir(local_dir) else remote_id

    def _load_pretrained(self, factory: Callable[..., Any], source: str) -> Any:
        if not self.token:
            return factory(source)

        try:
            parameters = inspect.signature(factory).parameters
        except (TypeError, ValueError):
            parameters = {}

        if "token" in parameters:
            return factory(source, token=self.token)
        if "use_auth_token" in parameters:
            return factory(source, use_auth_token=self.token)

        return factory(source)

    @property
    def available(self) -> bool:
        return self._available

    def diarize(self, audio: np.ndarray, sample_rate: int = 16000) -> list[dict]:
        """Return diarization segments using the exclusive output when available."""
        if not self._available or self.pipeline is None or audio.size == 0:
            return []

        waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        output = self.pipeline({
            "waveform": waveform,
            "sample_rate": sample_rate,
        })

        annotation = (
            getattr(output, "exclusive_speaker_diarization", None)
            or getattr(output, "speaker_diarization", None)
            or output
        )

        segments = []
        if hasattr(annotation, "itertracks"):
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                segments.append({
                    "start_s": float(turn.start),
                    "end_s": float(turn.end),
                    "speaker": str(speaker),
                })
        else:
            for entry in annotation:
                if len(entry) == 3:
                    turn, _, speaker = entry
                else:
                    turn, speaker = entry
                segments.append({
                    "start_s": float(turn.start),
                    "end_s": float(turn.end),
                    "speaker": str(speaker),
                })

        segments.sort(key=lambda item: (item["start_s"], item["end_s"], item["speaker"]))
        return segments

    def extract_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """Extract a speaker embedding from a single audio clip."""
        if not self._available or self.embedding_inference is None or audio.size == 0:
            return None

        waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        embedding = self.embedding_inference({
            "waveform": waveform,
            "sample_rate": sample_rate,
        })
        if embedding is None:
            return None

        embedding_array = np.asarray(embedding, dtype=np.float32).reshape(-1)
        return embedding_array if embedding_array.size > 0 else None
