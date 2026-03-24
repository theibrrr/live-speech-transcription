"""
nemo_diarization_model.py - Wrapper for NVIDIA NeMo speaker diarization.

Uses NeMo's TitaNet speaker embedding model for both diarization
(windowed embedding extraction + agglomerative clustering) and
session-level embedding stabilization.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class NemoDiarizationModel:
    """NeMo TitaNet-based speaker diarization and embedding extraction."""

    def __init__(
        self,
        embedding_model_id: str = "nvidia/speakerverification_en_titanet_large",
        device: str = "cpu",
        window_s: float = 1.5,
        hop_s: float = 0.75,
        cluster_threshold: float = 0.5,
    ):
        self.embedding_model_id = embedding_model_id
        self.device = device
        self.window_s = window_s
        self.hop_s = hop_s
        self.cluster_threshold = cluster_threshold
        self._model = None
        self._available = False

        try:
            from nemo.collections.asr.models import EncDecSpeakerLabelModel

            logger.info("Loading NeMo speaker embedding model: %s ...", embedding_model_id)
            self._model = EncDecSpeakerLabelModel.from_pretrained(embedding_model_id)
            if hasattr(self._model, "to"):
                self._model.to(torch.device(device))
            if hasattr(self._model, "eval"):
                self._model.eval()
            self._available = True
            logger.info("NeMo speaker embedding model loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load NeMo speaker model: %s", exc)

    @property
    def available(self) -> bool:
        return self._available

    def diarize(self, audio: np.ndarray, sample_rate: int = 16000) -> list[dict]:
        """Segment audio into speaker-labeled regions via windowed embeddings + clustering."""
        if not self._available or self._model is None or audio.size == 0:
            return []

        try:
            windows = self._create_windows(audio, sample_rate)
            if not windows:
                return []

            embeddings = self._extract_window_embeddings(windows, sample_rate)
            if embeddings is None or len(embeddings) == 0:
                return []

            labels = self._cluster_embeddings(embeddings)
            return self._windows_to_segments(windows, labels, sample_rate)
        except Exception as exc:
            logger.error("NeMo diarization failed: %s", exc)
            return []

    def extract_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """Extract a speaker embedding from a single audio clip."""
        if not self._available or self._model is None or audio.size == 0:
            return None

        try:
            audio_tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            audio_len = torch.tensor([audio_tensor.shape[1]])

            with torch.no_grad():
                _, emb = self._model.forward(
                    input_signal=audio_tensor.to(self.device),
                    input_signal_length=audio_len.to(self.device),
                )

            embedding = emb.squeeze().cpu().numpy().astype(np.float32).reshape(-1)
            return embedding if embedding.size > 0 else None
        except Exception as exc:
            logger.error("NeMo embedding extraction failed: %s", exc)
            return None

    # ── Internal helpers ────────────────────────────────────────────────

    def _create_windows(self, audio: np.ndarray, sample_rate: int) -> list[dict]:
        total_samples = audio.shape[0]
        window_samples = int(self.window_s * sample_rate)
        hop_samples = int(self.hop_s * sample_rate)

        if total_samples < window_samples // 2:
            return [{"start_sample": 0, "end_sample": total_samples, "audio": audio}]

        windows = []
        start = 0
        while start < total_samples:
            end = min(start + window_samples, total_samples)
            if end - start < window_samples // 4:
                break
            windows.append({
                "start_sample": start,
                "end_sample": end,
                "audio": audio[start:end],
            })
            start += hop_samples
        return windows

    def _extract_window_embeddings(self, windows: list[dict], sample_rate: int) -> Optional[np.ndarray]:
        max_len = max(w["audio"].shape[0] for w in windows)
        batch = torch.zeros(len(windows), max_len, dtype=torch.float32)
        lengths = torch.zeros(len(windows), dtype=torch.long)

        for i, w in enumerate(windows):
            n = w["audio"].shape[0]
            batch[i, :n] = torch.from_numpy(w["audio"].astype(np.float32))
            lengths[i] = n

        try:
            with torch.no_grad():
                _, embs = self._model.forward(
                    input_signal=batch.to(self.device),
                    input_signal_length=lengths.to(self.device),
                )
            return embs.cpu().numpy().astype(np.float32)
        except Exception as exc:
            logger.error("NeMo batch embedding extraction failed: %s", exc)
            return None

    def _cluster_embeddings(self, embeddings: np.ndarray) -> list[int]:
        n = embeddings.shape[0] if embeddings.ndim == 2 else len(embeddings)
        if n <= 1:
            return [0] * n

        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist

        if embeddings.ndim == 3:
            embeddings = embeddings.reshape(n, -1)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = embeddings / norms

        distances = pdist(normalized, metric="cosine")
        distances = np.maximum(distances, 0.0)

        Z = linkage(distances, method="average")
        labels = fcluster(Z, t=self.cluster_threshold, criterion="distance")
        return [int(l) - 1 for l in labels]

    @staticmethod
    def _windows_to_segments(windows: list[dict], labels: list[int], sample_rate: int) -> list[dict]:
        if not windows or not labels:
            return []

        raw = []
        for window, label in zip(windows, labels):
            raw.append({
                "start_sample": window["start_sample"],
                "end_sample": window["end_sample"],
                "speaker": f"SPEAKER_{label:02d}",
            })

        merged = [raw[0].copy()]
        for seg in raw[1:]:
            if seg["speaker"] == merged[-1]["speaker"]:
                merged[-1]["end_sample"] = seg["end_sample"]
            else:
                merged.append(seg.copy())

        segments = []
        for seg in merged:
            segments.append({
                "start_s": seg["start_sample"] / sample_rate,
                "end_s": seg["end_sample"] / sample_rate,
                "speaker": seg["speaker"],
            })

        segments.sort(key=lambda s: (s["start_s"], s["end_s"]))
        return segments
