"""
speechbrain_diarization_model.py - Wrapper for SpeechBrain speaker diarization.

Uses SpeechBrain's ECAPA-TDNN speaker embedding model for both diarization
(windowed embedding extraction + agglomerative clustering) and
session-level embedding stabilization.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SpeechBrainDiarizationModel:
    """SpeechBrain ECAPA-TDNN based speaker diarization and embedding extraction."""

    def __init__(
        self,
        embedding_model_id: str = "speechbrain/spkrec-ecapa-voxceleb",
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
            self._model = self._load_encoder(embedding_model_id, device)
            self._available = self._model is not None
            if self._available:
                logger.info("SpeechBrain speaker embedding model loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load SpeechBrain speaker model: %s", exc)

    @staticmethod
    def _load_encoder(source: str, device: str):
        logger.info("Loading SpeechBrain speaker embedding model: %s ...", source)
        run_opts = {"device": device}

        try:
            from speechbrain.inference.speaker import SpeakerRecognition
            return SpeakerRecognition.from_hparams(source=source, run_opts=run_opts)
        except Exception:
            pass

        try:
            from speechbrain.inference import EncoderClassifier
            return EncoderClassifier.from_hparams(source=source, run_opts=run_opts)
        except Exception:
            pass

        try:
            from speechbrain.pretrained import EncoderClassifier as LegacyEncoder
            return LegacyEncoder.from_hparams(source=source, run_opts=run_opts)
        except Exception as exc:
            logger.error("All SpeechBrain import paths failed: %s", exc)
            return None

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
            logger.error("SpeechBrain diarization failed: %s", exc)
            return []

    def extract_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """Extract a speaker embedding from a single audio clip."""
        if not self._available or self._model is None or audio.size == 0:
            return None

        try:
            waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                emb = self._model.encode_batch(waveform)
            embedding = emb.squeeze().cpu().numpy().astype(np.float32).reshape(-1)
            return embedding if embedding.size > 0 else None
        except Exception as exc:
            logger.error("SpeechBrain embedding extraction failed: %s", exc)
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

        for i, w in enumerate(windows):
            n = w["audio"].shape[0]
            batch[i, :n] = torch.from_numpy(w["audio"].astype(np.float32))

        try:
            with torch.no_grad():
                embs = self._model.encode_batch(batch)
            result = embs.squeeze(1).cpu().numpy().astype(np.float32)
            if result.ndim == 1:
                result = result.reshape(1, -1)
            return result
        except Exception as exc:
            logger.error("SpeechBrain batch embedding extraction failed: %s", exc)
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
