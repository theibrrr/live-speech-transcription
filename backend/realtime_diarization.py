"""
realtime_diarization.py - Session-level speaker diarization state.

Runs pyannote on a rolling audio window and stabilizes speaker labels with a
session speaker bank based on running embedding centroids. Speaker labels stay
as "loading..." until the backend has seen enough consistent evidence.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from threading import Lock
from typing import Optional

import numpy as np

from backend.config import Config

logger = logging.getLogger(__name__)


class SessionSpeakerBank:
    """Track stable session speakers using embedding centroids."""

    def __init__(
        self,
        similarity_threshold: float = 0.72,
        pending_similarity_threshold: float = 0.58,
        new_speaker_confirmations: int = 2,
        pending_profile_max_age_ms: int = 20_000,
    ):
        self.similarity_threshold = similarity_threshold
        self.pending_similarity_threshold = pending_similarity_threshold
        self.new_speaker_confirmations = max(1, int(new_speaker_confirmations))
        self.pending_profile_max_age_ms = max(0, int(pending_profile_max_age_ms))
        self._profiles: dict[str, dict] = {}
        self._pending_profiles: dict[str, dict] = {}
        self._next_speaker_index = 1
        self._next_pending_index = 1

    @staticmethod
    def _normalize(embedding: np.ndarray) -> Optional[np.ndarray]:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            return None
        return vector / norm

    def assign(self, embedding: np.ndarray, observation_ms: int = 0) -> Optional[str]:
        """Match the embedding to an existing speaker or confirm a new one."""
        normalized = self._normalize(embedding)
        if normalized is None:
            return None

        self._prune_pending(observation_ms)

        best_label, best_score = self._match_profile(normalized, self._profiles)
        if best_label is not None and best_score >= self.similarity_threshold:
            self._update_profile(self._profiles[best_label], normalized, observation_ms)
            return best_label

        pending_label, pending_score = self._match_profile(normalized, self._pending_profiles)
        if pending_label is not None and pending_score >= self.pending_similarity_threshold:
            pending_profile = self._pending_profiles[pending_label]
            self._update_profile(pending_profile, normalized, observation_ms)
            if pending_profile["count"] >= self.new_speaker_confirmations:
                label = f"speaker_{self._next_speaker_index}"
                self._next_speaker_index += 1
                self._profiles[label] = {
                    "centroid": pending_profile["centroid"].copy(),
                    "count": pending_profile["count"],
                }
                del self._pending_profiles[pending_label]
                return label
            return None

        pending_label = f"candidate_{self._next_pending_index}"
        self._next_pending_index += 1
        self._pending_profiles[pending_label] = {
            "centroid": normalized.astype(np.float32),
            "count": 1,
            "last_seen_ms": observation_ms,
        }
        return None

    def reserve_label(self) -> str:
        """Reserve the next stable speaker label for non-embedding fallback paths."""
        label = f"speaker_{self._next_speaker_index}"
        self._next_speaker_index += 1
        return label

    @staticmethod
    def _match_profile(embedding: np.ndarray, profiles: dict[str, dict]) -> tuple[Optional[str], float]:
        best_label = None
        best_score = -1.0
        for label, profile in profiles.items():
            score = float(np.dot(embedding, profile["centroid"]))
            if score > best_score:
                best_score = score
                best_label = label
        return best_label, best_score

    @staticmethod
    def _update_profile(profile: dict, embedding: np.ndarray, observation_ms: int):
        count = int(profile.get("count", 0))
        updated = profile["centroid"] * count + embedding
        updated /= max(np.linalg.norm(updated), 1e-8)
        profile["centroid"] = updated.astype(np.float32)
        profile["count"] = count + 1
        profile["last_seen_ms"] = observation_ms

    def _prune_pending(self, observation_ms: int):
        if self.pending_profile_max_age_ms <= 0:
            return

        stale_labels = [
            label
            for label, profile in self._pending_profiles.items()
            if observation_ms - int(profile.get("last_seen_ms", observation_ms)) > self.pending_profile_max_age_ms
        ]
        for label in stale_labels:
            del self._pending_profiles[label]


class RealtimeDiarizationSession:
    """Store rolling audio and pending transcript items for pyannote updates."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.window_ms = Config.DIARIZATION_WINDOW_MS
        self.min_window_ms = Config.DIARIZATION_MIN_WINDOW_MS
        self.update_interval_ms = Config.DIARIZATION_UPDATE_INTERVAL_MS
        self.audio_retention_ms = Config.DIARIZATION_AUDIO_RETENTION_MS
        self.embedding_min_ms = Config.DIARIZATION_EMBEDDING_MIN_MS
        self.embedding_max_ms = Config.DIARIZATION_EMBEDDING_MAX_MS
        self.finalization_delay_ms = Config.DIARIZATION_FINALIZATION_DELAY_MS
        self.min_label_observations = Config.DIARIZATION_MIN_LABEL_OBSERVATIONS
        self.min_label_win_ratio = Config.DIARIZATION_MIN_LABEL_WIN_RATIO
        self.min_label_margin_s = Config.DIARIZATION_MIN_LABEL_MARGIN_S
        self.force_assignment_ms = Config.DIARIZATION_FORCE_ASSIGNMENT_MS
        self.force_assignment_ratio = Config.DIARIZATION_FORCE_ASSIGNMENT_RATIO

        self._audio_chunks: deque[dict] = deque()
        self._items: dict[str, dict] = {}
        self._speaker_bank = SessionSpeakerBank(
            similarity_threshold=Config.DIARIZATION_SPEAKER_MATCH_THRESHOLD,
            pending_similarity_threshold=Config.DIARIZATION_PENDING_SPEAKER_THRESHOLD,
            new_speaker_confirmations=Config.DIARIZATION_NEW_SPEAKER_CONFIRMATIONS,
            pending_profile_max_age_ms=Config.DIARIZATION_PENDING_SPEAKER_MAX_AGE_MS,
        )

        self._lock = Lock()
        self._current_audio_end_ms = 0
        self._last_run_audio_end_ms = 0
        self._running = False
        self._disabled = False
        self._raw_local_prefix = "__local__:"
        self._fallback_local_labels: dict[str, str] = {}

    def append_audio_chunk(self, audio: Optional[np.ndarray], start_ms: int, end_ms: int):
        """Append raw session audio so diarization keeps wall-clock timing."""
        if audio is None or audio.size == 0:
            return

        with self._lock:
            self._audio_chunks.append({
                "start_ms": start_ms,
                "end_ms": end_ms,
                "audio": audio.astype(np.float32, copy=True),
            })
            self._current_audio_end_ms = max(self._current_audio_end_ms, end_ms)
            self._prune_audio_locked()

    def register_transcript_items(self, items: list[dict]):
        """Track transcript items until a stable speaker label is assigned."""
        if not items:
            return

        with self._lock:
            for item in items:
                self._items[item["id"]] = {
                    "id": item["id"],
                    "text": item["text"],
                    "start_ms": item["start_ms"],
                    "end_ms": item["end_ms"],
                    "speaker": item.get("speaker"),
                    "speaker_pending": bool(item.get("speaker_pending")),
                    "registered_at_ms": item["end_ms"],
                    "observation_count": 0,
                    "speaker_scores": {},
                    "speaker_hits": {},
                    "last_observed_ms": None,
                }

    def has_pending_work(self) -> bool:
        """Whether the session currently has enough data for another run."""
        with self._lock:
            if self._disabled or self._running:
                return False
            if self._current_audio_end_ms < self.min_window_ms:
                return False
            if self._current_audio_end_ms - self._last_run_audio_end_ms < self.update_interval_ms:
                return False
            return any(item["speaker_pending"] for item in self._items.values())

    def run_pending(self, diarization_model) -> list[dict]:
        """Resolve as many pending speaker labels as possible on the rolling window."""
        snapshot = self._take_snapshot()
        if snapshot is None:
            return []

        try:
            diar_segments = diarization_model.diarize(
                snapshot["audio"],
                sample_rate=self.sample_rate,
            )
            if not diar_segments:
                return self._finish_run(snapshot["audio_end_ms"], [])

            local_speaker_labels = self._resolve_local_speakers(
                diarization_model=diarization_model,
                audio=snapshot["audio"],
                diar_segments=diar_segments,
                items=snapshot["items"],
                audio_start_ms=snapshot["audio_start_ms"],
                audio_end_ms=snapshot["audio_end_ms"],
            )
            observations = self._collect_item_observations(snapshot, diar_segments, local_speaker_labels)
            return self._finish_run(snapshot["audio_end_ms"], observations)

        except Exception as exc:
            logger.error("Realtime diarization failed: %s", exc)
            self._finish_run(snapshot["audio_end_ms"], [])
            return []

    def disable(self) -> list[dict]:
        """Disable diarization and clear all pending labels from current items."""
        with self._lock:
            self._disabled = True
            self._running = False

            updates = []
            for item in self._items.values():
                if item["speaker_pending"]:
                    item["speaker_pending"] = False
                    item["speaker"] = None
                    updates.append({
                        "id": item["id"],
                        "speaker": None,
                        "speaker_pending": False,
                    })
            return updates

    def _take_snapshot(self) -> Optional[dict]:
        with self._lock:
            if self._disabled or self._running:
                return None
            if self._current_audio_end_ms < self.min_window_ms:
                return None
            if self._current_audio_end_ms - self._last_run_audio_end_ms < self.update_interval_ms:
                return None

            pending_items = [item for item in self._items.values() if item["speaker_pending"]]
            if not pending_items:
                return None

            window_end_ms = self._current_audio_end_ms
            window_start_ms = max(0, window_end_ms - self.window_ms)
            audio_chunks = [
                chunk for chunk in self._audio_chunks
                if chunk["end_ms"] > window_start_ms and chunk["start_ms"] < window_end_ms
            ]
            if not audio_chunks:
                return None

            snapshot_audio = self._compose_window_audio(audio_chunks, window_start_ms, window_end_ms)
            if snapshot_audio.size == 0:
                return None

            items = [
                dict(item)
                for item in self._items.values()
                if item["end_ms"] > window_start_ms and item["start_ms"] < window_end_ms
            ]
            if not items:
                return None

            self._running = True
            return {
                "audio": snapshot_audio,
                "audio_start_ms": window_start_ms,
                "audio_end_ms": window_end_ms,
                "items": items,
            }

    def _finish_run(self, audio_end_ms: int, observations: list[dict]) -> list[dict]:
        with self._lock:
            updates = []
            updated_item_ids = set()

            for observation in observations:
                item = self._items.get(observation["id"])
                if item is None or not item["speaker_pending"]:
                    continue
                update = self._apply_observation_locked(item, observation, audio_end_ms)
                if update is not None:
                    updates.append(update)
                    updated_item_ids.add(update["id"])

            for item in self._items.values():
                if not item["speaker_pending"] or item["id"] in updated_item_ids:
                    continue
                update = self._maybe_finalize_item_locked(item, audio_end_ms)
                if update is not None:
                    updates.append(update)

            self._prune_items_locked()
            self._last_run_audio_end_ms = max(self._last_run_audio_end_ms, audio_end_ms)
            self._running = False
            return updates

    def _compose_window_audio(self, chunks: list[dict], window_start_ms: int, window_end_ms: int) -> np.ndarray:
        total_samples = max(
            1,
            int(round((window_end_ms - window_start_ms) * self.sample_rate / 1000.0)),
        )
        canvas = np.zeros(total_samples, dtype=np.float32)

        for chunk in chunks:
            chunk_start_ms = max(chunk["start_ms"], window_start_ms)
            chunk_end_ms = min(chunk["end_ms"], window_end_ms)
            if chunk_end_ms <= chunk_start_ms:
                continue

            src_start = int(round((chunk_start_ms - chunk["start_ms"]) * self.sample_rate / 1000.0))
            src_end = int(round((chunk_end_ms - chunk["start_ms"]) * self.sample_rate / 1000.0))
            dst_start = int(round((chunk_start_ms - window_start_ms) * self.sample_rate / 1000.0))
            dst_end = dst_start + max(0, src_end - src_start)

            if src_end <= src_start or dst_end <= dst_start:
                continue

            canvas[dst_start:dst_end] = chunk["audio"][src_start:src_end]

        return canvas

    def _resolve_local_speakers(
        self,
        diarization_model,
        audio: np.ndarray,
        diar_segments: list[dict],
        items: list[dict],
        audio_start_ms: int,
        audio_end_ms: int,
    ) -> dict[str, Optional[str]]:
        grouped_segments: dict[str, list[dict]] = defaultdict(list)
        for segment in diar_segments:
            grouped_segments[segment["speaker"]].append(segment)

        labels: dict[str, Optional[str]] = {}
        for local_speaker, segments in grouped_segments.items():
            embedding_audio = self._collect_embedding_audio(audio, segments)
            session_label = None
            if embedding_audio is not None and embedding_audio.size > 0:
                embedding = diarization_model.extract_embedding(
                    embedding_audio,
                    sample_rate=self.sample_rate,
                )
                if embedding is not None:
                    session_label = self._speaker_bank.assign(embedding, observation_ms=audio_end_ms)

            if session_label is None:
                session_label = self._vote_from_existing_items(segments, items, audio_start_ms)
            if session_label is None:
                session_label = self._fallback_local_labels.get(local_speaker)

            labels[local_speaker] = session_label

        return labels

    def _collect_embedding_audio(self, audio: np.ndarray, segments: list[dict]) -> Optional[np.ndarray]:
        collected = []
        collected_ms = 0

        for segment in sorted(segments, key=lambda item: item["end_s"] - item["start_s"], reverse=True):
            start_ms = int(segment["start_s"] * 1000)
            end_ms = int(segment["end_s"] * 1000)
            duration_ms = max(0, end_ms - start_ms)
            if duration_ms <= 0:
                continue

            start_sample = int(round(start_ms * self.sample_rate / 1000.0))
            end_sample = int(round(end_ms * self.sample_rate / 1000.0))
            clip = audio[start_sample:end_sample]
            if clip.size == 0:
                continue

            collected.append(clip.astype(np.float32, copy=False))
            collected_ms += duration_ms
            if collected_ms >= self.embedding_max_ms:
                break

        if collected_ms < self.embedding_min_ms or not collected:
            return None

        return np.concatenate(collected)

    def _vote_from_existing_items(self, segments: list[dict], items: list[dict], audio_start_ms: int) -> Optional[str]:
        votes: dict[str, float] = defaultdict(float)

        for item in items:
            if item.get("speaker_pending") or not item.get("speaker"):
                continue

            item_start_s = max(0.0, (item["start_ms"] - audio_start_ms) / 1000.0)
            item_end_s = max(item_start_s, (item["end_ms"] - audio_start_ms) / 1000.0)
            for segment in segments:
                overlap = self._overlap_seconds(
                    item_start_s,
                    item_end_s,
                    segment["start_s"],
                    segment["end_s"],
                )
                if overlap > 0:
                    votes[item["speaker"]] += overlap

        if not votes:
            return None
        return max(votes.items(), key=lambda pair: pair[1])[0]

    def _collect_item_observations(
        self,
        snapshot: dict,
        diar_segments: list[dict],
        local_labels: dict[str, Optional[str]],
    ) -> list[dict]:
        observations = []

        for item in snapshot["items"]:
            if not item.get("speaker_pending"):
                continue

            item_start_s = max(0.0, (item["start_ms"] - snapshot["audio_start_ms"]) / 1000.0)
            item_end_s = max(item_start_s, (item["end_ms"] - snapshot["audio_start_ms"]) / 1000.0)
            candidate_scores: dict[str, float] = defaultdict(float)

            for segment in diar_segments:
                overlap = self._overlap_seconds(
                    item_start_s,
                    item_end_s,
                    segment["start_s"],
                    segment["end_s"],
                )
                if overlap <= 0:
                    continue

                speaker_label = local_labels.get(segment["speaker"])
                if not speaker_label:
                    speaker_label = self._raw_local_label(segment["speaker"])
                candidate_scores[speaker_label] += overlap

            if not candidate_scores:
                continue

            observations.append({
                "id": item["id"],
                "candidate_scores": dict(candidate_scores),
            })

        return observations

    def _apply_observation_locked(self, item: dict, observation: dict, audio_end_ms: int) -> Optional[dict]:
        candidate_scores = observation.get("candidate_scores") or {}
        if not candidate_scores:
            return None

        speaker_scores = item.setdefault("speaker_scores", {})
        speaker_hits = item.setdefault("speaker_hits", {})
        best_label = max(candidate_scores.items(), key=lambda pair: pair[1])[0]

        for label, score in candidate_scores.items():
            if score <= 0:
                continue
            speaker_scores[label] = float(speaker_scores.get(label, 0.0)) + float(score)

        speaker_hits[best_label] = int(speaker_hits.get(best_label, 0)) + 1
        item["observation_count"] = int(item.get("observation_count", 0)) + 1
        item["last_observed_ms"] = audio_end_ms

        return self._maybe_finalize_item_locked(item, audio_end_ms)

    def _maybe_finalize_item_locked(self, item: dict, audio_end_ms: int) -> Optional[dict]:
        if not item.get("speaker_pending"):
            return None

        speaker_scores = item.get("speaker_scores") or {}
        if not speaker_scores:
            return None

        ranked_scores = sorted(
            speaker_scores.items(),
            key=lambda pair: pair[1],
            reverse=True,
        )
        best_label, best_score = ranked_scores[0]
        second_score = ranked_scores[1][1] if len(ranked_scores) > 1 else 0.0
        total_score = sum(score for _, score in ranked_scores)
        observations = int(item.get("observation_count", 0))
        best_hits = int((item.get("speaker_hits") or {}).get(best_label, 0))
        age_ms = max(0, audio_end_ms - int(item.get("end_ms", 0)))
        win_ratio = (best_score / total_score) if total_score > 0 else 0.0
        margin_s = best_score - second_score

        ready_to_finalize = (
            age_ms >= self.finalization_delay_ms
            and observations >= self.min_label_observations
            and best_hits >= self.min_label_observations
            and win_ratio >= self.min_label_win_ratio
            and margin_s >= self.min_label_margin_s
        )
        forced_finalize = (
            age_ms >= self.force_assignment_ms
            and observations >= 1
            and best_hits >= 1
            and win_ratio >= self.force_assignment_ratio
        )

        if not ready_to_finalize and not forced_finalize:
            return None

        if best_label.startswith(self._raw_local_prefix):
            best_label = self._materialize_fallback_label_locked(best_label)

        item["speaker"] = best_label
        item["speaker_pending"] = False
        return {
            "id": item["id"],
            "speaker": best_label,
            "speaker_pending": False,
        }

    def _prune_audio_locked(self):
        keep_after_ms = self._current_audio_end_ms - self.audio_retention_ms
        while self._audio_chunks and self._audio_chunks[0]["end_ms"] < keep_after_ms:
            self._audio_chunks.popleft()

    def _prune_items_locked(self):
        keep_after_ms = self._current_audio_end_ms - self.audio_retention_ms
        removable_ids = [
            item_id
            for item_id, item in self._items.items()
            if not item.get("speaker_pending") and item["end_ms"] < keep_after_ms
        ]
        for item_id in removable_ids:
            del self._items[item_id]

    @staticmethod
    def _overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
        return max(0.0, min(a_end, b_end) - max(a_start, b_start))

    def _raw_local_label(self, local_speaker: str) -> str:
        return f"{self._raw_local_prefix}{local_speaker}"

    def _materialize_fallback_label_locked(self, raw_local_label: str) -> str:
        local_speaker = raw_local_label[len(self._raw_local_prefix):]
        label = self._fallback_local_labels.get(local_speaker)
        if label is not None:
            return label

        label = self._speaker_bank.reserve_label()
        self._fallback_local_labels[local_speaker] = label
        logger.info(
            "Fallback speaker assignment: local speaker '%s' -> %s",
            local_speaker,
            label,
        )
        return label
