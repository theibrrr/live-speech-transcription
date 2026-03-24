"""
base_task.py — Abstract base class for all pipeline tasks.

Defines the contract that every pipeline stage must implement,
following Luigi-style task architecture where each processing step
is an independent, replaceable unit.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseTask(ABC):
    """
    Abstract base class for pipeline processing tasks.

    Each task represents one stage in the audio processing pipeline:
        AudioInput → NoiseSuppression → VAD → STT

    Every task must implement `process()`, which takes input data
    and returns processed output. Tasks can be enabled/disabled
    dynamically.
    """

    def __init__(self, name: str, enabled: bool = True):
        """
        Initialize the base task.

        Args:
            name: Human-readable name of this pipeline stage.
            enabled: Whether this task is active in the pipeline.
        """
        self.name = name
        self.enabled = enabled

    @abstractmethod
    def process(self, data: Any) -> Optional[Any]:
        """
        Process the input data and return the result.

        Args:
            data: Input data from the previous pipeline stage.

        Returns:
            Processed data to pass to the next stage, or None if
            processing should stop (e.g., VAD detects silence).
        """
        pass

    def reset(self):
        """
        Reset any internal state between streaming sessions.

        Override in subclasses that maintain state (e.g., VAD models
        with internal buffers).
        """
        pass

    def __repr__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        return f"<{self.__class__.__name__} '{self.name}' [{status}]>"
