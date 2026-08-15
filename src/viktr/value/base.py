"""ValueEstimator interface: task-relative progress in [0, 1] for one frame.

Implementations plug into retrieval/fusion.py's value-alignment score
|v_hat_t - v'| and, at rollout time, into the retrieval query itself. The MVP
implementation is RobometerValueEstimator (value/robometer.py); a trained
IC-VFE (paper Sec III-C/D) is a future, non-MVP implementation of this same
interface.
"""

from __future__ import annotations

import abc

import numpy as np


class ValueEstimator(abc.ABC):
    @abc.abstractmethod
    def estimate(self, images: np.ndarray, instruction: str) -> float:
        """images: (T, H, W, 3) uint8 frames observed so far in the current rollout
        (or a demonstration, for offline annotation), ending at the current frame.
        Returns a scalar progress estimate in [0, 1] for the current (last) frame."""

    def estimate_episode(self, images: np.ndarray, instruction: str) -> np.ndarray:
        """Per-frame progress for every prefix [0, t] of a full episode.
        Default: one estimate() call per frame — override for a batched/cheaper path."""
        return np.array([self.estimate(images[: t + 1], instruction) for t in range(len(images))], dtype=np.float32)
