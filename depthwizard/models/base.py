"""Common interface for height-estimation methods.

Every baseline (affine, learned fusion head, RDAH-Net adapter, ...) implements
`HeightEstimator`, so the Phase-1 harness can swap methods without changing the
evaluation code. This directly serves the hard constraint: "keep the
implementation modular so the height-estimation method can be replaced if the
experiment disproves the current hypothesis."

A `Sample` is a lightweight dict with keys:
    rgb   : np.uint8/float  HxWx3
    depth : np.float32      HxW   (frozen Depth Anything V2 relative depth)
    gt    : np.float32      HxW   (reference nDSM/AGL in meters; NaN where invalid)
    cls   : np.int_ | None  HxW   (semantic labels; may be None)
    city  : str
    id    : str
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

Sample = dict


class HeightEstimator(ABC):
    name: str = "base"
    #: Does predict() return metric meters (True) or scale-free relative units (False)?
    metric: bool = False

    @abstractmethod
    def fit(self, train_samples: Iterable[Sample]) -> "HeightEstimator":
        ...

    @abstractmethod
    def predict(self, sample: Sample) -> np.ndarray:
        """Return a predicted height map (HxW) aligned to sample['gt']."""
        ...
