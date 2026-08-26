"""Baselines A and B: raw relative depth, and simple affine calibration.

Baseline A  (RawDepth)      : identity on Depth Anything V2 relative depth.
    Not metric. Only Pearson r vs GT is meaningful. This measures the CEILING
    of what any monotonic scaling could achieve -- if r is near zero here, no
    amount of calibration (affine or learned) can rescue it.

Baseline B  (GlobalAffine)  : h_hat = a * depth + b, with a single (a, b) fit
    on the TRAINING city's pixels and applied unchanged to the test city.
    This is the "simple scale calibration" the problem statement suggests and
    the bar that the learned head (Baseline C) must beat, especially cross-city.

Also provided: `fit_oracle_affine`, a per-image best-case affine fit. It is an
UPPER BOUND (it peeks at each test image's GT) and is NOT deployable -- we
report it only to separate "depth carries the signal but scale drifts per
scene" from "depth lacks the signal entirely".
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from .base import HeightEstimator, Sample
from ..metrics.height_metrics import valid_mask


def _fit_affine(d: np.ndarray, h: np.ndarray, robust: bool = True) -> Tuple[float, float]:
    """Least-squares fit of h ~ a*d + b. Optional trimmed-residual robustness."""
    A = np.stack([d, np.ones_like(d)], axis=1)
    sol, *_ = np.linalg.lstsq(A, h, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    if robust:
        # One IRLS-style trim: drop the worst 10% residuals and refit.
        resid = np.abs((a * d + b) - h)
        keep = resid <= np.quantile(resid, 0.90)
        if keep.sum() > 10:
            A2 = np.stack([d[keep], np.ones_like(d[keep])], axis=1)
            sol, *_ = np.linalg.lstsq(A2, h[keep], rcond=None)
            a, b = float(sol[0]), float(sol[1])
    return a, b


class RawDepth(HeightEstimator):
    """Baseline A: return relative depth unchanged (scale-free)."""

    name = "A_raw_depth"
    metric = False

    def fit(self, train_samples: Iterable[Sample]) -> "RawDepth":
        return self

    def predict(self, sample: Sample) -> np.ndarray:
        return np.asarray(sample["depth"], dtype=np.float32)


class GlobalAffine(HeightEstimator):
    """Baseline B: single global affine map depth -> meters, fit on train city."""

    name = "B_global_affine"
    metric = True

    def __init__(self, robust: bool = True, max_pixels: int = 2_000_000, seed: int = 0):
        self.robust = robust
        self.max_pixels = max_pixels
        self.rng = np.random.default_rng(seed)
        self.a: float = 1.0
        self.b: float = 0.0

    def fit(self, train_samples: Iterable[Sample]) -> "GlobalAffine":
        ds, hs = [], []
        for s in train_samples:
            d = np.asarray(s["depth"], dtype=np.float64)
            h = np.asarray(s["gt"], dtype=np.float64)
            m = valid_mask(h, d)
            if m.any():
                ds.append(d[m])
                hs.append(h[m])
        if not ds:
            raise ValueError("GlobalAffine.fit: no valid training pixels")
        d = np.concatenate(ds)
        h = np.concatenate(hs)
        if d.size > self.max_pixels:  # subsample for a fast, stable lstsq
            idx = self.rng.choice(d.size, self.max_pixels, replace=False)
            d, h = d[idx], h[idx]
        self.a, self.b = _fit_affine(d, h, robust=self.robust)
        return self

    def predict(self, sample: Sample) -> np.ndarray:
        d = np.asarray(sample["depth"], dtype=np.float32)
        return (self.a * d + self.b).astype(np.float32)


def fit_oracle_affine(sample: Sample, nodata: float | None = None) -> np.ndarray:
    """Per-image oracle affine (UPPER BOUND, not deployable)."""
    d = np.asarray(sample["depth"], dtype=np.float64)
    h = np.asarray(sample["gt"], dtype=np.float64)
    m = valid_mask(h, d, nodata=nodata)
    if m.sum() < 10:
        return np.full_like(d, np.nan, dtype=np.float32)
    a, b = _fit_affine(d[m], h[m], robust=True)
    return (a * d + b).astype(np.float32)
