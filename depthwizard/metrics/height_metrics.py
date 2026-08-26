"""Height-estimation metrics for DepthWizard Phase-1 feasibility.

All metrics operate on 2-D arrays (a predicted height map and a reference
height map) and are computed ONLY over valid pixels. "Valid" means the
ground-truth is finite and not equal to an optional `nodata` sentinel.

We deliberately keep this module numpy-only (no torch) so it is trivial to
unit-test and reuse anywhere (training loops, evaluation, notebooks).

Definitions
-----------
- MAE   : mean(|pred - gt|)              (meters, if gt is in meters)
- RMSE  : sqrt(mean((pred - gt)**2))     (meters)
- Pearson r : linear correlation between pred and gt over valid pixels.
              Sign matters: for overhead imagery we EXPECT r > 0 (taller
              objects -> larger predicted depth/height). A near-zero r is
              the single strongest "abandon" signal.

Nothing here assumes metric units; MAE/RMSE are only in meters if the inputs
are. For raw relative depth (Baseline A) MAE/RMSE are meaningless in meters,
so callers should report Pearson r (scale-free) for A.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np


def valid_mask(
    gt: np.ndarray,
    pred: Optional[np.ndarray] = None,
    nodata: Optional[float] = None,
    extra_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Boolean mask of pixels usable for evaluation."""
    m = np.isfinite(gt)
    if pred is not None:
        m &= np.isfinite(pred)
    if nodata is not None:
        m &= gt != nodata
    if extra_mask is not None:
        m &= extra_mask.astype(bool)
    return m


@dataclass
class HeightMetrics:
    """Container for a single (pred, gt) comparison over a pixel set."""

    mae: float
    rmse: float
    pearson: float
    n_pixels: int
    mean_gt: float
    mean_pred: float

    def as_dict(self) -> dict:
        return asdict(self)


def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: Optional[np.ndarray] = None,
    nodata: Optional[float] = None,
) -> HeightMetrics:
    """Compute MAE / RMSE / Pearson over valid pixels.

    Returns NaN metrics (with n_pixels=0) if no valid pixels exist, rather
    than raising -- callers aggregate across many tiles and must tolerate
    empty ones.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    m = valid_mask(gt, pred, nodata=nodata, extra_mask=mask)
    n = int(m.sum())
    if n == 0:
        return HeightMetrics(np.nan, np.nan, np.nan, 0, np.nan, np.nan)

    p = pred[m]
    g = gt[m]
    err = p - g
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    # Pearson r, guarding against zero-variance (constant) inputs.
    if np.std(p) < 1e-12 or np.std(g) < 1e-12:
        r = np.nan
    else:
        r = float(np.corrcoef(p, g)[0, 1])

    return HeightMetrics(mae, rmse, r, n, float(np.mean(g)), float(np.mean(p)))


def compute_class_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    cls: Optional[np.ndarray],
    building_label: int = 6,
    nodata: Optional[float] = None,
) -> dict:
    """Metrics split into building vs non-building (ground/other) pixels.

    DFC2019 CLS convention: 2=ground, 5=trees, 6=building, 9=water, 17=bridge.
    If `cls` is None we cannot split -> returns only the 'all' entry.
    """
    out = {"all": compute_metrics(pred, gt, nodata=nodata).as_dict()}
    if cls is None:
        return out
    cls = np.asarray(cls)
    building = cls == building_label
    non_building = ~building
    out["building"] = compute_metrics(
        pred, gt, mask=building, nodata=nodata
    ).as_dict()
    out["non_building"] = compute_metrics(
        pred, gt, mask=non_building, nodata=nodata
    ).as_dict()
    return out


def aggregate_scene_metrics(scene_metrics: list[dict]) -> dict:
    """Aggregate a list of per-scene metric dicts (each {mae,rmse,pearson,...}).

    Reports mean / median / std for each metric across scenes, plus a
    pixel-count-weighted mean (the honest "overall" number). Reporting BOTH
    the per-scene distribution and the pooled number is required by the
    experiment spec -- a single overall number hides per-scene failures.
    """
    keys = ["mae", "rmse", "pearson"]
    valid = [m for m in scene_metrics if m.get("n_pixels", 0) > 0]
    if not valid:
        return {"n_scenes": 0}

    agg: dict = {"n_scenes": len(valid)}
    for k in keys:
        vals = np.array(
            [m[k] for m in valid if np.isfinite(m.get(k, np.nan))], dtype=np.float64
        )
        if vals.size:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_median"] = float(np.median(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        else:
            agg[f"{k}_mean"] = agg[f"{k}_median"] = agg[f"{k}_std"] = np.nan

    # pixel-weighted pooled MAE/RMSE (approximate; exact for MAE, and RMSE
    # via weighted mean of squared errors reconstructed from per-scene rmse).
    w = np.array([m["n_pixels"] for m in valid], dtype=np.float64)
    mae_w = np.array([m["mae"] for m in valid], dtype=np.float64)
    rmse_w = np.array([m["rmse"] for m in valid], dtype=np.float64)
    finite = np.isfinite(mae_w) & np.isfinite(rmse_w)
    if finite.any():
        wf = w[finite]
        agg["mae_pooled"] = float(np.sum(mae_w[finite] * wf) / np.sum(wf))
        agg["rmse_pooled"] = float(
            np.sqrt(np.sum((rmse_w[finite] ** 2) * wf) / np.sum(wf))
        )
    return agg
