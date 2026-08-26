"""Evaluation loops for Phase-1.

For a fitted estimator we compute, per scene (tile):
  - all-pixel MAE / RMSE / Pearson,
  - building-only and non-building-only MAE / RMSE / Pearson (needs CLS),
then aggregate across scenes (mean/median/std + pixel-pooled) per city group.

We evaluate two groups separately and NEVER pool them:
  - IN-DOMAIN   : held-out tiles from the TRAIN city (val),
  - CROSS-CITY  : the fully held-out TEST city  <-- the generalization number.

For non-metric Baseline A, only Pearson is meaningful (MAE/RMSE are reported but
flagged as not-comparable, since raw depth is unit-less).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from ..metrics.height_metrics import (
    compute_metrics, compute_class_metrics, aggregate_scene_metrics,
)
from ..models.affine import fit_oracle_affine


def _scene_record(pred, gt, cls, nodata, building_label):
    if cls is not None:
        # compute_class_metrics already returns plain dicts per class
        return compute_class_metrics(pred, gt, cls, building_label=building_label,
                                     nodata=nodata)
    return {"all": compute_metrics(pred, gt, nodata=nodata).as_dict()}


def evaluate_estimator(est, samples: Iterable[dict], cfg, group: str) -> dict:
    """Run est over samples; return per-scene + aggregated metrics for a group."""
    nodata = cfg.data.nodata
    blabel = cfg.data.building_label
    per_scene = []
    for s in samples:
        pred = est.predict(s)
        rec = _scene_record(pred, s["gt"], s.get("cls"), nodata, blabel)
        rec["id"] = s["id"]
        rec["city"] = s["city"]
        per_scene.append(rec)

    def agg(kind):
        dicts = [p[kind] for p in per_scene if kind in p and p[kind]["n_pixels"] > 0]
        return aggregate_scene_metrics(dicts) if dicts else {}

    return {
        "group": group,
        "estimator": est.name,
        "metric_valid": bool(getattr(est, "metric", True)),
        "n_scenes": len(per_scene),
        "cities": sorted({p["city"] for p in per_scene}),
        "aggregate": {
            "all": agg("all"),
            "building": agg("building"),
            "non_building": agg("non_building"),
        },
        "per_scene": per_scene,
    }


def evaluate_oracle(samples: Iterable[dict], cfg, group: str) -> dict:
    """Per-image oracle affine UPPER BOUND (peeks at each tile's GT; not deployable)."""
    nodata = cfg.data.nodata
    blabel = cfg.data.building_label
    per_scene = []
    for s in samples:
        pred = fit_oracle_affine(s, nodata=nodata)
        rec = _scene_record(pred, s["gt"], s.get("cls"), nodata, blabel)
        rec["id"] = s["id"]; rec["city"] = s["city"]
        per_scene.append(rec)

    def agg(kind):
        dicts = [p[kind] for p in per_scene if kind in p and p[kind]["n_pixels"] > 0]
        return aggregate_scene_metrics(dicts) if dicts else {}

    return {
        "group": group, "estimator": "oracle_affine_upper_bound",
        "metric_valid": True, "n_scenes": len(per_scene),
        "cities": sorted({p["city"] for p in per_scene}),
        "aggregate": {"all": agg("all"), "building": agg("building"),
                      "non_building": agg("non_building")},
        "per_scene": per_scene,
    }
