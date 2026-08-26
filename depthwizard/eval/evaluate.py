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
    compute_binned_metrics, aggregate_binned,
)
from ..models.affine import fit_oracle_affine


def _scene_record(pred, gt, cls, nodata, building_label):
    if cls is not None:
        # compute_class_metrics already returns plain dicts per class
        return compute_class_metrics(pred, gt, cls, building_label=building_label,
                                     nodata=nodata)
    return {"all": compute_metrics(pred, gt, nodata=nodata).as_dict()}


def evaluate_estimator(est, samples: Iterable[dict], cfg, group: str,
                       bin_edges=None) -> dict:
    """Run est over samples; return per-scene + aggregated metrics for a group.

    If `bin_edges` is given (Phase-2 §10), also compute per-scene height-binned
    metrics (all pixels + building pixels) on the GT height axis and pixel-pool
    them into aggregate['binned_all'] / aggregate['binned_building']. Left None
    for Phase-1 so its behavior/runtime are unchanged.
    """
    nodata = cfg.data.nodata
    blabel = cfg.data.building_label
    per_scene = []
    binned_all_scenes, binned_bldg_scenes = [], []
    for s in samples:
        pred = est.predict(s)
        rec = _scene_record(pred, s["gt"], s.get("cls"), nodata, blabel)
        rec["id"] = s["id"]
        rec["city"] = s["city"]
        if bin_edges is not None:
            b_all = compute_binned_metrics(pred, s["gt"], bin_edges, nodata=nodata)
            rec["binned_all"] = b_all
            binned_all_scenes.append(b_all)
            cls = s.get("cls")
            if cls is not None:
                import numpy as _np
                bmask = _np.asarray(cls) == blabel
                b_bld = compute_binned_metrics(pred, s["gt"], bin_edges,
                                               mask=bmask, nodata=nodata)
                rec["binned_building"] = b_bld
                binned_bldg_scenes.append(b_bld)
        per_scene.append(rec)

    def agg(kind):
        dicts = [p[kind] for p in per_scene if kind in p and p[kind]["n_pixels"] > 0]
        return aggregate_scene_metrics(dicts) if dicts else {}

    aggregate = {
        "all": agg("all"),
        "building": agg("building"),
        "non_building": agg("non_building"),
    }
    if bin_edges is not None:
        aggregate["binned_all"] = aggregate_binned(binned_all_scenes)
        aggregate["binned_building"] = aggregate_binned(binned_bldg_scenes)

    return {
        "group": group,
        "estimator": est.name,
        "metric_valid": bool(getattr(est, "metric", True)),
        "n_scenes": len(per_scene),
        "cities": sorted({p["city"] for p in per_scene}),
        "aggregate": aggregate,
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
