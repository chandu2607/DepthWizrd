"""Qualitative visualization for Phase-1 (matplotlib, guarded import).

These are diagnostic plots for the human reviewer at the checkpoint -- NOT the
3D flythrough (that is deferred until after GO). We render, per selected tile:
RGB | reference nDSM | predicted nDSM | signed error, on a shared height scale,
plus a pred-vs-GT scatter. Error maps are where "good RMSE, wrong structure"
gets caught, so they carry as much weight as the scalar metrics.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..metrics.height_metrics import valid_mask

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False


def _require():
    if not _HAS_MPL:
        raise RuntimeError("viz requires matplotlib. `pip install matplotlib`.")


def save_qualitative(sample, pred, out_path, nodata=None, title=""):
    _require()
    gt = np.asarray(sample["gt"], dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    m = valid_mask(gt, pred, nodata=nodata)
    if m.any():
        vmax = float(np.percentile(gt[m], 98))
        vmin = float(min(0.0, np.percentile(gt[m], 2)))
    else:
        vmin, vmax = 0.0, 1.0
    err = np.where(m, pred - gt, np.nan)
    emax = float(np.percentile(np.abs(err[m]), 98)) if m.any() else 1.0

    fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
    ax[0].imshow(np.asarray(sample["rgb"])); ax[0].set_title("RGB")
    im1 = ax[1].imshow(np.where(m, gt, np.nan), vmin=vmin, vmax=vmax, cmap="viridis")
    ax[1].set_title("reference nDSM (m)"); fig.colorbar(im1, ax=ax[1], fraction=0.046)
    im2 = ax[2].imshow(pred, vmin=vmin, vmax=vmax, cmap="viridis")
    ax[2].set_title("predicted nDSM (m)"); fig.colorbar(im2, ax=ax[2], fraction=0.046)
    im3 = ax[3].imshow(err, vmin=-emax, vmax=emax, cmap="RdBu")
    ax[3].set_title("signed error (m)"); fig.colorbar(im3, ax=ax[3], fraction=0.046)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title)
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=90); plt.close(fig)


def save_scatter(pred, gt, out_path, nodata=None, title="pred vs reference (m)",
                 max_points=40000):
    _require()
    pred = np.asarray(pred, dtype=np.float64); gt = np.asarray(gt, dtype=np.float64)
    m = valid_mask(gt, pred, nodata=nodata)
    p, g = pred[m], gt[m]
    if p.size > max_points:
        idx = np.random.default_rng(0).choice(p.size, max_points, replace=False)
        p, g = p[idx], g[idx]
    fig, a = plt.subplots(figsize=(5, 5))
    a.scatter(g, p, s=2, alpha=0.15, edgecolors="none")
    if g.size:
        lim = [min(g.min(), p.min()), max(g.max(), p.max())]
        a.plot(lim, lim, "r--", lw=1, label="y=x")
        a.set_xlim(lim); a.set_ylim(lim); a.legend(loc="upper left")
    a.set_xlabel("reference nDSM (m)"); a.set_ylabel("predicted nDSM (m)")
    a.set_title(title)
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=100); plt.close(fig)


def select_scenes(per_scene, k=3):
    """Pick best/median/worst tiles by all-pixel MAE for qualitative panels."""
    rows = [(p["id"], p["all"]["mae"]) for p in per_scene
            if p.get("all", {}).get("n_pixels", 0) > 0 and np.isfinite(p["all"]["mae"])]
    if not rows:
        return []
    rows.sort(key=lambda t: t[1])
    picks = {rows[0][0], rows[len(rows) // 2][0], rows[-1][0]}
    return [r[0] for r in rows if r[0] in picks][:k]
