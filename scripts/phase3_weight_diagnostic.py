#!/usr/bin/env python
"""Phase-3 §10 diagnostic: height -> training loss weight, BEFORE the expensive run.

Read-only. NO model, NO torch needed. Uses ONLY the TRAINING split (JAX_train) to
derive/verify the height-aware weight -- the test city (OMA) is never touched (§4).

Reports, for w(h) = min(1 + max(h,0)/scale, w_max):
  * min / max / mean / median weight over training valid pixels
  * weight at representative heights (0,2,5,10,15,20,30,40 m)
  * fraction of PIXELS in each weight range
  * fraction of LOSS MASS (share of Sum(w)) in each height bin vs the unweighted
    pixel share -> shows the rebalancing is real but does NOT eliminate ground (§9)

    python scripts/phase3_weight_diagnostic.py --config configs/phase3_weighted.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.config import load_config
from depthwizard.data import fetch, datasets
from depthwizard.models.fusion_head import height_weight

STRIDE = 13  # pixel subsample (matches phase2 diagnostic scale; bounded memory)
REP_HEIGHTS = [0, 2, 5, 10, 15, 20, 30, 40]
WEIGHT_RANGES = [(1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 5.0)]
HEIGHT_BINS = [(0, 2), (2, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40),
               (40, float("inf"))]


def _collect_train_heights(records, tile_size, nodata, building_label):
    """Subsampled valid physical heights (meters) over the TRAIN tiles: all + building."""
    all_h, bld_h = [], []
    for rec in records:
        s = datasets.load_sample(rec, tile_size, nodata, depth_model=None)
        gt = np.asarray(s["gt"], dtype=np.float32)
        valid = np.isfinite(gt)
        if nodata is not None:
            valid &= gt != nodata
        all_h.append(gt[valid][::STRIDE].copy())
        cls = s.get("cls")
        if cls is not None:
            bmask = (np.asarray(cls) == building_label) & valid
            bld_h.append(gt[bmask][::STRIDE].copy())
    a = np.concatenate(all_h) if all_h else np.array([], np.float32)
    b = np.concatenate(bld_h) if bld_h else np.array([], np.float32)
    return a, b


def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase3_weighted.yaml")
    ap.add_argument("--out", default="runs/phase3_diag")
    ap.add_argument("--scale", type=float, default=None, help="override loss_weight_scale")
    ap.add_argument("--wmax", type=float, default=None, help="override loss_weight_max")
    args = ap.parse_args()

    cfg = load_config(args.config)
    scale = args.scale if args.scale is not None else float(
        getattr(cfg.train, "loss_weight_scale", 7.0))
    wmax = args.wmax if args.wmax is not None else float(
        getattr(cfg.train, "loss_weight_max", 5.0))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    source, records = fetch.resolve_records(cfg)
    tr, va, te = datasets.split_by_city(
        records, cfg.split.train_cities, cfg.split.val_cities,
        cfg.split.test_cities, cfg.split.val_fraction_within_train_city,
        cfg.split.seed, cfg.data.max_tiles_per_city)
    print(f"[diag] source={source} | TRAIN tiles used={len(tr)} "
          f"(val/test IGNORED here, §4) | scale={scale} w_max={wmax}")

    h_all, h_bld = _collect_train_heights(
        tr, cfg.data.tile_size, cfg.data.nodata, cfg.data.building_label)
    if h_all.size == 0:
        print("[diag] no training pixels; aborting."); sys.exit(2)

    w_all = height_weight(h_all, scale, wmax)
    w_bld = height_weight(h_bld, scale, wmax) if h_bld.size else np.array([], np.float32)

    L = []
    L.append("# Phase-3 §10 diagnostic — height → training loss weight\n")
    L.append(f"_Read-only, TRAINING split ONLY ({cfg.split.train_cities}); the test "
             f"city ({cfg.split.test_cities}) is untouched (§4). source=`{source}`, "
             f"subsample stride={STRIDE}._\n")
    L.append("## Weight definition\n")
    L.append("```\nw(h) = min(1 + max(h,0)/scale, w_max)   "
             f"[scale={scale:g} m, w_max={wmax:g}]\n```")
    L.append("- basis: **physical height (meters)**, computed BEFORE the log1p "
             "transform → transform-agnostic; encodes metric-space 'tall matters more'.")
    L.append("- ground (h=0) → w=1 (rebalanced, never eliminated); saturates at "
             f"w_max for h ≥ {(wmax-1)*scale:g} m.")
    L.append(f"- scale ≈ JAX-train building-pixel median (7.16 m, runs/phase2_diag) "
             f"→ training-derived, no leakage.\n")

    L.append("## Weight statistics over training valid pixels\n")
    L.append("| population | n (subsampled) | min w | max w | mean w | median w |")
    L.append("|---|--:|--:|--:|--:|--:|")
    for name, w in [("ALL pixels", w_all), ("BUILDING pixels", w_bld)]:
        if w.size:
            L.append(f"| {name} | {w.size:,} | {_fmt(float(w.min()))} | "
                     f"{_fmt(float(w.max()))} | {_fmt(float(w.mean()))} | "
                     f"{_fmt(float(np.median(w)))} |")
        else:
            L.append(f"| {name} | 0 | n/a | n/a | n/a | n/a |")
    L.append(f"\n_Mean w over ALL pixels = **{_fmt(float(w_all.mean()),3)}** — the "
             "weighted mean divides by Σw, so the loss magnitude stays ~unweighted "
             "(effective LR unchanged; the only change is relative emphasis)._\n")

    L.append("## Weight at representative heights\n")
    L.append("| height (m) | " + " | ".join(str(h) for h in REP_HEIGHTS) + " |")
    L.append("|---|" + "|".join(["--:"] * len(REP_HEIGHTS)) + "|")
    L.append("| weight w(h) | " + " | ".join(
        _fmt(float(height_weight(float(h), scale, wmax)), 2) for h in REP_HEIGHTS) + " |")
    L.append("")

    L.append("## Fraction of training pixels by weight range\n")
    L.append("| weight range | ALL pixels | BUILDING pixels |")
    L.append("|---|--:|--:|")
    for lo, hi in WEIGHT_RANGES:
        if lo == hi:  # the cap bucket
            fa = float((w_all >= lo - 1e-6).mean())
            fb = float((w_bld >= lo - 1e-6).mean()) if w_bld.size else float("nan")
            label = f"= {lo:g} (cap)"
        else:
            fa = float(((w_all >= lo) & (w_all < hi)).mean())
            fb = float(((w_bld >= lo) & (w_bld < hi)).mean()) if w_bld.size else float("nan")
            label = f"[{lo:g}, {hi:g})"
        L.append(f"| {label} | {100*fa:.2f}% | "
                 f"{'n/a' if fb != fb else f'{100*fb:.2f}%'} |")
    L.append("")

    L.append("## Rebalancing check — pixel share vs LOSS-MASS share by height bin (ALL px)\n")
    L.append("_Loss-mass share = Σw in bin / Σw total (how much each regime now counts). "
             "Ground share should DROP but stay substantial (not eliminated, §9); tall "
             "bins should RISE._\n")
    L.append("| GT height bin (m) | pixel share | loss-mass share | Δ (mass − pixels) |")
    L.append("|---|--:|--:|--:|")
    tot_w = float(w_all.sum())
    n_tot = int(w_all.size)
    for lo, hi in HEIGHT_BINS:
        inb = (h_all >= lo) & (h_all < hi)
        px_share = float(inb.mean())
        mass_share = float(w_all[inb].sum() / tot_w) if tot_w else float("nan")
        hi_s = "inf" if hi == float("inf") else f"{hi:g}"
        L.append(f"| {lo:g}–{hi_s} | {100*px_share:.2f}% | {100*mass_share:.2f}% "
                 f"| {100*(mass_share-px_share):+.2f}% |")
    L.append(f"\n_Total subsampled ALL pixels: {n_tot:,}. Ground-dominant bins lose "
             "emphasis to buildings/tall structures without being zeroed out._\n")

    md = out / "weight_diagnostic.md"
    md.write_text("\n".join(L), encoding="utf-8")
    print(f"[diag] wrote {md}")

    # figure: w(h) curve + pixel-share vs loss-mass-share bars
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        hs = np.linspace(0, 45, 460)
        ax[0].plot(hs, height_weight(hs, scale, wmax), color="#1f6feb", lw=2)
        for h in REP_HEIGHTS:
            wv = float(height_weight(float(h), scale, wmax))
            ax[0].scatter([h], [wv], color="#c1440e", zorder=5)
            ax[0].annotate(f"{wv:.2f}", (h, wv), textcoords="offset points",
                           xytext=(4, 4), fontsize=8)
        ax[0].axhline(wmax, color="k", ls="--", lw=0.7, label=f"cap w_max={wmax:g}")
        ax[0].axhline(1.0, color="gray", ls=":", lw=0.7, label="ground w=1")
        ax[0].set_xlabel("physical height h (m)"); ax[0].set_ylabel("loss weight w(h)")
        ax[0].set_title(f"Height-aware weight  w(h)=min(1+h/{scale:g}, {wmax:g})")
        ax[0].legend()
        labels, px, ms = [], [], []
        for lo, hi in HEIGHT_BINS:
            inb = (h_all >= lo) & (h_all < hi)
            labels.append(f"{lo:g}-{'inf' if hi == float('inf') else f'{hi:g}'}")
            px.append(100 * float(inb.mean()))
            ms.append(100 * float(w_all[inb].sum() / tot_w) if tot_w else 0.0)
        xs = np.arange(len(labels))
        ax[1].bar(xs - 0.2, px, 0.4, label="pixel share", color="#7d8590")
        ax[1].bar(xs + 0.2, ms, 0.4, label="loss-mass share (weighted)", color="#1f6feb")
        ax[1].set_xticks(xs); ax[1].set_xticklabels(labels, rotation=45)
        ax[1].set_ylabel("% of training pixels / loss mass")
        ax[1].set_title("Rebalancing: ground emphasis ↓, tall emphasis ↑ (not eliminated)")
        ax[1].legend()
        fig.suptitle("Phase-3 §10 — training-derived height-aware loss weight (JAX train only)")
        fig.tight_layout(); fig.savefig(out / "weight_diagnostic.png", dpi=100)
        plt.close(fig)
        print(f"[diag] wrote {out / 'weight_diagnostic.png'}")
    except Exception as e:
        print(f"[diag] figure skipped: {e}")

    # console summary
    print(f"[diag] ALL px: mean w={w_all.mean():.3f} median={np.median(w_all):.3f} "
          f"min={w_all.min():.3f} max={w_all.max():.3f}")
    if w_bld.size:
        print(f"[diag] BLD px: mean w={w_bld.mean():.3f} median={np.median(w_bld):.3f}")


if __name__ == "__main__":
    main()
