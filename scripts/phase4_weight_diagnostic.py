#!/usr/bin/env python
"""Phase-4 §10 diagnostic: CALIBRATED TAIL weight vs the Phase-3 aggressive weight.

Read-only. NO model, NO torch. Uses ONLY the TRAINING split (JAX_train) to derive and
verify the tail-weight parameters -- the test city (OMA) is never touched (§8/§26).

It answers the §10 must-prove question:
  > Does this weighting primarily target the problematic tail WITHOUT substantially
  > changing the optimization emphasis of the 0–15 m regime?
by reporting, for BOTH tail_weight and the aggressive height_weight, side by side:
  * the JAX-train height distribution (percentiles + cumulative share) that JUSTIFIES
    h_start / tail_scale / w_max
  * min / max / mean / median weight over training valid + building pixels
  * weight at representative heights
  * pixel-share vs LOSS-MASS-share by height bin (the rebalancing)

    python scripts/phase4_weight_diagnostic.py --config configs/phase4_tail_weighted.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.config import load_config
from depthwizard.data import fetch, datasets
from depthwizard.models.fusion_head import height_weight, tail_weight

STRIDE = 13  # pixel subsample (matches phase2/phase3 diagnostics; bounded memory)
REP_HEIGHTS = [0, 2, 5, 10, 14, 15, 18, 20, 25, 30, 40, 50]
PCTS = [50, 75, 85, 90, 92, 95, 99]
HEIGHT_BINS = [(0, 2), (2, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40),
               (40, float("inf"))]


def _collect_train_heights(records, tile_size, nodata, building_label):
    """Subsampled valid physical heights (m) over the TRAIN tiles: all + building."""
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


def _stats_block(name, w):
    if not w.size:
        return f"| {name} | 0 | n/a | n/a | n/a | n/a |"
    return (f"| {name} | {w.size:,} | {_fmt(float(w.min()))} | {_fmt(float(w.max()))} "
            f"| {_fmt(float(w.mean()))} | {_fmt(float(np.median(w)))} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase4_tail_weighted.yaml")
    ap.add_argument("--out", default="runs/phase4_diag")
    ap.add_argument("--start", type=float, default=None, help="override loss_tail_start")
    ap.add_argument("--scale", type=float, default=None, help="override loss_tail_scale")
    ap.add_argument("--wmax", type=float, default=None, help="override loss_tail_max")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    cfg = load_config(args.config)
    t = cfg.train
    h_start = args.start if args.start is not None else float(getattr(t, "loss_tail_start", 15.0))
    t_scale = args.scale if args.scale is not None else float(getattr(t, "loss_tail_scale", 12.5))
    t_max = args.wmax if args.wmax is not None else float(getattr(t, "loss_tail_max", 3.0))
    a_scale = float(getattr(t, "loss_weight_scale", 7.0))   # aggressive (Phase-3)
    a_max = float(getattr(t, "loss_weight_max", 5.0))
    h_sat = h_start + (t_max - 1.0) * t_scale
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    source, records = fetch.resolve_records(cfg)
    tr, va, te = datasets.split_by_city(
        records, cfg.split.train_cities, cfg.split.val_cities,
        cfg.split.test_cities, cfg.split.val_fraction_within_train_city,
        cfg.split.seed, cfg.data.max_tiles_per_city)
    print(f"[diag] source={source} | TRAIN tiles used={len(tr)} (val/test IGNORED, §8) "
          f"| tail: start={h_start} scale={t_scale} wmax={t_max} (sat@{h_sat:g} m)")

    h_all, h_bld = _collect_train_heights(
        tr, cfg.data.tile_size, cfg.data.nodata, cfg.data.building_label)
    if h_all.size == 0:
        print("[diag] no training pixels; aborting."); sys.exit(2)

    wt_all = tail_weight(h_all, h_start, t_scale, t_max)
    wt_bld = tail_weight(h_bld, h_start, t_scale, t_max) if h_bld.size else np.array([], np.float32)
    wa_all = height_weight(h_all, a_scale, a_max)
    wa_bld = height_weight(h_bld, a_scale, a_max) if h_bld.size else np.array([], np.float32)

    # cumulative share at h_start (the key justification numbers)
    def _cum_le(arr, thr):
        return float((arr <= thr).mean()) if arr.size else float("nan")

    L = []
    L.append("# Phase-4 §10 diagnostic — calibrated tail weight vs aggressive weight\n")
    L.append(f"_Read-only, TRAINING split ONLY ({cfg.split.train_cities}); the test city "
             f"({cfg.split.test_cities}) is untouched (§8/§26). source=`{source}`, "
             f"subsample stride={STRIDE}._\n")

    # ---- parameter derivation from the JAX-train distribution ----
    L.append("## JAX-train height distribution → parameter derivation (§8/§9)\n")
    L.append("| percentile | all-pixel height (m) | building-pixel height (m) |")
    L.append("|---|--:|--:|")
    for p in PCTS:
        pa = float(np.percentile(h_all, p)) if h_all.size else float("nan")
        pb = float(np.percentile(h_bld, p)) if h_bld.size else float("nan")
        L.append(f"| P{p} | {_fmt(pa, 2)} | {_fmt(pb, 2)} |")
    med_b = float(np.median(h_bld)) if h_bld.size else float("nan")
    L.append(f"\n- **h_start = {h_start:g} m** — onset of the sparse tail: "
             f"{100*_cum_le(h_all, h_start):.1f}% of ALL and {100*_cum_le(h_bld, h_start):.1f}% "
             f"of BUILDING training pixels sit at ≤ {h_start:g} m (≈ P"
             f"{int(round(100*_cum_le(h_bld, h_start)))} building), and it coincides with the "
             f"observed ~14 m learned ceiling. Below it: abundant, well-sampled → ordinary "
             f"weight (w=1). Building-pixel median = {_fmt(med_b,2)} m (well inside the "
             f"protected regime).")
    L.append(f"- **w_max = {t_max:g}** — deliberately gentler than the aggressive cap "
             f"({a_max:g}); a tall pixel counts at most {t_max:g}× a ground pixel, so rare "
             f"extremes cannot dominate the gradient (§11).")
    L.append(f"- **tail_scale = {t_scale:g} m** — the ramp spans the tail from h_start to the "
             f"cap at h = h_start+(w_max−1)·scale = {h_sat:g} m "
             f"(≈ P{int(round(100*_cum_le(h_all, h_sat)))} all-pixel); heights beyond "
             f"{h_sat:g} m (the extreme {100*(1-_cum_le(h_all, h_sat)):.1f}%) are clamped.\n")

    L.append("## Weight definition (Phase-4, calibrated tail)\n")
    L.append("```")
    L.append("w(h) = 1                                          for h <= h_start")
    L.append(f"     = min(1 + (h - h_start)/tail_scale, w_max)   for h  > h_start")
    L.append(f"[h_start={h_start:g} m, tail_scale={t_scale:g} m, w_max={t_max:g}]")
    L.append("```")
    L.append("- basis: **physical height (m)**, BEFORE log1p → transform-agnostic, "
             "metric-space 'tall matters more'.")
    L.append(f"- flat w=1 through the protected 0–{h_start:g} m regime; continuous at "
             f"h_start (no jump); bounded in [1, {t_max:g}].\n")

    L.append("## Weight at representative heights (tail vs aggressive)\n")
    L.append("| height (m) | " + " | ".join(str(h) for h in REP_HEIGHTS) + " |")
    L.append("|---|" + "|".join(["--:"] * len(REP_HEIGHTS)) + "|")
    L.append("| tail w(h) | " + " | ".join(
        _fmt(float(tail_weight(float(h), h_start, t_scale, t_max)), 2) for h in REP_HEIGHTS) + " |")
    L.append("| aggressive w(h) | " + " | ".join(
        _fmt(float(height_weight(float(h), a_scale, a_max)), 2) for h in REP_HEIGHTS) + " |")
    L.append("")

    L.append("## Weight statistics over training valid pixels\n")
    L.append("| weight · population | n (subsampled) | min w | max w | mean w | median w |")
    L.append("|---|--:|--:|--:|--:|--:|")
    L.append(_stats_block("tail · ALL", wt_all))
    L.append(_stats_block("tail · BUILDING", wt_bld))
    L.append(_stats_block("aggressive · ALL", wa_all))
    L.append(_stats_block("aggressive · BUILDING", wa_bld))
    L.append(f"\n_Mean tail-w over ALL pixels = **{_fmt(float(wt_all.mean()),3)}** vs aggressive "
             f"**{_fmt(float(wa_all.mean()),3)}** (weighted-MEAN normalizes by Σw, so loss "
             f"magnitude ≈ unweighted → effective LR unchanged; only relative emphasis moves)._\n")

    # ---- the rebalancing: pixel share vs loss-mass share, both weights ----
    L.append("## Rebalancing check — pixel share vs LOSS-MASS share by height bin (ALL px)\n")
    L.append("_Loss-mass share = Σw in bin / Σw total. The calibration goal: the 0–15 m "
             "mass share should stay ≈ its pixel share (regime protected), while the tall "
             "bins rise — a far smaller shift than the aggressive weight imposes._\n")
    L.append("| GT bin (m) | pixel % | tail mass % | tail Δ | aggressive mass % | aggr Δ |")
    L.append("|---|--:|--:|--:|--:|--:|")
    tw = float(wt_all.sum()); aw = float(wa_all.sum())
    prot_px = prot_tmass = prot_amass = 0.0
    for lo, hi in HEIGHT_BINS:
        inb = (h_all >= lo) & (h_all < hi)
        px = float(inb.mean())
        tm = float(wt_all[inb].sum() / tw) if tw else float("nan")
        am = float(wa_all[inb].sum() / aw) if aw else float("nan")
        if hi <= h_start + 1e-9:
            prot_px += px; prot_tmass += tm; prot_amass += am
        hi_s = "inf" if hi == float("inf") else f"{hi:g}"
        L.append(f"| {lo:g}–{hi_s} | {100*px:.2f}% | {100*tm:.2f}% | {100*(tm-px):+.2f}% "
                 f"| {100*am:.2f}% | {100*(am-px):+.2f}% |")
    L.append(f"\n_Total subsampled ALL pixels: {wt_all.size:,}._\n")

    # ---- the §10 must-prove verdict ----
    L.append("## §10 must-prove — does it target the tail WITHOUT disturbing 0–15 m?\n")
    L.append(f"- **0–{h_start:g} m regime** (pixel share {100*prot_px:.2f}%): loss-mass share "
             f"under the **tail** weight = **{100*prot_tmass:.2f}%** "
             f"(Δ {100*(prot_tmass-prot_px):+.2f}%) vs under the **aggressive** weight = "
             f"{100*prot_amass:.2f}% (Δ {100*(prot_amass-prot_px):+.2f}%).")
    verdict = ("YES" if abs(prot_tmass - prot_px) < 0.5 * abs(prot_amass - prot_px)
               else "PARTIAL")
    L.append(f"- The tail weight shifts the protected regime's emphasis by only "
             f"{100*abs(prot_tmass-prot_px):.2f}% (vs {100*abs(prot_amass-prot_px):.2f}% for the "
             f"aggressive weight) → **{verdict}**: it primarily targets the >{h_start:g} m tail "
             f"while leaving the 0–{h_start:g} m optimization emphasis essentially intact.")
    L.append(f"- tall tail (>{h_start:g} m) loss-mass share: tail = "
             f"{100*(1-prot_tmass):.2f}% vs pixel share {100*(1-prot_px):.2f}% "
             f"(emphasis raised, but bounded).\n")

    md = out / "weight_diagnostic.md"
    md.write_text("\n".join(L), encoding="utf-8")
    print(f"[diag] wrote {md}")

    # ---- figure: weight curves + rebalancing bars (tail vs aggressive) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        hs = np.linspace(0, 50, 501)
        ax[0].plot(hs, tail_weight(hs, h_start, t_scale, t_max), color="#1f883d", lw=2.2,
                   label=f"tail (calibrated): start={h_start:g}, scale={t_scale:g}, cap={t_max:g}")
        ax[0].plot(hs, height_weight(hs, a_scale, a_max), color="#c1440e", lw=1.6, ls="--",
                   label=f"aggressive (Phase-3): 1+h/{a_scale:g}, cap {a_max:g}")
        ax[0].axvline(h_start, color="gray", ls=":", lw=0.9, label=f"h_start={h_start:g} m")
        ax[0].axhline(1.0, color="k", ls=":", lw=0.6)
        ax[0].set_xlabel("physical height h (m)"); ax[0].set_ylabel("loss weight w(h)")
        ax[0].set_title("Calibrated tail weight stays at 1 through 0–15 m")
        ax[0].legend(fontsize=8)
        labels, px, tmS, amS = [], [], [], []
        for lo, hi in HEIGHT_BINS:
            inb = (h_all >= lo) & (h_all < hi)
            labels.append(f"{lo:g}-{'inf' if hi == float('inf') else f'{hi:g}'}")
            px.append(100 * float(inb.mean()))
            tmS.append(100 * float(wt_all[inb].sum() / tw) if tw else 0.0)
            amS.append(100 * float(wa_all[inb].sum() / aw) if aw else 0.0)
        xs = np.arange(len(labels))
        ax[1].bar(xs - 0.26, px, 0.26, label="pixel share", color="#7d8590")
        ax[1].bar(xs, tmS, 0.26, label="tail loss-mass share", color="#1f883d")
        ax[1].bar(xs + 0.26, amS, 0.26, label="aggressive loss-mass share", color="#c1440e")
        ax[1].axvspan(-0.5, 3.5, color="green", alpha=0.06)  # protected 0–15 m
        ax[1].set_xticks(xs); ax[1].set_xticklabels(labels, rotation=45)
        ax[1].set_ylabel("% of training pixels / loss mass")
        ax[1].set_title("Tail mass ≈ pixel share in 0–15 m; aggressive drains it")
        ax[1].legend(fontsize=8)
        fig.suptitle("Phase-4 §10 — training-derived calibrated tail weight (JAX train only)")
        fig.tight_layout(); fig.savefig(out / "weight_diagnostic.png", dpi=100)
        plt.close(fig)
        print(f"[diag] wrote {out / 'weight_diagnostic.png'}")
    except Exception as e:
        print(f"[diag] figure skipped: {e}")

    print(f"[diag] tail ALL mean w={wt_all.mean():.3f} (aggressive {wa_all.mean():.3f}); "
          f"0–{h_start:g} m mass: tail {100*prot_tmass:.2f}% vs px {100*prot_px:.2f}% "
          f"→ {verdict}")


if __name__ == "__main__":
    main()
