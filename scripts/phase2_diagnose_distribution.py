#!/usr/bin/env python
"""Phase-2 DIAGNOSIS (read-only): target-height distribution + composition.

This does NOT touch the model. It answers section 9 of the MODIFY prompt:
is the training signal dominated by low heights, and how much tall-structure
mass sits ABOVE the observed ~14 m prediction ceiling?

It reuses the EXACT same record resolution + city split as run_phase1 (so the
JAX train / JAX val / OMA test groups match the experiment), loads only GT +
CLS (no depth, no torch -> fast), and reports per-group / per-class:
  min, max, mean, median, std, percentiles, and P(height > {5,10,15,20,30,40} m)
for all pixels and for building pixels, JAX vs OMA. Writes a markdown table and
a histogram/CDF figure to runs/phase2_diag/.

Usage:  python scripts/phase2_diagnose_distribution.py --config configs/phase1_hf.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.config import load_config
from depthwizard.data import fetch, datasets

THRESHOLDS = [5, 10, 15, 20, 30, 40]
PCTS = [50, 75, 90, 95, 99, 99.9]
CEILING = 14.0  # the observed Baseline-C prediction ceiling we are diagnosing
STRIDE = 8      # subsample stride for percentile/histogram arrays (memory)


def _accumulate(records, tile_size, nodata, building_label):
    """Stream tiles; return dict with exact threshold counts + subsampled values.

    Exact: total pixel count, per-threshold exceed counts (all + building).
    Subsampled (stride): value arrays for percentiles/histograms.
    """
    n_all = n_bld = 0
    over_all = {t: 0 for t in THRESHOLDS}
    over_bld = {t: 0 for t in THRESHOLDS}
    over_ceil_all = over_ceil_bld = 0
    vals_all, vals_bld = [], []
    for rec in records:
        s = datasets.load_sample(rec, tile_size, nodata, depth_model=None)
        gt = np.asarray(s["gt"], dtype=np.float32)
        cls = s.get("cls")
        valid = np.isfinite(gt)
        g = gt[valid]
        n_all += g.size
        for t in THRESHOLDS:
            over_all[t] += int((g > t).sum())
        over_ceil_all += int((g > CEILING).sum())
        vals_all.append(g[::STRIDE].copy())
        if cls is not None:
            bmask = (np.asarray(cls) == building_label) & valid
            gb = gt[bmask]
            n_bld += gb.size
            for t in THRESHOLDS:
                over_bld[t] += int((gb > t).sum())
            over_ceil_bld += int((gb > CEILING).sum())
            vals_bld.append(gb[::STRIDE].copy())
    va = np.concatenate(vals_all) if vals_all else np.array([], np.float32)
    vb = np.concatenate(vals_bld) if vals_bld else np.array([], np.float32)
    return {
        "n_all": n_all, "n_bld": n_bld,
        "over_all": over_all, "over_bld": over_bld,
        "over_ceil_all": over_ceil_all, "over_ceil_bld": over_ceil_bld,
        "vals_all": va, "vals_bld": vb,
        "ground_frac": float((va <= 0.5).mean()) if va.size else float("nan"),
    }


def _stats(v):
    if v.size == 0:
        return {k: float("nan") for k in ["min", "max", "mean", "median", "std"]} | \
               {f"p{p}": float("nan") for p in PCTS}
    d = {"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()),
         "median": float(np.median(v)), "std": float(v.std())}
    for p in PCTS:
        d[f"p{p}"] = float(np.percentile(v, p))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1_hf.yaml")
    ap.add_argument("--out", default="runs/phase2_diag")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    source, records = fetch.resolve_records(cfg)
    tr, va, te = datasets.split_by_city(
        records, cfg.split.train_cities, cfg.split.val_cities,
        cfg.split.test_cities, cfg.split.val_fraction_within_train_city,
        cfg.split.seed, cfg.data.max_tiles_per_city)
    print(f"[diag] source={source} split: train={len(tr)} val={len(va)} test={len(te)}")

    ts, nd, bl = cfg.data.tile_size, cfg.data.nodata, cfg.data.building_label
    groups = {
        "JAX_train": _accumulate(tr, ts, nd, bl),
        "JAX_val": _accumulate(va, ts, nd, bl),
        "OMA_test": _accumulate(te, ts, nd, bl),
    }

    # ---- markdown report ----
    L = ["# Phase-2 Diagnosis — target-height distribution & composition\n",
         f"_Read-only analysis (no model). source=`{source}`, tile_size={ts}, "
         f"building_label={bl}. Groups match run_phase1's city split._\n",
         f"Observed Baseline-C prediction ceiling under diagnosis: **~{CEILING:.0f} m**.\n",
         "## Pixel composition\n",
         "| group | tiles | total px | ground(≤0.5m) frac | building px | building frac |",
         "|---|--:|--:|--:|--:|--:|"]
    counts = {"JAX_train": len(tr), "JAX_val": len(va), "OMA_test": len(te)}
    for name, G in groups.items():
        bf = G["n_bld"] / G["n_all"] if G["n_all"] else float("nan")
        L.append(f"| {name} | {counts[name]} | {G['n_all']:,} | {G['ground_frac']:.3f} "
                 f"| {G['n_bld']:,} | {bf:.3f} |")

    for cls_name, key_n, key_over, key_ceil, key_vals in [
            ("ALL pixels", "n_all", "over_all", "over_ceil_all", "vals_all"),
            ("BUILDING pixels", "n_bld", "over_bld", "over_ceil_bld", "vals_bld")]:
        L.append(f"\n## {cls_name}: height statistics (m)\n")
        L.append("| group | mean | median | std | max | " +
                 " | ".join(f"p{p}" for p in PCTS) + " |")
        L.append("|---|--:|--:|--:|--:|" + "|".join(["--:"] * len(PCTS)) + "|")
        for name, G in groups.items():
            st = _stats(G[key_vals])
            L.append(f"| {name} | {st['mean']:.2f} | {st['median']:.2f} | {st['std']:.2f} "
                     f"| {st['max']:.1f} | " +
                     " | ".join(f"{st[f'p{p}']:.1f}" for p in PCTS) + " |")
        L.append(f"\n### {cls_name}: P(height > threshold)\n")
        L.append("| group | " + " | ".join(f">{t}m" for t in THRESHOLDS) +
                 f" | >{CEILING:.0f}m (ceiling) |")
        L.append("|---|" + "|".join(["--:"] * (len(THRESHOLDS) + 1)) + "|")
        for name, G in groups.items():
            n = G[key_n] or 1
            row = " | ".join(f"{100*G[key_over][t]/n:.2f}%" for t in THRESHOLDS)
            ceil_pct = 100 * G[key_ceil] / n
            L.append(f"| {name} | {row} | {ceil_pct:.2f}% |")

    # key one-liners for the diagnosis
    jt = groups["JAX_train"]
    frac_bld_over_ceil = 100 * jt["over_ceil_bld"] / (jt["n_bld"] or 1)
    L.append("\n## Diagnosis-relevant summary\n")
    L.append(f"- JAX-train **building** pixels above the ~{CEILING:.0f} m ceiling: "
             f"**{frac_bld_over_ceil:.1f}%** — height mass a 14 m-saturating model cannot express.")
    L.append(f"- JAX-train building median height: {_stats(jt['vals_bld'])['median']:.1f} m, "
             f"p99: {_stats(jt['vals_bld'])['p99']:.1f} m, max: {_stats(jt['vals_bld'])['max']:.1f} m.")
    (out / "target_distribution.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[diag] wrote {out/'target_distribution.md'}")

    # ---- figure: histograms + CDF ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        bins = np.linspace(0, 50, 101)
        for name, G in groups.items():
            if G["vals_all"].size:
                ax[0].hist(G["vals_all"], bins=bins, histtype="step", density=True, label=name)
            if G["vals_bld"].size:
                ax[1].hist(G["vals_bld"], bins=bins, histtype="step", density=True, label=name)
        for a, ttl in [(ax[0], "ALL pixels"), (ax[1], "BUILDING pixels")]:
            a.axvline(CEILING, color="k", ls="--", lw=1, label=f"~{CEILING:.0f}m ceiling")
            a.set_yscale("log"); a.set_xlabel("nDSM height (m)"); a.set_ylabel("density (log)")
            a.set_title(ttl); a.legend(fontsize=8)
        # CDF of building pixels
        for name, G in groups.items():
            v = np.sort(G["vals_bld"])
            if v.size:
                ax[2].plot(v, np.linspace(0, 1, v.size), label=name)
        ax[2].axvline(CEILING, color="k", ls="--", lw=1, label=f"~{CEILING:.0f}m ceiling")
        ax[2].set_xlim(0, 50); ax[2].set_xlabel("building nDSM height (m)")
        ax[2].set_ylabel("CDF"); ax[2].set_title("BUILDING height CDF"); ax[2].legend(fontsize=8)
        fig.suptitle("Phase-2 diagnosis: nDSM target-height distribution (JAX train/val vs OMA test)")
        fig.tight_layout(); fig.savefig(out / "target_distribution.png", dpi=100); plt.close(fig)
        print(f"[diag] wrote {out/'target_distribution.png'}")
    except Exception as e:
        print(f"[diag] figure skipped: {e}")


if __name__ == "__main__":
    main()
