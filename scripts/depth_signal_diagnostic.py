#!/usr/bin/env python
"""DepthWizard INPUT-SIGNAL DIAGNOSTIC orchestrator -- DIAGNOSTIC ONLY (no training).

The single question (master prompt §2):
  > Does the frozen Depth Anything V2 relative-depth output contain enough useful
  > information about TALL building height for a SIMPLE mapping to recover that height?

It distinguishes (A) the information exists but the training setup fails to use it, from
(B) the information is not sufficiently present in the DA-V2 representation itself.

What it does (and ONLY this):
  * REUSES the existing frozen DA-V2 depth cache (runs/phase1_hf/depth_cache) -- NO recompute,
    NO torch needed if the cache is complete (§5).
  * Pairs (DA-V2 relative depth, GT nDSM height) per valid pixel.
  * Correlations (Pearson+Spearman) for all/building x JAX/OMA, within height bins, and
    tall-only (>15/>20/>30 m building px) (§6-§8).
  * Fits three SIMPLE mappings h~f(depth) on JAX-TRAIN ONLY -- affine, degree-2 poly,
    isotonic (best monotone) -- and evaluates FROZEN on held-out JAX + OMA (§10-§11).
  * Per-image ORACLE affine upper bound (peeks at each image's GT; diagnostic only) vs the
    global mapping, on buildings and the tall tail (§12/§21).
  * Tall-ordering (cross-bin ordering AUC, mean-depth monotonicity) (§13) and tall-bin depth
    distribution / overlap (§14/§20).
  * Cross-city representation shift JAX vs OMA (§15).
  * Figures + report + results.json (§18-§20/§24), CASE A/B/C/D classification (§25).

It does NOT train a model, touch the fusion head / loss / transform / split / DA-V2 weights,
or build any product (§3/§22/§26). It STOPS at the §27 checkpoint for human review.

Usage:
  python scripts/depth_signal_diagnostic.py --config configs/depth_signal_diagnostic.yaml
  python scripts/depth_signal_diagnostic.py --smoke --allow-fake-depth   # plumbing only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Force HF cache-only resolution BEFORE any transformers/hf import to avoid the documented
# HTTP 429 in the mirror loader. The snapshot is already cached from prior phases (§5).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.config import load_config, config_to_dict
from depthwizard.data import fetch, datasets
from depthwizard.models.affine import GlobalAffine, fit_oracle_affine, _fit_affine
from depthwizard.diagnostics.depth_signal import (
    pearson, spearman, apply_affine, fit_poly, apply_poly, fit_isotonic,
    bin_spans, bin_index, order_auc, cohens_d, map_metrics, disjoint_ids,
)

# Diagnostic height bins (master prompt §7/§14/§20) -- distinct from the model's
# DEFAULT_HEIGHT_EDGES; the diagnostic uses the coarser tall-focused edges.
DIAG_EDGES = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0]
TALL_THRESHOLDS = [15.0, 20.0, 30.0]
CEILING = 14.0                # observed Phase-1..5 prediction ceiling (context)
PLOT_CAP = 8000               # building points per domain in scatter figures
ALL_CAP = 20000               # all-valid pixels kept per tile (subsample; unbiased pool)


# ----------------------------------------------------------------- helpers
def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    if x == float("inf"):
        return "inf"
    return f"{x:.{nd}f}"


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _attach_depth(samples, cfg, depth_model, fake):
    ts = (cfg.data.tile_size, cfg.data.tile_size)
    for s in samples:
        if fake is not None:
            s["depth"] = fake.infer_from_gt(s["gt"], ts)
        else:
            s["depth"] = depth_model.infer(s["rgb"], key=s["id"], target_hw=ts)
    return samples


def _materialize(records, cfg, depth_model, fake):
    samples = [datasets.load_sample(r, cfg.data.tile_size, cfg.data.nodata,
                                    depth_model=None) for r in records]
    return _attach_depth(samples, cfg, depth_model, fake)


def _cache_report(records, cfg):
    """How many tile ids already have a cached DA-V2 output (§5). Returns (hits, misses)."""
    from depthwizard.depth.depth_anything import DepthAnythingV2
    dm = DepthAnythingV2(cfg.depth.model_id, cfg.depth.input_size,
                         cache_dir=cfg.depth.cache_dir, use_cache=True)
    hits = sum(1 for r in records if dm._cache_path(r.tile_id).exists())
    return hits, len(records) - hits


# ----------------------------------------------------------------- pixel extraction
def _extract(samples, blabel, nodata, cap, seed):
    """One pass over a domain's tiles -> pooled pixel arrays.

    Keeps ALL building pixels (they are the analysis focus, §6) and subsamples all-valid
    pixels per tile to `cap` (uniform per-tile subsampling is unbiased for the pooled
    correlation and keeps the 'all pixels' plot from being swamped, §18). Also evaluates the
    per-image ORACLE affine (§12) in the same pass so the oracle is genuinely per-image.
    """
    rng = np.random.default_rng(seed)
    d_all, h_all, o_all, ho_all = [], [], [], []
    d_b, h_b, o_b, ho_b = [], [], [], []
    ids = []
    for s in samples:
        ids.append(s["id"])
        d = np.asarray(s["depth"], np.float64)
        h = np.asarray(s["gt"], np.float64)
        valid = np.isfinite(d) & np.isfinite(h) & (h != nodata)
        if not valid.any():
            continue
        cls = s.get("cls")
        bmask = (valid & (np.asarray(cls) == blabel)) if cls is not None \
            else np.zeros_like(valid)
        opred = np.asarray(fit_oracle_affine(s, nodata=nodata), np.float64)  # per-image UB
        if bmask.any():
            d_b.append(d[bmask]); h_b.append(h[bmask])
            o_b.append(opred[bmask]); ho_b.append(h[bmask])
        vidx = np.flatnonzero(valid)
        if vidx.size > cap:
            vidx = rng.choice(vidx, cap, replace=False)
        dr, hr, orr = d.ravel(), h.ravel(), opred.ravel()
        d_all.append(dr[vidx]); h_all.append(hr[vidx])
        o_all.append(orr[vidx]); ho_all.append(hr[vidx])
    cat = lambda L: np.concatenate(L) if L else np.array([], np.float64)
    return {"ids": ids,
            "d_all": cat(d_all), "h_all": cat(h_all),
            "o_all": cat(o_all), "ho_all": cat(ho_all),
            "d_b": cat(d_b), "h_b": cat(h_b),
            "o_b": cat(o_b), "ho_b": cat(ho_b)}


def _merge(a, b):
    """Concatenate two domain dicts into a city-level pool (JAX = train + val)."""
    out = {"ids": list(a["ids"]) + list(b["ids"])}
    for k in ("d_all", "h_all", "o_all", "ho_all", "d_b", "h_b", "o_b", "ho_b"):
        out[k] = np.concatenate([a[k], b[k]]) if (a[k].size or b[k].size) \
            else np.array([], np.float64)
    return out


# ----------------------------------------------------------------- analyses
def _corr_block(dom):
    """Pearson + Spearman for all-valid and building pixels (§7)."""
    return {
        "all": {"n": int(dom["d_all"].size),
                "pearson": pearson(dom["d_all"], dom["h_all"]),
                "spearman": spearman(dom["d_all"], dom["h_all"])},
        "building": {"n": int(dom["d_b"].size),
                     "pearson": pearson(dom["d_b"], dom["h_b"]),
                     "spearman": spearman(dom["d_b"], dom["h_b"])},
    }


def _within_bin_corr(dom, edges):
    """Pearson + Spearman of depth vs height WITHIN each building height bin (§7)."""
    idx = bin_index(dom["h_b"], edges)
    out = []
    for bi, (lo, hi) in enumerate(bin_spans(edges)):
        sel = idx == bi
        dd, hh = dom["d_b"][sel], dom["h_b"][sel]
        out.append({"lo": lo, "hi": hi, "n": int(dd.size),
                    "pearson": pearson(dd, hh), "spearman": spearman(dd, hh)})
    return out


def _tall_only(dom):
    """Building-pixel correlations above 15 / 20 / 30 m (§8)."""
    out = {}
    for t in TALL_THRESHOLDS:
        sel = dom["h_b"] > t
        dd, hh = dom["d_b"][sel], dom["h_b"][sel]
        out[f"gt_{int(t)}"] = {"n": int(dd.size),
                               "pearson": pearson(dd, hh),
                               "spearman": spearman(dd, hh)}
    return out


def _bin_table(dom, ab, edges, seed):
    """Per building height bin: depth distribution + frozen-affine mapping error (§14/§20)."""
    idx = bin_index(dom["h_b"], edges)
    pred = apply_affine(ab, dom["d_b"])
    rows = []
    for bi, (lo, hi) in enumerate(bin_spans(edges)):
        sel = idx == bi
        n = int(sel.sum())
        if n == 0:
            rows.append({"lo": lo, "hi": hi, "n": 0, "depth_mean": None,
                         "depth_median": None, "depth_std": None, "gt_mean": None,
                         "map_mae": None, "map_rmse": None, "map_bias": None})
            continue
        dd, hh, pp = dom["d_b"][sel], dom["h_b"][sel], pred[sel]
        err = pp - hh
        rows.append({"lo": lo, "hi": hi, "n": n,
                     "depth_mean": float(np.mean(dd)),
                     "depth_median": float(np.median(dd)),
                     "depth_std": float(np.std(dd)),
                     "gt_mean": float(np.mean(hh)),
                     "map_mae": float(np.mean(np.abs(err))),
                     "map_rmse": float(np.sqrt(np.mean(err ** 2))),
                     "map_bias": float(np.mean(err))})
    return rows


def _ordering(dom, edges, seed):
    """Tall-ordering property (§13): mean-depth monotonicity + cross-bin ordering AUC.

    order_auc(lower_bin_depths, higher_bin_depths) = P(a higher-bin pixel has larger depth
    than a lower-bin pixel). 0.5 = indistinguishable; 1.0 = perfectly ordered. Adjacent-bin
    AUCs among the TALL bins (lo>=15 m) are the decisive tall-separability numbers.
    """
    idx = bin_index(dom["h_b"], edges)
    spans = bin_spans(edges)
    groups = [dom["d_b"][idx == bi] for bi in range(len(spans))]
    means = [float(np.mean(g)) if g.size else None for g in groups]
    adj, adj_tall = [], []
    for bi in range(len(spans) - 1):
        lo = spans[bi][0]
        if groups[bi].size and groups[bi + 1].size:
            auc, _, _ = order_auc(groups[bi], groups[bi + 1], seed=seed)
            d = cohens_d(groups[bi], groups[bi + 1])
        else:
            auc, d = None, None
        rec = {"low_lo": spans[bi][0], "high_lo": spans[bi + 1][0],
               "auc": auc, "cohens_d": d}
        adj.append(rec)
        if lo >= 15.0:
            adj_tall.append(auc)
    pair_aucs = []
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            if groups[i].size and groups[j].size:
                a, _, _ = order_auc(groups[i], groups[j], seed=seed)
                pair_aucs.append(a)
    # monotonicity fraction of adjacent mean-depth increases (over defined pairs)
    inc, tot = 0, 0
    for bi in range(len(means) - 1):
        if means[bi] is not None and means[bi + 1] is not None:
            tot += 1
            inc += 1 if means[bi + 1] > means[bi] else 0
    tall_aucs = [a for a in adj_tall if a is not None]
    return {"mean_depth_by_bin": means,
            "adjacent": adj,
            "adjacent_tall_auc_mean": float(np.mean(tall_aucs)) if tall_aucs else None,
            "all_pairs_auc_mean": float(np.mean(pair_aucs)) if pair_aucs else None,
            "mono_fraction": (inc / tot) if tot else None,
            "spearman_building": spearman(dom["d_b"], dom["h_b"])}


def _eval_mapping(predfn, dom):
    """Frozen mapping evaluated on a domain: overall / building / tall subsets (§11)."""
    out = {"all": map_metrics(predfn(dom["d_all"]), dom["h_all"]),
           "building": map_metrics(predfn(dom["d_b"]), dom["h_b"])}
    for t in TALL_THRESHOLDS:
        sel = dom["h_b"] > t
        out[f"tall_{int(t)}"] = map_metrics(predfn(dom["d_b"][sel]), dom["h_b"][sel])
    return out


def _eval_oracle(dom):
    """Per-image oracle affine evaluated on a domain (precomputed in _extract, §12/§21)."""
    out = {"all": map_metrics(dom["o_all"], dom["ho_all"]),
           "building": map_metrics(dom["o_b"], dom["ho_b"])}
    for t in TALL_THRESHOLDS:
        sel = dom["ho_b"] > t
        out[f"tall_{int(t)}"] = map_metrics(dom["o_b"][sel], dom["ho_b"][sel])
    return out


# ----------------------------------------------------------------- classification (§25)
def _classify(oma_tall_only, oma_ord, glob_oma, orac_oma):
    """Transparent, threshold-based CASE proposal from the CROSS-CITY (OMA) evidence.

    Thresholds are heuristics stated in the report; the human makes the final call (§27).
    """
    sp15 = oma_tall_only["gt_15"]["spearman"]
    auc_tall = oma_ord["adjacent_tall_auc_mean"]
    g_tall = glob_oma["tall_15"]["mae"]
    o_tall = orac_oma["tall_15"]["mae"]
    sp15 = sp15 if (sp15 is not None and sp15 == sp15) else 0.0
    auc_tall = auc_tall if (auc_tall is not None and auc_tall == auc_tall) else 0.5
    gap = (g_tall - o_tall) if (g_tall is not None and o_tall is not None) else None

    strong_order = (sp15 >= 0.40) and (auc_tall >= 0.70)
    weak_order = (auc_tall < 0.60) or (sp15 < 0.15)
    oracle_recovers = (o_tall is not None and o_tall < 8.0)
    big_oracle_gap = (gap is not None and gap >= 3.0 and g_tall and o_tall < 0.6 * g_tall)

    if strong_order and oracle_recovers:
        case = "A"
        one = ("STRONG tall-height signal: depth preserves tall ordering and a simple/oracle "
               "mapping recovers tall height. Next bottleneck is data/training/calibration, "
               "not the depth representation.")
    elif big_oracle_gap and oracle_recovers and not weak_order:
        case = "B"
        one = ("USEFUL RELATIVE signal, WEAK METRIC SCALE: ordering is meaningful and per-image "
               "oracle recovers tall height, but the GLOBAL mapping is unstable -> scene-dependent "
               "scale is the dominant bottleneck.")
    elif weak_order and not oracle_recovers:
        case = "C"
        one = ("WEAK tall signal: depth correlates at low/moderate heights but loses tall "
               "ordering, and EVEN the per-image oracle cannot recover the tall tail -> the frozen "
               "DA-V2 prior likely lacks sufficient absolute tall-height information.")
    else:
        case = "D"
        one = ("MIXED / INCONCLUSIVE: the tall-height evidence does not fall cleanly into A/B/C. "
               "State exactly what is uncertain; do not force a conclusion.")
    return {"case": case, "one_line": one,
            "signals": {"oma_tall15_spearman": sp15, "oma_tall_adj_auc": auc_tall,
                        "oma_global_tall15_mae": g_tall, "oma_oracle_tall15_mae": o_tall,
                        "oracle_minus_global_tall15_mae_gap": gap,
                        "strong_order": strong_order, "weak_order": weak_order,
                        "oracle_recovers_tall": oracle_recovers,
                        "big_oracle_gap": big_oracle_gap}}


# ----------------------------------------------------------------- figures
def _figures(fig_dir, jt, jv, ot, jax, oma, mappings, edges, evidence_valid):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[viz] skipped ({e})")
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if evidence_valid else "  [NON-EVIDENCE smoke]"
    ab = mappings["affine"]["ab"]
    iso = mappings["isotonic"]["map"]
    rng = np.random.default_rng(0)

    def sub(d, h, k):
        if d.size <= k:
            return d, h
        i = rng.choice(d.size, k, replace=False)
        return d[i], h[i]

    # ---- §18 MASTER figure: building depth vs height, tall highlighted, fitted train map,
    #      eval points (JAX-val, OMA) distinguished. Ground pixels excluded so they can't
    #      hide the building relationship.
    try:
        fig, ax = plt.subplots(figsize=(9, 7))
        dtr, htr = sub(jt["d_b"], jt["h_b"], PLOT_CAP)
        dvl, hvl = sub(jv["d_b"], jv["h_b"], PLOT_CAP)
        dom_, hom = sub(ot["d_b"], ot["h_b"], PLOT_CAP)
        ax.scatter(dtr, htr, s=3, alpha=0.25, c="#9aa0a6", label="JAX-train buildings (fit)")
        ax.scatter(dvl, hvl, s=6, alpha=0.5, c="#1f6feb",
                   label="JAX-val buildings (in-domain eval)")
        ax.scatter(dom_, hom, s=6, alpha=0.5, c="#d1242f",
                   label="OMA buildings (cross-city eval)")
        tsel = jt["h_b"] > 15.0
        dt, ht = sub(jt["d_b"][tsel], jt["h_b"][tsel], PLOT_CAP // 2)
        ax.scatter(dt, ht, s=8, alpha=0.5, c="#8250df",
                   label="JAX-train tall (>15 m) highlighted")
        xs = np.linspace(float(np.min(jt["d_b"])), float(np.max(jt["d_b"])), 200)
        ax.plot(xs, apply_affine(ab, xs), "k-", lw=2,
                label=f"fitted affine h={ab[0]:.2f}d{ab[1]:+.2f}")
        ax.plot(iso.x, iso.y, "g--", lw=2, label="fitted isotonic (best monotone)")
        ax.axhline(15, color="orange", lw=0.8, ls=":"); ax.axhline(CEILING, color="gray", lw=0.8, ls=":")
        ax.set_xlabel("Depth Anything V2 relative depth"); ax.set_ylabel("GT nDSM height (m)")
        ax.set_title(f"MASTER: DA-V2 depth vs building height -- fit on JAX-train, "
                     f"eval JAX-val/OMA{tag}")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout(); fig.savefig(fig_dir / "master_depth_vs_height.png", dpi=110)
        plt.close(fig)
    except Exception as e:
        print(f"[viz] master figure skipped: {e}")

    # ---- §9 scatter grid per domain: all / building / >15 / >20 / >30
    def grid(dom, name):
        try:
            fig, ax = plt.subplots(1, 5, figsize=(22, 4.2))
            da, ha = sub(dom["d_all"], dom["h_all"], 40000)
            ax[0].scatter(da, ha, s=2, alpha=0.2, c="#6e7781"); ax[0].set_title("all valid")
            db, hb = sub(dom["d_b"], dom["h_b"], 40000)
            ax[1].scatter(db, hb, s=2, alpha=0.3, c="#1f6feb"); ax[1].set_title("buildings")
            for k, t in enumerate(TALL_THRESHOLDS):
                sel = dom["h_b"] > t
                dd, hh = sub(dom["d_b"][sel], dom["h_b"][sel], 40000)
                ax[k + 2].scatter(dd, hh, s=3, alpha=0.4, c="#8250df")
                ax[k + 2].set_title(f"buildings > {int(t)} m")
            xs = np.linspace(float(np.min(dom["d_all"])), float(np.max(dom["d_all"])), 100)
            for a in ax:
                a.plot(xs, apply_affine(ab, xs), "k-", lw=1)
                a.set_xlabel("depth"); a.set_ylabel("GT height (m)")
            fig.suptitle(f"{name}: DA-V2 depth vs GT height (black = JAX-train affine){tag}")
            fig.tight_layout(); fig.savefig(fig_dir / f"scatter_{name}.png", dpi=100)
            plt.close(fig)
        except Exception as e:
            print(f"[viz] scatter {name} skipped: {e}")

    grid(jt, "jax_train"); grid(jv, "jax_val"); grid(ot, "oma_test")

    # ---- §14 tall-bin depth distribution: mean +/- std depth per height bin, JAX vs OMA
    try:
        spans = bin_spans(edges)
        labels = [f"{int(lo)}-{'inf' if hi == float('inf') else int(hi)}" for lo, hi in spans]
        fig, ax = plt.subplots(figsize=(10, 5))
        for dom, name, col in [(jax, "JAX", "#1f6feb"), (oma, "OMA", "#d1242f")]:
            idx = bin_index(dom["h_b"], edges)
            m = [float(np.mean(dom["d_b"][idx == bi])) if (idx == bi).any() else np.nan
                 for bi in range(len(spans))]
            sd = [float(np.std(dom["d_b"][idx == bi])) if (idx == bi).any() else np.nan
                  for bi in range(len(spans))]
            ax.errorbar(range(len(spans)), m, yerr=sd, marker="o", capsize=4,
                        color=col, label=name)
        ax.set_xticks(range(len(spans))); ax.set_xticklabels(labels, rotation=45)
        ax.set_xlabel("GT height bin (m)"); ax.set_ylabel("DA-V2 depth (mean +/- std)")
        ax.set_title(f"Tall-bin depth distribution -- do tall bins SEPARATE in depth?{tag}")
        ax.legend(); fig.tight_layout()
        fig.savefig(fig_dir / "tall_bin_depth_distribution.png", dpi=110); plt.close(fig)
    except Exception as e:
        print(f"[viz] bin distribution skipped: {e}")

    # ---- tall MAE: global affine vs per-image oracle, JAX-val vs OMA
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        xs = [f">{int(t)}" for t in TALL_THRESHOLDS]
        series = {
            "global affine / JAX-val": (mappings["affine"]["eval"]["jax_val"], "#1f6feb", "-"),
            "global affine / OMA": (mappings["affine"]["eval"]["oma_test"], "#1f6feb", "--"),
        }
        for lbl, (ev, col, ls) in series.items():
            ax.plot(xs, [ev[f"tall_{int(t)}"]["mae"] for t in TALL_THRESHOLDS],
                    marker="o", color=col, ls=ls, label=lbl)
        ax.plot(xs, [ORACLE_JV[f"tall_{int(t)}"]["mae"] for t in TALL_THRESHOLDS],
                marker="s", color="#1a7f37", ls="-", label="oracle / JAX-val")
        ax.plot(xs, [ORACLE_OT[f"tall_{int(t)}"]["mae"] for t in TALL_THRESHOLDS],
                marker="s", color="#1a7f37", ls="--", label="oracle / OMA")
        ax.set_xlabel("tall building threshold"); ax.set_ylabel("MAE (m)")
        ax.set_title(f"Tall-tail MAE: simple global mapping vs per-image oracle{tag}")
        ax.legend(); fig.tight_layout()
        fig.savefig(fig_dir / "tall_mae_global_vs_oracle.png", dpi=110); plt.close(fig)
    except Exception as e:
        print(f"[viz] tall MAE figure skipped: {e}")
    print(f"[viz] figures written to {fig_dir}")


# ORACLE eval dicts are needed inside _figures; set as module globals just before the call.
ORACLE_JV = {}
ORACLE_OT = {}


# ----------------------------------------------------------------- report
def _mtable_row(name, ev):
    a, b = ev["all"], ev["building"]
    return (f"| {name} | {_fmt(a['mae'],2)} | {_fmt(a['rmse'],2)} | {_fmt(a['pearson'])} "
            f"| {_fmt(a['spearman'])} | {_fmt(b['mae'],2)} | {_fmt(b['rmse'],2)} "
            f"| {_fmt(ev['tall_15']['mae'],2)} | {_fmt(ev['tall_20']['mae'],2)} "
            f"| {_fmt(ev['tall_30']['mae'],2)} |")


def _write_report(results, path):
    m = results["meta"]
    c = results["correlations"]
    cls = results["classification"]
    L = []
    L.append("# DepthWizard -- INPUT-SIGNAL DIAGNOSTIC")
    L.append("## Does the frozen Depth Anything V2 relative-depth output contain usable "
             "TALL-height information?\n")
    L.append("_Generated by `scripts/depth_signal_diagnostic.py`. DIAGNOSTIC ONLY: no model "
             "was trained; the fusion head, loss, target transform, dataset split and DA-V2 "
             "weights are untouched (master prompt §3/§22). Every number below is measured by "
             "this run from the REUSED frozen DA-V2 depth cache; missing values read `n/a`._\n")
    if not m["evidence_valid"]:
        L.append("> # WARNING -- NOT VALID EVIDENCE (synthetic/fake-depth smoke). "
                 "Numbers only prove the code executes.\n")
    L.append("### Run metadata (§24)\n")
    L.append(f"- dataset source: `{m['source']}` | evidence valid: **{m['evidence_valid']}**")
    L.append(f"- depth prior: `{m['depth_model']}` | cache: `{m['depth_cache']}` "
             f"(REUSED from Phase-1; hits={m['cache_hits']} misses={m['cache_misses']}, §5)")
    L.append(f"- split (city-held-out): fit on **JAX-train** ({m['n_train']} tiles); eval on "
             f"held-out **JAX-val** ({m['n_val']}) + **OMA-test** ({m['n_test']}) (§4/§16)")
    L.append(f"- no train/eval leakage (disjoint tile ids): "
             f"JAX-train vs JAX-val **{m['leakage_ok_val']}**, vs OMA **{m['leakage_ok_oma']}**")
    L.append(f"- diagnostic height bins (m): {DIAG_EDGES} + [40, inf) | tall thresholds: "
             f"{[int(t) for t in TALL_THRESHOLDS]} m | seed {m['seed']} | runtime {m['elapsed_s']}s\n")
    if m["source"] == "hf_blocks":
        L.append("> **Provenance caveat:** unofficial HF mirror `JasonXF/DFC2019-10k`, "
                 "preprocessed nDSM (ground floored to 0). Feasibility evidence only; re-confirm "
                 "on official IEEE GRSS DFC2019 before external reporting.\n")

    L.append("---\n## 1. Correlation: DA-V2 depth vs GT height (§6/§7)\n")
    L.append("| Pixels | JAX Pearson | JAX Spearman | OMA Pearson | OMA Spearman | JAX n | OMA n |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    for grp, lbl in [("all", "all valid"), ("building", "buildings")]:
        j, o = c["JAX"][grp], c["OMA"][grp]
        L.append(f"| {lbl} | {_fmt(j['pearson'])} | {_fmt(j['spearman'])} | "
                 f"{_fmt(o['pearson'])} | {_fmt(o['spearman'])} | {j['n']:,} | {o['n']:,} |")
    L.append("\n_All-pixel correlation is dominated by the ground/building contrast and is "
             "expected to look strong; the BUILDING and TALL-ONLY rows are the discriminators._\n")

    L.append("### Within-bin building correlation (§7) -- does depth track height INSIDE a bin?\n")
    L.append("| GT bin (m) | JAX n | JAX Pearson | JAX Spearman | OMA n | OMA Pearson | OMA Spearman |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    jb = {(r["lo"], r["hi"]): r for r in c["JAX_within_bin"]}
    for r in c["OMA_within_bin"]:
        j = jb.get((r["lo"], r["hi"]), {})
        hi = "inf" if r["hi"] == float("inf") else int(r["hi"])
        L.append(f"| {int(r['lo'])}-{hi} | {j.get('n',0):,} | {_fmt(j.get('pearson'))} "
                 f"| {_fmt(j.get('spearman'))} | {r['n']:,} | {_fmt(r['pearson'])} "
                 f"| {_fmt(r['spearman'])} |")

    L.append("\n---\n## 2. Tall-only building correlation (§8)\n")
    L.append("| Threshold | JAX n | JAX Pearson | JAX Spearman | OMA n | OMA Pearson | OMA Spearman |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    for t in TALL_THRESHOLDS:
        k = f"gt_{int(t)}"
        j, o = results["tall_only"]["JAX"][k], results["tall_only"]["OMA"][k]
        L.append(f"| > {int(t)} m | {j['n']:,} | {_fmt(j['pearson'])} | {_fmt(j['spearman'])} "
                 f"| {o['n']:,} | {_fmt(o['pearson'])} | {_fmt(o['spearman'])} |")
    L.append("\n_This is the crux: does DA-V2 preserve useful ordering AMONG genuinely tall "
             "structures, cross-city (OMA)?_\n")

    L.append("---\n## 3. Tall ordering property (§13)\n")
    for name in ("JAX", "OMA"):
        o = results["ordering"][name]
        L.append(f"- **{name}**: building Spearman={_fmt(o['spearman_building'])}, "
                 f"mean-depth monotonic across bins={_fmt(o['mono_fraction'],2)} "
                 f"(fraction of adjacent bins that increase), adjacent-bin ordering AUC among "
                 f"TALL bins (>=15 m)={_fmt(o['adjacent_tall_auc_mean'])}, all-pairs mean AUC="
                 f"{_fmt(o['all_pairs_auc_mean'])}")
    L.append("\n_Ordering AUC = P(a higher-bin building pixel has larger depth than a lower-bin "
             "one). 0.5 = indistinguishable, 1.0 = perfectly ordered._\n")

    L.append("---\n## 4. §19 required table -- correlations + affine/tall mapping (JAX vs OMA)\n")
    L.append("| Diagnostic | JAX In-domain | OMA Cross-city |")
    L.append("|---|--:|--:|")
    aff = results["mappings"]["affine"]["eval"]
    L.append(f"| Pearson -- all pixels | {_fmt(c['JAX']['all']['pearson'])} "
             f"| {_fmt(c['OMA']['all']['pearson'])} |")
    L.append(f"| Spearman -- all pixels | {_fmt(c['JAX']['all']['spearman'])} "
             f"| {_fmt(c['OMA']['all']['spearman'])} |")
    L.append(f"| Pearson -- buildings | {_fmt(c['JAX']['building']['pearson'])} "
             f"| {_fmt(c['OMA']['building']['pearson'])} |")
    L.append(f"| Spearman -- buildings | {_fmt(c['JAX']['building']['spearman'])} "
             f"| {_fmt(c['OMA']['building']['spearman'])} |")
    L.append(f"| Building MAE -- affine mapping | {_fmt(aff['jax_val']['building']['mae'],2)} "
             f"| {_fmt(aff['oma_test']['building']['mae'],2)} |")
    L.append(f"| Building RMSE -- affine mapping | {_fmt(aff['jax_val']['building']['rmse'],2)} "
             f"| {_fmt(aff['oma_test']['building']['rmse'],2)} |")
    for t in TALL_THRESHOLDS:
        L.append(f"| > {int(t)} m MAE (affine) | {_fmt(aff['jax_val'][f'tall_{int(t)}']['mae'],2)} "
                 f"| {_fmt(aff['oma_test'][f'tall_{int(t)}']['mae'],2)} |")

    L.append("\n### Simple mappings fit on JAX-train, evaluated FROZEN (§10/§11)\n")
    L.append("_all-pixel MAE/RMSE/Pearson/Spearman, then building MAE/RMSE, then tall MAE._\n")
    for domlbl, domkey in [("JAX-val (in-domain)", "jax_val"), ("OMA (cross-city)", "oma_test")]:
        L.append(f"\n**{domlbl}**\n")
        L.append("| Mapping | all MAE | all RMSE | all Pears | all Spear | bldg MAE | bldg RMSE "
                 "| >15 MAE | >20 MAE | >30 MAE |")
        L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
        for mk, mlbl in [("affine", "affine h=a·d+b"), ("poly2", "poly (deg 2)"),
                         ("isotonic", "isotonic (monotone)")]:
            L.append(_mtable_row(mlbl, results["mappings"][mk]["eval"][domkey]))
        L.append(_mtable_row("**oracle (per-image, UB)**", results["oracle"][domkey]))

    L.append("\n---\n## 5. §20 required height-bin table (buildings; affine mapping error)\n")
    for domlbl, rows in [("JAX-train (training-derived)", results["bin_tables"]["jax_train"]),
                         ("JAX-val (in-domain eval)", results["bin_tables"]["jax_val"]),
                         ("OMA (cross-city eval)", results["bin_tables"]["oma_test"])]:
        L.append(f"\n**{domlbl}**\n")
        L.append("| Height bin (m) | Pixel count | Mean depth | Median depth | Depth std "
                 "| Mapping MAE | Mapping RMSE |")
        L.append("|---|--:|--:|--:|--:|--:|--:|")
        for r in rows:
            hi = "inf" if r["hi"] == float("inf") else int(r["hi"])
            L.append(f"| {int(r['lo'])}-{hi} | {r['n']:,} | {_fmt(r['depth_mean'])} "
                     f"| {_fmt(r['depth_median'])} | {_fmt(r['depth_std'])} "
                     f"| {_fmt(r['map_mae'],2)} | {_fmt(r['map_rmse'],2)} |")

    L.append("\n---\n## 6. Oracle vs global mapping -- scene-scale ambiguity? (§12/§21)\n")
    L.append("| Metric | global affine (OMA) | per-image oracle (OMA) | improvement |")
    L.append("|---|--:|--:|--:|")
    go, oo = results["mappings"]["affine"]["eval"]["oma_test"], results["oracle"]["oma_test"]
    for label, key, sub in [("building MAE", "building", "mae"), ("building RMSE", "building", "rmse"),
                            (">15 m MAE", "tall_15", "mae"), (">20 m MAE", "tall_20", "mae"),
                            (">30 m MAE", "tall_30", "mae")]:
        g, o = go[key][sub], oo[key][sub]
        imp = (g - o) if (g is not None and o is not None) else None
        L.append(f"| {label} | {_fmt(g,2)} | {_fmt(o,2)} | {_fmt(imp,2)} |")
    L.append("\n_Oracle peeks at each image's own GT (upper bound, NOT deployable, labelled "
             "diagnostic-only, §16). A large oracle improvement = scene-dependent scale ambiguity; "
             "oracle STILL poor = the representation itself lacks the tall-height information._\n")

    L.append("---\n## 7. Cross-city representation shift (§15)\n")
    sh = results["cross_city_shift"]
    L.append(f"- JAX-train affine: h = {_fmt(sh['jax_affine'][0])}·d + {_fmt(sh['jax_affine'][1])}")
    L.append(f"- OMA affine (diagnostic-only, NOT used for production): "
             f"h = {_fmt(sh['oma_affine'][0])}·d + {_fmt(sh['oma_affine'][1])}")
    L.append(f"- same depth -> different height? predicted height at reference depths under each "
             f"fit: {sh['ref_table']}")
    L.append(f"- per-bin mean-depth shift (OMA mean depth - JAX mean depth) by height bin: "
             f"{[ _fmt(x) for x in sh['bin_mean_depth_delta'] ]}\n")

    L.append("---\n## 8. CASE classification (§25)\n")
    L.append(f"### PROPOSED: **CASE {cls['case']}**\n")
    L.append(f"{cls['one_line']}\n")
    L.append("Decisive cross-city signals:\n")
    s = cls["signals"]
    L.append(f"- OMA building >15 m Spearman: **{_fmt(s['oma_tall15_spearman'])}**")
    L.append(f"- OMA tall adjacent-bin ordering AUC (>=15 m): **{_fmt(s['oma_tall_adj_auc'])}**")
    L.append(f"- OMA global affine >15 m MAE: **{_fmt(s['oma_global_tall15_mae'],2)} m** "
             f"vs per-image oracle >15 m MAE: **{_fmt(s['oma_oracle_tall15_mae'],2)} m** "
             f"(oracle gap {_fmt(s['oracle_minus_global_tall15_mae_gap'],2)} m)")
    L.append(f"- strong_order={s['strong_order']} | weak_order={s['weak_order']} | "
             f"oracle_recovers_tall={s['oracle_recovers_tall']} | big_oracle_gap={s['big_oracle_gap']}")
    L.append("\n_Thresholds (heuristic, stated for auditability): strong_order = tall Spearman "
             ">=0.40 AND tall AUC >=0.70; weak_order = tall AUC <0.60 OR tall Spearman <0.15; "
             "oracle_recovers = oracle >15 m MAE <8 m; big_oracle_gap = global-oracle >=3 m and "
             "oracle <60% of global. The human makes the final call (§27)._\n")

    L.append("---\n## 9. Connection to prior experiments (§17)\n")
    L.append("- The ~14 m prediction ceiling survived the target transform (Phase-2), loss "
             "reweighting (Phase-3/4) and capacity/receptive-field (Phase-5). This diagnostic "
             "tests the remaining upstream hypothesis: the INPUT depth signal itself.")
    L.append("- If the signal is strong here -> the next lever is data/training/calibration.")
    L.append("- If tall signal is weak but oracle recovers it -> scene-scale ambiguity (per-image "
             "normalization / metric cue).")
    L.append("- If BOTH global and oracle are weak on the tall tail -> the frozen DA-V2 prior "
             "lacks sufficient absolute tall-height information and a different metric/physical "
             "cue is required.\n")

    L.append("---\n## 10. Limitations\n")
    L.append("- Unofficial DFC2019 mirror, preprocessed nDSM (ground floored to 0); single "
             "training city (JAX), single cross-city test (OMA).")
    L.append("- DA-V2 depth is scale/shift-ambiguous and near-nadir camera-depth variation is "
             "negligible, so it encodes learned appearance priors, not measured geometry.")
    L.append("- Tall pixels are sparse; tall-bin statistics have wider uncertainty. Oracle is an "
             "upper bound that peeks at GT and is not deployable.")
    L.append("- This diagnostic does NOT establish final model / ISRO / DSM / DEM / GeoTIFF / 3D "
             "/ production accuracy (§26).\n")
    L.append("> MANDATORY STOP (§27): diagnostic only. No new training or architecture was "
             "launched. Await human decision on the single next direction.\n")
    Path(path).write_text("\n".join(L), encoding="utf-8")
    print(f"[report] wrote {path}")


def _append_experiment_results(results, path="EXPERIMENT_RESULTS.md"):
    """Append a concise diagnostic section; NEVER overwrite Phases 1-5 (§24)."""
    p = Path(path)
    m, cls = results["meta"], results["classification"]
    c = results["correlations"]
    aff = results["mappings"]["affine"]["eval"]
    oo = results["oracle"]["oma_test"]
    s = cls["signals"]
    block = [
        "\n\n---\n\n## INPUT-SIGNAL DIAGNOSTIC — does DA-V2 contain tall-height signal?\n",
        f"_Diagnostic only (no training); generated by `scripts/depth_signal_diagnostic.py`. "
        f"Reused frozen DA-V2 cache (hits={m['cache_hits']}/{m['cache_hits']+m['cache_misses']}). "
        f"Fit on JAX-train ({m['n_train']} tiles), eval held-out JAX-val ({m['n_val']}) + OMA "
        f"({m['n_test']}). Evidence valid: {m['evidence_valid']}._\n",
        f"- **Buildings, cross-city (OMA):** Pearson {_fmt(c['OMA']['building']['pearson'])}, "
        f"Spearman {_fmt(c['OMA']['building']['spearman'])}. Tall >15 m Spearman "
        f"{_fmt(s['oma_tall15_spearman'])}; tall adjacent-bin ordering AUC "
        f"{_fmt(s['oma_tall_adj_auc'])}.",
        f"- **Simple global affine (OMA):** building MAE {_fmt(aff['oma_test']['building']['mae'],2)} m, "
        f">15 m MAE {_fmt(aff['oma_test']['tall_15']['mae'],2)} m, "
        f">30 m MAE {_fmt(aff['oma_test']['tall_30']['mae'],2)} m.",
        f"- **Per-image oracle (UB, OMA):** building MAE {_fmt(oo['building']['mae'],2)} m, "
        f">15 m MAE {_fmt(oo['tall_15']['mae'],2)} m (oracle-minus-global >15 m gap "
        f"{_fmt(s['oracle_minus_global_tall15_mae_gap'],2)} m).",
        f"- **PROPOSED CASE {cls['case']}** — {cls['one_line']}",
        f"- Full report + figures + tables: `runs/depth_signal_diagnostic/`. STOP for human "
        f"review (§27); no architecture chosen automatically.\n",
    ]
    with p.open("a", encoding="utf-8") as f:
        f.write("\n".join(block))
    print(f"[results] appended diagnostic section to {path}")


# ----------------------------------------------------------------- main
def main():
    global ORACLE_JV, ORACLE_OT
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/depth_signal_diagnostic.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="force synthetic data + tiny run (NOT evidence)")
    ap.add_argument("--allow-fake-depth", action="store_true",
                    help="fabricate depth from GT if the cache/torch is unavailable (NOT evidence)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    cfg = load_config(args.config)
    if args.smoke:
        cfg.data.source = "synthetic"
        cfg.seeds = [0]
    seed = int(cfg.seeds[0]) if cfg.seeds else 0
    out_dir = Path(args.out or cfg.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    source, records = fetch.resolve_records(cfg)
    is_synth = (source == "synthetic")

    # --- depth: reuse cache; only touch torch if a cache entry is genuinely missing (§5) ---
    fake = None
    depth_model = None
    cache_hits = cache_misses = 0
    if is_synth:
        fake_needed = True
    else:
        cache_hits, cache_misses = _cache_report(records, cfg)
        print(f"[cache] DA-V2 depth cache: hits={cache_hits} misses={cache_misses} "
              f"(of {len(records)} tiles)")
        fake_needed = False
        if cache_misses > 0 and not _has_torch():
            if args.allow_fake_depth:
                print("[warn] cache misses + no torch -> using FAKE depth (NOT evidence).")
                fake_needed = True
            else:
                print("[error] missing depth-cache entries and torch/transformers unavailable. "
                      "Cannot reproduce the exact frozen representation. Re-run where the cache "
                      "is complete, or pass --allow-fake-depth for a NON-EVIDENCE smoke.")
                sys.exit(2)
    if fake_needed or (args.allow_fake_depth and is_synth):
        from depthwizard.depth.fake import FakeDepth
        fake = FakeDepth(seed=cfg.split.seed)
    elif not is_synth:
        from depthwizard.depth.depth_anything import DepthAnythingV2
        depth_model = DepthAnythingV2(
            cfg.depth.model_id, cfg.depth.input_size,
            cache_dir=cfg.depth.cache_dir if cfg.depth.use_cache else None,
            use_cache=cfg.depth.use_cache)

    # --- materialize the city-held-out split (same seed/cap as Phase-1..5) ---
    if is_synth:
        train, val, test = fetch.synthetic_samples(cfg)
        for grp in (train, val, test):
            _attach_depth(grp, cfg, depth_model, fake)
        tr_ids = [s["id"] for s in train]; va_ids = [s["id"] for s in val]
        te_ids = [s["id"] for s in test]
    else:
        tr_rec, va_rec, te_rec = datasets.split_by_city(
            records, cfg.split.train_cities, cfg.split.val_cities,
            cfg.split.test_cities, cfg.split.val_fraction_within_train_city,
            cfg.split.seed, cfg.data.max_tiles_per_city)
        print(f"[split] train={len(tr_rec)} val={len(va_rec)} test={len(te_rec)}")
        tr_ids = [r.tile_id for r in tr_rec]; va_ids = [r.tile_id for r in va_rec]
        te_ids = [r.tile_id for r in te_rec]
        train = _materialize(tr_rec, cfg, depth_model, fake)
        val = _materialize(va_rec, cfg, depth_model, fake)
        test = _materialize(te_rec, cfg, depth_model, fake)
    if not train or not test:
        print("[error] empty train/test split."); sys.exit(3)

    leakage_ok_val = disjoint_ids(tr_ids, va_ids)   # §16 hard guard
    leakage_ok_oma = disjoint_ids(tr_ids, te_ids)
    if not (leakage_ok_val and leakage_ok_oma):
        print("[error] train/eval tile-id overlap detected -> would leak. Aborting.")
        sys.exit(5)
    evidence_valid = (not is_synth) and (fake is None)

    # --- extract pooled pixel arrays per domain (+ per-image oracle in the same pass) ---
    blabel, nodata = cfg.data.building_label, cfg.data.nodata
    jt = _extract(train, blabel, nodata, ALL_CAP, seed)   # JAX-train (fit)
    jv = _extract(val, blabel, nodata, ALL_CAP, seed)     # JAX-val (in-domain eval)
    ot = _extract(test, blabel, nodata, ALL_CAP, seed)    # OMA (cross-city eval)
    jax = _merge(jt, jv)                                  # JAX city descriptive pool
    oma = ot
    print(f"[pixels] JAX buildings={jax['d_b'].size:,} OMA buildings={oma['d_b'].size:,}")

    # --- descriptive correlations (no fitting -> no leakage) ---
    correlations = {"JAX": _corr_block(jax), "OMA": _corr_block(oma),
                    "JAX_within_bin": _within_bin_corr(jax, DIAG_EDGES),
                    "OMA_within_bin": _within_bin_corr(oma, DIAG_EDGES)}
    tall_only = {"JAX": _tall_only(jax), "OMA": _tall_only(oma)}
    ordering = {"JAX": _ordering(jax, DIAG_EDGES, seed),
                "OMA": _ordering(oma, DIAG_EDGES, seed)}

    # --- simple mappings fit on JAX-TRAIN ONLY, then frozen (§10) ---
    ab = _fit_affine(jt["d_all"], jt["h_all"], robust=True)
    gb = GlobalAffine(max_pixels=cfg.train.max_train_pixels_affine,
                      seed=cfg.split.seed).fit(train)   # canonical Baseline-B anchor
    print(f"[affine] diagnostic fit a={ab[0]:.4g} b={ab[1]:.4g} | Baseline-B a={gb.a:.4g} b={gb.b:.4g}")
    poly = fit_poly(jt["d_all"], jt["h_all"], degree=2, seed=seed)
    iso = fit_isotonic(jt["d_all"], jt["h_all"], seed=seed)
    predfns = {"affine": lambda d: apply_affine(ab, d),
               "poly2": lambda d: apply_poly(poly, d),
               "isotonic": lambda d: iso.predict(d)}
    mappings = {}
    for mk, fn in predfns.items():
        mappings[mk] = {"eval": {"jax_train": _eval_mapping(fn, jt),
                                 "jax_val": _eval_mapping(fn, jv),
                                 "oma_test": _eval_mapping(fn, ot)}}
    mappings["affine"]["ab"] = [float(ab[0]), float(ab[1])]
    mappings["affine"]["baseline_b"] = [float(gb.a), float(gb.b)]
    mappings["poly2"]["coeffs"] = [float(x) for x in poly]
    mappings["isotonic"]["map"] = iso
    mappings["isotonic"]["grid"] = iso.as_dict()

    # --- per-image oracle (§12/§21) ---
    oracle = {"jax_train": _eval_oracle(jt), "jax_val": _eval_oracle(jv),
              "oma_test": _eval_oracle(ot)}
    ORACLE_JV, ORACLE_OT = oracle["jax_val"], oracle["oma_test"]

    # --- height-bin tables (frozen JAX affine mapping error), separated by domain (§20) ---
    bin_tables = {"jax_train": _bin_table(jt, ab, DIAG_EDGES, seed),
                  "jax_val": _bin_table(jv, ab, DIAG_EDGES, seed),
                  "oma_test": _bin_table(ot, ab, DIAG_EDGES, seed)}

    # --- cross-city shift (§15): JAX affine vs OMA-fit affine (diagnostic-only) ---
    ab_oma = _fit_affine(ot["d_all"], ot["h_all"], robust=True)
    refs = np.quantile(jt["d_b"], [0.1, 0.5, 0.9]) if jt["d_b"].size else np.array([0., 0., 0.])
    ref_table = [{"depth": float(r),
                  "h_jax": float(apply_affine(ab, r)),
                  "h_oma": float(apply_affine(ab_oma, r))} for r in refs]
    jrows, orows = bin_tables["jax_train"], bin_tables["oma_test"]
    bin_mean_delta = []
    for jr, orr in zip(jrows, orows):
        if jr["depth_mean"] is not None and orr["depth_mean"] is not None:
            bin_mean_delta.append(orr["depth_mean"] - jr["depth_mean"])
        else:
            bin_mean_delta.append(None)
    cross_city_shift = {"jax_affine": [float(ab[0]), float(ab[1])],
                        "oma_affine": [float(ab_oma[0]), float(ab_oma[1])],
                        "ref_table": ref_table, "bin_mean_depth_delta": bin_mean_delta}

    classification = _classify(tall_only["OMA"], ordering["OMA"],
                               mappings["affine"]["eval"]["oma_test"], oracle["oma_test"])

    results = {
        "meta": {"source": source, "evidence_valid": evidence_valid,
                 "depth_model": (cfg.depth.model_id if fake is None else "FAKE_STUB"),
                 "depth_cache": cfg.depth.cache_dir, "cache_hits": cache_hits,
                 "cache_misses": cache_misses, "n_train": len(train), "n_val": len(val),
                 "n_test": len(test), "seed": seed, "diag_edges": DIAG_EDGES,
                 "tall_thresholds": TALL_THRESHOLDS, "leakage_ok_val": leakage_ok_val,
                 "leakage_ok_oma": leakage_ok_oma, "elapsed_s": round(time.time() - t0, 1),
                 "config": config_to_dict(cfg)},
        "correlations": correlations, "tall_only": tall_only, "ordering": ordering,
        "mappings": {k: {kk: vv for kk, vv in v.items() if kk != "map"}
                     for k, v in mappings.items()},  # drop non-serialisable IsotonicMap obj
        "oracle": oracle, "bin_tables": bin_tables, "cross_city_shift": cross_city_shift,
        "classification": classification,
    }

    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    _write_report(results, str(out_dir / "DEPTH_SIGNAL_DIAGNOSTIC.md"))
    _figures(fig_dir, jt, jv, ot, jax, oma, mappings, DIAG_EDGES, evidence_valid)
    _append_experiment_results(results)

    # --- console summary + §27 STOP ---
    s = classification["signals"]
    print("\n" + "=" * 74)
    print("INPUT-SIGNAL DIAGNOSTIC -- does DA-V2 contain usable tall-height signal?")
    print(f"  buildings OMA: Pearson={_fmt(correlations['OMA']['building']['pearson'])} "
          f"Spearman={_fmt(correlations['OMA']['building']['spearman'])}")
    print(f"  tall >15m OMA: Spearman={_fmt(s['oma_tall15_spearman'])} "
          f"adj-bin ordering AUC={_fmt(s['oma_tall_adj_auc'])}")
    print(f"  >15m MAE OMA: global affine={_fmt(s['oma_global_tall15_mae'],2)}m "
          f"oracle={_fmt(s['oma_oracle_tall15_mae'],2)}m "
          f"(gap={_fmt(s['oracle_minus_global_tall15_mae_gap'],2)}m)")
    print(f"  PROPOSED CASE {classification['case']}: {classification['one_line']}")
    if not evidence_valid:
        print("\n[!] evidence_valid=False -- plumbing only; NOT a real result.")
    print("=" * 74)
    print("STOP (§27): diagnostic complete. No training/architecture launched. Await human decision.")


if __name__ == "__main__":
    main()
