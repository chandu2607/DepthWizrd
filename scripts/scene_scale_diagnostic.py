#!/usr/bin/env python
"""DepthWizard INFERENCE-TIME SCENE-SCALE DIAGNOSTIC (Experiment #6) -- DIAGNOSTIC ONLY.

The single question (master prompt §4):
  > Can DepthWizard estimate the required scene-specific metric scale at inference time
  > WITHOUT using ground-truth height for that scene?

This bridges the Exp-5 finding "a per-image ORACLE affine recovers tall height, a single
GLOBAL affine does not" (proposed CASE B) toward a deployable method: is that per-scene
calibration PREDICTABLE from information a real user actually has (RGB + frozen DA-V2 depth)?

What it does (and ONLY this -- §3/§22):
  * REUSES the frozen DA-V2 depth cache (runs/phase1_hf/depth_cache) -- NO recompute, NO torch
    if the cache is complete.
  * Derives the per-scene ORACLE scale target from JAX-train GT ONLY (§6/§7): per-scene robust
    affine (a_i, b_i) == the Exp-5 per-image oracle's own parameters. Compares candidate scale
    statistics for stability and documents the choice.
  * Extracts inference-time features from RGB + depth ONLY (§9): the extractor is structurally
    incapable of seeing GT/CLS, so the target can never leak into the features.
  * Fits the simplest predictors on JAX-train, FROZEN (§8/§10): A = constant global scale,
    B = ridge regression from features (alpha chosen by leave-one-out on JAX-train), and a
    RandomForest nonlinear check (Method C) only reported when B shows signal.
  * Evaluates predicted-vs-oracle scene scale on held-out JAX-val + OMA (§10/§12): MAE, RMSE,
    Pearson, Spearman, relative error, systematic bias -- never fit or tuned on OMA (§20).
  * Three-way DIAGNOSTIC reconstruction Global / Predicted-adaptive / Oracle (§15-§18): apply
    each scene's (a, b) to depth -> metric height; all-pixel / building / tall (>15/>20/>30/40)
    MAE/RMSE/bias -- does adaptive scale attack the actual tall-tail bottleneck?
  * Figures (§13/§14), summary table (§24), CASE A/B/C/D classification (§19), 12 answers (§25).

It does NOT train the fusion head / touch C_log1p / loss / transform / split / DA-V2 weights, or
build any product (§3/§28). It STOPS at the §28 checkpoint for human review. The append to
EXPERIMENT_RESULTS.md is GATED on evidence_valid so a smoke run can never pollute the record.

Usage:
  python scripts/scene_scale_diagnostic.py --config configs/scene_scale_diagnostic.yaml
  python scripts/scene_scale_diagnostic.py --smoke --allow-fake-depth   # plumbing only (NOT evidence)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Force HF cache-only resolution BEFORE any transformers/hf import (documented mirror 429).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.config import load_config, config_to_dict
from depthwizard.data import fetch, datasets
from depthwizard.models.affine import GlobalAffine, _fit_affine
from depthwizard.metrics.height_metrics import valid_mask
from depthwizard.diagnostics.depth_signal import map_metrics, disjoint_ids
from depthwizard.diagnostics import scene_scale as ss

# Reconstruction tall thresholds (§18: >15/>20/>30, plus 40+ where sufficient samples exist).
TALL_THRESHOLDS = [15.0, 20.0, 30.0, 40.0]
ALPHAS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
ALL_CAP = 20000            # all-valid pixels kept per tile for the reconstruction "all" pool
PLOT_CAP = 8000
MIN_SCENE_VALID = 200      # a scene needs this many valid px to yield a stable oracle affine


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
    from depthwizard.depth.depth_anything import DepthAnythingV2
    dm = DepthAnythingV2(cfg.depth.model_id, cfg.depth.input_size,
                         cache_dir=cfg.depth.cache_dir, use_cache=True)
    hits = sum(1 for r in records if dm._cache_path(r.tile_id).exists())
    return hits, len(records) - hits


# ----------------------------------------------------------------- per-scene extraction
def _scene_records(samples, blabel, nodata, cap, seed):
    """One record per SCENE/tile (§11): inference-time features + the GT-derived oracle scale.

    Per scene we store:
      * feat      : the 15-D inference-time feature vector (depth + RGB ONLY -> no GT/CLS leak)
      * a_or,b_or : the per-scene robust affine over ALL valid pixels == Exp-5 oracle params
      * cand      : candidate scale statistics for the §6 stability comparison
      * d_all/h_all (subsampled), d_b/h_b (all building pixels) : pixel pools for reconstruction

    The oracle affine is fit on ALL valid pixels (identical to models.affine.fit_oracle_affine),
    so it reproduces the Exp-5 per-image oracle; the building/tall pools are only for scoring.
    """
    rng = np.random.default_rng(seed)
    scenes = []
    dropped = 0
    for s in samples:
        d = np.asarray(s["depth"], np.float64)
        h = np.asarray(s["gt"], np.float64)
        vm = valid_mask(h, d, nodata=nodata)
        n_valid = int(vm.sum())
        if n_valid < MIN_SCENE_VALID:
            dropped += 1
            continue
        aff = ss.scene_affine(d, h, nodata=nodata, robust=True)
        if not aff["ok"]:
            dropped += 1
            continue
        cand = ss.scene_scale_candidates(d, h, nodata=nodata)
        feat = ss.scene_features(d, s["rgb"])            # depth + RGB only
        cls = s.get("cls")
        bmask = (vm & (np.asarray(cls) == blabel)) if cls is not None else np.zeros_like(vm)
        # all-valid pool (subsample per scene, unbiased) for the reconstruction "all" metric
        vidx = np.flatnonzero(vm)
        if vidx.size > cap:
            vidx = rng.choice(vidx, cap, replace=False)
        dr, hr = d.ravel(), h.ravel()
        scenes.append({
            "id": s["id"], "city": s.get("city"),
            "feat": feat, "fvec": ss.features_to_vector(feat),
            "a_or": aff["a"], "b_or": aff["b"], "n_valid": n_valid,
            "cand": cand,
            "d_all": dr[vidx].copy(), "h_all": hr[vidx].copy(),
            "d_b": d[bmask].copy(), "h_b": h[bmask].copy(),
        })
    return scenes, dropped


def _matrix(scenes):
    X = np.stack([s["fvec"] for s in scenes], axis=0) if scenes else np.zeros((0, len(ss.FEATURE_NAMES)))
    a = np.array([s["a_or"] for s in scenes], np.float64)
    b = np.array([s["b_or"] for s in scenes], np.float64)
    return X, a, b


# ----------------------------------------------------------------- §6 target stability
def _target_stability(scenes):
    """Compare candidate per-scene SCALE statistics on JAX-train (§6) -> justify the choice.

    Reports central tendency + spread (coeff. of variation) and cross-correlation of the
    candidates. A stable, meaningful target is one with moderate (not degenerate) spread that
    tracks the robust affine slope we ultimately predict.
    """
    keys = ["a_robust", "a_ols", "median_ratio"]
    cols = {k: np.array([s["cand"].get(k, np.nan) for s in scenes], np.float64) for k in keys}
    stats = {}
    for k, v in cols.items():
        vv = v[np.isfinite(v)]
        mean = float(np.mean(vv)) if vv.size else float("nan")
        std = float(np.std(vv)) if vv.size else float("nan")
        stats[k] = {"n": int(vv.size), "mean": mean, "median": float(np.median(vv)) if vv.size else float("nan"),
                    "std": std, "cv": (std / abs(mean)) if (vv.size and abs(mean) > 1e-9) else float("nan"),
                    "p10": float(np.percentile(vv, 10)) if vv.size else float("nan"),
                    "p90": float(np.percentile(vv, 90)) if vv.size else float("nan")}
    corr = {}
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            m = np.isfinite(cols[k1]) & np.isfinite(cols[k2])
            corr[f"{k1}~{k2}"] = ss.pearson(cols[k1][m], cols[k2][m]) if m.sum() > 2 else float("nan")
    return {"stats": stats, "cross_corr": corr,
            "choice": "a_robust",
            "rationale": ("per-scene robust affine slope: identical to the Exp-5 per-image oracle "
                          "parameters, robust to tall-pixel outliers, and directly usable in the "
                          "reconstruction h = a*d + b. median(h/d) conflates scale with the depth "
                          "offset and is unstable when depth ~ 0.")}


# ----------------------------------------------------------------- reconstruction (§15-§18)
def _pool_reconstruct(scenes, ab_of):
    """Apply a per-scene affine (a,b)=ab_of(scene) -> metric height; pool errors by subset.

    Returns MAE/RMSE/bias for all-valid / building / tall(>15/>20/>30/40). Same subsample per
    scene is shared across methods, so Global/Predicted/Oracle are compared on identical pixels.
    """
    pa, ph = [], []           # all-valid
    pb, hb = [], []           # building
    for s in scenes:
        a, b = ab_of(s)
        if a is None or not np.isfinite(a) or not np.isfinite(b):
            continue
        if s["d_all"].size:
            pa.append(a * s["d_all"] + b); ph.append(s["h_all"])
        if s["d_b"].size:
            pb.append(a * s["d_b"] + b); hb.append(s["h_b"])
    cat = lambda L: np.concatenate(L) if L else np.array([], np.float64)
    pa, ph, pb, hb = cat(pa), cat(ph), cat(pb), cat(hb)
    out = {"all": map_metrics(pa, ph), "building": map_metrics(pb, hb)}
    for t in TALL_THRESHOLDS:
        sel = hb > t
        out[f"tall_{int(t)}"] = map_metrics(pb[sel], hb[sel])
    return out


# ----------------------------------------------------------------- CASE classification (§19)
def _classify(scale_val, scale_oma, recon_oma_global, recon_oma_pred, recon_oma_oracle):
    """Transparent thresholds -> CASE A/B/C/D from held-out JAX + OMA (human decides, §28).

    A: scene scale predictable INCLUDING OMA and adaptive reconstruction helps -> deployable plausible.
    B: predictable on held-out JAX but NOT OMA -> domain-robustness question, not solved.
    C: not predictable even on held-out JAX -> need an additional metric/physical cue.
    D: scale predictable (incl OMA) but tall height still poor -> scale is only part of the problem.
    """
    val_sp = scale_val["pred_a"]["spearman"]
    oma_sp = scale_oma["pred_a"]["spearman"]
    oma_rel = scale_oma["pred_a"]["rel_median"]
    val_sp = val_sp if (val_sp is not None and val_sp == val_sp) else 0.0
    oma_sp = oma_sp if (oma_sp is not None and oma_sp == oma_sp) else 0.0
    oma_rel = oma_rel if (oma_rel is not None and oma_rel == oma_rel) else float("inf")

    g15 = recon_oma_global["tall_15"]["mae"]
    p15 = recon_oma_pred["tall_15"]["mae"]
    o15 = recon_oma_oracle["tall_15"]["mae"]
    gb = recon_oma_global["building"]["mae"]
    pb = recon_oma_pred["building"]["mae"]

    def _n(x):
        return x if (x is not None and x == x) else None
    g15, p15, o15, gb, pb = map(_n, (g15, p15, o15, gb, pb))

    # "predictable" = rank agreement with the oracle scale (Spearman) above a modest floor.
    val_predictable = val_sp >= 0.35
    oma_predictable = (oma_sp >= 0.35) and (oma_rel <= 0.60)
    # "helps tall" = predicted-adaptive lowers OMA tall MAE meaningfully vs global, toward oracle.
    helps_tall = (g15 is not None and p15 is not None and (g15 - p15) >= 1.0)
    helps_bldg = (gb is not None and pb is not None and (gb - pb) >= 0.25)

    if not val_predictable:
        case = "C"
        one = ("SCALE NOT PREDICTABLE: observable RGB/depth features cannot recover the per-scene "
               "oracle scale even on held-out JAX -> scene-specific calibration needs an additional "
               "metric/physical cue not present in a bare image.")
    elif val_predictable and not oma_predictable:
        case = "B"
        one = ("PREDICTABLE ON JAX, NOT ON OMA: scene scale is real and learnable in-domain but the "
               "cues do not survive the cross-city depth-representation shift -> this is now a domain-"
               "robustness problem; adaptive scale is NOT demonstrated to solve it.")
    elif oma_predictable and helps_tall:
        case = "A"
        one = ("STRONG SCENE-SCALE PREDICTABILITY: features predict the oracle scale on held-out JAX "
               "AND OMA, and applying the predicted scale improves the tall tail -> a deployable "
               "adaptive-scale approach is plausible (next: integrate carefully).")
    else:
        case = "D"
        one = ("SCALE PREDICTABLE BUT HEIGHT STILL POOR: even with a reasonable predicted scale the "
               "tall-building error does not improve enough -> scale ambiguity is only part of the "
               "problem; do not declare success.")
    return {"case": case, "one_line": one,
            "signals": {"jaxval_scale_spearman": val_sp, "oma_scale_spearman": oma_sp,
                        "oma_scale_rel_median": oma_rel,
                        "val_predictable": val_predictable, "oma_predictable": oma_predictable,
                        "helps_tall_oma": helps_tall, "helps_building_oma": helps_bldg,
                        "oma_global_tall15_mae": g15, "oma_pred_tall15_mae": p15,
                        "oma_oracle_tall15_mae": o15}}


# ----------------------------------------------------------------- figures
def _figures(fig_dir, results, packs, evidence_valid):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[viz] skipped ({e})")
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if evidence_valid else "  [NON-EVIDENCE smoke]"

    # ---- §13 predicted vs oracle scene scale: JAX-train(LOO) / JAX-val / OMA, with y=x ----
    try:
        fig, ax = plt.subplots(1, 3, figsize=(16, 5.0))
        panels = [("JAX-train (LOO, in-sample)", packs["train_loo_a"], packs["train_a"], "#9aa0a6"),
                  ("held-out JAX-val", packs["val_pred_a"], packs["val_a"], "#1f6feb"),
                  ("OMA (cross-city)", packs["oma_pred_a"], packs["oma_a"], "#d1242f")]
        for k, (name, pred, orac, col) in enumerate(panels):
            if orac.size:
                lo = float(min(np.min(orac), np.min(pred))) if pred.size else float(np.min(orac))
                hi = float(max(np.max(orac), np.max(pred))) if pred.size else float(np.max(orac))
                pad = 0.05 * (hi - lo + 1e-6)
                ax[k].plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, label="y = x")
                ax[k].scatter(orac, pred, s=22, alpha=0.6, c=col, edgecolors="none")
            ax[k].set_xlabel("oracle scene scale a (GT-fit)")
            ax[k].set_ylabel("predicted scene scale a")
            ax[k].set_title(name)
            ax[k].legend(loc="upper left", fontsize=8)
        fig.suptitle(f"Predicted vs oracle scene scale -- freeze on JAX-train, eval JAX-val/OMA{tag}")
        fig.tight_layout(); fig.savefig(fig_dir / "scale_pred_vs_oracle.png", dpi=110); plt.close(fig)
    except Exception as e:
        print(f"[viz] scatter skipped: {e}")

    # ---- §14 JAX vs OMA scene-scale distribution ----
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        ja, oa = packs["train_a"], packs["oma_a"]
        va = packs["val_a"]
        bins = np.linspace(float(min(ja.min(), oa.min())), float(max(ja.max(), oa.max())), 30) \
            if (ja.size and oa.size) else 20
        ax.hist(ja, bins=bins, alpha=0.5, color="#1f6feb", label=f"JAX-train oracle a (n={ja.size})")
        if va.size:
            ax.hist(va, bins=bins, alpha=0.5, color="#54aeff", label=f"JAX-val oracle a (n={va.size})")
        ax.hist(oa, bins=bins, alpha=0.5, color="#d1242f", label=f"OMA oracle a (n={oa.size})")
        ax.axvline(results["global"]["a"], color="black", lw=1.5, ls=":",
                   label=f"global Baseline-B a={results['global']['a']:.2f}")
        ax.set_xlabel("per-scene oracle scale a (height per unit depth)")
        ax.set_ylabel("scene count")
        ax.set_title(f"Scene-scale distribution shift: JAX vs OMA{tag}")
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(fig_dir / "scale_distribution_jax_vs_oma.png", dpi=110); plt.close(fig)
    except Exception as e:
        print(f"[viz] distribution skipped: {e}")

    # ---- §18 tall-tail MAE: Global vs Predicted vs Oracle (OMA + JAX-val) ----
    try:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        xs = ["bldg", ">15", ">20", ">30", ">40"]
        keys = ["building", "tall_15", "tall_20", "tall_30", "tall_40"]
        for k, (dom, title) in enumerate([("jax_val", "JAX-val (in-domain)"),
                                          ("oma_test", "OMA (cross-city)")]):
            for lbl, col, mk in [("Global", "#1f6feb", "o"), ("Predicted", "#8250df", "s"),
                                 ("Oracle (UB)", "#1a7f37", "^")]:
                rec = results["reconstruction"][dom][lbl.split()[0].lower()]
                ax[k].plot(xs, [rec[kk]["mae"] if rec[kk]["mae"] is not None else np.nan for kk in keys],
                           marker=mk, color=col, label=lbl)
            ax[k].set_title(title); ax[k].set_ylabel("MAE (m)"); ax[k].set_xlabel("subset")
            ax[k].legend(fontsize=8)
        fig.suptitle(f"Does adaptive scale attack the tall tail? Global vs Predicted vs Oracle{tag}")
        fig.tight_layout(); fig.savefig(fig_dir / "recon_tall_mae.png", dpi=110); plt.close(fig)
    except Exception as e:
        print(f"[viz] recon MAE skipped: {e}")

    # ---- univariate feature signal vs scale (top feature) ----
    try:
        screen = results["feature_screen"]
        top = screen[0]["feature"] if screen else None
        if top is not None:
            j = ss.FEATURE_NAMES.index(top)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(packs["train_X"][:, j], packs["train_a"], s=20, alpha=0.6,
                       c="#1f6feb", label="JAX-train")
            if packs["oma_X"].size:
                ax.scatter(packs["oma_X"][:, j], packs["oma_a"], s=20, alpha=0.6,
                           c="#d1242f", label="OMA")
            ax.set_xlabel(f"{top} (top |Spearman| feature)")
            ax.set_ylabel("oracle scene scale a")
            ax.set_title(f"Strongest single scale cue: {top}{tag}")
            ax.legend(fontsize=8); fig.tight_layout()
            fig.savefig(fig_dir / "top_feature_vs_scale.png", dpi=110); plt.close(fig)
    except Exception as e:
        print(f"[viz] feature scatter skipped: {e}")
    print(f"[viz] figures written to {fig_dir}")


# ----------------------------------------------------------------- report
def _scale_row(name, m):
    return (f"| {name} | {_fmt(m['mae'],3)} | {_fmt(m['rmse'],3)} | {_fmt(m['bias'],3)} "
            f"| {_fmt(m['pearson'])} | {_fmt(m['spearman'])} | {_fmt(m['rel_median'],3)} | {m['n']} |")


def _recon_row(name, ev):
    a, b = ev["all"], ev["building"]
    return (f"| {name} | {_fmt(a['mae'],2)} | {_fmt(a['rmse'],2)} | {_fmt(b['mae'],2)} "
            f"| {_fmt(b['rmse'],2)} | {_fmt(b['bias'],2)} | {_fmt(ev['tall_15']['mae'],2)} "
            f"| {_fmt(ev['tall_20']['mae'],2)} | {_fmt(ev['tall_30']['mae'],2)} "
            f"| {_fmt(ev['tall_40']['mae'],2)} |")


def _write_report(results, path):
    m = results["meta"]
    cls = results["classification"]
    st = results["target_stability"]
    L = []
    L.append("# DepthWizard -- INFERENCE-TIME SCENE-SCALE DIAGNOSTIC (Experiment #6)")
    L.append("## Can the per-scene metric SCALE be predicted from inference-time RGB + depth, "
             "without that scene's ground truth?\n")
    L.append("_Generated by `scripts/scene_scale_diagnostic.py`. DIAGNOSTIC ONLY: no model was "
             "trained; C_log1p, the fusion head, loss, target transform, dataset split and DA-V2 "
             "weights are untouched (§3/§28). Every number below is measured by this run from the "
             "REUSED frozen DA-V2 depth cache; missing values read `n/a`._\n")
    if not m["evidence_valid"]:
        L.append("> # WARNING -- NOT VALID EVIDENCE (synthetic/fake-depth smoke). "
                 "Numbers only prove the code executes; not appended to EXPERIMENT_RESULTS.md.\n")
    L.append("### Run metadata (§10/§23)\n")
    L.append(f"- dataset source: `{m['source']}` | evidence valid: **{m['evidence_valid']}**")
    L.append(f"- depth prior: `{m['depth_model']}` | cache: `{m['depth_cache']}` "
             f"(REUSED from Phase-1; hits={m['cache_hits']} misses={m['cache_misses']})")
    L.append(f"- split (city-held-out): fit predictor on **JAX-train** ({m['n_train']} scenes); "
             f"eval on held-out **JAX-val** ({m['n_val']}) + **OMA-test** ({m['n_test']}) "
             f"(dropped {m['dropped']} scenes with < {MIN_SCENE_VALID} valid px)")
    L.append(f"- no fit/eval leakage (disjoint tile ids): JAX-train vs JAX-val "
             f"**{m['leakage_ok_val']}**, vs OMA **{m['leakage_ok_oma']}**")
    L.append(f"- ridge alpha (LOO-selected on JAX-train): **{m['alpha']}** | features: "
             f"{len(ss.FEATURE_NAMES)} | seed {m['seed']} | runtime {m['elapsed_s']}s\n")
    if m["source"] == "hf_blocks":
        L.append("> **Provenance caveat:** unofficial HF mirror `JasonXF/DFC2019-10k`, preprocessed "
                 "nDSM (ground floored to 0). Feasibility evidence only; re-confirm on official IEEE "
                 "GRSS DFC2019 before external reporting.\n")

    # --- §6/§7 target definition
    L.append("---\n## 1. Scene-scale target: definition + stability (§6/§7)\n")
    L.append("The per-scene ORACLE scale is the **robust affine slope `a_i`** from `h = a_i·d + b_i` "
             "fit over ALL valid pixels of scene *i* -- exactly the Exp-5 per-image oracle's own "
             "parameters. It is derived from **JAX-train GT only** and is the quantity we try to "
             "PREDICT; it is never a feature and never the deployable answer (§7).\n")
    L.append("Candidate scale statistics compared on JAX-train for stability (§6):\n")
    L.append("| Candidate | n | mean | median | std | CV (std/|mean|) | p10 | p90 |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for k in ["a_robust", "a_ols", "median_ratio"]:
        s = st["stats"][k]
        L.append(f"| {k} | {s['n']} | {_fmt(s['mean'])} | {_fmt(s['median'])} | {_fmt(s['std'])} "
                 f"| {_fmt(s['cv'])} | {_fmt(s['p10'])} | {_fmt(s['p90'])} |")
    L.append(f"\nCross-correlation (Pearson) among candidates: "
             f"{ {k: _fmt(v) for k, v in st['cross_corr'].items()} }.")
    L.append(f"\n**Chosen target: `{st['choice']}`.** {st['rationale']}\n")

    # --- §9 features + leakage
    L.append("---\n## 2. Inference-time features + leakage discipline (§5/§9)\n")
    L.append(f"All {len(ss.FEATURE_NAMES)} features are computed by `scene_features(depth, rgb)` "
             "from ONLY the RGB image and the frozen DA-V2 relative depth. The function signature "
             "excludes GT and CLS, so the target cannot leak into the features (§9). Features:\n")
    L.append(f"`{', '.join(ss.FEATURE_NAMES)}`\n")
    L.append("Leakage checks performed: (a) structural -- extractor takes no GT/CLS; (b) unit test "
             "`test_features_ignore_gt` perturbs GT and confirms features are byte-identical; "
             "(c) tile-id disjointness between fit and eval; (d) predictor fit on JAX-train only, "
             "never on OMA (§20).\n")
    L.append("### Univariate feature signal vs oracle scale (JAX-train, sorted by |Spearman|)\n")
    L.append("| Feature | Pearson | Spearman |")
    L.append("|---|--:|--:|")
    for r in results["feature_screen"]:
        L.append(f"| {r['feature']} | {_fmt(r['pearson'])} | {_fmt(r['spearman'])} |")

    # --- §12 scale-prediction results
    L.append("\n---\n## 3. Scale-prediction accuracy: predicted vs oracle scale (§12)\n")
    L.append("_Scale target = per-scene `a`. Method A = constant (global Baseline-B slope applied "
             "to every scene); Method B = ridge from features; Oracle = 0 by construction. JAX-train "
             "row is leave-one-out (honest in-sample); JAX-val + OMA are the generalization evidence "
             "(§13: do not read the train panel as generalization)._\n")
    for dom, lbl in [("train_loo", "JAX-train (LOO)"), ("jax_val", "held-out JAX-val"),
                     ("oma_test", "OMA (cross-city)")]:
        L.append(f"\n**{lbl} -- scale `a`**\n")
        L.append("| Method | MAE | RMSE | bias | Pearson | Spearman | rel.err (median) | n |")
        L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
        L.append(_scale_row("A: constant global", results["scale"][dom]["global_a"]))
        L.append(_scale_row("B: ridge (features)", results["scale"][dom]["pred_a"]))
        if results["scale"][dom].get("rf_a") is not None:
            L.append(_scale_row("C: random forest", results["scale"][dom]["rf_a"]))
    L.append("\n_Offset `b` (predicted jointly by the same ridge, used in reconstruction):_\n")
    L.append("| Domain | ridge b MAE | ridge b Pearson | ridge b Spearman |")
    L.append("|---|--:|--:|--:|")
    for dom, lbl in [("train_loo", "JAX-train (LOO)"), ("jax_val", "JAX-val"), ("oma_test", "OMA")]:
        mb = results["scale"][dom]["pred_b"]
        L.append(f"| {lbl} | {_fmt(mb['mae'],3)} | {_fmt(mb['pearson'])} | {_fmt(mb['spearman'])} |")

    # --- §15-§18 three-way reconstruction
    L.append("\n---\n## 4. Three-way reconstruction: Global vs Predicted vs Oracle (§15-§18)\n")
    L.append("_Apply each scene's affine `h = a·d + b` to the frozen depth. Global = one pooled "
             "Baseline-B affine for every scene; Predicted = per-scene ridge (a,b); Oracle = per-scene "
             "GT-fit (a,b), diagnostic-only upper bound. Same pixels across methods._\n")
    for dom, lbl in [("jax_val", "JAX-val (in-domain)"), ("oma_test", "OMA (cross-city)")]:
        L.append(f"\n**{lbl}**\n")
        L.append("| Method | all MAE | all RMSE | bldg MAE | bldg RMSE | bldg bias | >15 MAE "
                 "| >20 MAE | >30 MAE | >40 MAE |")
        L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
        L.append(_recon_row("Global (Baseline-B)", results["reconstruction"][dom]["global"]))
        L.append(_recon_row("Predicted adaptive", results["reconstruction"][dom]["predicted"]))
        L.append(_recon_row("Oracle (UB, diag-only)", results["reconstruction"][dom]["oracle"]))

    # --- §24 summary table
    L.append("\n---\n## 5. §24 required summary table\n")
    L.append("| Method | JAX val scale err (MAE) | OMA scale err (MAE) | Building MAE | "
             "Building RMSE | >15m MAE | >30m MAE |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    for method, skey, rkey, note in [("Global", "global_a", "global", ""),
                                     ("Predicted adaptive", "pred_a", "predicted", ""),
                                     ("Oracle (diagnostic-only)", None, "oracle", "diag")]:
        vse = results["scale"]["jax_val"][skey]["mae"] if skey else 0.0
        ose = results["scale"]["oma_test"][skey]["mae"] if skey else 0.0
        rec = results["reconstruction"]["oma_test"][rkey]
        L.append(f"| {method} | {_fmt(vse,3)} | {_fmt(ose,3)} | {_fmt(rec['building']['mae'],2)} "
                 f"| {_fmt(rec['building']['rmse'],2)} | {_fmt(rec['tall_15']['mae'],2)} "
                 f"| {_fmt(rec['tall_30']['mae'],2)} |")
    L.append("\n_Oracle scale error is 0 by construction (it IS the target); reconstruction columns "
             "are OMA. Oracle rows are diagnostic-only upper bounds, NOT deployable (§7/§24)._\n")

    # --- §14 cross-city scale shift
    L.append("---\n## 6. Cross-city scene-scale shift (§14)\n")
    sh = results["scale_shift"]
    L.append(f"- JAX oracle scale a: mean {_fmt(sh['jax_mean'])}, median {_fmt(sh['jax_median'])}, "
             f"std {_fmt(sh['jax_std'])} (n={sh['jax_n']})")
    L.append(f"- OMA oracle scale a: mean {_fmt(sh['oma_mean'])}, median {_fmt(sh['oma_median'])}, "
             f"std {_fmt(sh['oma_std'])} (n={sh['oma_n']})")
    L.append(f"- global Baseline-B slope (pooled JAX-train): a={_fmt(results['global']['a'])}, "
             f"b={_fmt(results['global']['b'])}")
    L.append(f"- JAX->OMA mean-scale ratio: {_fmt(sh['ratio'])}x -- the predictor must recognise this "
             f"shift from image/depth features alone to generalize cross-city.\n")

    # --- §19 CASE
    L.append("---\n## 7. CASE classification (§19)\n")
    L.append(f"### PROPOSED: **CASE {cls['case']}**\n")
    L.append(f"{cls['one_line']}\n")
    s = cls["signals"]
    L.append("Decisive signals:\n")
    L.append(f"- held-out JAX scale Spearman (pred vs oracle a): **{_fmt(s['jaxval_scale_spearman'])}**")
    L.append(f"- OMA scale Spearman: **{_fmt(s['oma_scale_spearman'])}** | OMA median relative "
             f"scale error: **{_fmt(s['oma_scale_rel_median'],3)}**")
    L.append(f"- OMA tall >15m MAE: global **{_fmt(s['oma_global_tall15_mae'],2)}** -> predicted "
             f"**{_fmt(s['oma_pred_tall15_mae'],2)}** (oracle **{_fmt(s['oma_oracle_tall15_mae'],2)}**)")
    L.append(f"- val_predictable={s['val_predictable']} | oma_predictable={s['oma_predictable']} "
             f"| helps_tall_oma={s['helps_tall_oma']} | helps_building_oma={s['helps_building_oma']}")
    L.append("\n_Thresholds (heuristic, for auditability): predictable = scale Spearman >= 0.35 "
             "(OMA also requires median rel. error <= 0.60); helps_tall = predicted lowers OMA >15m "
             "MAE by >= 1.0 m vs global. The human makes the final call (§28)._\n")

    # --- §25 twelve questions
    L.append("---\n## 8. §25 required final questions\n")
    for q, a in results["answers"]:
        L.append(f"**Q{q}**")
        L.append(f"{a}\n")

    # --- limitations
    L.append("---\n## 9. Limitations\n")
    L.append("- Unofficial DFC2019 mirror, preprocessed nDSM (ground floored to 0); single training "
             "city (JAX), single cross-city test (OMA). Scene count is small (per-scene unit): "
             f"{m['n_train']}/{m['n_val']}/{m['n_test']}.")
    L.append("- The oracle scale peeks at each scene's GT and is a NON-deployable upper bound.")
    L.append("- Reconstruction uses the frozen relative depth directly (diagnostic), not C_log1p; it "
             "measures the value of scale ADAPTATION, not a production accuracy.")
    L.append("- DA-V2 depth is scale/shift-ambiguous; features describe distribution/appearance, not "
             "measured geometry. No camera/sensor metadata is assumed (§21/§22).")
    L.append("- This diagnostic does NOT establish final model / ISRO / DSM / DEM / 3D / production "
             "accuracy (§26).\n")
    L.append("> MANDATORY STOP (§28): diagnostic only. No training/architecture/product was launched. "
             "Await human decision before any scale-predictor integration.\n")
    Path(path).write_text("\n".join(L), encoding="utf-8")
    print(f"[report] wrote {path}")


def _append_experiment_results(results, path="EXPERIMENT_RESULTS.md"):
    """Append a concise section; GATED on evidence_valid by the caller (Observation 10 fix)."""
    p = Path(path)
    m, cls = results["meta"], results["classification"]
    s = cls["signals"]
    gv = results["reconstruction"]["oma_test"]["global"]
    pv = results["reconstruction"]["oma_test"]["predicted"]
    ov = results["reconstruction"]["oma_test"]["oracle"]
    block = [
        "\n\n---\n\n## SCENE-SCALE DIAGNOSTIC (Experiment #6) — is per-scene metric scale "
        "inferable at inference time?\n",
        f"_Diagnostic only (no training); generated by `scripts/scene_scale_diagnostic.py`. "
        f"Reused frozen DA-V2 cache (hits={m['cache_hits']}/{m['cache_hits']+m['cache_misses']}). "
        f"Predictor fit on JAX-train ({m['n_train']} scenes), frozen, eval held-out JAX-val "
        f"({m['n_val']}) + OMA ({m['n_test']}). Target = per-scene robust affine slope (Exp-5 "
        f"oracle params). Ridge alpha={m['alpha']} (LOO). Evidence valid: {m['evidence_valid']}._\n",
        f"- **Scale predictability (pred vs oracle a, Spearman):** held-out JAX "
        f"{_fmt(s['jaxval_scale_spearman'])}, OMA {_fmt(s['oma_scale_spearman'])} "
        f"(OMA median relative scale error {_fmt(s['oma_scale_rel_median'],3)}).",
        f"- **OMA reconstruction (Global -> Predicted -> Oracle):** building MAE "
        f"{_fmt(gv['building']['mae'],2)} -> {_fmt(pv['building']['mae'],2)} -> "
        f"{_fmt(ov['building']['mae'],2)} m; >15m MAE {_fmt(gv['tall_15']['mae'],2)} -> "
        f"{_fmt(pv['tall_15']['mae'],2)} -> {_fmt(ov['tall_15']['mae'],2)} m; >30m MAE "
        f"{_fmt(gv['tall_30']['mae'],2)} -> {_fmt(pv['tall_30']['mae'],2)} -> "
        f"{_fmt(ov['tall_30']['mae'],2)} m.",
        f"- **PROPOSED CASE {cls['case']}** — {cls['one_line']}",
        f"- Full report + figures + tables: `runs/scene_scale_diagnostic/`. STOP for human "
        f"review (§28); no architecture/predictor integrated automatically.\n",
    ]
    with p.open("a", encoding="utf-8") as f:
        f.write("\n".join(block))
    print(f"[results] appended Experiment #6 section to {path}")


# ----------------------------------------------------------------- answers (§25)
def _build_answers(results):
    sc = results["scale"]; rec = results["reconstruction"]; s = results["classification"]["signals"]
    fs = results["feature_screen"]
    top3 = ", ".join(f"{r['feature']} (S={_fmt(r['spearman'],2)})" for r in fs[:3]) if fs else "n/a"
    go, po, oo = rec["oma_test"]["global"], rec["oma_test"]["predicted"], rec["oma_test"]["oracle"]

    def cmp(a, b):
        if a is None or b is None or a != a or b != b:
            return "n/a"
        d = a - b
        return f"{_fmt(a,2)}->{_fmt(b,2)} ({'improves' if d > 0 else 'worsens'} {_fmt(abs(d),2)} m)"

    A = []
    A.append((1, f"Partially. On held-out JAX the predicted scale tracks the oracle with Spearman "
                 f"{_fmt(sc['jax_val']['pred_a']['spearman'])} (MAE {_fmt(sc['jax_val']['pred_a']['mae'],3)}); "
                 f"on OMA Spearman {_fmt(sc['oma_test']['pred_a']['spearman'])}, median relative error "
                 f"{_fmt(sc['oma_test']['pred_a']['rel_median'],3)}. See CASE "
                 f"{results['classification']['case']} for the confidence-qualified verdict."))
    A.append((2, f"By univariate |Spearman| vs the oracle scale on JAX-train, the strongest cues are: "
                 f"{top3}. These are depth-distribution / appearance statistics (no GT)."))
    A.append((3, f"Held-out JAX generalization: ridge scale Spearman {_fmt(sc['jax_val']['pred_a']['spearman'])} "
                 f"vs the constant-global floor MAE {_fmt(sc['jax_val']['global_a']['mae'],3)} "
                 f"(ridge MAE {_fmt(sc['jax_val']['pred_a']['mae'],3)}). "
                 f"{'Yes' if s['val_predictable'] else 'Weak/no'} in-domain signal."))
    A.append((4, f"OMA generalization: scale Spearman {_fmt(sc['oma_test']['pred_a']['spearman'])}, "
                 f"median relative error {_fmt(sc['oma_test']['pred_a']['rel_median'],3)}, bias "
                 f"{_fmt(sc['oma_test']['pred_a']['bias'],3)}. "
                 f"{'Generalizes' if s['oma_predictable'] else 'Does NOT generalize cleanly'} cross-city."))
    A.append((5, f"Predicted scene scale accuracy (a): JAX-val MAE {_fmt(sc['jax_val']['pred_a']['mae'],3)} "
                 f"/ RMSE {_fmt(sc['jax_val']['pred_a']['rmse'],3)}; OMA MAE {_fmt(sc['oma_test']['pred_a']['mae'],3)} "
                 f"/ RMSE {_fmt(sc['oma_test']['pred_a']['rmse'],3)} (vs constant-global OMA MAE "
                 f"{_fmt(sc['oma_test']['global_a']['mae'],3)})."))
    A.append((6, f"Building height MAE on OMA under adaptive scale: {cmp(go['building']['mae'], po['building']['mae'])} "
                 f"(oracle upper bound {_fmt(oo['building']['mae'],2)} m)."))
    A.append((7, f"Tall buildings on OMA (Global->Predicted): >15m {cmp(go['tall_15']['mae'], po['tall_15']['mae'])}; "
                 f">20m {cmp(go['tall_20']['mae'], po['tall_20']['mae'])}; "
                 f">30m {cmp(go['tall_30']['mae'], po['tall_30']['mae'])}."))
    A.append((8, f"High-height underestimation (bias, OMA >30m): global {_fmt(go['tall_30']['bias'],2)} m -> "
                 f"predicted {_fmt(po['tall_30']['bias'],2)} m (oracle {_fmt(oo['tall_30']['bias'],2)} m). "
                 f"Negative bias = underestimation."))
    A.append((9, f"Distance to oracle on OMA >15m MAE: global {_fmt(go['tall_15']['mae'],2)}, predicted "
                 f"{_fmt(po['tall_15']['mae'],2)}, oracle {_fmt(oo['tall_15']['mae'],2)} m. Predicted closes "
                 f"{_fmt(_frac_closed(go['tall_15']['mae'], po['tall_15']['mae'], oo['tall_15']['mae']),1)}% "
                 f"of the global->oracle gap."))
    A.append((10, f"No -- scene scale is {'necessary but not sufficient' if results['classification']['case'] in ('B','D') else 'a genuine lever'}. "
                  f"CASE {results['classification']['case']}: {results['classification']['one_line']}"))
    A.append((11, "Missing information: a per-scene absolute metric anchor that survives cross-city depth-"
                  "representation shift (e.g. genuine GSD/geometry when available, or a small amount of "
                  "in-domain target-city calibration). A bare PNG/JPG provides no physical scale (§22)."))
    A.append((12, results["next_experiment"]))
    return A


def _frac_closed(g, p, o):
    if None in (g, p, o) or g != g or p != p or o != o or abs(g - o) < 1e-9:
        return float("nan")
    return 100.0 * (g - p) / (g - o)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/scene_scale_diagnostic.yaml")
    ap.add_argument("--smoke", action="store_true", help="force synthetic data (NOT evidence)")
    ap.add_argument("--allow-fake-depth", action="store_true",
                    help="fabricate depth from GT if cache/torch unavailable (NOT evidence)")
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

    # --- depth: reuse cache; only touch torch if an entry is genuinely missing ---
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
                      "Re-run where the cache is complete, or pass --allow-fake-depth for a "
                      "NON-EVIDENCE smoke.")
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

    leakage_ok_val = disjoint_ids(tr_ids, va_ids)
    leakage_ok_oma = disjoint_ids(tr_ids, te_ids)
    if not (leakage_ok_val and leakage_ok_oma):
        print("[error] train/eval tile-id overlap -> would leak. Aborting."); sys.exit(5)
    evidence_valid = (not is_synth) and (fake is None)

    blabel, nodata = cfg.data.building_label, cfg.data.nodata
    tr_sc, dtr = _scene_records(train, blabel, nodata, ALL_CAP, seed)
    va_sc, dva = _scene_records(val, blabel, nodata, ALL_CAP, seed)
    ot_sc, dot = _scene_records(test, blabel, nodata, ALL_CAP, seed)
    dropped = dtr + dva + dot
    if not tr_sc or not ot_sc:
        print("[error] no usable scenes after valid-pixel filter."); sys.exit(4)
    print(f"[scenes] train={len(tr_sc)} val={len(va_sc)} oma={len(ot_sc)} (dropped {dropped})")

    Xtr, atr, btr = _matrix(tr_sc)
    Xva, ava, bva = _matrix(va_sc)
    Xot, aot, bot = _matrix(ot_sc)

    # --- §6 target stability + univariate feature screen (JAX-train only) ---
    target_stability = _target_stability(tr_sc)
    feature_screen = ss.univariate_screen(Xtr, atr)

    # --- global (Baseline-B) anchor: pooled JAX-train affine, one scale for everything (§15) ---
    gb = GlobalAffine(max_pixels=getattr(cfg.train, "max_train_pixels_affine", 2_000_000),
                      seed=cfg.split.seed).fit(train)
    a_g, b_g = float(gb.a), float(gb.b)
    print(f"[global] Baseline-B pooled affine a={a_g:.4g} b={b_g:.4g}")

    # --- Method B: ridge (a,b) ~ features, alpha by LOO on JAX-train scale target (§8/§10) ---
    Y = np.stack([atr, btr], axis=1)
    alpha, alpha_mae, alpha_scores = ss.select_alpha(Xtr, atr, ALPHAS)
    mu, sd = ss.standardize_fit(Xtr)
    ridge = ss.fit_ridge(ss.standardize_apply(Xtr, mu, sd), Y, alpha)
    print(f"[ridge] alpha={alpha} (LOO scale MAE={alpha_mae:.4g})")

    def _pred_ab(X):
        if X.shape[0] == 0:
            return np.zeros((0, 2))
        return ss.predict_ridge(ridge, ss.standardize_apply(X, mu, sd))

    pred_tr = _pred_ab(Xtr); pred_va = _pred_ab(Xva); pred_ot = _pred_ab(Xot)
    loo_ab = ss.loo_predict(Xtr, Y, alpha)                    # honest in-sample (§13 train panel)
    loo_a = loo_ab[:, 0] if loo_ab.ndim == 2 else loo_ab
    loo_b = loo_ab[:, 1] if loo_ab.ndim == 2 else loo_ab

    # --- Method C: RandomForest nonlinear check, reported only if B shows in-domain signal ---
    rf_loo_a = None; rf_va_a = None; rf_ot_a = None
    b_has_signal = (ss.spearman(loo_a, atr) or 0.0) >= 0.35
    if b_has_signal:
        rf_loo_a = ss.rf_loo_predict(Xtr, atr, seed=seed)
        rf = ss.fit_random_forest(Xtr, atr, seed=seed)
        if rf is not None:
            rf_va_a = rf.predict(Xva) if Xva.shape[0] else np.zeros(0)
            rf_ot_a = rf.predict(Xot) if Xot.shape[0] else np.zeros(0)

    # --- scale-prediction metrics per domain (constant-global vs ridge vs [rf]) ---
    def _scale_block(pred_a, pred_b, true_a, true_b):
        blk = {"global_a": ss.scale_metrics(np.full_like(true_a, a_g), true_a),
               "pred_a": ss.scale_metrics(pred_a, true_a),
               "pred_b": ss.scale_metrics(pred_b, true_b)}
        return blk
    scale = {
        "train_loo": _scale_block(loo_a, loo_b, atr, btr),
        "jax_val": _scale_block(pred_va[:, 0], pred_va[:, 1], ava, bva),
        "oma_test": _scale_block(pred_ot[:, 0], pred_ot[:, 1], aot, bot),
    }
    if rf_loo_a is not None:
        scale["train_loo"]["rf_a"] = ss.scale_metrics(rf_loo_a, atr)
    if rf_va_a is not None:
        scale["jax_val"]["rf_a"] = ss.scale_metrics(rf_va_a, ava)
    if rf_ot_a is not None:
        scale["oma_test"]["rf_a"] = ss.scale_metrics(rf_ot_a, aot)

    # --- three-way reconstruction (§15-§18) ---
    pred_map = {}
    for scv, pv in [(va_sc, pred_va), (ot_sc, pred_ot)]:
        for i, s in enumerate(scv):
            pred_map[s["id"]] = (float(pv[i, 0]), float(pv[i, 1]))

    def recon(scenes):
        return {
            "global": _pool_reconstruct(scenes, lambda s: (a_g, b_g)),
            "predicted": _pool_reconstruct(scenes, lambda s: pred_map.get(s["id"], (np.nan, np.nan))),
            "oracle": _pool_reconstruct(scenes, lambda s: (s["a_or"], s["b_or"])),
        }
    reconstruction = {"jax_val": recon(va_sc), "oma_test": recon(ot_sc)}

    # --- cross-city scale shift (§14) ---
    def _sstats(a):
        a = a[np.isfinite(a)]
        return (float(np.mean(a)), float(np.median(a)), float(np.std(a)), int(a.size)) if a.size \
            else (float("nan"),) * 3 + (0,)
    jm, jmd, jstd, jn = _sstats(atr)
    om, omd, ostd, on = _sstats(aot)
    scale_shift = {"jax_mean": jm, "jax_median": jmd, "jax_std": jstd, "jax_n": jn,
                   "oma_mean": om, "oma_median": omd, "oma_std": ostd, "oma_n": on,
                   "ratio": (jm / om) if (om and abs(om) > 1e-9) else float("nan")}

    classification = _classify(scale["jax_val"], scale["oma_test"],
                               reconstruction["oma_test"]["global"],
                               reconstruction["oma_test"]["predicted"],
                               reconstruction["oma_test"]["oracle"])

    # --- smallest next experiment justified by the evidence (§25 Q12) ---
    case = classification["case"]
    if case == "A":
        next = ("Integrate the frozen scale predictor as a per-scene affine head on top of the "
                "existing depth/features and re-measure on the held-out split (still no new city).")
    elif case == "B":
        next = ("Quantify how little TARGET-CITY calibration closes the gap: fit the SAME feature "
                "ridge with k in {1,2,4,8} labelled OMA scenes added, to see whether a few in-domain "
                "anchors restore predictability -- a domain-adaptation probe, not a redesign.")
    elif case == "C":
        next = ("Test whether a genuinely available metric cue (e.g. EXIF/GSD where present, or a "
                "known ground-sampling distance for the intended input type) correlates with the "
                "oracle scale -- a data-availability probe, kept separate from RGB/depth-only (§21).")
    else:
        next = ("Diagnose the residual after oracle scale: with the oracle affine applied, measure "
                "the remaining per-scene tall error vs depth compression (mean-depth plateau 15-30 m) "
                "to see what beyond a global slope+offset the tall tail needs (shape, not scale).")

    meta = {"source": source, "evidence_valid": evidence_valid,
            "depth_model": (cfg.depth.model_id if fake is None else "FAKE_STUB"),
            "depth_cache": cfg.depth.cache_dir, "cache_hits": cache_hits, "cache_misses": cache_misses,
            "n_train": len(tr_sc), "n_val": len(va_sc), "n_test": len(ot_sc), "dropped": dropped,
            "seed": seed, "alpha": alpha, "alpha_loo_mae": alpha_mae, "alpha_scores": alpha_scores,
            "leakage_ok_val": leakage_ok_val, "leakage_ok_oma": leakage_ok_oma,
            "elapsed_s": round(time.time() - t0, 1), "config": config_to_dict(cfg)}

    results = {"meta": meta, "target_stability": target_stability, "feature_screen": feature_screen,
               "global": {"a": a_g, "b": b_g}, "ridge": {"W": ridge["W"], "ymean": ridge["ymean"],
                                                         "mu": mu, "sd": sd, "alpha": alpha,
                                                         "feature_names": ss.FEATURE_NAMES},
               "scale": scale, "reconstruction": reconstruction, "scale_shift": scale_shift,
               "classification": classification, "next_experiment": next}
    results["answers"] = _build_answers(results)

    (out_dir / "results.json").write_text(
        json.dumps({k: v for k, v in results.items() if k != "ridge"} |
                   {"ridge_alpha": alpha, "feature_names": ss.FEATURE_NAMES},
                   indent=2, default=_json_default), encoding="utf-8")
    _write_report(results, str(out_dir / "SCENE_SCALE_DIAGNOSTIC.md"))

    packs = {"train_X": Xtr, "oma_X": Xot,
             "train_a": atr, "val_a": ava, "oma_a": aot,
             "train_loo_a": loo_a, "val_pred_a": pred_va[:, 0], "oma_pred_a": pred_ot[:, 0]}
    _figures(fig_dir, results, packs, evidence_valid)

    # Observation-10 fix: only mutate the canonical record with REAL evidence.
    if evidence_valid:
        _append_experiment_results(results)
    else:
        print("[results] evidence_valid=False -> NOT appending to EXPERIMENT_RESULTS.md (smoke).")

    # --- console summary + §28 STOP ---
    s = classification["signals"]
    print("\n" + "=" * 78)
    print("SCENE-SCALE DIAGNOSTIC (Exp #6) -- is per-scene metric scale inferable at inference?")
    print(f"  scale predictability (pred vs oracle a, Spearman): JAX-val="
          f"{_fmt(s['jaxval_scale_spearman'])} OMA={_fmt(s['oma_scale_spearman'])} "
          f"(OMA rel.err {_fmt(s['oma_scale_rel_median'],3)})")
    go = reconstruction["oma_test"]["global"]; po = reconstruction["oma_test"]["predicted"]
    oo = reconstruction["oma_test"]["oracle"]
    print(f"  OMA building MAE: global={_fmt(go['building']['mae'],2)} "
          f"pred={_fmt(po['building']['mae'],2)} oracle={_fmt(oo['building']['mae'],2)}")
    print(f"  OMA >15m MAE: global={_fmt(go['tall_15']['mae'],2)} "
          f"pred={_fmt(po['tall_15']['mae'],2)} oracle={_fmt(oo['tall_15']['mae'],2)}")
    print(f"  PROPOSED CASE {classification['case']}: {classification['one_line']}")
    if not evidence_valid:
        print("\n[!] evidence_valid=False -- plumbing only; NOT a real result.")
    print("=" * 78)
    print("STOP (§28): diagnostic complete. No training/architecture/product launched. Await review.")


if __name__ == "__main__":
    main()
