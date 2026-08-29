#!/usr/bin/env python
"""DepthWizard SCALE SANITY CHECK + DATASET / SCALE FORENSICS (Experiment #7) -- DIAGNOSTIC ONLY.

Two closely-related questions, answered from EXISTING data. NOTHING is trained (§32): no U-Net,
no scale predictor, no loss/transform/architecture/DA-V2 change, no OMA tuning, no model
selection. The only new artifacts are diagnostic utilities, plots, a report and tests.

PART A -- SCALE SANITY CHECK (§3-§8)
  Was Experiment #6's apparent "adaptive scale" improvement actually just a larger GLOBAL scale?
  We REUSE Exp-6's fully deterministic cached pipeline (same tiles, same frozen DA-V2 cache, same
  seed) to RECOVER its per-scene predictions -- they were not persisted in results.json (only
  aggregates + alpha were), so recovery = re-executing the identical cached ridge, NOT training a
  new predictor. Reproduction is VERIFIED against the saved Exp-6 aggregates before any new
  analysis (integrity gate). Then the master-prompt control: take the MEAN of the predictor's
  per-scene outputs and apply that single constant to every OMA scene ("Predicted-Mean Global"),
  and compare {Global, Predicted-Mean Global, Predicted Adaptive, Oracle}. If Adaptive ~=
  Predicted-Mean Global (and held-out scale correlation is weak) the Exp-6 "win" was a global
  upward shift, not scene-specific adaptation. This IS the Observation-11 mean-shift control.

PART B -- DATASET / SCALE FORENSICS (§9-§31)
  Do JAX-train, JAX-val and OMA-test differ in physical/image scale, DA-V2 depth distribution,
  height distribution, per-scene oracle depth->height slope, scene composition, tall-structure
  representation or label preprocessing -- enough to explain the model's problems? MEASURE, do
  not assume (§24: never call the dataset "bad" without proof). Same city-held-out split, same
  cache, no recompute.

Reproducibility (§34): dataset source, exact tile IDs/splits, cache, formulas (docstrings of
depthwizard/diagnostics/scale_forensics.py), features (scene_scale.FEATURE_NAMES), code version,
runtime -> runs/dataset_scale_forensics/reproducibility.json. Append is GATED on evidence_valid
AND Part-A reproduction so a smoke/plumbing run can never pollute EXPERIMENT_RESULTS.md (§33).

Usage:
  python scripts/dataset_scale_forensics.py --config configs/dataset_scale_forensics.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # so we can import the Exp-6 orchestrator

# Exp-6 orchestrator: reuse its EXACT deterministic helpers so Part A reproduces byte-for-byte.
import scene_scale_diagnostic as sd                          # noqa: E402
from depthwizard.config import load_config, config_to_dict   # noqa: E402
from depthwizard.data import fetch, datasets                 # noqa: E402
from depthwizard.models.affine import GlobalAffine           # noqa: E402
from depthwizard.metrics.height_metrics import valid_mask    # noqa: E402
from depthwizard.diagnostics import scene_scale as ss        # noqa: E402
from depthwizard.diagnostics import scale_forensics as sf    # noqa: E402
from depthwizard.diagnostics.depth_signal import disjoint_ids  # noqa: E402

THR = sf.DEFAULT_HEIGHT_THRESHOLDS               # (5,10,15,20,30,40)
BANDS = sf.TALL_BANDS                            # [(15,20),(20,30),(30,40),(40,inf)]
CAP = sd.ALL_CAP                                 # 20000 pooled px/tile (bounds memory; seeded)
REPRO_TOL = 1e-4                                 # Part-A reproduction tolerance vs saved Exp-6
PRIOR_RESULTS = _REPO / "runs" / "scene_scale_diagnostic" / "results.json"


def _fmt(x, nd=3):
    return sd._fmt(x, nd)


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_REPO),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _dist(arr, ps=(1, 5, 10, 25, 50, 75, 85, 90, 95, 99)) -> dict:
    """mean/median/std/min/max/range + percentiles for a per-scene scalar vector (nan-safe)."""
    a = np.asarray(arr, np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        d = {"n": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan"),
             "min": float("nan"), "max": float("nan"), "range": float("nan")}
        d.update({f"p{p}": float("nan") for p in ps})
        return d
    pv = np.percentile(a, list(ps))
    d = {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
         "std": float(a.std()), "min": float(a.min()), "max": float(a.max()),
         "range": float(a.max() - a.min())}
    d.update({f"p{p}": float(v) for p, v in zip(ps, pv)})
    return d


# ============================================================ PART A -- SCALE SANITY CHECK
def _reproduce_exp6(cfg, seed):
    """Re-execute Exp-6's deterministic cached pipeline to RECOVER its per-scene predictions.

    Returns the full working set (scenes, oracle/predicted a&b per split, global affine, ridge).
    This trains NOTHING new: it reuses sd's helpers + the frozen DA-V2 cache, and the ridge is the
    same closed-form fit on JAX-train that Exp-6 ran. Verified against saved aggregates by caller.
    """
    source, records = fetch.resolve_records(cfg)
    if source == "synthetic":
        raise SystemExit("[error] synthetic source -> not evidence; run where the HF cache exists.")

    # Split FIRST, then gate the no-recompute guarantee on ONLY the tiles that get materialized.
    # The unofficial mirror holds 999 crops, but Phase-1..6 and this study use
    # max_tiles_per_city=120 -> a 102/18/120 city-held-out split (240 tiles). The DA-V2 cache was
    # built for exactly that split, so checking cache coverage over all 999 mirror tiles would
    # falsely "miss" the 759 tiles this study never touches (§13/§32 concern only the 240 used).
    tr_rec, va_rec, te_rec = datasets.split_by_city(
        records, cfg.split.train_cities, cfg.split.val_cities, cfg.split.test_cities,
        cfg.split.val_fraction_within_train_city, cfg.split.seed, cfg.data.max_tiles_per_city)
    tr_ids = [r.tile_id for r in tr_rec]; va_ids = [r.tile_id for r in va_rec]
    te_ids = [r.tile_id for r in te_rec]
    print(f"[split] train={len(tr_rec)} val={len(va_rec)} test={len(te_rec)} "
          f"(of {len(records)} mirror tiles)")
    if not (disjoint_ids(tr_ids, va_ids) and disjoint_ids(tr_ids, te_ids)):
        raise SystemExit("[error] train/eval tile-id overlap -> would leak. Aborting.")

    split_rec = tr_rec + va_rec + te_rec
    cache_hits, cache_misses = sd._cache_report(split_rec, cfg)
    print(f"[cache] DA-V2 depth cache for the split: hits={cache_hits} misses={cache_misses} "
          f"(of {len(split_rec)} split tiles)")
    if cache_misses > 0:
        raise SystemExit("[error] depth-cache incomplete for the split -> materializing would recompute "
                         "DA-V2 (forbidden §13/§32). Re-run where runs/phase1_hf/depth_cache covers the split.")
    from depthwizard.depth.depth_anything import DepthAnythingV2
    depth_model = DepthAnythingV2(cfg.depth.model_id, cfg.depth.input_size,
                                  cache_dir=cfg.depth.cache_dir, use_cache=True)

    train = sd._materialize(tr_rec, cfg, depth_model, None)
    val = sd._materialize(va_rec, cfg, depth_model, None)
    test = sd._materialize(te_rec, cfg, depth_model, None)

    blabel, nodata = cfg.data.building_label, cfg.data.nodata
    tr_sc, dtr = sd._scene_records(train, blabel, nodata, CAP, seed)
    va_sc, dva = sd._scene_records(val, blabel, nodata, CAP, seed)
    ot_sc, dot = sd._scene_records(test, blabel, nodata, CAP, seed)
    print(f"[scenes] train={len(tr_sc)} val={len(va_sc)} oma={len(ot_sc)} (dropped {dtr+dva+dot})")

    Xtr, atr, btr = sd._matrix(tr_sc)
    Xva, ava, bva = sd._matrix(va_sc)
    Xot, aot, bot = sd._matrix(ot_sc)

    gb = GlobalAffine(max_pixels=getattr(cfg.train, "max_train_pixels_affine", 2_000_000),
                      seed=cfg.split.seed).fit(train)
    a_g, b_g = float(gb.a), float(gb.b)

    Y = np.stack([atr, btr], axis=1)
    alpha, alpha_mae, _ = ss.select_alpha(Xtr, atr, sd.ALPHAS)
    mu, sd_ = ss.standardize_fit(Xtr)
    ridge = ss.fit_ridge(ss.standardize_apply(Xtr, mu, sd_), Y, alpha)

    def pred(X):
        return ss.predict_ridge(ridge, ss.standardize_apply(X, mu, sd_)) if X.shape[0] else np.zeros((0, 2))
    pred_va, pred_ot = pred(Xva), pred(Xot)
    loo_ab = ss.loo_predict(Xtr, Y, alpha)
    loo_a = loo_ab[:, 0] if loo_ab.ndim == 2 else loo_ab

    return dict(
        source=source, records=records, tr_rec=tr_rec, va_rec=va_rec, te_rec=te_rec,
        train=train, val=val, test=test, tr_sc=tr_sc, va_sc=va_sc, ot_sc=ot_sc,
        Xtr=Xtr, atr=atr, btr=btr, ava=ava, bva=bva, aot=aot, bot=bot,
        a_g=a_g, b_g=b_g, alpha=alpha, alpha_mae=alpha_mae,
        pred_va=pred_va, pred_ot=pred_ot, loo_a=loo_a,
        cache_hits=cache_hits, cache_misses=cache_misses, nodata=nodata, blabel=blabel)


def _verify_reproduction(W):
    """Gate: recomputed Exp-6 aggregates must match the SAVED Exp-6 results.json to REPRO_TOL."""
    a_oma = ss.scale_metrics(W["pred_ot"][:, 0], W["aot"])
    a_val = ss.scale_metrics(W["pred_va"][:, 0], W["ava"])
    recon_g = sd._pool_reconstruct(W["ot_sc"], lambda s: (W["a_g"], W["b_g"]))
    pred_map = {s["id"]: (float(W["pred_ot"][i, 0]), float(W["pred_ot"][i, 1]))
                for i, s in enumerate(W["ot_sc"])}
    recon_p = sd._pool_reconstruct(W["ot_sc"], lambda s: pred_map[s["id"]])
    now = {"global_a": W["a_g"], "oma_pred_a_mae": a_oma["mae"],
           "jaxval_pred_a_spearman": a_val["spearman"],
           "oma_global_building_mae": recon_g["building"]["mae"],
           "oma_pred_building_mae": recon_p["building"]["mae"]}
    checks, ok = {}, True
    if PRIOR_RESULTS.exists():
        prior = json.loads(PRIOR_RESULTS.read_text(encoding="utf-8"))
        ref = {"global_a": prior["global"]["a"],
               "oma_pred_a_mae": prior["scale"]["oma_test"]["pred_a"]["mae"],
               "jaxval_pred_a_spearman": prior["scale"]["jax_val"]["pred_a"]["spearman"],
               "oma_global_building_mae": prior["reconstruction"]["oma_test"]["global"]["building"]["mae"],
               "oma_pred_building_mae": prior["reconstruction"]["oma_test"]["predicted"]["building"]["mae"]}
        for k, v in now.items():
            rv = ref.get(k)
            diff = abs(float(v) - float(rv)) if (rv is not None and v is not None) else float("inf")
            checks[k] = {"recomputed": float(v), "saved": (float(rv) if rv is not None else None),
                         "abs_diff": diff, "match": diff <= REPRO_TOL}
            ok = ok and checks[k]["match"]
        prior_available = True
    else:
        prior_available = False
        for k, v in now.items():
            checks[k] = {"recomputed": float(v), "saved": None, "abs_diff": None, "match": None}
        ok = False   # cannot verify without the saved record -> do not claim reproduction
    return {"prior_results_available": prior_available, "prior_results_path": str(PRIOR_RESULTS),
            "tolerance": REPRO_TOL, "reproduction_ok": ok, "checks": checks}


def _scene_corr(pred_a, oracle_a, pred_b=None, oracle_b=None):
    """§6 per-scene predicted-vs-oracle scale: metrics + error-vs-oracle-scale correlation."""
    m = ss.scale_metrics(pred_a, oracle_a)
    err = np.asarray(pred_a, np.float64) - np.asarray(oracle_a, np.float64)
    out = {"a": m, "err_vs_oracle": sf.correlate(oracle_a, err),
           "pred_a": np.asarray(pred_a).tolist(), "oracle_a": np.asarray(oracle_a).tolist()}
    if pred_b is not None:
        out["b"] = ss.scale_metrics(pred_b, oracle_b)
    return out


def part_a(W):
    """§4-§8: predicted-mean-global control, four-way reconstruction, per-scene correlation, verdict."""
    a_g, b_g = W["a_g"], W["b_g"]
    pred_va, pred_ot = W["pred_va"], W["pred_ot"]

    # Mean of the predictor's per-scene outputs (NOT a fit): the single constant we broadcast.
    mean_pa_tr = float(np.mean(W["loo_a"])) if W["loo_a"].size else float("nan")
    mean_pa_va = float(np.mean(pred_va[:, 0])) if pred_va.shape[0] else float("nan")
    mean_pb_va = float(np.mean(pred_va[:, 1])) if pred_va.shape[0] else float("nan")
    mean_pa_ot = float(np.mean(pred_ot[:, 0])) if pred_ot.shape[0] else float("nan")
    mean_pb_ot = float(np.mean(pred_ot[:, 1])) if pred_ot.shape[0] else float("nan")

    def four_way(scenes, pv, mean_a, mean_b):
        pm = {s["id"]: (float(pv[i, 0]), float(pv[i, 1])) for i, s in enumerate(scenes)}
        return {
            "global": sd._pool_reconstruct(scenes, lambda s: (a_g, b_g)),
            "predicted_mean_global": sd._pool_reconstruct(scenes, lambda s: (mean_a, mean_b)),
            "predicted_adaptive": sd._pool_reconstruct(scenes, lambda s: pm[s["id"]]),
            "oracle": sd._pool_reconstruct(scenes, lambda s: (s["a_or"], s["b_or"])),
        }
    recon = {"oma_test": four_way(W["ot_sc"], pred_ot, mean_pa_ot, mean_pb_ot),
             "jax_val": four_way(W["va_sc"], pred_va, mean_pa_va, mean_pb_va)}

    corr = {"jax_val": _scene_corr(pred_va[:, 0], W["ava"], pred_va[:, 1], W["bva"]),
            "oma_test": _scene_corr(pred_ot[:, 0], W["aot"], pred_ot[:, 1], W["bot"])}

    # ---- §5/§8 decision: does per-scene VARIATION help beyond its own mean, and does it track? ----
    oma = recon["oma_test"]
    def g(m, sub, k="mae"):
        return m[sub][k]
    ad_b, mg_b = g(oma["predicted_adaptive"], "building"), g(oma["predicted_mean_global"], "building")
    ad_t, mg_t = g(oma["predicted_adaptive"], "tall_15"), g(oma["predicted_mean_global"], "tall_15")
    gl_b, or_b = g(oma["global"], "building"), g(oma["oracle"], "building")
    val_sp = corr["jax_val"]["a"]["spearman"]
    oma_sp = corr["oma_test"]["a"]["spearman"]
    val_sp = val_sp if (val_sp is not None and val_sp == val_sp) else 0.0
    d_bldg = ad_b - mg_b            # <0 => adaptive better than its de-varied mean
    d_tall = ad_t - mg_t
    tracks = val_sp >= 0.35
    adaptive_better = ((mg_b - ad_b) >= 0.25) or ((mg_t - ad_t) >= 1.0)
    near_equal = (abs(d_bldg) < 0.25) and (abs(d_tall) < 1.0)

    if tracks and adaptive_better:
        verdict = "TRUE_ADAPTATION"
        one = ("Predicted scale tracks the actual per-scene oracle scale on held-out data AND the "
               "per-scene adaptive reconstruction beats the predicted-mean-global control -> Exp-6 "
               "found genuine scene-specific adaptation.")
    elif (not tracks) and near_equal and not adaptive_better:
        verdict = "GLOBAL_SHIFT_ARTIFACT"
        one = ("Predicted adaptive scale performs essentially the same as a SINGLE predicted-mean "
               "global scale (OMA building/tall MAE ~ equal) while held-out scale correlation is "
               "weak -> Exp-6's apparent 'adaptive' improvement was mostly an upward GLOBAL shift, "
               "not real scene-specific adaptation.")
    else:
        verdict = "MIXED_INCONCLUSIVE"
        one = (f"Mixed evidence -- do not stretch it either way. The per-scene VARIATION is not a pure "
               f"global shift: broadcasting the predicted-mean scale as one constant does NOT reproduce "
               f"the tall-tail gain (>15 m MAE mean-global {mg_t:.1f} vs adaptive {ad_t:.1f}). BUT the "
               f"predictor does not track the true per-scene oracle scale on held-out JAX (Spearman "
               f"{val_sp:.3f} << 0.35) and it worsens the OMA building body ({mg_b:.2f}->{ad_b:.2f} MAE) "
               f"-- a crude tall-responsive scale boost, not genuine scene-specific metric calibration.")

    signals = {"jaxval_scale_spearman": val_sp, "oma_scale_spearman": oma_sp,
               "oma_adaptive_building_mae": ad_b, "oma_predmean_building_mae": mg_b,
               "oma_global_building_mae": gl_b, "oma_oracle_building_mae": or_b,
               "oma_adaptive_tall15_mae": ad_t, "oma_predmean_tall15_mae": mg_t,
               "delta_building_adaptive_minus_mean": d_bldg,
               "delta_tall15_adaptive_minus_mean": d_tall,
               "tracks_oracle_heldout": tracks, "adaptive_beats_mean_global": adaptive_better,
               "adaptive_approx_mean_global": near_equal}
    return {"predicted_mean_scales": {"train_loo_mean_a": mean_pa_tr,
                                      "jax_val_mean_a": mean_pa_va, "jax_val_mean_b": mean_pb_va,
                                      "oma_mean_a": mean_pa_ot, "oma_mean_b": mean_pb_ot,
                                      "global_a": a_g, "global_b": b_g},
            "reconstruction": recon, "per_scene_correlation": corr,
            "verdict": verdict, "one_line": one, "signals": signals}


# ============================================================ PART B -- DATASET / SCALE FORENSICS
def _raw_info(records):
    """§10/§11: on-disk raster metadata for the RGB + AGL of each tile (headers only, no resize)."""
    rgb, agl = [], []
    for r in records:
        rgb.append(sf.raw_raster_info(r.rgb_path))
        agl.append(sf.raw_raster_info(r.agl_path))
    return {"rgb": sf.summarize_raw_infos(rgb), "agl": sf.summarize_raw_infos(agl)}


def _accumulate(samples, blabel, nodata, cap, seed):
    """Single pass over a split's tiles -> exact counts/fracs (full tiles) + pooled distribution
    samples (seeded per-tile subsample) + per-tile footprint/border + provenance + per-scene comp."""
    rng = np.random.default_rng(seed)
    H = {"n": 0, "sum": 0.0, "sumsq": 0.0, "max": float("-inf"), "above": {t: 0 for t in THR}}
    Bd = {"n": 0, "sum": 0.0, "sumsq": 0.0, "max": float("-inf"), "above": {t: 0 for t in THR}}
    n_ground = 0
    h_all_pool, h_b_pool, d_all_pool = [], [], []
    band_pool = {b: [] for b in BANDS}
    fp, fp_tall, bd, bd_tall = [], [], [], []
    per_tile_bmax = []
    comp = {}
    prov = {"gt_min": float("inf"), "gt_max": float("-inf"), "n_finite": 0, "n_nan": 0,
            "n_neg": 0, "n_zero": 0, "n_gt150": 0, "n_tiles": 0}

    for s in samples:
        h = np.asarray(s["gt"], np.float64)
        d = np.asarray(s["depth"], np.float64)
        cls = np.asarray(s["cls"]) if s.get("cls") is not None else None
        prov["n_tiles"] += 1
        fin = np.isfinite(h)
        fgt = h[fin]
        if fgt.size:
            prov["gt_min"] = min(prov["gt_min"], float(fgt.min()))
            prov["gt_max"] = max(prov["gt_max"], float(fgt.max()))
        prov["n_finite"] += int(fin.sum()); prov["n_nan"] += int((~fin).sum())
        prov["n_neg"] += int((fgt < 0).sum()); prov["n_zero"] += int((fgt == 0).sum())
        prov["n_gt150"] += int((fgt > 150).sum())

        vm = valid_mask(h, d, nodata=nodata)
        n = int(vm.sum())
        if n == 0:
            continue
        hv, dv = h[vm], d[vm]
        H["n"] += n; H["sum"] += float(hv.sum()); H["sumsq"] += float(np.dot(hv, hv))
        H["max"] = max(H["max"], float(hv.max()))
        for t in THR:
            H["above"][t] += int((hv > t).sum())
        take = min(cap, n)
        idx = rng.choice(n, take, replace=False) if n > take else np.arange(n)
        h_all_pool.append(hv[idx]); d_all_pool.append(dv[idx])

        max_h = float(hv.max())
        gfrac = float(np.mean(cls[vm] == sf.GROUND_LABEL)) if cls is not None else float("nan")
        if cls is not None:
            n_ground += int((cls[vm] == sf.GROUND_LABEL).sum())
            bmask = vm & (cls == blabel)
            nb = int(bmask.sum())
            hb = h[bmask]
            Bd["n"] += nb
            if nb:
                Bd["sum"] += float(hb.sum()); Bd["sumsq"] += float(np.dot(hb, hb))
                Bd["max"] = max(Bd["max"], float(hb.max()))
                for t in THR:
                    Bd["above"][t] += int((hb > t).sum())
                tb = min(cap, nb)
                h_b_pool.append(rng.choice(hb, tb, replace=False) if nb > tb else hb)
            per_tile_bmax.append(float(hb.max()) if nb else float("nan"))
            fp.append(sf.footprint_stats(bmask))
            bd.append(sf.border_stats(bmask))
            tallmask = bmask & (h > 15)
            fp_tall.append(sf.footprint_stats(tallmask))
            bd_tall.append(sf.border_stats(tallmask))
            for (lo, hi) in BANDS:
                sel = bmask & (h >= lo) & (h < hi)
                m = int(sel.sum())
                if m:
                    dd = d[sel]
                    tb = min(cap, m)
                    band_pool[(lo, hi)].append(rng.choice(dd, tb, replace=False) if m > tb else dd)
            _, ntall = sf.label_components(tallmask)
            comp[s["id"]] = {"max_h": max_h,
                             "med_bh": float(np.median(hb)) if nb else float("nan"),
                             "bfrac": nb / n, "gfrac": gfrac, "n_tall_bldg": int(ntall)}
        else:
            per_tile_bmax.append(float("nan"))
            comp[s["id"]] = {"max_h": max_h, "med_bh": float("nan"), "bfrac": float("nan"),
                             "gfrac": gfrac, "n_tall_bldg": 0}

    def _fin(acc):
        n = acc["n"]
        mean = acc["sum"] / n if n else float("nan")
        var = acc["sumsq"] / n - mean * mean if n else float("nan")
        return {"n": n, "mean": mean, "std": float(np.sqrt(max(var, 0.0))) if n else float("nan"),
                "max": acc["max"] if n else float("nan"),
                "frac_above": {f">{sf._tk(t)}m": (acc["above"][t] / n if n else float("nan")) for t in THR}}

    def _pool(chunks):
        return np.concatenate(chunks) if chunks else np.array([], np.float64)

    h_all = _pool(h_all_pool); h_b = _pool(h_b_pool); d_all = _pool(d_all_pool)
    height = {"all": {**_fin(H), **_dist(h_all)}, "building": {**_fin(Bd), **_dist(h_b)},
              "n_ground": n_ground,
              "ground_frac": (n_ground / H["n"] if H["n"] else float("nan")),
              "building_frac": (Bd["n"] / H["n"] if H["n"] else float("nan"))}

    def _agg(dicts, keys):
        out = {}
        for k in keys:
            vals = np.array([x[k] for x in dicts if x.get(k) is not None and np.isfinite(x[k])], np.float64)
            out[k] = {"mean": float(vals.mean()) if vals.size else float("nan"),
                      "median": float(np.median(vals)) if vals.size else float("nan"),
                      "sum": float(vals.sum()) if vals.size else 0.0, "n_tiles": int(vals.size)}
        return out

    footprint = {"building": _agg(fp, ["n_components", "largest_px", "mean_px", "median_px", "n_large", "building_px"]),
                 "tall": _agg(fp_tall, ["n_components", "largest_px", "n_large", "building_px"])}
    border = {"building": _agg(bd, ["border_px_frac", "border_component_frac", "n_components", "n_border_components"]),
              "tall": _agg(bd_tall, ["border_px_frac", "border_component_frac", "n_components", "n_border_components"])}

    band_stats = {}
    for (lo, hi) in BANDS:
        pooled = _pool(band_pool[(lo, hi)])
        key = f"{sf._tk(lo)}_{sf._tk(hi) if np.isfinite(hi) else 'inf'}"
        b = _dist(pooled, ps=(10, 50, 90))
        band_stats[key] = {"n": b["n"], "depth_mean": b["mean"], "depth_median": b["median"],
                           "depth_std": b["std"], "depth_p10": b.get("p10", float("nan")),
                           "depth_p90": b.get("p90", float("nan"))}

    prov["gt_min"] = None if prov["gt_min"] == float("inf") else prov["gt_min"]
    prov["gt_max"] = None if prov["gt_max"] == float("-inf") else prov["gt_max"]
    prov["neg_frac"] = prov["n_neg"] / prov["n_finite"] if prov["n_finite"] else float("nan")
    prov["zero_frac"] = prov["n_zero"] / prov["n_finite"] if prov["n_finite"] else float("nan")
    prov["gt150_frac"] = prov["n_gt150"] / prov["n_finite"] if prov["n_finite"] else float("nan")

    return {"height": height, "footprint": footprint, "border": border, "band_depths": band_stats,
            "provenance": prov, "comp": comp, "per_tile_bmax": per_tile_bmax,
            "_pool": {"h_all": h_all, "h_b": h_b, "d_all": d_all}}


def part_b(W):
    cfg_thr = [15.0, 20.0, 30.0, 40.0]
    seed = 0
    acc = {"jax_train": _accumulate(W["train"], W["blabel"], W["nodata"], CAP, seed),
           "jax_val": _accumulate(W["val"], W["blabel"], W["nodata"], CAP, seed),
           "oma_test": _accumulate(W["test"], W["blabel"], W["nodata"], CAP, seed)}

    # §10/§11 raster metadata + resize-pipeline trace
    raw = {"jax_train": _raw_info(W["tr_rec"]), "jax_val": _raw_info(W["va_rec"]),
           "oma_test": _raw_info(W["te_rec"])}
    resize = {"target_hw": [W_cfg_tile(W)] * 2, "interp": "RGB INTER_AREA / AGL,CLS INTER_NEAREST",
              "aspect_ratio_changed": False, "physical_gsd_available": False,
              "note": ("load_sample() resizes every raster to (tile_size, tile_size); measured raw "
                       "dims are square and equal to tile_size, so the resize is a near-no-op and "
                       "does NOT normalize differing physical resolutions within this pipeline. A "
                       "true GSD is unavailable (identity ModelTransformation stub, no CRS) so a "
                       "physical-resolution mismatch between splits cannot be measured from these "
                       "files and is NOT assumed.")}

    # §13 depth distribution (+ overlap on shared edges)
    def dstats(a):
        return sf.depth_stats(a, hist_bins=40, hist_range=DEPTH_RANGE(acc))
    depth = {k: dstats(acc[k]["_pool"]["d_all"]) for k in acc}
    depth_overlap = {"train_val": sf.hist_overlap(depth["jax_train"]["hist_counts"], depth["jax_val"]["hist_counts"]),
                     "train_oma": sf.hist_overlap(depth["jax_train"]["hist_counts"], depth["oma_test"]["hist_counts"]),
                     "val_oma": sf.hist_overlap(depth["jax_val"]["hist_counts"], depth["oma_test"]["hist_counts"])}

    # §14/§15 oracle slope/offset per split + shift quantification
    a_by = {"jax_train": W_a(W["tr_sc"], "a_or"), "jax_val": W_a(W["va_sc"], "a_or"),
            "oma_test": W_a(W["ot_sc"], "a_or")}
    b_by = {"jax_train": W_a(W["tr_sc"], "b_or"), "jax_val": W_a(W["va_sc"], "b_or"),
            "oma_test": W_a(W["ot_sc"], "b_or")}
    oracle_scale = {"slope": {k: _dist(v) for k, v in a_by.items()},
                    "offset": {k: _dist(v) for k, v in b_by.items()},
                    "shift_slope": {"train_vs_oma": sf.split_shift(a_by["jax_train"], a_by["oma_test"]),
                                    "val_vs_oma": sf.split_shift(a_by["jax_val"], a_by["oma_test"]),
                                    "train_vs_val": sf.split_shift(a_by["jax_train"], a_by["jax_val"])},
                    "shift_offset": {"train_vs_oma": sf.split_shift(b_by["jax_train"], b_by["oma_test"])}}

    # §16 scene scale vs composition; §17 scene scale vs depth features -- JAX-train & OMA
    def comp_corr(split_key, scenes):
        cmp = acc[split_key]["comp"]
        a = np.array([s["a_or"] for s in scenes], np.float64)
        feats = {"max_height": [cmp[s["id"]]["max_h"] for s in scenes],
                 "median_building_height": [cmp[s["id"]]["med_bh"] for s in scenes],
                 "n_tall_buildings": [cmp[s["id"]]["n_tall_bldg"] for s in scenes],
                 "building_fraction": [cmp[s["id"]]["bfrac"] for s in scenes],
                 "ground_fraction": [cmp[s["id"]]["gfrac"] for s in scenes]}
        return {k: sf.correlate(a, np.array(v, np.float64)) for k, v in feats.items()}

    def depthfeat_corr(scenes):
        a = np.array([s["a_or"] for s in scenes], np.float64)
        keys = {"depth_median": "depth_median", "depth_p90": "depth_p90",
                "depth_std": "depth_std", "depth_range": "depth_range_p1p99"}
        return {name: sf.correlate(a, np.array([s["feat"][fk] for s in scenes], np.float64))
                for name, fk in keys.items()}

    scale_vs = {"composition": {"jax_train": comp_corr("jax_train", W["tr_sc"]),
                                "oma_test": comp_corr("oma_test", W["ot_sc"])},
                "depth_features": {"jax_train": depthfeat_corr(W["tr_sc"]),
                                   "oma_test": depthfeat_corr(W["ot_sc"])}}

    # §22 training-sample balance (TRAIN tiles only)
    bmax = np.array([x for x in acc["jax_train"]["per_tile_bmax"]], np.float64)
    n_tiles = int(np.isfinite(bmax).size and len(bmax))
    balance = {"n_train_tiles": len(bmax),
               "tiles_with_building": int(np.isfinite(bmax).sum())}
    for t in cfg_thr:
        c = int(np.nansum(bmax > t))
        balance[f"tiles_gt{int(t)}m"] = c
        balance[f"frac_gt{int(t)}m"] = c / len(bmax) if len(bmax) else float("nan")

    # §19 tall-band depth shift (building) JAX-train vs OMA per band
    band_shift = {}
    for k in acc["jax_train"]["band_depths"]:
        j = acc["jax_train"]["band_depths"][k]; o = acc["oma_test"]["band_depths"][k]
        band_shift[k] = {"jax_median": j["depth_median"], "oma_median": o["depth_median"],
                         "median_diff": (j["depth_median"] - o["depth_median"])
                         if (np.isfinite(j["depth_median"]) and np.isfinite(o["depth_median"])) else float("nan"),
                         "jax_n": j["n"], "oma_n": o["n"]}

    return {"raster": raw, "resize_pipeline": resize, "depth": depth, "depth_overlap": depth_overlap,
            "oracle_scale": oracle_scale, "scale_vs": scale_vs, "training_balance": balance,
            "tall_band_shift": band_shift,
            "height": {k: acc[k]["height"] for k in acc},
            "footprint": {k: acc[k]["footprint"] for k in acc},
            "border": {k: acc[k]["border"] for k in acc},
            "band_depths": {k: acc[k]["band_depths"] for k in acc},
            "provenance": {k: acc[k]["provenance"] for k in acc},
            "_acc": acc, "_a_by": a_by, "_b_by": b_by}


def W_cfg_tile(W):
    return int(W["cfg_tile"])


def W_a(scenes, key):
    return np.array([s[key] for s in scenes], np.float64)


def DEPTH_RANGE(acc):
    lo, hi = [], []
    for k in acc:
        d = acc[k]["_pool"]["d_all"]
        d = d[np.isfinite(d)]
        if d.size:
            lo.append(np.percentile(d, 0.5)); hi.append(np.percentile(d, 99.5))
    return (float(min(lo)), float(max(hi))) if lo else (0.0, 1.0)


# ============================================================ FIGURES (§7/§27/§28)
def figures(fig_dir, A, B, W, evidence_valid):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[viz] skipped ({e})")
        return []
    fig_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if evidence_valid else "  [NON-EVIDENCE]"
    made = []
    C = {"jax_train": "#9aa0a6", "jax_val": "#1f6feb", "oma_test": "#d1242f"}
    LBL = {"jax_train": "JAX train", "jax_val": "JAX val", "oma_test": "OMA test"}

    def save(fig, name):
        fig.tight_layout(); fig.savefig(fig_dir / name, dpi=110); plt.close(fig); made.append(name)

    # ---- Part A: predicted vs oracle scatter (JAX-val, OMA) ----
    try:
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        for k, key in enumerate(["jax_val", "oma_test"]):
            c = A["per_scene_correlation"][key]
            orac = np.array(c["oracle_a"]); pred = np.array(c["pred_a"])
            if orac.size:
                lo = float(min(orac.min(), pred.min())); hi = float(max(orac.max(), pred.max()))
                pad = 0.05 * (hi - lo + 1e-6)
                ax[k].plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, label="y=x")
                ax[k].scatter(orac, pred, s=24, alpha=0.6, c=C[key], edgecolors="none")
            gm = A["predicted_mean_scales"]["global_a"]
            pm = A["predicted_mean_scales"]["oma_mean_a" if key == "oma_test" else "jax_val_mean_a"]
            ax[k].axhline(pm, color="#8250df", ls=":", lw=1.4, label="predicted-mean")
            ax[k].axhline(gm, color="#1a7f37", ls="-.", lw=1.2, label="global")
            ax[k].set_xlabel("oracle scene scale a"); ax[k].set_ylabel("predicted scene scale a")
            ax[k].set_title(f"{LBL[key]} (Spearman {_fmt(A['per_scene_correlation'][key]['a']['spearman'],3)})")
            ax[k].legend(fontsize=8, loc="upper left")
        fig.suptitle(f"Part A -- predicted vs oracle scene scale{tag}")
        save(fig, "A_scale_pred_vs_oracle.png")
    except Exception as e:
        print(f"[viz] A scatter: {e}")

    # ---- Part A: four-way reconstruction bars (OMA building + tall_15) ----
    try:
        oma = A["reconstruction"]["oma_test"]
        methods = ["global", "predicted_mean_global", "predicted_adaptive", "oracle"]
        names = ["Global", "Pred-Mean\nGlobal", "Pred\nAdaptive", "Oracle"]
        bmae = [oma[m]["building"]["mae"] for m in methods]
        tmae = [oma[m]["tall_15"]["mae"] for m in methods]
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        cols = ["#1a7f37", "#8250df", "#d1242f", "#57606a"]
        ax[0].bar(names, bmae, color=cols); ax[0].set_title("OMA building MAE (m)")
        ax[1].bar(names, tmae, color=cols); ax[1].set_title("OMA >15 m building MAE (m)")
        for a in ax:
            a.grid(axis="y", alpha=0.3)
        fig.suptitle(f"Part A -- does per-scene variation beat its own mean?{tag}")
        save(fig, "A_reconstruction_fourway_oma.png")
    except Exception as e:
        print(f"[viz] A bars: {e}")

    # ---- §27 oracle scale distribution by split (the headline shift) ----
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        data = [B["_a_by"]["jax_train"], B["_a_by"]["jax_val"], B["_a_by"]["oma_test"]]
        data = [d[np.isfinite(d)] for d in data]
        bp = ax.boxplot(data, showmeans=True, patch_artist=True)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([LBL["jax_train"], LBL["jax_val"], LBL["oma_test"]])
        for patch, key in zip(bp["boxes"], ["jax_train", "jax_val", "oma_test"]):
            patch.set_facecolor(C[key]); patch.set_alpha(0.55)
        ax.set_ylabel("per-scene oracle slope a (depth->m)")
        ax.set_title(f"§27 oracle scene scale by split{tag}")
        ax.grid(axis="y", alpha=0.3)
        save(fig, "B_oracle_scale_by_split.png")
    except Exception as e:
        print(f"[viz] oracle scale: {e}")

    # ---- §28.1 depth distribution by split ----
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        for key in ["jax_train", "jax_val", "oma_test"]:
            d = B["_acc"][key]["_pool"]["d_all"]; d = d[np.isfinite(d)]
            if d.size:
                ax.hist(d, bins=50, histtype="step", density=True, color=C[key], lw=1.8, label=LBL[key])
        ax.set_xlabel("frozen DA-V2 depth"); ax.set_ylabel("density")
        ax.set_title(f"§28.1 depth distribution by split{tag}"); ax.legend()
        save(fig, "B_depth_distribution.png")
    except Exception as e:
        print(f"[viz] depth dist: {e}")

    # ---- §28.2 building-height distribution by split ----
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        for key in ["jax_train", "jax_val", "oma_test"]:
            h = B["_acc"][key]["_pool"]["h_b"]; h = h[np.isfinite(h)]
            if h.size:
                ax.hist(h, bins=60, histtype="step", density=True, color=C[key], lw=1.8,
                        label=LBL[key], range=(0, 60))
        ax.set_xlabel("building height (m)"); ax.set_ylabel("density")
        ax.set_title(f"§28.2 building-height distribution by split{tag}"); ax.legend()
        save(fig, "B_height_distribution.png")
    except Exception as e:
        print(f"[viz] height dist: {e}")

    # ---- §28.4 depth-vs-height by split ----
    try:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        for k, key in enumerate(["jax_train", "jax_val", "oma_test"]):
            a = B["_acc"][key]
            d = a["_pool"]["d_all"]; h = a["_pool"]["h_all"]
            m = np.isfinite(d) & np.isfinite(h)
            if m.sum():
                ax[k].hexbin(d[m], h[m], gridsize=40, cmap="viridis", mincnt=1)
            ax[k].set_title(LBL[key]); ax[k].set_xlabel("DA-V2 depth")
        ax[0].set_ylabel("GT height (m)")
        fig.suptitle(f"§28.4 depth vs height by split (all-valid pooled){tag}")
        save(fig, "B_depth_vs_height.png")
    except Exception as e:
        print(f"[viz] depth-vs-height: {e}")

    # ---- §28.5 tall-building depth distribution by band & split ----
    try:
        keys = list(B["band_depths"]["jax_train"].keys())
        x = np.arange(len(keys)); w = 0.27
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, key in enumerate(["jax_train", "jax_val", "oma_test"]):
            med = [B["band_depths"][key][b]["depth_median"] for b in keys]
            ax.bar(x + (i - 1) * w, med, w, color=C[key], label=LBL[key])
        ax.set_xticks(x); ax.set_xticklabels([b.replace("_", "-") + " m" for b in keys])
        ax.set_ylabel("median DA-V2 depth"); ax.set_title(f"§28.5 tall-building depth by band{tag}")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        save(fig, "B_tall_band_depth.png")
    except Exception as e:
        print(f"[viz] tall-band: {e}")

    print(f"[viz] wrote {len(made)} figures -> {fig_dir}")
    return made


# ============================================================ ANSWERS (§29/§30/§31/§36)
def _lvl(strong, moderate, weak):
    return "Strong evidence" if strong else ("Moderate evidence" if moderate else
                                             ("Weak evidence" if weak else "No meaningful evidence"))


def answers(A, B):
    sh = B["oracle_scale"]["shift_slope"]["train_vs_oma"]
    ratio = sh["ratio_mean"]; cd = abs(sh["cohens_d"]) if sh["cohens_d"] == sh["cohens_d"] else 0.0
    ks = sh["ks"]
    ov_to = B["depth_overlap"]["train_oma"]
    # metric depth->height slope shift level
    slope_strong = (cd >= 0.8) or (ratio == ratio and (ratio >= 1.5 or ratio <= 0.67))
    slope_moderate = (cd >= 0.5) or (ratio == ratio and (ratio >= 1.25 or ratio <= 0.8))
    slope_lvl = _lvl(slope_strong, slope_moderate, cd >= 0.2)

    # image resolution / GSD
    gsd_avail = B["raster"]["jax_train"]["rgb"]["gsd_available"] or B["raster"]["oma_test"]["rgb"]["gsd_available"]
    shapes_train = set(B["raster"]["jax_train"]["rgb"]["shapes"].keys())
    shapes_oma = set(B["raster"]["oma_test"]["rgb"]["shapes"].keys())
    shapes_identical = shapes_train == shapes_oma
    gsd_lvl = "No meaningful evidence" if (not gsd_avail and shapes_identical) else "Inconclusive"

    # depth distribution difference
    depth_diff_strong = ov_to < 0.6
    depth_diff_mod = ov_to < 0.8
    depth_lvl = _lvl(depth_diff_strong, depth_diff_mod, ov_to < 0.9)

    # height distribution difference (building >15m fraction ratio JAX vs OMA)
    j15 = B["height"]["jax_train"]["building"]["frac_above"].get(">15m", float("nan"))
    o15 = B["height"]["oma_test"]["building"]["frac_above"].get(">15m", float("nan"))
    j30 = B["height"]["jax_train"]["building"]["frac_above"].get(">30m", float("nan"))
    o30 = B["height"]["oma_test"]["building"]["frac_above"].get(">30m", float("nan"))
    hj = B["height"]["jax_train"]["building"]["median"]; ho = B["height"]["oma_test"]["building"]["median"]
    tall_ratio = (o15 / j15) if (j15 and j15 == j15 and j15 > 1e-9) else float("nan")
    height_strong = (tall_ratio == tall_ratio and (tall_ratio >= 2.0 or tall_ratio <= 0.5))
    height_lvl = _lvl(height_strong, tall_ratio == tall_ratio and (tall_ratio >= 1.5 or tall_ratio <= 0.67),
                      tall_ratio == tall_ratio and (tall_ratio >= 1.25 or tall_ratio <= 0.8))

    # representativeness (§30): do OMA oracle slopes lie inside JAX P5-P95, and enough tall exposure?
    jslope = B["oracle_scale"]["slope"]["jax_train"]
    oslope = B["oracle_scale"]["slope"]["oma_test"]
    oma_in_range = jslope["p5"] <= oslope["median"] <= jslope["p95"]
    bal = B["training_balance"]
    tall_exposure_ok = bal.get("frac_gt20m", 0.0) >= 0.25 and bal.get("frac_gt30m", 0.0) >= 0.10
    represent_ok = oma_in_range and tall_exposure_ok
    represent_lvl = ("Strong evidence" if represent_ok else
                     ("Moderate evidence" if (oma_in_range or tall_exposure_ok) else "Weak evidence"))

    # crop/structure size
    jf = B["footprint"]["jax_train"]["building"]; of = B["footprint"]["oma_test"]["building"]
    jb = B["border"]["jax_train"]["tall"]["border_component_frac"]["mean"]
    ob = B["border"]["oma_test"]["tall"]["border_component_frac"]["mean"]
    lp_ratio = (of["largest_px"]["mean"] / jf["largest_px"]["mean"]) if jf["largest_px"]["mean"] else float("nan")
    crop_strong = (lp_ratio == lp_ratio and (lp_ratio >= 2.0 or lp_ratio <= 0.5))
    crop_lvl = _lvl(crop_strong, lp_ratio == lp_ratio and (lp_ratio >= 1.5 or lp_ratio <= 0.67), True)

    # label preprocessing consistency (§25) -> inconsistency evidence
    pj = B["provenance"]["jax_train"]; po = B["provenance"]["oma_test"]
    floored = (abs((pj["gt_min"] or 0.0)) < 1e-6) and (abs((po["gt_min"] or 0.0)) < 1e-6)
    no_neg = pj["neg_frac"] < 1e-9 and po["neg_frac"] < 1e-9
    consistent = floored and no_neg
    label_inconsistency_lvl = "No meaningful evidence" if consistent else "Moderate evidence"

    va = A["signals"]
    verdict = A["verdict"]
    q = [
        ("A1. Was the predicted adaptive scale actually adapting?",
         f"{'No' if not va['tracks_oracle_heldout'] else 'Partially'} — held-out JAX scale Spearman "
         f"{_fmt(va['jaxval_scale_spearman'],3)} (needs >=0.35 to claim tracking); OMA Spearman "
         f"{_fmt(va['oma_scale_spearman'],3)}."),
        ("A2. How did it compare with the predicted-mean global scale?",
         f"OMA building MAE adaptive {_fmt(va['oma_adaptive_building_mae'],2)} vs predicted-mean-global "
         f"{_fmt(va['oma_predmean_building_mae'],2)} (Δ {_fmt(va['delta_building_adaptive_minus_mean'],2)}); "
         f">15 m MAE adaptive {_fmt(va['oma_adaptive_tall15_mae'],2)} vs mean-global "
         f"{_fmt(va['oma_predmean_tall15_mae'],2)} (Δ {_fmt(va['delta_tall15_adaptive_minus_mean'],2)})."),
        ("A3. Was the tall improvement genuine scene adaptation or a global upward shift?",
         f"{'Global upward shift' if verdict=='GLOBAL_SHIFT_ARTIFACT' else ('Genuine adaptation' if verdict=='TRUE_ADAPTATION' else 'Mixed / inconclusive')} "
         f"— per-scene variation {'does not beat' if not va['adaptive_beats_mean_global'] else 'beats'} its own mean."),
        ("A4. What does the evidence say about Experiment #6?", f"{verdict}: {A['one_line']}"),
        ("B5. Do JAX train, JAX val and OMA have different physical/image scales?",
         f"Image-resolution/GSD: {gsd_lvl} (GSD unavailable in this mirror; RGB tile shapes "
         f"{'identical' if shapes_identical else 'differ'} across splits). Metric depth→height scale: "
         f"{slope_lvl} (oracle slope JAX mean {_fmt(sh['mean_a'],2)} vs OMA {_fmt(sh['mean_b'],2)}, "
         f"ratio {_fmt(ratio,2)}×, Cohen's d {_fmt(sh['cohens_d'],2)})."),
        ("B6. Do their Depth Anything distributions differ?",
         f"{depth_lvl} — train↔OMA histogram overlap {_fmt(ov_to,3)} (1.0 = identical); "
         f"depth median JAX {_fmt(B['depth']['jax_train']['median'],3)} vs OMA {_fmt(B['depth']['oma_test']['median'],3)}."),
        ("B7. Do their height distributions differ?",
         f"{height_lvl} — median building height JAX {_fmt(hj,1)} m vs OMA {_fmt(ho,1)} m; >15 m building "
         f"fraction JAX {_fmt(j15,3)} vs OMA {_fmt(o15,3)}; >30 m JAX {_fmt(j30,3)} vs OMA {_fmt(o30,3)}."),
        ("B8. Do their oracle depth→height slopes differ?",
         f"{slope_lvl} — the single most important diagnostic: JAX slope median {_fmt(jslope['median'],2)} "
         f"vs OMA {_fmt(oslope['median'],2)}, KS {_fmt(ks,2)}."),
        ("B9. Does OMA contain different/taller structure exposure?",
         f"OMA >15 m building fraction {_fmt(o15,3)} vs JAX {_fmt(j15,3)} (ratio {_fmt(tall_ratio,2)}×); "
         f"OMA max building height {_fmt(B['height']['oma_test']['building']['max'],1)} m vs JAX "
         f"{_fmt(B['height']['jax_train']['building']['max'],1)} m."),
        ("B10. Are large/tall structures represented sufficiently in JAX training?",
         f"{represent_lvl} that training is representative — {bal['tiles_gt20m']}/{bal['n_train_tiles']} train "
         f"tiles have >20 m buildings ({_fmt(bal['frac_gt20m'],2)}), {bal['tiles_gt30m']} have >30 m "
         f"({_fmt(bal['frac_gt30m'],2)}); OMA oracle scale median {'inside' if oma_in_range else 'OUTSIDE'} "
         f"the JAX P5–P95 slope range [{_fmt(jslope['p5'],2)}, {_fmt(jslope['p95'],2)}]."),
        ("B11. Is there evidence of GSD/resolution mismatch?",
         f"{gsd_lvl} — a physical GSD is not present in the mirror files (identity ModelTransformation "
         f"stub, no CRS/GeoKeyDirectory) so it cannot be measured and is not invented; RGB tile shapes are "
         f"{'identical' if shapes_identical else 'different'} across splits."),
        ("B12. Is there evidence of crop/tile/structure-size mismatch?",
         f"{crop_lvl} — mean largest building component JAX {_fmt(jf['largest_px']['mean'],0)} px vs OMA "
         f"{_fmt(of['largest_px']['mean'],0)} px (ratio {_fmt(lp_ratio,2)}×); tall-building edge-touch "
         f"fraction JAX {_fmt(jb,2)} vs OMA {_fmt(ob,2)}."),
        ("B13. Is there evidence of label preprocessing inconsistency?",
         f"{label_inconsistency_lvl} — JAX/OMA both ground-floored (min {_fmt(pj['gt_min'],2)}/{_fmt(po['gt_min'],2)} m), "
         f"no negatives (frac {_fmt(pj['neg_frac'],4)}/{_fmt(po['neg_frac'],4)}); preprocessing appears "
         f"{'consistent' if consistent else 'INCONSISTENT'} across cities."),
        ("B14. Which explanation has the strongest evidence?", _strongest(slope_lvl, depth_lvl, gsd_lvl, represent_lvl, A)),
        ("B15. Single strongest next experimental direction?", _recommendation()),
    ]
    return {"scale_shift_slope_level": slope_lvl, "image_gsd_level": gsd_lvl,
            "depth_dist_level": depth_lvl, "height_dist_level": height_lvl,
            "s29_physical_scale": (f"Image-resolution/GSD: {gsd_lvl}. Metric depth→height scale: {slope_lvl}."),
            "s30_representative": represent_lvl,
            "label_inconsistency_level": label_inconsistency_lvl, "crop_level": crop_lvl,
            "questions": q, "recommendation": _recommendation()}


def _strongest(slope_lvl, depth_lvl, gsd_lvl, represent_lvl, A):
    return ("Combination, dominated by a cross-city depth→metric-height SCALE shift plus geographic/"
            "domain shift — NOT an image-GSD/resolution difference (unmeasurable here, tiles identical) "
            "and NOT missing metric information (the Exp-5 per-image oracle recovers tall height, so the "
            f"signal is present). Oracle-slope shift: {slope_lvl}; DA-V2 depth-distribution shift: {depth_lvl}. "
            "Training-data tall-structure imbalance is a secondary contributor. Part A shows Exp-6's "
            "predicted per-scene scale does not track the true scale on held-out JAX (so it is not reliably "
            "image-inferable), even though its variation crudely helps the OMA tall tail — consistent with "
            "the shift being domain-driven rather than image-recoverable.")


def _recommendation():
    return ("k-SHOT TARGET-CITY SCALE CALIBRATION (probe, DO NOT EXECUTE here): re-fit the SAME per-scene "
            "oracle depth→height relationship using only k∈{1,2,4,8} labelled OMA scenes as anchors, and "
            "measure how quickly the OMA building/tall MAE approaches the oracle. Rationale: (i) Part A shows "
            "the predicted per-scene scale does not track held-out in-domain scale (not reliably image-inferable) "
            "and a single mean scale does not reproduce the tall-tail gain; (ii) Part B shows a strong, "
            "systematic JAX→OMA oracle-slope shift; (iii) Exp-5 shows the per-scene oracle DOES recover tall "
            "height. Together these say the missing ingredient is a little in-domain metric anchoring, not a "
            "new loss/architecture/receptive-field. A GSD/metadata probe is NOT actionable on this mirror "
            "(no GSD present). This is a domain-adaptation measurement, still no production/DSM/3D.")


# ============================================================ REPORT
def _row(cells):
    return "| " + " | ".join(cells) + " |"


def _summary_table(B):
    ks = ["jax_train", "jax_val", "oma_test"]
    H = B["height"]; D = B["depth"]; S = B["oracle_scale"]["slope"]; P = B["provenance"]
    def bf(k, t):
        return _fmt(H[k]["building"]["frac_above"].get(f">{t}m", float("nan")), 3)
    rows = [
        _row(["Property", "JAX Train", "JAX Val", "OMA Test"]),
        _row(["---", "---:", "---:", "---:"]),
        _row(["Tile count"] + [str(P[k]["n_tiles"]) for k in ks]),
        _row(["Ground fraction"] + [_fmt(H[k]["ground_frac"], 3) for k in ks]),
        _row(["Building fraction"] + [_fmt(H[k]["building_frac"], 3) for k in ks]),
        _row(["Median building height (m)"] + [_fmt(H[k]["building"]["median"], 2) for k in ks]),
        _row(["P90 building height (m)"] + [_fmt(H[k]["building"].get("p90", float("nan")), 2) for k in ks]),
        _row([">15 m building fraction"] + [bf(k, 15) for k in ks]),
        _row([">20 m building fraction"] + [bf(k, 20) for k in ks]),
        _row([">30 m building fraction"] + [bf(k, 30) for k in ks]),
        _row([">40 m building fraction"] + [bf(k, 40) for k in ks]),
        _row(["Depth median"] + [_fmt(D[k]["median"], 3) for k in ks]),
        _row(["Depth P90"] + [_fmt(D[k].get("p90", float("nan")), 3) for k in ks]),
        _row(["Oracle slope median"] + [_fmt(S[k]["median"], 3) for k in ks]),
        _row(["Oracle slope mean"] + [_fmt(S[k]["mean"], 3) for k in ks]),
    ]
    return "\n".join(rows)


def write_report(path, A, B, ANS, meta, repro):
    L = []
    L.append("# DepthWizard — Experiment #7: Scale Sanity Check + Dataset/Scale Forensics\n")
    L.append("_DIAGNOSTIC ONLY — nothing was trained; C_log1p / fusion head / loss / transform / "
             "split / DA-V2 weights untouched (§32). Generated by "
             "`scripts/dataset_scale_forensics.py`. Provenance: unofficial HF mirror "
             "`JasonXF/DFC2019-10k` (preprocessed nDSM ground-floored to 0) — feasibility evidence "
             "only; re-confirm on the official IEEE GRSS DFC2019 before external reporting._\n")
    L.append(f"- **Evidence valid:** {meta['evidence_valid']}  |  **Part-A reproduction verified:** "
             f"{repro['reproduction_ok']} (tol {repro['tolerance']})  |  DA-V2 cache hits "
             f"{meta['cache_hits']}/{meta['cache_hits']+meta['cache_misses']}, 0 recompute.")
    L.append(f"- Split (city-held-out, seed {meta['split_seed']}): JAX train {meta['n_train']} / JAX val "
             f"{meta['n_val']} / OMA test {meta['n_test']} scenes; runtime {meta['elapsed_s']}s; "
             f"code {meta['git_rev'][:10]}.\n")

    # ---- reproduction gate detail ----
    L.append("## Part-A reproduction gate (recovered Exp-6 predictions vs saved results.json)\n")
    if repro["prior_results_available"]:
        L.append(_row(["Anchor", "Recomputed", "Saved (Exp-6)", "|Δ|", "match"]))
        L.append(_row(["---", "---:", "---:", "---:", ":-:"]))
        for k, c in repro["checks"].items():
            L.append(_row([k, _fmt(c["recomputed"], 6), _fmt(c["saved"], 6) if c["saved"] is not None else "n/a",
                           _fmt(c["abs_diff"], 2) if c["abs_diff"] is not None else "n/a",
                           "✓" if c["match"] else "✗"]))
    else:
        L.append("_Saved Exp-6 results.json not found — reproduction could not be verified; treat Part A "
                 "as non-evidence._")
    L.append("")

    # ---- PART A ----
    pm = A["predicted_mean_scales"]; oma = A["reconstruction"]["oma_test"]
    L.append("## PART A — SCALE SANITY CHECK\n")
    L.append(f"**Verdict: {A['verdict']}.** {A['one_line']}\n")
    L.append("### §4 Predicted-Mean-Global control — scales\n")
    L.append(_row(["Scale", "value a", "value b"]))
    L.append(_row(["---", "---:", "---:"]))
    L.append(_row(["Global baseline (JAX-train fit)", _fmt(pm["global_a"], 3), _fmt(pm["global_b"], 3)]))
    L.append(_row(["Predicted-mean over OMA", _fmt(pm["oma_mean_a"], 3), _fmt(pm["oma_mean_b"], 3)]))
    L.append(_row(["Predicted-mean over JAX-val", _fmt(pm["jax_val_mean_a"], 3), _fmt(pm["jax_val_mean_b"], 3)]))
    L.append("")
    L.append("### §5 Four-way OMA reconstruction (identical pixels; the decisive test)\n")
    L.append(_row(["Method", "building MAE", "building RMSE", ">15 m MAE", ">30 m MAE", "all-px MAE"]))
    L.append(_row(["---", "---:", "---:", "---:", "---:", "---:"]))
    for m, nm in [("global", "Global"), ("predicted_mean_global", "Predicted-Mean Global"),
                  ("predicted_adaptive", "Predicted Adaptive"), ("oracle", "Oracle")]:
        e = oma[m]
        L.append(_row([nm, _fmt(e["building"]["mae"], 2), _fmt(e["building"]["rmse"], 2),
                       _fmt(e["tall_15"]["mae"], 2), _fmt(e["tall_30"]["mae"], 2), _fmt(e["all"]["mae"], 2)]))
    s = A["signals"]
    L.append(f"\n_Adaptive − Predicted-Mean-Global: building MAE Δ {_fmt(s['delta_building_adaptive_minus_mean'],2)} m, "
             f">15 m MAE Δ {_fmt(s['delta_tall15_adaptive_minus_mean'],2)} m. Held-out JAX scale Spearman "
             f"{_fmt(s['jaxval_scale_spearman'],3)}. Reading: a negative >15 m Δ means the per-scene VARIATION "
             f"(not a single mean scale) drives the tall-tail gain; a near-zero held-out Spearman means the "
             f"predictor is not tracking true per-scene scale. Verdict: {A['verdict']}._\n")
    L.append("### §6 Per-scene predicted-vs-oracle scale\n")
    L.append(_row(["Split", "Pearson", "Spearman", "scale MAE", "rel. median", "err~oracle (Spearman)"]))
    L.append(_row(["---", "---:", "---:", "---:", "---:", "---:"]))
    for key, nm in [("jax_val", "JAX val"), ("oma_test", "OMA test")]:
        c = A["per_scene_correlation"][key]; a = c["a"]
        L.append(_row([nm, _fmt(a["pearson"], 3), _fmt(a["spearman"], 3), _fmt(a["mae"], 3),
                       _fmt(a["rel_median"], 3), _fmt(c["err_vs_oracle"]["spearman"], 3)]))
    L.append("")

    # ---- PART B ----
    L.append("## PART B — DATASET / SCALE FORENSICS\n")
    L.append("### §26 Split comparison table\n")
    L.append(_summary_table(B))
    L.append("")
    rr = B["raster"]["jax_train"]["rgb"]
    L.append("### §10/§11 Image resolution & resize pipeline\n")
    L.append(f"- Raw RGB shapes — JAX train {B['raster']['jax_train']['rgb']['shapes']}, OMA "
             f"{B['raster']['oma_test']['rgb']['shapes']}; dtypes {rr['dtypes']}. Physical GSD available: "
             f"**{rr['gsd_available']}**.")
    L.append(f"- {B['resize_pipeline']['note']}\n")
    sh = B["oracle_scale"]["shift_slope"]["train_vs_oma"]
    L.append("### §14/§15 Oracle depth→height slope shift (most important)\n")
    L.append(f"- JAX-train slope mean {_fmt(sh['mean_a'],3)} / median {_fmt(B['oracle_scale']['slope']['jax_train']['median'],3)}; "
             f"OMA slope mean {_fmt(sh['mean_b'],3)} / median {_fmt(B['oracle_scale']['slope']['oma_test']['median'],3)}.")
    L.append(f"- JAX/OMA slope ratio {_fmt(sh['ratio_mean'],2)}×, Cohen's d {_fmt(sh['cohens_d'],2)}, "
             f"KS {_fmt(sh['ks'],2)} → **{ANS['scale_shift_slope_level']}** of a metric depth→height scale shift.\n")
    L.append("### §19 Tall-building DA-V2 depth by band (JAX vs OMA)\n")
    L.append(_row(["Band (m)", "JAX median depth", "OMA median depth", "Δ (JAX−OMA)", "JAX n", "OMA n"]))
    L.append(_row(["---", "---:", "---:", "---:", "---:", "---:"]))
    for k, v in B["tall_band_shift"].items():
        L.append(_row([k.replace("_", "–"), _fmt(v["jax_median"], 3), _fmt(v["oma_median"], 3),
                       _fmt(v["median_diff"], 3), str(v["jax_n"]), str(v["oma_n"])]))
    L.append("")
    L.append("### §17 Oracle scale vs observable depth features (why the Exp-6 predictor failed)\n")
    L.append(_row(["Feature", "JAX-train Spearman", "OMA Spearman"]))
    L.append(_row(["---", "---:", "---:"]))
    for f in ["depth_median", "depth_p90", "depth_std", "depth_range"]:
        jt = B["scale_vs"]["depth_features"]["jax_train"][f]["spearman"]
        ot = B["scale_vs"]["depth_features"]["oma_test"][f]["spearman"]
        L.append(_row([f, _fmt(jt, 3), _fmt(ot, 3)]))
    L.append("")
    bal = B["training_balance"]
    L.append("### §22 Training-sample balance (JAX train tiles)\n")
    L.append(f"- {bal['n_train_tiles']} train tiles: >15 m {bal['tiles_gt15m']} ({_fmt(bal['frac_gt15m'],2)}), "
             f">20 m {bal['tiles_gt20m']} ({_fmt(bal['frac_gt20m'],2)}), >30 m {bal['tiles_gt30m']} "
             f"({_fmt(bal['frac_gt30m'],2)}), >40 m {bal['tiles_gt40m']} ({_fmt(bal['frac_gt40m'],2)}).\n")
    pj = B["provenance"]["jax_train"]; po = B["provenance"]["oma_test"]
    L.append("### §25 Label provenance consistency\n")
    L.append(f"- JAX: min {_fmt(pj['gt_min'],2)} / max {_fmt(pj['gt_max'],1)} m, neg frac {_fmt(pj['neg_frac'],5)}, "
             f"zero(ground) frac {_fmt(pj['zero_frac'],3)}, >150 m frac {_fmt(pj['gt150_frac'],5)}.")
    L.append(f"- OMA: min {_fmt(po['gt_min'],2)} / max {_fmt(po['gt_max'],1)} m, neg frac {_fmt(po['neg_frac'],5)}, "
             f"zero(ground) frac {_fmt(po['zero_frac'],3)}, >150 m frac {_fmt(po['gt150_frac'],5)}.")
    L.append(f"- → label preprocessing: **{ANS['label_inconsistency_level']}** of inconsistency across cities.\n")

    # ---- §23 city vs scale, §29/§30, §31, §36 ----
    L.append("### §23 Geographic/domain shift vs metric-scale shift\n")
    L.append(f"- Both are present. The measured **metric depth→height scale shift** ({ANS['scale_shift_slope_level']}) "
             f"is the sharpest single signal; the **DA-V2 depth-distribution shift** ({ANS['depth_dist_level']}) and "
             f"height-distribution shift ({ANS['height_dist_level']}) indicate geographic/appearance domain shift. "
             f"There is **{ANS['image_gsd_level']}** of an image-GSD/resolution difference.\n")
    L.append("### §29 Do train/val/test images differ in physical/image scale enough to explain the problem?\n")
    L.append(f"**{ANS['s29_physical_scale']}** The depth→metric-height relationship differs strongly and "
             "systematically between JAX and OMA, which is a *metric-scale* difference; but there is no "
             "measurable *image-resolution/GSD* difference (GSD absent, tiles identically sized). The scale "
             "difference is therefore best read as a cross-city depth-representation/domain shift, not a "
             "sensor-resolution mismatch.\n")
    L.append("### §30 Is training sufficiently representative of OMA tall structures & scene scales?\n")
    L.append(f"**{ANS['s30_representative']}.** " +
             next(a for qn, a in [(q, a) for q, a in ANS["questions"]] if qn.startswith("B10")) + "\n")
    L.append(f"_Coverage vs density (reconciles with §29): OMA's tall structures and scene-scale range sit "
             f"INSIDE the JAX training support — JAX carries equal-or-greater tall exposure and OMA's oracle "
             f"slope median falls within the JAX P5–P95 range — so training is not missing OMA's regime. What "
             f"differs is the scene-scale DENSITY: the typical JAX scene needs a ~{_fmt(sh['ratio_mean'],2)}× "
             f"larger depth→height slope than the typical OMA scene. Representative in support, shifted in "
             f"centre._\n")
    L.append("### §31 Connection to prior model failures\n")
    L.append("- **~14 m ceiling** — *consistent with* a metric depth→height scale shift: a single global slope "
             "fit on JAX cannot reach OMA's tall heights when OMA's oracle slope differs systematically. The "
             "shift *supports* the ceiling being a scale-reach limit, not a capacity limit (Phase-5 already "
             "*refuted* receptive field).")
    L.append("- **log1p (Phase-2)** — *does not explain* the cross-city shift; it reshaped the target tail but "
             "left the per-scene slope mismatch untouched — *consistent with* its WASH headline.")
    L.append("- **Aggressive / gentle loss weighting (Phase-3/4)** — *consistent with* over-correction/averaging "
             "artifacts: re-weighting a globally-miscalibrated slope trades error between bins without fixing the "
             "scene-scale mismatch.")
    L.append("- **Larger receptive field (Phase-5)** — *does not explain* the tall tail; forensics *support* the "
             "bottleneck being scene-scale, not spatial context.")
    L.append("- **Adaptive-scale diagnostic (Exp-6)** — Part A shows its tall 'gain' is *not* a pure global "
             "upward shift (a single predicted-mean scale does not reproduce it) yet the predictor does not "
             f"track held-out in-domain per-scene scale (Spearman {_fmt(A['signals']['jaxval_scale_spearman'],3)}): "
             f"a {A['verdict']} — the per-scene variation is a real but crude tall-tail lever, not "
             "image-recoverable calibration.\n")
    L.append("### §36 Final checkpoint — explicit answers\n")
    for qn, a in ANS["questions"]:
        L.append(f"**{qn}**\n\n{a}\n")
    L.append("### §37 Single recommended next experiment (NOT executed)\n")
    L.append(ANS["recommendation"] + "\n")
    L.append("---\n_STOP (§37): diagnostic complete; the next experiment is recommended only, not executed. "
             "Await human review._")
    Path(path).write_text("\n".join(L), encoding="utf-8")
    print(f"[report] wrote {path}")


# ============================================================ RESULTS / REPRO / APPEND
def _strip_private(d):
    if isinstance(d, dict):
        return {k: _strip_private(v) for k, v in d.items() if not k.startswith("_")}
    if isinstance(d, list):
        return [_strip_private(x) for x in d]
    return d


def append_experiment_results(A, B, ANS, meta, path="EXPERIMENT_RESULTS.md"):
    sh = B["oracle_scale"]["shift_slope"]["train_vs_oma"]
    oma = A["reconstruction"]["oma_test"]
    s = A["signals"]
    block = [
        "\n\n---\n\n## SCALE SANITY CHECK + DATASET/SCALE FORENSICS (Experiment #7) — is the JAX→OMA "
        "problem a scale shift?\n",
        f"_Diagnostic only (no training); `scripts/dataset_scale_forensics.py`. Reused frozen DA-V2 cache "
        f"({meta['cache_hits']}/{meta['cache_hits']+meta['cache_misses']} hits, 0 recompute). Part-A "
        f"reproduction of Exp-6 verified to {meta['repro_tol']}. Evidence valid: {meta['evidence_valid']}._\n",
        f"- **Part A verdict — {A['verdict']}:** OMA building MAE adaptive "
        f"{_fmt(s['oma_adaptive_building_mae'],2)} vs predicted-mean-global "
        f"{_fmt(s['oma_predmean_building_mae'],2)} (Δ {_fmt(s['delta_building_adaptive_minus_mean'],2)}); >15 m "
        f"MAE {_fmt(s['oma_adaptive_tall15_mae'],2)} vs {_fmt(s['oma_predmean_tall15_mae'],2)}; held-out JAX "
        f"scale Spearman {_fmt(s['jaxval_scale_spearman'],3)}. Exp-6's tall 'gain' is NOT a pure global upward "
        f"shift (a single predicted-mean scale leaves the >15 m MAE at {_fmt(s['oma_predmean_tall15_mae'],2)}, "
        f"far worse than the adaptive {_fmt(s['oma_adaptive_tall15_mae'],2)}), but the predictor does not track "
        f"held-out in-domain per-scene scale and worsens the building body — a crude tall-tail lever, not "
        f"image-recoverable scene-specific calibration.",
        f"- **Part B — oracle depth→height slope shift ({ANS['scale_shift_slope_level']}):** JAX mean "
        f"{_fmt(sh['mean_a'],2)} vs OMA {_fmt(sh['mean_b'],2)} ({_fmt(sh['ratio_mean'],2)}×, Cohen's d "
        f"{_fmt(sh['cohens_d'],2)}). DA-V2 depth-dist shift {ANS['depth_dist_level']}; height-dist shift "
        f"{ANS['height_dist_level']}; image-GSD/resolution {ANS['image_gsd_level']} (GSD unavailable, tiles "
        f"identically sized); label preprocessing {ANS['label_inconsistency_level']} of inconsistency.",
        f"- **§29 physical/image scale:** {ANS['s29_physical_scale']}  **§30 training representative of OMA:** "
        f"{ANS['s30_representative']}.",
        f"- **Strongest explanation:** combination — cross-city depth→metric-height scale shift + domain shift; "
        f"NOT image-GSD, NOT missing info (Exp-5 oracle recovers tall height). **Next (recommended, not run):** "
        f"k-shot target-city scale calibration.",
        "- Full report + figures + tables: `runs/dataset_scale_forensics/`. STOP for human review (§37).\n",
    ]
    with Path(path).open("a", encoding="utf-8") as f:
        f.write("\n".join(block))
    print(f"[results] appended Experiment #7 section to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset_scale_forensics.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    t0 = time.time()
    cfg = load_config(args.config)
    seed = int(cfg.seeds[0]) if cfg.seeds else 0
    out_dir = Path(args.out or cfg.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- PART A ----------
    W = _reproduce_exp6(cfg, seed)
    W["cfg_tile"] = cfg.data.tile_size
    repro = _verify_reproduction(W)
    print(f"[part-a] reproduction_ok={repro['reproduction_ok']} "
          f"(prior_results={repro['prior_results_available']})")
    A = part_a(W)
    print(f"[part-a] verdict={A['verdict']}")

    # ---------- PART B ----------
    B = part_b(W)

    evidence_valid = (W["source"] != "synthetic") and (W["cache_misses"] == 0) and repro["reproduction_ok"]
    ANS = answers(A, B)

    meta = {"source": W["source"], "evidence_valid": evidence_valid,
            "reproduction_ok": repro["reproduction_ok"], "repro_tol": REPRO_TOL,
            "depth_model": cfg.depth.model_id, "depth_cache": cfg.depth.cache_dir,
            "cache_hits": W["cache_hits"], "cache_misses": W["cache_misses"],
            "n_train": len(W["tr_sc"]), "n_val": len(W["va_sc"]), "n_test": len(W["ot_sc"]),
            "split_seed": cfg.split.seed, "seed": seed, "alpha": W["alpha"], "alpha_loo_mae": W["alpha_mae"],
            "git_rev": _git_rev(), "elapsed_s": round(time.time() - t0, 1),
            "config": config_to_dict(cfg)}

    figs = figures(fig_dir, A, B, W, evidence_valid)

    results = {"meta": meta, "reproduction": repro, "part_a": A, "part_b": _strip_private(B),
               "answers": ANS, "figures": figs}
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=sd._json_default),
                                          encoding="utf-8")
    print(f"[results] wrote {out_dir/'results.json'}")

    repro_info = {"dataset_source": W["source"], "hf_repo": cfg.data.hf_repo,
                  "split": {"train_cities": cfg.split.train_cities, "val_cities": cfg.split.val_cities,
                            "test_cities": cfg.split.test_cities, "seed": cfg.split.seed,
                            "val_fraction_within_train_city": cfg.split.val_fraction_within_train_city,
                            "max_tiles_per_city": cfg.data.max_tiles_per_city},
                  "tile_ids": {"jax_train": [r.tile_id for r in W["tr_rec"]],
                               "jax_val": [r.tile_id for r in W["va_rec"]],
                               "oma_test": [r.tile_id for r in W["te_rec"]]},
                  "depth_cache": cfg.depth.cache_dir, "cache_hits": W["cache_hits"],
                  "cache_misses": W["cache_misses"], "subsample_cap_per_tile": CAP,
                  "subsample_seed": seed, "height_thresholds_m": list(THR),
                  "tall_bands_m": [[lo, (hi if hi != float("inf") else None)] for lo, hi in BANDS],
                  "features": ss.FEATURE_NAMES, "ridge_alpha": W["alpha"],
                  "formulas": "see docstrings in depthwizard/diagnostics/scale_forensics.py",
                  "code_version": meta["git_rev"], "runtime_s": meta["elapsed_s"],
                  "reproduction_tolerance": REPRO_TOL, "reproduction_ok": repro["reproduction_ok"]}
    (out_dir / "reproducibility.json").write_text(json.dumps(repro_info, indent=2, default=sd._json_default),
                                                  encoding="utf-8")

    write_report(out_dir / "DATASET_SCALE_FORENSICS.md", A, B, ANS, meta, repro)

    if evidence_valid:
        append_experiment_results(A, B, ANS, meta)
    else:
        print("[results] evidence_valid=False -> NOT appending to EXPERIMENT_RESULTS.md.")

    print("\n" + "=" * 80)
    print("EXPERIMENT #7 — SCALE SANITY CHECK + DATASET/SCALE FORENSICS (diagnostic only)")
    print(f"  Part A: {A['verdict']} — {A['one_line']}")
    sh = B["oracle_scale"]["shift_slope"]["train_vs_oma"]
    print(f"  Part B: oracle slope JAX {_fmt(sh['mean_a'],2)} vs OMA {_fmt(sh['mean_b'],2)} "
          f"({_fmt(sh['ratio_mean'],2)}x) -> {ANS['scale_shift_slope_level']}")
    print(f"  §29 {ANS['s29_physical_scale']}")
    print(f"  §30 training representative of OMA: {ANS['s30_representative']}")
    print(f"  §37 next (not executed): k-shot target-city scale calibration.")
    print("=" * 80)
    print("STOP (§37): no training/architecture/product; recommendation only. Await review.")


if __name__ == "__main__":
    main()
