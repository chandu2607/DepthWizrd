#!/usr/bin/env python
"""DepthWizard PHASE-4 orchestrator -- controlled CALIBRATED-TAIL-WEIGHT test.

Question (§4): can a GENTLER, THRESHOLDED tail weight preserve the tall-height gains
the aggressive Phase-3 weight produced WHILE avoiding the broad upward bias it imposed
on the abundant 0–15 m building population -- i.e. improve >15 m behaviour without
distorting the whole height distribution?

Protocol (identical to Phase-3 except the added variant; §5 single variable):
  * Reuse the EXACT Phase-1/2/3 data + city-held-out split + DA-V2 depth cache.
  * Anchors: A (raw depth, signal), B (global affine, bar), oracle (upper bound).
  * Baseline C trained FOUR ways per seed; only (target_transform, loss_type) differ:
        C_none                 = "none",  "standard"        (Phase-1 original)
        C_log1p                = "log1p", "standard"        (Phase-2 reference; PRIMARY CONTROL §12)
        C_log1p_weighted       = "log1p", "height_weighted" (Phase-3 aggressive ablation §24)
        C_log1p_tail_weighted  = "log1p", "tail_weighted"   (NEW, tested §13)
    KEY comparison: C_log1p vs C_log1p_tail_weighted (isolates the calibrated weight).
  * Strict eval: all/building/non-building, in-domain AND cross-city, per-GT-height-bin
    MAE/bias. BOTH seeds -> mean +/- std; never a single favorable seed.
  * Tail weight w(h)=min(1+max(h-h_start,0)/scale, w_max) on PHYSICAL height, ALL params
    derived ONLY from JAX training stats; see scripts/phase4_weight_diagnostic.py.

Does NOT build the product. STOPS at the Phase-4 scientific checkpoint (§28/§30).

Usage:
  python scripts/run_phase4_tail_weighted.py --config configs/phase4_tail_weighted.yaml
  python scripts/run_phase4_tail_weighted.py --smoke --allow-fake-depth   # plumbing only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.config import load_config, config_to_dict
from depthwizard.data import fetch, datasets
from depthwizard.eval.evaluate import evaluate_estimator, evaluate_oracle
from depthwizard.eval.decision import decide
from depthwizard.metrics.height_metrics import DEFAULT_HEIGHT_EDGES

CEILING = 14.0  # the observed Phase-1 Baseline-C prediction ceiling under test

# The four C variants: (key, target_transform, loss_type, human label)
VARIANTS = [
    ("C_none", "none", "standard", "C · original (none)"),
    ("C_log1p", "log1p", "standard", "C · log1p (control)"),
    ("C_log1p_weighted", "log1p", "height_weighted", "C · log1p + aggressive weight (Phase-3)"),
    ("C_log1p_tail_weighted", "log1p", "tail_weighted", "C · log1p + calibrated tail weight (NEW)"),
]
# variants displayed in tables/figures (all four), and the trio for the §22 bin figure.
TABLE_KEYS = [k for k, *_ in VARIANTS]
TRIO = ["C_log1p", "C_log1p_weighted", "C_log1p_tail_weighted"]


# ---------------------------------------------------------------- utilities
def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
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


def _mean_std(vals):
    a = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return None, None
    return float(a.mean()), float(a.std())


def _get(agg, key):
    v = (agg or {}).get(key)
    return v if (v is not None and v == v) else None


# ---------------------------------------------------------------- per-variant
def _fit_eval_variant(transform, loss_type, cfg, train, val, test, edges, keep_fitted):
    """Train Baseline C with a given (target_transform, loss_type) once per seed.

    ONLY these two knobs vary across variants; the tail/aggressive weight params travel
    on cfg.train (each loss_type reads its own fields) so everything else is identical
    -> any delta is attributable to the loss weighting. Returns (per_seed, fitted_seed0).
    """
    from depthwizard.models.fusion_head import LearnedFusionHead
    per_seed = []
    fitted0 = None
    tcfg = replace(cfg.train, target_transform=transform, loss_type=loss_type)
    for si, seed in enumerate(cfg.seeds):
        c = LearnedFusionHead(tcfg, nodata=cfg.data.nodata, seed=seed)
        n_params = c.n_params()
        c.fit(train)
        idn = evaluate_estimator(c, val, cfg, "indomain", bin_edges=edges)
        xc = evaluate_estimator(c, test, cfg, "xcity", bin_edges=edges)
        per_seed.append({"seed": seed, "n_params": n_params,
                         "indomain": idn, "xcity": xc})
        xa = xc["aggregate"]["all"]
        xb = xc["aggregate"].get("building", {})
        print(f"[{transform}/{loss_type}][seed={seed}] xcity all MAE={_get(xa,'mae_pooled')} "
              f"| building MAE={_get(xb,'mae_pooled')}")
        if keep_fitted and si == 0:
            fitted0 = c
    return per_seed, fitted0


def _summarize(per_seed):
    """mean +/- std across seeds for the headline scalars + per-bin MAE/bias."""
    def scal(group, kind, key):
        return _mean_std([ps[group]["aggregate"].get(kind, {}).get(key)
                          for ps in per_seed])

    def binned(group, which):
        lists = [ps[group]["aggregate"].get(which, []) for ps in per_seed]
        lists = [l for l in lists if l]
        if not lists:
            return []
        nb = len(lists[0])
        out = []
        for i in range(nb):
            mae_m, mae_s = _mean_std([l[i]["mae"] for l in lists])
            bias_m, bias_s = _mean_std([l[i]["bias"] for l in lists])
            out.append({"lo": lists[0][i]["lo"], "hi": lists[0][i]["hi"],
                        "n_pixels": lists[0][i]["n_pixels"],
                        "mae_mean": mae_m, "mae_std": mae_s,
                        "bias_mean": bias_m, "bias_std": bias_s})
        return out

    s = {"seeds": [ps["seed"] for ps in per_seed],
         "n_params": per_seed[0]["n_params"] if per_seed else None}
    for group in ("xcity", "indomain"):
        for kind in ("all", "building", "non_building"):
            for key in ("mae_pooled", "rmse_pooled"):
                m, sd = scal(group, kind, key)
                s[f"{group}_{kind}_{key}_mean"] = m
                s[f"{group}_{kind}_{key}_std"] = sd
        s[f"{group}_binned_all"] = binned(group, "binned_all")
        s[f"{group}_binned_building"] = binned(group, "binned_building")
    s["xcity_building_pearson_mean"], _ = _mean_std(
        [ps["xcity"]["aggregate"].get("building", {}).get("pearson_mean")
         for ps in per_seed])
    s["xcity_all_pearson_mean"], _ = _mean_std(
        [ps["xcity"]["aggregate"].get("all", {}).get("pearson_mean")
         for ps in per_seed])
    return s


def _variant_verdict(A, B, variant_summary):
    """Reuse decide() with this variant's C plugged into the C slot (mean numbers)."""
    def c_agg():
        return {"aggregate": {
            "all": {"mae_pooled": variant_summary["xcity_all_mae_pooled_mean"],
                    "rmse_pooled": variant_summary["xcity_all_rmse_pooled_mean"]},
            "building": {"pearson_mean": variant_summary["xcity_building_pearson_mean"]}}}

    def c_id():
        return {"aggregate": {"all": {
            "mae_pooled": variant_summary["indomain_all_mae_pooled_mean"]}}}

    synth = {
        "indomain": {
            "A_raw_depth": A["indomain"], "B_global_affine": B["indomain"],
            "C_learned_fusion": c_id()},
        "xcity": {
            "A_raw_depth": A["xcity"], "B_global_affine": B["xcity"],
            "C_learned_fusion": c_agg()},
    }
    return decide(synth, evidence_valid=True)


def _pool_bldg(S, want_tall):
    """Pixel-weighted building MAE & bias pooled over tall (lo>=CEILING) OR low
    (lo<CEILING) bins -- the low pool is the 0–15 m regime the tail weight protects."""
    mnum = bnum = den = 0.0
    for bb in S["xcity_binned_building"]:
        is_tall = bb["lo"] >= CEILING
        if is_tall != want_tall:
            continue
        if bb["n_pixels"] and bb["mae_mean"] is not None:
            mnum += bb["mae_mean"] * bb["n_pixels"]
            bnum += (bb["bias_mean"] or 0.0) * bb["n_pixels"]
            den += bb["n_pixels"]
    return ((mnum / den) if den else None, (bnum / den) if den else None)


# ---------------------------------------------------------------- challenging viz
def _challenge_ids(test, nodata, blabel, ceiling=CEILING, k=2):
    scored = []
    for s in test:
        gt = np.asarray(s["gt"], np.float32)
        valid = np.isfinite(gt) & (gt != nodata)
        cls = s.get("cls")
        if cls is not None:
            tall = int(((np.asarray(cls) == blabel) & valid & (gt > ceiling)).sum())
        else:
            tall = int((valid & (gt > ceiling)).sum())
        scored.append((tall, s["id"]))
    scored.sort(reverse=True)
    return [sid for _, sid in scored[:k] if scored]


def _make_figures(fitted, test, cfg, fig_dir, summaries, evidence_valid):
    """§21 challenge panels/scatter (C_log1p, aggressive, tail) + §22 the decisive
    3-variant per-height-bin building MAE/bias comparison."""
    try:
        from depthwizard.viz import plots
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[viz] skipped ({e})")
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if evidence_valid else "  [NON-EVIDENCE]"
    ids = _challenge_ids(test, cfg.data.nodata, cfg.data.building_label, k=2)
    by_id = {s["id"]: s for s in test}
    # §21: for each challenging tall tile, render RGB|ref|pred|error for the control,
    # the aggressive model, and the new calibrated model, plus a pred-vs-ref scatter.
    focus = {k: fitted.get(k) for k in TRIO}
    for sid in ids:
        s = by_id.get(sid)
        if s is None:
            continue
        for vname, head in focus.items():
            if head is None:
                continue
            pred = head.predict(s)
            plots.save_qualitative(
                s, pred, str(fig_dir / f"challenge_{sid}_{vname}.png"),
                nodata=cfg.data.nodata,
                title=f"{vname} (seed0) · {sid} · cross-city{tag}")
            plots.save_scatter(
                pred, np.asarray(s["gt"], np.float32),
                str(fig_dir / f"scatter_{sid}_{vname}.png"), nodata=cfg.data.nodata,
                title=f"{vname} · {sid} · pred vs ref (m){tag}")
    # §22: cross-city BUILDING MAE and signed bias per height bin, THREE variants.
    try:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        colors = {"C_log1p": "#1f6feb", "C_log1p_weighted": "#c1440e",
                  "C_log1p_tail_weighted": "#1f883d"}
        for which, a, ylab in [("mae_mean", ax[0], "cross-city building MAE (m)"),
                               ("bias_mean", ax[1], "cross-city building signed bias (m)")]:
            for vname in TRIO:
                bins = summaries[vname].get("xcity_binned_building", [])
                if not bins:
                    continue
                labels = [f"{int(b['lo'])}-{'inf' if b['hi']==float('inf') else int(b['hi'])}"
                          for b in bins]
                y = [b[which] if b[which] is not None else np.nan for b in bins]
                a.plot(labels, y, marker="o", label=vname, color=colors.get(vname))
            a.axhline(0, color="k", lw=0.6)
            a.axvspan(-0.5, 3.5, color="green", alpha=0.05)  # protected 0–15 m bins
            a.set_xlabel("GT height bin (m)"); a.set_ylabel(ylab)
            a.tick_params(axis="x", rotation=45); a.legend()
        ax[0].set_title("Where does building error live? (lower=better)")
        ax[1].set_title(f"Bias by bin (neg=under-predict; ceiling≈{CEILING:.0f} m){tag}")
        fig.suptitle("Phase-4: calibrated tail vs aggressive vs control — cross-city building "
                     "error by height (does tail flatten bias without the upward shift?)")
        fig.tight_layout()
        fig.savefig(fig_dir / "binned_building_3variant.png", dpi=100)
        plt.close(fig)
        print(f"[viz] wrote challenge panels + scatters + 3-variant binned comparison to {fig_dir}")
    except Exception as e:
        print(f"[viz] binned figure skipped: {e}")


# ---------------------------------------------------------------- report
def _fmt(x, nd=2):
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    if x == float("inf"):
        return "inf"
    return f"{x:.{nd}f}"


def _ms(m, s, nd=2):
    if m is None:
        return "n/a"
    return f"{_fmt(m, nd)} ± {_fmt(s, nd)}"


def _bin_label(b):
    hi = "inf" if b["hi"] == float("inf") else f"{int(b['hi'])}"
    return f"{int(b['lo'])}–{hi}"


def _write_report(results, path):
    meta = results["meta"]
    ev = meta["evidence_valid"]
    A, B = results["context"]["A"], results["context"]["B"]
    ora = results["context"]["oracle"]
    V = {k: results["variants"][k]["summary"] for k in TABLE_KEYS}
    tr = meta["config"]["train"]
    hs, sc, wm = tr.get("loss_tail_start"), tr.get("loss_tail_scale"), tr.get("loss_tail_max")
    L = []
    L.append("# DepthWizard — PHASE 4 results (calibrated tail-weighted loss)\n")
    L.append("_Generated by `scripts/run_phase4_tail_weighted.py`. Every number is measured "
             "by that run; placeholders read `n/a`. Controlled single-variable test: "
             "C_log1p vs the SAME model with a THRESHOLDED tail-weighted masked-L1 "
             "(`loss_type=tail_weighted`), all else identical. The aggressive Phase-3 "
             "`height_weighted` model is retained as an ablation (§24)._\n")
    if not ev:
        L.append("> # ⚠️ NOT VALID EVIDENCE — synthetic/fake-depth smoke run. "
                 "Numbers only prove the code executes.\n")
    L.append("## Run metadata\n")
    L.append(f"- data source: `{meta['source']}` | evidence valid: **{ev}** | "
             f"device: `{meta['device']}`")
    L.append(f"- depth prior: `{meta['depth_model']}` | depth cache: "
             f"`{meta['depth_cache']}` (reused from Phase-1)")
    L.append(f"- split (city-held-out): train `{meta['train_cities']}` / in-domain "
             f"val `{meta['val_cities']}` / **test `{meta['test_cities']}`**")
    L.append(f"- tiles: train={meta['n_train']}, val={meta['n_val']}, test={meta['n_test']} "
             f"| seeds: `{meta['seeds']}` | head params: {V['C_none'].get('n_params')} "
             f"| runtime: {meta['elapsed_s']}s")
    L.append(f"- **calibrated tail weight** (NEW): `w(h)=min(1+max(h-{hs:g},0)/{sc:g}, {wm:g})` "
             f"on PHYSICAL height; ALL params JAX-train-derived (h_start ≈ ~14 m ceiling / "
             f"P92 all-pixel; sat @ {hs + (wm-1)*sc:g} m). See `runs/phase4_diag/`.")
    L.append(f"- aggressive ablation weight (Phase-3): `w(h)=min(1+max(h,0)/"
             f"{tr.get('loss_weight_scale'):g}, {tr.get('loss_weight_max'):g})`.\n")
    if meta["source"] == "hf_blocks":
        L.append("> **Provenance caveat:** unofficial HF mirror `JasonXF/DFC2019-10k`, "
                 "preprocessed nDSM (ground floored to 0). Valid feasibility evidence; "
                 "re-confirm on official IEEE GRSS DFC2019 before external reporting.\n")

    # §23 required 5-model comparison table
    L.append("## Comparison table (§23) — mean ± std over seeds\n")
    L.append("| Model | In-domain MAE | In-domain RMSE | Cross-city MAE | Cross-city RMSE "
             "| Building MAE (xcity) | Building RMSE (xcity) |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    L.append(f"| B affine | {_fmt(_get(B['indomain']['aggregate']['all'],'mae_pooled'))} "
             f"| {_fmt(_get(B['indomain']['aggregate']['all'],'rmse_pooled'))} "
             f"| {_fmt(_get(B['xcity']['aggregate']['all'],'mae_pooled'))} "
             f"| {_fmt(_get(B['xcity']['aggregate']['all'],'rmse_pooled'))} "
             f"| {_fmt(_get(B['xcity']['aggregate'].get('building',{}),'mae_pooled'))} "
             f"| {_fmt(_get(B['xcity']['aggregate'].get('building',{}),'rmse_pooled'))} |")
    labels = {"C_none": "C none", "C_log1p": "C log1p (control)",
              "C_log1p_weighted": "C log1p weighted (aggressive)",
              "C_log1p_tail_weighted": "**C log1p tail-weighted (NEW)**"}
    for key in TABLE_KEYS:
        S = V[key]
        L.append(f"| {labels[key]} "
                 f"| {_ms(S['indomain_all_mae_pooled_mean'], S['indomain_all_mae_pooled_std'])} "
                 f"| {_ms(S['indomain_all_rmse_pooled_mean'], S['indomain_all_rmse_pooled_std'])} "
                 f"| {_ms(S['xcity_all_mae_pooled_mean'], S['xcity_all_mae_pooled_std'])} "
                 f"| {_ms(S['xcity_all_rmse_pooled_mean'], S['xcity_all_rmse_pooled_std'])} "
                 f"| {_ms(S['xcity_building_mae_pooled_mean'], S['xcity_building_mae_pooled_std'])} "
                 f"| {_ms(S['xcity_building_rmse_pooled_mean'], S['xcity_building_rmse_pooled_std'])} |")
    L.append(f"| oracle affine (UB) "
             f"| {_fmt(_get(ora['indomain']['aggregate']['all'],'mae_pooled'))} "
             f"| {_fmt(_get(ora['indomain']['aggregate']['all'],'rmse_pooled'))} "
             f"| {_fmt(_get(ora['xcity']['aggregate']['all'],'mae_pooled'))} "
             f"| {_fmt(_get(ora['xcity']['aggregate']['all'],'rmse_pooled'))} "
             f"| {_fmt(_get(ora['xcity']['aggregate'].get('building',{}),'mae_pooled'))} "
             f"| {_fmt(_get(ora['xcity']['aggregate'].get('building',{}),'rmse_pooled'))} |")
    L.append(f"\n_A · raw-depth cross-city Pearson r (signal check): "
             f"{_fmt(_get(A['xcity']['aggregate']['all'],'pearson_mean'))}. Building is the "
             f"PRIMARY analysis (§19), not pooled all-pixel MAE._\n")

    # per-seed transparency (never hide a disagreeing seed)
    L.append("## Per-seed cross-city all-pixel MAE (transparency, §19)\n")
    L.append("| variant | " + " | ".join(f"seed {s}" for s in meta["seeds"]) + " | mean ± std |")
    L.append("|---|" + "|".join(["---"] * (len(meta["seeds"]) + 1)) + "|")
    for key in TABLE_KEYS:
        ps = results["variants"][key]["per_seed"]
        cells = [_fmt(_get(p["xcity"]["aggregate"]["all"], "mae_pooled")) for p in ps]
        S = V[key]
        L.append(f"| {key} | " + " | ".join(cells) +
                 f" | {_ms(S['xcity_all_mae_pooled_mean'], S['xcity_all_mae_pooled_std'])} |")
    L.append("")

    # the decisive isolation: building error by height bin, control vs tail (vs aggressive)
    L.append("## Cross-city BUILDING MAE & bias by GT-height bin — the loss isolation (§19,§22)\n")
    L.append("_The key question: does the calibrated tail weight lift the tall bins while "
             "leaving the dominant 0–15 m bins ≈ the C_log1p control (unlike the aggressive "
             "weight, which regressed them)? bias < 0 = under-prediction (ceiling signature)._\n")
    bl = V["C_log1p"]["xcity_binned_building"]
    ba = V["C_log1p_weighted"]["xcity_binned_building"]
    bt = V["C_log1p_tail_weighted"]["xcity_binned_building"]
    L.append("| GT bin (m) | px | C_log1p MAE | aggr MAE | **tail MAE** | ΔMAE tail−ctrl "
             "| C_log1p bias | aggr bias | **tail bias** |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for i in range(min(len(bl), len(ba), len(bt))):
        dm = (bl[i]["mae_mean"] - bt[i]["mae_mean"]) if (bl[i]["mae_mean"] is not None
              and bt[i]["mae_mean"] is not None) else None
        arrow = ""
        if dm is not None:
            arrow = " ✅" if dm > 0.05 else (" ❌" if dm < -0.05 else "")
        L.append(f"| {_bin_label(bl[i])} | {bl[i]['n_pixels']:,} "
                 f"| {_fmt(bl[i]['mae_mean'])} | {_fmt(ba[i]['mae_mean'])} | {_fmt(bt[i]['mae_mean'])} "
                 f"| {_fmt(dm)}{arrow} "
                 f"| {_fmt(bl[i]['bias_mean'])} | {_fmt(ba[i]['bias_mean'])} | {_fmt(bt[i]['bias_mean'])} |")
    L.append("\n_ΔMAE tail−ctrl = C_log1p − tail (positive = tail better). ✅/❌ mark "
             "|ΔMAE|>0.05 m. The 0–15 m rows should be ≈0 (protected); the >15 m rows "
             "positive (lifted). Compare the aggressive column to see what over-correction "
             "looked like._\n")

    # in-domain sanity + building
    L.append("## In-domain (held-out TRAIN-city tiles) — mean ± std\n")
    L.append("| variant | all MAE | all RMSE | building MAE |")
    L.append("|---|---|---|---|")
    for key in TABLE_KEYS:
        S = V[key]
        L.append(f"| {key} "
                 f"| {_ms(S['indomain_all_mae_pooled_mean'], S['indomain_all_mae_pooled_std'])} "
                 f"| {_ms(S['indomain_all_rmse_pooled_mean'], S['indomain_all_rmse_pooled_std'])} "
                 f"| {_ms(S['indomain_building_mae_pooled_mean'], S['indomain_building_mae_pooled_std'])} |")
    L.append("")

    # per-variant auditable verdict
    L.append("## Per-variant auditable verdict (decide(), context only — keys on overall "
             "cross-city MAE, which is dominated by ground pixels)\n")
    for key, _, _, label in VARIANTS:
        v = results["variants"][key]["verdict"]
        L.append(f"- **{label}: {v['verdict']}** — {v['summary']}")
    L.append("")

    # comparison summary: tail vs control (primary) + tail vs aggressive (§24)
    cmp = results["comparison"]
    L.append("## Comparison summary — calibrated tail vs C_log1p control (the isolated change)\n")
    L.append(f"- cross-city all MAE Δ (ctrl−tail): **{_fmt(cmp['d_all_mae'])} m** (pos = tail better)")
    L.append(f"- cross-city building MAE Δ (ctrl−tail): **{_fmt(cmp['d_bldg_mae'])} m**")
    L.append(f"- **low 0–{int(CEILING)} m** building MAE: ctrl={_fmt(cmp['low_mae_ctrl'])} "
             f"tail={_fmt(cmp['low_mae_tail'])} (Δ {_fmt(cmp['d_low_bldg_mae'])} m; ≈0 = regime protected)")
    L.append(f"- low 0–{int(CEILING)} m building bias: ctrl={_fmt(cmp['low_bias_ctrl'])} "
             f"tail={_fmt(cmp['low_bias_tail'])} (aggressive drove this strongly positive)")
    L.append(f"- **tall >{int(CEILING)} m** building MAE: ctrl={_fmt(cmp['tall_mae_ctrl'])} "
             f"tail={_fmt(cmp['tall_mae_tail'])} (Δ {_fmt(cmp['d_tall_bldg_mae'])} m; pos = tail better)")
    L.append(f"- tall >{int(CEILING)} m building signed bias: ctrl={_fmt(cmp['tall_bias_ctrl'])} "
             f"tail={_fmt(cmp['tall_bias_tail'])} (Δ {_fmt(cmp['d_tall_bldg_bias'])} m; pos = less under-prediction)")
    L.append(f"- beats affine cross-city (≥10% all-MAE cut over B={_fmt(cmp['b_all_mae'])})? "
             f"ctrl: **{cmp['ctrl_beats_B']}**, tail: **{cmp['tail_beats_B']}**\n")

    L.append("### §20 success signals — the trade-off (tall gain AND 0–15 m stability AND xcity)\n")
    for k, desc in [("s_tall_mae_better", "tall-bin building MAE improved vs control"),
                    ("s_tall_bias_better", "tall-bin building bias less negative vs control"),
                    ("s_low_stable", "0–15 m building MAE stayed ≈ control (within 0.5 m)"),
                    ("s_building_ok", "overall building MAE not worse than control (within 0.3 m)"),
                    ("s_xcity_useful", "cross-city all-MAE still beats affine ≥10%"),
                    ("s_variance_ok", "seed variance not worse than control")]:
        L.append(f"- {'✅' if cmp.get(k) else '❌'} {desc}")
    L.append("")

    # §24 explicit: did we avoid the aggressive failure modes?
    L.append("### §24 — did the calibrated weight AVOID the aggressive failure modes?\n")
    for k, desc in [("avoid_short_overpred", "no short-building over-prediction (low-bin bias not strongly positive)"),
                    ("avoid_broad_bias", "no broad positive bias (overall building bias ≈ control)"),
                    ("avoid_rmse_explosion", "no in-domain RMSE explosion (tail ≈ control, not the aggressive blow-up)"),
                    ("avoid_instability", "no loss instability (seed std controlled)")]:
        L.append(f"- {'✅' if cmp.get(k) else '❌'} {desc}")
    L.append(f"\n_Context: aggressive in-domain RMSE was {_fmt(cmp['agg_indomain_rmse'])} m "
             f"(vs control {_fmt(cmp['ctrl_indomain_rmse'])} m); tail = {_fmt(cmp['tail_indomain_rmse'])} m. "
             f"Aggressive low-bin bias = {_fmt(cmp['low_bias_agg'])} m (strongly positive = the "
             f"over-prediction); tail low-bin bias = {_fmt(cmp['low_bias_tail'])} m._\n")

    L.append("> This run does NOT establish ISRO/Indian-sensor readiness and does NOT "
             "build the product. It is one controlled modification, city-held-out on the "
             "unofficial DFC2019 mirror only. STOP for human review.\n")
    Path(path).write_text("\n".join(L), encoding="utf-8")
    print(f"[report] wrote {path}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase4_tail_weighted.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="force synthetic data + tiny run (NOT evidence)")
    ap.add_argument("--allow-fake-depth", action="store_true",
                    help="fabricate depth from GT if torch/transformers absent (NOT evidence)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Windows consoles default to cp1252 and choke on the Δ/±/≈ glyphs in the console
    # summary; make stdout tolerant (the report .md is already written UTF-8).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    cfg = load_config(args.config)
    if args.smoke:
        cfg.data.source = "synthetic"
        cfg.seeds = [0]
        cfg.train = replace(cfg.train, epochs=2)
    out_dir = Path(args.out or cfg.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    has_torch = _has_torch()
    edges = DEFAULT_HEIGHT_EDGES
    t0 = time.time()

    source, records = fetch.resolve_records(cfg)
    is_synth = (source == "synthetic")

    use_fake = args.allow_fake_depth or (not has_torch)
    if use_fake and not args.allow_fake_depth:
        print("[warn] torch/transformers unavailable. Baseline C (the whole point of "
              "Phase-4) cannot run. Re-run with --allow-fake-depth for a NON-EVIDENCE "
              "plumbing smoke only. Aborting.")
        sys.exit(2)
    from depthwizard.depth.fake import FakeDepth
    fake = FakeDepth(seed=cfg.split.seed) if use_fake else None
    depth_model = None
    if not use_fake:
        from depthwizard.depth.depth_anything import DepthAnythingV2
        depth_model = DepthAnythingV2(
            cfg.depth.model_id, cfg.depth.input_size,
            cache_dir=cfg.depth.cache_dir if cfg.depth.use_cache else None,
            use_cache=cfg.depth.use_cache)  # <-- REUSE Phase-1 cache verbatim

    if is_synth:
        train, val, test = fetch.synthetic_samples(cfg)
        for grp in (train, val, test):
            _attach_depth(grp, cfg, depth_model, fake)
    else:
        tr_rec, va_rec, te_rec = datasets.split_by_city(
            records, cfg.split.train_cities, cfg.split.val_cities,
            cfg.split.test_cities, cfg.split.val_fraction_within_train_city,
            cfg.split.seed, cfg.data.max_tiles_per_city)
        print(f"[split] train={len(tr_rec)} val={len(va_rec)} test={len(te_rec)}")
        train = _materialize(tr_rec, cfg, depth_model, fake)
        val = _materialize(va_rec, cfg, depth_model, fake)
        test = _materialize(te_rec, cfg, depth_model, fake)

    if not train or not test:
        print("[error] empty train/test split."); sys.exit(3)
    if not has_torch:
        print("[error] torch required for Phase-4 (Baseline C). Aborting."); sys.exit(4)

    evidence_valid = (not is_synth) and (fake is None)
    device = "cuda" if _cuda() else "cpu"

    # ---- anchors: A (signal), B (bar), oracle (upper bound) ----
    from depthwizard.models.affine import RawDepth, GlobalAffine
    a = RawDepth().fit(train)
    A = {"indomain": evaluate_estimator(a, val, cfg, "indomain", bin_edges=edges),
         "xcity": evaluate_estimator(a, test, cfg, "xcity", bin_edges=edges)}
    b = GlobalAffine(max_pixels=cfg.train.max_train_pixels_affine,
                     seed=cfg.split.seed).fit(train)
    B = {"indomain": evaluate_estimator(b, val, cfg, "indomain", bin_edges=edges),
         "xcity": evaluate_estimator(b, test, cfg, "xcity", bin_edges=edges)}
    print(f"[B] a={b.a:.4g} b={b.b:.4g} | xcity all MAE="
          f"{_get(B['xcity']['aggregate']['all'],'mae_pooled')}")
    oracle = {"indomain": evaluate_oracle(val, cfg, "indomain"),
              "xcity": evaluate_oracle(test, cfg, "xcity")}

    # ---- the controlled comparison: 4 C variants (only loss/transform differ) ----
    variants = {}
    fitted = {}
    for key, transform, loss_type, _label in VARIANTS:
        ps, fit0 = _fit_eval_variant(transform, loss_type, cfg, train, val, test,
                                     edges, keep_fitted=True)
        S = _summarize(ps)
        variants[key] = {"per_seed": ps, "summary": S,
                         "verdict": _variant_verdict(A, B, S)}
        fitted[key] = fit0

    Sl = variants["C_log1p"]["summary"]           # control
    Sa = variants["C_log1p_weighted"]["summary"]  # aggressive
    St = variants["C_log1p_tail_weighted"]["summary"]  # NEW calibrated tail

    # ---- comparison scalars: tail vs control (primary) + §24 aggressive-failure checks ----
    b_all_mae = _get(B["xcity"]["aggregate"]["all"], "mae_pooled")
    b_bldg_mae = _get(B["xcity"]["aggregate"].get("building", {}), "mae_pooled")

    def _beats_B(S):
        c = S["xcity_all_mae_pooled_mean"]
        return bool(b_all_mae and c is not None and (b_all_mae - c) / b_all_mae >= 0.10)

    def _d(S1, S2, key):  # S1 - S2
        v1, v2 = S1[key], S2[key]
        return (v1 - v2) if (v1 is not None and v2 is not None) else None

    tall_mae_l, tall_bias_l = _pool_bldg(Sl, want_tall=True)
    tall_mae_t, tall_bias_t = _pool_bldg(St, want_tall=True)
    low_mae_l, low_bias_l = _pool_bldg(Sl, want_tall=False)
    low_mae_t, low_bias_t = _pool_bldg(St, want_tall=False)
    low_mae_a, low_bias_a = _pool_bldg(Sa, want_tall=False)

    def _sub(x, y):
        return (x - y) if (x is not None and y is not None) else None

    d_tall_mae = _sub(tall_mae_l, tall_mae_t)      # pos = tail better
    d_tall_bias = _sub(tall_bias_t, tall_bias_l)   # pos = tail less negative
    d_low_mae = _sub(low_mae_t, low_mae_l)         # ~0 = protected (tail - ctrl)
    d_bldg_mae = _d(Sl, St, "xcity_building_mae_pooled_mean")  # ctrl - tail

    ovb_l = Sl["xcity_building_pearson_mean"]
    ctrl_id_rmse = Sl["indomain_all_rmse_pooled_mean"]
    agg_id_rmse = Sa["indomain_all_rmse_pooled_mean"]
    tail_id_rmse = St["indomain_all_rmse_pooled_mean"]
    var_l = Sl["xcity_all_mae_pooled_std"]
    var_t = St["xcity_all_mae_pooled_std"]
    # overall building bias proxy: pool low+tall
    ovbias_l = _mean_std([low_bias_l, tall_bias_l])[0]
    ovbias_t = _mean_std([low_bias_t, tall_bias_t])[0]

    comparison = {
        "b_all_mae": b_all_mae, "b_bldg_mae": b_bldg_mae,
        "d_all_mae": _d(Sl, St, "xcity_all_mae_pooled_mean"),
        "d_bldg_mae": d_bldg_mae,
        "tall_mae_ctrl": tall_mae_l, "tall_mae_tail": tall_mae_t, "d_tall_bldg_mae": d_tall_mae,
        "tall_bias_ctrl": tall_bias_l, "tall_bias_tail": tall_bias_t, "d_tall_bldg_bias": d_tall_bias,
        "low_mae_ctrl": low_mae_l, "low_mae_tail": low_mae_t, "d_low_bldg_mae": d_low_mae,
        "low_bias_ctrl": low_bias_l, "low_bias_tail": low_bias_t, "low_bias_agg": low_bias_a,
        "ctrl_beats_B": _beats_B(Sl), "tail_beats_B": _beats_B(St),
        "ctrl_indomain_rmse": ctrl_id_rmse, "agg_indomain_rmse": agg_id_rmse,
        "tail_indomain_rmse": tail_id_rmse,
        # §20 success signals (the trade-off; each necessary, none alone sufficient)
        "s_tall_mae_better": bool(d_tall_mae is not None and d_tall_mae > 0),
        "s_tall_bias_better": bool(d_tall_bias is not None and d_tall_bias > 0),
        "s_low_stable": bool(d_low_mae is not None and abs(d_low_mae) <= 0.5),
        "s_building_ok": bool(d_bldg_mae is not None and d_bldg_mae >= -0.3),
        "s_xcity_useful": _beats_B(St),
        "s_variance_ok": bool(var_l is not None and var_t is not None and var_t <= var_l + 0.1),
        # §24 aggressive-failure-avoidance checks
        "avoid_short_overpred": bool(low_bias_t is not None and low_bias_t <= 2.0),
        "avoid_broad_bias": bool(ovbias_l is not None and ovbias_t is not None
                                 and (ovbias_t - ovbias_l) <= 1.5),
        "avoid_rmse_explosion": bool(tail_id_rmse is not None and ctrl_id_rmse is not None
                                     and tail_id_rmse <= ctrl_id_rmse + 2.0),
        "avoid_instability": bool(var_t is not None and var_l is not None and var_t <= var_l + 0.5),
    }

    results = {
        "meta": {
            "source": source, "evidence_valid": evidence_valid, "device": device,
            "depth_model": (cfg.depth.model_id if fake is None else "FAKE_STUB"),
            "fake_depth": fake is not None, "depth_cache": cfg.depth.cache_dir,
            "train_cities": cfg.split.train_cities, "val_cities": cfg.split.val_cities,
            "test_cities": cfg.split.test_cities, "n_train": len(train),
            "n_val": len(val), "n_test": len(test), "seeds": list(cfg.seeds),
            "ceiling": CEILING, "height_edges": edges, "has_torch": has_torch,
            "elapsed_s": round(time.time() - t0, 1), "config": config_to_dict(cfg),
        },
        "context": {"A": A, "B": B, "oracle": oracle},
        "variants": variants,
        "comparison": comparison,
    }

    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    _write_report(results, str(out_dir / "PHASE4_COMPARISON.md"))
    _make_figures(fitted, test, cfg, fig_dir,
                  {k: variants[k]["summary"] for k in variants}, evidence_valid)

    print("\n" + "=" * 72)
    print("PHASE-4 CHECKPOINT — calibrated tail weight vs C_log1p control")
    print(f"  cross-city all MAE:  ctrl={_fmt(Sl['xcity_all_mae_pooled_mean'])}±"
          f"{_fmt(Sl['xcity_all_mae_pooled_std'])}  "
          f"tail={_fmt(St['xcity_all_mae_pooled_mean'])}±{_fmt(St['xcity_all_mae_pooled_std'])}  "
          f"(B={_fmt(b_all_mae)})")
    print(f"  cross-city bldg MAE: ctrl={_fmt(Sl['xcity_building_mae_pooled_mean'])}  "
          f"tail={_fmt(St['xcity_building_mae_pooled_mean'])}  (B={_fmt(b_bldg_mae)})")
    print(f"  LOW 0–{CEILING:.0f}m bldg MAE:  ctrl={_fmt(low_mae_l)}  tail={_fmt(low_mae_t)}  "
          f"(Δ={_fmt(d_low_mae)}; ≈0=protected)  | bias ctrl={_fmt(low_bias_l)} tail={_fmt(low_bias_t)} agg={_fmt(low_bias_a)}")
    print(f"  TALL >{CEILING:.0f}m bldg MAE: ctrl={_fmt(tall_mae_l)}  tail={_fmt(tall_mae_t)}  "
          f"(Δ={_fmt(d_tall_mae)}, pos=better) | bias ctrl={_fmt(tall_bias_l)} tail={_fmt(tall_bias_t)}")
    print(f"  §20 signals: tall_mae↓={comparison['s_tall_mae_better']} "
          f"tall_bias↑={comparison['s_tall_bias_better']} low_stable={comparison['s_low_stable']} "
          f"bldg_ok={comparison['s_building_ok']} xcity_useful={comparison['s_xcity_useful']} "
          f"var_ok={comparison['s_variance_ok']}")
    print(f"  §24 avoids: short_overpred={comparison['avoid_short_overpred']} "
          f"broad_bias={comparison['avoid_broad_bias']} rmse_ok={comparison['avoid_rmse_explosion']} "
          f"stable={comparison['avoid_instability']}")
    if not evidence_valid:
        print("\n[!] evidence_valid=False — plumbing only; NOT a real result.")
    print("=" * 72)
    print("STOP: human reviews these numbers + error maps before any GO decision.")


if __name__ == "__main__":
    main()
