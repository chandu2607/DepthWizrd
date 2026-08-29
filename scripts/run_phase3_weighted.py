#!/usr/bin/env python
"""DepthWizard PHASE-3 MODIFY-AGAIN orchestrator -- controlled single-variable test.

Question: does a physical-height-aware LOSS WEIGHTING, applied on top of the
log1p target transform, improve the learned head's BUILDING / tall-structure
accuracy (the product-critical regime) WITHOUT wrecking the dominant low-height
bins or cross-city generalization -- and does it beat trivial global-affine?

Protocol (matches the Phase-3 spec; identical to Phase-2 except the loss):
  * Reuse the EXACT Phase-1/2 data + city-held-out split + DA-V2 depth cache.
  * Anchors: Baseline A (raw depth, signal), B (global affine, bar), oracle (UB).
  * Baseline C trained THREE ways per seed, ONLY the labeled variable differs:
        C_none            = transform "none",  loss "standard"  (Phase-1 original)
        C_log1p           = transform "log1p", loss "standard"  (Phase-2 reference)
        C_log1p_weighted  = transform "log1p", loss "height_weighted" (NEW, tested)
    The KEY comparison is C_log1p vs C_log1p_weighted -> isolates the loss change.
  * Strict eval: all/building/non-building, in-domain AND cross-city, per-GT-height
    bin MAE/bias. BOTH seeds -> mean +/- std; never a single favorable seed.
  * Weight w(h)=min(1+max(h,0)/scale, w_max) on PHYSICAL height, derived ONLY from
    training stats (scale~train building median); see scripts/phase3_weight_diagnostic.py.

Does NOT build the product. STOPS at the Phase-3 checkpoint.

Usage:
  python scripts/run_phase3_weighted.py --config configs/phase3_weighted.yaml
  python scripts/run_phase3_weighted.py --smoke --allow-fake-depth   # plumbing only
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

# The three C variants: (key, target_transform, loss_type, human label)
VARIANTS = [
    ("C_none", "none", "standard", "C · original (none)"),
    ("C_log1p", "log1p", "standard", "C · log1p"),
    ("C_log1p_weighted", "log1p", "height_weighted", "C · log1p + height-weighted loss"),
]


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

    ONLY these two knobs vary across variants; everything else (arch, lr, epochs,
    res, seeds, data, split) is identical -> any delta is attributable to the loss.
    Returns (per_seed_list, fitted_seed0_or_None).
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


def _tall_bldg_mae(S):
    """Pixel-weighted building MAE pooled over the tall (>=CEILING) bins."""
    num = den = 0.0
    for bb in S["xcity_binned_building"]:
        if bb["lo"] >= CEILING and bb["mae_mean"] is not None and bb["n_pixels"]:
            num += bb["mae_mean"] * bb["n_pixels"]; den += bb["n_pixels"]
    return (num / den) if den else None


def _tall_bldg_bias(S):
    """Pixel-weighted building signed bias pooled over the tall (>=CEILING) bins."""
    num = den = 0.0
    for bb in S["xcity_binned_building"]:
        if bb["lo"] >= CEILING and bb["bias_mean"] is not None and bb["n_pixels"]:
            num += bb["bias_mean"] * bb["n_pixels"]; den += bb["n_pixels"]
    return (num / den) if den else None


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
    """§18 challenge panels + scatters (C_log1p vs C_log1p_weighted) and §19 the
    decisive per-height-bin building MAE/bias comparison."""
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
    # §18: for each challenging tall tile, render RGB|ref|pred|error for BOTH the
    # log1p reference and the weighted model, plus a pred-vs-ref scatter for both.
    focus = {k: fitted.get(k) for k in ("C_log1p", "C_log1p_weighted")}
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
    # §19: cross-city BUILDING MAE and signed bias per height bin, log1p vs weighted.
    try:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        colors = {"C_log1p": "#1f6feb", "C_log1p_weighted": "#c1440e"}
        for which, a, ylab in [("mae_mean", ax[0], "cross-city building MAE (m)"),
                               ("bias_mean", ax[1], "cross-city building signed bias (m)")]:
            for vname in ("C_log1p", "C_log1p_weighted"):
                bins = summaries[vname].get("xcity_binned_building", [])
                if not bins:
                    continue
                labels = [f"{int(b['lo'])}-{'inf' if b['hi']==float('inf') else int(b['hi'])}"
                          for b in bins]
                y = [b[which] if b[which] is not None else np.nan for b in bins]
                a.plot(labels, y, marker="o", label=vname, color=colors.get(vname))
            a.axhline(0, color="k", lw=0.6)
            a.axvspan(-0.5, 3.5, color="green", alpha=0.05)  # low bins (dominant)
            a.set_xlabel("GT height bin (m)"); a.set_ylabel(ylab)
            a.tick_params(axis="x", rotation=45); a.legend()
        ax[0].set_title("Where does building error live? (lower=better)")
        ax[1].set_title(f"Bias by bin (neg=under-predict; ceiling≈{CEILING:.0f} m){tag}")
        fig.suptitle("Phase-3: height-weighted vs log1p — cross-city building error by height")
        fig.tight_layout(); fig.savefig(fig_dir / "binned_building_log1p_vs_weighted.png", dpi=100)
        plt.close(fig)
        print(f"[viz] wrote challenge panels + scatters + binned comparison to {fig_dir}")
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
    V = {k: results["variants"][k]["summary"] for k, *_ in VARIANTS}
    scale = meta["config"]["train"].get("loss_weight_scale")
    wmax = meta["config"]["train"].get("loss_weight_max")
    L = []
    L.append("# DepthWizard — PHASE 3 MODIFY-AGAIN results (height-aware loss)\n")
    L.append("_Generated by `scripts/run_phase3_weighted.py`. Every number is measured "
             "by that run; placeholders read `n/a`. Controlled single-variable test: "
             "C_log1p vs the SAME model with a physical-height-weighted masked-L1 "
             "(`loss_type=height_weighted`), all else identical._\n")
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
    L.append(f"- weight: `w(h)=min(1+max(h,0)/{scale:g}, {wmax:g})` on PHYSICAL height, "
             f"training-derived (scale ≈ JAX-train building median 7.16 m); "
             f"see `runs/phase3_diag/weight_diagnostic.md`\n")
    if meta["source"] == "hf_blocks":
        L.append("> **Provenance caveat:** unofficial HF mirror `JasonXF/DFC2019-10k`, "
                 "preprocessed nDSM (ground floored to 0). Valid feasibility evidence; "
                 "re-confirm on official IEEE GRSS DFC2019 before external reporting.\n")

    # §20 required comparison table (the canonical 4-model table)
    L.append("## Comparison table (§20) — mean ± std over seeds\n")
    L.append("| Model | In-domain MAE | In-domain RMSE | Cross-city MAE | Cross-city RMSE "
             "| Building MAE (xcity) | Building RMSE (xcity) |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    L.append(f"| B affine | {_fmt(_get(B['indomain']['aggregate']['all'],'mae_pooled'))} "
             f"| {_fmt(_get(B['indomain']['aggregate']['all'],'rmse_pooled'))} "
             f"| {_fmt(_get(B['xcity']['aggregate']['all'],'mae_pooled'))} "
             f"| {_fmt(_get(B['xcity']['aggregate']['all'],'rmse_pooled'))} "
             f"| {_fmt(_get(B['xcity']['aggregate'].get('building',{}),'mae_pooled'))} "
             f"| {_fmt(_get(B['xcity']['aggregate'].get('building',{}),'rmse_pooled'))} |")
    labels = {"C_none": "C none", "C_log1p": "C log1p",
              "C_log1p_weighted": "**C log1p weighted**"}
    for key in ("C_none", "C_log1p", "C_log1p_weighted"):
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
             f"PRIMARY analysis (§16), not pooled all-pixel MAE._\n")

    # per-seed transparency (never hide a disagreeing seed)
    L.append("## Per-seed cross-city all-pixel MAE (transparency, §15)\n")
    L.append("| variant | " + " | ".join(f"seed {s}" for s in meta["seeds"]) + " | mean ± std |")
    L.append("|---|" + "|".join(["---"] * (len(meta["seeds"]) + 1)) + "|")
    for key in ("C_none", "C_log1p", "C_log1p_weighted"):
        ps = results["variants"][key]["per_seed"]
        cells = [_fmt(_get(p["xcity"]["aggregate"]["all"], "mae_pooled")) for p in ps]
        S = V[key]
        L.append(f"| {key} | " + " | ".join(cells) +
                 f" | {_ms(S['xcity_all_mae_pooled_mean'], S['xcity_all_mae_pooled_std'])} |")
    L.append("")

    # the decisive isolation: building error by height bin, log1p vs weighted
    L.append("## Cross-city BUILDING MAE & bias by GT-height bin — the loss isolation (§17,§19)\n")
    L.append("_The key comparison: does the weighted loss lift the tall bins WITHOUT "
             "hurting the dominant low bins? bias < 0 = under-prediction (ceiling signature)._\n")
    bl = V["C_log1p"]["xcity_binned_building"]
    bw = V["C_log1p_weighted"]["xcity_binned_building"]
    L.append("| GT bin (m) | px | C_log1p MAE | weighted MAE | ΔMAE | C_log1p bias | weighted bias | Δbias |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for i in range(min(len(bl), len(bw))):
        dm = (bl[i]["mae_mean"] - bw[i]["mae_mean"]) if (bl[i]["mae_mean"] is not None
              and bw[i]["mae_mean"] is not None) else None
        db = (bw[i]["bias_mean"] - bl[i]["bias_mean"]) if (bl[i]["bias_mean"] is not None
              and bw[i]["bias_mean"] is not None) else None  # +ve = less negative = better
        arrow = ""
        if dm is not None:
            arrow = " ✅" if dm > 0.05 else (" ❌" if dm < -0.05 else "")
        L.append(f"| {_bin_label(bl[i])} | {bl[i]['n_pixels']:,} "
                 f"| {_fmt(bl[i]['mae_mean'])} | {_fmt(bw[i]['mae_mean'])} | {_fmt(dm)}{arrow} "
                 f"| {_fmt(bl[i]['bias_mean'])} | {_fmt(bw[i]['bias_mean'])} | {_fmt(db)} |")
    L.append("\n_ΔMAE = C_log1p − weighted (positive = weighted better). "
             "Δbias = weighted − C_log1p (positive = less under-prediction). "
             "✅/❌ mark |ΔMAE|>0.05 m._\n")

    # in-domain sanity + building
    L.append("## In-domain (held-out TRAIN-city tiles) — mean ± std\n")
    L.append("| variant | all MAE | building MAE |")
    L.append("|---|---|---|")
    for key in ("C_none", "C_log1p", "C_log1p_weighted"):
        S = V[key]
        L.append(f"| {key} "
                 f"| {_ms(S['indomain_all_mae_pooled_mean'], S['indomain_all_mae_pooled_std'])} "
                 f"| {_ms(S['indomain_building_mae_pooled_mean'], S['indomain_building_mae_pooled_std'])} |")
    L.append("")

    # per-variant auditable verdict
    L.append("## Per-variant auditable verdict (decide(), context only — keys on overall "
             "cross-city MAE, which is dominated by ground pixels)\n")
    for key, _, _, label in VARIANTS:
        v = results["variants"][key]["verdict"]
        L.append(f"- **{label}: {v['verdict']}** — {v['summary']}")
    L.append("")

    # comparison summary (weighted vs log1p) + §21 success signals
    cmp = results["comparison"]
    L.append("## Comparison summary — weighted vs C_log1p (the isolated loss change)\n")
    L.append(f"- cross-city all MAE Δ: **{_fmt(cmp['d_all_mae'])} m** (pos = weighted better)")
    L.append(f"- cross-city building MAE Δ: **{_fmt(cmp['d_bldg_mae'])} m**")
    L.append(f"- cross-city tall-building (>{CEILING:.0f} m) MAE Δ: **{_fmt(cmp['d_tall_bldg_mae'])} m**")
    L.append(f"- cross-city tall-building signed bias Δ: **{_fmt(cmp['d_tall_bldg_bias'])} m** "
             f"(pos = less under-prediction; log1p={_fmt(cmp['tall_bias_log1p'])}, "
             f"weighted={_fmt(cmp['tall_bias_weighted'])})")
    L.append(f"- beats affine cross-city (≥10% all-MAE cut over B)? "
             f"log1p: **{cmp['log1p_beats_B']}**, weighted: **{cmp['weighted_beats_B']}**")
    L.append(f"- beats affine on BUILDINGS cross-city? "
             f"log1p: **{cmp['log1p_beats_B_bldg']}**, weighted: **{cmp['weighted_beats_B_bldg']}** "
             f"(B building MAE = {_fmt(cmp['b_bldg_mae'])})\n")
    L.append("### §21 success signals (success is NOT merely lower overall MAE)\n")
    for k, desc in [("s_building_improved", "building MAE decreased"),
                    ("s_tall_bias_better", "tall-bin bias less negative"),
                    ("s_tall_mae_better", "tall-bin MAE improved"),
                    ("s_xcity_useful", "cross-city still beats affine overall"),
                    ("s_variance_ok", "seed variance not worse than log1p")]:
        L.append(f"- {'✅' if cmp.get(k) else '❌'} {desc}")
    L.append("")

    L.append("> This run does NOT establish ISRO/Indian-sensor readiness and does NOT "
             "build the product. It is one controlled modification, city-held-out on the "
             "unofficial DFC2019 mirror only. STOP for human review.\n")
    Path(path).write_text("\n".join(L), encoding="utf-8")
    print(f"[report] wrote {path}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase3_weighted.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="force synthetic data + tiny run (NOT evidence)")
    ap.add_argument("--allow-fake-depth", action="store_true",
                    help="fabricate depth from GT if torch/transformers absent (NOT evidence)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Windows consoles default to cp1252 and choke on the Δ/± glyphs in the
    # console summary; make stdout tolerant (the report .md is already UTF-8).
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
              "Phase-3) cannot run. Re-run with --allow-fake-depth for a NON-EVIDENCE "
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
        print("[error] torch required for Phase-3 (Baseline C). Aborting."); sys.exit(4)

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

    # ---- the controlled comparison: 3 C variants (only loss/transform differ) ----
    variants = {}
    fitted = {}
    for key, transform, loss_type, _label in VARIANTS:
        ps, fit0 = _fit_eval_variant(transform, loss_type, cfg, train, val, test,
                                     edges, keep_fitted=True)
        S = _summarize(ps)
        variants[key] = {"per_seed": ps, "summary": S,
                         "verdict": _variant_verdict(A, B, S)}
        fitted[key] = fit0

    Sl = variants["C_log1p"]["summary"]
    Sw = variants["C_log1p_weighted"]["summary"]

    # ---- comparison scalars: weighted vs log1p (the isolated loss change) ----
    b_all_mae = _get(B["xcity"]["aggregate"]["all"], "mae_pooled")
    b_bldg_mae = _get(B["xcity"]["aggregate"].get("building", {}), "mae_pooled")

    def _beats_B(S):
        c = S["xcity_all_mae_pooled_mean"]
        return bool(b_all_mae and c is not None and (b_all_mae - c) / b_all_mae >= 0.10)

    def _beats_B_bldg(S):
        c = S["xcity_building_mae_pooled_mean"]
        return bool(b_bldg_mae and c is not None and (b_bldg_mae - c) / b_bldg_mae >= 0.10)

    def _d(a_key):
        la, wa = Sl[a_key], Sw[a_key]
        return (la - wa) if (la is not None and wa is not None) else None

    tall_l, tall_w = _tall_bldg_mae(Sl), _tall_bldg_mae(Sw)
    tbias_l, tbias_w = _tall_bldg_bias(Sl), _tall_bldg_bias(Sw)
    d_bldg_mae = _d("xcity_building_mae_pooled_mean")
    d_tall_mae = (tall_l - tall_w) if (tall_l is not None and tall_w is not None) else None
    d_tall_bias = (tbias_w - tbias_l) if (tbias_l is not None and tbias_w is not None) else None
    var_l = Sl["xcity_all_mae_pooled_std"]
    var_w = Sw["xcity_all_mae_pooled_std"]
    comparison = {
        "d_all_mae": _d("xcity_all_mae_pooled_mean"),
        "d_bldg_mae": d_bldg_mae,
        "d_tall_bldg_mae": d_tall_mae,
        "d_tall_bldg_bias": d_tall_bias,
        "tall_bias_log1p": tbias_l, "tall_bias_weighted": tbias_w,
        "tall_mae_log1p": tall_l, "tall_mae_weighted": tall_w,
        "log1p_beats_B": _beats_B(Sl), "weighted_beats_B": _beats_B(Sw),
        "log1p_beats_B_bldg": _beats_B_bldg(Sl), "weighted_beats_B_bldg": _beats_B_bldg(Sw),
        "b_bldg_mae": b_bldg_mae,
        # §21 success signals (each necessary, none alone sufficient)
        "s_building_improved": bool(d_bldg_mae is not None and d_bldg_mae > 0),
        "s_tall_bias_better": bool(d_tall_bias is not None and d_tall_bias > 0),
        "s_tall_mae_better": bool(d_tall_mae is not None and d_tall_mae > 0),
        "s_xcity_useful": _beats_B(Sw),
        "s_variance_ok": bool(var_l is not None and var_w is not None and var_w <= var_l + 0.1),
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
    _write_report(results, str(out_dir / "PHASE3_COMPARISON.md"))
    _make_figures(fitted, test, cfg, fig_dir,
                  {k: variants[k]["summary"] for k in variants}, evidence_valid)

    print("\n" + "=" * 72)
    print("PHASE-3 MODIFY-AGAIN CHECKPOINT — height-weighted loss vs C_log1p")
    print(f"  cross-city all MAE:  log1p={_fmt(Sl['xcity_all_mae_pooled_mean'])}±"
          f"{_fmt(Sl['xcity_all_mae_pooled_std'])}  "
          f"weighted={_fmt(Sw['xcity_all_mae_pooled_mean'])}±{_fmt(Sw['xcity_all_mae_pooled_std'])}  "
          f"(B={_fmt(b_all_mae)})")
    print(f"  cross-city bldg MAE: log1p={_fmt(Sl['xcity_building_mae_pooled_mean'])}  "
          f"weighted={_fmt(Sw['xcity_building_mae_pooled_mean'])}  (B={_fmt(b_bldg_mae)})")
    print(f"  tall (>{CEILING:.0f}m) bldg MAE:  log1p={_fmt(tall_l)}  weighted={_fmt(tall_w)}  "
          f"(Δ={_fmt(d_tall_mae)})")
    print(f"  tall (>{CEILING:.0f}m) bldg bias: log1p={_fmt(tbias_l)}  weighted={_fmt(tbias_w)}  "
          f"(Δ={_fmt(d_tall_bias)}, pos=better)")
    print(f"  §21 signals: bldg↓={comparison['s_building_improved']} "
          f"tall_bias↑={comparison['s_tall_bias_better']} "
          f"tall_mae↓={comparison['s_tall_mae_better']} "
          f"xcity_useful={comparison['s_xcity_useful']} "
          f"var_ok={comparison['s_variance_ok']}")
    if not evidence_valid:
        print("\n[!] evidence_valid=False — plumbing only; NOT a real result.")
    print("=" * 72)
    print("STOP: human reviews these numbers + error maps before any GO decision.")


if __name__ == "__main__":
    main()
