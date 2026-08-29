#!/usr/bin/env python
"""DepthWizard PHASE-5 orchestrator -- controlled RECONSTRUCTION-FIDELITY test.

Question (§18): does giving the fusion head a larger effective receptive field / stronger
multiscale reconstruction (ONE extra encoder+decoder level: SmallFusionUNet -> SmallReconUNet)
let it place the correct height across the FULL FOOTPRINT of a large structure, instead of
collapsing the body and spiking the edges (the Phase-4 failure) -- WITHOUT the aggressive
weight's upward distribution shift?

Protocol (identical to Phase-1..4 except the ONE architectural variable; §5):
  * Reuse the EXACT data + city-held-out split + DA-V2 depth cache.
  * Anchors: A (raw depth, signal), B (global affine, bar), oracle (upper bound).
  * Baseline C trained TWO ways per seed; ONLY `arch` differs (transform=log1p,
    loss=standard, lr/epochs/batch/width/res all identical):
        C_log1p        = arch "unet3"  (PRIMARY CONTROL §4, kept EXACTLY)
        C_log1p_recon  = arch "unet4"  (NEW, one extra level, ~2x receptive field §10)
    KEY comparison: C_log1p vs C_log1p_recon (isolates reconstruction capacity).
  * The Phase-3 aggressive + Phase-4 tail ablations are NOT retrained (§17); their
    numbers are cited from runs/phase4_tail_weighted/results.json for context.
  * Strict eval: all/building/non-building, in-domain AND cross-city, per-GT-height-bin
    MAE/bias. BOTH seeds -> mean +/- std; never a single favorable seed.
  * §8 hardware: measure per-variant peak VRAM + runtime; confirm the deeper net still
    fits the RTX 3050 (~4 GB).

CRITICAL (§24): a lower signed bias is NOT success if RMSE/MAE rise -- that is the Phase-4
compensating-artifact pattern (edge overshoot cancels body collapse). This script inspects
MAE and RMSE FIRST and explicitly flags the bias-down/RMSE-up failure mode (§27F).

Does NOT build the product. STOPS at the Phase-5 scientific checkpoint (§30/§32).

Usage:
  python scripts/run_phase5_recon.py --config configs/phase5_recon.yaml
  python scripts/run_phase5_recon.py --smoke --allow-fake-depth   # plumbing only
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

CEILING = 14.0  # the observed Phase-1 Baseline-C prediction ceiling

# The two C variants under test: (key, target_transform, loss_type, arch, human label).
# ONLY `arch` differs -> any delta is attributable to reconstruction capacity.
VARIANTS = [
    ("C_log1p", "log1p", "standard", "unet3", "C · log1p (control, unet3)"),
    ("C_log1p_recon", "log1p", "standard", "unet4", "C · log1p + recon (unet4, NEW)"),
]
TABLE_KEYS = [k for k, *_ in VARIANTS]
# Historical ablations (NOT retrained; cited from Phase-4's results.json for context, §17).
HIST_PATH = Path("runs/phase4_tail_weighted/results.json")
HIST_KEYS = ["C_none", "C_log1p_weighted", "C_log1p_tail_weighted"]


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
def _fit_eval_variant(transform, loss_type, arch, cfg, train, val, test, edges, keep_fitted):
    """Train Baseline C with a given architecture once per seed.

    ONLY `arch` varies across the two variants (transform=log1p, loss=standard fixed) ->
    any delta is attributable to reconstruction capacity. Measures per-seed peak VRAM and
    fit wall-time (§8/§9). Returns (per_seed, fitted_seed0, runtime_s, vram_mb).
    """
    from depthwizard.models.fusion_head import LearnedFusionHead
    on_cuda = _cuda()
    if on_cuda:
        import torch
    per_seed = []
    fitted0 = None
    runtime_s = 0.0
    vram_mb = None
    tcfg = replace(cfg.train, target_transform=transform, loss_type=loss_type, arch=arch)
    for si, seed in enumerate(cfg.seeds):
        c = LearnedFusionHead(tcfg, nodata=cfg.data.nodata, seed=seed)
        n_params = c.n_params()
        if on_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        t_fit = time.time()
        c.fit(train)
        runtime_s += time.time() - t_fit
        if on_cuda:
            peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            vram_mb = peak if vram_mb is None else max(vram_mb, peak)
        idn = evaluate_estimator(c, val, cfg, "indomain", bin_edges=edges)
        xc = evaluate_estimator(c, test, cfg, "xcity", bin_edges=edges)
        per_seed.append({"seed": seed, "n_params": n_params,
                         "indomain": idn, "xcity": xc})
        xa = xc["aggregate"]["all"]
        xb = xc["aggregate"].get("building", {})
        print(f"[{arch}/{transform}/{loss_type}][seed={seed}] xcity all MAE={_get(xa,'mae_pooled')} "
              f"| building MAE={_get(xb,'mae_pooled')} | building RMSE={_get(xb,'rmse_pooled')}")
        if keep_fitted and si == 0:
            fitted0 = c
    return per_seed, fitted0, round(runtime_s, 1), (round(vram_mb, 1) if vram_mb else None)


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
            rmse_m, rmse_s = _mean_std([l[i]["rmse"] for l in lists])
            bias_m, bias_s = _mean_std([l[i]["bias"] for l in lists])
            out.append({"lo": lists[0][i]["lo"], "hi": lists[0][i]["hi"],
                        "n_pixels": lists[0][i]["n_pixels"],
                        "mae_mean": mae_m, "mae_std": mae_s,
                        "rmse_mean": rmse_m, "rmse_std": rmse_s,
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


def _pool_bldg(S, want_tall, key="xcity_binned_building"):
    """Pixel-weighted building MAE, RMSE & bias pooled over tall (lo>=CEILING) OR low
    (lo<CEILING) bins. RMSE is pooled in quadrature (sqrt of pixel-weighted mean of
    rmse^2) so it stays a true magnitude -- the metric §24 says to read BEFORE bias."""
    mnum = bnum = r2num = den = 0.0
    for bb in S.get(key, []):
        is_tall = bb["lo"] >= CEILING
        if is_tall != want_tall:
            continue
        n = bb["n_pixels"]
        if n and bb["mae_mean"] is not None:
            mnum += bb["mae_mean"] * n
            bnum += (bb["bias_mean"] or 0.0) * n
            if bb.get("rmse_mean") is not None:
                r2num += (bb["rmse_mean"] ** 2) * n
            den += n
    if not den:
        return None, None, None
    return mnum / den, (r2num / den) ** 0.5, bnum / den


# ---------------------------------------------------------------- ceiling probe
def _max_pred_over(head, samples):
    """Max finite predicted height across a sample set (the empirical prediction ceiling,
    §6/§22). Inference only -- no training."""
    mx = None
    for s in samples:
        p = head.predict(s)
        if np.isfinite(p).any():
            m = float(np.nanmax(p))
            mx = m if mx is None else max(mx, m)
    return mx


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
    """§21 challenge panels + §22 scatter (C_log1p vs C_log1p_recon) + §23 the decisive
    2-variant per-height-bin building MAE + signed-bias comparison (with the protected
    0–15 m band shaded)."""
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
    focus = {k: fitted.get(k) for k in TABLE_KEYS}
    # §21/§22: same challenging tall scene, control vs recon -> RGB|ref|pred|error + scatter.
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
    # §23: cross-city BUILDING MAE and signed bias per height bin, control vs recon.
    try:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        colors = {"C_log1p": "#1f6feb", "C_log1p_recon": "#1f883d"}
        for which, a, ylab in [("mae_mean", ax[0], "cross-city building MAE (m)"),
                               ("bias_mean", ax[1], "cross-city building signed bias (m)")]:
            for vname in TABLE_KEYS:
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
        ax[0].set_title("Building MAE by bin (lower=better) — read THIS before bias (§24)")
        ax[1].set_title(f"Signed bias by bin (neg=under-predict; ceiling≈{CEILING:.0f} m){tag}")
        fig.suptitle("Phase-5: reconstruction (unet4) vs control (unet3) — cross-city building "
                     "error by height (does the deeper net lift tall MAE without upward shift?)")
        fig.tight_layout()
        fig.savefig(fig_dir / "binned_building_recon_vs_ctrl.png", dpi=100)
        plt.close(fig)
        print(f"[viz] wrote challenge panels + scatters + binned comparison to {fig_dir}")
    except Exception as e:
        print(f"[viz] binned figure skipped: {e}")


# ---------------------------------------------------------------- historical context
def _load_historical():
    """Load the Phase-4 run's summaries for the retained ablations (§17: cite, don't
    retrain). Returns {key: summary} or {} if the file is absent/unreadable."""
    if not HIST_PATH.exists():
        return {}
    try:
        data = json.loads(HIST_PATH.read_text(encoding="utf-8"))
        return {k: data["variants"][k]["summary"]
                for k in HIST_KEYS if k in data.get("variants", {})}
    except Exception as e:
        print(f"[hist] could not load {HIST_PATH}: {e}")
        return {}


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
    RT = {k: results["variants"][k]["runtime_s"] for k in TABLE_KEYS}
    VR = {k: results["variants"][k]["vram_mb"] for k in TABLE_KEYS}
    hist = results.get("historical", {})
    cmp = results["comparison"]
    L = []
    L.append("# DepthWizard — PHASE 5 results (reconstruction-fidelity / receptive-field)\n")
    L.append("_Generated by `scripts/run_phase5_recon.py`. Every number is measured by that "
             "run; placeholders read `n/a`. Controlled single-variable test: C_log1p (arch "
             "`unet3`) vs the SAME model with ONE extra encoder/decoder level (arch `unet4`, "
             "`C_log1p_recon`) — bottleneck 32→16 at 256 px, ~2× effective receptive field. "
             "Target transform (log1p), loss (plain masked-L1), optimizer, LR, batch, width, "
             "resolution, data, split and DA-V2 cache are ALL identical. The Phase-3 aggressive "
             "and Phase-4 tail ablations are cited from Phase-4 (NOT retrained, §17)._\n")
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
             f"| seeds: `{meta['seeds']}` | total runtime: {meta['elapsed_s']}s")
    L.append(f"- **architecture change** (the ONE variable): `unet3` SmallFusionUNet "
             f"(3 pool levels, bottleneck 32² @256, RF≈66px) → `unet4` SmallReconUNet "
             f"(4 pool levels, bottleneck 16² @256, RF≈138px). New deepest level held at "
             f"w·4 channels (NOT widened) so added params buy receptive field, not width.")
    L.append(f"- params: control `unet3`={V['C_log1p'].get('n_params'):,} | recon "
             f"`unet4`={V['C_log1p_recon'].get('n_params'):,} "
             f"(Δ +{cmp['d_params']:,}, +{cmp['d_params_pct']:.0f}%)")
    L.append(f"- peak VRAM: control={_fmt(VR['C_log1p'],1)} MB | recon="
             f"{_fmt(VR['C_log1p_recon'],1)} MB | runtime: control={_fmt(RT['C_log1p'],1)}s "
             f"recon={_fmt(RT['C_log1p_recon'],1)}s (§8/§9)\n")
    if meta["source"] == "hf_blocks":
        L.append("> **Provenance caveat (§29):** unofficial HF mirror `JasonXF/DFC2019-10k`, "
                 "preprocessed nDSM (ground floored to 0). Valid feasibility evidence only; "
                 "does NOT establish official IEEE GRSS DFC2019, Indian/ISRO, or production "
                 "metric-height accuracy. Re-confirm on official data before external reporting.\n")

    # §25 required comparison table
    L.append("## §25 comparison table — mean ± std over seeds\n")
    L.append("| Model | Params | In-domain MAE | In-domain RMSE | Cross-city MAE "
             "| Cross-city RMSE | Building MAE (xcity) | Building RMSE (xcity) |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    L.append(f"| B affine | — "
             f"| {_fmt(_get(B['indomain']['aggregate']['all'],'mae_pooled'))} "
             f"| {_fmt(_get(B['indomain']['aggregate']['all'],'rmse_pooled'))} "
             f"| {_fmt(_get(B['xcity']['aggregate']['all'],'mae_pooled'))} "
             f"| {_fmt(_get(B['xcity']['aggregate']['all'],'rmse_pooled'))} "
             f"| {_fmt(_get(B['xcity']['aggregate'].get('building',{}),'mae_pooled'))} "
             f"| {_fmt(_get(B['xcity']['aggregate'].get('building',{}),'rmse_pooled'))} |")
    labels = {"C_log1p": "C log1p (control, unet3)",
              "C_log1p_recon": "**C log1p recon (unet4, NEW)**"}
    for key in TABLE_KEYS:
        S = V[key]
        L.append(f"| {labels[key]} | {S.get('n_params'):,} "
                 f"| {_ms(S['indomain_all_mae_pooled_mean'], S['indomain_all_mae_pooled_std'])} "
                 f"| {_ms(S['indomain_all_rmse_pooled_mean'], S['indomain_all_rmse_pooled_std'])} "
                 f"| {_ms(S['xcity_all_mae_pooled_mean'], S['xcity_all_mae_pooled_std'])} "
                 f"| {_ms(S['xcity_all_rmse_pooled_mean'], S['xcity_all_rmse_pooled_std'])} "
                 f"| {_ms(S['xcity_building_mae_pooled_mean'], S['xcity_building_mae_pooled_std'])} "
                 f"| {_ms(S['xcity_building_rmse_pooled_mean'], S['xcity_building_rmse_pooled_std'])} |")
    L.append(f"| oracle affine (UB) | — "
             f"| {_fmt(_get(ora['indomain']['aggregate']['all'],'mae_pooled'))} "
             f"| {_fmt(_get(ora['indomain']['aggregate']['all'],'rmse_pooled'))} "
             f"| {_fmt(_get(ora['xcity']['aggregate']['all'],'mae_pooled'))} "
             f"| {_fmt(_get(ora['xcity']['aggregate']['all'],'rmse_pooled'))} "
             f"| {_fmt(_get(ora['xcity']['aggregate'].get('building',{}),'mae_pooled'))} "
             f"| {_fmt(_get(ora['xcity']['aggregate'].get('building',{}),'rmse_pooled'))} |")
    L.append(f"\n_A · raw-depth cross-city Pearson r (signal check): "
             f"{_fmt(_get(A['xcity']['aggregate']['all'],'pearson_mean'))}. Building is the "
             f"PRIMARY analysis (§19)._\n")

    # historical ablations, cited (not retrained)
    if hist:
        L.append("### Historical ablations (Phase-3/4, cited from `runs/phase4_tail_weighted/`, NOT retrained, §17)\n")
        L.append("| Model | Cross-city MAE | Cross-city RMSE | Building MAE | Building RMSE |")
        L.append("|---|--:|--:|--:|--:|")
        hlabels = {"C_none": "C none (Phase-1)", "C_log1p_weighted": "C log1p weighted (Phase-3 aggressive)",
                   "C_log1p_tail_weighted": "C log1p tail-weighted (Phase-4)"}
        for k in HIST_KEYS:
            if k not in hist:
                continue
            S = hist[k]
            L.append(f"| {hlabels.get(k,k)} "
                     f"| {_ms(S.get('xcity_all_mae_pooled_mean'), S.get('xcity_all_mae_pooled_std'))} "
                     f"| {_ms(S.get('xcity_all_rmse_pooled_mean'), S.get('xcity_all_rmse_pooled_std'))} "
                     f"| {_ms(S.get('xcity_building_mae_pooled_mean'), S.get('xcity_building_mae_pooled_std'))} "
                     f"| {_ms(S.get('xcity_building_rmse_pooled_mean'), S.get('xcity_building_rmse_pooled_std'))} |")
        L.append("\n_Note: cross-run GPU nondeterminism means these historical C numbers are "
                 "not bit-comparable to this run's control; they are context, and the decisive "
                 "comparison is the within-run C_log1p vs C_log1p_recon below._\n")

    # per-seed transparency (never hide a disagreeing seed)
    L.append("## Per-seed cross-city MAE (all / building) — transparency (§14,§19)\n")
    L.append("| variant | " + " | ".join(f"seed {s} all / bldg" for s in meta["seeds"]) + " |")
    L.append("|---|" + "|".join(["---"] * len(meta["seeds"])) + "|")
    for key in TABLE_KEYS:
        ps = results["variants"][key]["per_seed"]
        cells = []
        for p in ps:
            a = _get(p["xcity"]["aggregate"]["all"], "mae_pooled")
            bb = _get(p["xcity"]["aggregate"].get("building", {}), "mae_pooled")
            cells.append(f"{_fmt(a)} / {_fmt(bb)}")
        L.append(f"| {key} | " + " | ".join(cells) + " |")
    L.append("")

    # §23/§24 the decisive isolation: building MAE, RMSE, bias by height bin
    L.append("## Cross-city BUILDING error by GT-height bin — the reconstruction isolation (§19,§23,§24)\n")
    L.append("_Read MAE and RMSE FIRST, bias last (§24): a bias improvement with rising MAE/RMSE "
             "is the Phase-4 compensating-artifact pattern, NOT reconstruction. bias<0 = "
             "under-prediction (the ceiling/body-collapse signature)._\n")
    bl = V["C_log1p"]["xcity_binned_building"]
    br = V["C_log1p_recon"]["xcity_binned_building"]
    L.append("| GT bin (m) | px | ctrl MAE | **recon MAE** | ΔMAE (ctrl−recon) | ctrl RMSE "
             "| **recon RMSE** | ΔRMSE | ctrl bias | **recon bias** |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for i in range(min(len(bl), len(br))):
        dm = (bl[i]["mae_mean"] - br[i]["mae_mean"]) if (bl[i]["mae_mean"] is not None
              and br[i]["mae_mean"] is not None) else None
        dr = (bl[i].get("rmse_mean") - br[i].get("rmse_mean")) if (bl[i].get("rmse_mean") is not None
              and br[i].get("rmse_mean") is not None) else None
        arrow = ""
        if dm is not None:
            arrow = " ✅" if dm > 0.05 else (" ❌" if dm < -0.05 else "")
        L.append(f"| {_bin_label(bl[i])} | {bl[i]['n_pixels']:,} "
                 f"| {_fmt(bl[i]['mae_mean'])} | {_fmt(br[i]['mae_mean'])} | {_fmt(dm)}{arrow} "
                 f"| {_fmt(bl[i].get('rmse_mean'))} | {_fmt(br[i].get('rmse_mean'))} | {_fmt(dr)} "
                 f"| {_fmt(bl[i]['bias_mean'])} | {_fmt(br[i]['bias_mean'])} |")
    L.append("\n_ΔMAE = ctrl − recon (positive = recon better). ✅/❌ mark |ΔMAE|>0.05 m. "
             "The 0–15 m rows should stay ≈0 (§20 do-not-sacrifice); the >15 m rows are the "
             "target. Watch for a row where bias shrinks but RMSE grows — that is edge overshoot "
             "cancelling body collapse (§24)._\n")

    # in-domain sanity + building
    L.append("## In-domain (held-out TRAIN-city tiles) — mean ± std\n")
    L.append("| variant | all MAE | all RMSE | building MAE | building RMSE |")
    L.append("|---|---|---|---|---|")
    for key in TABLE_KEYS:
        S = V[key]
        L.append(f"| {key} "
                 f"| {_ms(S['indomain_all_mae_pooled_mean'], S['indomain_all_mae_pooled_std'])} "
                 f"| {_ms(S['indomain_all_rmse_pooled_mean'], S['indomain_all_rmse_pooled_std'])} "
                 f"| {_ms(S['indomain_building_mae_pooled_mean'], S['indomain_building_mae_pooled_std'])} "
                 f"| {_ms(S['indomain_building_rmse_pooled_mean'], S['indomain_building_rmse_pooled_std'])} |")
    L.append("")

    # per-variant auditable verdict
    L.append("## Per-variant auditable verdict (decide(), context only — keys on overall "
             "cross-city MAE, which is dominated by ground pixels)\n")
    for key, _, _, _arch, label in VARIANTS:
        v = results["variants"][key]["verdict"]
        L.append(f"- **{label}: {v['verdict']}** — {v['summary']}")
    L.append("")

    # comparison summary: recon vs control (primary), MAE/RMSE FIRST then bias/ceiling
    L.append("## Comparison summary — reconstruction (unet4) vs C_log1p control (the isolated change)\n")
    L.append(f"- cross-city all MAE Δ (ctrl−recon): **{_fmt(cmp['d_all_mae'])} m** (pos = recon better)")
    L.append(f"- cross-city building MAE Δ (ctrl−recon): **{_fmt(cmp['d_bldg_mae'])} m**")
    L.append(f"- cross-city building RMSE Δ (ctrl−recon): **{_fmt(cmp['d_bldg_rmse'])} m** "
             f"(ctrl={_fmt(cmp['bldg_rmse_ctrl'])} recon={_fmt(cmp['bldg_rmse_recon'])}; pos = recon better)")
    L.append(f"- **low 0–{int(CEILING)} m** building: MAE ctrl={_fmt(cmp['low_mae_ctrl'])} "
             f"recon={_fmt(cmp['low_mae_recon'])} (Δ {_fmt(cmp['d_low_bldg_mae'])} m; ≈0 = §20 preserved) "
             f"| RMSE ctrl={_fmt(cmp['low_rmse_ctrl'])} recon={_fmt(cmp['low_rmse_recon'])}")
    L.append(f"- **tall >{int(CEILING)} m** building: MAE ctrl={_fmt(cmp['tall_mae_ctrl'])} "
             f"recon={_fmt(cmp['tall_mae_recon'])} (Δ {_fmt(cmp['d_tall_bldg_mae'])} m; pos = recon better) "
             f"| RMSE ctrl={_fmt(cmp['tall_rmse_ctrl'])} recon={_fmt(cmp['tall_rmse_recon'])} "
             f"(Δ {_fmt(cmp['d_tall_bldg_rmse'])} m)")
    L.append(f"- tall >{int(CEILING)} m building signed bias: ctrl={_fmt(cmp['tall_bias_ctrl'])} "
             f"recon={_fmt(cmp['tall_bias_recon'])} (Δ {_fmt(cmp['d_tall_bldg_bias'])} m; pos = less under-prediction)")
    L.append(f"- empirical prediction ceiling (max pred over test, seed0): ctrl="
             f"{_fmt(cmp['max_pred_ctrl'])} m recon={_fmt(cmp['max_pred_recon'])} m "
             f"(§6/§22; higher recon = ceiling raised)")
    L.append(f"- beats affine cross-city (≥10% all-MAE cut over B={_fmt(cmp['b_all_mae'])})? "
             f"ctrl: **{cmp['ctrl_beats_B']}**, recon: **{cmp['recon_beats_B']}** | beats affine "
             f"on BUILDING MAE (B={_fmt(cmp['b_bldg_mae'])})? ctrl: **{cmp['ctrl_beats_B_bldg']}**, "
             f"recon: **{cmp['recon_beats_B_bldg']}**\n")

    L.append("### §26 success signals — reconstruction without regression\n")
    for k, desc in [("s_bldg_mae_ok", "building MAE improved or stable (≥ −0.3 m vs control)"),
                    ("s_bldg_rmse_better", "building RMSE improved (the anti-Phase-4 signal)"),
                    ("s_tall_mae_better", "tall >15 m building MAE improved"),
                    ("s_tall_bias_better", "tall >15 m under-prediction reduced (bias less negative)"),
                    ("s_ceiling_up", "prediction ceiling raised (recon max pred > control)"),
                    ("s_low_stable", "0–15 m building MAE stayed ≈ control (within 0.5 m)"),
                    ("s_edge_ok", "no edge-overshoot explosion (tall RMSE not worse by >2 m)"),
                    ("s_xcity_useful", "cross-city all-MAE still beats affine ≥10%"),
                    ("s_variance_ok", "seed variance not materially worse than control")]:
        L.append(f"- {'✅' if cmp.get(k) else '❌'} {desc}")
    L.append("")

    # §24/§27F explicit compensating-artifact guard
    L.append("### §24/§27F — is any 'improvement' a compensating artifact? (bias↓ but MAE/RMSE↑)\n")
    L.append(f"- {'⚠️ YES' if cmp['compensating_artifact'] else '✅ no'}: "
             f"tall bias {'improved' if cmp['tall_bias_improved'] else 'did not improve'} "
             f"while tall {'MAE/RMSE worsened' if cmp['tall_magnitude_worse'] else 'MAE/RMSE held/improved'}. "
             f"{'A lower bias here is NOT reconstruction — treat as failure mode F.' if cmp['compensating_artifact'] else 'Bias and magnitude move together (or magnitude improved) — not the Phase-4 pattern.'}")
    L.append(f"- building RMSE explosion guard: recon xcity building RMSE="
             f"{_fmt(cmp['bldg_rmse_recon'])} m vs control {_fmt(cmp['bldg_rmse_ctrl'])} m "
             f"({'⚠️ worse' if (cmp['d_bldg_rmse'] is not None and cmp['d_bldg_rmse'] < 0) else 'ok'}).\n")

    L.append("> This run does NOT establish ISRO/Indian-sensor readiness and does NOT "
             "build the product. It is one controlled architectural modification, "
             "city-held-out on the unofficial DFC2019 mirror only. STOP for human review.\n")
    Path(path).write_text("\n".join(L), encoding="utf-8")
    print(f"[report] wrote {path}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase5_recon.yaml")
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
              "Phase-5) cannot run. Re-run with --allow-fake-depth for a NON-EVIDENCE "
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
        print("[error] torch required for Phase-5 (Baseline C). Aborting."); sys.exit(4)

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

    # ---- the controlled comparison: C_log1p (unet3) vs C_log1p_recon (unet4) ----
    variants = {}
    fitted = {}
    for key, transform, loss_type, arch, _label in VARIANTS:
        ps, fit0, rt, vram = _fit_eval_variant(transform, loss_type, arch, cfg, train, val,
                                               test, edges, keep_fitted=True)
        S = _summarize(ps)
        variants[key] = {"per_seed": ps, "summary": S, "runtime_s": rt, "vram_mb": vram,
                         "verdict": _variant_verdict(A, B, S)}
        fitted[key] = fit0

    # ---- empirical prediction ceiling (max pred over test, seed0) ----
    max_pred = {}
    for key in TABLE_KEYS:
        max_pred[key] = _max_pred_over(fitted[key], test) if fitted.get(key) else None

    Sc = variants["C_log1p"]["summary"]        # control (unet3)
    Sr = variants["C_log1p_recon"]["summary"]  # recon (unet4)

    # ---- comparison scalars: recon vs control, MAE/RMSE FIRST, then bias/ceiling ----
    b_all_mae = _get(B["xcity"]["aggregate"]["all"], "mae_pooled")
    b_bldg_mae = _get(B["xcity"]["aggregate"].get("building", {}), "mae_pooled")

    def _beats_B_all(S):
        c = S["xcity_all_mae_pooled_mean"]
        return bool(b_all_mae and c is not None and (b_all_mae - c) / b_all_mae >= 0.10)

    def _beats_B_bldg(S):
        c = S["xcity_building_mae_pooled_mean"]
        return bool(b_bldg_mae and c is not None and c < b_bldg_mae)

    def _d(S1, S2, key):  # S1 - S2
        v1, v2 = S1[key], S2[key]
        return (v1 - v2) if (v1 is not None and v2 is not None) else None

    def _sub(x, y):
        return (x - y) if (x is not None and y is not None) else None

    tall_mae_c, tall_rmse_c, tall_bias_c = _pool_bldg(Sc, want_tall=True)
    tall_mae_r, tall_rmse_r, tall_bias_r = _pool_bldg(Sr, want_tall=True)
    low_mae_c, low_rmse_c, low_bias_c = _pool_bldg(Sc, want_tall=False)
    low_mae_r, low_rmse_r, low_bias_r = _pool_bldg(Sr, want_tall=False)

    d_tall_mae = _sub(tall_mae_c, tall_mae_r)    # pos = recon better
    d_tall_rmse = _sub(tall_rmse_c, tall_rmse_r)  # pos = recon better
    d_tall_bias = _sub(tall_bias_r, tall_bias_c)  # pos = recon less negative (less under-pred)
    d_low_mae = _sub(low_mae_r, low_mae_c)        # ~0 = protected (recon - ctrl)

    bldg_rmse_c = Sc["xcity_building_rmse_pooled_mean"]
    bldg_rmse_r = Sr["xcity_building_rmse_pooled_mean"]
    d_bldg_rmse = _sub(bldg_rmse_c, bldg_rmse_r)  # pos = recon better
    var_c = Sc["xcity_all_mae_pooled_std"]
    var_r = Sr["xcity_all_mae_pooled_std"]

    n_c = Sc.get("n_params") or 0
    n_r = Sr.get("n_params") or 0

    # §24/§27F compensating-artifact detector: bias improved but magnitude (MAE or RMSE) worse.
    tall_bias_improved = bool(d_tall_bias is not None and d_tall_bias > 0.25)
    tall_magnitude_worse = bool((d_tall_mae is not None and d_tall_mae < -0.25)
                                or (d_tall_rmse is not None and d_tall_rmse < -0.5))
    compensating_artifact = bool(tall_bias_improved and tall_magnitude_worse)

    comparison = {
        "b_all_mae": b_all_mae, "b_bldg_mae": b_bldg_mae,
        "d_params": int(n_r - n_c),
        "d_params_pct": (100.0 * (n_r - n_c) / n_c) if n_c else None,
        "d_all_mae": _d(Sc, Sr, "xcity_all_mae_pooled_mean"),
        "d_bldg_mae": _d(Sc, Sr, "xcity_building_mae_pooled_mean"),
        "d_bldg_rmse": d_bldg_rmse,
        "bldg_rmse_ctrl": bldg_rmse_c, "bldg_rmse_recon": bldg_rmse_r,
        "tall_mae_ctrl": tall_mae_c, "tall_mae_recon": tall_mae_r, "d_tall_bldg_mae": d_tall_mae,
        "tall_rmse_ctrl": tall_rmse_c, "tall_rmse_recon": tall_rmse_r, "d_tall_bldg_rmse": d_tall_rmse,
        "tall_bias_ctrl": tall_bias_c, "tall_bias_recon": tall_bias_r, "d_tall_bldg_bias": d_tall_bias,
        "low_mae_ctrl": low_mae_c, "low_mae_recon": low_mae_r, "d_low_bldg_mae": d_low_mae,
        "low_rmse_ctrl": low_rmse_c, "low_rmse_recon": low_rmse_r,
        "low_bias_ctrl": low_bias_c, "low_bias_recon": low_bias_r,
        "max_pred_ctrl": max_pred.get("C_log1p"), "max_pred_recon": max_pred.get("C_log1p_recon"),
        "ctrl_beats_B": _beats_B_all(Sc), "recon_beats_B": _beats_B_all(Sr),
        "ctrl_beats_B_bldg": _beats_B_bldg(Sc), "recon_beats_B_bldg": _beats_B_bldg(Sr),
        # §26 success signals (each necessary; none alone sufficient)
        "s_bldg_mae_ok": bool(_d(Sc, Sr, "xcity_building_mae_pooled_mean") is not None
                              and _d(Sc, Sr, "xcity_building_mae_pooled_mean") >= -0.3),
        "s_bldg_rmse_better": bool(d_bldg_rmse is not None and d_bldg_rmse > 0),
        "s_tall_mae_better": bool(d_tall_mae is not None and d_tall_mae > 0),
        "s_tall_bias_better": bool(d_tall_bias is not None and d_tall_bias > 0),
        "s_ceiling_up": bool(max_pred.get("C_log1p") is not None
                             and max_pred.get("C_log1p_recon") is not None
                             and max_pred["C_log1p_recon"] > max_pred["C_log1p"]),
        "s_low_stable": bool(d_low_mae is not None and abs(d_low_mae) <= 0.5),
        "s_edge_ok": bool(d_tall_rmse is not None and d_tall_rmse >= -2.0),
        "s_xcity_useful": _beats_B_all(Sr),
        "s_variance_ok": bool(var_c is not None and var_r is not None and var_r <= var_c + 0.3),
        # §24/§27F compensating-artifact guard
        "tall_bias_improved": tall_bias_improved,
        "tall_magnitude_worse": tall_magnitude_worse,
        "compensating_artifact": compensating_artifact,
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
        "historical": _load_historical(),
        "comparison": comparison,
    }

    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    _write_report(results, str(out_dir / "PHASE5_COMPARISON.md"))
    _make_figures(fitted, test, cfg, fig_dir,
                  {k: variants[k]["summary"] for k in variants}, evidence_valid)

    print("\n" + "=" * 72)
    print("PHASE-5 CHECKPOINT — reconstruction (unet4) vs C_log1p control (unet3)")
    print(f"  params: ctrl={n_c:,}  recon={n_r:,}  (+{comparison['d_params']:,}, "
          f"+{_fmt(comparison['d_params_pct'],0)}%)  | VRAM ctrl="
          f"{_fmt(variants['C_log1p']['vram_mb'],0)}MB recon={_fmt(variants['C_log1p_recon']['vram_mb'],0)}MB")
    print(f"  cross-city all MAE:  ctrl={_fmt(Sc['xcity_all_mae_pooled_mean'])}±"
          f"{_fmt(Sc['xcity_all_mae_pooled_std'])}  "
          f"recon={_fmt(Sr['xcity_all_mae_pooled_mean'])}±{_fmt(Sr['xcity_all_mae_pooled_std'])}  "
          f"(B={_fmt(b_all_mae)})")
    print(f"  cross-city bldg MAE: ctrl={_fmt(Sc['xcity_building_mae_pooled_mean'])}  "
          f"recon={_fmt(Sr['xcity_building_mae_pooled_mean'])}  (B={_fmt(b_bldg_mae)})")
    print(f"  cross-city bldg RMSE: ctrl={_fmt(bldg_rmse_c)}  recon={_fmt(bldg_rmse_r)}  "
          f"(Δ={_fmt(d_bldg_rmse)}, pos=better)")
    print(f"  LOW 0–{CEILING:.0f}m bldg MAE:  ctrl={_fmt(low_mae_c)}  recon={_fmt(low_mae_r)}  "
          f"(Δ={_fmt(d_low_mae)}; ≈0=preserved)")
    print(f"  TALL >{CEILING:.0f}m bldg MAE: ctrl={_fmt(tall_mae_c)}  recon={_fmt(tall_mae_r)}  "
          f"(Δ={_fmt(d_tall_mae)}, pos=better) | RMSE ctrl={_fmt(tall_rmse_c)} recon={_fmt(tall_rmse_r)}")
    print(f"  TALL bias: ctrl={_fmt(tall_bias_c)}  recon={_fmt(tall_bias_r)}  "
          f"| ceiling(maxpred): ctrl={_fmt(max_pred.get('C_log1p'))} recon={_fmt(max_pred.get('C_log1p_recon'))}")
    print(f"  §24 compensating-artifact (bias↓ but MAE/RMSE↑)? "
          f"{'⚠️ YES — failure mode F' if compensating_artifact else 'no'}")
    print(f"  §26 signals: bldg_mae_ok={comparison['s_bldg_mae_ok']} "
          f"bldg_rmse↓={comparison['s_bldg_rmse_better']} tall_mae↓={comparison['s_tall_mae_better']} "
          f"tall_bias↑={comparison['s_tall_bias_better']} ceiling↑={comparison['s_ceiling_up']} "
          f"low_stable={comparison['s_low_stable']} edge_ok={comparison['s_edge_ok']} "
          f"xcity_useful={comparison['s_xcity_useful']} var_ok={comparison['s_variance_ok']}")
    if not evidence_valid:
        print("\n[!] evidence_valid=False — plumbing only; NOT a real result.")
    print("=" * 72)
    print("STOP: human reviews these numbers + error maps before any GO decision.")


if __name__ == "__main__":
    main()
