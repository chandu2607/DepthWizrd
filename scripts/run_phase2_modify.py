#!/usr/bin/env python
"""DepthWizard PHASE-2 MODIFY orchestrator -- controlled single-variable test.

Question: does the ONE evidence-justified modification (log1p target transform)
lift the learned head's tall-structure accuracy WITHOUT wrecking the dominant
low-height bins or the cross-city generalization -- enough to beat trivial
global-affine calibration on a fully held-out city?

Protocol (matches the Phase-2 MODIFY spec):
  * Reuse the EXACT Phase-1 data + city-held-out split + DA-V2 depth cache.
  * Anchors: Baseline A (raw depth, signal check), B (global affine, bar to beat),
    per-image oracle affine (upper bound).
  * The comparison: Baseline C trained TWICE per seed --
        C_none   = target_transform "none"  (the ORIGINAL Phase-1 C, reproduced)
        C_log1p  = target_transform "log1p" (the modification under test)
    ONLY that variable differs. Everything else (arch, loss, lr, epochs, res,
    seeds, data) is identical.
  * Strict evaluation: all / building / non-building, in-domain (held-out train
    city tiles) AND cross-city (unseen city), PLUS per-GT-height-bin MAE/bias so
    we can see the tall-structure regime directly. BOTH seeds -> mean +/- std;
    never headline a single favorable seed.
  * Challenging tall-building scenes rendered for original vs modified C.

This does NOT build the product. It STOPS at the Phase-2 checkpoint again.

Usage:
  python scripts/run_phase2_modify.py --config configs/phase2_modify.yaml
  python scripts/run_phase2_modify.py --smoke --allow-fake-depth   # plumbing only
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
def _fit_eval_variant(transform, cfg, train, val, test, edges, keep_fitted):
    """Train Baseline C with a given target_transform once per seed; evaluate.

    Returns (per_seed_list, fitted_seed0_or_None). per_seed entries carry the
    full in-domain + cross-city evaluate_estimator dicts (incl. binned metrics).
    """
    from depthwizard.models.fusion_head import LearnedFusionHead
    per_seed = []
    fitted0 = None
    tcfg = replace(cfg.train, target_transform=transform)
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
        print(f"[C:{transform}][seed={seed}] xcity all MAE={_get(xa,'mae_pooled')} "
              f"RMSE={_get(xa,'rmse_pooled')} | building MAE={_get(xb,'mae_pooled')}")
        if keep_fitted and si == 0:
            fitted0 = c
    return per_seed, fitted0


def _summarize(per_seed):
    """mean +/- std across seeds for the headline scalars + per-bin MAE/bias."""
    def scal(group, kind, key):
        return _mean_std([ps[group]["aggregate"].get(kind, {}).get(key)
                          for ps in per_seed])

    def binned(group, which):
        # per_seed[i][group]['aggregate'][which] is a list of per-bin dicts
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
    # cross-city building Pearson (signal on the hard class), mean across seeds
    s["xcity_building_pearson_mean"], _ = _mean_std(
        [ps["xcity"]["aggregate"].get("building", {}).get("pearson_mean")
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


# ---------------------------------------------------------------- challenging viz
def _challenge_ids(test, nodata, blabel, ceiling=CEILING, k=3):
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
    ids = _challenge_ids(test, cfg.data.nodata, cfg.data.building_label, k=3)
    by_id = {s["id"]: s for s in test}
    # tall-building challenging scenes: original C vs modified C, same tile.
    for sid in ids:
        s = by_id.get(sid)
        if s is None:
            continue
        for vname, head in fitted.items():
            if head is None:
                continue
            pred = head.predict(s)
            plots.save_qualitative(
                s, pred, str(fig_dir / f"challenge_{sid}_{vname}.png"),
                nodata=cfg.data.nodata,
                title=f"{vname} (seed0) · {sid} · cross-city{tag}")
    # decision figure: cross-city BUILDING MAE and bias per height bin, none vs log1p
    try:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        colors = {"C_none": "#c1440e", "C_log1p": "#1f6feb"}
        for which, a, ylab in [("mae_mean", ax[0], "cross-city building MAE (m)"),
                               ("bias_mean", ax[1], "cross-city building bias (m)")]:
            for vname, summ in summaries.items():
                bins = summ.get("xcity_binned_building", [])
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
        fig.suptitle("Phase-2: log1p vs original C — cross-city building error by height")
        fig.tight_layout(); fig.savefig(fig_dir / "binned_building_none_vs_log1p.png", dpi=100)
        plt.close(fig)
        print(f"[viz] wrote challenge panels + binned comparison to {fig_dir}")
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
    Sn = results["variants"]["C_none"]["summary"]
    Sl = results["variants"]["C_log1p"]["summary"]
    L = []
    L.append("# DepthWizard — PHASE 2 MODIFY results (log1p target transform)\n")
    L.append("_Generated by `scripts/run_phase2_modify.py`. Every number is measured "
             "by that run; placeholders read `n/a`. This is a controlled single-"
             "variable test: original Baseline C vs the SAME C with a log1p target "
             "transform, all else identical._\n")
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
             f"| seeds: `{meta['seeds']}` | head params: {Sn.get('n_params')}")
    L.append(f"- diagnosed ceiling under test: **~{CEILING:.0f} m** | modification: "
             f"**log1p** target transform (only variable changed)\n")
    if meta["source"] == "hf_blocks":
        L.append("> **Provenance caveat:** unofficial HF mirror `JasonXF/DFC2019-10k`, "
                 "preprocessed nDSM (ground floored to 0). Valid feasibility evidence; "
                 "re-confirm on official IEEE GRSS DFC2019 before external reporting.\n")

    # headline comparison table
    L.append("## Headline — cross-city (held-out TEST city), mean ± std over seeds\n")
    L.append("| method | all MAE | all RMSE | building MAE | building RMSE |")
    L.append("|---|---|---|---|---|")
    L.append(f"| B · global affine | {_fmt(_get(B['xcity']['aggregate']['all'],'mae_pooled'))} "
             f"| {_fmt(_get(B['xcity']['aggregate']['all'],'rmse_pooled'))} "
             f"| {_fmt(_get(B['xcity']['aggregate'].get('building',{}),'mae_pooled'))} "
             f"| {_fmt(_get(B['xcity']['aggregate'].get('building',{}),'rmse_pooled'))} |")
    for label, S in [("C · original (none)", Sn), ("C · MODIFIED (log1p)", Sl)]:
        L.append(f"| {label} "
                 f"| {_ms(S['xcity_all_mae_pooled_mean'], S['xcity_all_mae_pooled_std'])} "
                 f"| {_ms(S['xcity_all_rmse_pooled_mean'], S['xcity_all_rmse_pooled_std'])} "
                 f"| {_ms(S['xcity_building_mae_pooled_mean'], S['xcity_building_mae_pooled_std'])} "
                 f"| {_ms(S['xcity_building_rmse_pooled_mean'], S['xcity_building_rmse_pooled_std'])} |")
    L.append(f"| oracle affine (upper bound) "
             f"| {_fmt(_get(ora['xcity']['aggregate']['all'],'mae_pooled'))} "
             f"| {_fmt(_get(ora['xcity']['aggregate']['all'],'rmse_pooled'))} "
             f"| {_fmt(_get(ora['xcity']['aggregate'].get('building',{}),'mae_pooled'))} "
             f"| {_fmt(_get(ora['xcity']['aggregate'].get('building',{}),'rmse_pooled'))} |")
    L.append(f"\n_A · raw-depth cross-city Pearson r (signal check): "
             f"{_fmt(_get(A['xcity']['aggregate']['all'],'pearson_mean'))}._\n")

    # per-seed transparency (never hide a disagreeing seed)
    L.append("## Per-seed cross-city all-pixel MAE (transparency, §14)\n")
    L.append("| variant | " + " | ".join(f"seed {s}" for s in meta["seeds"]) + " | mean ± std |")
    L.append("|---|" + "|".join(["---"] * (len(meta["seeds"]) + 1)) + "|")
    for label, key in [("C_none", "C_none"), ("C_log1p", "C_log1p")]:
        ps = results["variants"][key]["per_seed"]
        cells = [_fmt(_get(p["xcity"]["aggregate"]["all"], "mae_pooled")) for p in ps]
        S = results["variants"][key]["summary"]
        L.append(f"| {label} | " + " | ".join(cells) +
                 f" | {_ms(S['xcity_all_mae_pooled_mean'], S['xcity_all_mae_pooled_std'])} |")
    L.append("")

    # the decisive table: building error by height bin (the ceiling regime)
    L.append("## Cross-city BUILDING MAE by GT-height bin (mean over seeds) — the ceiling test\n")
    L.append("_Does log1p help the tall bins WITHOUT hurting the dominant low bins? "
             "Bias<0 = under-prediction (the ceiling signature)._\n")
    bn = Sn["xcity_binned_building"]; bl = Sl["xcity_binned_building"]
    L.append("| GT bin (m) | px | C_none MAE | C_log1p MAE | ΔMAE | C_none bias | C_log1p bias |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    for i in range(min(len(bn), len(bl))):
        d = (bn[i]["mae_mean"] - bl[i]["mae_mean"]) if (bn[i]["mae_mean"] is not None
                                                        and bl[i]["mae_mean"] is not None) else None
        arrow = ""
        if d is not None:
            arrow = " ✅" if d > 0.05 else (" ❌" if d < -0.05 else "")
        L.append(f"| {_bin_label(bn[i])} | {bn[i]['n_pixels']:,} "
                 f"| {_fmt(bn[i]['mae_mean'])} | {_fmt(bl[i]['mae_mean'])} "
                 f"| {_fmt(d)}{arrow} | {_fmt(bn[i]['bias_mean'])} | {_fmt(bl[i]['bias_mean'])} |")
    L.append("\n_ΔMAE = C_none − C_log1p (positive = log1p better). ✅/❌ mark |Δ|>0.05 m._\n")

    # in-domain sanity
    L.append("## In-domain (held-out TRAIN-city tiles) — mean ± std\n")
    L.append("| variant | all MAE | building MAE |")
    L.append("|---|---|---|")
    for label, S in [("C_none", Sn), ("C_log1p", Sl)]:
        L.append(f"| {label} "
                 f"| {_ms(S['indomain_all_mae_pooled_mean'], S['indomain_all_mae_pooled_std'])} "
                 f"| {_ms(S['indomain_building_mae_pooled_mean'], S['indomain_building_mae_pooled_std'])} |")
    L.append("")

    # per-variant auditable verdict
    L.append("## Per-variant auditable verdict (decide(), context only)\n")
    for label, key in [("C_none (original)", "C_none"), ("C_log1p (modified)", "C_log1p")]:
        v = results["variants"][key]["verdict"]
        L.append(f"- **{label}: {v['verdict']}** — {v['summary']}")
    L.append("")
    cmp = results["comparison"]
    L.append("## Comparison summary (log1p − none)\n")
    L.append(f"- cross-city all MAE Δ: **{_fmt(cmp['d_all_mae'])} m** "
             f"(neg = log1p worse; pos = log1p better)")
    L.append(f"- cross-city building MAE Δ: **{_fmt(cmp['d_bldg_mae'])} m**")
    L.append(f"- cross-city tall-building (>{CEILING:.0f} m bins) MAE Δ: "
             f"**{_fmt(cmp['d_tall_bldg_mae'])} m**")
    L.append(f"- beats affine cross-city? none: **{cmp['none_beats_B']}**, "
             f"log1p: **{cmp['log1p_beats_B']}** (≥10% MAE cut over B)\n")
    L.append("> This run does NOT establish ISRO/Indian-sensor readiness and does NOT "
             "build the product. It is one controlled modification, city-held-out on "
             "DFC2019 only. STOP for human review.\n")
    Path(path).write_text("\n".join(L), encoding="utf-8")
    print(f"[report] wrote {path}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2_modify.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="force synthetic data + tiny run (NOT evidence)")
    ap.add_argument("--allow-fake-depth", action="store_true",
                    help="fabricate depth from GT if torch/transformers absent (NOT evidence)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

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
              "Phase-2) cannot run. Re-run with --allow-fake-depth for a NON-EVIDENCE "
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
        print("[error] torch required for Phase-2 (Baseline C). Aborting."); sys.exit(4)

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

    # ---- the controlled comparison: C none vs C log1p ----
    ps_none, fit_none = _fit_eval_variant("none", cfg, train, val, test, edges, True)
    ps_log, fit_log = _fit_eval_variant("log1p", cfg, train, val, test, edges, True)
    Sn, Sl = _summarize(ps_none), _summarize(ps_log)
    Vn = _variant_verdict(A, B, Sn)
    Vl = _variant_verdict(A, B, Sl)

    # ---- comparison scalars ----
    b_all_mae = _get(B["xcity"]["aggregate"]["all"], "mae_pooled")

    def _beats_B(S):
        c = S["xcity_all_mae_pooled_mean"]
        return bool(b_all_mae and c is not None and (b_all_mae - c) / b_all_mae >= 0.10)

    def _tall_bldg_mae(S):
        bins = S["xcity_binned_building"]
        num = den = 0.0
        for bb in bins:
            if bb["lo"] >= CEILING and bb["mae_mean"] is not None and bb["n_pixels"]:
                num += bb["mae_mean"] * bb["n_pixels"]; den += bb["n_pixels"]
        return (num / den) if den else None

    tall_n, tall_l = _tall_bldg_mae(Sn), _tall_bldg_mae(Sl)
    comparison = {
        "d_all_mae": (Sn["xcity_all_mae_pooled_mean"] - Sl["xcity_all_mae_pooled_mean"])
        if (Sn["xcity_all_mae_pooled_mean"] is not None
            and Sl["xcity_all_mae_pooled_mean"] is not None) else None,
        "d_bldg_mae": (Sn["xcity_building_mae_pooled_mean"] - Sl["xcity_building_mae_pooled_mean"])
        if (Sn["xcity_building_mae_pooled_mean"] is not None
            and Sl["xcity_building_mae_pooled_mean"] is not None) else None,
        "d_tall_bldg_mae": (tall_n - tall_l) if (tall_n is not None and tall_l is not None) else None,
        "none_beats_B": _beats_B(Sn), "log1p_beats_B": _beats_B(Sl),
        "tall_bldg_mae_none": tall_n, "tall_bldg_mae_log1p": tall_l,
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
        "variants": {
            "C_none": {"per_seed": ps_none, "summary": Sn, "verdict": Vn},
            "C_log1p": {"per_seed": ps_log, "summary": Sl, "verdict": Vl},
        },
        "comparison": comparison,
    }

    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    _write_report(results, str(out_dir / "PHASE2_COMPARISON.md"))
    _make_figures({"C_none": fit_none, "C_log1p": fit_log}, test, cfg, fig_dir,
                  {"C_none": Sn, "C_log1p": Sl}, evidence_valid)

    print("\n" + "=" * 72)
    print("PHASE-2 MODIFY CHECKPOINT — controlled log1p vs original C")
    print(f"  cross-city all MAE:  none={_fmt(Sn['xcity_all_mae_pooled_mean'])}±"
          f"{_fmt(Sn['xcity_all_mae_pooled_std'])}  "
          f"log1p={_fmt(Sl['xcity_all_mae_pooled_mean'])}±{_fmt(Sl['xcity_all_mae_pooled_std'])}  "
          f"(B={_fmt(b_all_mae)})")
    print(f"  cross-city bldg MAE: none={_fmt(Sn['xcity_building_mae_pooled_mean'])}  "
          f"log1p={_fmt(Sl['xcity_building_mae_pooled_mean'])}")
    print(f"  tall (>{CEILING:.0f}m) bldg MAE: none={_fmt(tall_n)}  log1p={_fmt(tall_l)}")
    print(f"  beats affine xcity: none={comparison['none_beats_B']} "
          f"log1p={comparison['log1p_beats_B']}")
    if not evidence_valid:
        print("\n[!] evidence_valid=False — plumbing only; NOT a real result.")
    print("=" * 72)
    print("STOP: human reviews these numbers + error maps before any GO decision.")


if __name__ == "__main__":
    main()
