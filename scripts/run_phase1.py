#!/usr/bin/env python
"""DepthWizard Phase-1 feasibility orchestrator.

Pipeline:
  resolve dataset (real DFC2019 preferred; synthetic fallback)
   -> city-held-out split (train city / in-domain val / held-out test city)
   -> frozen Depth Anything V2 relative-depth prior (cached), or fake-depth stub
   -> Baseline A (raw depth), B (global affine), C (small learned fusion) [, D]
   -> evaluate IN-DOMAIN and CROSS-CITY (all / building / non-building, per-scene)
   -> GO/MODIFY/ABANDON recommendation (computed from measured metrics)
   -> results.json, qualitative figures, EXPERIMENT_RESULTS.md

STOP after this: the Phase-2 checkpoint verdict is produced here for a HUMAN to
act on. This script does not build the full pipeline / 3D flythrough.

Usage:
  python scripts/run_phase1.py --config configs/phase1.yaml
  python scripts/run_phase1.py --smoke --allow-fake-depth   # offline plumbing test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# make `import depthwizard` work when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.config import load_config, config_to_dict
from depthwizard.data import fetch, datasets
from depthwizard.depth.depth_anything import DepthAnythingV2
from depthwizard.depth.fake import FakeDepth
from depthwizard.models.affine import RawDepth, GlobalAffine
from depthwizard.eval.evaluate import evaluate_estimator, evaluate_oracle
from depthwizard.eval.decision import decide
from depthwizard.eval import report as report_mod


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


def materialize_real(records, cfg, depth_model, fake):
    samples = [datasets.load_sample(r, cfg.data.tile_size, cfg.data.nodata,
                                    depth_model=None) for r in records]
    return _attach_depth(samples, cfg, depth_model, fake)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="force synthetic smoke-test data (NOT evidence)")
    ap.add_argument("--allow-fake-depth", action="store_true",
                    help="fabricate depth from GT when torch/transformers absent (NOT evidence)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        cfg.data.source = "synthetic"
    out_dir = Path(args.out or cfg.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    has_torch = _has_torch()
    t0 = time.time()

    # ---- resolve dataset ----
    source, records = fetch.resolve_records(cfg)
    is_synthetic = (source == "synthetic")

    # ---- depth prior (real frozen DA-V2, or fake stub for offline plumbing) ----
    use_fake = args.allow_fake_depth or (not has_torch)
    if use_fake and not args.allow_fake_depth:
        print("[warn] torch/transformers unavailable. Re-run with --allow-fake-depth "
              "for a NON-EVIDENCE offline smoke test. Aborting to avoid silent fakery.")
        sys.exit(2)
    fake = FakeDepth(seed=cfg.split.seed) if use_fake else None
    depth_model = None if use_fake else DepthAnythingV2(
        cfg.depth.model_id, cfg.depth.input_size,
        cache_dir=str(out_dir / "depth_cache") if cfg.depth.use_cache else None,
        use_cache=cfg.depth.use_cache)

    # ---- build splits + materialize samples ----
    if is_synthetic:
        train, val, test = fetch.synthetic_samples(cfg)
        for grp in (train, val, test):
            _attach_depth(grp, cfg, depth_model, fake)
    else:
        tr_rec, va_rec, te_rec = datasets.split_by_city(
            records, cfg.split.train_cities, cfg.split.val_cities,
            cfg.split.test_cities, cfg.split.val_fraction_within_train_city,
            cfg.split.seed, cfg.data.max_tiles_per_city)
        print(f"[split] train={len(tr_rec)} val={len(va_rec)} test={len(te_rec)} "
              f"(train {cfg.split.train_cities} / test {cfg.split.test_cities})")
        train = materialize_real(tr_rec, cfg, depth_model, fake)
        val = materialize_real(va_rec, cfg, depth_model, fake)
        test = materialize_real(te_rec, cfg, depth_model, fake)

    if not train or not test:
        print("[error] empty train or test split; check dataset / city names.")
        sys.exit(3)

    evidence_valid = (not is_synthetic) and (fake is None)
    device = "cuda" if (has_torch and _cuda()) else "cpu"

    results: dict = {"indomain": {}, "xcity": {}, "reproducibility": {}}
    letters = [b.upper() for b in cfg.baselines]
    fig_estimator = None  # a live metric estimator we can re-predict for figures

    # ---- Baseline A: raw relative depth (scale-free) ----
    if "A" in letters:
        a = RawDepth().fit(train)
        results["indomain"][a.name] = evaluate_estimator(a, val, cfg, "indomain")
        results["xcity"][a.name] = evaluate_estimator(a, test, cfg, "xcity")
        print(f"[A] cross-city Pearson(mean)="
              f"{results['xcity'][a.name]['aggregate']['all'].get('pearson_mean')}")

    # ---- Baseline B: global affine calibration ----
    if "B" in letters:
        b = GlobalAffine(max_pixels=cfg.train.max_train_pixels_affine,
                         seed=cfg.split.seed).fit(train)
        results["indomain"][b.name] = evaluate_estimator(b, val, cfg, "indomain")
        results["xcity"][b.name] = evaluate_estimator(b, test, cfg, "xcity")
        fig_estimator = b
        print(f"[B] a={b.a:.4g} b={b.b:.4g} | cross-city MAE(pooled)="
              f"{results['xcity'][b.name]['aggregate']['all'].get('mae_pooled')}")

    # ---- Baseline C: small learned fusion head (THE hypothesis) ----
    head_params = None
    if "C" in letters:
        if not has_torch:
            print("[C] skipped: torch not installed. The hypothesis test needs C, so "
                  "the decision will be INCONCLUSIVE for the learned head.")
        else:
            from depthwizard.models.fusion_head import LearnedFusionHead
            xcity_mae, xcity_rmse = [], []
            rep_idn = rep_xc = None
            for si, seed in enumerate(cfg.seeds):
                c = LearnedFusionHead(cfg.train, nodata=cfg.data.nodata, seed=seed)
                head_params = c.n_params()
                c.fit(train)
                idn = evaluate_estimator(c, val, cfg, "indomain")
                xc = evaluate_estimator(c, test, cfg, "xcity")
                m = xc["aggregate"]["all"]
                xcity_mae.append(m.get("mae_pooled"))
                xcity_rmse.append(m.get("rmse_pooled"))
                if si == 0:
                    rep_idn, rep_xc, fig_estimator = idn, xc, c
                print(f"[C][seed={seed}] cross-city MAE(pooled)={m.get('mae_pooled')}")
            results["indomain"]["C_learned_fusion"] = rep_idn
            results["xcity"]["C_learned_fusion"] = rep_xc
            mae_arr = np.array([v for v in xcity_mae if v is not None], float)
            rmse_arr = np.array([v for v in xcity_rmse if v is not None], float)
            results["reproducibility"] = {
                "seeds": list(cfg.seeds),
                "xcity_mae_mean": float(mae_arr.mean()) if mae_arr.size else None,
                "xcity_mae_std": float(mae_arr.std()) if mae_arr.size else None,
                "xcity_rmse_mean": float(rmse_arr.mean()) if rmse_arr.size else None,
                "xcity_rmse_std": float(rmse_arr.std()) if rmse_arr.size else None,
            }

    # ---- Baseline D: RDAH-Net adapter (optional, bounded effort) ----
    if "D" in letters:
        _maybe_run_rdah(cfg, val, test, results)

    # ---- oracle upper bound (per-image affine; peeks at GT; not deployable) ----
    results["xcity"]["oracle_affine_upper_bound"] = evaluate_oracle(test, cfg, "xcity")
    results["indomain"]["oracle_affine_upper_bound"] = evaluate_oracle(val, cfg, "indomain")

    # ---- decision + metadata ----
    dec = decide(results, evidence_valid=evidence_valid)
    results["decision"] = dec
    results["meta"] = {
        "source": source, "evidence_valid": evidence_valid,
        "fake_depth": fake is not None,
        "depth_model": (cfg.depth.model_id if fake is None else "FAKE_STUB"),
        "device": device, "has_torch": has_torch,
        "train_cities": cfg.split.train_cities, "val_cities": cfg.split.val_cities,
        "test_cities": cfg.split.test_cities,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "tile_size": cfg.data.tile_size, "seeds": list(cfg.seeds),
        "head_params": head_params, "elapsed_s": round(time.time() - t0, 1),
        "config": config_to_dict(cfg),
    }

    # ---- persist results + report ----
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    report_mod.write_report(results, str(out_dir / "EXPERIMENT_RESULTS.md"))
    # Only an evidence-valid run may overwrite the repo-root record; a synthetic /
    # fake-depth smoke run must never clobber the committed template.
    if evidence_valid:
        report_mod.write_report(
            results, str(Path(__file__).resolve().parents[1] / "EXPERIMENT_RESULTS.md"))
    else:
        print("[report] non-evidence run: repo-root EXPERIMENT_RESULTS.md left untouched "
              f"(see {out_dir / 'EXPERIMENT_RESULTS.md'}).")

    # ---- qualitative figures from a LIVE estimator (C if available, else B) ----
    _make_figures(fig_estimator, results, test, cfg, fig_dir, evidence_valid)

    # ---- checkpoint banner ----
    print("\n" + "=" * 72)
    print(f"PHASE-2 CHECKPOINT — verdict: {dec['verdict']}")
    print(dec["summary"])
    if not evidence_valid:
        print("\n[!] evidence_valid=False — this run does NOT justify any verdict; it "
              "only proves the code runs. Re-run on real DFC2019 + real depth prior.")
    print("=" * 72)
    print("STOP: do not proceed to the full pipeline / 3D flythrough until a HUMAN "
          "reviews these numbers and the error maps and confirms GO.")


def _maybe_run_rdah(cfg, val, test, results):
    from depthwizard.models.rdah_net import RDAHNetAdapter
    try:
        d = RDAHNetAdapter()  # repo_dir/checkpoint come from config in a real wiring
        if not d.available():
            print("[D] RDAH-Net not configured/available; skipping (Baseline C already "
                  "tests the same principle). See depthwizard/models/rdah_net.py.")
            return
        d.fit(train_samples=[])
        results["indomain"][d.name] = evaluate_estimator(d, val, cfg, "indomain")
        results["xcity"][d.name] = evaluate_estimator(d, test, cfg, "xcity")
    except NotImplementedError:
        print("[D] RDAH-Net adapter not wired (TODOs in rdah_net.py); skipping per "
              "bounded-effort policy.")
    except Exception as e:
        print(f"[D] RDAH-Net attempt failed ({e}); skipping per bounded-effort policy.")


def _make_figures(estimator, results, test, cfg, fig_dir, evidence_valid):
    if estimator is None:
        print("[viz] no metric estimator available; skipping figures.")
        return
    try:
        from depthwizard.viz import plots
    except Exception as e:
        print(f"[viz] skipped ({e})")
        return
    ev = results["xcity"].get(getattr(estimator, "name", ""))
    if not ev:
        return
    ids = plots.select_scenes(ev["per_scene"], k=3)
    by_id = {s["id"]: s for s in test}
    tag = "" if evidence_valid else "  [NON-EVIDENCE RUN]"
    ps, gs = [], []
    for sid in ids:
        s = by_id.get(sid)
        if s is None:
            continue
        pred = estimator.predict(s)
        plots.save_qualitative(
            s, pred, str(fig_dir / f"scene_{sid}_{estimator.name}.png"),
            nodata=cfg.data.nodata, title=f"{estimator.name} · {sid} · cross-city{tag}")
        ps.append(np.asarray(pred).ravel())
        gs.append(np.asarray(s["gt"], dtype=np.float32).ravel())
    if ps:
        plots.save_scatter(
            np.concatenate(ps), np.concatenate(gs),
            str(fig_dir / f"scatter_{estimator.name}_xcity.png"),
            nodata=cfg.data.nodata,
            title=f"{estimator.name} cross-city: pred vs reference (m){tag}")
        print(f"[viz] wrote {len(ids)} scene panels + scatter to {fig_dir}")


if __name__ == "__main__":
    main()
