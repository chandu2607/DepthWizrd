import os
import sys
import json
import time
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.config import TrainConfig
from depthwizard.models.fusion_head import LearnedFusionHead
from depthwizard.eval.evaluate import evaluate_estimator
from depthwizard.viz import plots

MANIFEST_PATH = "runs/dfc2023_multicity_prep/split_manifest.csv"
DATA_DIR = Path("data/dfc2023_multicity")
OUT_DIR = Path("runs/dfc2023_multicity")

CEILING = 14.0
HEIGHT_EDGES = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, float("inf")]

def load_split(manifest_path, split_type, arm_a_only=False):
    df = pd.read_csv(manifest_path)
    if split_type == 'train':
        if arm_a_only:
            df = df[(df['split'] == 'train') & (df['train_arm_a'] == 'yes')]
        else:
            df = df[df['split'] == 'train']
    else:
        df = df[df['split'] == split_type]
    return df['tile_id'].tolist()

def load_samples(tile_ids):
    from depthwizard.depth.depth_anything import DepthAnythingV2
    from depthwizard.config import DepthConfig
    dcfg = DepthConfig(cache_dir=str(DATA_DIR / "depth_cache"))
    depth_model = DepthAnythingV2(dcfg.model_id, dcfg.input_size, dcfg.cache_dir, True)

    samples = []
    for tid in tile_ids:
        rgb_path = DATA_DIR / "rgb" / tid
        dsm_path = DATA_DIR / "dsm" / tid
        
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        gt = cv2.imread(str(dsm_path), cv2.IMREAD_UNCHANGED)
        if gt is None:
            continue
        gt = gt.astype(np.float32)
        
        depth = depth_model.infer(rgb, tid, target_hw=rgb.shape[:2])
        
        city = next((c for c in ["Barcelona", "Berlin", "Brasilia", "Copenhagen", "NewDelhi", "NewYork", "Portsmouth", "Rio", "SanDiego", "SaoLuis", "Sydney"] if c in tid), "Unknown")
        
        # DFC2023 doesn't have CLS rasters extracted; proxy building pixels as nDSM > 2.0
        cls = (gt > 2.0).astype(np.uint8) * 6
        
        samples.append({"id": tid, "city": city, "rgb": rgb, "gt": gt, "depth": depth, "nodata": -999.0, "cls": cls})
    return samples

def _get(agg, key):
    v = (agg or {}).get(key)
    return v if (v is not None and v == v) else None

def _mean_std(vals):
    a = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return None, None
    return float(a.mean()), float(a.std())

def _challenge_ids(test, nodata, ceiling=30.0, k=3):
    scored = []
    for s in test:
        gt = np.asarray(s["gt"], np.float32)
        valid = np.isfinite(gt) & (gt != nodata)
        tall = int((valid & (gt > ceiling)).sum())
        scored.append((tall, s["id"]))
    scored.sort(reverse=True)
    return [sid for _, sid in scored[:k] if scored]

def compute_exposure(samples):
    counts = {15: 0, 20: 0, 30: 0, 40: 0}
    for s in samples:
        gt = s["gt"]
        valid = np.isfinite(gt) & (gt != -999.0)
        gt_v = gt[valid]
        counts[15] += int((gt_v > 15).sum())
        counts[20] += int((gt_v > 20).sum())
        counts[30] += int((gt_v > 30).sum())
        counts[40] += int((gt_v > 40).sum())
    return counts

def main():
    if not HAS_TORCH:
        print("ERROR: torch required.")
        sys.exit(1)
        
    print("Loading data...")
    arm_a_ids = load_split(MANIFEST_PATH, 'train', arm_a_only=True)
    arm_b_ids = load_split(MANIFEST_PATH, 'train', arm_a_only=False)
    val_ids = load_split(MANIFEST_PATH, 'val')
    test_ids = load_split(MANIFEST_PATH, 'test')
    
    print(f"Arm A: {len(arm_a_ids)}, Arm B: {len(arm_b_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    train_a = load_samples(arm_a_ids)
    train_b = load_samples(arm_b_ids)
    val = load_samples(val_ids)
    test = load_samples(test_ids)
    
    exposure_a = compute_exposure(train_a)
    exposure_b = compute_exposure(train_b)
    print("Exposure A:", exposure_a)
    print("Exposure B:", exposure_b)
    
    class FakeCfg:
        class data:
            nodata = -999.0
            building_label = 6
    cfg_eval = FakeCfg()

    tcfg = TrainConfig(
        arch="unet3",
        target_transform="log1p",
        loss_type="standard",
        epochs=15,
        batch_size=4,
        lr=1e-3,
        amp=True
    )

    results = {"runs": []}
    
    # Store fitted models from seed 0 for plotting
    model_a_s0 = None
    model_b_s0 = None
    
    for arm_name, train_data in [("ArmA", train_a), ("ArmB", train_b)]:
        for seed in [0, 1]:
            print(f"\n--- Starting {arm_name} Seed {seed} ---")
            t0 = time.time()
            model = LearnedFusionHead(tcfg, nodata=-999.0, seed=seed)
            model.fit(train_data)
            
            vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
            runtime = time.time() - t0
            
            print(f"Evaluating Validation (Copenhagen)...")
            val_res = evaluate_estimator(model, val, cfg_eval, "indomain", bin_edges=HEIGHT_EDGES)
            
            print(f"Evaluating Test (NewYork)...")
            test_res = evaluate_estimator(model, test, cfg_eval, "xcity", bin_edges=HEIGHT_EDGES)
            
            run_info = {
                "arm": arm_name,
                "seed": seed,
                "tiles": len(train_data),
                "runtime": runtime,
                "vram": vram,
                "val": val_res,
                "test": test_res
            }
            results["runs"].append(run_info)
            
            if seed == 0:
                if arm_name == "ArmA": model_a_s0 = model
                if arm_name == "ArmB": model_b_s0 = model
                
            # Quick summary
            xa = test_res["aggregate"]["all"]
            print(f"[{arm_name}|seed={seed}] Test MAE: {_get(xa,'mae_pooled'):.3f} | VRAM: {vram:.1f}MB | Time: {runtime:.1f}s")
            
    # Compute summaries
    summary = {}
    for arm_name in ["ArmA", "ArmB"]:
        runs = [r for r in results["runs"] if r["arm"] == arm_name]
        arm_sum = {}
        
        def agg_scalar(grp, kind, metric):
            return _mean_std([_get(r[grp]["aggregate"].get(kind, {}), metric) for r in runs])
            
        for kind in ["all", "building"]:
            for metric in ["mae_pooled", "rmse_pooled", "pearson_mean", "bias_mean"]:
                m, s = agg_scalar("test", kind, metric)
                arm_sum[f"{kind}_{metric}"] = (m, s)
                
        # Binned metrics
        bins = []
        for i in range(len(HEIGHT_EDGES)-1):
            lo = HEIGHT_EDGES[i]
            hi = HEIGHT_EDGES[i+1]
            # test['aggregate']['binned_all'][i]['mae']
            # Find the bin index for each run
            maes = []
            biases = []
            for r in runs:
                blist = r["test"]["aggregate"].get("binned_all", [])
                if i < len(blist):
                    maes.append(blist[i].get("mae"))
                    biases.append(blist[i].get("bias"))
            m_mae, s_mae = _mean_std(maes)
            m_bias, s_bias = _mean_std(biases)
            
            bins.append({
                "lo": lo, "hi": hi,
                "mae_m": m_mae, "mae_s": s_mae,
                "bias_m": m_bias, "bias_s": s_bias
            })
            
        # Extra >15, >20, >30, >40 metrics
        def gt_threshold(thresh):
            maes, rmses, biases = [], [], []
            for r in runs:
                # evaluate_estimator provides a way to compute this if we requested it, 
                # but we didn't natively group by >X. We'll approximate by summing bins
                # Actually, I can just recalculate manually for the model or rely on bins.
                pass 
                
        arm_sum["bins"] = bins
        summary[arm_name] = arm_sum
        
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Custom post-eval for thresholds using model_a_s0 and model_b_s0
    def eval_threshold(model, samples, t):
        errs = []
        for s in samples:
            pred = model.predict(s)
            gt = s["gt"]
            valid = np.isfinite(gt) & (gt != -999.0) & (gt > t)
            if valid.sum() > 0:
                errs.extend(np.abs(pred[valid] - gt[valid]).tolist())
        return np.mean(errs) if errs else None
        
    print("Generating visual results...")
    import matplotlib
    matplotlib.use("Agg")
    
    c_ids = _challenge_ids(test, -999.0, ceiling=25.0, k=4)
    test_dict = {s["id"]: s for s in test}
    for sid in c_ids:
        s = test_dict[sid]
        for name, model in [("ArmA", model_a_s0), ("ArmB", model_b_s0)]:
            if model is None: continue
            pred = model.predict(s)
            plots.save_qualitative(s, pred, str(fig_dir / f"challenge_{sid}_{name}.png"), nodata=-999.0, title=f"{name} (seed0) · {sid}")
            
    print("Writing report...")
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump({"runs": results["runs"], "exposure_a": exposure_a, "exposure_b": exposure_b}, f, indent=2)
        
    def fmt(x): return f"{x[0]:.3f}±{x[1]:.3f}" if (x and x[0] is not None) else "n/a"
    
    with open(OUT_DIR / "REPORT.md", "w") as f:
        f.write("# PHASE 9 - MULTI-CITY TRAINING EXPERIMENT\n\n")
        f.write("## Hypothesis\nBroader training across multiple cities exposes the model to more diverse depth->height relationships and improves unseen-city generalization.\n\n")
        
        f.write("## Exposure\n")
        f.write(f"- Arm A >15m px: {exposure_a[15]}, >30m: {exposure_a[30]}\n")
        f.write(f"- Arm B >15m px: {exposure_b[15]}, >30m: {exposure_b[30]}\n\n")
        
        f.write("## Results (Mean ± Std over Seeds)\n")
        f.write("| Metric | Arm A | Arm B |\n")
        f.write("|---|---|---|\n")
        for metric in ["all_mae_pooled", "all_rmse_pooled", "building_mae_pooled", "building_rmse_pooled", "building_bias_mean"]:
            f.write(f"| {metric} | {fmt(summary['ArmA'].get(metric))} | {fmt(summary['ArmB'].get(metric))} |\n")
            
        f.write("\n## Height Bins (All pixels)\n")
        f.write("| Bin | Arm A MAE | Arm B MAE | Arm A Bias | Arm B Bias |\n")
        f.write("|---|---|---|---|---|\n")
        for i in range(len(summary['ArmA']['bins'])):
            ba = summary['ArmA']['bins'][i]
            bb = summary['ArmB']['bins'][i]
            f.write(f"| {ba['lo']}-{ba['hi']} | {ba['mae_m']:.3f} | {bb['mae_m']:.3f} | {ba['bias_m']:.3f} | {bb['bias_m']:.3f} |\n")

    print(f"Done. Results in {OUT_DIR}")

if __name__ == "__main__":
    main()
