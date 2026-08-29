import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.config import TrainConfig
from depthwizard.models.fusion_head import LearnedFusionHead
from depthwizard.eval.evaluate import evaluate_estimator
from depthwizard.viz import plots

MANIFEST_PATH = "runs/dfc2023_multicity_prep/split_manifest.csv"
DATA_DIR = Path("data/dfc2023_multicity")
OUT_DIR = Path("runs/phase11_input_ablation")

HEIGHT_EDGES = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, float("inf")]

def load_split(manifest_path, split_type):
    df = pd.read_csv(manifest_path)
    if split_type == 'train':
        # Train on the 9 cities from Arm B (multi-city)
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
        
        cls = (gt > 2.0).astype(np.uint8) * 6
        
        samples.append({"id": tid, "city": city, "rgb": rgb, "gt": gt, "depth": depth, "nodata": -999.0, "cls": cls})
    return samples

def _get(agg, key):
    v = (agg or {}).get(key)
    return v if (v is not None and v == v) else None

def _mean_std(vals):
    a = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0: return None, None
    return float(a.mean()), float(a.std())

def _challenge_ids(test, nodata, ceiling=30.0, k=4):
    scored = []
    for s in test:
        gt = np.asarray(s["gt"], np.float32)
        valid = np.isfinite(gt) & (gt != nodata)
        tall = int((valid & (gt > ceiling)).sum())
        scored.append((tall, s["id"]))
    scored.sort(reverse=True)
    return [sid for _, sid in scored[:k] if scored]

def main():
    print("Loading data...")
    train_ids = load_split(MANIFEST_PATH, 'train')
    val_ids = load_split(MANIFEST_PATH, 'val')
    test_ids = load_split(MANIFEST_PATH, 'test')
    
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    train = load_samples(train_ids)
    val = load_samples(val_ids)
    test = load_samples(test_ids)
    
    class FakeCfg:
        class data:
            nodata = -999.0
            building_label = 6
    cfg_eval = FakeCfg()

    results = {"runs": []}
    models_seed0 = {}
    
    modes = ["rgb", "depth", "rgb_depth"]
    seeds = [0, 1]
    
    for mode in modes:
        mode_dir = OUT_DIR / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        
        for seed in seeds:
            print(f"\n=== Starting Mode: {mode.upper()} | Seed: {seed} ===")
            tcfg = TrainConfig(
                arch="unet3",
                target_transform="log1p",
                loss_type="standard",
                epochs=15,
                batch_size=4,
                lr=1e-3,
                amp=True
            )
            setattr(tcfg, "input_mode", mode)
            
            t0 = time.time()
            model = LearnedFusionHead(tcfg, nodata=-999.0, seed=seed)
            model.fit(train)
            
            vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
            runtime = time.time() - t0
            
            print(f"Evaluating Validation (Copenhagen)...")
            val_res = evaluate_estimator(model, val, cfg_eval, "indomain", bin_edges=HEIGHT_EDGES)
            
            print(f"Evaluating Test (NewYork)...")
            test_res = evaluate_estimator(model, test, cfg_eval, "xcity", bin_edges=HEIGHT_EDGES)
            
            run_info = {
                "mode": mode,
                "seed": seed,
                "runtime": runtime,
                "vram": vram,
                "val": val_res,
                "test": test_res
            }
            results["runs"].append(run_info)
            
            if seed == 0:
                models_seed0[mode] = model
                
            xa = test_res["aggregate"]["all"]
            print(f"[{mode}|seed={seed}] Test MAE: {_get(xa,'mae_pooled'):.3f} | VRAM: {vram:.1f}MB | Time: {runtime:.1f}s")
            
    # Compute summaries
    summary = {}
    for mode in modes:
        runs = [r for r in results["runs"] if r["mode"] == mode]
        mode_sum = {}
        
        def agg_scalar(grp, kind, metric):
            return _mean_std([_get(r[grp]["aggregate"].get(kind, {}), metric) for r in runs])
            
        for kind in ["all", "building"]:
            for metric in ["mae_pooled", "rmse_pooled", "pearson_mean", "bias_mean"]:
                m, s = agg_scalar("test", kind, metric)
                mode_sum[f"{kind}_{metric}"] = (m, s)
                
        bins = []
        for i in range(len(HEIGHT_EDGES)-1):
            lo = HEIGHT_EDGES[i]
            hi = HEIGHT_EDGES[i+1]
            maes, biases = [], []
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
            
        mode_sum["bins"] = bins
        summary[mode] = mode_sum
        
    print("Generating visual results...")
    import matplotlib
    matplotlib.use("Agg")
    
    c_ids = _challenge_ids(test, -999.0, ceiling=25.0, k=4)
    test_dict = {s["id"]: s for s in test}
    for sid in c_ids:
        s = test_dict[sid]
        for mode in modes:
            model = models_seed0[mode]
            pred = model.predict(s)
            mode_dir = OUT_DIR / mode
            plots.save_qualitative(s, pred, str(mode_dir / f"challenge_{sid}_{mode}.png"), nodata=-999.0, title=f"{mode.upper()} (seed0) | {sid}")
            
    print("Writing report...")
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump({"runs": results["runs"]}, f, indent=2)
        
    def fmt(x): return f"{x[0]:.3f}±{x[1]:.3f}" if (x and x[0] is not None) else "n/a"
    
    with open(OUT_DIR / "REPORT.md", "w") as f:
        f.write("# PHASE 11 - INPUT ABLATION EXPERIMENT\n\n")
        f.write("## Results (Test: NewYork, Mean ± Std over Seeds)\n")
        f.write("| Metric | RGB-only | Depth-only | RGB+Depth |\n")
        f.write("|---|---|---|---|\n")
        for metric in ["all_mae_pooled", "all_rmse_pooled", "all_pearson_mean", "building_mae_pooled", "building_rmse_pooled", "building_bias_mean"]:
            f.write(f"| {metric} | {fmt(summary['rgb'].get(metric))} | {fmt(summary['depth'].get(metric))} | {fmt(summary['rgb_depth'].get(metric))} |\n")
            
        f.write("\n## Height Bins (All pixels)\n")
        f.write("| Bin | RGB MAE | Depth MAE | RGB+Depth MAE | RGB Bias | Depth Bias | RGB+Depth Bias |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for i in range(len(summary['rgb']['bins'])):
            br = summary['rgb']['bins'][i]
            bd = summary['depth']['bins'][i]
            brd = summary['rgb_depth']['bins'][i]
            f.write(f"| {br['lo']}-{br['hi']} | {br['mae_m']:.3f} | {bd['mae_m']:.3f} | {brd['mae_m']:.3f} | {br['bias_m']:.3f} | {bd['bias_m']:.3f} | {brd['bias_m']:.3f} |\n")

    print(f"Done. Results in {OUT_DIR}")

if __name__ == "__main__":
    main()
