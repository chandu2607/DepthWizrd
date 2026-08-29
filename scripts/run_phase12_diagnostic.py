import os
import sys
import json
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT_DIR = Path("runs/phase12_scene_regime")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data/dfc2023_multicity")

def get_binned_errs(binned_list, thresholds):
    # Calculate MAE for >T by aggregating bins
    res = {}
    for t in thresholds:
        total_mae = 0.0
        total_px = 0
        for b in binned_list:
            if b.get("lo", 0) >= t and b.get("mae") == b.get("mae"): # not nan
                n = b.get("n_pixels", 0)
                mae = b.get("mae", 0.0)
                if n > 0:
                    total_mae += mae * n
                    total_px += n
        res[f">{int(t)}m_MAE"] = (total_mae / total_px) if total_px > 0 else np.nan
    return res

def main():
    print("Loading Phase 11 results.json...")
    with open("runs/phase11_input_ablation/results.json", "r") as f:
        res = json.load(f)
        
    # Extract seed 0 results for Depth-only (the best model) and RGB+Depth
    depth_run = next(r for r in res["runs"] if r["mode"] == "depth" and r["seed"] == 0)
    rgbd_run = next(r for r in res["runs"] if r["mode"] == "rgb_depth" and r["seed"] == 0)
    
    depth_scenes = {s["id"]: s for s in depth_run["test"]["per_scene"]}
    rgbd_scenes = {s["id"]: s for s in rgbd_run["test"]["per_scene"]}
    
    records = []
    
    for sid, d_scene in depth_scenes.items():
        if "NewYork" not in sid:
            continue
            
        r_scene = rgbd_scenes.get(sid)
        
        dsm_path = DATA_DIR / "dsm" / sid
        from depthwizard.depth.depth_anything import DepthAnythingV2
        from depthwizard.config import DepthConfig
        dcfg = DepthConfig(cache_dir=str(DATA_DIR / "depth_cache"))
        depth_model = DepthAnythingV2(dcfg.model_id, dcfg.input_size, dcfg.cache_dir, True)
        
        depth_path = depth_model._cache_path(sid)
        
        gt = cv2.imread(str(dsm_path), cv2.IMREAD_UNCHANGED)
        if gt is None:
            continue
            
        gt = gt.astype(np.float32)
        valid = np.isfinite(gt) & (gt != -999.0)
        gt_v = gt[valid]
        
        if len(gt_v) == 0:
            continue
            
        # load depth cache
        try:
            depth_data = np.load(depth_path)
            depth_v = depth_data[valid]
        except Exception:
            depth_v = None

        # Statistics
        true_max = gt_v.max()
        p90 = np.percentile(gt_v, 90)
        p95 = np.percentile(gt_v, 95)
        p99 = np.percentile(gt_v, 99)
        
        f15 = (gt_v > 15).sum() / len(gt_v)
        f20 = (gt_v > 20).sum() / len(gt_v)
        f30 = (gt_v > 30).sum() / len(gt_v)
        f40 = (gt_v > 40).sum() / len(gt_v)
        
        bld_mask = gt_v > 2.0
        bld_frac = bld_mask.sum() / len(gt_v)
        bld_mean = gt_v[bld_mask].mean() if bld_mask.sum() > 0 else 0.0
        
        oracle_scale = np.nan
        if depth_v is not None and len(depth_v) > 2:
            # simple robust scale: cov(depth, gt) / var(depth)
            cov = np.cov(depth_v, gt_v)[0,1]
            var = np.var(depth_v)
            if var > 1e-6:
                oracle_scale = cov / var
                
        # Depth-only metrics
        d_all = d_scene["all"]
        d_mae = d_all.get("mae")
        d_rmse = d_all.get("rmse")
        d_bias = d_all.get("mean_pred", 0) - d_all.get("mean_gt", 0)
        d_tall = get_binned_errs(d_scene.get("binned_all", []), [15, 20, 30, 40])
        
        # RGBD metrics
        r_mae = np.nan
        if r_scene:
            r_all = r_scene["all"]
            r_mae = r_all.get("mae")
            
        record = {
            "tile_id": sid,
            "true_max": true_max,
            "true_p90": p90,
            "true_p95": p95,
            "true_p99": p99,
            "frac_15": f15,
            "frac_20": f20,
            "frac_30": f30,
            "frac_40": f40,
            "bld_frac": bld_frac,
            "bld_mean": bld_mean,
            "oracle_scale": oracle_scale,
            "scene_mae_depth": d_mae,
            "scene_rmse_depth": d_rmse,
            "scene_bias_depth": d_bias,
            "scene_mae_rgbd": r_mae,
            "mae_15_depth": d_tall.get(">15m_MAE"),
            "mae_20_depth": d_tall.get(">20m_MAE"),
            "mae_30_depth": d_tall.get(">30m_MAE"),
            "mae_40_depth": d_tall.get(">40m_MAE"),
        }
        records.append(record)
        
    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "scene_analysis.csv", index=False)
    
    print("\n=== CORRELATION ANALYSIS (Spearman) ===")
    targets = ["scene_mae_depth", "mae_30_depth"]
    features = ["true_max", "frac_30", "bld_frac", "oracle_scale"]
    
    for t in targets:
        print(f"\nTarget: {t}")
        for f in features:
            v_t = df[t].values
            v_f = df[f].values
            valid = np.isfinite(v_t) & np.isfinite(v_f)
            if valid.sum() > 2:
                corr, p = spearmanr(v_f[valid], v_t[valid])
                print(f"  vs {f}: {corr:.3f} (p={p:.4f})")
                
    print("\n=== IDENTIFY FAILURE REGIMES ===")
    def regime(max_h):
        if max_h < 15: return "Low-Rise (<15m)"
        if max_h < 40: return "Mid-Rise (15-40m)"
        if max_h < 80: return "High-Rise (40-80m)"
        return "Extreme High-Rise (>80m)"
        
    df["regime"] = df["true_max"].apply(regime)
    grp = df.groupby("regime").agg({
        "tile_id": "count",
        "scene_mae_depth": "mean",
        "scene_rmse_depth": "mean",
        "scene_bias_depth": "mean",
        "mae_30_depth": "mean",
        "oracle_scale": "mean"
    }).round(2)
    print(grp)
    
    print("\n=== SCALE-REGIME TEST ===")
    # Group by oracle scale
    df["scale_bin"] = pd.qcut(df["oracle_scale"], 3, labels=["Low Scale", "Medium Scale", "High Scale"])
    grp_scale = df.groupby("scale_bin").agg({
        "tile_id": "count",
        "scene_mae_depth": "mean",
        "scene_bias_depth": "mean",
        "true_max": "mean",
        "mae_30_depth": "mean"
    }).round(2)
    print(grp_scale)
    
    print("\n=== RGB VS DEPTH ===")
    # Are there scenes where RGB+Depth is better?
    rgb_better = df[df["scene_mae_rgbd"] < df["scene_mae_depth"]]
    print(f"Scenes where RGB+Depth beats Depth-only: {len(rgb_better)} out of {len(df)}")
    if len(rgb_better) > 0:
        print("Mean true_max of these scenes:", rgb_better["true_max"].mean())
        print("Mean true_max of depth_better scenes:", df[df["scene_mae_rgbd"] >= df["scene_mae_depth"]]["true_max"].mean())

if __name__ == "__main__":
    main()
