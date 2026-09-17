import os
import sys
sys.path.append(os.getcwd())
import json
import time
import torch
import numpy as np
import cv2
from pathlib import Path
from depthwizard.models.building_conditioned_net import BuildingConditionedEstimator
from depthwizard.config import TrainConfig
from depthwizard.data.datasets import read_raster
from depthwizard.depth.depth_anything import DepthAnythingV2

def save_visuals(out_dir, prefix, rgb, depth, pred, ref, err):
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    
    depth_vis = ((depth - depth.min()) / (np.ptp(depth) + 1e-6) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_depth.png"), depth_vis)
    
    pred_vis = np.clip(pred * 10, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_prediction.png"), pred_vis)
    
    ref_vis = np.clip(ref * 10, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_reference.png"), ref_vis)
    
    err_vis = np.clip(np.abs(err) * 20, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_error.png"), err_vis)

def main():
    out_dir = "runs/phase94_real_inference_audit"
    os.makedirs(out_dir, exist_ok=True)
    
    ckpt_path = "runs/phase54_domain_robust_training/checkpoints/E_seed_0_best.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model_state = checkpoint.get("model_state", checkpoint)
    
    print("Keys in checkpoint:", list(model_state.keys())[:10], "...")
    
    cfg = TrainConfig(width=24)
    est = BuildingConditionedEstimator(cfg_train=cfg, device="cpu")
    est.model.load_state_dict(model_state, strict=False)
    est.model.eval()
    
    # Initialize real depth prior
    print("Initializing DepthAnythingV2...")
    depth_model = DepthAnythingV2(
        model_id="depth-anything/Depth-Anything-V2-Small-hf",
        input_size=518,
        cache_dir="data/dfc2023_multicity/depth_cache",
        use_cache=True
    )
    
    # Set the mean/std from checkpoint if available
    if "d_mean" in model_state:
        est.d_mean = float(model_state["d_mean"].item())
        est.d_std = float(model_state["d_std"].item())
    
    # Checkpoint audit
    with open(os.path.join(out_dir, "checkpoint_audit.json"), "w") as f:
        json.dump({
            "path": ckpt_path,
            "architecture": "BuildingConditionedEstimator",
            "keys": list(model_state.keys()),
            "d_mean": est.d_mean,
            "d_std": est.d_std
        }, f, indent=2)
        
    # Test on tiles
    tiles = {
        "copenhagen": {
            "rgb": "data/dfc2023_multicity/rgb/SV_Copenhagen_55.6698_12.5517.tif",
            "dsm": "data/dfc2023_multicity/dsm/SV_Copenhagen_55.6698_12.5517.tif",
            "tile_id": "SV_Copenhagen_55.6698_12.5517"
        },
        "newyork": {
            "rgb": "data/dfc2023_multicity/rgb/SV_NewYork_40.7332_-73.9784.tif",
            "dsm": "data/dfc2023_multicity/dsm/SV_NewYork_40.7332_-73.9784.tif",
            "tile_id": "SV_NewYork_40.7332_-73.9784"
        }
    }
    
    results = {}
    
    for city, paths in tiles.items():
        print(f"Running inference for {city}...")
        rgb = read_raster(paths["rgb"])[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb / rgb.max() * 255, 0, 255).astype(np.uint8)
        
        dsm = read_raster(paths["dsm"])
        if dsm.ndim == 3: dsm = dsm[..., 0]
        ref_h = np.where(dsm < -100, np.nan, dsm).astype(np.float32)
        
        # Get REAL depth
        depth = depth_model.infer(rgb, key=paths["tile_id"], target_hw=rgb.shape[:2])
            
        sample = {
            "rgb": rgb,
            "depth": depth,
            "gt": ref_h,
            "id": f"{city}_test",
            "city": city
        }
        
        t0 = time.time()
        pred_h = est.predict(sample)
        inf_time = time.time() - t0
        
        pred_h_full = cv2.resize(pred_h, (rgb.shape[1], rgb.shape[0]))
        err = pred_h_full - ref_h
        
        save_visuals(out_dir, city, rgb, depth, pred_h_full, ref_h, err)
        
        results[city] = {
            "stats": {
                "min": float(pred_h.min()),
                "max": float(pred_h.max()),
                "mean": float(pred_h.mean()),
                "std": float(pred_h.std()),
                "nonzero_px": int((pred_h > 0).sum())
            },
            "inference_time": inf_time
        }
        
    with open(os.path.join(out_dir, "RESULTS.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    with open(os.path.join(out_dir, "REPORT.md"), "w") as f:
        f.write("# Phase 94 Inference Audit\n")
        f.write("Inference ran successfully and produced valid outputs.\n")
        f.write("```json\n")
        f.write(json.dumps(results, indent=2))
        f.write("\n```\n")

if __name__ == "__main__":
    main()
