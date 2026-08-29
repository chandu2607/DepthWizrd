import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from transformers import AutoModelForDepthEstimation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.metrics.height_metrics import compute_metrics, compute_class_metrics, aggregate_scene_metrics, compute_binned_metrics, aggregate_binned

DATA_DIR = Path("data/dfc2023_multicity")
OUT_DIR = Path("runs/phase14_depth_decoder_adapt")

def load_split(manifest_path, split_type):
    df = pd.read_csv(manifest_path)
    if split_type == 'train':
        df = df[df['split'] == 'train']
    else:
        df = df[df['split'] == split_type]
    return df['tile_id'].tolist()

def load_data_in_memory(tile_ids):
    samples = []
    for tid in tile_ids:
        rgb_path = DATA_DIR / "rgb" / tid
        dsm_path = DATA_DIR / "dsm" / tid
        rgb = cv2.imread(str(rgb_path))
        if rgb is None: continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(dsm_path), cv2.IMREAD_UNCHANGED)
        if gt is None: continue
        gt = gt.astype(np.float32)
        samples.append({"id": tid, "rgb": rgb, "gt": gt, "nodata": -999.0})
    return samples

def prep_rgb(rgb_np, target_size=518):
    rgb = cv2.resize(rgb_np, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    rgb = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    return rgb.transpose(2, 0, 1)

@torch.no_grad()
def predict_full(model, rgb_np, original_shape, device):
    model.eval()
    x = prep_rgb(rgb_np)
    x = torch.from_numpy(x).unsqueeze(0).float().to(device)
    out = model(x)
    pred = out.predicted_depth.unsqueeze(1)
    
    pred = F.interpolate(pred, size=original_shape, mode='bilinear', align_corners=False)
    pred = pred.squeeze().cpu().numpy()
    pred = np.expm1(pred)
    pred = np.maximum(pred, 0.0)
    return pred

def main():
    manifest = Path("runs/dfc2023_multicity_prep/split_manifest.csv")
    test_tids = load_split(manifest, "test")
    test_samples = load_data_in_memory(test_tids)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    all_results = []
    bin_edges = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, np.inf]
    
    for seed in [0, 1]:
        print(f"Evaluating Seed {seed}...")
        seed_dir = OUT_DIR / f"seed_{seed}"
        ckpt_path = seed_dir / "best_model.pt"
        if not ckpt_path.exists():
            print(f"Skipping Seed {seed} (not found)")
            continue
            
        model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(device)
        model.load_state_dict(torch.load(ckpt_path))
        
        per_scene = []
        binned_scenes = []
        
        for s in test_samples:
            pred = predict_full(model, s['rgb'], s['gt'].shape[:2], device)
            # overall metrics
            metrics = compute_metrics(pred, s['gt'], nodata=-999.0).as_dict()
            # binned
            binned = compute_binned_metrics(pred, s['gt'], edges=bin_edges, nodata=-999.0)
            
            per_scene.append({"metrics": metrics})
            binned_scenes.append(binned)
            
            if "NewYork" in s["id"] and s["gt"].max() > 30:
                valid = (np.isfinite(s["gt"])) & (s["gt"] != -999.0)
                err = np.zeros_like(pred)
                err[valid] = pred[valid] - s["gt"][valid]
                err[~valid] = 0.0
                
                # We could render figures here but let's just do it in script if needed
        
        agg_metrics = aggregate_scene_metrics([p["metrics"] for p in per_scene])
        agg_binned = aggregate_binned(binned_scenes)
        
        res_dict = {
            "seed": seed,
            "agg_metrics": agg_metrics,
            "binned": agg_binned
        }
        all_results.append(res_dict)
        
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Evaluation complete.")

if __name__ == "__main__":
    main()
