import os
import sys
import json
import time
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.config import TrainConfig
from scripts.phase42_augment import augment_sample

MANIFEST_PATH = "runs/dfc2023_multicity_prep/split_manifest.csv"
DATA_DIR = Path("data/dfc2023_multicity")
OUT_DIR = Path("runs/phase42_augmentation")

class PeakRecoveryMLP(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.net(x).squeeze(-1)

def load_split(manifest_path, split_type):
    df = pd.read_csv(manifest_path)
    return df[df['split'] == split_type]['tile_id'].tolist()

def load_samples(tile_ids, max_samples=None):
    from depthwizard.depth.depth_anything import DepthAnythingV2
    from depthwizard.config import DepthConfig
    dcfg = DepthConfig(cache_dir=str(DATA_DIR / "depth_cache"))
    depth_model = DepthAnythingV2(dcfg.model_id, dcfg.input_size, dcfg.cache_dir, True)

    samples = []
    tids_to_load = tile_ids[:max_samples] if max_samples is not None else tile_ids
    for tid in tids_to_load:
        rgb_path = DATA_DIR / "rgb" / tid
        dsm_path = DATA_DIR / "dsm" / tid
        
        rgb = cv2.imread(str(rgb_path))
        if rgb is None: continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        gt = cv2.imread(str(dsm_path), cv2.IMREAD_UNCHANGED)
        if gt is None: continue
        gt = gt.astype(np.float32)
        
        depth = depth_model.infer(rgb, tid, target_hw=rgb.shape[:2])
        city = next((c for c in ["Barcelona", "Berlin", "Brasilia", "Copenhagen", "NewDelhi", "NewYork", "Portsmouth", "Rio", "SanDiego", "SaoLuis", "Sydney"] if c in tid), "Unknown")
        
        samples.append({"id": tid, "city": city, "rgb": rgb, "gt": gt, "depth": depth, "nodata": -999.0})
    return samples

def create_coarse_dem(gt, downsample_factor=30, nodata=-999.0):
    h, w = gt.shape
    valid = (gt != nodata) & np.isfinite(gt)
    th, tw = max(1, h // downsample_factor), max(1, w // downsample_factor)
    coarse = np.zeros((th, tw), dtype=np.float32)
    for r in range(th):
        for c in range(tw):
            r_start, r_end = r * downsample_factor, min((r + 1) * downsample_factor, h)
            c_start, c_end = c * downsample_factor, min((c + 1) * downsample_factor, w)
            block = gt[r_start:r_end, c_start:c_end]
            block_valid = valid[r_start:r_end, c_start:c_end]
            if block_valid.sum() > 0:
                coarse[r, c] = np.mean(block[block_valid])
    return coarse

def extract_building_features(s, b_mask, dem_up, d_rel):
    area = float(b_mask.sum())
    if area < 10: return None
        
    dem_b = dem_up[b_mask]
    dem_mean = float(np.mean(dem_b))
    dem_median = float(np.median(dem_b))
    dem_p95 = float(np.percentile(dem_b, 95))
    dem_range = float(np.max(dem_b) - np.min(dem_b))
    dem_std = float(np.std(dem_b))
    
    d_b = d_rel[b_mask]
    d_mean = float(np.mean(d_b))
    d_median = float(np.median(d_b))
    d_p90 = float(np.percentile(d_b, 90))
    d_p95 = float(np.percentile(d_b, 95))
    d_p99 = float(np.percentile(d_b, 99))
    d_std = float(np.std(d_b))
    d_range = float(np.max(d_b) - np.min(d_b))
    
    ys, xs = np.where(b_mask)
    w_box = float(xs.max() - xs.min() + 1)
    h_box = float(ys.max() - ys.min() + 1)
    aspect_ratio = w_box / (h_box + 1e-6)
    
    contours, _ = cv2.findContours(b_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(cv2.arcLength(contours[0], True)) if contours else 0.0
    compactness = (perimeter ** 2) / (4.0 * np.pi * area + 1e-6)
    
    gt_b = s["gt"][b_mask]
    true_p95 = float(np.percentile(gt_b, 95))
    
    delta_h = true_p95 - dem_mean
    
    return {
        "dem_mean": dem_mean, "dem_median": dem_median, "dem_p95": dem_p95, "dem_range": dem_range, "dem_std": dem_std,
        "d_mean": d_mean, "d_median": d_median, "d_p90": d_p90, "d_p95": d_p95, "d_p99": d_p99, "d_std": d_std, "d_range": d_range,
        "area": area, "w_box": w_box, "h_box": h_box, "aspect_ratio": aspect_ratio, "perimeter": perimeter, "compactness": compactness,
        "true_p95": true_p95, "delta_h": delta_h
    }

def get_building_mask(s, estimator):
    res = estimator.cfg.train_res
    x = estimator._prep_x(s, res)
    xt = torch.from_numpy(x[None]).float().to(estimator.device)
    depth = np.asarray(s["depth"], dtype=np.float32)
    depth_r = cv2.resize(depth, (res, res), interpolation=cv2.INTER_LINEAR)
    raw_d = torch.from_numpy(depth_r[None]).float().to(estimator.device)
    with torch.no_grad():
        mask_logits, _, _, _, _ = estimator.model(xt, raw_d, device=estimator.device)
    probs = torch.sigmoid(mask_logits).squeeze(0).cpu().numpy()
    h, w = s["gt"].shape[:2]
    return cv2.resize((probs > 0.5).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0.5

def process_samples(samples, estimator, config_mode, rng, name):
    features_list = []
    print(f"Processing {name} ({len(samples)}) with Mode {config_mode}...")
    for i, s_base in enumerate(samples):
        if config_mode == 'A':
            s = s_base
        else:
            # We augment mask as well! So let's run the U-Net on original first to get mask, then augment all
            s = s_base.copy()
            if "mask_bldg" not in s:
                s["mask_bldg"] = get_building_mask(s_base, estimator)
            s = augment_sample(s, config_mode, rng)
            
        gt = s["gt"]
        mask_bldg = s["mask_bldg"] if "mask_bldg" in s else get_building_mask(s, estimator)
        
        coarse = create_coarse_dem(gt)
        dem_up = cv2.resize(coarse, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        num_labels, labels_im = cv2.connectedComponents(mask_bldg.astype(np.uint8))
        for label in range(1, num_labels):
            b_mask = labels_im == label
            feat = extract_building_features(s, b_mask, dem_up, s["depth"])
            if feat:
                features_list.append(feat)
    return pd.DataFrame(features_list)

def main():
    print("================ PHASE 42: PEAK-RECOVERY AUGMENTATION ABLATION ================")
    feature_cols = [
        "dem_mean", "dem_median", "dem_p95", "dem_range", "dem_std",
        "d_mean", "d_median", "d_p90", "d_p95", "d_p99", "d_std", "d_range",
        "area", "w_box", "h_box", "aspect_ratio", "perimeter", "compactness"
    ]
    
    train_ids = load_split(MANIFEST_PATH, 'train')
    val_ids = load_split(MANIFEST_PATH, 'val')
    test_ids = load_split(MANIFEST_PATH, 'test')
    
    print("Loading samples...")
    train_samples = load_samples(train_ids, max_samples=128)
    val_samples = load_samples(val_ids)
    test_samples = load_samples(test_ids)
    
    from depthwizard.models.building_conditioned_net import BuildingConditionedEstimator
    tcfg = TrainConfig(arch="unet3", target_transform="none", epochs=1, batch_size=8, lr=1e-3, amp=True)
    estimator = BuildingConditionedEstimator(tcfg, nodata=-999.0, seed=0)
    checkpoint_path = Path("runs/phase24_moe/seed_0/model.pt")
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=estimator.device)
        estimator.model.load_state_dict(state)
        estimator.model.eval()
        print("Loaded Phase 24 model.")
    
    rng = np.random.default_rng(42)
    # Validation and Test are ALWAYS Mode A (No augmentation)
    df_val = process_samples(val_samples, estimator, 'A', rng, "val")
    df_test = process_samples(test_samples, estimator, 'A', rng, "test")
    
    # We must normalize using un-augmented train features for consistency
    df_train_base = process_samples(train_samples, estimator, 'A', rng, "train_base")
    mu_train = df_train_base[feature_cols].values.mean(axis=0)
    sigma_train = df_train_base[feature_cols].values.std(axis=0)
    
    X_val = (df_val[feature_cols].values - mu_train) / (sigma_train + 1e-6)
    X_test = (df_test[feature_cols].values - mu_train) / (sigma_train + 1e-6)
    
    results = []
    
    # Ablations
    for config_mode in ['A', 'B', 'C', 'D']:
        print(f"\\n--- Running CONFIGURATION {config_mode} ---")
        best_overall_mae = float('inf')
        test_metrics_for_config = None
        
        for seed in [0, 1]:
            torch.manual_seed(seed)
            model = PeakRecoveryMLP()
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
            best_val_mae = float("inf")
            
            df_train_aug = process_samples(train_samples, estimator, config_mode, np.random.default_rng(seed), f"train_{config_mode}_seed_{seed}")
            X_tr = (df_train_aug[feature_cols].values - mu_train) / (sigma_train + 1e-6)
            y_tr = df_train_aug["delta_h"].values
            w_tr = np.ones_like(y_tr)
            w_tr[(df_train_aug["true_p95"] >= 10) & (df_train_aug["true_p95"] < 20)] = 1.5
            w_tr[(df_train_aug["true_p95"] >= 20) & (df_train_aug["true_p95"] < 30)] = 2.0
            w_tr[(df_train_aug["true_p95"] >= 30) & (df_train_aug["true_p95"] < 40)] = 2.5
            w_tr[df_train_aug["true_p95"] >= 40] = 3.0
            
            inputs_tr = torch.from_numpy(X_tr).float()
            targets_tr = torch.from_numpy(y_tr).float()
            weights_tr = torch.from_numpy(w_tr).float()
            inputs_val = torch.from_numpy(X_val).float()
            
            for epoch in range(120):
                model.train()
                optimizer.zero_grad()
                p = model(inputs_tr)
                loss = (F.huber_loss(p, targets_tr, reduction="none") * weights_tr).mean()
                loss.backward()
                optimizer.step()
                
                model.eval()
                with torch.no_grad():
                    p_val = model(inputs_val).numpy()
                val_mae = np.mean(np.abs(df_val["dem_mean"].values + p_val - df_val["true_p95"].values))
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_state = model.state_dict().copy()
            
            # Eval on TEST zero-shot
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                p_test = model(torch.from_numpy(X_test).float()).numpy()
            pred_recon_test = df_test["dem_mean"].values + p_test
            true_test = df_test["true_p95"].values
            
            test_mae = np.mean(np.abs(pred_recon_test - true_test))
            test_bias = np.mean(pred_recon_test - true_test)
            sky_mask = true_test >= 40.0
            sky_mae = np.mean(np.abs(pred_recon_test[sky_mask] - true_test[sky_mask])) if sky_mask.sum() > 0 else 0
            
            results.append({
                "Config": config_mode,
                "Seed": seed,
                "Val_MAE": best_val_mae,
                "Test_MAE": test_mae,
                "Test_Bias": test_bias,
                "Skyscraper_MAE": sky_mae
            })
            print(f"  Seed {seed} | Val MAE: {best_val_mae:.4f} | Test MAE: {test_mae:.4f} | Skyscraper MAE: {sky_mae:.4f}")
            
    pd.DataFrame(results).to_csv(OUT_DIR / "augmentation_ablation.csv", index=False)
    print("Done. Saved results.")

if __name__ == "__main__":
    main()
