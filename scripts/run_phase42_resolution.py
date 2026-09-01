import os
import sys
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.config import TrainConfig
from scripts.run_phase42_peak_recovery import PeakRecoveryMLP, load_split, load_samples, get_building_mask, create_coarse_dem, extract_building_features

MANIFEST_PATH = "runs/dfc2023_multicity_prep/split_manifest.csv"
DATA_DIR = Path("data/dfc2023_multicity")
OUT_DIR = Path("runs/phase42_augmentation")

def degrade_sample(s_base, factor):
    if factor == 1:
        return s_base
    s = s_base.copy()
    gt = s["gt"]
    h, w = gt.shape[:2]
    sh, sw = h // factor, w // factor
    coarse = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_AREA)
    s["gt"] = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
    return s

def process_samples_degraded(samples, estimator, factor, name):
    features_list = []
    print(f"Processing {name} ({len(samples)}) with Factor {factor}x...")
    for s_base in samples:
        s = degrade_sample(s_base, factor)
        gt = s["gt"]
        if "mask_bldg" not in s:
            s["mask_bldg"] = get_building_mask(s, estimator)
        mask_bldg = s["mask_bldg"]
        
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
    print("================ PHASE 42: RESOLUTION DEGRADATION ABLATION ================")
    feature_cols = [
        "dem_mean", "dem_median", "dem_p95", "dem_range", "dem_std",
        "d_mean", "d_median", "d_p90", "d_p95", "d_p99", "d_std", "d_range",
        "area", "w_box", "h_box", "aspect_ratio", "perimeter", "compactness"
    ]
    
    train_ids = load_split(MANIFEST_PATH, 'train')
    val_ids = load_split(MANIFEST_PATH, 'val')
    test_ids = load_split(MANIFEST_PATH, 'test')
    
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
    
    # We always evaluate on 1x for Val/Test
    df_val = process_samples_degraded(val_samples, estimator, 1, "val")
    df_test = process_samples_degraded(test_samples, estimator, 1, "test")
    
    results = []
    
    for factor in [1, 2, 4, 8]:
        print(f"\\n--- Running DEGRADATION FACTOR {factor}x ---")
        df_train = process_samples_degraded(train_samples, estimator, factor, "train")
        
        mu_train = df_train[feature_cols].values.mean(axis=0)
        sigma_train = df_train[feature_cols].values.std(axis=0)
        
        X_tr = (df_train[feature_cols].values - mu_train) / (sigma_train + 1e-6)
        X_val = (df_val[feature_cols].values - mu_train) / (sigma_train + 1e-6)
        X_test = (df_test[feature_cols].values - mu_train) / (sigma_train + 1e-6)
        
        y_tr = df_train["delta_h"].values
        w_tr = np.ones_like(y_tr)
        w_tr[(df_train["true_p95"] >= 10) & (df_train["true_p95"] < 20)] = 1.5
        w_tr[(df_train["true_p95"] >= 20) & (df_train["true_p95"] < 30)] = 2.0
        w_tr[(df_train["true_p95"] >= 30) & (df_train["true_p95"] < 40)] = 2.5
        w_tr[df_train["true_p95"] >= 40] = 3.0
        
        torch.manual_seed(42)
        model = PeakRecoveryMLP()
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        
        inputs_tr = torch.from_numpy(X_tr).float()
        targets_tr = torch.from_numpy(y_tr).float()
        weights_tr = torch.from_numpy(w_tr).float()
        inputs_val = torch.from_numpy(X_val).float()
        
        best_val_mae = float("inf")
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
                
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            p_test = model(torch.from_numpy(X_test).float()).numpy()
        pred_recon_test = df_test["dem_mean"].values + p_test
        true_test = df_test["true_p95"].values
        
        test_mae = np.mean(np.abs(pred_recon_test - true_test))
        sky_mask = true_test >= 40.0
        sky_mae = np.mean(np.abs(pred_recon_test[sky_mask] - true_test[sky_mask])) if sky_mask.sum() > 0 else 0
        
        results.append({
            "Factor": factor, "Val_MAE": best_val_mae, "Test_MAE": test_mae, "Skyscraper_MAE": sky_mae
        })
        print(f"  Factor {factor}x | Val MAE: {best_val_mae:.4f} | Test MAE: {test_mae:.4f} | Skyscraper MAE: {sky_mae:.4f}")
        
    pd.DataFrame(results).to_csv(OUT_DIR / "resolution_ablation.csv", index=False)
    print("Done. Saved resolution results.")

if __name__ == "__main__":
    main()
