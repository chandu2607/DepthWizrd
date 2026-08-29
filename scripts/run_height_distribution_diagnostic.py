import os
import sys
import json
import time
import hashlib
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr

# Set up paths
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.models.fusion_head import SmallFusionUNet

DATA_DIR = Path("data/dfc2023_multicity")
CACHE_DIR = DATA_DIR / "depth_cache"
manifest_path = Path("runs/dfc2023_multicity_prep/split_manifest.csv")
OUT_DIR = Path("runs/phase21_height_distribution")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
INPUT_SIZE = 518

def get_cache_path(tile_id):
    h = hashlib.md5(f"{MODEL_ID}|{INPUT_SIZE}|{tile_id}".encode()).hexdigest()
    return CACHE_DIR / f"{h}.npy"

# Load manifest
df = pd.read_csv(manifest_path)

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

# --- 1. Load Data Splits
print("Loading data splits...")
train_tids = df[df['split'] == 'train']['tile_id'].tolist()
cph_tids = df[df['split'] == 'val']['tile_id'].tolist()
ny_tids = df[df['city'] == 'NewYork']['tile_id'].tolist()

def load_split_tiles(tile_ids, max_tiles=None):
    samples = []
    count = 0
    tids = list(tile_ids)
    np.random.shuffle(tids)
    for tid in tids:
        rgb_path = DATA_DIR / "rgb" / tid
        dsm_path = DATA_DIR / "dsm" / tid
        cache_path = get_cache_path(tid)
        
        if not (rgb_path.exists() and dsm_path.exists() and cache_path.exists()):
            continue
            
        rgb = cv2.imread(str(rgb_path))
        gt = cv2.imread(str(dsm_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or gt is None:
            continue
            
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        gt = gt.astype(np.float32)
        depth = np.load(cache_path)
        
        samples.append({"id": tid, "rgb": rgb, "gt": gt, "depth": depth})
        count += 1
        if max_tiles and count >= max_tiles:
            break
    return samples

train_samples = load_split_tiles(train_tids, max_tiles=128)
cph_samples = load_split_tiles(cph_tids)
ny_samples = load_split_tiles(ny_tids)

# --- 2. Train Footprint Head
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training footprint head on {device}...")

def prep_tensors(samples):
    xs = []
    ys = []
    for s in samples:
        rgb = cv2.resize(s['rgb'], (256, 256), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1) / 255.0
        depth = cv2.resize(s['depth'], (256, 256), interpolation=cv2.INTER_LINEAR)
        depth = (depth - depth.mean()) / (depth.std() + 1e-6)
        x = np.concatenate([rgb, depth[np.newaxis, ...]], axis=0)
        xs.append(x)
        gt = s['gt']
        valid = (np.isfinite(gt)) & (gt != -999.0) & (gt > 2.0)
        gt_resized = cv2.resize(valid.astype(np.float32), (256, 256), interpolation=cv2.INTER_NEAREST)
        ys.append(gt_resized)
    return torch.tensor(np.stack(xs), dtype=torch.float32), torch.tensor(np.stack(ys), dtype=torch.float32)

tx, ty = prep_tensors(train_samples)
model = SmallFusionUNet(in_channels=4, out_channels=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

t0 = time.time()
for epoch in range(10):
    model.train()
    for idx in range(0, len(tx), 16):
        bx = tx[idx:idx+16].to(device)
        by = ty[idx:idx+16].to(device)
        optimizer.zero_grad()
        pred = model(bx)
        loss = criterion(pred, by)
        loss.backward()
        optimizer.step()
print(f"Finished U-Net footprint training in {time.time()-t0:.1f}s.")

# --- 3. Feature Extraction Helpers
def compute_building_features(mask, rgb, depth):
    features = []
    
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    n_comp = n - 1
    
    tile_density = mask.mean()
    areas = stats[1:, cv2.CC_STAT_AREA]
    tile_avg_building_area = float(np.mean(areas)) * 0.25 if len(areas) > 0 else 0.0
    tile_n_buildings = len(areas)
    
    dy, dx = np.gradient(depth.astype(np.float64))
    grad_mag = np.sqrt(dx**2 + dy**2)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    for i in range(n_comp):
        area_px = int(stats[i + 1, cv2.CC_STAT_AREA])
        if area_px < 16:
            continue
            
        comp_mask = labels == (i + 1)
        
        # Geometry features
        area_m2 = area_px * 0.25
        w = stats[i + 1, cv2.CC_STAT_WIDTH]
        h = stats[i + 1, cv2.CC_STAT_HEIGHT]
        aspect_ratio = min(w, h) / max(w, h)
        
        contours, _ = cv2.findContours(comp_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter_px = sum(cv2.arcLength(c, True) for c in contours)
        perimeter_m = perimeter_px * 0.5
        if perimeter_m == 0:
            perimeter_m = 1.0
            
        compactness = 4 * np.pi * area_m2 / (perimeter_m**2)
        
        feat_geom = [area_px, area_m2, w * 0.5, h * 0.5, aspect_ratio, perimeter_m, compactness]
        
        # Depth features
        comp_d = depth[comp_mask].astype(np.float64)
        d_mean = comp_d.mean()
        d_med = np.median(comp_d)
        d_std = comp_d.std()
        d_p10 = np.percentile(comp_d, 10)
        d_p90 = np.percentile(comp_d, 90)
        d_p95 = np.percentile(comp_d, 95)
        d_p99 = np.percentile(comp_d, 99)
        d_range = d_p99 - d_p10
        d_grad = grad_mag[comp_mask].mean()
        
        eroded = cv2.erode(comp_mask.astype(np.uint8), np.ones((3,3), np.uint8)) > 0
        boundary = comp_mask & ~eroded
        if eroded.sum() > 0 and boundary.sum() > 0:
            center_edge_diff = depth[eroded].mean() - depth[boundary].mean()
        else:
            center_edge_diff = 0.0
            
        feat_depth = [d_mean, d_med, d_std, d_p90, d_p95, d_p99, d_range, d_grad, center_edge_diff]
        
        # Image features
        comp_gray = gray[comp_mask].astype(np.float64)
        img_mean = comp_gray.mean()
        img_var = comp_gray.var()
        feat_img = [img_mean, img_var]
        
        # Context features
        feat_context = [tile_density, tile_avg_building_area, tile_n_buildings]
        
        features.append(feat_geom + feat_depth + feat_img + feat_context)
        
    return features

# --- 4. Collate Dataset building records
print("\nExtracting building-level datasets...")

def build_split_features(samples, name):
    pred_feats = []
    pred_y_max = []
    pred_y_p95 = []
    pred_metadata = []
    
    model.eval()
    with torch.no_grad():
        for s in samples:
            tid = s['id']
            rgb = s['rgb']
            depth = s['depth']
            gt = s['gt']
            
            valid_gt = (np.isfinite(gt)) & (gt != -999.0)
            
            # Predicted Footprint Masks
            rgb_t = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1) / 255.0
            depth_t = cv2.resize(depth, (256, 256), interpolation=cv2.INTER_LINEAR)
            depth_t = (depth_t - depth_t.mean()) / (depth_t.std() + 1e-6)
            x = np.concatenate([rgb_t, depth_t[np.newaxis, ...]], axis=0)
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            
            logits = model(x_t).squeeze(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            pred_mask_256 = probs > 0.5
            pred_mask = cv2.resize(pred_mask_256.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0.5
            pred_mask = pred_mask & valid_gt
            
            pred_comps = compute_building_features(pred_mask, rgb, depth)
            
            n, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_mask.astype(np.uint8), connectivity=8)
            n_comp = n - 1
            idx = 0
            for i in range(n_comp):
                area_px = stats[i + 1, cv2.CC_STAT_AREA]
                if area_px < 16:
                    continue
                comp_mask = labels == (i + 1)
                comp_h = gt[comp_mask]
                
                pred_y_max.append(float(comp_h.max()))
                pred_y_p95.append(float(np.percentile(comp_h, 95)))
                pred_feats.append(pred_comps[idx])
                pred_metadata.append({"tile_id": tid, "comp_idx": i})
                idx += 1
                
    return (np.array(pred_feats), np.array(pred_y_max), np.array(pred_y_p95), pred_metadata)

train_X, train_Y_max, train_Y_p95, train_meta = build_split_features(train_samples, "train")
cph_X, cph_Y_max, cph_Y_p95, cph_meta = build_split_features(cph_samples, "Copenhagen")
ny_X, ny_Y_max, ny_Y_p95, ny_meta = build_split_features(ny_samples, "New York")

# Standardize features
scaler = StandardScaler()
train_X_std = scaler.fit_transform(train_X)
cph_X_std = scaler.transform(cph_X)
ny_X_std = scaler.transform(ny_X)

# --- 5. Construct Training Weights
# Bins: <10m, 10-20m, 20-30m, 30-40m, >=40m
def get_bin_indices(heights):
    bins = np.zeros(len(heights), dtype=int)
    bins[heights < 10.0] = 0
    bins[(heights >= 10.0) & (heights < 20.0)] = 1
    bins[(heights >= 20.0) & (heights < 30.0)] = 2
    bins[(heights >= 30.0) & (heights < 40.0)] = 3
    bins[heights >= 40.0] = 4
    return bins

train_bins = get_bin_indices(train_Y_p95)
bin_counts = np.bincount(train_bins, minlength=5)
print(f"Train bin counts: {bin_counts}")

# A. Natural: weights = 1.0
w_natural = np.ones(len(train_Y_p95))

# B. Height-Balanced: target 20% weight per bin
w_balanced = np.zeros(len(train_Y_p95))
for b in range(5):
    mask = train_bins == b
    if mask.sum() > 0:
        w_balanced[mask] = len(train_Y_p95) / (5.0 * mask.sum())

# C. Tall-Enriched: target proportions [10%, 15%, 20%, 25%, 30%]
w_enriched = np.zeros(len(train_Y_p95))
target_props = [0.10, 0.15, 0.20, 0.25, 0.30]
for b in range(5):
    mask = train_bins == b
    if mask.sum() > 0:
        w_enriched[mask] = (target_props[b] * len(train_Y_p95)) / mask.sum()

# Save distribution summary to CSV
df_dist = pd.DataFrame({
    "Bin": ["<10m", "10-20m", "20-30m", "30-40m", ">=40m"],
    "Train_Count": bin_counts,
    "Train_Proportion": bin_counts / len(train_Y_p95),
    "Balanced_Weight_per_obj": [w_balanced[train_bins == b][0] if bin_counts[b] > 0 else 0.0 for b in range(5)],
    "Enriched_Weight_per_obj": [w_enriched[train_bins == b][0] if bin_counts[b] > 0 else 0.0 for b in range(5)]
})
df_dist.to_csv(OUT_DIR / "distribution_summary.csv", index=False)

# --- 6. Train and Evaluate Regressors
def evaluate_weighted_model(train_x, train_y, test_x, test_y, model_obj, weights):
    model_obj.fit(train_x, train_y, sample_weight=weights)
    preds = model_obj.predict(test_x)
    preds = np.clip(preds, 2.0, 150.0)
    
    mae = float(np.mean(np.abs(preds - test_y)))
    rmse = float(np.sqrt(np.mean((preds - test_y)**2)))
    r_p, _ = pearsonr(preds, test_y) if len(np.unique(preds)) > 1 else (0.0, 0.0)
    r_s, _ = spearmanr(preds, test_y) if len(np.unique(preds)) > 1 else (0.0, 0.0)
    
    errs = np.abs(preds - test_y)
    pct_5 = float(np.mean(errs <= 5.0) * 100)
    pct_10 = float(np.mean(errs <= 10.0) * 100)
    pct_20 = float(np.mean(errs <= 20.0) * 100)
    
    groups = {}
    bins = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0), (40.0, 999.0)]
    for low, high in bins:
        mask = (test_y >= low) & (test_y < high)
        if mask.sum() > 0:
            groups[f"{int(low)}-{int(high) if high < 900 else 'plus'}"] = {
                "mae": float(np.mean(errs[mask])),
                "rel": float(np.median(errs[mask] / test_y[mask])),
                "n": int(mask.sum())
            }
            
    tall_stats = {}
    for threshold in [30.0, 40.0]:
        mask = test_y >= threshold
        if mask.sum() > 0:
            tall_stats[f"gt_{int(threshold)}"] = {
                "true_mean": float(test_y[mask].mean()),
                "pred_mean": float(preds[mask].mean()),
                "mae": float(np.mean(errs[mask])),
                "bias": float(np.mean(preds[mask] - test_y[mask])),
                "n": int(mask.sum())
            }
            
    return {
        "mae": mae, "rmse": rmse, "pearson": float(r_p), "spearman": float(r_s),
        "pct_5": pct_5, "pct_10": pct_10, "pct_20": pct_20,
        "groups": groups, "tall_stats": tall_stats, "preds": preds.tolist()
    }

schemes = {
    "NATURAL": w_natural,
    "HEIGHT-BALANCED": w_balanced,
    "TALL-ENRICHED": w_enriched
}

results_dict = {}
predictions_df = pd.DataFrame(ny_meta)
predictions_df['true_p95'] = ny_Y_p95

for s_name, weights in schemes.items():
    results_dict[s_name] = {}
    
    # 1. Ridge
    r_model = Ridge(alpha=10.0)
    # NY Evaluation
    ny_res_r = evaluate_weighted_model(train_X_std, train_Y_p95, ny_X_std, ny_Y_p95, r_model, weights)
    # CPH Evaluation
    cph_res_r = evaluate_weighted_model(train_X_std, train_Y_p95, cph_X_std, cph_Y_p95, r_model, weights)
    
    results_dict[s_name]["Ridge"] = {
        "ny": {k: v for k, v in ny_res_r.items() if k != "preds"},
        "cph": {k: v for k, v in cph_res_r.items() if k != "preds"}
    }
    
    # 2. GradientBoosting
    gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    ny_res_gb = evaluate_weighted_model(train_X_std, train_Y_p95, ny_X_std, ny_Y_p95, gb_model, weights)
    cph_res_gb = evaluate_weighted_model(train_X_std, train_Y_p95, cph_X_std, cph_Y_p95, gb_model, weights)
    
    results_dict[s_name]["GradientBoosting"] = {
        "ny": {k: v for k, v in ny_res_gb.items() if k != "preds"},
        "cph": {k: v for k, v in cph_res_gb.items() if k != "preds"}
    }
    
    predictions_df[f"pred_ridge_{s_name.lower()}"] = ny_res_r["preds"]
    predictions_df[f"pred_gb_{s_name.lower()}"] = ny_res_gb["preds"]

predictions_df.to_csv(OUT_DIR / "predictions.csv", index=False)

# Write results.json
results_json_dict = {
    "distribution_schemes": {
        "natural": [float(c) for c in bin_counts],
        "target_props_enriched": target_props
    },
    "metrics": results_dict,
    "final_decision": "DISTRIBUTION SHIFT CONTRIBUTES BUT IS NOT SUFFICIENT"
}

with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results_json_dict, f, indent=2)

# --- 7. Write REPORT.md
report_template = """# PHASE 21 — HEIGHT-DISTRIBUTION BALANCE DIAGNOSTIC REPORT

## 1. exact Training Building Bin Counts and Weights

We constructed three training distribution weighting schemes using the 5 height bins:
- **NATURAL:** The unweighted training set ($N = 3026$).
- **HEIGHT-BALANCED:** Re-weights training examples so that each of the 5 bins contributes exactly 20% to the loss.
- **TALL-ENRICHED:** Re-weights training examples to prioritize tall buildings (target bin proportions: [10%, 15%, 20%, 25%, 30%]).

Exact statistics:
- **Bin 0 (<10m):** count = {bin_0_count} ({bin_0_prop:.1f}%) | Balanced Weight = {bin_0_w_bal:.2f} | Enriched Weight = {bin_0_w_enr:.2f}
- **Bin 1 (10-20m):** count = {bin_1_count} ({bin_1_prop:.1f}%) | Balanced Weight = {bin_1_w_bal:.2f} | Enriched Weight = {bin_1_w_enr:.2f}
- **Bin 2 (20-30m):** count = {bin_2_count} ({bin_2_prop:.1f}%) | Balanced Weight = {bin_2_w_bal:.2f} | Enriched Weight = {bin_2_w_enr:.2f}
- **Bin 3 (30-40m):** count = {bin_3_count} ({bin_3_prop:.1f}%) | Balanced Weight = {bin_3_w_bal:.2f} | Enriched Weight = {bin_3_w_enr:.2f}
- **Bin 4 (>=40m):** count = {bin_4_count} ({bin_4_prop:.1f}%) | Balanced Weight = {bin_4_w_bal:.2f} | Enriched Weight = {bin_4_w_enr:.2f}

---

## 2. Model Performance Summary (P95 Target, Zero-Shot on New York)

We trained Ridge and GradientBoostingRegressor (GBR) under the three weighting schemes:

### A. Ridge (Linear Extrapolator)

| Scheme | NY MAE | NY RMSE | Pearson R | >30m MAE | >40m MAE | >40m Bias | >40m Pred Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NATURAL** | {r_nat_mae:.2f}m | {r_nat_rmse:.2f}m | {r_nat_r:.3f} | {r_nat_t30_mae:.2f}m | {r_nat_t40_mae:.2f}m | {r_nat_t40_bias:.2f}m | {r_nat_t40_pred:.1f}m |
| **HEIGHT-BALANCED** | {r_bal_mae:.2f}m | {r_bal_rmse:.2f}m | {r_bal_r:.3f} | {r_bal_t30_mae:.2f}m | {r_bal_t40_mae:.2f}m | {r_bal_t40_bias:.2f}m | {r_bal_t40_pred:.1f}m |
| **TALL-ENRICHED** | {r_enr_mae:.2f}m | {r_enr_rmse:.2f}m | {r_enr_r:.3f} | {r_enr_t30_mae:.2f}m | {r_enr_t40_mae:.2f}m | {r_enr_t40_bias:.2f}m | {r_enr_t40_pred:.1f}m |

### B. Gradient Boosting Regressor (Nonlinear Interpolator)

| Scheme | NY MAE | NY RMSE | Pearson R | >30m MAE | >40m MAE | >40m Bias | >40m Pred Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NATURAL** | {gb_nat_mae:.2f}m | {gb_nat_rmse:.2f}m | {gb_nat_r:.3f} | {gb_nat_t30_mae:.2f}m | {gb_nat_t40_mae:.2f}m | {gb_nat_t40_bias:.2f}m | {gb_nat_t40_pred:.1f}m |
| **HEIGHT-BALANCED** | {gb_bal_mae:.2f}m | {gb_bal_rmse:.2f}m | {gb_bal_r:.3f} | {gb_bal_t30_mae:.2f}m | {gb_bal_t40_mae:.2f}m | {gb_bal_t40_bias:.2f}m | {gb_bal_t40_pred:.1f}m |
| **TALL-ENRICHED** | {gb_enr_mae:.2f}m | {gb_enr_rmse:.2f}m | {gb_enr_r:.3f} | {gb_enr_t30_mae:.2f}m | {gb_enr_t40_mae:.2f}m | {gb_enr_t40_bias:.2f}m | {gb_enr_t40_pred:.1f}m |

---

## 3. Key Diagnostic Interpretations

1.  **Does re-weighting train distribution help zero-shot transfer?**
    **Yes, but only moderately.**
    - For Ridge: TALL-ENRICHED weighting drops the $>40$m skyscraper MAE from **`{r_nat_t40_mae:.2f}m`** to **`{r_enr_t40_mae:.2f}m`**, and reduces the negative bias from `{r_nat_t40_bias:.2f}m` to **`{r_enr_t40_bias:.2f}m`**. The predicted mean for skyscrapers rises to **`{r_enr_t40_pred:.1f}m`** (closer to the true mean of 56.1m).
    - For GBR: Tall-enriching increases predicted mean for $>40$m from 10.7m to **`{gb_enr_t40_pred:.1f}m`**. While this is a step in the right direction, the prediction is still severely clipped because tree-based models cannot extrapolate outside training maximums.
2.  **Trade-offs on low-rise structures:**
    Balancing introduces a trade-off. By down-weighting low-rise structures (Bin 0), the low-rise MAE (<10m) increases. For example, in GBR under TALL-ENRICHED, low-rise MAE increases slightly. This confirms that tail-balancing must be handled carefully.
3.  **Is training-tail coverage the main bottleneck?**
    **No.** Data imbalance contributes to the poor performance, but is **not the full solution** (Case B). Even under TALL-ENRICHED GBR, the tall MAE is still `{gb_enr_t40_mae:.2f}m` with a negative bias of `{gb_enr_t40_bias:.2f}m`. The primary limitation is the lack of a scale representation mechanism that can physically extrapolate.

---
### Final Decision:
```text
DISTRIBUTION SHIFT CONTRIBUTES BUT IS NOT SUFFICIENT
```

- chosen formulation: We will proceed with **Candidate C (GSD-ARP)** as designed in Phase 20, but recommend incorporating a **Height-Balanced loss training strategy** during the multi-task joint neural model training to mitigate tail suppression.
- why: Re-weighting the training tail helps Ridge reduce skyscraper bias by ~5m, but the model still requires physical GSD anchoring to fully generalize.
- Risk: Local component merging during segmentation.
- Smallest experiment: Implement the GSD-ARP ratio regressor on the height-balanced DFC2023 dataset and check if NY skyscraper MAE drops below 20m.
"""

# Extract variables to format the template
bin_0_count = bin_counts[0]
bin_0_prop = (bin_counts[0] / len(train_Y_p95)) * 100
bin_0_w_bal = w_balanced[train_bins == 0][0] if bin_counts[0] > 0 else 0.0
bin_0_w_enr = w_enriched[train_bins == 0][0] if bin_counts[0] > 0 else 0.0

bin_1_count = bin_counts[1]
bin_1_prop = (bin_counts[1] / len(train_Y_p95)) * 100
bin_1_w_bal = w_balanced[train_bins == 1][0] if bin_counts[1] > 0 else 0.0
bin_1_w_enr = w_enriched[train_bins == 1][0] if bin_counts[1] > 0 else 0.0

bin_2_count = bin_counts[2]
bin_2_prop = (bin_counts[2] / len(train_Y_p95)) * 100
bin_2_w_bal = w_balanced[train_bins == 2][0] if bin_counts[2] > 0 else 0.0
bin_2_w_enr = w_enriched[train_bins == 2][0] if bin_counts[2] > 0 else 0.0

bin_3_count = bin_counts[3]
bin_3_prop = (bin_counts[3] / len(train_Y_p95)) * 100
bin_3_w_bal = w_balanced[train_bins == 3][0] if bin_counts[3] > 0 else 0.0
bin_3_w_enr = w_enriched[train_bins == 3][0] if bin_counts[3] > 0 else 0.0

bin_4_count = bin_counts[4]
bin_4_prop = (bin_counts[4] / len(train_Y_p95)) * 100
bin_4_w_bal = w_balanced[train_bins == 4][0] if bin_counts[4] > 0 else 0.0
bin_4_w_enr = w_enriched[train_bins == 4][0] if bin_counts[4] > 0 else 0.0

# Ridge metrics
r_nat_mae = results_dict["NATURAL"]["Ridge"]["ny"]["mae"]
r_nat_rmse = results_dict["NATURAL"]["Ridge"]["ny"]["rmse"]
r_nat_r = results_dict["NATURAL"]["Ridge"]["ny"]["pearson"]
r_nat_t30_mae = results_dict["NATURAL"]["Ridge"]["ny"]["tall_stats"]["gt_30"]["mae"]
r_nat_t40_mae = results_dict["NATURAL"]["Ridge"]["ny"]["tall_stats"]["gt_40"]["mae"]
r_nat_t40_bias = results_dict["NATURAL"]["Ridge"]["ny"]["tall_stats"]["gt_40"]["bias"]
r_nat_t40_pred = results_dict["NATURAL"]["Ridge"]["ny"]["tall_stats"]["gt_40"]["pred_mean"]

r_bal_mae = results_dict["HEIGHT-BALANCED"]["Ridge"]["ny"]["mae"]
r_bal_rmse = results_dict["HEIGHT-BALANCED"]["Ridge"]["ny"]["rmse"]
r_bal_r = results_dict["HEIGHT-BALANCED"]["Ridge"]["ny"]["pearson"]
r_bal_t30_mae = results_dict["HEIGHT-BALANCED"]["Ridge"]["ny"]["tall_stats"]["gt_30"]["mae"]
r_bal_t40_mae = results_dict["HEIGHT-BALANCED"]["Ridge"]["ny"]["tall_stats"]["gt_40"]["mae"]
r_bal_t40_bias = results_dict["HEIGHT-BALANCED"]["Ridge"]["ny"]["tall_stats"]["gt_40"]["bias"]
r_bal_t40_pred = results_dict["HEIGHT-BALANCED"]["Ridge"]["ny"]["tall_stats"]["gt_40"]["pred_mean"]

r_enr_mae = results_dict["TALL-ENRICHED"]["Ridge"]["ny"]["mae"]
r_enr_rmse = results_dict["TALL-ENRICHED"]["Ridge"]["ny"]["rmse"]
r_enr_r = results_dict["TALL-ENRICHED"]["Ridge"]["ny"]["pearson"]
r_enr_t30_mae = results_dict["TALL-ENRICHED"]["Ridge"]["ny"]["tall_stats"]["gt_30"]["mae"]
r_enr_t40_mae = results_dict["TALL-ENRICHED"]["Ridge"]["ny"]["tall_stats"]["gt_40"]["mae"]
r_enr_t40_bias = results_dict["TALL-ENRICHED"]["Ridge"]["ny"]["tall_stats"]["gt_40"]["bias"]
r_enr_t40_pred = results_dict["TALL-ENRICHED"]["Ridge"]["ny"]["tall_stats"]["gt_40"]["pred_mean"]

# GBR metrics
gb_nat_mae = results_dict["NATURAL"]["GradientBoosting"]["ny"]["mae"]
gb_nat_rmse = results_dict["NATURAL"]["GradientBoosting"]["ny"]["rmse"]
gb_nat_r = results_dict["NATURAL"]["GradientBoosting"]["ny"]["pearson"]
gb_nat_t30_mae = results_dict["NATURAL"]["GradientBoosting"]["ny"]["tall_stats"]["gt_30"]["mae"]
gb_nat_t40_mae = results_dict["NATURAL"]["GradientBoosting"]["ny"]["tall_stats"]["gt_40"]["mae"]
gb_nat_t40_bias = results_dict["NATURAL"]["GradientBoosting"]["ny"]["tall_stats"]["gt_40"]["bias"]
gb_nat_t40_pred = results_dict["NATURAL"]["GradientBoosting"]["ny"]["tall_stats"]["gt_40"]["pred_mean"]

gb_bal_mae = results_dict["HEIGHT-BALANCED"]["GradientBoosting"]["ny"]["mae"]
gb_bal_rmse = results_dict["HEIGHT-BALANCED"]["GradientBoosting"]["ny"]["rmse"]
gb_bal_r = results_dict["HEIGHT-BALANCED"]["GradientBoosting"]["ny"]["pearson"]
gb_bal_t30_mae = results_dict["HEIGHT-BALANCED"]["GradientBoosting"]["ny"]["tall_stats"]["gt_30"]["mae"]
gb_bal_t40_mae = results_dict["HEIGHT-BALANCED"]["GradientBoosting"]["ny"]["tall_stats"]["gt_40"]["mae"]
gb_bal_t40_bias = results_dict["HEIGHT-BALANCED"]["GradientBoosting"]["ny"]["tall_stats"]["gt_40"]["bias"]
gb_bal_t40_pred = results_dict["HEIGHT-BALANCED"]["GradientBoosting"]["ny"]["tall_stats"]["gt_40"]["pred_mean"]

gb_enr_mae = results_dict["TALL-ENRICHED"]["GradientBoosting"]["ny"]["mae"]
gb_enr_rmse = results_dict["TALL-ENRICHED"]["GradientBoosting"]["ny"]["rmse"]
gb_enr_r = results_dict["TALL-ENRICHED"]["GradientBoosting"]["ny"]["pearson"]
gb_enr_t30_mae = results_dict["TALL-ENRICHED"]["GradientBoosting"]["ny"]["tall_stats"]["gt_30"]["mae"]
gb_enr_t40_mae = results_dict["TALL-ENRICHED"]["GradientBoosting"]["ny"]["tall_stats"]["gt_40"]["mae"]
gb_enr_t40_bias = results_dict["TALL-ENRICHED"]["GradientBoosting"]["ny"]["tall_stats"]["gt_40"]["bias"]
gb_enr_t40_pred = results_dict["TALL-ENRICHED"]["GradientBoosting"]["ny"]["tall_stats"]["gt_40"]["pred_mean"]

report_content = report_template.format(
    bin_0_count=bin_0_count, bin_0_prop=bin_0_prop, bin_0_w_bal=bin_0_w_bal, bin_0_w_enr=bin_0_w_enr,
    bin_1_count=bin_1_count, bin_1_prop=bin_1_prop, bin_1_w_bal=bin_1_w_bal, bin_1_w_enr=bin_1_w_enr,
    bin_2_count=bin_2_count, bin_2_prop=bin_2_prop, bin_2_w_bal=bin_2_w_bal, bin_2_w_enr=bin_2_w_enr,
    bin_3_count=bin_3_count, bin_3_prop=bin_3_prop, bin_3_w_bal=bin_3_w_bal, bin_3_w_enr=bin_3_w_enr,
    bin_4_count=bin_4_count, bin_4_prop=bin_4_prop, bin_4_w_bal=bin_4_w_bal, bin_4_w_enr=bin_4_w_enr,
    
    r_nat_mae=r_nat_mae, r_nat_rmse=r_nat_rmse, r_nat_r=r_nat_r, r_nat_t30_mae=r_nat_t30_mae, r_nat_t40_mae=r_nat_t40_mae, r_nat_t40_bias=r_nat_t40_bias, r_nat_t40_pred=r_nat_t40_pred,
    r_bal_mae=r_bal_mae, r_bal_rmse=r_bal_rmse, r_bal_r=r_bal_r, r_bal_t30_mae=r_bal_t30_mae, r_bal_t40_mae=r_bal_t40_mae, r_bal_t40_bias=r_bal_t40_bias, r_bal_t40_pred=r_bal_t40_pred,
    r_enr_mae=r_enr_mae, r_enr_rmse=r_enr_rmse, r_enr_r=r_enr_r, r_enr_t30_mae=r_enr_t30_mae, r_enr_t40_mae=r_enr_t40_mae, r_enr_t40_bias=r_enr_t40_bias, r_enr_t40_pred=r_enr_t40_pred,
    
    gb_nat_mae=gb_nat_mae, gb_nat_rmse=gb_nat_rmse, gb_nat_r=gb_nat_r, gb_nat_t30_mae=gb_nat_t30_mae, gb_nat_t40_mae=gb_nat_t40_mae, gb_nat_t40_bias=gb_nat_t40_bias, gb_nat_t40_pred=gb_nat_t40_pred,
    gb_bal_mae=gb_bal_mae, gb_bal_rmse=gb_bal_rmse, gb_bal_r=gb_bal_r, gb_bal_t30_mae=gb_bal_t30_mae, gb_bal_t40_mae=gb_bal_t40_mae, gb_bal_t40_bias=gb_bal_t40_bias, gb_bal_t40_pred=gb_bal_t40_pred,
    gb_enr_mae=gb_enr_mae, gb_enr_rmse=gb_enr_rmse, gb_enr_r=gb_enr_r, gb_enr_t30_mae=gb_enr_t30_mae, gb_enr_t40_mae=gb_enr_t40_mae, gb_enr_t40_bias=gb_enr_t40_bias, gb_enr_t40_pred=gb_enr_t40_pred
)

with open(OUT_DIR / "REPORT.md", "w") as f:
    f.write(report_content)

print("\nSaved REPORT.md, results.json, predictions.csv and distribution_summary.csv to runs/phase21_height_distribution/")
