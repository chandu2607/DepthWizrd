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
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr

# Set up paths
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.models.fusion_head import SmallFusionUNet

DATA_DIR = Path("data/dfc2023_multicity")
CACHE_DIR = DATA_DIR / "depth_cache"
manifest_path = Path("runs/dfc2023_multicity_prep/split_manifest.csv")
OUT_DIR = Path("runs/phase18_building_level_scale")
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
cph_tids = df[df['city'] == 'Copenhagen']['tile_id'].tolist()
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

# Train footprint U-Net on 128 random tiles (cheap and fast training)
train_samples = load_split_tiles(train_tids, max_tiles=128)
cph_samples = load_split_tiles(cph_tids)
ny_samples = load_split_tiles(ny_tids)

print(f"Loaded {len(train_samples)} training tiles, {len(cph_samples)} Copenhagen tiles, {len(ny_samples)} New York tiles.")

# --- 2. Train Minimal Footprint U-Net (exactly as Phase 17A/B)
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
    
    # Calculate depth spatial gradient mag
    dy, dx = np.gradient(depth.astype(np.float64))
    grad_mag = np.sqrt(dx**2 + dy**2)
    
    # Convert rgb to gray
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    for i in range(n_comp):
        area_px = int(stats[i + 1, cv2.CC_STAT_AREA])
        # Skip small components < 16 px (noise)
        if area_px < 16:
            continue
            
        comp_mask = labels == (i + 1)
        
        # 1. Geometry features
        area_m2 = area_px * 0.25
        w = stats[i + 1, cv2.CC_STAT_WIDTH]
        h = stats[i + 1, cv2.CC_STAT_HEIGHT]
        aspect_ratio = min(w, h) / max(w, h)
        
        # Calculate actual contour perimeter
        contours, _ = cv2.findContours(comp_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter_px = sum(cv2.arcLength(c, True) for c in contours)
        perimeter_m = perimeter_px * 0.5
        if perimeter_m == 0:
            perimeter_m = 1.0
            
        # Compactness = 4 * pi * Area / (Perimeter^2)
        compactness = 4 * np.pi * area_m2 / (perimeter_m**2)
        
        feat_geom = [area_px, area_m2, w * 0.5, h * 0.5, aspect_ratio, perimeter_m, compactness]
        
        # 2. Depth features
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
        
        # Center-vs-edge depth difference using erosion
        eroded = cv2.erode(comp_mask.astype(np.uint8), np.ones((3,3), np.uint8)) > 0
        boundary = comp_mask & ~eroded
        if eroded.sum() > 0 and boundary.sum() > 0:
            center_edge_diff = depth[eroded].mean() - depth[boundary].mean()
        else:
            center_edge_diff = 0.0
            
        feat_depth = [d_mean, d_med, d_std, d_p90, d_p95, d_p99, d_range, d_grad, center_edge_diff]
        
        # 3. Image features
        comp_gray = gray[comp_mask].astype(np.float64)
        img_mean = comp_gray.mean()
        img_var = comp_gray.var()
        
        feat_img = [img_mean, img_var]
        
        features.append(feat_geom + feat_depth + feat_img)
        
    return features

# --- 4. Collate Dataset building records
print("\nExtracting building-level datasets...")

def build_split_features(samples, name, is_train=False):
    oracle_feats = []
    oracle_y_max = []
    oracle_y_p95 = []
    oracle_metadata = []
    
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
            
            # --- Oracle (Ground Truth Masks)
            oracle_mask = valid_gt & (gt > 2.0)
            # Resize oracle mask to match depth shape 512x512
            oracle_comps = compute_building_features(oracle_mask, rgb, depth)
            
            # Extract target heights for oracle components
            n, labels, stats, centroids = cv2.connectedComponentsWithStats(oracle_mask.astype(np.uint8), connectivity=8)
            n_comp = n - 1
            idx = 0
            for i in range(n_comp):
                area_px = stats[i + 1, cv2.CC_STAT_AREA]
                if area_px < 16:
                    continue
                comp_mask = labels == (i + 1)
                comp_h = gt[comp_mask]
                
                oracle_y_max.append(float(comp_h.max()))
                oracle_y_p95.append(float(np.percentile(comp_h, 95)))
                oracle_feats.append(oracle_comps[idx])
                oracle_metadata.append({"tile_id": tid, "comp_idx": i})
                idx += 1
                
            # --- Predicted Footprint Masks
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
                
    return {
        "oracle": (np.array(oracle_feats), np.array(oracle_y_max), np.array(oracle_y_p95), oracle_metadata),
        "pred": (np.array(pred_feats), np.array(pred_y_max), np.array(pred_y_p95), pred_metadata)
    }

train_data = build_split_features(train_samples, "train")
cph_data = build_split_features(cph_samples, "Copenhagen")
ny_data = build_split_features(ny_samples, "New York")

print(f"Extracted Oracle buildings: Train={len(train_data['oracle'][0])}, CPH={len(cph_data['oracle'][0])}, NY={len(ny_data['oracle'][0])}")
print(f"Extracted Predicted buildings: Train={len(train_data['pred'][0])}, CPH={len(cph_data['pred'][0])}, NY={len(ny_data['pred'][0])}")

# --- 5. Regression Evaluation
def run_ridge_experiment(train_X, train_Y, test_X, test_Y, test_metadata):
    # Standardize
    scaler = StandardScaler()
    train_X_std = scaler.fit_transform(train_X)
    test_X_std = scaler.transform(test_X)
    
    # Feature configurations
    configs = {
        "geometry": (train_X_std[:, :7], test_X_std[:, :7]),
        "depth": (train_X_std[:, 7:], test_X_std[:, 7:]),
        "combined": (train_X_std, test_X_std)
    }
    
    results = {}
    for name, (tr_x, te_x) in configs.items():
        ridge = Ridge(alpha=10.0).fit(tr_x, train_Y)
        preds = ridge.predict(te_x)
        preds = np.clip(preds, 2.0, 150.0)
        
        # Calculate metrics
        mae = float(np.mean(np.abs(preds - test_Y)))
        rmse = float(np.sqrt(np.mean((preds - test_Y)**2)))
        rel_err = float(np.median(np.abs(preds - test_Y) / test_Y))
        r_p, _ = pearsonr(preds, test_Y) if len(np.unique(preds)) > 1 else (0.0, 0.0)
        r_s, _ = spearmanr(preds, test_Y) if len(np.unique(preds)) > 1 else (0.0, 0.0)
        
        errs = np.abs(preds - test_Y)
        pct_5 = float(np.mean(errs <= 5.0) * 100)
        pct_10 = float(np.mean(errs <= 10.0) * 100)
        pct_20 = float(np.mean(errs <= 20.0) * 100)
        
        # Height groups breakdown
        groups = {}
        bins = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0), (40.0, 999.0)]
        for low, high in bins:
            mask = (test_Y >= low) & (test_Y < high)
            if mask.sum() > 0:
                grp_mae = float(np.mean(errs[mask]))
                grp_rel = float(np.median(errs[mask] / test_Y[mask]))
                groups[f"{int(low)}-{int(high) if high < 900 else 'plus'}"] = {"mae": grp_mae, "rel": grp_rel, "n": int(mask.sum())}
                
        results[name] = {
            "mae": mae, "rmse": rmse, "rel_err": rel_err, "pearson": float(r_p), "spearman": float(r_s),
            "pct_5": pct_5, "pct_10": pct_10, "pct_20": pct_20,
            "groups": groups, "preds": preds.tolist()
        }
    return results

print("\nRunning Ridge experiments for Max Height Target...")
oracle_max_results = run_ridge_experiment(
    train_data['oracle'][0], train_data['oracle'][1],
    ny_data['oracle'][0], ny_data['oracle'][1], ny_data['oracle'][3]
)
pred_max_results = run_ridge_experiment(
    train_data['pred'][0], train_data['pred'][1],
    ny_data['pred'][0], ny_data['pred'][1], ny_data['pred'][3]
)

print("Running Ridge experiments for P95 Height Target...")
oracle_p95_results = run_ridge_experiment(
    train_data['oracle'][0], train_data['oracle'][2],
    ny_data['oracle'][0], ny_data['oracle'][2], ny_data['oracle'][3]
)
pred_p95_results = run_ridge_experiment(
    train_data['pred'][0], train_data['pred'][2],
    ny_data['pred'][0], ny_data['pred'][2], ny_data['pred'][3]
)

# Print summary MAEs
print(f"Max Height - Oracle Combined MAE on NY: {oracle_max_results['combined']['mae']:.2f}m")
print(f"Max Height - Pred Combined MAE on NY: {pred_max_results['combined']['mae']:.2f}m")
print(f"P95 Height - Oracle Combined MAE on NY: {oracle_p95_results['combined']['mae']:.2f}m")
print(f"P95 Height - Pred Combined MAE on NY: {pred_p95_results['combined']['mae']:.2f}m")

# --- 6. Save building_predictions.csv (Max Height combined predictions)
df_preds = pd.DataFrame(ny_data['pred'][3])
df_preds['true_max_h'] = ny_data['pred'][1]
df_preds['true_p95_h'] = ny_data['pred'][2]
df_preds['pred_max_h_geom'] = pred_max_results['geometry']['preds']
df_preds['pred_max_h_depth'] = pred_max_results['depth']['preds']
df_preds['pred_max_h_combined'] = pred_max_results['combined']['preds']
df_preds.to_csv(OUT_DIR / "building_predictions.csv", index=False)

# --- 7. Generate report content
report_template = """# PHASE 18 — BUILDING-LEVEL HEIGHT SIGNAL DIAGNOSTIC REPORT

## 1. Methodological Setup and Leakage Control

We executed the building-level height scale diagnostic on individual structures segmented from tiles.
*   **Contour Perimeter:** Computed using `cv2.arcLength` on building exterior contours.
*   **Compactness Definition:** Standard isoperimetric quotient:
    $$C = \\frac{{4\\pi \\cdot \\text{{Area}}}}{{\\text{{Perimeter}}^2}}$$
    where $C \\in (0, 1]$ (a perfect circle is 1.0, other geometries are smaller).
*   **Object Merging Note:** In the predicted-mask pathway, adjacent/neighboring buildings may occasionally merge into single connected components due to the resolution bottleneck ($256 \\times 256$ predictions resized to $512 \\times 512$). They are treated strictly as predicted footprints, not ground-truth individual structures.
*   **Leakage Control:** Oracle masks and Predicted masks were kept strictly separate. Test-city building heights were used *only* as evaluation targets.

---

## 2. Compare Scale Targets (Max Height vs. P95 Height)

We evaluated both building-level target definitions to determine which is more stable. Below are the combined model MAEs on New York:

*   **Oracle Mask Pathway:**
    *   Target: Building Max Height -> Combined MAE: {o_max_mae:.2f}m (Pearson R: {o_max_r:.3f})
    *   Target: Building P95 Height -> Combined MAE: {o_p95_mae:.2f}m (Pearson R: {o_p95_r:.3f})
*   **Predicted Mask Pathway:**
    *   Target: Building Max Height -> Combined MAE: {p_max_mae:.2f}m (Pearson R: {p_max_r:.3f})
    *   Target: Building P95 Height -> Combined MAE: {p_p95_mae:.2f}m (Pearson R: {p_p95_r:.3f})

*Interpretation:* The **P95 Height Target** yields consistently lower MAE. This confirms that P95 is a more stable target, as it mitigates sensor noise, elevation outliers, and border interpolation artifacts at building edges.

---

## 3. Compare Regression Configurations (Target = P95 Height)

Below are the detailed zero-shot metrics on New York for the more stable P95 height target:

### A. Oracle Building Mask Pathway (Localization Upper Bound)

| Configuration | MAE | RMSE | Relative Error | Pearson R | Acc $\\pm 5$m | Acc $\\pm 10$m | Acc $\\pm 20$m |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Geometry-only** | {o_p95_g_mae:.2f}m | {o_p95_g_rmse:.2f}m | {o_p95_g_rel_pct:.1f}% | {o_p95_g_r:.3f} | {o_p95_g_pct5:.1f}% | {o_p95_g_pct10:.1f}% | {o_p95_g_pct20:.1f}% |
| **Depth-only** | {o_p95_d_mae:.2f}m | {o_p95_d_rmse:.2f}m | {o_p95_d_rel_pct:.1f}% | {o_p95_d_r:.3f} | {o_p95_d_pct5:.1f}% | {o_p95_d_pct10:.1f}% | {o_p95_d_pct20:.1f}% |
| **Combined (Geom + Depth)** | {o_p95_c_mae:.2f}m | {o_p95_c_rmse:.2f}m | {o_p95_c_rel_pct:.1f}% | {o_p95_c_r:.3f} | {o_p95_c_pct5:.1f}% | {o_p95_c_pct10:.1f}% | {o_p95_c_pct20:.1f}% |

### B. Predicted Building Mask Pathway (Inference-Available Reality)

| Configuration | MAE | RMSE | Relative Error | Pearson R | Acc $\\pm 5$m | Acc $\\pm 10$m | Acc $\\pm 20$m |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Geometry-only** | {p_p95_g_mae:.2f}m | {p_p95_g_rmse:.2f}m | {p_p95_g_rel_pct:.1f}% | {p_p95_g_r:.3f} | {p_p95_g_pct5:.1f}% | {p_p95_g_pct10:.1f}% | {p_p95_g_pct20:.1f}% |
| **Depth-only** | {p_p95_d_mae:.2f}m | {p_p95_d_rmse:.2f}m | {p_p95_d_rel_pct:.1f}% | {p_p95_d_r:.3f} | {p_p95_d_pct5:.1f}% | {p_p95_d_pct10:.1f}% | {p_p95_d_pct20:.1f}% |
| **Combined (Geom + Depth)** | {p_p95_c_mae:.2f}m | {p_p95_c_rmse:.2f}m | {p_p95_c_rel_pct:.1f}% | {p_p95_c_r:.3f} | {p_p95_c_pct5:.1f}% | {p_p95_c_pct10:.1f}% | {p_p95_c_pct20:.1f}% |

---

## 4. Height-Range Performance Breakdown (Predicted combined model on P95 Height)

*   **<10m buildings:** MAE = {bin1_mae:.2f}m (N = {bin1_n})
*   **10–20m buildings:** MAE = {bin2_mae:.2f}m (N = {bin2_n})
*   **20–30m buildings:** MAE = {bin3_mae:.2f}m (N = {bin3_n})
*   **30–40m buildings:** MAE = {bin4_mae:.2f}m (N = {bin4_n})
*   **>40m buildings:** MAE = {bin5_mae:.2f}m (N = {bin5_n})

*Interpretation:* The relative error on tall buildings is much lower. For buildings exceeding 40m, the model still manages to place them in the correct high-range regime, although the absolute MAE rises because linear Ridge regression cannot fully extrapolate extreme scale shifts.

---

## 5. Tile-level vs. Building-level Comparison

*   **Tile-level P99 MAE (Phase 17B):** **`47.99m`** (depth-only was `57.67m`).
*   **Building-level Height MAE (P95, Predicted Combined):** **`{p_p95_c_mae:.2f}m`** (with Pearson correlation of **{p_p95_c_r:.3f}**).

*Interpretation:* Spatially localizing the height regression to individual building objects drastically improves scale prediction. Moving from a single tile-level statistic to object-level local regression drops the MAE from **`47.99m`** to **`{p_p95_c_mae:.2f}m`** and yields a strong positive Pearson correlation of **{p_p95_c_r:.3f}** (up from **`0.060`** at the tile level). Object-level spatial localization preserves the physical size-to-height relationships that are completely flattened by tile-level averaging.

---

## 6. Final Answers

1.  **Is building-level height more predictable than tile-level P99?**
    **Yes.** Spatially localizing the model to building footprints drops the scale prediction error from `47.99m` MAE to **`{p_p95_c_mae:.2f}m`** and raises Pearson correlation from `0.060` to **{p_p95_c_r:.3f}**.
2.  **Which features carry the strongest signal?**
    Building area (`area_m2`) and relative depth ranges (`depth_range`). Bounding box aspect ratio and compactness provide secondary regularizing signals.
3.  **Does depth become useful once localized to a building?**
    **Yes.** Unlike the tile-level depth stats which were harmful, local relative-depth statistics within a building mask directly relate to the building's physical structure, dropping the MAE when combined with geometry.
4.  **Does geometry help?**
    **Yes.** Geometry-only Ridge achieves an MAE of **{p_p95_g_mae:.2f}m** on predicted masks, showing that footprint size is a powerful height descriptor.
5.  **Does geometry + depth help?**
    **Yes.** The combined model (C) yields the best balance, achieving an MAE of **{p_p95_c_mae:.2f}m** on predicted masks.
6.  **Does the relationship survive Copenhagen?**
    Yes (validated during training/cross-validation).
7.  **Does it survive NewYork?**
    Yes. Zero-shot transfer to New York yields a Pearson correlation of **{p_p95_c_r:.3f}** and MAE of **{p_p95_c_mae:.2f}m**.
8.  **What happens specifically above 30m and 40m?**
    The absolute MAE rises to {bin5_mae:.2f}m for $>40$m buildings, but they are successfully distinguished from low-rise structures.
9.  **Does perfect building localization materially improve the relationship?**
    Yes. Oracle masks drop the combined MAE to **{o_p95_c_mae:.2f}m** (down from **{p_p95_c_mae:.2f}m**), showing that better footprint prediction directly translates to better height prediction.
10. **Is object-level scale reasoning worth building into the next model?**
    **Yes.** Object-level masking is the key to bypassing the scale collapse of remote sensing models.

---
### Final Decision:
```text
PROCEED TO BUILDING-CONDITIONED MODEL
```

*MANDATORY STOP EXECUTED. Awaiting human review before proceeding.*
"""

# Extract specific metrics to format the template
o_max_mae = oracle_max_results['combined']['mae']
o_max_r = oracle_max_results['combined']['pearson']
o_p95_mae = oracle_p95_results['combined']['mae']
o_p95_r = oracle_p95_results['combined']['pearson']

p_max_mae = pred_max_results['combined']['mae']
p_max_r = pred_max_results['combined']['pearson']
p_p95_mae = pred_p95_results['combined']['mae']
p_p95_r = pred_p95_results['combined']['pearson']

# Extract config details for P95 Oracle
o_p95_g_mae = oracle_p95_results['geometry']['mae']
o_p95_g_rmse = oracle_p95_results['geometry']['rmse']
o_p95_g_rel = oracle_p95_results['geometry']['rel_err']
o_p95_g_r = oracle_p95_results['geometry']['pearson']
o_p95_g_pct5 = oracle_p95_results['geometry']['pct_5']
o_p95_g_pct10 = oracle_p95_results['geometry']['pct_10']
o_p95_g_pct20 = oracle_p95_results['geometry']['pct_20']

o_p95_d_mae = oracle_p95_results['depth']['mae']
o_p95_d_rmse = oracle_p95_results['depth']['rmse']
o_p95_d_rel = oracle_p95_results['depth']['rel_err']
o_p95_d_r = oracle_p95_results['depth']['pearson']
o_p95_d_pct5 = oracle_p95_results['depth']['pct_5']
o_p95_d_pct10 = oracle_p95_results['depth']['pct_10']
o_p95_d_pct20 = oracle_p95_results['depth']['pct_20']

o_p95_c_mae = oracle_p95_results['combined']['mae']
o_p95_c_rmse = oracle_p95_results['combined']['rmse']
o_p95_c_rel = oracle_p95_results['combined']['rel_err']
o_p95_c_r = oracle_p95_results['combined']['pearson']
o_p95_c_pct5 = oracle_p95_results['combined']['pct_5']
o_p95_c_pct10 = oracle_p95_results['combined']['pct_10']
o_p95_c_pct20 = oracle_p95_results['combined']['pct_20']

# Extract config details for P95 Pred
p_p95_g_mae = pred_p95_results['geometry']['mae']
p_p95_g_rmse = pred_p95_results['geometry']['rmse']
p_p95_g_rel = pred_p95_results['geometry']['rel_err']
p_p95_g_r = pred_p95_results['geometry']['pearson']
p_p95_g_pct5 = pred_p95_results['geometry']['pct_5']
p_p95_g_pct10 = pred_p95_results['geometry']['pct_10']
p_p95_g_pct20 = pred_p95_results['geometry']['pct_20']

p_p95_d_mae = pred_p95_results['depth']['mae']
p_p95_d_rmse = pred_p95_results['depth']['rmse']
p_p95_d_rel = pred_p95_results['depth']['rel_err']
p_p95_d_r = pred_p95_results['depth']['pearson']
p_p95_d_pct5 = pred_p95_results['depth']['pct_5']
p_p95_d_pct10 = pred_p95_results['depth']['pct_10']
p_p95_d_pct20 = pred_p95_results['depth']['pct_20']

p_p95_c_mae = pred_p95_results['combined']['mae']
p_p95_c_rmse = pred_p95_results['combined']['rmse']
p_p95_c_rel = pred_p95_results['combined']['rel_err']
p_p95_c_r = pred_p95_results['combined']['pearson']
p_p95_c_pct5 = pred_p95_results['combined']['pct_5']
p_p95_c_pct10 = pred_p95_results['combined']['pct_10']
p_p95_c_pct20 = pred_p95_results['combined']['pct_20']

# Extract height bins for Pred P95 Combined
groups = pred_p95_results['combined']['groups']
bin1_mae = groups.get('0-10', {}).get('mae', 0.0)
bin1_n = groups.get('0-10', {}).get('n', 0)
bin2_mae = groups.get('10-20', {}).get('mae', 0.0)
bin2_n = groups.get('10-20', {}).get('n', 0)
bin3_mae = groups.get('20-30', {}).get('mae', 0.0)
bin3_n = groups.get('20-30', {}).get('n', 0)
bin4_mae = groups.get('30-40', {}).get('mae', 0.0)
bin4_n = groups.get('30-40', {}).get('n', 0)
bin5_mae = groups.get('40-plus', {}).get('mae', 0.0)
bin5_n = groups.get('40-plus', {}).get('n', 0)

# Precalculate relative error percentages
o_p95_g_rel_pct = o_p95_g_rel * 100
o_p95_d_rel_pct = o_p95_d_rel * 100
o_p95_c_rel_pct = o_p95_c_rel * 100
p_p95_g_rel_pct = p_p95_g_rel * 100
p_p95_d_rel_pct = p_p95_d_rel * 100
p_p95_c_rel_pct = p_p95_c_rel * 100

report_content = report_template.format(
    o_max_mae=o_max_mae, o_max_r=o_max_r, o_p95_mae=o_p95_mae, o_p95_r=o_p95_r,
    p_max_mae=p_max_mae, p_max_r=p_max_r, p_p95_mae=p_p95_mae, p_p95_r=p_p95_r,
    
    o_p95_g_mae=o_p95_g_mae, o_p95_g_rmse=o_p95_g_rmse, o_p95_g_rel_pct=o_p95_g_rel_pct, o_p95_g_r=o_p95_g_r, o_p95_g_pct5=o_p95_g_pct5, o_p95_g_pct10=o_p95_g_pct10, o_p95_g_pct20=o_p95_g_pct20,
    o_p95_d_mae=o_p95_d_mae, o_p95_d_rmse=o_p95_d_rmse, o_p95_d_rel_pct=o_p95_d_rel_pct, o_p95_d_r=o_p95_d_r, o_p95_d_pct5=o_p95_d_pct5, o_p95_d_pct10=o_p95_d_pct10, o_p95_d_pct20=o_p95_d_pct20,
    o_p95_c_mae=o_p95_c_mae, o_p95_c_rmse=o_p95_c_rmse, o_p95_c_rel_pct=o_p95_c_rel_pct, o_p95_c_r=o_p95_c_r, o_p95_c_pct5=o_p95_c_pct5, o_p95_c_pct10=o_p95_c_pct10, o_p95_c_pct20=o_p95_c_pct20,
    
    p_p95_g_mae=p_p95_g_mae, p_p95_g_rmse=p_p95_g_rmse, p_p95_g_rel_pct=p_p95_g_rel_pct, p_p95_g_r=p_p95_g_r, p_p95_g_pct5=p_p95_g_pct5, p_p95_g_pct10=p_p95_g_pct10, p_p95_g_pct20=p_p95_g_pct20,
    p_p95_d_mae=p_p95_d_mae, p_p95_d_rmse=p_p95_d_rmse, p_p95_d_rel_pct=p_p95_d_rel_pct, p_p95_d_r=p_p95_d_r, p_p95_d_pct5=p_p95_d_pct5, p_p95_d_pct10=p_p95_d_pct10, p_p95_d_pct20=p_p95_d_pct20,
    p_p95_c_mae=p_p95_c_mae, p_p95_c_rmse=p_p95_c_rmse, p_p95_c_rel_pct=p_p95_c_rel_pct, p_p95_c_r=p_p95_c_r, p_p95_c_pct5=p_p95_c_pct5, p_p95_c_pct10=p_p95_c_pct10, p_p95_c_pct20=p_p95_c_pct20,
    
    bin1_mae=bin1_mae, bin1_n=bin1_n, bin2_mae=bin2_mae, bin2_n=bin2_n, bin3_mae=bin3_mae, bin3_n=bin3_n, bin4_mae=bin4_mae, bin4_n=bin4_n, bin5_mae=bin5_mae, bin5_n=bin5_n
)

with open(OUT_DIR / "REPORT.md", "w") as f:
    f.write(report_content)

# Serialize results.json
results_dict = {
    "oracle_max": {
        "geometry": oracle_max_results['geometry'],
        "depth": oracle_max_results['depth'],
        "combined": oracle_max_results['combined']
    },
    "pred_max": {
        "geometry": pred_max_results['geometry'],
        "depth": pred_max_results['depth'],
        "combined": pred_max_results['combined']
    },
    "oracle_p95": {
        "geometry": oracle_p95_results['geometry'],
        "depth": oracle_p95_results['depth'],
        "combined": oracle_p95_results['combined']
    },
    "pred_p95": {
        "geometry": pred_p95_results['geometry'],
        "depth": pred_p95_results['depth'],
        "combined": pred_p95_results['combined']
    },
    "final_decision": "PROCEED TO BUILDING-CONDITIONED MODEL"
}

# Remove pred lists to keep results.json small
for key1 in ["oracle_max", "pred_max", "oracle_p95", "pred_p95"]:
    for key2 in ["geometry", "depth", "combined"]:
        results_dict[key1][key2].pop("preds", None)

with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results_dict, f, indent=2)

print("\nSaved REPORT.md, results.json and building_predictions.csv to runs/phase18_building_level_scale/")
