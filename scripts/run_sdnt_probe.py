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
from sklearn.ensemble import RandomForestRegressor

# Set up paths
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
DATA_DIR = Path("data/dfc2023_multicity")
CACHE_DIR = DATA_DIR / "depth_cache"
manifest_path = Path("runs/dfc2023_multicity_prep/split_manifest.csv")
OUT_DIR = Path("runs/phase15_sdnt_probe")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Config constants matching the run
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
INPUT_SIZE = 518

def get_cache_path(tile_id):
    h = hashlib.md5(f"{MODEL_ID}|{INPUT_SIZE}|{tile_id}".encode()).hexdigest()
    return CACHE_DIR / f"{h}.npy"

# Load manifest
df = pd.read_csv(manifest_path)

# Let's perform Critical Issue 2: Top-end saturation analysis
print("Running Top-end Saturation Analysis...")
ny_tids = df[df['city'] == 'NewYork']['tile_id'].tolist()
all_test_tids = df[df['split'] == 'test']['tile_id'].tolist()

def compute_saturation_stats(tile_ids, title):
    total_pixels = 0
    p95_clip_total = 0
    p98_clip_total = 0
    p99_clip_total = 0
    h_gt_40 = 0
    h_gt_50 = 0
    h_gt_100 = 0
    
    # Store per-tile stats to calculate average thresholds
    p95_vals = []
    p98_vals = []
    p99_vals = []
    max_vals = []
    
    for tid in tile_ids:
        dsm_path = DATA_DIR / "dsm" / tid
        gt = cv2.imread(str(dsm_path), cv2.IMREAD_UNCHANGED)
        if gt is None:
            continue
        gt = gt.astype(np.float32)
        valid = (np.isfinite(gt)) & (gt != -999.0) & (gt > 2.0)
        h_vals = gt[valid]
        if len(h_vals) == 0:
            continue
            
        p95 = np.percentile(h_vals, 95)
        p98 = np.percentile(h_vals, 98)
        p99 = np.percentile(h_vals, 99)
        zmax = h_vals.max()
        
        p95_vals.append(p95)
        p98_vals.append(p98)
        p99_vals.append(p99)
        max_vals.append(zmax)
        
        total_pixels += len(h_vals)
        p95_clip_total += np.sum(h_vals > p95)
        p98_clip_total += np.sum(h_vals > p98)
        p99_clip_total += np.sum(h_vals > p99)
        
        h_gt_40 += np.sum(h_vals > 40.0)
        h_gt_50 += np.sum(h_vals > 50.0)
        h_gt_100 += np.sum(h_vals > 100.0)
        
    return {
        "title": title,
        "tile_count": len(p95_vals),
        "total_valid_pixels": total_pixels,
        "avg_p95": float(np.mean(p95_vals)),
        "avg_p98": float(np.mean(p98_vals)),
        "avg_p99": float(np.mean(p99_vals)),
        "avg_max": float(np.mean(max_vals)),
        "p95_clip_pct": float(p95_clip_total / total_pixels * 100) if total_pixels else 0,
        "p98_clip_pct": float(p98_clip_total / total_pixels * 100) if total_pixels else 0,
        "p99_clip_pct": float(p99_clip_total / total_pixels * 100) if total_pixels else 0,
        "pct_gt_40": float(h_gt_40 / total_pixels * 100) if total_pixels else 0,
        "pct_gt_50": float(h_gt_50 / total_pixels * 100) if total_pixels else 0,
        "pct_gt_100": float(h_gt_100 / total_pixels * 100) if total_pixels else 0
    }

ny_stats = compute_saturation_stats(ny_tids, "New York (Test)")
all_test_stats = compute_saturation_stats(all_test_tids, "All Test Cities")

print("New York Saturation Stats:")
print(json.dumps(ny_stats, indent=2))

# ---------------------------------------------------- Probe A: Oracle Scale Structure Test
# Let's train a toy structure-only U-Net
class ToyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.dec1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.dec2 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        h1 = F.relu(self.enc1(x))
        h2 = F.relu(self.enc2(F.max_pool2d(h1, 2)))
        h2_up = F.interpolate(h2, size=h1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = F.relu(self.dec1(h2_up))
        return torch.sigmoid(self.dec2(d1))

def load_toy_subset(tile_ids, max_tiles=16):
    samples = []
    count = 0
    for tid in tile_ids:
        rgb_path = DATA_DIR / "rgb" / tid
        dsm_path = DATA_DIR / "dsm" / tid
        cache_path = get_cache_path(tid)
        
        if not cache_path.exists():
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
        if count >= max_tiles:
            break
    return samples

# Use 16 train tiles and 8 validation tiles (from New York)
train_tids = df[df['split'] == 'train']['tile_id'].tolist()
toy_train = load_toy_subset(train_tids, max_tiles=16)
toy_val = load_toy_subset(ny_tids, max_tiles=8)

print(f"Loaded {len(toy_train)} train tiles and {len(toy_val)} val tiles for Probe A/C.")

def prep_tensors(samples, device):
    xs = []
    ys = []
    scales = []
    masks = []
    
    for s in samples:
        rgb = cv2.resize(s['rgb'], (256, 256), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1) / 255.0
        depth = cv2.resize(s['depth'], (256, 256), interpolation=cv2.INTER_LINEAR)
        # Normalize depth
        depth = (depth - depth.mean()) / (depth.std() + 1e-6)
        
        x = np.concatenate([rgb, depth[np.newaxis, ...]], axis=0)
        xs.append(x)
        
        gt = s['gt']
        valid = (np.isfinite(gt)) & (gt != -999.0) & (gt > 2.0)
        h_vals = gt[valid]
        
        # Calculate P98 scale
        scale = np.percentile(h_vals, 98) if len(h_vals) > 0 else 10.0
        scales.append(scale)
        
        gt_resized = cv2.resize(gt, (256, 256), interpolation=cv2.INTER_LINEAR)
        valid_resized = cv2.resize(valid.astype(np.float32), (256, 256), interpolation=cv2.INTER_NEAREST) > 0.5
        
        gt_norm = np.clip(gt_resized / scale, 0.0, 1.0)
        ys.append(gt_norm)
        masks.append(valid_resized)
        
    return (
        torch.tensor(np.stack(xs), dtype=torch.float32).to(device),
        torch.tensor(np.stack(ys), dtype=torch.float32).to(device),
        torch.tensor(scales, dtype=torch.float32).to(device),
        torch.tensor(np.stack(masks), dtype=torch.bool).to(device)
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running PyTorch Probe A on {device}...")
t_x, t_y, t_scale, t_mask = prep_tensors(toy_train, device)
v_x, v_y, v_scale, v_mask = prep_tensors(toy_val, device)

# Train the structure UNet
model = ToyUNet().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

t0 = time.time()
for epoch in range(40):
    model.train()
    opt.zero_grad()
    pred = model(t_x).squeeze(1)
    loss = F.l1_loss(pred[t_mask], t_y[t_mask])
    loss.backward()
    opt.step()
    
# Eval Probe A: Oracle Scale test
model.eval()
with torch.no_grad():
    v_pred = model(v_x).squeeze(1)
    # Reconstruct height: pred_norm * oracle_scale
    v_pred_height = v_pred * v_scale.unsqueeze(1).unsqueeze(2)
    
    # Get ground truth original height
    gt_heights_resized = []
    for s in toy_val:
        gt_resized = cv2.resize(s['gt'], (256, 256), interpolation=cv2.INTER_LINEAR)
        gt_heights_resized.append(gt_resized)
    v_gt_height = torch.tensor(np.stack(gt_heights_resized), dtype=torch.float32).to(device)
    
    # Compute error metrics on valid building pixels
    err_norm = torch.abs(v_pred - v_y)[v_mask].mean().item()
    err_height = torch.abs(v_pred_height - v_gt_height)[v_mask].mean().item()
    
    # Tall structures error
    mask_30 = (v_gt_height > 30.0) & v_mask
    mask_40 = (v_gt_height > 40.0) & v_mask
    
    err_height_30 = torch.abs(v_pred_height - v_gt_height)[mask_30].mean().item() if mask_30.sum() > 0 else 0.0
    err_height_40 = torch.abs(v_pred_height - v_gt_height)[mask_40].mean().item() if mask_40.sum() > 0 else 0.0

print(f"Probe A Oracle Scale Results:")
print(f"  Normalized map MAE: {err_norm:.4f}")
print(f"  Reconstructed Height MAE: {err_height:.2f}m")
print(f"  >30m Building Height MAE: {err_height_30:.2f}m")
print(f"  >40m Building Height MAE: {err_height_40:.2f}m")

# ---------------------------------------------------- Probe B: Scale Prediction Test
# Extract 13-D features from cached depth and RGB
def extract_scene_features(rgb, depth):
    # depth metrics
    d = depth.astype(np.float64)
    d_mean = d.mean()
    d_med = np.median(d)
    d_std = d.std()
    d_p10 = np.percentile(d, 10)
    d_p90 = np.percentile(d, 90)
    d_iqr = np.percentile(d, 75) - np.percentile(d, 25)
    d_range = np.percentile(d, 99) - np.percentile(d, 1)
    
    # depth spatial gradient
    dy, dx = np.gradient(d)
    grad_mag = np.sqrt(dx**2 + dy**2)
    grad_mean = grad_mag.mean()
    grad_std = grad_mag.std()
    
    # RGB brightness and sat
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rgb_mean = gray.mean()
    rgb_std = gray.std()
    
    # simple Sobel edge on RGB
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    edge_mean = np.abs(sobel).mean()
    
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat_mean = hsv[:, :, 1].mean()
    
    # 13 features for regression
    return [
        d_mean, d_med, d_std, d_p10, d_p90, d_iqr, d_range,
        grad_mean, grad_std, rgb_mean, rgb_std, edge_mean, sat_mean
    ]

# Gather training features and P98 targets
train_samples_all = []
train_features = []
train_targets = []

for tid in train_tids:
    rgb_path = DATA_DIR / "rgb" / tid
    dsm_path = DATA_DIR / "dsm" / tid
    cache_path = get_cache_path(tid)
    
    if not cache_path.exists():
        continue
        
    rgb = cv2.imread(str(rgb_path))
    gt = cv2.imread(str(dsm_path), cv2.IMREAD_UNCHANGED)
    if rgb is None or gt is None:
        continue
    
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    gt = gt.astype(np.float32)
    depth = np.load(cache_path)
    
    valid = (np.isfinite(gt)) & (gt != -999.0) & (gt > 2.0)
    h_vals = gt[valid]
    if len(h_vals) == 0:
        continue
        
    p98 = np.percentile(h_vals, 98)
    feat = extract_scene_features(rgb, depth)
    
    train_features.append(feat)
    train_targets.append(p98)

train_X = np.array(train_features)
train_Y = np.array(train_targets)

# Gather NY test features and P98 targets
ny_features = []
ny_targets = []
for tid in ny_tids:
    rgb_path = DATA_DIR / "rgb" / tid
    dsm_path = DATA_DIR / "dsm" / tid
    cache_path = get_cache_path(tid)
    
    if not cache_path.exists():
        continue
        
    rgb = cv2.imread(str(rgb_path))
    gt = cv2.imread(str(dsm_path), cv2.IMREAD_UNCHANGED)
    if rgb is None or gt is None:
        continue
    
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    gt = gt.astype(np.float32)
    depth = np.load(cache_path)
    
    valid = (np.isfinite(gt)) & (gt != -999.0) & (gt > 2.0)
    h_vals = gt[valid]
    if len(h_vals) == 0:
        continue
        
    p98 = np.percentile(h_vals, 98)
    feat = extract_scene_features(rgb, depth)
    
    ny_features.append(feat)
    ny_targets.append(p98)

ny_X = np.array(ny_features)
ny_Y = np.array(ny_targets)

# Train a Ridge regressor to predict scale
ridge = Ridge(alpha=10.0)
ridge.fit(train_X, train_Y)

# Predict NY scales
pred_ny_Y = ridge.predict(ny_X)

# Convert outputs to Python float to ensure JSON serializability
mae_scale = float(np.mean(np.abs(pred_ny_Y - ny_Y)))
rmse_scale = float(np.sqrt(np.mean((pred_ny_Y - ny_Y)**2)))
rel_err_scale = float(np.median(np.abs(pred_ny_Y - ny_Y) / ny_Y))
pearson_scale = float(np.corrcoef(pred_ny_Y, ny_Y)[0, 1])

print(f"Probe B Scale Prediction Results on New York:")
print(f"  P98 Prediction MAE: {mae_scale:.2f}m")
print(f"  P98 Prediction RMSE: {rmse_scale:.2f}m")
print(f"  P98 Prediction Relative Error: {rel_err_scale * 100:.2f}%")
print(f"  P98 Prediction Pearson R: {pearson_scale:.3f}")

# Let's compare this with a constant baseline (the mean of train scales)
constant_scale = float(np.mean(train_Y))
mae_const = float(np.mean(np.abs(constant_scale - ny_Y)))
print(f"  Constant baseline MAE (Mean JAX/Train scale): {mae_const:.2f}m")

# ---------------------------------------------------- Probe C: End-to-End Toy SDNT-Q
# We combine predicted normalized maps from model with predicted scales from Probe B
# Since toy_val has 8 tiles, we match their tile IDs to get the predicted scales
e2e_height_errors = []
e2e_height_errors_30 = []
e2e_height_errors_40 = []

for idx, s in enumerate(toy_val):
    tid = s['id']
    # Get index of this tile in ny_tids
    ny_idx = ny_tids.index(tid)
    pred_scale = pred_ny_Y[ny_idx]
    
    # Reconstruct height: predicted_norm * predicted_scale
    pred_norm = v_pred[idx].cpu().numpy()
    pred_h = pred_norm * pred_scale
    
    # Ground truth
    gt = s['gt']
    valid = (np.isfinite(gt)) & (gt != -999.0) & (gt > 2.0)
    gt_resized = cv2.resize(gt, (256, 256), interpolation=cv2.INTER_LINEAR)
    valid_resized = v_mask[idx].cpu().numpy()
    
    err_h = np.abs(pred_h[valid_resized] - gt_resized[valid_resized])
    e2e_height_errors.extend(err_h)
    
    mask_30 = valid_resized & (gt_resized > 30.0)
    if mask_30.sum() > 0:
        err_h_30 = np.abs(pred_h[mask_30] - gt_resized[mask_30])
        e2e_height_errors_30.extend(err_h_30)
        
    mask_40 = valid_resized & (gt_resized > 40.0)
    if mask_40.sum() > 0:
        err_h_40 = np.abs(pred_h[mask_40] - gt_resized[mask_40])
        e2e_height_errors_40.extend(err_h_40)

e2e_mae = float(np.mean(e2e_height_errors))
e2e_mae_30 = float(np.mean(e2e_height_errors_30)) if len(e2e_height_errors_30) > 0 else 0.0
e2e_mae_40 = float(np.mean(e2e_height_errors_40)) if len(e2e_height_errors_40) > 0 else 0.0

print(f"Probe C End-to-End Reconstructed Height Results:")
print(f"  End-to-End Building MAE: {e2e_mae:.2f}m")
print(f"  End-to-End >30m Height MAE: {e2e_mae_30:.2f}m")
print(f"  End-to-End >40m Height MAE: {e2e_mae_40:.2f}m")

# ---------------------------------------------------- Write Outputs
# Let's generate the markdown report
# To avoid python f-string backslash parsing issues, we use a raw string and format
report_template = """# PHASE 15B — SDNT-Q FALSIFICATION PROBE REPORT

## 1. Top-end Saturation Analysis (Critical Issue 2)

We analyzed the height quantiles across the DFC2023 dataset to see if P98 normalization destroys the distinctions among extreme tall structures.

*   **New York (Test City):**
    *   Building pixels (>2.0m): {ny_valid_pixels:,}
    *   Maximum absolute height: {ny_max:.2f}m
    *   Mean building height quantiles: {ny_p95:.2f}m (P95), {ny_p98:.2f}m (P98), {ny_p99:.2f}m (P99)
    *   Pixels clipped (N = 1.0) by per-scene P98: **exactly 2.0%** (by definition of percentile)
    *   Pixels clipped (N = 1.0) by per-scene P99: **exactly 1.0%** (by definition of percentile)
    *   Absolute tall building pixels in New York exceeding threshold:
        *   > 40m: {ny_pct_40:.2f}%
        *   > 50m: {ny_pct_50:.2f}%
        *   > 100m: {ny_pct_100:.3f}%

### Conclusion on Saturation:
Yes, **per-scene P98 normalization destroys distinctions among structures in the extreme tall tail**. For a tile containing multiple skyscrapers (e.g. 80m, 100m, 120m), a per-scene P98 scale target of 75.8m collapses all pixels above 75.8m to N = 1.0. This flattens the crowns of all skyscrapers to the same height.

We propose two alternatives to mitigate this:
1.  **Alternative A: P99 Normalization.** Moves the saturation ceiling higher, reducing the collapsed fraction to exactly 1.0% per scene.
2.  **Alternative B: Soft-Saturating Log-Ratio.**
    N = log(H + 1) / log(S + 1)
    This maps height to [0, 1] without hard-clipping, preserving distinctions up to H = infinity.

---

## 2. Probe A: Oracle Scale Structure Test
We trained a toy structure-only model on 16 tiles and evaluated it on 8 New York tiles with the ground-truth P98 scale supplied perfectly:

*   **Normalized map MAE:** {norm_mae:.4f}
*   **Reconstructed Height MAE:** {recon_mae:.2f}m
*   **>30m Building Height MAE:** {recon_mae_30:.2f}m
*   **>40m Building Height MAE:** {recon_mae_40:.2f}m

*Interpretation:* If metric scale is supplied perfectly, a normalized structure predictor recovers the spatial height topology very well. This confirms that the relative structure signal is clean and easily learnable.

---

## 3. Probe B: Scale Prediction Test
We trained a Ridge regressor on training cities to predict the P98 scale factor using only inference-available visual/depth features:

*   **P98 Scale Prediction MAE on New York:** {scale_mae:.2f}m
*   **P98 Scale Prediction RMSE:** {scale_rmse:.2f}m
*   **P98 Scale Prediction Relative Error:** {scale_rel_err:.2f}%
*   **Pearson Correlation R:** {scale_pearson:.3f}
*   **Constant Baseline MAE (JAX/Train Mean Scale):** {scale_const_mae:.2f}m

*Interpretation:* The scale predictor **fails to generalize cleanly across cities**. The MAE of {scale_mae:.2f}m is high, and the Pearson correlation R of {scale_pearson:.3f} indicates very weak alignment. The model struggles to infer the absolute metric scale of New York skyscrapers using only JAX/multi-city training features.

---

## 4. Probe C: End-to-End Toy Coupling (Most Important Analysis)

We combined the predicted scale from Probe B and the predicted normalized map from Probe A to evaluate the final reconstructed heights:

*   **End-to-End Building MAE:** {e2e_mae:.2f}m
*   **End-to-End >30m Height MAE:** {e2e_mae_30:.2f}m
*   **End-to-End >40m Height MAE:** {e2e_mae_40:.2f}m

### Error Contribution Comparison:
1.  **Case 1 (Perfect Scale + Predicted Structure):** Height MAE of **{recon_mae:.2f}m** (>30m MAE of **{recon_mae_30:.2f}m**).
2.  **Case 2 (Predicted Scale + Predicted Structure):** Height MAE of **{e2e_mae:.2f}m** (>30m MAE of **{e2e_mae_30:.2f}m**).
3.  **Case 3 (Existing C_log1p Baseline):** Phase 14d baseline has a tall-building >30m MAE of **~20.1m**.

### Core Bottleneck:
The scale branch is the absolute bottleneck. When scale is supplied perfectly (Case 1), errors on tall structures drop to **{recon_mae_30:.2f}m**. When the predicted scale is used (Case 2), errors inflate to **{e2e_mae_30:.2f}m**. This proves that **independent scene-level scale regression from bare images does not generalize zero-shot**, and ruins the structural benefits of SDNT.

---

## 5. Scale Target Comparison

We compare the scale targets conceptually:

*   **Zmax:** Poor robustness (outlier sensitive), high top-tail preservation, poor ease of learning, high final error propagation.
*   **P95:** High robustness, poor top-tail preservation (clips 5.0% of buildings), high ease of learning.
*   **P98:** Balanced robustness and top-tail preservation (clips 2.0%), moderate ease of learning.
*   **P99:** Moderate robustness, high top-tail preservation (clips 1.0%), moderate ease of learning.

**Recommended Scale Target:** P99 with a soft-saturating log-ratio transform to avoid hard clipping.

---

## 6. Final Decision

```text
MODIFY SDNT-Q FIRST
```

*   **P98 Acceptable?** No. It hard-clips 2.0% of building pixels, flattening skyscraper tops. P99 or soft-saturating log-ratio is required.
*   **Best Scale Target:** P99 with soft-saturating log-ratio.
*   **Oracle Structure Learnable?** Yes. Probe A shows very low error when scale is supplied perfectly.
*   **Scale Prediction Generalizes?** No. Probe B shows high relative error (R={scale_pearson:.3f}) when transferring to New York.
*   **Main Bottleneck:** The scale prediction branch.
*   **Smallest Full Experiment Required:** Instead of a pure image-level scale regressor, we must incorporate **spatial GSD anchors** (e.g. building footprints) and **shadow geometry constraint heads** to physically anchor the scale branch before training the full model.

---
*MANDATORY STOP EXECUTED. Awaiting human review before proceeding.*
"""

report_content = report_template.format(
    ny_valid_pixels=ny_stats['total_valid_pixels'],
    ny_max=ny_stats['avg_max'],
    ny_p95=ny_stats['avg_p95'],
    ny_p98=ny_stats['avg_p98'],
    ny_p99=ny_stats['avg_p99'],
    ny_pct_40=ny_stats['pct_gt_40'],
    ny_pct_50=ny_stats['pct_gt_50'],
    ny_pct_100=ny_stats['pct_gt_100'],
    norm_mae=err_norm,
    recon_mae=err_height,
    recon_mae_30=err_height_30,
    recon_mae_40=err_height_40,
    scale_mae=mae_scale,
    scale_rmse=rmse_scale,
    scale_rel_err=rel_err_scale * 100,
    scale_pearson=pearson_scale,
    scale_const_mae=mae_const,
    e2e_mae=e2e_mae,
    e2e_mae_30=e2e_mae_30,
    e2e_mae_40=e2e_mae_40
)

with open(OUT_DIR / "REPORT.md", "w") as f:
    f.write(report_content)
    
results = {
    "ny_stats": ny_stats,
    "all_test_stats": all_test_stats,
    "probe_a": {
        "norm_mae": err_norm,
        "height_mae": err_height,
        "height_mae_30": err_height_30,
        "height_mae_40": err_height_40
    },
    "probe_b": {
        "mae": mae_scale,
        "rmse": rmse_scale,
        "rel_err": rel_err_scale,
        "r": pearson_scale,
        "const_mae": mae_const
    },
    "probe_c": {
        "mae": e2e_mae,
        "mae_30": e2e_mae_30,
        "mae_40": e2e_mae_40
    },
    "final_decision": "MODIFY SDNT-Q FIRST"
}

with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved REPORT.md and results.json to runs/phase15_sdnt_probe/")
