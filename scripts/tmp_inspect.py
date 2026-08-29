import os
import sys
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

DATA_DIR = Path("data/dfc2023_multicity")
manifest_path = Path("runs/dfc2023_multicity_prep/split_manifest.csv")
df = pd.read_csv(manifest_path)

# Let's inspect the cities and splits
print("Cities and splits:")
print(df.groupby(['city', 'split']).size())

# Let's write a function to analyze quantiles and clipping for a set of files
def analyze_clipping(tile_ids, title):
    p95_clip_pcts = []
    p98_clip_pcts = []
    p99_clip_pcts = []
    
    total_valid_pixels = 0
    gt_all_valid = []
    
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
        
        gt_all_valid.extend(h_vals[::10]) # subsample to save memory
        
        p95 = np.percentile(h_vals, 95)
        p98 = np.percentile(h_vals, 98)
        p99 = np.percentile(h_vals, 99)
        
        p95_clip_pcts.append(np.sum(h_vals > p95) / len(h_vals) * 100)
        p98_clip_pcts.append(np.sum(h_vals > p98) / len(h_vals) * 100)
        p99_clip_pcts.append(np.sum(h_vals > p99) / len(h_vals) * 100)
        
    gt_all_valid = np.array(gt_all_valid)
    if len(gt_all_valid) == 0:
        print(f"No valid pixels found for {title}")
        return
        
    print(f"\n=== {title} ===")
    print(f"Total tiles analyzed: {len(tile_ids)}")
    print(f"Global stats of building pixels (>2.0m):")
    print(f"  Min: {gt_all_valid.min():.2f}m, Max: {gt_all_valid.max():.2f}m")
    print(f"  Mean: {gt_all_valid.mean():.2f}m, Median: {np.median(gt_all_valid):.2f}m")
    print(f"  P95: {np.percentile(gt_all_valid, 95):.2f}m")
    print(f"  P98: {np.percentile(gt_all_valid, 98):.2f}m")
    print(f"  P99: {np.percentile(gt_all_valid, 99):.2f}m")
    
    # Let's check how many absolute tall pixels are above P98
    print(f"Percentage of building pixels (>2.0m) exceeding absolute heights:")
    for h_thresh in [15, 20, 30, 40, 50, 100]:
        pct = np.sum(gt_all_valid > h_thresh) / len(gt_all_valid) * 100
        print(f"  > {h_thresh}m: {pct:.3f}%")

# Let's filter NewYork tiles
ny_df = df[df['city'] == 'NewYork']
print(f"\nNew York tiles count: {len(ny_df)}")
analyze_clipping(ny_df['tile_id'].tolist(), "NEW YORK CITY (TEST SET)")

# Let's filter all test tiles
test_df = df[df['split'] == 'test']
for city in test_df['city'].unique():
    city_tids = test_df[test_df['city'] == city]['tile_id'].tolist()
    analyze_clipping(city_tids, f"{city.upper()} (TEST)")
