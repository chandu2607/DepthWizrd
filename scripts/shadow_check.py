import cv2
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/dfc2023_multicity")
manifest_path = Path("runs/dfc2023_multicity_prep/split_manifest.csv")
df = pd.read_csv(manifest_path)

# Filter NewYork tiles
ny_tids = df[df['city'] == 'NewYork']['tile_id'].tolist()

print(f"Auditing shadows in New York ({len(ny_tids)} tiles):")
shadow_pixels_fractions = []
for tid in ny_tids[:10]:
    rgb_path = DATA_DIR / "rgb" / tid
    dsm_path = DATA_DIR / "dsm" / tid
    
    rgb = cv2.imread(str(rgb_path))
    if rgb is None:
        continue
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    # Calculate grayscale intensity
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    # Shadows are typically very dark (e.g. intensity < 40 in 8-bit)
    # Let's count the fraction of pixels below different thresholds
    shadow_mask = gray < 40
    fraction = np.sum(shadow_mask) / shadow_mask.size
    shadow_pixels_fractions.append(fraction)
    
    # Let's check the height map associated with this tile
    gt = cv2.imread(str(dsm_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    valid = (np.isfinite(gt)) & (gt != -999.0)
    max_h = gt[valid].max() if np.any(valid) else 0.0
    
    print(f"  Tile {tid}: Max Height = {max_h:.1f}m | Grayscale Mean = {gray.mean():.1f} | Shadow Fraction (<40) = {fraction*100:.2f}%")

print(f"\nAverage shadow pixel fraction (<40) in NY: {np.mean(shadow_pixels_fractions)*100:.2f}%")
