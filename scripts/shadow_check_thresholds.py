import cv2
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/dfc2023_multicity")
manifest_path = Path("runs/dfc2023_multicity_prep/split_manifest.csv")
df = pd.read_csv(manifest_path)
ny_tids = df[df['city'] == 'NewYork']['tile_id'].tolist()

print("Auditing shadows in New York with multiple thresholds:")
for tid in ny_tids[:5]:
    rgb_path = DATA_DIR / "rgb" / tid
    rgb = cv2.imread(str(rgb_path))
    if rgb is None:
        continue
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    frac_40 = np.sum(gray < 40) / gray.size * 100
    frac_60 = np.sum(gray < 60) / gray.size * 100
    frac_80 = np.sum(gray < 80) / gray.size * 100
    frac_100 = np.sum(gray < 100) / gray.size * 100
    
    print(f"  Tile {tid}: Mean={gray.mean():.1f} | <40: {frac_40:.2f}% | <60: {frac_60:.2f}% | <80: {frac_80:.2f}% | <100: {frac_100:.2f}%")
