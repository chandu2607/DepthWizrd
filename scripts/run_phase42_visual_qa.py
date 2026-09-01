import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys

sys.path.insert(0, ".")
from depthwizard.data.datasets import load_sample, Record
from scripts.phase42_augment import augment_sample

OUT_DIR = Path("runs/phase42_augmentation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fake a record
rec = Record(
    tile_id="NewYork_1.tif",
    city="NewYork",
    rgb_path="data/dfc2023_multicity/rgb/JAX_001_RGB.tif", # we'll just mock one
    agl_path="data/dfc2023_multicity/dsm/JAX_001_AGL.tif",
    cls_path=None
)

def create_mock_sample():
    # Since we can't easily grab a specific dataset image dynamically without paths,
    # let's find the first valid training tile
    import pandas as pd
    manifest_path = "runs/dfc2023_multicity_prep/split_manifest.csv"
    if not os.path.exists(manifest_path):
        # Generate synthetic fallback
        rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.rectangle(rgb, (50, 50), (150, 150), (100, 100, 100), -1)
        gt = np.zeros((256, 256), dtype=np.float32)
        cv2.rectangle(gt, (50, 50), (150, 150), 30.0, -1)
        depth = np.zeros((256, 256), dtype=np.float32)
        cv2.rectangle(depth, (50, 50), (150, 150), 5.0, -1)
        mask_bldg = (gt > 2.0)
        return {"rgb": rgb, "gt": gt, "depth": depth, "nodata": -999.0, "mask_bldg": mask_bldg, "city": "Mock"}
        
    df = pd.read_csv(manifest_path)
    tid = df[df['split']=='train'].iloc[0]['tile_id']
    rgb_path = f"data/dfc2023_multicity/rgb/{tid}"
    dsm_path = f"data/dfc2023_multicity/dsm/{tid}"
    
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    gt = cv2.imread(dsm_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    
    # generate a dummy depth for visual QA
    depth = gt * 0.1
    mask_bldg = (gt > 5.0).astype(np.uint8)
    
    return {"rgb": rgb, "gt": gt, "depth": depth, "nodata": -999.0, "mask_bldg": mask_bldg, "city": "Test"}

def save_visual_qa(s, name):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(s["rgb"])
    axes[0].set_title("RGB")
    axes[1].imshow(s["gt"], cmap="turbo")
    axes[1].set_title("DSM")
    axes[2].imshow(s["gt"], cmap="turbo") # dtm dummy
    axes[2].set_title("DTM (mock)")
    axes[3].imshow(s["depth"], cmap="plasma")
    axes[3].set_title("Depth")
    axes[4].imshow(s["mask_bldg"], cmap="gray")
    axes[4].set_title("Building Mask")
    
    for ax in axes:
        ax.axis("off")
        
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}.png")
    plt.close()

def main():
    s = create_mock_sample()
    rng = np.random.default_rng(42)
    
    # Original
    save_visual_qa(s, "original")
    
    # Geometric
    s_geom = augment_sample(s, 'B', rng)
    save_visual_qa(s_geom, "geometric")
    
    # Photometric
    s_photo = augment_sample(s, 'C', rng)
    save_visual_qa(s_photo, "photometric")
    
    # Multiscale
    s_multi = augment_sample(s, 'D', rng)
    save_visual_qa(s_multi, "multiscale")
    
    # Resolution (2x, 4x, 8x)
    for scale in [2, 4, 8]:
        s_res = s.copy()
        h, w = s["gt"].shape[:2]
        sh, sw = h // scale, w // scale
        coarse = cv2.resize(s["gt"], (sw, sh), interpolation=cv2.INTER_AREA)
        s_res["gt"] = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
        save_visual_qa(s_res, f"resolution_{scale}x")
        
    print("Visual QA Generation Complete.")

if __name__ == "__main__":
    main()
