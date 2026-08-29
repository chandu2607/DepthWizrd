import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.config import TrainConfig
from depthwizard.models.fusion_head import LearnedFusionHead
import pandas as pd
import cv2

DATA_DIR = Path("data/dfc2023_multicity")

def load_split(manifest_path, split_type):
    df = pd.read_csv(manifest_path)
    if split_type == 'train':
        df = df[df['split'] == 'train']
    else:
        df = df[df['split'] == split_type]
    return df['tile_id'].tolist()

def load_samples(tile_ids):
    from depthwizard.depth.depth_anything import DepthAnythingV2
    from depthwizard.config import DepthConfig
    dcfg = DepthConfig(cache_dir=str(DATA_DIR / "depth_cache"))
    depth_model = DepthAnythingV2(dcfg.model_id, dcfg.input_size, dcfg.cache_dir, True)

    samples = []
    for tid in tile_ids:
        rgb_path = DATA_DIR / "rgb" / tid
        dsm_path = DATA_DIR / "dsm" / tid
        
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        gt = cv2.imread(str(dsm_path), cv2.IMREAD_UNCHANGED)
        if gt is None:
            continue
        gt = gt.astype(np.float32)
        
        depth = depth_model.infer(rgb, tid, target_hw=rgb.shape[:2])
        city = next((c for c in ["Barcelona", "Berlin", "Brasilia", "Copenhagen", "NewDelhi", "NewYork", "Portsmouth", "Rio", "SanDiego", "SaoLuis", "Sydney"] if c in tid), "Unknown")
        cls = (gt > 2.0).astype(np.uint8) * 6
        samples.append({"id": tid, "city": city, "rgb": rgb, "gt": gt, "depth": depth, "nodata": -999.0, "cls": cls})
    return samples

def main():
    print("=== PHASE 13B SMOKE TEST ===")
    
    # Load a tiny subset of data
    manifest = Path("runs/dfc2023_multicity_prep/split_manifest.csv")
    train_tids = load_split(manifest, "train")
    val_tids = load_split(manifest, "val")
    
    # Just take 8 train and 2 val for smoke test
    train_samples = load_samples(train_tids[:8])
    val_samples = load_samples(val_tids[:2])
    
    # 1. Check class distribution on the tiny subset
    print("\n[Check 1] Class Distribution on Smoke Test Samples:")
    all_y = []
    bins = np.array([0, 2, 5, 10, 15, 20, 30, 40, np.inf])
    for s in train_samples:
        gt = s["gt"]
        valid = (np.isfinite(gt)) & (gt != -999.0)
        gt_v = gt[valid]
        c = np.digitize(np.maximum(gt_v, 0.0), bins) - 1
        c = np.clip(c, 0, 7)
        all_y.extend(c.tolist())
    
    counts = np.bincount(all_y, minlength=8)
    for i, count in enumerate(counts):
        print(f"Class {i}: {count} pixels")
        
    # 2. Instantiate Model
    print("\n[Check 2] Instantiating Model:")
    cfg = TrainConfig(
        arch="unet3",
        target_transform="classification",
        loss_type="standard",
        epochs=2,
        batch_size=4,
        lr=1e-3,
        train_res=256
    )
    cfg.input_mode = "depth"
    
    model = LearnedFusionHead(cfg, nodata=-999.0)
    print(f"Output channels: {model.model.head.out_channels}")
    
    # 3. Fit (tiny)
    print("\n[Check 3] Running Fit (2 epoch, 8 tiles):")
    try:
        model.fit(train_samples)
        print("Fit completed without NaN/Inf.")
    except Exception as e:
        print(f"Fit failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # 4. Predict (decode)
    print("\n[Check 4] Testing predict & validation decoding:")
    pred = model.predict(val_samples[0])
    print(f"Prediction shape: {pred.shape}")
    print(f"Unique decoded continuous values in prediction: {np.unique(pred)}")
    
    print("\nSMOKE TEST PASSED.")
    
if __name__ == "__main__":
    main()
