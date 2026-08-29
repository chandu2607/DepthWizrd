import os
import sys
import pandas as pd
import numpy as np
import time
from pathlib import Path
import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.models.fusion_head import LearnedFusionHead
from depthwizard.config import TrainConfig


MANIFEST_PATH = "runs/dfc2023_multicity_prep/split_manifest.csv"
DATA_DIR = Path("data/dfc2023_multicity")

def load_tiny_split():
    df = pd.read_csv(MANIFEST_PATH)
    train_ids = df[df['split'] == 'train'].head(4)['tile_id'].tolist()
    val_ids = df[df['split'] == 'val'].head(2)['tile_id'].tolist()
    return train_ids, val_ids

def load_sample(tile_id):
    rgb_path = DATA_DIR / "rgb" / tile_id
    dsm_path = DATA_DIR / "dsm" / tile_id
    depth_path = DATA_DIR / "depth_cache" / f"{tile_id}.npy"
    
    # Read RGB
    rgb = cv2.imread(str(rgb_path))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    # Read GT
    gt = cv2.imread(str(dsm_path), cv2.IMREAD_UNCHANGED)
    if gt is None:
        raise FileNotFoundError(f"Could not read {dsm_path}")
    gt = gt.astype(np.float32)
            
    # Read Depth Cache (Depth Anything creates a hash name in real config, but here it's named tile_id.npy in my script? Wait, I used depth_model.infer with key=t, which saves it as hash. Let me load it correctly).
    # Ah, the prepare script used DepthAnythingV2 which hashes the key!
    # I should instantiate DepthAnythingV2 with use_cache=True and call infer to fetch it.
    from depthwizard.depth.depth_anything import DepthAnythingV2
    from depthwizard.config import DepthConfig
    dcfg = DepthConfig(cache_dir=str(DATA_DIR / "depth_cache"))
    depth_model = DepthAnythingV2(dcfg.model_id, dcfg.input_size, dcfg.cache_dir, True)
    
    depth = depth_model.infer(rgb, tile_id, target_hw=rgb.shape[:2])
    
    return {"id": tile_id, "rgb": rgb, "gt": gt, "depth": depth, "nodata": -999.0}

def main():
    print("Loading tiny dataset...")
    train_ids, val_ids = load_tiny_split()
    train_samples = [load_sample(tid) for tid in train_ids]
    val_samples = [load_sample(tid) for tid in val_ids]
    
    print("Setting up C_log1p model...")
    tcfg = TrainConfig(
        arch="unet3",
        target_transform="log1p",
        loss_type="standard",
        epochs=3,
        batch_size=2,
        lr=1e-3,
        amp=True
    )
    
    model = LearnedFusionHead(tcfg, nodata=-999.0, seed=42)
    print("Model initialized. VRAM before fit:", torch.cuda.memory_allocated()/(1024**2), "MB")
    
    # Run fit
    print("Running training loop (smoke test)...")
    t0 = time.time()
    
    # We will manually capture loss to verify it decreases
    # We can rely on the model.fit but let's intercept printed output or just check train runs.
    model.fit(train_samples)
    t_train = time.time() - t0
    
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Training finished in {t_train:.2f}s. Peak VRAM: {peak_vram:.2f} MB")
    
    # Verification
    print("Running validation inference...")
    val_sample = val_samples[0]
    pred = model.predict(val_sample)
    
    print(f"Output shape: {pred.shape}, GT shape: {val_sample['gt'].shape}")
    assert pred.shape == val_sample['gt'].shape[:2]
    
    nan_inf = np.isnan(pred).any() or np.isinf(pred).any()
    print(f"NaN/Inf in prediction: {nan_inf}")
    assert not nan_inf
    
    print("Smoke test successfully completed.")
    
if __name__ == "__main__":
    main()
