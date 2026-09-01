import os
import sys
import json
import time
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.config import TrainConfig
from depthwizard.models.building_conditioned_net import BuildingConditionedEstimator
from scripts.phase42_augment import augment_sample

MANIFEST_PATH = "runs/dfc2023_multicity_prep/split_manifest.csv"
DATA_DIR = Path("data/dfc2023_multicity")
OUT_DIR = Path("runs/phase42_augmentation")

def load_split(manifest_path, split_type):
    df = pd.read_csv(manifest_path)
    return df[df['split'] == split_type]['tile_id'].tolist()

def load_samples(tile_ids, max_samples=None):
    from depthwizard.depth.depth_anything import DepthAnythingV2
    from depthwizard.config import DepthConfig
    dcfg = DepthConfig(cache_dir=str(DATA_DIR / "depth_cache"))
    depth_model = DepthAnythingV2(dcfg.model_id, dcfg.input_size, dcfg.cache_dir, True)

    samples = []
    tids_to_load = tile_ids[:max_samples] if max_samples is not None else tile_ids
    for tid in tids_to_load:
        rgb_path = DATA_DIR / "rgb" / tid
        dsm_path = DATA_DIR / "dsm" / tid
        
        rgb = cv2.imread(str(rgb_path))
        if rgb is None: continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        gt = cv2.imread(str(dsm_path), cv2.IMREAD_UNCHANGED)
        if gt is None: continue
        gt = gt.astype(np.float32)
        
        depth = depth_model.infer(rgb, tid, target_hw=rgb.shape[:2])
        mask_bldg = (gt > 2.0).astype(np.uint8)
        
        samples.append({"rgb": rgb, "gt": gt, "depth": depth, "mask_bldg": mask_bldg, "nodata": -999.0})
    return samples

def evaluate_masks(preds, gts):
    ious, dices, precs, recs = [], [], [], []
    for pred, gt in zip(preds, gts):
        intersection = (pred & gt).sum()
        union = (pred | gt).sum()
        if union > 0:
            ious.append(intersection / union)
            dices.append(2 * intersection / (pred.sum() + gt.sum() + 1e-6))
            precs.append(intersection / (pred.sum() + 1e-6))
            recs.append(intersection / (gt.sum() + 1e-6))
    return np.mean(ious), np.mean(dices), np.mean(precs), np.mean(recs)

def main():
    print("================ PHASE 42: U-NET FOOTPRINT AUGMENTATION ABLATION ================")
    
    train_ids = load_split(MANIFEST_PATH, 'train')
    val_ids = load_split(MANIFEST_PATH, 'val')
    test_ids = load_split(MANIFEST_PATH, 'test')
    
    print("Loading samples...")
    train_samples = load_samples(train_ids, max_samples=64)
    val_samples = load_samples(val_ids, max_samples=32)
    test_samples = load_samples(test_ids, max_samples=32)
    
    rng = np.random.default_rng(42)
    results = []
    
    for config_mode in ['A', 'B', 'C', 'D']:
        print(f"\\n--- Running UNET CONFIGURATION {config_mode} ---")
        torch.manual_seed(42)
        tcfg = TrainConfig(arch="unet3", target_transform="none", epochs=3, batch_size=8, lr=1e-3, amp=True)
        estimator = BuildingConditionedEstimator(tcfg, nodata=-999.0, seed=42)
        model = estimator.model
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        res = 256
        for epoch in range(3):
            model.train()
            for s_base in train_samples:
                s = augment_sample(s_base, config_mode, rng)
                x = estimator._prep_x(s, res)
                xt = torch.from_numpy(x[None]).float().to(estimator.device)
                d = cv2.resize(s["depth"], (res, res))
                dt = torch.from_numpy(d[None]).float().to(estimator.device)
                m = cv2.resize(s["mask_bldg"], (res, res))
                mt = torch.from_numpy(m[None]).float().to(estimator.device)
                
                mask_logits, _, _, _, _ = model(xt, dt, device=estimator.device)
                loss_footprint = F.binary_cross_entropy_with_logits(mask_logits.squeeze(1), mt)
                
                opt.zero_grad()
                loss_footprint.backward()
                opt.step()
                
        # Evaluate
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for s in test_samples:
                x = estimator._prep_x(s, res)
                xt = torch.from_numpy(x[None]).float().to(estimator.device)
                d = cv2.resize(s["depth"], (res, res))
                dt = torch.from_numpy(d[None]).float().to(estimator.device)
                m = cv2.resize(s["mask_bldg"], (res, res))
                
                mask_logits, _, _, _, _ = model(xt, dt, device=estimator.device)
                p = (torch.sigmoid(mask_logits).squeeze().cpu().numpy() > 0.5).astype(np.uint8)
                preds.append(p)
                gts.append(m)
                
        iou, dice, prec, rec = evaluate_masks(preds, gts)
        print(f"  Test IoU: {iou:.4f} | Dice: {dice:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
        results.append({
            "Config": config_mode,
            "IoU": iou, "Dice": dice, "Precision": prec, "Recall": rec
        })
        
    pd.DataFrame(results).to_csv(OUT_DIR / "building_model_results.csv", index=False)
    print("Done. Saved UNet results.")

if __name__ == "__main__":
    main()
