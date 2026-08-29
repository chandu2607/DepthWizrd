import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from transformers import AutoModelForDepthEstimation

DATA_DIR = Path("data/dfc2023_multicity")
OUT_DIR = Path("runs/phase14a_depth_adaptation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_split(manifest_path, split_type):
    df = pd.read_csv(manifest_path)
    if split_type == 'train':
        df = df[df['split'] == 'train']
    else:
        df = df[df['split'] == split_type]
    return df['tile_id'].tolist()

def load_tiny_samples(tile_ids, max_samples=4):
    samples = []
    for tid in tile_ids[:max_samples]:
        rgb_path = DATA_DIR / "rgb" / tid
        dsm_path = DATA_DIR / "dsm" / tid
        rgb = cv2.imread(str(rgb_path))
        if rgb is None: continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(dsm_path), cv2.IMREAD_UNCHANGED)
        if gt is None: continue
        gt = gt.astype(np.float32)
        samples.append({"id": tid, "rgb": rgb, "gt": gt, "nodata": -999.0})
    return samples

def prep_rgb(rgb_np, target_size=518):
    # standard imagenet norm
    rgb = cv2.resize(rgb_np, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    rgb = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    return rgb.transpose(2, 0, 1)

def prep_gt(gt_np, target_size=518, nodata=-999.0):
    valid = (np.isfinite(gt_np)) & (gt_np != nodata)
    gt_f = np.where(valid, gt_np, 0.0)
    gt_r = cv2.resize(gt_f, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    valid_r = cv2.resize(valid.astype(np.float32), (target_size, target_size), interpolation=cv2.INTER_NEAREST) > 0.5
    gt_log1p = np.log1p(np.maximum(gt_r, 0.0))
    return gt_log1p, valid_r

def main():
    print("=== PHASE 14A SMOKE TEST (DECODER ONLY) ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    print(f"Loading {model_id}...")
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    
    # Freeze backbone, unfreeze rest
    trainable_modules = set()
    frozen_modules = set()
    trainable_params = 0
    frozen_params = 0
    
    for name, param in model.named_parameters():
        if name.startswith("backbone."):
            param.requires_grad = False
            frozen_params += param.numel()
            frozen_modules.add(name.split('.')[0])
        else:
            param.requires_grad = True
            trainable_params += param.numel()
            trainable_modules.add(name.split('.')[0])
            
    print(f"\nModules frozen: {sorted(list(frozen_modules))}")
    print(f"Modules trainable: {sorted(list(trainable_modules))}")
    print(f"Frozen parameters: {frozen_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {frozen_params + trainable_params:,}")
    
    # Load tiny subset
    manifest = Path("runs/dfc2023_multicity_prep/split_manifest.csv")
    train_tids = load_split(manifest, "train")
    samples = load_tiny_samples(train_tids, max_samples=4)
    
    # Batch size 2
    bs = 2
    xs, ys, ms = [], [], []
    for s in samples[:bs]:
        xs.append(prep_rgb(s['rgb']))
        y, m = prep_gt(s['gt'])
        ys.append(y)
        ms.append(m)
        
    x = torch.from_numpy(np.stack(xs)).float().to(device)
    y = torch.from_numpy(np.stack(ys)).float().to(device)
    m = torch.from_numpy(np.stack(ms)).bool().to(device)
    
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    torch.cuda.reset_peak_memory_stats()
    
    print("\nRunning smoke iterations...")
    losses = []
    
    # Check requires_grad
    print("\n[Check 4 & 5] requires_grad")
    neck_param = next(p for n, p in model.named_parameters() if "neck" in n)
    backbone_param = next(p for n, p in model.named_parameters() if "backbone" in n)
    print(f"Backbone param requires_grad: {backbone_param.requires_grad}")
    print(f"Neck param requires_grad: {neck_param.requires_grad}")
    
    for i in range(5):
        opt.zero_grad()
        
        # Forward
        out = model(x)
        pred = out.predicted_depth.unsqueeze(1) # [B, 1, H, W]
        
        if pred.shape[2:] != y.shape[1:]:
            pred = F.interpolate(pred, size=y.shape[1:], mode='bilinear', align_corners=False)
            
        pred = pred.squeeze(1) # [B, H, W]
        
        # Masked L1
        diff = torch.abs(pred - y)
        loss = diff[m].mean()
        
        # Backward
        loss.backward()
        
        if i == 0:
            print("\n[Check 6 & 7] Gradient verification (iter 0)")
            neck_grad = neck_param.grad
            backbone_grad = backbone_param.grad
            print(f"Backbone grad is None: {backbone_grad is None}")
            print(f"Neck grad is None: {neck_grad is None}")
            if neck_grad is not None:
                print(f"Neck grad mean abs: {neck_grad.abs().mean().item():.6f}")
                
        opt.step()
        
        l_val = loss.item()
        losses.append(l_val)
        print(f"Step {i+1}: masked-L1 = {l_val:.4f}")
        
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"\nPeak VRAM: {peak_mem:.1f} MB")
    
    print("\n[Check 15] Checkpoint save/load")
    ckpt_path = OUT_DIR / "smoke_test.pt"
    torch.save(model.state_dict(), ckpt_path)
    model.load_state_dict(torch.load(ckpt_path))
    print("Checkpoint saved and loaded successfully.")
    
    res = {
        "trainable_modules": sorted(list(trainable_modules)),
        "frozen_modules": sorted(list(frozen_modules)),
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "total_params": trainable_params + frozen_params,
        "losses": losses,
        "peak_vram_mb": peak_mem,
        "backbone_requires_grad": backbone_param.requires_grad,
        "neck_requires_grad": neck_param.requires_grad,
        "backbone_grad_is_none": backbone_param.grad is None,
        "neck_grad_is_none": neck_param.grad is None
    }
    
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(res, f, indent=2)
        
    print("Smoke test complete.")

if __name__ == "__main__":
    main()
