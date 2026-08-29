import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from transformers import AutoModelForDepthEstimation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.eval.evaluate import evaluate_estimator

DATA_DIR = Path("data/dfc2023_multicity")
OUT_DIR = Path("runs/phase14_depth_decoder_adapt")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_split(manifest_path, split_type):
    df = pd.read_csv(manifest_path)
    if split_type == 'train':
        df = df[df['split'] == 'train']
    else:
        df = df[df['split'] == split_type]
    return df['tile_id'].tolist()

def load_data_in_memory(tile_ids):
    samples = []
    for tid in tile_ids:
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

def train_epoch(model, opt, scaler, samples, bs, device, use_amp):
    model.train()
    rng = np.random.default_rng()
    order = rng.permutation(len(samples))
    ep_loss, nb = 0.0, 0
    
    for i in range(0, len(order), bs):
        idx = order[i : i + bs]
        xs, ys, ms = [], [], []
        for j in idx:
            s = samples[j]
            xs.append(prep_rgb(s['rgb']))
            y, m = prep_gt(s['gt'])
            ys.append(y)
            ms.append(m)
            
        x = torch.from_numpy(np.stack(xs)).float().to(device)
        y = torch.from_numpy(np.stack(ys)).float().to(device)
        m = torch.from_numpy(np.stack(ms)).bool().to(device)
        
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(x)
            pred = out.predicted_depth.unsqueeze(1)
            if pred.shape[2:] != y.shape[1:]:
                pred = F.interpolate(pred, size=y.shape[1:], mode='bilinear', align_corners=False)
            pred = pred.squeeze(1)
            
            diff = torch.abs(pred - y)
            loss = diff[m].mean()
            
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        
        ep_loss += float(loss.detach())
        nb += 1
        
    return ep_loss / max(nb, 1)

@torch.no_grad()
def eval_loss(model, samples, bs, device):
    model.eval()
    ep_loss, nb = 0.0, 0
    for i in range(0, len(samples), bs):
        idx = range(i, min(i + bs, len(samples)))
        xs, ys, ms = [], [], []
        for j in idx:
            s = samples[j]
            xs.append(prep_rgb(s['rgb']))
            y, m = prep_gt(s['gt'])
            ys.append(y)
            ms.append(m)
            
        x = torch.from_numpy(np.stack(xs)).float().to(device)
        y = torch.from_numpy(np.stack(ys)).float().to(device)
        m = torch.from_numpy(np.stack(ms)).bool().to(device)
        
        out = model(x)
        pred = out.predicted_depth.unsqueeze(1)
        if pred.shape[2:] != y.shape[1:]:
            pred = F.interpolate(pred, size=y.shape[1:], mode='bilinear', align_corners=False)
        pred = pred.squeeze(1)
        
        diff = torch.abs(pred - y)
        loss = diff[m].mean()
        
        ep_loss += float(loss)
        nb += 1
        
    return ep_loss / max(nb, 1)

@torch.no_grad()
def predict_full(model, rgb_np, original_shape, device):
    model.eval()
    x = prep_rgb(rgb_np)
    x = torch.from_numpy(x).unsqueeze(0).float().to(device)
    out = model(x)
    pred = out.predicted_depth.unsqueeze(1)
    
    pred = F.interpolate(pred, size=original_shape, mode='bilinear', align_corners=False)
    pred = pred.squeeze().cpu().numpy()
    pred = np.expm1(pred)
    pred = np.maximum(pred, 0.0)
    return pred

def render_comparison(rgb, gt, pred, err, out_path, vmax=45.0):
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(rgb)
    plt.title("RGB")
    plt.axis("off")
    
    plt.subplot(1, 4, 2)
    plt.imshow(np.where(np.isfinite(gt) & (gt != -999.0), gt, -1), cmap="nipy_spectral", vmin=0, vmax=vmax)
    plt.title("GT Height")
    plt.axis("off")
    
    plt.subplot(1, 4, 3)
    plt.imshow(pred, cmap="nipy_spectral", vmin=0, vmax=vmax)
    plt.title("Predicted Height")
    plt.axis("off")
    
    plt.subplot(1, 4, 4)
    plt.imshow(err, cmap="bwr", vmin=-20, vmax=20)
    plt.title("Error (Red=Under, Blue=Over)")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def main():
    manifest = Path("runs/dfc2023_multicity_prep/split_manifest.csv")
    train_tids = load_split(manifest, "train")
    val_tids = load_split(manifest, "val")
    test_tids = load_split(manifest, "test")
    
    print("Loading datasets...")
    train_samples = load_data_in_memory(train_tids)
    val_samples = load_data_in_memory(val_tids)
    test_samples = load_data_in_memory(test_tids)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 15
    bs = 4
    lr = 1e-4
    use_amp = (device == "cuda")
    
    all_results = []
    
    for seed in [1]:
        print(f"\n{'='*40}")
        print(f"SEED {seed} - PHASE 14B")
        print(f"{'='*40}")
        
        seed_dir = OUT_DIR / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "figures").mkdir(parents=True, exist_ok=True)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model_id = "depth-anything/Depth-Anything-V2-Small-hf"
        model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
        
        trainable_modules, frozen_modules = set(), set()
        t_params, f_params = 0, 0
        for name, param in model.named_parameters():
            if name.startswith("backbone."):
                param.requires_grad = False
                f_params += param.numel()
                frozen_modules.add(name.split('.')[0])
            else:
                param.requires_grad = True
                t_params += param.numel()
                trainable_modules.add(name.split('.')[0])
                
        print(f"Trainable params: {t_params:,} | Frozen params: {f_params:,}")
        
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
        best_val = float('inf')
        best_epoch = -1
        ckpt_path = seed_dir / "best_model.pt"
        
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        
        for ep in range(epochs):
            ep_t0 = time.time()
            train_l1 = train_epoch(model, opt, scaler, train_samples, bs, device, use_amp)
            val_l1 = eval_loss(model, val_samples, bs, device)
            ep_time = time.time() - ep_t0
            
            print(f"[Seed {seed}] Epoch {ep+1}/{epochs} | Train L1: {train_l1:.4f} | Val L1: {val_l1:.4f} | Time: {ep_time:.1f}s")
            
            if val_l1 < best_val:
                best_val = val_l1
                best_epoch = ep + 1
                torch.save(model.state_dict(), ckpt_path)
                
        total_time = time.time() - t0
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Finished seed {seed}. Best epoch: {best_epoch}. Peak VRAM: {peak_vram:.1f} MB.")
        
    print("Training finished.")

if __name__ == "__main__":
    main()
