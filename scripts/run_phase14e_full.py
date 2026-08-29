import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from transformers import AutoModelForDepthEstimation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.models.fusion_head import SmallFusionUNet
from depthwizard.metrics.height_metrics import compute_metrics, compute_binned_metrics, aggregate_scene_metrics, aggregate_binned, compute_class_metrics

DATA_DIR = Path("data/dfc2023_multicity")
OUT_DIR = Path("runs/phase14e_vit_unfreeze")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_split(manifest_path, split_type):
    df = pd.read_csv(manifest_path)
    if split_type == 'train':
        df = df[df['split'] == 'train']
    else:
        df = df[df['split'] == split_type]
    return df['tile_id'].tolist()

def load_data_in_memory(tile_ids, load_cls=False):
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
        s = {"id": tid, "rgb": rgb, "gt": gt, "nodata": -999.0}
        
        if load_cls:
            cls_path = DATA_DIR / "cls" / tid
            if cls_path.exists():
                cls = cv2.imread(str(cls_path), cv2.IMREAD_UNCHANGED)
                if cls is not None:
                    s["cls"] = cls
        samples.append(s)
    return samples

def prep_rgb_dav2(rgb_np, target_size=518):
    rgb = cv2.resize(rgb_np, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    rgb = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    return rgb.transpose(2, 0, 1)

def prep_rgb_unet(rgb_np, target_size=256):
    rgb = rgb_np.astype(np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    rgb = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return rgb.transpose(2, 0, 1)

def prep_gt(gt_np, target_size=256, nodata=-999.0):
    valid = (np.isfinite(gt_np)) & (gt_np != nodata)
    gt_f = np.where(valid, gt_np, 0.0)
    gt_r = cv2.resize(gt_f, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    valid_r = cv2.resize(valid.astype(np.float32), (target_size, target_size), interpolation=cv2.INTER_NEAREST) > 0.5
    gt_log1p = np.log1p(np.maximum(gt_r, 0.0))
    return gt_log1p, valid_r

class Phase14EModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.da_v2 = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        
        self.f_params = 0
        self.t_params = 0
        self.trainable_modules = set()
        self.frozen_modules = set()
        for name, param in self.da_v2.named_parameters():
            if "backbone.encoder.layer.11" in name or "backbone.layernorm" in name:
                param.requires_grad = True
                self.t_params += param.numel()
                self.trainable_modules.add(name.split('.')[0])
            elif name.startswith("backbone."):
                param.requires_grad = False
                self.f_params += param.numel()
                self.frozen_modules.add(name.split('.')[0])
            else:
                param.requires_grad = True
                self.t_params += param.numel()
                self.trainable_modules.add(name.split('.')[0])
                
        self.unet = SmallFusionUNet(w=24, in_channels=4, out_channels=1)
        for name, param in self.unet.named_parameters():
            self.t_params += param.numel()
            self.trainable_modules.add("unet." + name.split('.')[0])
            
        self.register_buffer("d_mean", torch.tensor(0.0))
        self.register_buffer("d_std", torch.tensor(1.0))

    def forward(self, rgb_dav2, rgb_unet):
        da_out = self.da_v2(rgb_dav2).predicted_depth.unsqueeze(1)
        depth_256 = F.interpolate(da_out, size=rgb_unet.shape[-2:], mode='bilinear', align_corners=False)
        depth_256 = (depth_256 - self.d_mean) / (self.d_std + 1e-6)
        x = torch.cat([rgb_unet, depth_256], dim=1)
        return self.unet(x)
        
    def predict_full(self, rgb_np, original_shape, device):
        self.eval()
        rgb_d = torch.from_numpy(prep_rgb_dav2(rgb_np)).unsqueeze(0).to(device)
        rgb_u = torch.from_numpy(prep_rgb_unet(rgb_np)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = self(rgb_d, rgb_u)
            
        pred = F.interpolate(pred.unsqueeze(1), size=original_shape, mode='bilinear', align_corners=False)
        pred = pred.squeeze().cpu().numpy()
        pred = np.expm1(pred)
        pred = np.maximum(pred, 0.0)
        return pred

class Phase14DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.da_v2 = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        for name, param in self.da_v2.named_parameters():
            if name.startswith("backbone."):
                param.requires_grad = False
            else:
                param.requires_grad = True
        self.unet = SmallFusionUNet(w=24, in_channels=4, out_channels=1)
        self.register_buffer("d_mean", torch.tensor(0.0))
        self.register_buffer("d_std", torch.tensor(1.0))

    def forward(self, rgb_dav2, rgb_unet):
        da_out = self.da_v2(rgb_dav2).predicted_depth.unsqueeze(1)
        depth_256 = F.interpolate(da_out, size=rgb_unet.shape[-2:], mode='bilinear', align_corners=False)
        depth_256 = (depth_256 - self.d_mean) / (self.d_std + 1e-6)
        x = torch.cat([rgb_unet, depth_256], dim=1)
        return self.unet(x)
        
    def predict_full(self, rgb_np, original_shape, device):
        self.eval()
        rgb_d = torch.from_numpy(prep_rgb_dav2(rgb_np)).unsqueeze(0).to(device)
        rgb_u = torch.from_numpy(prep_rgb_unet(rgb_np)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = self(rgb_d, rgb_u)
        pred = F.interpolate(pred.unsqueeze(1), size=original_shape, mode='bilinear', align_corners=False)
        pred = pred.squeeze().cpu().numpy()
        pred = np.expm1(pred)
        return np.maximum(pred, 0.0)

def train_epoch(model, opt, scaler, samples, bs, device, use_amp):
    model.train()
    rng = np.random.default_rng()
    order = rng.permutation(len(samples))
    ep_loss, nb = 0.0, 0
    
    for i in range(0, len(order), bs):
        idx = order[i : i + bs]
        xd, xu, ys, ms = [], [], [], []
        for j in idx:
            s = samples[j]
            xd.append(prep_rgb_dav2(s['rgb']))
            xu.append(prep_rgb_unet(s['rgb']))
            y, m = prep_gt(s['gt'])
            ys.append(y)
            ms.append(m)
            
        rgb_d = torch.from_numpy(np.stack(xd)).float().to(device)
        rgb_u = torch.from_numpy(np.stack(xu)).float().to(device)
        y = torch.from_numpy(np.stack(ys)).float().to(device)
        m = torch.from_numpy(np.stack(ms)).bool().to(device)
        
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            pred = model(rgb_d, rgb_u)
            if pred.shape != y.shape:
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
        xd, xu, ys, ms = [], [], [], []
        for j in idx:
            s = samples[j]
            xd.append(prep_rgb_dav2(s['rgb']))
            xu.append(prep_rgb_unet(s['rgb']))
            y, m = prep_gt(s['gt'])
            ys.append(y)
            ms.append(m)
            
        rgb_d = torch.from_numpy(np.stack(xd)).float().to(device)
        rgb_u = torch.from_numpy(np.stack(xu)).float().to(device)
        y = torch.from_numpy(np.stack(ys)).float().to(device)
        m = torch.from_numpy(np.stack(ms)).bool().to(device)
        
        pred = model(rgb_d, rgb_u)
        if pred.shape != y.shape:
            pred = pred.squeeze(1)
        diff = torch.abs(pred - y)
        loss = diff[m].mean()
        
        ep_loss += float(loss)
        nb += 1
        
    return ep_loss / max(nb, 1)

def render_comparison(rgb, gt, pred_base, err_base, pred_adapt, err_adapt, out_path, vmax=45.0):
    plt.figure(figsize=(24, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(rgb)
    plt.title("RGB")
    plt.axis("off")
    
    plt.subplot(2, 3, 2)
    plt.imshow(np.where(np.isfinite(gt) & (gt != -999.0), gt, -1), cmap="nipy_spectral", vmin=0, vmax=vmax)
    plt.title("GT Height")
    plt.axis("off")
    
    plt.subplot(2, 3, 4)
    plt.imshow(pred_base, cmap="nipy_spectral", vmin=0, vmax=vmax)
    plt.title("Phase 14D (Baseline) Prediction")
    plt.axis("off")
    
    plt.subplot(2, 3, 5)
    plt.imshow(err_base, cmap="bwr", vmin=-20, vmax=20)
    plt.title("Baseline Error (Red=Under, Blue=Over)")
    plt.axis("off")
    
    plt.subplot(2, 3, 3)
    plt.imshow(pred_adapt, cmap="nipy_spectral", vmin=0, vmax=vmax)
    plt.title("Phase 14E (Adapted) Prediction")
    plt.axis("off")
    
    plt.subplot(2, 3, 6)
    plt.imshow(err_adapt, cmap="bwr", vmin=-20, vmax=20)
    plt.title("Adapted Error (Red=Under, Blue=Over)")
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
    train_samples = load_data_in_memory(train_tids, load_cls=False)
    val_samples = load_data_in_memory(val_tids, load_cls=False)
    test_samples = load_data_in_memory(test_tids, load_cls=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 15
    bs = 4
    lr = 1e-4
    use_amp = (device == "cuda")
    
    all_results = []
    if (OUT_DIR / "results.json").exists():
        with open(OUT_DIR / "results.json", "r") as f:
            all_results = json.load(f)
            
    bin_edges = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, np.inf]
    
    # Run Seeds 0 and 1
    for seed in [0, 1]:
        if any(r['seed'] == seed for r in all_results):
            print(f"Skipping Seed {seed} as it already exists in results.json.")
            continue
            
        print(f"\n{'='*40}")
        print(f"SEED {seed} - PHASE 14E")
        print(f"{'='*40}")
        
        seed_dir = OUT_DIR / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "figures").mkdir(parents=True, exist_ok=True)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = Phase14EModel().to(device)
        print(f"Trainable params: {model.t_params:,} | Frozen params: {model.f_params:,}")
        
        print("Initializing depth stats from training set...")
        model.eval()
        ds = []
        with torch.no_grad():
            for i in range(min(64, len(train_samples))):
                rgb_d = torch.from_numpy(prep_rgb_dav2(train_samples[i]['rgb'])).unsqueeze(0).to(device)
                d_out = model.da_v2(rgb_d).predicted_depth
                ds.append(d_out.cpu().numpy().ravel()[::37])
        allc = np.concatenate(ds)
        model.d_mean.fill_(float(np.mean(allc)))
        model.d_std.fill_(float(np.std(allc) + 1e-6))
        
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        
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
        
        # Load Phase 14D baseline for visual comparison
        baseline_model = Phase14DModel().to(device)
        baseline_ckpt = Path(f"runs/phase14d_correct_decoder_adapt/seed_{seed}/best_model.pt")
        if baseline_ckpt.exists():
            baseline_model.load_state_dict(torch.load(baseline_ckpt))
        else:
            print("WARNING: Baseline checkpoint not found, baseline visualizations will be untrained.")
        
        # Evaluate
        print(f"Evaluating Seed {seed} on New York (Test Set)...")
        model.load_state_dict(torch.load(ckpt_path))
        
        per_scene = []
        binned_scenes = []
        
        # We need to collect raw predictions to compute overall Max, P95, P99
        all_valid_preds = []
        
        for i, s in enumerate(test_samples):
            pred_adapt = model.predict_full(s['rgb'], s['gt'].shape[:2], device)
            
            # metrics
            cls_mask = s.get('cls', None)
            metrics = compute_class_metrics(pred_adapt, s['gt'], cls_mask, building_label=6, nodata=-999.0)
            binned = compute_binned_metrics(pred_adapt, s['gt'], edges=bin_edges, nodata=-999.0)
            
            per_scene.append(metrics)
            binned_scenes.append(binned)
            
            valid = (np.isfinite(s['gt'])) & (s['gt'] != -999.0)
            all_valid_preds.append(pred_adapt[valid])
            
            # Visualization on difficult New York scenes (>30m max)
            if "NewYork" in s["id"] and s["gt"][valid].max() > 30 and i < 20:
                pred_base = baseline_model.predict_full(s['rgb'], s['gt'].shape[:2], device)
                
                err_base = np.zeros_like(pred_base)
                err_base[valid] = pred_base[valid] - s["gt"][valid]
                err_base[~valid] = 0.0
                
                err_adapt = np.zeros_like(pred_adapt)
                err_adapt[valid] = pred_adapt[valid] - s["gt"][valid]
                err_adapt[~valid] = 0.0
                
                render_comparison(s['rgb'], s['gt'], pred_base, err_base, pred_adapt, err_adapt, seed_dir / f"figures/vis_{s['id'].replace('.tif', '')}.png")
                
        agg_all = aggregate_scene_metrics([p["all"] for p in per_scene])
        agg_building = aggregate_scene_metrics([p["building"] for p in per_scene if "building" in p])
        agg_metrics = {"all": agg_all, "building": agg_building}
        agg_binned = aggregate_binned(binned_scenes)
        
        flat_preds = np.concatenate(all_valid_preds)
        extra_metrics = {
            "predicted_max": float(np.max(flat_preds)),
            "predicted_p95": float(np.percentile(flat_preds, 95)),
            "predicted_p99": float(np.percentile(flat_preds, 99))
        }
        
        res_dict = {
            "seed": seed,
            "trainable_modules": sorted(list(model.trainable_modules)),
            "trainable_params": model.t_params,
            "frozen_params": model.f_params,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "runtime_s": total_time,
            "peak_vram_mb": peak_vram,
            "agg_metrics": agg_metrics,
            "binned": agg_binned,
            "extra_metrics": extra_metrics
        }
        all_results.append(res_dict)
        
        with open(OUT_DIR / "results.json", "w") as f:
            json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
