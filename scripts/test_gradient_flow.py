import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.models.building_conditioned_net import BuildingConditionedEstimator

MANIFEST_PATH = "runs/dfc2023_multicity_prep/split_manifest.csv"
DATA_DIR = Path("data/dfc2023_multicity")
OUT_DIR = Path("runs/phase22b_gradient_audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_tiny_split():
    df = pd.read_csv(MANIFEST_PATH)
    return df[df['split'] == 'train'].head(2)['tile_id'].tolist()

def load_sample(tile_id):
    rgb_path = DATA_DIR / "rgb" / tile_id
    dsm_path = DATA_DIR / "dsm" / tile_id
    
    rgb = cv2.imread(str(rgb_path))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    gt = cv2.imread(str(dsm_path), cv2.IMREAD_UNCHANGED)
    gt = gt.astype(np.float32)
            
    from depthwizard.depth.depth_anything import DepthAnythingV2
    from depthwizard.config import DepthConfig
    dcfg = DepthConfig(cache_dir=str(DATA_DIR / "depth_cache"))
    depth_model = DepthAnythingV2(dcfg.model_id, dcfg.input_size, dcfg.cache_dir, True)
    depth = depth_model.infer(rgb, tile_id, target_hw=rgb.shape[:2])
    
    return {"id": tile_id, "rgb": rgb, "gt": gt, "depth": depth, "nodata": -999.0}

def main():
    print("Loading test batch...")
    tids = load_tiny_split()
    samples = [load_sample(tid) for tid in tids]
    
    from depthwizard.config import TrainConfig
    cfg = TrainConfig(arch="unet3", target_transform="none", epochs=1, batch_size=2, lr=1e-3)
    estimator = BuildingConditionedEstimator(cfg, nodata=-999.0, seed=42)
    model = estimator.model
    
    # Global depth normalization
    estimator.d_mean, estimator.d_std = 0.0, 1.0
    
    # Prepare batch
    xs, ys, ms, ds_raw = [], [], [], []
    for s in samples:
        xs.append(estimator._prep_x(s, 256))
        gt = np.asarray(s["gt"], dtype=np.float32)
        valid = np.isfinite(gt) & (gt != -999.0)
        gt_f = np.where(valid, gt, 0.0)
        gt_r = cv2.resize(gt_f, (256, 256), interpolation=cv2.INTER_LINEAR)
        valid_r = cv2.resize(valid.astype(np.float32), (256, 256), interpolation=cv2.INTER_NEAREST) > 0.5
        ys.append(gt_r)
        ms.append(valid_r)
        
        depth = np.asarray(s["depth"], dtype=np.float32)
        depth_r = cv2.resize(depth, (256, 256), interpolation=cv2.INTER_LINEAR)
        ds_raw.append(depth_r)
        
    x = torch.from_numpy(np.stack(xs)).float().to(estimator.device)
    y = torch.from_numpy(np.stack(ys)).float().to(estimator.device)
    m = torch.from_numpy(np.stack(ms)).bool().to(estimator.device)
    raw_d = torch.from_numpy(np.stack(ds_raw)).float().to(estimator.device)
    
    # --- TEST A: Footprint Loss Only
    print("\n--- Running TEST A: Footprint Loss Backward ---")
    model.zero_grad()
    mask_logits, preds, targets, regimes, weights = model(x, raw_d, y, device=estimator.device)
    
    gt_footprint = (y > 2.0).float()
    loss_footprint = F.binary_cross_entropy_with_logits(mask_logits, gt_footprint, reduction='none')
    loss_footprint = loss_footprint[m].mean()
    loss_footprint.backward(retain_graph=True)
    
    # Measure grads
    grad_e1 = model.backbone.e1[0].weight.grad
    grad_head_mask = model.backbone.head.weight.grad[16, :, :, :] # mask channel (channel 16 out of 17)
    grad_head_feat = model.backbone.head.weight.grad[0, :, :, :]  # feature channel 0
    grad_mlp = model.mlp[0].weight.grad
    
    print("Gradients from Footprint Loss:")
    print(f"  Shared encoder e1 weight grad norm: {grad_e1.norm().item() if grad_e1 is not None else 'None'}")
    print(f"  Footprint head channel grad norm: {grad_head_mask.norm().item() if grad_head_mask is not None else 'None'}")
    print(f"  Feature head channel 0 grad norm: {grad_head_feat.norm().item() if grad_head_feat is not None else 'None'}")
    print(f"  MLP weights grad norm: {grad_mlp.norm().item() if grad_mlp is not None else 'None'}")
    
    # --- TEST B: Height Loss Only
    print("\n--- Running TEST B: Height Loss Backward ---")
    model.zero_grad()
    mask_logits, preds, targets, regimes, weights = model(x, raw_d, y, device=estimator.device)
    
    loss_regime = torch.tensor(0.0).to(estimator.device)
    loss_height = torch.tensor(0.0).to(estimator.device)
    if len(preds) > 0:
        regime_losses = []
        height_losses = []
        for k in range(len(preds)):
            pred_h, pred_regime_logit = preds[k]
            target_h = targets[k]
            target_regime = regimes[k]
            weight = weights[k]
            
            ce = F.cross_entropy(pred_regime_logit.unsqueeze(0), torch.tensor([target_regime]).to(estimator.device))
            regime_losses.append(ce * weight)
            
            sl1 = F.smooth_l1_loss(pred_h, torch.tensor([target_h], dtype=torch.float32).to(estimator.device))
            height_losses.append(sl1 * weight)
            
        loss_regime = torch.stack(regime_losses).mean()
        loss_height = torch.stack(height_losses).mean()
        
    total_height_loss = 0.5 * loss_regime + 0.1 * loss_height
    total_height_loss.backward()
    
    # Measure grads
    grad_e1_h = model.backbone.e1[0].weight.grad
    grad_head_mask_h = model.backbone.head.weight.grad[16, :, :, :] # mask channel
    grad_head_feat_h = model.backbone.head.weight.grad[0, :, :, :]  # feature channel 0
    grad_mlp_h = model.mlp[0].weight.grad
    
    print("Gradients from Height/Regime Loss:")
    print(f"  Shared encoder e1 weight grad norm: {grad_e1_h.norm().item() if grad_e1_h is not None else 'None'}")
    print(f"  Footprint head channel grad norm: {grad_head_mask_h.norm().item() if grad_head_mask_h is not None else 'None'}")
    print(f"  Feature head channel 0 grad norm: {grad_head_feat_h.norm().item() if grad_head_feat_h is not None else 'None'}")
    print(f"  MLP weights grad norm: {grad_mlp_h.norm().item() if grad_mlp_h is not None else 'None'}")
    
    # Save statistics for json
    stats = {
        "footprint_loss": float(loss_footprint.detach().cpu().numpy()),
        "height_loss": float(loss_height.detach().cpu().numpy()) if len(preds) > 0 else 0.0,
        "regime_loss": float(loss_regime.detach().cpu().numpy()) if len(preds) > 0 else 0.0,
        "grad_footprint_loss": {
            "shared_encoder_e1": float(grad_e1.norm().cpu().numpy()) if grad_e1 is not None else None,
            "footprint_head": float(grad_head_mask.norm().cpu().numpy()) if grad_head_mask is not None else None,
            "feature_head_ch0": float(grad_head_feat.norm().cpu().numpy()) if grad_head_feat is not None else None,
            "mlp_weights": float(grad_mlp.norm().cpu().numpy()) if grad_mlp is not None else None
        },
        "grad_height_loss": {
            "shared_encoder_e1": float(grad_e1_h.norm().cpu().numpy()) if grad_e1_h is not None else None,
            "footprint_head": float(grad_head_mask_h.norm().cpu().numpy()) if grad_head_mask_h is not None else None,
            "feature_head_ch0": float(grad_head_feat_h.norm().cpu().numpy()) if grad_head_feat_h is not None else None,
            "mlp_weights": float(grad_mlp_h.norm().cpu().numpy()) if grad_mlp_h is not None else None
        }
    }
    
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(stats, f, indent=2)
        
    print(f"\nSaved results.json to {OUT_DIR}")

if __name__ == "__main__":
    main()
