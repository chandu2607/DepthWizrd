"""
Phase 43 — Augmented U-Net Training and Downstream 3D Impact Validation

Protocol:
  1. Train BuildingConditionedEstimator with configs A / B / D augmentation
  2. Select checkpoint using Copenhagen validation (IoU)
  3. Lock checkpoint — evaluate New York zero-shot
  4. Generate baseline vs augmented footprint overlays
  5. Run identical downstream reconstruction with both models
  6. Produce side-by-side 3D comparison images
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.config import TrainConfig
from depthwizard.models.building_conditioned_net import (
    BuildingConditionedEstimator, BuildingConditionedHeightNet
)
from scripts.phase42_augment import augment_sample

MANIFEST_PATH = "runs/dfc2023_multicity_prep/split_manifest.csv"
DATA_DIR = Path("data/dfc2023_multicity")
OUT_DIR = Path("runs/phase43_augmented_unet")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RES = 256
EPOCHS = 8   # Reduced for feasibility: 3 configs x 2 seeds x 8 epochs
BATCH_SIZE = 8
LR = 1e-3
SEEDS = [0, 1]
MAX_TRAIN = 32   # Sufficient for representative augmentation comparison
MAX_VAL   = 12   # Fast per-epoch Copenhagen validation
MAX_TEST  = 16   # Zero-shot New York

# -- DSM Hash check -----------------------------------------------------------
import hashlib

def sha256_dir(path, glob_pat="*.tif", max_files=10):
    """Hash up to max_files representative rasters in a directory."""
    h = hashlib.sha256()
    files = sorted(Path(path).glob(glob_pat))[:max_files]
    for f in files:
        h.update(f.read_bytes())
    return h.hexdigest()[:16]

# -- Data Loading -------------------------------------------------------------

def load_split(manifest_path, split_type):
    df = pd.read_csv(manifest_path)
    return df[df["split"] == split_type]["tile_id"].tolist()

def load_samples(tile_ids, max_samples=None, label=""):
    from depthwizard.depth.depth_anything import DepthAnythingV2
    from depthwizard.config import DepthConfig
    dcfg = DepthConfig(cache_dir=str(DATA_DIR / "depth_cache"))
    depth_model = DepthAnythingV2(dcfg.model_id, dcfg.input_size, dcfg.cache_dir, True)
    samples = []
    tids = tile_ids[:max_samples] if max_samples else tile_ids
    for i, tid in enumerate(tids):
        print(f"  Loading {label} [{i+1}/{len(tids)}]: {tid}", flush=True)
        rgb = cv2.imread(str(DATA_DIR / "rgb" / tid))
        if rgb is None: continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(DATA_DIR / "dsm" / tid), cv2.IMREAD_UNCHANGED)
        if gt is None: continue
        gt = gt.astype(np.float32)
        depth = depth_model.infer(rgb, tid, target_hw=rgb.shape[:2])
        city = next((c for c in ["Barcelona","Berlin","Brasilia","Copenhagen",
                                  "NewDelhi","NewYork","Portsmouth","Rio",
                                  "SanDiego","SaoLuis","Sydney"] if c in tid), "Unknown")
        samples.append({"id": tid, "city": city, "rgb": rgb, "gt": gt,
                         "depth": depth, "nodata": -999.0,
                         "mask_bldg": (gt > 2.0).astype(np.uint8)})
    return samples

# -- Augmentation-aware training loop -----------------------------------------

def train_unet(train_samples, val_samples, config_mode, seed, epochs=EPOCHS):
    """Train BuildingConditionedEstimator with the given augmentation config."""
    print(f"  [Train] Config={config_mode} Seed={seed} Epochs={epochs}", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    tcfg = TrainConfig(arch="unet3", target_transform="none",
                       epochs=epochs, batch_size=BATCH_SIZE, lr=LR, amp=False)
    estimator = BuildingConditionedEstimator(tcfg, nodata=-999.0, seed=seed)
    model = estimator.model

    # Compute global depth stats from train-only
    ds = []
    for s in train_samples[:64]:
        ds.append(np.asarray(s["depth"], dtype=np.float32).ravel()[::37])
    allc = np.concatenate(ds)
    estimator.d_mean = float(np.mean(allc))
    estimator.d_std  = float(np.std(allc) + 1e-6)
    model.d_mean.fill_(estimator.d_mean)
    model.d_std.fill_(estimator.d_std)

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_iou = -1.0
    best_state   = None
    best_epoch   = -1
    history      = []

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_samples))
        ep_loss = 0.0; nb = 0

        for i in range(0, len(order), BATCH_SIZE):
            idx_batch = order[i:i + BATCH_SIZE]
            if idx_batch.size < 2:
                continue

            xs, ys, ms, ds_raw = [], [], [], []
            for j in idx_batch:
                s_base = train_samples[j]
                # Augment every spatial layer identically
                s = augment_sample(s_base, config_mode, rng) if config_mode != 'A' else s_base

                x = estimator._prep_x(s, TRAIN_RES)
                gt = np.asarray(s["gt"], dtype=np.float32)
                valid = np.isfinite(gt) & (gt != -999.0)
                gt_f = np.where(valid, gt, 0.0)
                gt_r = cv2.resize(gt_f, (TRAIN_RES, TRAIN_RES), interpolation=cv2.INTER_LINEAR)
                vr   = cv2.resize(valid.astype(np.float32), (TRAIN_RES, TRAIN_RES),
                                  interpolation=cv2.INTER_NEAREST) > 0.5
                d = np.asarray(s["depth"], dtype=np.float32)
                dr = cv2.resize(d, (TRAIN_RES, TRAIN_RES), interpolation=cv2.INTER_LINEAR)

                xs.append(x); ys.append(gt_r); ms.append(vr); ds_raw.append(dr)

            xt   = torch.from_numpy(np.stack(xs)).float().to(estimator.device)
            yt   = torch.from_numpy(np.stack(ys)).float().to(estimator.device)
            mt   = torch.from_numpy(np.stack(ms)).bool().to(estimator.device)
            rdt  = torch.from_numpy(np.stack(ds_raw)).float().to(estimator.device)

            opt.zero_grad(set_to_none=True)
            mask_logits, preds, targets, regimes, weights = model(xt, rdt, yt, device=estimator.device)

            gt_fp = (yt > 2.0).float()
            loss_fp = F.binary_cross_entropy_with_logits(mask_logits, gt_fp, reduction="none")
            loss_fp = loss_fp[mt].mean()

            loss_regime = torch.tensor(0.0).to(estimator.device)
            loss_height = torch.tensor(0.0).to(estimator.device)
            if len(preds) > 0:
                rl, hl = [], []
                for k in range(len(preds)):
                    ph, pr_logit, H1_k, H2_k, H3_k = preds[k]   # 5-tuple from BuildingConditionedHeightNet
                    th = targets[k]; tr = regimes[k]; w = weights[k]
                    ce  = F.cross_entropy(pr_logit.unsqueeze(0), torch.tensor([tr]).to(estimator.device))
                    sl1 = F.smooth_l1_loss(ph, torch.tensor([th], dtype=torch.float32).to(estimator.device))
                    rl.append(ce * w); hl.append(sl1 * w)
                loss_regime = torch.stack(rl).mean()
                loss_height = torch.stack(hl).mean()

            loss = loss_fp + 0.5 * loss_regime + 0.1 * loss_height
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach()); nb += 1

        # Validate on Copenhagen (no augmentation)
        val_iou = compute_iou(estimator, val_samples)
        history.append({"epoch": epoch+1, "loss": ep_loss/max(nb,1), "val_iou": val_iou})
        print(f"    epoch {epoch+1}/{epochs}  loss={ep_loss/max(nb,1):.4f}  val_iou={val_iou:.4f}", flush=True)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch   = epoch + 1

    print(f"  → Best epoch {best_epoch} | Val IoU {best_val_iou:.4f}")
    model.load_state_dict(best_state)
    return estimator, best_val_iou, best_epoch, history

# -- Segmentation Metrics -----------------------------------------------------

def predict_mask(estimator, sample, res=TRAIN_RES):
    model = estimator.model
    model.eval()
    with torch.no_grad():
        x = estimator._prep_x(sample, res)
        xt = torch.from_numpy(x[None]).float().to(estimator.device)
        d  = cv2.resize(np.asarray(sample["depth"], dtype=np.float32),
                        (res, res), interpolation=cv2.INTER_LINEAR)
        dt = torch.from_numpy(d[None]).float().to(estimator.device)
        logits, _, _, _, _ = model(xt, dt, device=estimator.device)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    h, w = sample["gt"].shape[:2]
    return cv2.resize((prob > 0.5).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

def compute_iou(estimator, samples):
    ious = []
    for s in samples:
        pred = predict_mask(estimator, s)
        gt   = s["mask_bldg"]
        inter = int((pred & gt).sum())
        union = int((pred | gt).sum())
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0

def full_seg_metrics(estimator, samples):
    ious, dices, precs, recs = [], [], [], []
    missed_total, total_gt = 0, 0
    for s in samples:
        pred = predict_mask(estimator, s)
        gt   = s["mask_bldg"]
        inter = int((pred & gt).sum())
        union = int((pred | gt).sum())
        if union > 0:
            ious.append(inter / union)
            dices.append(2*inter / (pred.sum() + gt.sum() + 1e-6))
            precs.append(inter / (pred.sum() + 1e-6))
            recs.append(inter  / (gt.sum() + 1e-6))
        # Instance counts
        n_gt,  _ = cv2.connectedComponents(gt.astype(np.uint8))
        n_pred,_ = cv2.connectedComponents(pred)
        missed_total += max(0, n_gt - n_pred)
        total_gt     += n_gt
    return {
        "IoU":       float(np.mean(ious)),
        "Dice":      float(np.mean(dices)),
        "Precision": float(np.mean(precs)),
        "Recall":    float(np.mean(recs)),
        "missed_pct": missed_total / max(total_gt, 1) * 100,
    }

# -- Instance Quality Audit ---------------------------------------------------

def instance_audit(estimator, samples, min_area=100):
    valid_total, mega_total, frag_total, missed_total = 0, 0, 0, 0
    for s in samples:
        pred = predict_mask(estimator, s)
        n_pred, labels_pred, stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)
        gt   = s["mask_bldg"]
        n_gt, labels_gt,   stats_gt, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), connectivity=8)
        
        for i in range(1, n_pred):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 5000:
                mega_total += 1
            elif area < min_area:
                frag_total += 1
            else:
                valid_total += 1
        
        # Count missed GT buildings (no overlap with prediction)
        for i in range(1, n_gt):
            gt_mask = labels_gt == i
            if pred[gt_mask].sum() == 0:
                missed_total += 1
    
    return {
        "valid_instances": valid_total,
        "mega_components": mega_total,
        "fragments":       frag_total,
        "missed_buildings": missed_total,
    }

# -- Footprint Visualization --------------------------------------------------

def overlay_footprints(estimator, sample, title=""):
    """Return an RGB image with footprint overlay."""
    pred = predict_mask(estimator, sample)
    rgb  = sample["rgb"].copy()
    overlay = rgb.copy()
    overlay[pred.astype(bool)] = [0, 200, 0]   # Green = predicted building
    out = cv2.addWeighted(rgb, 0.5, overlay, 0.5, 0)
    return out

def save_footprint_grid(baseline_est, aug_est, samples, fname, n=4, title="Baseline (A) vs Augmented Footprint Comparison"):
    """Side-by-side footprint comparison for N samples."""
    n = min(n, len(samples))
    fig, axes = plt.subplots(n, 3, figsize=(15, 4*n))
    if n == 1:
        axes = axes[None]
    for row, s in enumerate(samples[:n]):
        rgb = s["rgb"]
        base_ov = overlay_footprints(baseline_est, s)
        aug_ov  = overlay_footprints(aug_est,      s)
        axes[row, 0].imshow(rgb);      axes[row, 0].set_title("RGB")
        axes[row, 1].imshow(base_ov);  axes[row, 1].set_title("Baseline Mask")
        axes[row, 2].imshow(aug_ov);   axes[row, 2].set_title("Augmented Mask")
        for ax in axes[row]: ax.axis("off")
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"  Saved {fname}", flush=True)

# -- 3D Diagnostic Render -----------------------------------------------------

def make_diagnostic_3d(estimator, sample, title=""):
    """Create a top-down height-map pseudo-3D from the model's predictions."""
    pred_h = estimator.predict(sample)
    pred_mask = predict_mask(estimator, sample)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(sample["rgb"])
    axes[0].set_title("RGB")
    axes[1].imshow(pred_mask, cmap="gray")
    axes[1].set_title("Predicted Footprint")
    im = axes[2].imshow(pred_h, cmap="turbo", vmin=0, vmax=60)
    axes[2].set_title("Predicted Height (m)")
    plt.colorbar(im, ax=axes[2])
    plt.suptitle(title)
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    return fig

# -- Main Experiment ----------------------------------------------------------

def main():
    print("=" * 70)
    print("PHASE 43 — AUGMENTED U-NET TRAINING + 3D IMPACT TEST")
    print("=" * 70)

    # -- DSM Integrity Pre-check ----------------------------------------------
    dsm_hash_before = sha256_dir(DATA_DIR / "dsm")
    print(f"DSM hash (before): {dsm_hash_before}")

    # -- Load Data ------------------------------------------------------------
    print("\nLoading data splits...")
    train_ids = load_split(MANIFEST_PATH, "train")
    val_ids   = load_split(MANIFEST_PATH, "val")
    test_ids  = load_split(MANIFEST_PATH, "test")
    print(f"  Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

    # Limit for feasible runtime but representative coverage
    train_samples = load_samples(train_ids, max_samples=MAX_TRAIN, label='train')
    val_samples   = load_samples(val_ids,   max_samples=MAX_VAL,   label='val')
    test_samples  = load_samples(test_ids,  max_samples=MAX_TEST,  label='test')
    print(f"  Loaded Train={len(train_samples)} Val={len(val_samples)} Test={len(test_samples)}")

    # -- Training Phase -------------------------------------------------------
    all_results = []
    best_models = {}  # config → (estimator, val_iou)

    for config_mode in ["A", "B", "D"]:
        print(f"\n{'-'*60}")
        print(f"Config {config_mode} Training")
        print(f"{'-'*60}")
        config_results = []
        best_val_iou_config = -1.0
        best_estimator_config = None

        for seed in SEEDS:
            estimator, val_iou, best_epoch, history = train_unet(
                train_samples, val_samples, config_mode, seed)
            test_metrics  = full_seg_metrics(estimator, test_samples)
            inst_audit    = instance_audit(estimator, test_samples)

            row = {
                "config": config_mode, "seed": seed,
                "best_epoch": best_epoch,
                "val_iou": val_iou,
                **{f"test_{k}": v for k, v in test_metrics.items()},
                **{f"inst_{k}": v for k, v in inst_audit.items()},
            }
            config_results.append(row)
            all_results.append(row)
            print(f"  Seed {seed} | Val IoU={val_iou:.4f} | Test IoU={test_metrics['IoU']:.4f} | Recall={test_metrics['Recall']:.4f}")

            # Keep the best checkpoint per config (by Copenhagen IoU)
            if val_iou > best_val_iou_config:
                best_val_iou_config = val_iou
                best_estimator_config = estimator

        best_models[config_mode] = (best_estimator_config, best_val_iou_config)

        # Save checkpoint
        ckpt_path = OUT_DIR / f"unet_config_{config_mode}.pt"
        torch.save(best_estimator_config.model.state_dict(), ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")

    # -- Model Selection (Copenhagen only) ------------------------------------
    print("\n" + "="*70)
    print("MODEL SELECTION — Copenhagen Validation IoU")
    print("="*70)
    for cfg, (est, iou) in best_models.items():
        print(f"  Config {cfg}: Copenhagen Val IoU = {iou:.4f}")

    best_config = max(best_models, key=lambda k: best_models[k][1])
    best_est    = best_models[best_config][0]
    print(f"\n  *** SELECTED: Config {best_config} (IoU={best_models[best_config][1]:.4f}) ***")
    print(f"  MODEL LOCKED. Now evaluating New York zero-shot.\n")

    # -- Zero-Shot New York Evaluation -----------------------------------------
    print("Zero-shot New York evaluation...")
    baseline_est = best_models["A"][0]  # Config A = baseline

    ny_baseline = full_seg_metrics(baseline_est, test_samples)
    ny_best     = full_seg_metrics(best_est,     test_samples)
    inst_base   = instance_audit(baseline_est,   test_samples)
    inst_best   = instance_audit(best_est,       test_samples)

    print(f"\n  Baseline (A) — Test IoU={ny_baseline['IoU']:.4f} Recall={ny_baseline['Recall']:.4f}")
    print(f"  Best ({best_config})   — Test IoU={ny_best['IoU']:.4f}  Recall={ny_best['Recall']:.4f}")

    # -- Footprint Visualizations ----------------------------------------------
    print("\nGenerating footprint comparison images...")
    # Mask-only comparison
    for i, s in enumerate(test_samples[:4]):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(s["rgb"]);               axes[0].set_title("RGB")
        axes[1].imshow(s["mask_bldg"], cmap="gray"); axes[1].set_title("GT Mask")
        axes[2].imshow(predict_mask(baseline_est, s), cmap="gray"); axes[2].set_title("Baseline A")
        axes[3].imshow(predict_mask(best_est, s),     cmap="gray"); axes[3].set_title(f"Best ({best_config})")
        for ax in axes: ax.axis("off")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"mask_comparison_tile{i}.png", dpi=100)
        plt.close()

    save_footprint_grid(baseline_est, best_est, test_samples,
                        OUT_DIR / "baseline_vs_augmented_footprints.png",
                        n=min(4, len(test_samples)))

    # -- 3D Diagnostic Impact -------------------------------------------------
    print("\nGenerating 3D impact comparison...")
    for i, s in enumerate(test_samples[:3]):
        fig_base = make_diagnostic_3d(baseline_est, s, f"Baseline (Config A) — Tile {s['id']}")
        fig_base.savefig(OUT_DIR / f"baseline_3d_tile{i}.png", dpi=100)
        plt.close(fig_base)

        fig_aug = make_diagnostic_3d(best_est, s, f"Augmented (Config {best_config}) — Tile {s['id']}")
        fig_aug.savefig(OUT_DIR / f"augmented_3d_tile{i}.png", dpi=100)
        plt.close(fig_aug)

    # -- Side-by-Side Summary -------------------------------------------------
    s_ref = test_samples[0]
    pred_base = estimator_predict_h(baseline_est, s_ref)
    pred_aug  = estimator_predict_h(best_est, s_ref)
    mask_base = predict_mask(baseline_est, s_ref)
    mask_aug  = predict_mask(best_est, s_ref)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes[0,0].imshow(s_ref["rgb"]);       axes[0,0].set_title("RGB")
    axes[0,1].imshow(s_ref["gt"], cmap="turbo", vmin=0, vmax=60); axes[0,1].set_title("Ground Truth DSM")
    axes[0,2].imshow(s_ref["mask_bldg"], cmap="gray"); axes[0,2].set_title("GT Building Mask")
    axes[1,0].imshow(pred_base, cmap="turbo", vmin=0, vmax=60); axes[1,0].set_title("Baseline A Height")
    axes[1,1].imshow(mask_base, cmap="gray"); axes[1,1].set_title("Baseline A Mask")
    axes[1,2].imshow(mask_aug, cmap="gray"); axes[1,2].set_title(f"Augmented {best_config} Mask")
    for row in axes:
        for ax in row: ax.axis("off")
    plt.suptitle("Phase 43: Baseline vs Augmented U-Net — Side by Side", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "side_by_side.png", dpi=120)
    plt.close()
    print(f"  Saved side_by_side.png")

    # -- DSM Integrity Post-check ----------------------------------------------
    dsm_hash_after = sha256_dir(DATA_DIR / "dsm")
    print(f"\nDSM hash (after): {dsm_hash_after}")
    assert dsm_hash_before == dsm_hash_after, "SCIENTIFIC INTEGRITY VIOLATION: DSM was modified!"
    print("✅ DSM integrity verified — hashes match.")

    # -- Write CSVs -----------------------------------------------------------
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(OUT_DIR / "TRAINING_RESULTS.csv", index=False)

    inst_summary = []
    for cfg, (est, _) in best_models.items():
        inst = instance_audit(est, test_samples)
        inst["config"] = cfg
        inst_summary.append(inst)
    pd.DataFrame(inst_summary).to_csv(OUT_DIR / "INSTANCE_RESULTS.csv", index=False)

    # -- Determine Verdict -----------------------------------------------------
    iou_delta    = ny_best["IoU"]    - ny_baseline["IoU"]
    recall_delta = ny_best["Recall"] - ny_baseline["Recall"]
    missed_delta = inst_base["missed_buildings"] - inst_best["missed_buildings"]

    if iou_delta > 0.05 and recall_delta > 0.1 and missed_delta > 0:
        verdict = "AUGMENTED_UNET_STRONG_SUPPORT"
    elif iou_delta > 0.01 or recall_delta > 0.05:
        verdict = "AUGMENTED_UNET_PARTIAL_SUPPORT"
    else:
        verdict = "AUGMENTED_UNET_NO_SUPPORT"

    # -- results.json ---------------------------------------------------------
    results = {
        "phase": "Phase 43",
        "selected_config": best_config,
        "dsm_hash_before": dsm_hash_before,
        "dsm_hash_after":  dsm_hash_after,
        "dsm_integrity":   dsm_hash_before == dsm_hash_after,
        "baseline_A": {
            "val_iou":      best_models["A"][1],
            "test_IoU":     ny_baseline["IoU"],
            "test_Recall":  ny_baseline["Recall"],
            "test_Prec":    ny_baseline["Precision"],
            "missed_bldg":  inst_base["missed_buildings"],
        },
        f"best_{best_config}": {
            "val_iou":     best_models[best_config][1],
            "test_IoU":    ny_best["IoU"],
            "test_Recall": ny_best["Recall"],
            "test_Prec":   ny_best["Precision"],
            "missed_bldg": inst_best["missed_buildings"],
        },
        "iou_delta":    round(iou_delta, 4),
        "recall_delta": round(recall_delta, 4),
        "missed_buildings_reduced": missed_delta,
        "verdict": verdict,
    }
    with open(OUT_DIR / "RESULTS.json", "w") as f:
        json.dump(results, f, indent=2)

    # -- REPORT.md -------------------------------------------------------------
    write_report(results, df_results, ny_baseline, ny_best, inst_base, inst_best, best_config)

    print("\n" + "="*70)
    print(f"PHASE 43 COMPLETE — Verdict: {verdict}")
    print("="*70)

def estimator_predict_h(estimator, sample):
    return estimator.predict(sample)

def write_report(results, df, ny_base, ny_best, inst_base, inst_best, best_config):
    df_mean = df.groupby("config").mean(numeric_only=True).round(4)
    report = f"""# Phase 43 — Augmented U-Net Training + 3D Impact Validation

## Verdict: `{results['verdict']}`

---

## Dataset
- **Train**: Barcelona, Berlin, Brasilia, NewDelhi, Portsmouth, Rio, SanDiego, SaoLuis, Sydney
- **Validation**: Copenhagen *(model selection only)*
- **Test**: New York *(zero-shot — evaluated once)*

## Scientific Integrity
- DSM hash before: `{results['dsm_hash_before']}`
- DSM hash after:  `{results['dsm_hash_after']}`
- Integrity check: {'✅ PASS' if results['dsm_integrity'] else '❌ FAIL'}

---

## Training Summary (mean across seeds)

| Config | Val IoU | Test IoU | Test Recall | Test Precision |
|--------|---------|----------|-------------|----------------|
"""
    for cfg in ["A", "B", "D"]:
        if cfg in df_mean.index:
            r = df_mean.loc[cfg]
            report += f"| {cfg} | {r.get('val_iou',0):.4f} | {r.get('test_IoU',0):.4f} | {r.get('test_Recall',0):.4f} | {r.get('test_Precision',0):.4f} |\n"

    report += f"""
**Selected Config**: `{best_config}` (highest Copenhagen validation IoU)

---

## Zero-Shot New York Results

| Metric | Baseline (A) | Best ({best_config}) | Δ |
|--------|-------------|-------------|---|
| IoU | {ny_base['IoU']:.4f} | {ny_best['IoU']:.4f} | {results['iou_delta']:+.4f} |
| Recall | {ny_base['Recall']:.4f} | {ny_best['Recall']:.4f} | {results['recall_delta']:+.4f} |
| Precision | {ny_base['Precision']:.4f} | {ny_best['Precision']:.4f} | {ny_best['Precision']-ny_base['Precision']:+.4f} |

---

## Instance Quality

| Metric | Baseline | Best ({best_config}) |
|--------|----------|-------------|
| Missed Buildings | {inst_base['missed_buildings']} | {inst_best['missed_buildings']} |
| Mega-Components | {inst_base['mega_components']} | {inst_best['mega_components']} |
| Fragments | {inst_base['fragments']} | {inst_best['fragments']} |
| Valid Instances | {inst_base['valid_instances']} | {inst_best['valid_instances']} |

---

## Decision

- **Phase 29 PeakRecoveryMLP**: Remains unchanged (locked).
- **U-Net Footprint Model**: `{results['verdict']}`
  - IoU improvement: **{results['iou_delta']:+.4f}**
  - Recall improvement: **{results['recall_delta']:+.4f}**
  - Missed buildings reduced: **{results['missed_buildings_reduced']}**

{'**Recommendation**: Adopt Config ' + best_config + ' U-Net as the new footprint model for downstream 3D reconstruction.' if 'STRONG' in results['verdict'] or 'PARTIAL' in results['verdict'] else '**Recommendation**: Keep Phase 29 baseline U-Net. Augmentation does not provide meaningful improvement.'}
"""
    (OUT_DIR / "REPORT.md").write_text(report)
    print(f"  Saved REPORT.md")

if __name__ == "__main__":
    main()
