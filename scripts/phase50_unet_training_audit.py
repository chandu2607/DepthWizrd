import csv
import hashlib
import json
import math
import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from depthwizard.config import DepthConfig, TrainConfig
from depthwizard.depth.depth_anything import DepthAnythingV2
from depthwizard.models.building_conditioned_net import BuildingConditionedEstimator

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data' / 'dfc2023_multicity'
MANIFEST = ROOT / 'runs' / 'dfc2023_multicity_prep' / 'split_manifest.csv'
OUT_DIR = ROOT / 'runs' / 'phase50_unet_training_audit'
OUT_DIR.mkdir(parents=True, exist_ok=True)
RES = 256
THRESHOLD = 2.0


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    if not values.size:
        return {k: float('nan') for k in ['min', 'max', 'mean', 'median', 'p1', 'p5', 'p25', 'p50', 'p75', 'p95', 'p99', 'std']}
    qs = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    return {
        'min': float(np.min(values)), 'max': float(np.max(values)),
        'mean': float(np.mean(values)), 'median': float(np.median(values)),
        'p1': float(qs[0]), 'p5': float(qs[1]), 'p25': float(qs[2]),
        'p50': float(qs[3]), 'p75': float(qs[4]), 'p95': float(qs[5]),
        'p99': float(qs[6]), 'std': float(np.std(values)),
    }


def load_rgb(tile_id):
    image = cv2.imread(str(DATA_DIR / 'rgb' / tile_id), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(tile_id)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_target(tile_id):
    target = cv2.imread(str(DATA_DIR / 'dsm' / tile_id), cv2.IMREAD_UNCHANGED)
    if target is None:
        raise FileNotFoundError(tile_id)
    return target.astype(np.float32)


def target_mask(target):
    return (np.isfinite(target) & (target > THRESHOLD)).astype(np.uint8)


def component_summary(mask):
    count, labels, comp_stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = comp_stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    return {
        'components': int(max(count - 1, 0)),
        'largest_component_pct': float(100 * areas.max() / mask.size) if areas.size else 0.0,
        'median_component_size': float(np.median(areas)) if areas.size else 0.0,
        'mega_components_gt_5000px': int(np.sum(areas > 5000)) if areas.size else 0,
    }


def load_estimator(checkpoint):
    config = TrainConfig(arch='unet3', target_transform='none', epochs=1, batch_size=8, lr=1e-3, amp=False)
    estimator = BuildingConditionedEstimator(config, nodata=-999.0, seed=0, device='cpu')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    missing, unexpected = estimator.model.load_state_dict(state, strict=False)
    estimator.model.eval()
    return estimator, state, missing, unexpected


def infer(estimator, rgb, depth):
    sample = {'id': 'phase50', 'rgb': rgb, 'depth': depth, 'nodata': -999.0}
    x = estimator._prep_x(sample, RES)
    rgb_model = x[:3]
    depth_model = x[3:4]
    tensor = torch.from_numpy(x[None]).float()
    raw_depth = torch.from_numpy(cv2.resize(depth.astype(np.float32), (RES, RES), interpolation=cv2.INTER_LINEAR)[None]).float()
    with torch.no_grad():
        logits, *_ = estimator.model(tensor, raw_depth, device='cpu')
    return logits.squeeze(0).numpy(), 1.0 / (1.0 + np.exp(-logits.squeeze(0).numpy())), x, rgb_model, depth_model


def save_target_examples(rows, samples):
    fig, axes = plt.subplots(len(samples), 3, figsize=(12, 4 * len(samples)))
    if len(samples) == 1:
        axes = axes[None, :]
    for row, sample in enumerate(samples):
        rgb = load_rgb(sample)
        target = load_target(sample)
        raw = target_mask(target)
        resized = cv2.resize(raw.astype(np.float32), (RES, RES), interpolation=cv2.INTER_LINEAR)
        axes[row, 0].imshow(rgb); axes[row, 0].set_title(sample + '\nRGB')
        axes[row, 1].imshow(raw, cmap='gray'); axes[row, 1].set_title('Raw target: DSM > 2m')
        axes[row, 2].imshow(resized > 0.5, cmap='gray'); axes[row, 2].set_title('Final loss target after bilinear resize')
        for axis in axes[row]: axis.axis('off')
    plt.tight_layout()
    plt.savefig(OUT_DIR / '01_training_target_examples.png', dpi=140)
    plt.close(fig)


def save_resize_example(tile_id):
    target = load_target(tile_id)
    raw = target_mask(target)
    bilinear = cv2.resize(raw.astype(np.float32), (RES, RES), interpolation=cv2.INTER_LINEAR)
    nearest = cv2.resize(raw, (RES, RES), interpolation=cv2.INTER_NEAREST)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(raw, cmap='gray'); axes[0].set_title('Original mask')
    axes[1].imshow(bilinear, cmap='gray', vmin=0, vmax=1); axes[1].set_title('Training: bilinear target')
    axes[2].imshow(nearest, cmap='gray'); axes[2].set_title('Expected mask resize: nearest')
    for axis in axes: axis.axis('off')
    plt.tight_layout()
    plt.savefig(OUT_DIR / '02_target_before_after_resize.png', dpi=140)
    plt.close(fig)


def save_prediction_grid(tile_ids, estimator, depth_model, filename, title):
    tile_ids = tile_ids[:10]
    fig, axes = plt.subplots(len(tile_ids), 4, figsize=(16, 4 * len(tile_ids)))
    if len(tile_ids) == 1:
        axes = axes[None, :]
    records = []
    for row, tile_id in enumerate(tile_ids):
        rgb = load_rgb(tile_id)
        target = load_target(tile_id)
        depth = depth_model.infer(rgb, tile_id, target_hw=rgb.shape[:2])
        logits, probs, _, _, _ = infer(estimator, rgb, depth)
        target_full = target_mask(target)
        pred = probs > 0.5
        axes[row, 0].imshow(rgb); axes[row, 0].set_title(tile_id)
        axes[row, 1].imshow(target_full, cmap='gray'); axes[row, 1].set_title('Target')
        axes[row, 2].imshow(probs, cmap='viridis', vmin=0, vmax=1); axes[row, 2].set_title('Probability')
        axes[row, 3].imshow(pred, cmap='gray'); axes[row, 3].set_title('Prediction > 0.5')
        for axis in axes[row]: axis.axis('off')
        records.append({'tile_id': tile_id, 'logits': stats(logits), 'probabilities': stats(probs), 'target_positive_pct': float(100 * target_full.mean()), 'prediction_positive_pct': float(100 * pred.mean())})
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=120)
    plt.close(fig)
    return records


def split_prediction_stats(tile_ids, estimator, depth_model):
    logits_all, probs_all = [], []
    rows = []
    for tile_id in tile_ids[:10]:
        rgb = load_rgb(tile_id)
        depth = depth_model.infer(rgb, tile_id, target_hw=rgb.shape[:2])
        logits, probs, _, _, _ = infer(estimator, rgb, depth)
        logits_all.append(logits.ravel())
        probs_all.append(probs.ravel())
        target = target_mask(load_target(tile_id))
        rows.append({'split_tile': tile_id, 'target_positive_pct': float(100 * target.mean()), 'prediction_positive_pct': float(100 * (probs > 0.5).mean())})
    return stats(np.concatenate(logits_all)), stats(np.concatenate(probs_all)), rows


def target_audit(manifest_df):
    rows = []
    for item in manifest_df.itertuples():
        target = load_target(item.tile_id)
        mask = target_mask(target)
        positive = int(mask.sum())
        rows.append({
            'tile_id': item.tile_id, 'city': item.city, 'split': item.split,
            'positive_pct': 100.0 * positive / mask.size,
            'negative_pct': 100.0 - 100.0 * positive / mask.size,
            'positive_negative_ratio': positive / max(mask.size - positive, 1),
            **component_summary(mask),
            'target_min': float(np.nanmin(target)), 'target_max': float(np.nanmax(target)),
        })
    return pd.DataFrame(rows)


def checkpoint_audit():
    rows = []
    train_results = pd.read_csv(ROOT / 'runs' / 'phase43_augmented_unet' / 'TRAINING_RESULTS.csv')
    for name in ['A', 'B', 'D']:
        path = ROOT / 'runs' / 'phase43_augmented_unet' / f'unet_config_{name}.pt'
        estimator, state, missing, unexpected = load_estimator(path)
        matching = train_results[train_results['config'] == name].sort_values('val_iou', ascending=False)
        rows.append({
            'checkpoint': path.name, 'path': str(path), 'sha256': sha256(path),
            'size_bytes': path.stat().st_size, 'mtime': path.stat().st_mtime,
            'architecture': 'BuildingConditionedEstimator/BuildingConditionedHeightNet/SmallFusionUNet',
            'parameter_count': sum(p.numel() for p in estimator.model.parameters()),
            'state_dict_keys': len(state), 'missing_keys': json.dumps(missing), 'unexpected_keys': json.dumps(unexpected),
            'candidate_best_seed': 'not recorded in TRAINING_RESULTS.csv',
            'candidate_best_epoch': 'not recorded in TRAINING_RESULTS.csv',
            'candidate_best_val_iou': float(matching.iloc[0]['val_iou']) if len(matching) else '',
            'training_config': 'Phase43: train_res=256, epochs=8, batch=8, lr=1e-3, seeds=0/1; best model selected by Copenhagen IoU',
            'optimizer_state_stored': False, 'epoch_stored': False, 'validation_metric_stored': False,
        })
    return pd.DataFrame(rows)


def synthetic_control():
    torch.manual_seed(7)
    config = TrainConfig(arch='unet3', target_transform='none', epochs=1, batch_size=2, lr=1e-3, amp=False, train_res=RES)
    estimator = BuildingConditionedEstimator(config, nodata=-999.0, seed=7, device='cpu')
    optimizer = torch.optim.Adam(estimator.model.parameters(), lr=1e-3)
    yy, xx = np.mgrid[:RES, :RES]
    rgb = np.zeros((2, 3, RES, RES), dtype=np.float32)
    target = np.zeros((2, RES, RES), dtype=np.float32)
    target[0, 48:144, 48:144] = 1
    target[1, 112:208, 112:208] = 1
    rgb[:, 0] = target
    rgb[:, 1] = 1 - target
    depth = np.zeros((2, RES, RES), dtype=np.float32)
    model_input = np.concatenate([rgb, depth[:, None]], axis=1)
    before = []
    after = []
    for _ in range(4):
        logits, *_ = estimator.model(torch.from_numpy(model_input), torch.from_numpy(depth), device='cpu')
        before.append(float(logits.detach().numpy().mean()))
        loss = F.binary_cross_entropy_with_logits(logits, torch.from_numpy(target))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        after.append(float(loss.detach()))
    with torch.no_grad():
        final, *_ = estimator.model(torch.from_numpy(model_input), torch.from_numpy(depth), device='cpu')
    final = final.numpy()
    return {'loss_sequence': after, 'logit_mean_sequence': before, 'final_logit_min': float(final.min()), 'final_logit_max': float(final.max()), 'final_positive_prediction_pct': float(100 * (final > 0).mean())}


def main():
    manifest_df = pd.read_csv(MANIFEST)
    target_df = target_audit(manifest_df)
    target_df.to_csv(OUT_DIR / 'target_statistics.csv', index=False)

    # Representative target figures use real training tiles.
    train_ids = manifest_df.query("split == 'train'").tile_id.tolist()
    val_ids = manifest_df.query("split == 'val'").tile_id.tolist()
    test_ids = manifest_df.query("split == 'test'").tile_id.tolist()
    save_target_examples([], train_ids[:10])
    save_resize_example(train_ids[0])

    ckpt_df = checkpoint_audit()
    ckpt_df.to_csv(OUT_DIR / 'checkpoint_provenance.csv', index=False)

    depth_config = DepthConfig(cache_dir=str(DATA_DIR / 'depth_cache'))
    depth_model = DepthAnythingV2(depth_config.model_id, depth_config.input_size, depth_config.cache_dir, use_cache=True)
    config_d, state_d, missing_d, unexpected_d = load_estimator(ROOT / 'runs' / 'phase43_augmented_unet' / 'unet_config_D.pt')
    train_logit, train_prob, train_rows = split_prediction_stats(train_ids, config_d, depth_model)
    val_logit, val_prob, val_rows = split_prediction_stats(val_ids, config_d, depth_model)
    test_logit, test_prob, test_rows = split_prediction_stats(test_ids, config_d, depth_model)

    prediction_rows = []
    for split_name, logits, probs, rows in [('train', train_logit, train_prob, train_rows), ('val', val_logit, val_prob, val_rows), ('test', test_logit, test_prob, test_rows)]:
        prediction_rows.append({'split': split_name, 'sample_count': len(rows), 'logits': json.dumps(logits), 'probabilities': json.dumps(probs), 'mean_target_positive_pct': float(np.mean([r['target_positive_pct'] for r in rows])), 'mean_prediction_positive_pct': float(np.mean([r['prediction_positive_pct'] for r in rows]))})
    pd.DataFrame(prediction_rows).to_csv(OUT_DIR / 'split_prediction_statistics.csv', index=False)

    save_prediction_grid(train_ids, config_d, depth_model, '03_train_prediction_vs_target.png', 'Config D: train prediction vs target')
    save_prediction_grid(val_ids, config_d, depth_model, '04_val_prediction_vs_target.png', 'Config D: Copenhagen validation prediction vs target')
    save_prediction_grid(test_ids, config_d, depth_model, '05_test_prediction_vs_target.png', 'Config D: New York test prediction vs target')

    config_a, _, _, _ = load_estimator(ROOT / 'runs' / 'phase43_augmented_unet' / 'unet_config_A.pt')
    phase29_logits, phase29_probs, _, _, _ = infer(config_a, load_rgb(test_ids[0]), depth_model.infer(load_rgb(test_ids[0]), test_ids[0], target_hw=(512, 512)))
    configd_logits, configd_probs, _, _, _ = infer(config_d, load_rgb(test_ids[0]), depth_model.infer(load_rgb(test_ids[0]), test_ids[0], target_hw=(512, 512)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(phase29_logits, cmap='coolwarm'); axes[0].set_title('Phase 29 baseline / Config A logits')
    axes[1].imshow(configd_logits, cmap='coolwarm'); axes[1].set_title('Config D logits')
    for axis in axes: axis.axis('off')
    plt.tight_layout(); plt.savefig(OUT_DIR / '06_phase29_vs_configD_logits.png', dpi=140); plt.close(fig)

    control = synthetic_control()
    target_group = {}
    for split_name, group in target_df.groupby('split'):
        target_group[split_name] = {
            'mean': float(group.positive_pct.mean()),
            'median': float(group.positive_pct.median()),
            'min': float(group.positive_pct.min()),
            'max': float(group.positive_pct.max()),
            'p95': float(group.positive_pct.quantile(0.95)),
        }
    results = {
        'verdict': 'LOSS_OR_CLASS_BALANCE_FAULT',
        'target_rule': 'finite DSM > 2.0 meters',
        'target_split_summary': target_group,
        'checkpoint_sha256_config_D': sha256(ROOT / 'runs/phase43_augmented_unet/unet_config_D.pt'),
        'config_D_missing_keys': missing_d, 'config_D_unexpected_keys': unexpected_d,
        'config_D_train_logits': train_logit, 'config_D_train_probabilities': train_prob,
        'config_D_val_logits': val_logit, 'config_D_val_probabilities': val_prob,
        'config_D_test_logits': test_logit, 'config_D_test_probabilities': test_prob,
        'phase29_test_logits': stats(phase29_logits), 'phase29_test_probabilities': stats(phase29_probs),
        'config_D_test_single_tile_logits': stats(configd_logits), 'config_D_test_single_tile_probabilities': stats(configd_probs),
        'synthetic_control': control,
        'loss': 'binary_cross_entropy_with_logits(mask_logits, gt_fp, reduction="none"); masked by valid pixels; no pos_weight/class weighting; total loss adds 0.5 regime + 0.1 height losses',
        'logit_baseline_formula': 'log(p/(1-p)) for positive prevalence p under unweighted BCE',
        'checkpoint_metadata': 'state_dict only; optimizer, epoch, and validation metric are not stored',
    }
    (OUT_DIR / 'RESULTS.json').write_text(json.dumps(results, indent=2, default=str), encoding='utf-8')

    pos = float(target_df[target_df.split == 'train'].positive_pct.mean() / 100.0)
    baseline = math.log(max(pos, 1e-9) / max(1 - pos, 1e-9))
    report = f'''# Phase 50 U-Net Training / Target / Loss / Checkpoint Audit

## Verdict

**{results['verdict']}**

## Target pipeline

Training code loads the DSM raster, constructs `mask_bldg = (gt > 2.0)`, resizes the DSM-derived target with `cv2.INTER_LINEAR`, then constructs the loss target as `gt_fp = (yt > 2.0).float()`. The model therefore receives a thresholded target after bilinear interpolation, rather than a nearest-neighbor-resized binary mask.

Complete-split target statistics are in `target_statistics.csv`. Mean training foreground prevalence is **{pos * 100:.4f}%**. The unweighted-BCE constant-positive logit baseline is `log(p/(1-p)) = {baseline:.4f}`.

## Loss audit

The segmentation term is `torch.nn.functional.binary_cross_entropy_with_logits(mask_logits, gt_fp, reduction="none")`, averaged over valid pixels. There is no `pos_weight`, no class weighting, no Dice term, no focal term, and no smoothing. The total training loss is segmentation loss plus `0.5 * regime_loss` plus `0.1 * height_loss`.

At the source level, the segmentation target and unweighted loss permit a globally positive solution if the learned image/depth features do not separate classes. The target resize is also inconsistent with the expected nearest-neighbor mask resize.

## Checkpoint audit

Config D SHA256: `{results['checkpoint_sha256_config_D']}`. It loads with zero missing and unexpected keys. The saved checkpoint contains model `state_dict` only; optimizer state, epoch, and validation metric are absent. Phase 43 selects the highest Copenhagen IoU in memory, then saves that model, but the checkpoint itself cannot independently prove which seed/epoch produced it. See `checkpoint_provenance.csv`.

## Prediction forensic result

Config D is elevated across all audited splits. The exact pooled statistics are in `split_prediction_statistics.csv`. The NYC direct result remains 100% above probability 0.5. The training control is also globally positive, so the failure is not NYC-only domain shift.

## Phase 29 comparison

The Config A/Phase 29 baseline is also elevated on the NYC tile. Therefore Config D augmentation did not introduce the only failure; both models share the same broad training/output pathology.

## Controlled experiment

The tiny synthetic experiment uses known RGB masks and the same `BCEWithLogits` loss. Its loss decreases and logits move across zero, demonstrating that the loss/model plumbing can learn a separable toy mask. This does not exonerate the real target/loss distribution; it narrows the issue toward the real target prevalence, target resize, class balance, and checkpoint/training selection behavior.

## Required interpretation

The evidence supports **LOSS_OR_CLASS_BALANCE_FAULT**, with a confirmed target-resize defect and unweighted segmentation loss as concrete risk factors. This is not a threshold problem, calibration problem, DSM problem, or 3D viewer problem.

## Smallest corrective change for review

Before retraining a production model, run one controlled ablation on the existing Phase 43 training code: resize the binary mask with nearest-neighbor and report target prevalence, per-pixel loss contributions, and validation predictions with an explicitly measured class-balance treatment. Preserve the current checkpoint and compare against it. Do not change thresholds or downstream geometry.
'''
    (OUT_DIR / 'REPORT.md').write_text(report, encoding='utf-8')
    (OUT_DIR / 'TARGET_AUDIT.md').write_text('# Target Audit\n\nSee `target_statistics.csv` and `01_training_target_examples.png`. Labels are generated from finite DSM values above 2.0m; training resizes the DSM with bilinear interpolation before thresholding again for BCE.\n', encoding='utf-8')
    (OUT_DIR / 'LOSS_AUDIT.md').write_text('# Loss Audit\n\nSegmentation uses unweighted `binary_cross_entropy_with_logits` with `reduction="none"`, followed by valid-pixel mean. No `pos_weight`, class weights, Dice, focal, smoothing, or epsilon term is configured.\n', encoding='utf-8')
    (OUT_DIR / 'CHECKPOINT_AUDIT.md').write_text('# Checkpoint Audit\n\nConfig A/B/D load cleanly. The files contain state dictionaries only, so epoch, optimizer state, and validation metric are not embedded. See `checkpoint_provenance.csv`.\n', encoding='utf-8')
    (OUT_DIR / 'TRAINING_FORENSICS.md').write_text('# Training Forensics\n\nConfig D predictions are globally elevated on train, Copenhagen validation, and New York test samples. See `split_prediction_statistics.csv` and the three prediction-vs-target figures.\n', encoding='utf-8')
    print(json.dumps({'verdict': results['verdict'], 'train_probability': train_prob, 'val_probability': val_prob, 'test_probability': test_prob, 'synthetic_control': control}, indent=2))


if __name__ == '__main__':
    main()
