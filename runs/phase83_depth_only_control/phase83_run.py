from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depthwizard.depth.depth_anything import DepthAnythingV2
from runs.phase79_spatial_terrain_baseline.phase79_run import (
    PHASE72,
    TRAIN_REGION,
    build_crop,
    compute_metrics,
    deterministic_split,
    gradient_abs_error,
    open_raster,
)

OUT = Path(__file__).resolve().parent
VIS = OUT / 'VISUALS'
VIS.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = 512
SEED = 1337


def depth_pipeline(rgb: np.ndarray) -> np.ndarray:
    model = DepthAnythingV2(
        model_id='depth-anything/Depth-Anything-V2-Small-hf',
        cache_dir=str(REPO_ROOT / 'data' / 'depth_cache'),
        use_cache=True,
    )
    rgb_u8 = (rgb.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    depth = model.infer(rgb_u8, key=f'phase83_{TRAIN_REGION}_{TARGET_SIZE}', target_hw=(TARGET_SIZE, TARGET_SIZE)).astype(np.float32)
    return depth


def rank_transform(values: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    if ref is None:
        ref = values
    sorted_ref = np.sort(np.asarray(ref).reshape(-1))
    arr = np.asarray(values).reshape(-1)
    idx = np.searchsorted(sorted_ref, arr, side='left')
    if sorted_ref.size == 1:
        return np.zeros_like(arr, dtype=np.float32)
    return (idx / (sorted_ref.size - 1)).astype(np.float32)


def fit_linear_rank(train_rank: np.ndarray, train_relief: np.ndarray):
    if train_rank.size < 2:
        return 1.0, 0.0
    X = np.column_stack([train_rank, np.ones_like(train_rank)])
    coeffs, _, _, _ = np.linalg.lstsq(X, train_relief, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def eval_pred(pred: np.ndarray, target: np.ndarray, valid: np.ndarray):
    metrics = compute_metrics(pred, target, valid)
    grad = gradient_abs_error(pred, target)
    return {
        'mae': float(metrics['mae']),
        'rmse': float(metrics['rmse']),
        'pearson': float(metrics['pearson']) if metrics['pearson'] is not None and not np.isnan(metrics['pearson']) else None,
        'prediction_mean': float(metrics['prediction_mean']),
        'prediction_std': float(metrics['prediction_std']),
        'target_std': float(metrics['target_std']),
        'mean_bias': float(metrics['mean_bias']),
        'gradient_x_mae': float(grad['gradient_x_mae']),
        'gradient_y_mae': float(grad['gradient_y_mae']),
    }


def save_visuals(true_relief: np.ndarray, raw_depth: np.ndarray, inverted_depth: np.ndarray, best_pred: np.ndarray, abs_err: np.ndarray):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes[0, 0].imshow(true_relief, cmap='terrain')
    axes[0, 0].set_title('True local relief')
    axes[0, 0].axis('off')

    im_raw = axes[0, 1].imshow(raw_depth, cmap='terrain')
    axes[0, 1].set_title('Raw depth')
    axes[0, 1].axis('off')
    fig.colorbar(im_raw, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im_inv = axes[0, 2].imshow(inverted_depth, cmap='terrain')
    axes[0, 2].set_title('Inverted depth')
    axes[0, 2].axis('off')
    fig.colorbar(im_inv, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im_best = axes[1, 0].imshow(best_pred, cmap='terrain')
    axes[1, 0].set_title('Best deterministic depth-derived relief')
    axes[1, 0].axis('off')
    fig.colorbar(im_best, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im_err = axes[1, 1].imshow(abs_err, cmap='magma')
    axes[1, 1].set_title('Absolute error')
    axes[1, 1].axis('off')
    fig.colorbar(im_err, ax=axes[1, 1], fraction=0.046, pad=0.04)

    axes[1, 2].axis('off')
    fig.tight_layout()
    fig.savefig(VIS / 'depth_controls_summary.png', dpi=180)
    plt.close(fig)


def load_phase82_metrics():
    phase82_path = REPO_ROOT / 'runs' / 'phase82_learning_trajectory' / 'RESULTS.json'
    payload = json.loads(phase82_path.read_text(encoding='utf-8'))
    history = payload['epoch_history']
    final_epoch = history[-1]
    return {
        'terrainunet_epoch5': {
            'mae': float(final_epoch['validation_mae']),
            'rmse': float(final_epoch['validation_rmse']),
            'pearson': float(final_epoch['validation_pearson']),
            'prediction_std': float(final_epoch['prediction_std']),
            'gradient_x_mae': float(final_epoch['gradient_x_mae']),
            'gradient_y_mae': float(final_epoch['gradient_y_mae']),
            'epoch': int(final_epoch['epoch']),
        },
        'zero_baseline': payload['baseline_comparison']['zero_baseline'],
        'training_mean_baseline': payload['baseline_comparison']['training_mean_baseline'],
        'training_median_baseline': payload['baseline_comparison']['training_median_baseline'],
    }


def classify_depth_signal(raw_corr: float, best_det_corr: float, best_det_mae: float, baseline_mae_values: list[float]):
    best_baseline_mae = min(baseline_mae_values) if baseline_mae_values else float('inf')
    if best_det_corr > 0.20 and best_det_mae < 0.90 * best_baseline_mae:
        return 'DEPTH_PRIOR_CONTAINS_TRANSFERABLE_TERRAIN_STRUCTURE'
    if abs(raw_corr) > 0.10 and best_det_corr > 0.05:
        return 'DEPTH_HAS_SIGNAL_BUT_REQUIRES_SPATIAL_CALIBRATION'
    return 'DEPTH_PRIOR_INSUFFICIENT_FOR_TERRAIN'


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    _, mask = open_raster(PHASE72 / TRAIN_REGION / 'valid_mask.tif')
    mask = mask[0].astype(bool)
    train_bbox, val_bbox = deterministic_split(mask)
    train_crop = build_crop(TRAIN_REGION, train_bbox)
    val_crop = build_crop(TRAIN_REGION, val_bbox)

    train_rgb = train_crop['rgb']
    val_rgb = val_crop['rgb']

    train_depth = depth_pipeline(train_rgb)
    val_depth = depth_pipeline(val_rgb)

    train_relief = train_crop['dem'] - train_crop['local_median']
    val_relief = val_crop['dem'] - val_crop['local_median']
    train_mask = train_crop['mask']
    val_mask = val_crop['mask']

    train_depth_valid = train_depth[train_mask]
    val_depth_valid = val_depth[val_mask]
    train_relief_valid = train_relief[train_mask]
    val_relief_valid = val_relief[val_mask]

    raw_depth_stats = {
        'train': {
            'min': float(train_depth_valid.min()),
            'max': float(train_depth_valid.max()),
            'mean': float(train_depth_valid.mean()),
            'std': float(train_depth_valid.std()),
            'finite': bool(np.isfinite(train_depth_valid).all()),
            'non_constant': bool(np.std(train_depth_valid) > 1e-8),
            'spatial_variation': bool(np.std(train_depth_valid) > 1e-8),
        },
        'validation': {
            'min': float(val_depth_valid.min()),
            'max': float(val_depth_valid.max()),
            'mean': float(val_depth_valid.mean()),
            'std': float(val_depth_valid.std()),
            'finite': bool(np.isfinite(val_depth_valid).all()),
            'non_constant': bool(np.std(val_depth_valid) > 1e-8),
            'spatial_variation': bool(np.std(val_depth_valid) > 1e-8),
        },
    }

    def pearson(x: np.ndarray, y: np.ndarray):
        if x.size < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
            return float('nan')
        return float(np.corrcoef(x, y)[0, 1])

    def spearman(x: np.ndarray, y: np.ndarray):
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        return pearson(rx.astype(np.float64), ry.astype(np.float64))

    corr_depth = {
        'pearson_depth_vs_relief': pearson(val_depth_valid, val_relief_valid),
        'spearman_depth_vs_relief': spearman(val_depth_valid, val_relief_valid),
        'pearson_minus_depth_vs_relief': pearson(-val_depth_valid, val_relief_valid),
        'spearman_minus_depth_vs_relief': spearman(-val_depth_valid, val_relief_valid),
    }

    depth_mean = float(train_depth_valid.mean())
    depth_std = float(train_depth_valid.std())
    if depth_std < 1e-8:
        depth_std = 1.0
    target_mean = float(train_relief_valid.mean())
    target_std = float(train_relief_valid.std())
    if target_std < 1e-8:
        target_std = 1.0

    depth_norm = (val_depth - depth_mean) / depth_std
    depth_norm_inv = -depth_norm
    pred_a = depth_norm * target_std + target_mean
    pred_b = depth_norm_inv * target_std + target_mean
    pred_a = np.asarray(pred_a, dtype=np.float32)
    pred_b = np.asarray(pred_b, dtype=np.float32)

    a_metrics = eval_pred(pred_a, val_relief, val_mask)
    b_metrics = eval_pred(pred_b, val_relief, val_mask)

    train_rank = rank_transform(train_depth_valid, train_depth_valid)
    val_rank = rank_transform(val_depth_valid, train_depth_valid)
    a_rank, b_rank = fit_linear_rank(train_rank, train_relief_valid)
    pred_rank = a_rank * val_rank + b_rank
    pred_rank = pred_rank.reshape(val_relief.shape).astype(np.float32)
    rank_metrics = eval_pred(pred_rank, val_relief, val_mask)
    rank_metrics['a'] = a_rank
    rank_metrics['b'] = b_rank

    controls = {
        'A_normalized_depth': a_metrics,
        'B_inverted_normalized_depth': b_metrics,
        'C_rank_linear_depth_mapping': rank_metrics,
    }

    best_control_name = max(
        controls,
        key=lambda name: (controls[name]['pearson'] if controls[name]['pearson'] is not None else -1.0, -controls[name]['mae'])
    )
    best_control = controls[best_control_name]

    phase82 = load_phase82_metrics()
    baseline_maes = [
        float(phase82['zero_baseline']['mae']),
        float(phase82['training_mean_baseline']['mae']),
        float(phase82['training_median_baseline']['mae']),
    ]

    final_label = classify_depth_signal(
        max(abs(corr_depth['pearson_depth_vs_relief']), abs(corr_depth['pearson_minus_depth_vs_relief'])) if corr_depth['pearson_depth_vs_relief'] is not None else 0.0,
        best_control['pearson'] if best_control['pearson'] is not None else 0.0,
        best_control['mae'],
        baseline_maes,
    )

    # Prepare visuals
    true_relief = val_relief.astype(np.float32)
    raw_depth_map = val_depth.astype(np.float32)
    inverted_depth_map = (-val_depth).astype(np.float32)
    best_pred_map = pred_a if best_control_name == 'A_normalized_depth' else pred_b if best_control_name == 'B_inverted_normalized_depth' else pred_rank
    best_pred_map = best_pred_map.astype(np.float32)
    abs_error = np.abs(best_pred_map - true_relief)
    save_visuals(true_relief, raw_depth_map, inverted_depth_map, best_pred_map, abs_error)

    comparison = {
        'DEPTH_CONTROL': {
            'method': best_control_name,
            'metrics': best_control,
        },
        'TERRAINUNET': phase82['terrainunet_epoch5'],
        'ZERO_BASELINE': phase82['zero_baseline'],
        'TRAINING_MEAN_BASELINE': phase82['training_mean_baseline'],
        'TRAINING_MEDIAN_BASELINE': phase82['training_median_baseline'],
    }

    depth_controls = {
        'raw_depth_summary': raw_depth_stats,
        'depth_vs_local_relief_correlation': corr_depth,
        'controls': controls,
        'best_control_name': best_control_name,
        'best_control_metrics': best_control,
        'phase82_reference': phase82,
    }

    (OUT / 'DEPTH_CONTROLS.json').write_text(json.dumps(depth_controls, indent=2), encoding='utf-8')
    (OUT / 'COMPARISON.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
    (OUT / 'RESULTS.json').write_text(json.dumps({
        'phase': 'PHASE_83',
        'status': 'DEPTH_ONLY_CONTROL',
        'final_label': final_label,
        'raw_depth_stats': raw_depth_stats,
        'depth_vs_local_relief_correlation': corr_depth,
        'controls': controls,
        'best_control_name': best_control_name,
        'best_control_metrics': best_control,
        'comparison': comparison,
    }, indent=2), encoding='utf-8')

    report_lines = [
        '# Phase 83 depth-only terrain structure control',
        '',
        '## Data and validation setup',
        '- Same held-out Uttarakhand validation split as Phase 77/82.',
        '- Exact Phase 78 depth generation path from Depth Anything V2 raw relative-depth inference.',
        '- No Sikkim evaluation; no retraining; no production edits.',
        '',
        '## Raw depth summary',
        f"- train depth min/max/mean/std: {raw_depth_stats['train']['min']:.6f}/{raw_depth_stats['train']['max']:.6f}/{raw_depth_stats['train']['mean']:.6f}/{raw_depth_stats['train']['std']:.6f}",
        f"- validation depth min/max/mean/std: {raw_depth_stats['validation']['min']:.6f}/{raw_depth_stats['validation']['max']:.6f}/{raw_depth_stats['validation']['mean']:.6f}/{raw_depth_stats['validation']['std']:.6f}",
        f"- validation finite: {raw_depth_stats['validation']['finite']}",
        f"- validation non-constant: {raw_depth_stats['validation']['non_constant']}",
        f"- validation spatial variation: {raw_depth_stats['validation']['spatial_variation']}",
        '',
        '## Depth-to-terrain correlation',
        f"- Pearson(depth, local_relief): {corr_depth['pearson_depth_vs_relief']}",
        f"- Spearman(depth, local_relief): {corr_depth['spearman_depth_vs_relief']}",
        f"- Pearson(-depth, local_relief): {corr_depth['pearson_minus_depth_vs_relief']}",
        f"- Spearman(-depth, local_relief): {corr_depth['spearman_minus_depth_vs_relief']}",
        '',
        '## Deterministic depth controls',
        f"- A normalized depth: MAE={a_metrics['mae']:.6f}, RMSE={a_metrics['rmse']:.6f}, Pearson={a_metrics['pearson']}, prediction_std={a_metrics['prediction_std']:.6f}, gradient_x_mae={a_metrics['gradient_x_mae']:.6f}, gradient_y_mae={a_metrics['gradient_y_mae']:.6f}",
        f"- B inverted normalized depth: MAE={b_metrics['mae']:.6f}, RMSE={b_metrics['rmse']:.6f}, Pearson={b_metrics['pearson']}, prediction_std={b_metrics['prediction_std']:.6f}, gradient_x_mae={b_metrics['gradient_x_mae']:.6f}, gradient_y_mae={b_metrics['gradient_y_mae']:.6f}",
        f"- C rank-linear depth mapping: MAE={rank_metrics['mae']:.6f}, RMSE={rank_metrics['rmse']:.6f}, Pearson={rank_metrics['pearson']}, a={rank_metrics['a']:.6f}, b={rank_metrics['b']:.6f}, prediction_std={rank_metrics['prediction_std']:.6f}, gradient_x_mae={rank_metrics['gradient_x_mae']:.6f}, gradient_y_mae={rank_metrics['gradient_y_mae']:.6f}",
        '',
        '## Comparison against Phase 82 metrics',
        f"- DEPTH_CONTROL ({best_control_name}): MAE={best_control['mae']:.6f}, RMSE={best_control['rmse']:.6f}, Pearson={best_control['pearson']}, prediction_std={best_control['prediction_std']:.6f}, gradient_x_mae={best_control['gradient_x_mae']:.6f}, gradient_y_mae={best_control['gradient_y_mae']:.6f}",
        f"- TERRAINUNET: MAE={phase82['terrainunet_epoch5']['mae']:.6f}, RMSE={phase82['terrainunet_epoch5']['rmse']:.6f}, Pearson={phase82['terrainunet_epoch5']['pearson']}, prediction_std={phase82['terrainunet_epoch5']['prediction_std']:.6f}, gradient_x_mae={phase82['terrainunet_epoch5']['gradient_x_mae']:.6f}, gradient_y_mae={phase82['terrainunet_epoch5']['gradient_y_mae']:.6f}",
        f"- ZERO_BASELINE: MAE={phase82['zero_baseline']['mae']:.6f}, RMSE={phase82['zero_baseline']['rmse']:.6f}, Pearson={phase82['zero_baseline']['pearson']}, prediction_std={phase82['zero_baseline']['prediction_std']:.6f}",
        f"- TRAINING_MEAN_BASELINE: MAE={phase82['training_mean_baseline']['mae']:.6f}, RMSE={phase82['training_mean_baseline']['rmse']:.6f}, Pearson={phase82['training_mean_baseline']['pearson']}, prediction_std={phase82['training_mean_baseline']['prediction_std']:.6f}",
        f"- TRAINING_MEDIAN_BASELINE: MAE={phase82['training_median_baseline']['mae']:.6f}, RMSE={phase82['training_median_baseline']['rmse']:.6f}, Pearson={phase82['training_median_baseline']['pearson']}, prediction_std={phase82['training_median_baseline']['prediction_std']:.6f}",
        '',
        '## Interpretation',
        f"- Best deterministic depth control: {best_control_name}",
        f"- Final decision: {final_label}",
        '',
        '## Final label',
        final_label,
    ]
    (OUT / 'REPORT.md').write_text('\n'.join(report_lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
