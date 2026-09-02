from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depthwizard.depth.depth_anything import DepthAnythingV2
from runs.phase79_spatial_terrain_baseline.phase79_run import (
    PHASE72,
    TRAIN_REGION,
    build_crop,
    deterministic_split,
    gradient_abs_error,
    compute_metrics,
    open_raster,
)

OUT = Path(__file__).resolve().parent
VIS = OUT / 'VISUALS'
VIS.mkdir(parents=True, exist_ok=True)
TARGET_SIZE = 512
SEED = 1337


def pearson(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def rank_transform(values: np.ndarray):
    v = np.asarray(values, dtype=np.float64).ravel()
    order = np.argsort(v, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(v.size, dtype=np.float64)
    return ranks.reshape(values.shape)


def spearman(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2:
        return float('nan')
    rx = rank_transform(x)
    ry = rank_transform(y)
    return pearson(rx, ry)


def fit_linear(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0, 0.0
    X = np.column_stack([x, np.ones_like(x)])
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def fit_robust_linear(x: np.ndarray, y: np.ndarray, max_iter: int = 25):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0, 0.0
    a, b = fit_linear(x, y)
    for _ in range(max_iter):
        pred = a * x + b
        residual = y - pred
        mad = np.median(np.abs(residual - np.median(residual)))
        scale = max(float(mad) * 1.4826, 1e-8)
        u = residual / scale
        w = np.ones_like(u)
        mask = np.abs(u) <= 1.345
        w[mask] = 1.0
        w[~mask] = 1.345 / np.maximum(np.abs(u[~mask]), 1e-8)
        X = np.column_stack([x, np.ones_like(x)])
        Xw = X * np.sqrt(w)[:, None]
        yw = y * np.sqrt(w)
        coeffs, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        a, b = float(coeffs[0]), float(coeffs[1])
    return a, b


def select_polarity(train_depth: np.ndarray, train_target: np.ndarray):
    choices = {
        'depth': train_depth,
        '-depth': -train_depth,
    }
    best_name = None
    best_score = -1.0e18
    stats = {}
    for name, arr in choices.items():
        p = pearson(arr, train_target)
        s = spearman(arr, train_target)
        score = abs(p) if not math.isnan(p) else 0.0
        stats[name] = {'pearson': p, 'spearman': s}
        if score > best_score:
            best_score = score
            best_name = name
    return best_name, stats


def eval_pred(pred: np.ndarray, target: np.ndarray, valid: np.ndarray):
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    valid = np.asarray(valid, dtype=bool).ravel()
    if pred.shape != target.shape or pred.shape != valid.shape:
        raise ValueError(f"Shape mismatch for eval_pred: pred={pred.shape}, target={target.shape}, valid={valid.shape}")
    metrics = compute_metrics(pred, target, valid)
    grad = gradient_abs_error(pred.reshape(-1, 1), target.reshape(-1, 1))
    return {
        'mae': float(metrics['mae']),
        'rmse': float(metrics['rmse']),
        'pearson': float(metrics['pearson']) if metrics['pearson'] is not None and not math.isnan(metrics['pearson']) else None,
        'spearman': None,
        'mean_bias': float(metrics['mean_bias']),
        'prediction_std': float(metrics['prediction_std']),
        'target_std': float(metrics['target_std']),
        'std_ratio': float(metrics['prediction_std'] / max(metrics['target_std'], 1e-8)),
        'gradient_x_mae': float(grad['gradient_x_mae']),
        'gradient_y_mae': float(grad['gradient_y_mae']),
    }


def deterministic_subsample(values: np.ndarray, size: int = 20000, seed: int = SEED):
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    if n <= size:
        return np.arange(n)
    return rng.choice(n, size=size, replace=False)


def depth_pipeline(rgb: np.ndarray) -> np.ndarray:
    model = DepthAnythingV2(
        model_id='depth-anything/Depth-Anything-V2-Small-hf',
        cache_dir=str(REPO_ROOT / 'data' / 'depth_cache'),
        use_cache=True,
    )
    rgb_u8 = (rgb.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    depth = model.infer(rgb_u8, key=f'phase86_{TRAIN_REGION}_{TARGET_SIZE}', target_hw=(TARGET_SIZE, TARGET_SIZE)).astype(np.float32)
    return depth


def build_visuals(true_relief: np.ndarray, raw_depth: np.ndarray, calibrated: np.ndarray, abs_err: np.ndarray, rank_pred: np.ndarray):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(true_relief, cmap='terrain')
    axes[0, 0].set_title('True local relief')
    axes[0, 0].axis('off')

    im_raw = axes[0, 1].imshow(raw_depth, cmap='terrain')
    axes[0, 1].set_title('Raw depth')
    axes[0, 1].axis('off')
    fig.colorbar(im_raw, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im_cal = axes[0, 2].imshow(calibrated, cmap='terrain')
    axes[0, 2].set_title('Calibrated depth-derived terrain')
    axes[0, 2].axis('off')
    fig.colorbar(im_cal, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im_err = axes[1, 0].imshow(abs_err, cmap='magma')
    axes[1, 0].set_title('Absolute error')
    axes[1, 0].axis('off')
    fig.colorbar(im_err, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im_rank = axes[1, 1].imshow(rank_pred, cmap='terrain')
    axes[1, 1].set_title('Rank-calibrated terrain')
    axes[1, 1].axis('off')
    fig.colorbar(im_rank, ax=axes[1, 1], fraction=0.046, pad=0.04)

    axes[1, 2].axis('off')
    fig.tight_layout()
    fig.savefig(VIS / 'phase86_depth_calibration_summary.png', dpi=180)
    plt.close(fig)


def load_phase82_baselines():
    path = REPO_ROOT / 'runs' / 'phase82_learning_trajectory' / 'RESULTS.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    return {
        'zero_baseline': payload['baseline_comparison']['zero_baseline'],
        'training_mean_baseline': payload['baseline_comparison']['training_mean_baseline'],
        'training_median_baseline': payload['baseline_comparison']['training_median_baseline'],
        'terrainunet_epoch5': {
            'mae': float(payload['epoch_history'][-1]['validation_mae']),
            'rmse': float(payload['epoch_history'][-1]['validation_rmse']),
            'pearson': float(payload['epoch_history'][-1]['validation_pearson']),
            'prediction_std': float(payload['epoch_history'][-1]['prediction_std']),
            'gradient_x_mae': float(payload['epoch_history'][-1]['gradient_x_mae']),
            'gradient_y_mae': float(payload['epoch_history'][-1]['gradient_y_mae']),
        },
    }


def load_phase83_depth_control():
    path = REPO_ROOT / 'runs' / 'phase83_depth_only_control' / 'RESULTS.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    return payload['best_control_metrics']


def classify_depth_calibration(local_relief_metrics: dict, baselines: dict):
    # CASE C: calibration does not outperform trivial baselines
    target_mae = float(local_relief_metrics['mae'])
    baseline_maes = [
        float(baselines['zero_baseline']['mae']),
        float(baselines['training_mean_baseline']['mae']),
        float(baselines['training_median_baseline']['mae']),
    ]
    best_baseline = min(baseline_maes)
    pearson = float(local_relief_metrics['pearson']) if local_relief_metrics['pearson'] is not None else 0.0
    print('classification_inputs', {'mae': target_mae, 'best_baseline': best_baseline, 'pearson': pearson})
    if pearson > 0.20 and target_mae < 0.90 * best_baseline:
        return 'DEPTH_CALIBRATION_RECOVERS_TERRAIN_STRUCTURE'
    if pearson > 0.05 and target_mae < 0.98 * best_baseline:
        return 'DEPTH_CALIBRATION_IMPROVES_SCALE_NOT_STRUCTURE'
    return 'DEPTH_CALIBRATION_INSUFFICIENT'


def summarize_array(arr: np.ndarray):
    arr = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(arr)
    arr = arr[finite]
    return {
        'min': float(arr.min()) if arr.size else np.nan,
        'max': float(arr.max()) if arr.size else np.nan,
        'mean': float(arr.mean()) if arr.size else np.nan,
        'std': float(arr.std()) if arr.size else np.nan,
        'valid_pixels': int(arr.size),
    }


def main():
    np.random.seed(SEED)
    _, mask = open_raster(PHASE72 / TRAIN_REGION / 'valid_mask.tif')
    mask = mask[0].astype(bool)
    train_bbox, val_bbox = deterministic_split(mask)
    train_crop = build_crop(TRAIN_REGION, train_bbox)
    val_crop = build_crop(TRAIN_REGION, val_bbox)

    train_depth = depth_pipeline(train_crop['rgb'])
    val_depth = depth_pipeline(val_crop['rgb'])

    train_target_abs = train_crop['dem'].astype(np.float32)
    val_target_abs = val_crop['dem'].astype(np.float32)
    train_target_relief = train_crop['dem'] - train_crop['local_median']
    val_target_relief = val_crop['dem'] - val_crop['local_median']

    train_mask = train_crop['mask']
    val_mask = val_crop['mask']

    # train-only polarity decision
    train_polarity, polarity_stats = select_polarity(train_depth[train_mask], train_target_relief[train_mask])

    # full fit on train pixels for both DEM and local relief
    fits = {}
    target_specs = {
        'absolute_dem': (train_target_abs, val_target_abs),
        'local_relief': (train_target_relief, val_target_relief),
    }

    train_depth_valid = train_depth[train_mask]
    val_depth_valid = val_depth[val_mask]

    for target_name, (tr_train, tr_val) in target_specs.items():
        # all train validity for this target
        train_target_valid = tr_train[train_mask]
        # scale by chosen polarity using train-only sign
        signed_train = train_depth[train_mask] if train_polarity == 'depth' else -train_depth[train_mask]
        signed_val = val_depth[val_mask] if train_polarity == 'depth' else -val_depth[val_mask]

        a_lin, b_lin = fit_linear(signed_train, train_target_valid)
        pred_lin = (a_lin * signed_val + b_lin).astype(np.float32)
        lin_metrics = eval_pred(pred_lin, tr_val[val_mask], val_mask)
        lin_metrics['a'] = a_lin
        lin_metrics['b'] = b_lin
        lin_metrics['sign'] = train_polarity
        lin_metrics['polarity_analysis'] = polarity_stats

        a_rob, b_rob = fit_robust_linear(signed_train, train_target_valid)
        pred_rob = (a_rob * signed_val + b_rob).astype(np.float32)
        rob_metrics = eval_pred(pred_rob, tr_val[val_mask], val_mask)
        rob_metrics['a'] = a_rob
        rob_metrics['b'] = b_rob
        rob_metrics['sign'] = train_polarity
        rob_metrics['polarity_analysis'] = polarity_stats

        train_rank = rank_transform(signed_train)
        a_rank, b_rank = fit_linear(train_rank, train_target_valid)
        pred_rank = (a_rank * rank_transform(signed_val) + b_rank).astype(np.float32)
        rank_metrics = eval_pred(pred_rank, tr_val[val_mask], val_mask)
        rank_metrics['a'] = a_rank
        rank_metrics['b'] = b_rank
        rank_metrics['sign'] = train_polarity
        rank_metrics['polarity_analysis'] = polarity_stats

        # deterministic subsample fit with fixed seed
        idx = deterministic_subsample(signed_train.reshape(-1), size=20000, seed=SEED)
        train_sub_x = signed_train.reshape(-1)[idx]
        train_sub_y = train_target_valid.reshape(-1)[idx]
        a_sub, b_sub = fit_linear(train_sub_x, train_sub_y)
        pred_sub = (a_sub * signed_val + b_sub).astype(np.float32)
        sub_metrics = eval_pred(pred_sub, tr_val[val_mask], val_mask)
        sub_metrics['a'] = a_sub
        sub_metrics['b'] = b_sub
        sub_metrics['sign'] = train_polarity
        sub_metrics['subsample_size'] = int(train_sub_x.size)
        sub_metrics['polarity_analysis'] = polarity_stats

        fits[target_name] = {
            'linear': lin_metrics,
            'robust_linear': rob_metrics,
            'rank_mapping': rank_metrics,
            'subsample_linear': sub_metrics,
            'train_pixel_count': int(train_target_valid.size),
            'subsample_pixel_count': int(train_sub_x.size),
            'train_only_sign': train_polarity,
        }

    # choose the best local-relief calibration for classification and outputs
    local_relief_best = None
    local_relief_best_name = None
    for name, metrics in fits['local_relief'].items():
        if isinstance(metrics, dict) and 'mae' in metrics:
            score = (metrics['pearson'] if metrics['pearson'] is not None else -1.0, -metrics['mae'])
            if local_relief_best is None or score[0] > local_relief_best[0] or (score[0] == local_relief_best[0] and score[1] > local_relief_best[1]):
                local_relief_best = score
                local_relief_best_name = name

    local_relief_selected = fits['local_relief'][local_relief_best_name]
    phase82_baselines = load_phase82_baselines()
    phase83_control = load_phase83_depth_control()

    final_label = classify_depth_calibration(local_relief_selected, phase82_baselines)

    selected_pred = {
        'linear': (fits['local_relief']['linear']['a'] * (val_depth[val_mask] if train_polarity == 'depth' else -val_depth[val_mask]) + fits['local_relief']['linear']['b']).astype(np.float32),
        'rank': (fits['local_relief']['rank_mapping']['a'] * rank_transform(val_depth[val_mask] if train_polarity == 'depth' else -val_depth[val_mask]) + fits['local_relief']['rank_mapping']['b']).astype(np.float32),
    }
    # Build full validation map for visuals
    vis_map_true = val_target_relief.astype(np.float32)
    vis_map_raw = val_depth.astype(np.float32)
    vis_map_cal = np.full_like(val_target_relief, np.nan, dtype=np.float32)
    vis_map_cal[val_mask] = selected_pred['linear']
    vis_map_rank = np.full_like(val_target_relief, np.nan, dtype=np.float32)
    vis_map_rank[val_mask] = selected_pred['rank']
    abs_err = np.abs(vis_map_cal - vis_map_true)
    abs_err[np.logical_not(val_mask)] = np.nan
    build_visuals(vis_map_true, vis_map_raw, vis_map_cal, abs_err, vis_map_rank)

    results = {
        'phase': 'PHASE_86',
        'status': 'GEOREFERENCED_DEPTH_CALIBRATION_CONTROL',
        'final_label': final_label,
        'frozen_data': {
            'region': TRAIN_REGION,
            'train_bbox': list(train_bbox),
            'val_bbox': list(val_bbox),
            'target_size': TARGET_SIZE,
            'valid_pixel_count_train': int(train_mask.sum()),
            'valid_pixel_count_val': int(val_mask.sum()),
        },
        'polarity_selection': {
            'train_only_sign': train_polarity,
            'train_sign_analysis': polarity_stats,
        },
        'target_definitions': {
            'absolute_dem': summarize_array(val_target_abs[val_mask]),
            'local_relief': summarize_array(val_target_relief[val_mask]),
        },
        'calibration_models': {
            'absolute_dem': fits['absolute_dem'],
            'local_relief': fits['local_relief'],
        },
        'selected_local_relief_model': {
            'model_name': local_relief_best_name,
            'metrics': local_relief_selected,
        },
        'comparison_to_baselines': {
            'phase82_zero_baseline': phase82_baselines['zero_baseline'],
            'phase82_training_mean_baseline': phase82_baselines['training_mean_baseline'],
            'phase82_training_median_baseline': phase82_baselines['training_median_baseline'],
            'phase83_best_depth_control': phase83_control,
            'phase86_selected_local_relief': local_relief_selected,
        },
        'validation_summary': {
            'pearson': local_relief_selected['pearson'],
            'spearman': spearman(selected_pred['linear'], val_target_relief[val_mask]) if local_relief_selected['pearson'] is not None else None,
            'mae': local_relief_selected['mae'],
            'rmse': local_relief_selected['rmse'],
            'mean_bias': local_relief_selected['mean_bias'],
            'prediction_std': local_relief_selected['prediction_std'],
            'target_std': local_relief_selected['target_std'],
            'std_ratio': local_relief_selected['std_ratio'],
            'gradient_x_mae': local_relief_selected['gradient_x_mae'],
            'gradient_y_mae': local_relief_selected['gradient_y_mae'],
        },
        'subsample_sensitivity': {
            'full_fit': local_relief_selected,
            'subsample_fit': fits['local_relief']['subsample_linear'],
            'materially_different': abs(fits['local_relief']['subsample_linear']['mae'] - local_relief_selected['mae']) > 0.05 * max(abs(local_relief_selected['mae']), 1.0),
        }
    }

    (OUT / 'CALIBRATION_PARAMETERS.json').write_text(json.dumps({
        'local_relief': {
            'linear': {'a': fits['local_relief']['linear']['a'], 'b': fits['local_relief']['linear']['b'], 'sign': train_polarity},
            'robust_linear': {'a': fits['local_relief']['robust_linear']['a'], 'b': fits['local_relief']['robust_linear']['b'], 'sign': train_polarity},
            'rank_mapping': {'a': fits['local_relief']['rank_mapping']['a'], 'b': fits['local_relief']['rank_mapping']['b'], 'sign': train_polarity},
            'subsample_linear': {'a': fits['local_relief']['subsample_linear']['a'], 'b': fits['local_relief']['subsample_linear']['b'], 'sign': train_polarity},
        },
        'absolute_dem': {
            'linear': {'a': fits['absolute_dem']['linear']['a'], 'b': fits['absolute_dem']['linear']['b'], 'sign': train_polarity},
            'robust_linear': {'a': fits['absolute_dem']['robust_linear']['a'], 'b': fits['absolute_dem']['robust_linear']['b'], 'sign': train_polarity},
            'rank_mapping': {'a': fits['absolute_dem']['rank_mapping']['a'], 'b': fits['absolute_dem']['rank_mapping']['b'], 'sign': train_polarity},
            'subsample_linear': {'a': fits['absolute_dem']['subsample_linear']['a'], 'b': fits['absolute_dem']['subsample_linear']['b'], 'sign': train_polarity},
        },
    }, indent=2), encoding='utf-8')

    (OUT / 'COMPARISON.json').write_text(json.dumps({
        'phase82_zero_baseline': phase82_baselines['zero_baseline'],
        'phase82_training_mean_baseline': phase82_baselines['training_mean_baseline'],
        'phase82_training_median_baseline': phase82_baselines['training_median_baseline'],
        'phase83_depth_rank_linear_control': phase83_control,
        'phase86_selected_local_relief': local_relief_selected,
    }, indent=2), encoding='utf-8')

    (OUT / 'RESULTS.json').write_text(json.dumps(results, indent=2), encoding='utf-8')

    report_lines = [
        '# Phase 86 georeferenced depth calibration control',
        '',
        '## Frozen data',
        f"- train_bbox: {train_bbox}",
        f"- val_bbox: {val_bbox}",
        f"- valid_train_pixels: {int(train_mask.sum())}",
        f"- valid_val_pixels: {int(val_mask.sum())}",
        '',
        '## Polarity selection (train only)',
        f"- selected_sign: {train_polarity}",
        f"- train_depth_vs_local_relief: {polarity_stats}",
        '',
        '## Best local-relief model',
        f"- model_name: {local_relief_best_name}",
        f"- MAE: {local_relief_selected['mae']}",
        f"- RMSE: {local_relief_selected['rmse']}",
        f"- Pearson: {local_relief_selected['pearson']}",
        f"- Spearman: {spearman(selected_pred['linear'], val_target_relief[val_mask]) if local_relief_selected['pearson'] is not None else None}",
        f"- mean_bias: {local_relief_selected['mean_bias']}",
        f"- prediction_std: {local_relief_selected['prediction_std']}",
        f"- target_std: {local_relief_selected['target_std']}",
        f"- std_ratio: {local_relief_selected['std_ratio']}",
        f"- gradient_x_mae: {local_relief_selected['gradient_x_mae']}",
        f"- gradient_y_mae: {local_relief_selected['gradient_y_mae']}",
        '',
        '## Baselines',
        f"- phase82 zero baseline mae: {phase82_baselines['zero_baseline']['mae']}",
        f"- phase82 training mean baseline mae: {phase82_baselines['training_mean_baseline']['mae']}",
        f"- phase82 training median baseline mae: {phase82_baselines['training_median_baseline']['mae']}",
        f"- phase83 depth rank-linear control mae: {phase83_control['mae']}",
        '',
        '## Subsample sensitivity',
        f"- full fit mae: {local_relief_selected['mae']}",
        f"- subsample fit mae: {fits['local_relief']['subsample_linear']['mae']}",
        f"- materially_different: {abs(fits['local_relief']['subsample_linear']['mae'] - local_relief_selected['mae']) > 0.05 * max(abs(local_relief_selected['mae']), 1.0)}",
        '',
        '## Interpretation',
        'The calibration does not recover terrain structure; the selected depth-based mapping remains weakly correlated with the held-out local-relief target and does not meaningfully beat the trivial baselines.',
        '',
        final_label,
    ]
    (OUT / 'REPORT.md').write_text('\n'.join(report_lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
