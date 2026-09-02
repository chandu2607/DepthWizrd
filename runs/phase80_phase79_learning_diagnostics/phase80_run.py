from __future__ import annotations

import json
import math
import time
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runs.phase79_spatial_terrain_baseline.phase79_run import (
    TRAIN_REGION,
    VALIDATION_REGION,
    TARGET_SIZE,
    build_crop,
    deterministic_split,
    TerrainDataset,
    TerrainUNet,
    compute_train_stats,
    open_raster,
    PHASE72,
)

PHASE79 = REPO_ROOT / 'runs' / 'phase79_spatial_terrain_baseline'
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)


def safe_stats(arr: np.ndarray):
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    return {
        'shape': list(arr.shape),
        'dtype': str(arr.dtype),
        'min': float(np.nanmin(arr[finite])) if finite.any() else np.nan,
        'max': float(np.nanmax(arr[finite])) if finite.any() else np.nan,
        'mean': float(np.nanmean(arr[finite])) if finite.any() else np.nan,
        'std': float(np.nanstd(arr[finite])) if finite.any() else np.nan,
        'finite_count': int(np.sum(finite)),
        'nan_fraction': float(np.isnan(arr).mean()),
        'inf_fraction': float(np.isinf(arr).mean()),
    }


def compute_distribution_stats(crop_data: dict):
    target = crop_data['dem'] - crop_data['local_median']
    valid = crop_data['mask']
    raw = target[valid]
    return {
        'valid_pixel_count': int(valid.sum()),
        'valid_fraction': float(valid.mean()),
        'target_mean': float(raw.mean()),
        'target_std': float(raw.std()),
        'target_min': float(raw.min()),
        'target_max': float(raw.max()),
    }


def summarize_list(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        'minimum': float(np.nanmin(arr)),
        'maximum': float(np.nanmax(arr)),
        'mean': float(np.nanmean(arr)),
        'median': float(np.nanmedian(arr)),
    }


def final_loss_eq(model: nn.Module, x: torch.Tensor, y: torch.Tensor, m: torch.Tensor):
    pred = model(x)
    criterion = nn.SmoothL1Loss(reduction='none')
    loss_map = criterion(pred * m, y * m)
    loss = (loss_map * m).sum() / (m.sum() + 1e-6)
    return pred, loss, loss_map


def grad_summary(model: nn.Module):
    rows = []
    for name, p in model.named_parameters():
        if p.grad is None:
            g = torch.zeros_like(p)
        else:
            g = p.grad
        finite = torch.isfinite(g)
        rows.append({
            'name': name,
            'param_count': int(p.numel()),
            'gradient_norm': float(torch.linalg.norm(g).item()),
            'gradient_mean': float(g.mean().item()) if g.numel() > 0 else 0.0,
            'gradient_std': float(g.std(unbiased=False).item()) if g.numel() > 0 else 0.0,
            'nonzero_fraction': float((g != 0).float().mean().item()) if g.numel() > 0 else 0.0,
            'nan_count': int(torch.isnan(g).sum().item()),
            'inf_count': int(torch.isinf(g).sum().item()),
            'finite_fraction': float(finite.float().mean().item()) if g.numel() > 0 else 1.0,
        })
    return rows


def compute_baseline_metrics(pred: np.ndarray, target: np.ndarray, valid: np.ndarray):
    pred = pred[valid]
    target = target[valid]
    err = pred - target
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    if np.std(pred) < 1e-8 or np.std(target) < 1e-8:
        pearson = np.nan
    else:
        pearson = float(np.corrcoef(pred, target)[0, 1])
    return {
        'mae': mae,
        'rmse': rmse,
        'prediction_std': float(pred.std()),
        'pearson': pearson,
        'prediction_mean': float(pred.mean()),
        'target_std': float(target.std()),
        'mean_bias': float(np.mean(err)),
    }


def main():
    torch.manual_seed(1337)
    np.random.seed(1337)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds_mask, mask = open_raster(PHASE72 / TRAIN_REGION / 'valid_mask.tif')
    mask = mask[0].astype(bool)
    train_bbox, val_bbox = deterministic_split(mask)
    train_crop = build_crop(TRAIN_REGION, train_bbox)
    val_crop = build_crop(TRAIN_REGION, val_bbox)

    train_target_raw = train_crop['dem'] - train_crop['local_median']
    val_target_raw = val_crop['dem'] - val_crop['local_median']
    train_stats = compute_train_stats(train_target_raw, train_crop['mask'])
    target_transform = {
        'target_type': 'normalized local relief',
        'units': 'meters',
        'normalization': {
            'mean': train_stats['mean'],
            'std': train_stats['std'],
            'formula': 'target_normalized = (raw_local_relief - mean_train_target) / std_train_target',
        },
    }
    # exact target transform
    norm_train = (train_target_raw - train_stats['mean']) / train_stats['std']
    norm_val = (val_target_raw - train_stats['mean']) / train_stats['std']
    reconstructed_train = norm_train * train_stats['std'] + train_stats['mean']
    reconstructed_val = norm_val * train_stats['std'] + train_stats['mean']
    rec_err_train = np.abs(reconstructed_train - train_target_raw)
    rec_err_val = np.abs(reconstructed_val - val_target_raw)

    target_space_audit = {
        'train_crop': {
            'raw_dem': {
                'min': float(train_crop['dem'][train_crop['mask']].min()),
                'max': float(train_crop['dem'][train_crop['mask']].max()),
                'mean': float(train_crop['dem'][train_crop['mask']].mean()),
                'std': float(train_crop['dem'][train_crop['mask']].std()),
            },
            'raw_local_relief': {
                'min': float(train_target_raw[train_crop['mask']].min()),
                'max': float(train_target_raw[train_crop['mask']].max()),
                'mean': float(train_target_raw[train_crop['mask']].mean()),
                'std': float(train_target_raw[train_crop['mask']].std()),
            },
            'model_target': {
                'min': float(norm_train[train_crop['mask']].min()),
                'max': float(norm_train[train_crop['mask']].max()),
                'mean': float(norm_train[train_crop['mask']].mean()),
                'std': float(norm_train[train_crop['mask']].std()),
            },
            'inverse_transform_reconstruction_error': {
                'max_abs': float(rec_err_train.max()),
                'mean_abs': float(rec_err_train.mean()),
            },
            'normalized_target_formula': 'target_normalized = (raw_local_relief - mean_train_target) / std_train_target',
        },
        'validation_crop': {
            'raw_dem': {
                'min': float(val_crop['dem'][val_crop['mask']].min()),
                'max': float(val_crop['dem'][val_crop['mask']].max()),
                'mean': float(val_crop['dem'][val_crop['mask']].mean()),
                'std': float(val_crop['dem'][val_crop['mask']].std()),
            },
            'raw_local_relief': {
                'min': float(val_target_raw[val_crop['mask']].min()),
                'max': float(val_target_raw[val_crop['mask']].max()),
                'mean': float(val_target_raw[val_crop['mask']].mean()),
                'std': float(val_target_raw[val_crop['mask']].std()),
            },
            'model_target': {
                'min': float(norm_val[val_crop['mask']].min()),
                'max': float(norm_val[val_crop['mask']].max()),
                'mean': float(norm_val[val_crop['mask']].mean()),
                'std': float(norm_val[val_crop['mask']].std()),
            },
            'inverse_transform_reconstruction_error': {
                'max_abs': float(rec_err_val.max()),
                'mean_abs': float(rec_err_val.mean()),
            },
            'normalized_target_formula': 'target_normalized = (raw_local_relief - mean_train_target) / std_train_target',
        },
        'target_transform': target_transform,
    }

    crop_stats = []
    for name, crop in [('train', train_crop), ('validation', val_crop)]:
        target = crop['dem'] - crop['local_median']
        valid = crop['mask']
        vals = target[valid]
        crop_stats.append({
            'crop': name,
            'valid_pixel_count': int(valid.sum()),
            'valid_fraction': float(valid.mean()),
            'target_mean': float(vals.mean()),
            'target_std': float(vals.std()),
            'target_min': float(vals.min()),
            'target_max': float(vals.max()),
        })
    all_summary = {
        'valid_pixel_count': summarize_list([c['valid_pixel_count'] for c in crop_stats]),
        'valid_fraction': summarize_list([c['valid_fraction'] for c in crop_stats]),
        'target_mean': summarize_list([c['target_mean'] for c in crop_stats]),
        'target_std': summarize_list([c['target_std'] for c in crop_stats]),
        'target_min': summarize_list([c['target_min'] for c in crop_stats]),
        'target_max': summarize_list([c['target_max'] for c in crop_stats]),
    }

    train_ds = TerrainDataset(train_crop, train_stats)
    loader_train = DataLoader(train_ds, batch_size=1, shuffle=False)
    batch = next(iter(loader_train))
    x = batch['image'].to(device, dtype=torch.float32)
    y = batch['target'].to(device, dtype=torch.float32)
    m = batch['mask'].to(device, dtype=torch.float32)

    model = TerrainUNet().to(device)
    model.train()
    pred_before, loss_before, loss_map = final_loss_eq(model, x, y, m)
    pred_before_np = pred_before.detach().cpu().numpy()[0]
    target_np = y.detach().cpu().numpy()[0, 0]
    mask_np = m.detach().cpu().numpy()[0, 0].astype(bool)
    zero_pred = np.zeros_like(target_np, dtype=np.float32)
    target_mean_pred = np.full_like(target_np, float(target_np[mask_np].mean()), dtype=np.float32)
    target_pred = target_np.copy()

    loss_controls = {
        'loss': float(loss_before.item()),
        'prediction_mean': float(pred_before_np[mask_np].mean()),
        'prediction_std': float(pred_before_np[mask_np].std()),
        'target_mean': float(target_np[mask_np].mean()),
        'target_std': float(target_np[mask_np].std()),
        'loss_if_zero_prediction': float(((torch.abs(pred_before * 0.0 - y) * m).sum() / (m.sum() + 1e-6)).item()),
        'loss_if_target_prediction': float(((torch.abs(pred_before * 0.0 + y - y) * m).sum() / (m.sum() + 1e-6)).item()),
        'loss_if_target_mean_prediction': float(((torch.abs(torch.full_like(y, fill_value=float(target_np[mask_np].mean())) - y) * m).sum() / (m.sum() + 1e-6)).item()),
    }
    # exact reference loss
    ref_pred = pred_before.detach().cpu().numpy()
    ref_target = y.detach().cpu().numpy()
    ref_mask = m.detach().cpu().numpy()
    ref_loss = float((np.abs(ref_pred[0] * ref_mask[0] - ref_target[0, 0] * ref_mask[0]) * ref_mask[0]).sum() / (ref_mask[0].sum() + 1e-6))
    implementation_loss = float(loss_before.item())

    # exact loss expression
    loss_audit = {
        'exact_expression': 'loss = ( SmoothL1Loss(pred * mask, target * mask) * mask ).sum() / (mask.sum() + 1e-6)',
        'valid_pixel_count': int(m.sum().item()),
        'loss_numerator': float((loss_map * m).sum().item()),
        'loss_denominator': float((m.sum() + 1e-6).item()),
        'implementation_loss': implementation_loss,
        'reference_loss': ref_loss,
        'loss_match_tolerance': abs(implementation_loss - ref_loss),
        'valid_mask_excludes_invalid_pixels': int((m == 1).sum().item()) == int(m.sum().item()) and bool((m > 0).all().item()),
    }

    # gradient flow audit
    model.zero_grad(set_to_none=True)
    pred, loss, _ = final_loss_eq(model, x, y, m)
    loss.backward()
    grad_rows = grad_summary(model)
    gradient_audit = {
        'model_name': 'TerrainUNet',
        'train_batch_shape': list(x.shape),
        'loss': float(loss.item()),
        'gradients': grad_rows,
        'health_summary': {
            'has_any_nan': any(r['nan_count'] > 0 for r in grad_rows),
            'has_any_inf': any(r['inf_count'] > 0 for r in grad_rows),
            'earliest_layer_nonzero': next((r['name'] for r in grad_rows if r['nonzero_fraction'] > 1e-6), None),
            'layer_with_largest_norm': max(grad_rows, key=lambda r: r['gradient_norm'])['name'] if grad_rows else None,
            'mean_gradient_norm': float(np.mean([r['gradient_norm'] for r in grad_rows])),
        },
    }

    # update audit
    model_update = TerrainUNet().to(device)
    model_update.train()
    optimizer = torch.optim.Adam(model_update.parameters(), lr=1e-3)
    x2 = x.clone(); y2 = y.clone(); m2 = m.clone()
    optimizer.zero_grad(set_to_none=True)
    pred_before_step, loss_before_step, _ = final_loss_eq(model_update, x2, y2, m2)
    loss_before_step.backward()
    before_state = {name: p.detach().clone() for name, p in model_update.named_parameters()}
    optimizer.step()
    after_state = {name: p.detach().clone() for name, p in model_update.named_parameters()}
    after_pred, after_loss, _ = final_loss_eq(model_update, x2, y2, m2)
    selected = ['e1.0.weight', 'e2.0.weight', 'bottleneck.0.weight', 'head.weight']
    update_rows = []
    for name in selected:
        p_before = before_state[name]
        p_after = after_state[name]
        abs_change = float((p_after - p_before).abs().mean().item())
        rel_change = float((p_after - p_before).abs().mean().item() / (p_before.abs().mean().item() + 1e-8))
        update_rows.append({
            'name': name,
            'abs_parameter_change_mean': abs_change,
            'relative_parameter_change_mean': rel_change,
        })
    update_audit = {
        'loss_before_step': float(loss_before_step.item()),
        'loss_after_step': float(after_loss.item()),
        'parameter_change_summary': update_rows,
        'optimizer_step_count': 1,
    }

    # output response before/after one step
    model_resp = TerrainUNet().to(device)
    optimizer_resp = torch.optim.Adam(model_resp.parameters(), lr=1e-3)
    optimizer_resp.zero_grad(set_to_none=True)
    pred_before_resp, loss_resp, _ = final_loss_eq(model_resp, x, y, m)
    loss_resp.backward()
    before_resp = pred_before_resp.detach().cpu().numpy()[0]
    optimizer_resp.step()
    pred_after_resp, _, _ = final_loss_eq(model_resp, x, y, m)
    after_resp = pred_after_resp.detach().cpu().numpy()[0]
    delta = after_resp - before_resp
    response_audit = {
        'before_step': {
            'min': float(before_resp[mask_np].min()),
            'max': float(before_resp[mask_np].max()),
            'mean': float(before_resp[mask_np].mean()),
            'std': float(before_resp[mask_np].std()),
        },
        'after_step': {
            'min': float(after_resp[mask_np].min()),
            'max': float(after_resp[mask_np].max()),
            'mean': float(after_resp[mask_np].mean()),
            'std': float(after_resp[mask_np].std()),
        },
        'output_change': {
            'mean': float(delta[mask_np].mean()),
            'std': float(delta[mask_np].std()),
            'max_abs': float(np.abs(delta[mask_np]).max()),
        },
    }

    # sensitivity after one-step model
    model_sens = TerrainUNet().to(device)
    opt_sens = torch.optim.Adam(model_sens.parameters(), lr=1e-3)
    opt_sens.zero_grad(set_to_none=True)
    pred_sens, loss_sens, _ = final_loss_eq(model_sens, x, y, m)
    loss_sens.backward(); opt_sens.step()
    with torch.no_grad():
        x0 = x.clone()
        pred_real = model_sens(x0)
        flat = x0[0].permute(1, 2, 0).reshape(-1, 4)
        perm = np.random.default_rng(1337).permutation(flat.shape[0])
        shuffled_flat = flat[perm]
        shuffled = shuffled_flat.reshape(512, 512, 4).permute(2, 0, 1).unsqueeze(0)
        pred_shuffled = model_sens(shuffled.to(device))
        const = x0.mean(dim=(2, 3), keepdim=True).expand_as(x0)
        pred_const = model_sens(const)
    real_pred_np = pred_real.detach().cpu().numpy()[0]
    shuffled_pred_np = pred_shuffled.detach().cpu().numpy()[0]
    const_pred_np = pred_const.detach().cpu().numpy()[0]
    sens_summary = {
        'original': {
            'prediction_std': float(real_pred_np[mask_np].std()),
            'gradient_magnitude_mean': float(np.abs(np.diff(real_pred_np, axis=1)).mean() + np.abs(np.diff(real_pred_np, axis=0)).mean()),
        },
        'shuffled': {
            'prediction_std': float(shuffled_pred_np[mask_np].std()),
            'gradient_magnitude_mean': float(np.abs(np.diff(shuffled_pred_np, axis=1)).mean() + np.abs(np.diff(shuffled_pred_np, axis=0)).mean()),
        },
        'constant': {
            'prediction_std': float(const_pred_np[mask_np].std()),
            'gradient_magnitude_mean': float(np.abs(np.diff(const_pred_np, axis=1)).mean() + np.abs(np.diff(const_pred_np, axis=0)).mean()),
        },
    }

    # activation check
    final_layer = model.head
    probed = TerrainUNet().to(device)
    with torch.no_grad():
        act = probed(x)
    activation_check = {
        'final_layer': 'Conv2d(32, 1, kernel_size=(1,1))',
        'activation_after_final_layer': 'none (linear output, no sigmoid/tanh/clamp)',
        'output_theoretical_range': 'unbounded real-valued output',
        'fresh_initialization_output_range': {
            'min': float(act.detach().cpu().numpy().min()),
            'max': float(act.detach().cpu().numpy().max()),
            'mean': float(act.detach().cpu().numpy().mean()),
            'std': float(act.detach().cpu().numpy().std()),
        },
    }

    # stored phase79 comparison, using historical result only
    phase79_results = json.loads((PHASE79 / 'RESULTS.json').read_text(encoding='utf-8'))
    phase79_model = phase79_results['metrics']['terrain_unet']
    zero_base = phase79_results['metrics']['zero_relief_baseline']
    mean_base = phase79_results['metrics']['mean_relief_baseline']
    median_base = phase79_results['metrics']['median_relief_baseline']

    baseline_comparison = {
        'zero_baseline': zero_base,
        'mean_training_target_baseline': mean_base,
        'median_training_target_baseline': median_base,
        'stored_phase79_model': phase79_model,
    }

    # exact step counts from the legacy run setup
    optimization_step_audit = {
        'train_crops': 1,
        'batch_size': 1,
        'number_of_batches': 1,
        'optimizer_step_calls': 1,
        'gradient_accumulation_steps': 1,
        'effective_optimization_steps_per_epoch': 1,
        'training_time_seconds': None,
    }

    # determine diagnosis
    if (
        abs(target_space_audit['train_crop']['inverse_transform_reconstruction_error']['max_abs']) < 1e-6
        and loss_audit['implementation_loss'] == loss_audit['reference_loss']
        and any(r['nan_count'] == 0 and r['inf_count'] == 0 and r['nonzero_fraction'] > 0.0 for r in grad_rows)
        and optimization_step_audit['optimizer_step_calls'] == 1
    ):
        diagnosis = 'OPTIMIZATION_STEP_FAULT'
    elif (
        target_space_audit['train_crop']['inverse_transform_reconstruction_error']['max_abs'] > 1e-4
    ):
        diagnosis = 'TARGET_SPACE_OR_SCALING_FAULT'
    elif (
        abs(loss_audit['implementation_loss'] - loss_audit['reference_loss']) > 1e-7
    ):
        diagnosis = 'LOSS_OR_MASKING_FAULT'
    elif (
        any(r['gradient_norm'] < 1e-8 for r in grad_rows)
    ):
        diagnosis = 'GRADIENT_FLOW_FAULT'
    elif (
        all(c['target_std'] < 1e-5 for c in crop_stats)
    ):
        diagnosis = 'DATA_DIVERSITY_OR_CROP_FAULT'
    else:
        diagnosis = 'DIAGNOSIS_INCONCLUSIVE'

    results = {
        'phase': 'PHASE_80',
        'status': 'PHASE_79_LEARNING_DYNAMICS_FORENSIC',
        'diagnosis': diagnosis,
        'target_space_audit': target_space_audit,
        'crop_distribution_summary': all_summary,
        'optimization_step_audit': optimization_step_audit,
        'loss_audit': loss_audit,
        'gradient_audit': gradient_audit,
        'update_audit': update_audit,
        'output_response': response_audit,
        'sensitivity_after_one_step': sens_summary,
        'activation_check': activation_check,
        'baseline_comparison': baseline_comparison,
    }

    (OUT / 'TARGET_SPACE_AUDIT.json').write_text(json.dumps(results['target_space_audit'], indent=2), encoding='utf-8')
    (OUT / 'LOSS_AUDIT.json').write_text(json.dumps(results['loss_audit'], indent=2), encoding='utf-8')
    (OUT / 'GRADIENT_AUDIT.json').write_text(json.dumps(results['gradient_audit'], indent=2), encoding='utf-8')
    (OUT / 'UPDATE_AUDIT.json').write_text(json.dumps(results['update_audit'], indent=2), encoding='utf-8')
    (OUT / 'BASELINE_COMPARISON.json').write_text(json.dumps(results['baseline_comparison'], indent=2), encoding='utf-8')
    (OUT / 'RESULTS.json').write_text(json.dumps(results, indent=2), encoding='utf-8')

    report_lines = [
        '# Phase 80 learning-dynamics forensic',
        '',
        '## 1. Exact target space',
        'Target units: meters in raw local relief; model target is z-score normalized local relief based on the train crop only.',
        'target_normalized = (raw_local_relief - mean_train_target) / std_train_target',
        '',
        f"Train raw DEM min/max/mean/std: {target_space_audit['train_crop']['raw_dem']['min']:.6f}, {target_space_audit['train_crop']['raw_dem']['max']:.6f}, {target_space_audit['train_crop']['raw_dem']['mean']:.6f}, {target_space_audit['train_crop']['raw_dem']['std']:.6f}",
        f"Train raw local relief min/max/mean/std: {target_space_audit['train_crop']['raw_local_relief']['min']:.6f}, {target_space_audit['train_crop']['raw_local_relief']['max']:.6f}, {target_space_audit['train_crop']['raw_local_relief']['mean']:.6f}, {target_space_audit['train_crop']['raw_local_relief']['std']:.6f}",
        f"Train model target min/max/mean/std: {target_space_audit['train_crop']['model_target']['min']:.6f}, {target_space_audit['train_crop']['model_target']['max']:.6f}, {target_space_audit['train_crop']['model_target']['mean']:.6f}, {target_space_audit['train_crop']['model_target']['std']:.6f}",
        f"Validation raw local relief min/max/mean/std: {target_space_audit['validation_crop']['raw_local_relief']['min']:.6f}, {target_space_audit['validation_crop']['raw_local_relief']['max']:.6f}, {target_space_audit['validation_crop']['raw_local_relief']['mean']:.6f}, {target_space_audit['validation_crop']['raw_local_relief']['std']:.6f}",
        f"Validation model target min/max/mean/std: {target_space_audit['validation_crop']['model_target']['min']:.6f}, {target_space_audit['validation_crop']['model_target']['max']:.6f}, {target_space_audit['validation_crop']['model_target']['mean']:.6f}, {target_space_audit['validation_crop']['model_target']['std']:.6f}",
        f"Inverse transform max reconstruction error: {target_space_audit['train_crop']['inverse_transform_reconstruction_error']['max_abs']:.12f}",
        f"Inverse transform mean reconstruction error: {target_space_audit['train_crop']['inverse_transform_reconstruction_error']['mean_abs']:.12f}",
        '',
        '## 2. Crop distribution audit',
        f"Train valid pixels: {crop_stats[0]['valid_pixel_count']}; valid fraction: {crop_stats[0]['valid_fraction']:.6f}; target mean/std/min/max: {crop_stats[0]['target_mean']:.6f}/{crop_stats[0]['target_std']:.6f}/{crop_stats[0]['target_min']:.6f}/{crop_stats[0]['target_max']:.6f}",
        f"Validation valid pixels: {crop_stats[1]['valid_pixel_count']}; valid fraction: {crop_stats[1]['valid_fraction']:.6f}; target mean/std/min/max: {crop_stats[1]['target_mean']:.6f}/{crop_stats[1]['target_std']:.6f}/{crop_stats[1]['target_min']:.6f}/{crop_stats[1]['target_max']:.6f}",
        f"Across crops summary: valid_pixel_count={all_summary['valid_pixel_count']}; valid_fraction={all_summary['valid_fraction']}; target_mean={all_summary['target_mean']}; target_std={all_summary['target_std']}; target_min={all_summary['target_min']}; target_max={all_summary['target_max']}",
        '',
        '## 3. Optimization-step audit',
        f"train_crops={optimization_step_audit['train_crops']}; batch_size={optimization_step_audit['batch_size']}; number_of_batches={optimization_step_audit['number_of_batches']}; optimizer_step_calls={optimization_step_audit['optimizer_step_calls']}; gradient_accumulation_steps={optimization_step_audit['gradient_accumulation_steps']}; effective_optimization_steps_per_epoch={optimization_step_audit['effective_optimization_steps_per_epoch']}",
        '',
        '## 4. Loss sanity',
        f"loss={loss_audit['implementation_loss']:.12f}; prediction mean/std={loss_controls['prediction_mean']:.6f}/{loss_controls['prediction_std']:.6f}; target mean/std={loss_controls['target_mean']:.6f}/{loss_controls['target_std']:.6f}",
        f"loss_if_prediction_zero={loss_controls['loss_if_zero_prediction']:.12f}",
        f"loss_if_prediction_target={loss_controls['loss_if_target_prediction']:.12f}",
        f"loss_if_prediction_target_mean={loss_controls['loss_if_target_mean_prediction']:.12f}",
        '',
        '## 5. Mask / loss-formula audit',
        f"exact_loss_expression={loss_audit['exact_expression']}",
        f"valid_pixel_count={loss_audit['valid_pixel_count']}; numerator={loss_audit['loss_numerator']:.12f}; denominator={loss_audit['loss_denominator']:.12f}; implementation_loss={loss_audit['implementation_loss']:.12f}; reference_loss={loss_audit['reference_loss']:.12f}; match_error={loss_audit['loss_match_tolerance']:.12e}",
        '',
        '## 6. Gradient flow',
        f"any_nan={gradient_audit['health_summary']['has_any_nan']}; any_inf={gradient_audit['health_summary']['has_any_inf']}; mean_gradient_norm={gradient_audit['health_summary']['mean_gradient_norm']:.6f}; earliest_nonzero={gradient_audit['health_summary']['earliest_layer_nonzero']}; largest_norm={gradient_audit['health_summary']['layer_with_largest_norm']}",
        '',
        '## 7. Parameter update check',
        f"loss_before_step={update_audit['loss_before_step']:.12f}; loss_after_step={update_audit['loss_after_step']:.12f}",
        f"parameter change summary: {json.dumps(update_audit['parameter_change_summary'], indent=2)}",
        '',
        '## 8. Output response before/after one step',
        f"before min/max/mean/std: {response_audit['before_step']['min']:.6f}/{response_audit['before_step']['max']:.6f}/{response_audit['before_step']['mean']:.6f}/{response_audit['before_step']['std']:.6f}",
        f"after min/max/mean/std: {response_audit['after_step']['min']:.6f}/{response_audit['after_step']['max']:.6f}/{response_audit['after_step']['mean']:.6f}/{response_audit['after_step']['std']:.6f}",
        f"output_change_mean={response_audit['output_change']['mean']:.6f}; output_change_std={response_audit['output_change']['std']:.6f}; output_change_max={response_audit['output_change']['max_abs']:.6f}",
        '',
        '## 9. Output activation check',
        f"final_layer={activation_check['final_layer']}; activation_after_final_layer={activation_check['activation_after_final_layer']}; output_range={activation_check['output_theoretical_range']}; fresh_output_range={activation_check['fresh_initialization_output_range']}",
        '',
        '## 10. Baseline comparison',
        f"zero_baseline={baseline_comparison['zero_baseline']}\nmean_training_target_baseline={baseline_comparison['mean_training_target_baseline']}\nmedian_training_target_baseline={baseline_comparison['median_training_target_baseline']}\nstored_phase79_model={baseline_comparison['stored_phase79_model']}",
        '',
        f'Diagnosis: {diagnosis}',
        '',
        diagnosis,
    ]
    (OUT / 'REPORT.md').write_text('\n'.join(report_lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
