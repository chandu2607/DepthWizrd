from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runs.phase79_spatial_terrain_baseline.phase79_run import (
    TRAIN_REGION,
    build_crop,
    deterministic_split,
    TerrainDataset,
    TerrainUNet,
    compute_train_stats,
    open_raster,
    PHASE72,
)

OUT = Path(__file__).resolve().parent
VIS = OUT / 'VISUALS'
VIS.mkdir(parents=True, exist_ok=True)


def to_4d(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 3:
        return t.unsqueeze(1)
    return t


def actual_phase79_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred4 = to_4d(pred)
    target4 = to_4d(target)
    mask4 = to_4d(mask)
    criterion = nn.SmoothL1Loss(reduction='none')
    loss_map = criterion(pred4 * mask4, target4 * mask4)
    return (loss_map * mask4).sum() / (mask4.sum() + 1e-6)


def reference_masked_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred4 = to_4d(pred)
    target4 = to_4d(target)
    mask4 = to_4d(mask)
    elementwise = nn.SmoothL1Loss(reduction='none')(pred4, target4)
    return (elementwise * mask4).sum() / (mask4.sum() + 1e-6)


def grad_summary(model: nn.Module):
    rows = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad
        rows.append({
            'name': name,
            'gradient_norm': float(torch.linalg.norm(g).item()),
            'gradient_mean': float(g.mean().item()),
            'gradient_std': float(g.std(unbiased=False).item()),
            'nonzero_fraction': float((g != 0).float().mean().item()),
        })
    return rows


def make_visuals(rgb, target, mask, loss_map, out_dir: Path):
    rgb_np = rgb.transpose(1, 2, 0)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].imshow(rgb_np)
    axes[0, 0].set_title('RGB')
    axes[0, 0].axis('off')

    im1 = axes[0, 1].imshow(target, cmap='terrain')
    axes[0, 1].set_title('Target')
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    axes[0, 1].axis('off')

    im2 = axes[1, 0].imshow(mask.astype(np.float32), cmap='gray')
    axes[1, 0].set_title('Mask')
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    axes[1, 0].axis('off')

    im3 = axes[1, 1].imshow(loss_map, cmap='magma')
    axes[1, 1].set_title('Loss map')
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    axes[1, 1].axis('off')

    fig.tight_layout()
    fig.savefig(out_dir / 'diagnostic_grid.png', dpi=200)
    plt.close(fig)


def main():
    torch.manual_seed(1337)
    np.random.seed(1337)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, mask0 = open_raster(PHASE72 / TRAIN_REGION / 'valid_mask.tif')
    mask0 = mask0[0].astype(bool)
    train_bbox, _ = deterministic_split(mask0)
    train_crop = build_crop(TRAIN_REGION, train_bbox)

    train_target = train_crop['dem'] - train_crop['local_median']
    train_stats = compute_train_stats(train_target, train_crop['mask'])
    train_ds = TerrainDataset(train_crop, train_stats)
    batch = next(iter(DataLoader(train_ds, batch_size=1, shuffle=False)))

    x = batch['image'].to(device, dtype=torch.float32)
    y = batch['target'].to(device, dtype=torch.float32)
    m = batch['mask'].to(device, dtype=torch.float32)

    model = TerrainUNet().to(device)
    with torch.no_grad():
        pred = model(x)

    pred4 = to_4d(pred)
    y4 = to_4d(y)
    m4 = to_4d(m)
    criterion = nn.SmoothL1Loss(reduction='none')

    actual_loss_map = criterion(pred4 * m4, y4 * m4)
    actual_loss = (actual_loss_map * m4).sum() / (m4.sum() + 1e-6)
    elementwise_loss = criterion(pred4, y4)
    reference_loss = (elementwise_loss * m4).sum() / (m4.sum() + 1e-6)

    A_correct = reference_loss
    B_current = actual_loss
    C_mask_before_smoothl1 = criterion(pred4 * m4, y4 * m4).mean()
    D_unmasked = criterion(pred4, y4).mean()
    valid_mask = m4.bool()
    E_valid_pixels_only = criterion(pred4[valid_mask], y4[valid_mask]).mean()

    total_pixels = int(m4.numel())
    valid_pixels = int(m4.sum().item())
    invalid_pixels = total_pixels - valid_pixels
    valid_fraction = valid_pixels / total_pixels
    valid_contribution = float((elementwise_loss * m4).sum().item() / (m4.sum().item() + 1e-6))
    invalid_contribution = float((elementwise_loss * (1.0 - m4)).sum().item() / (m4.sum().item() + 1e-6))

    unique_values = np.unique(m4.detach().cpu().numpy().reshape(-1)).tolist()
    mask_audit = {
        'mask_dtype': str(m4.dtype),
        'mask_min': float(m4.min().item()),
        'mask_max': float(m4.max().item()),
        'unique_values': [float(v) for v in unique_values],
        'nonzero_count': int((m4 > 0).sum().item()),
        'total_pixels': total_pixels,
        'valid_pixels': valid_pixels,
        'invalid_pixels': invalid_pixels,
        'valid_fraction': valid_fraction,
        'shape': list(m4.shape),
    }

    valid_pred = pred4[valid_mask].detach().cpu().numpy()
    valid_target = y4[valid_mask].detach().cpu().numpy()
    abs_err = np.abs(valid_pred - valid_target)
    elem_loss_valid = criterion(pred4[valid_mask], y4[valid_mask]).detach().cpu().numpy()
    loss_scale = {
        'target_mean': float(valid_target.mean()),
        'target_std': float(valid_target.std()),
        'prediction_mean': float(valid_pred.mean()),
        'prediction_std': float(valid_pred.std()),
        'absolute_error_mean': float(abs_err.mean()),
        'absolute_error_std': float(abs_err.std()),
        'smoothl1_elementwise_mean': float(elem_loss_valid.mean()),
        'smoothl1_elementwise_std': float(elem_loss_valid.std()),
        'smoothl1_elementwise_max': float(elem_loss_valid.max()),
    }

    zero_pred = torch.zeros_like(pred4)
    mean_pred = torch.full_like(pred4, float(valid_target.mean()))
    constant_check = {
        'loss_constant_zero': float(reference_masked_loss(zero_pred, y4, m4).item()),
        'loss_constant_target_mean': float(reference_masked_loss(mean_pred, y4, m4).item()),
        'loss_current_model': float(reference_loss.item()),
        'delta_model_vs_zero': float((reference_loss - reference_masked_loss(zero_pred, y4, m4)).item()),
        'delta_model_vs_mean': float((reference_loss - reference_masked_loss(mean_pred, y4, m4)).item()),
    }

    model_actual = TerrainUNet().to(device)
    model_ref = TerrainUNet().to(device)
    model_ref.load_state_dict(copy.deepcopy(model_actual.state_dict()))
    pred_a = model_actual(x)
    pred_r = model_ref(x)
    loss_a = actual_phase79_loss(pred_a, y, m)
    loss_r = reference_masked_loss(pred_r, y, m)
    model_actual.zero_grad(set_to_none=True)
    loss_a.backward()
    actual_grads = grad_summary(model_actual)
    model_ref.zero_grad(set_to_none=True)
    loss_r.backward()
    ref_grads = grad_summary(model_ref)

    actual_map = {g['name']: g for g in actual_grads}
    ref_map = {g['name']: g for g in ref_grads}
    grad_compare = []
    for name in sorted(set(actual_map) | set(ref_map)):
        a = actual_map.get(name)
        r = ref_map.get(name)
        if a is None or r is None:
            continue
        grad_compare.append({
            'name': name,
            'actual_gradient_norm': a['gradient_norm'],
            'reference_gradient_norm': r['gradient_norm'],
            'actual_gradient_mean': a['gradient_mean'],
            'reference_gradient_mean': r['gradient_mean'],
            'actual_gradient_std': a['gradient_std'],
            'reference_gradient_std': r['gradient_std'],
            'actual_nonzero_fraction': a['nonzero_fraction'],
            'reference_nonzero_fraction': r['nonzero_fraction'],
            'gradient_difference_norm': float(abs(a['gradient_norm'] - r['gradient_norm'])),
        })

    model_a = TerrainUNet().to(device)
    model_b = TerrainUNet().to(device)
    model_b.load_state_dict(copy.deepcopy(model_a.state_dict()))
    x0 = x.clone(); y0 = y.clone(); m0 = m.clone()
    pred_a0 = model_a(x0)
    pred_b0 = model_b(x0)
    before_a = actual_phase79_loss(pred_a0, y0, m0)
    before_b = reference_masked_loss(pred_b0, y0, m0)
    opt_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)
    opt_a.zero_grad(set_to_none=True)
    before_a.backward(); opt_a.step()
    opt_b.zero_grad(set_to_none=True)
    before_b.backward(); opt_b.step()
    pred_a1 = model_a(x0)
    pred_b1 = model_b(x0)
    after_a = actual_phase79_loss(pred_a1, y0, m0)
    after_b = reference_masked_loss(pred_b1, y0, m0)

    param_change_summary = []
    a_params = {k: v.detach().clone() for k, v in model_a.named_parameters()}
    b_params = {k: v.detach().clone() for k, v in model_b.named_parameters()}
    for name in sorted(a_params):
        diff = (a_params[name] - b_params[name]).abs().mean().item()
        param_change_summary.append({'name': name, 'abs_parameter_change_mean': float(diff)})

    one_step_control = {
        'MODEL_A_current_loss': {
            'loss_before': float(before_a.item()),
            'loss_after': float(after_a.item()),
            'prediction_mean_before': float(pred_a0.detach().cpu().numpy().mean()),
            'prediction_mean_after': float(pred_a1.detach().cpu().numpy().mean()),
            'prediction_std_before': float(pred_a0.detach().cpu().numpy().std()),
            'prediction_std_after': float(pred_a1.detach().cpu().numpy().std()),
            'gradient_magnitude_mean_before': float(np.abs(np.diff(pred_a0.detach().cpu().numpy()[0], axis=1)).mean() + np.abs(np.diff(pred_a0.detach().cpu().numpy()[0], axis=0)).mean()),
            'gradient_magnitude_mean_after': float(np.abs(np.diff(pred_a1.detach().cpu().numpy()[0], axis=1)).mean() + np.abs(np.diff(pred_a1.detach().cpu().numpy()[0], axis=0)).mean()),
        },
        'MODEL_B_reference_loss': {
            'loss_before': float(before_b.item()),
            'loss_after': float(after_b.item()),
            'prediction_mean_before': float(pred_b0.detach().cpu().numpy().mean()),
            'prediction_mean_after': float(pred_b1.detach().cpu().numpy().mean()),
            'prediction_std_before': float(pred_b0.detach().cpu().numpy().std()),
            'prediction_std_after': float(pred_b1.detach().cpu().numpy().std()),
            'gradient_magnitude_mean_before': float(np.abs(np.diff(pred_b0.detach().cpu().numpy()[0], axis=1)).mean() + np.abs(np.diff(pred_b0.detach().cpu().numpy()[0], axis=0)).mean()),
            'gradient_magnitude_mean_after': float(np.abs(np.diff(pred_b1.detach().cpu().numpy()[0], axis=1)).mean() + np.abs(np.diff(pred_b1.detach().cpu().numpy()[0], axis=0)).mean()),
        },
        'parameter_change_summary': param_change_summary,
    }

    exact_loss_report = {
        'loss_class': 'nn.SmoothL1Loss',
        'reduction_mode': 'none',
        'exact_expression': 'loss_map = criterion(pred * mask, target * mask); loss = (loss_map * mask).sum() / (mask.sum() + 1e-6)',
        'mask_shape': list(m4.shape),
        'prediction_shape': list(pred4.shape),
        'target_shape': list(y4.shape),
        'mask_dtype': str(m4.dtype),
        'prediction_dtype': str(pred4.dtype),
        'target_dtype': str(y4.dtype),
        'valid_pixel_count': valid_pixels,
        'total_pixel_count': total_pixels,
        'implementation_loss': float(actual_loss.item()),
        'reference_loss': float(reference_loss.item()),
        'absolute_difference': float(abs(actual_loss.item() - reference_loss.item())),
        'relative_difference': float(abs(actual_loss.item() - reference_loss.item()) / (abs(reference_loss.item()) + 1e-8)),
    }

    controls = {
        'A_correct': float(A_correct.item()),
        'B_current': float(B_current.item()),
        'C_mask_before_smoothl1': float(C_mask_before_smoothl1.item()),
        'D_unmasked': float(D_unmasked.item()),
        'E_valid_pixels_only': float(E_valid_pixels_only.item()),
    }

    final_label = 'LOSS_IMPLEMENTATION_IS_CORRECT' if abs(actual_loss.item() - reference_loss.item()) <= 1e-8 else 'ACTUAL_LOSS_IMPLEMENTATION_BUG'

    results = {
        'phase': 'PHASE_81',
        'status': 'LOSS_MASK_ISOLATION',
        'final_label': final_label,
        'loss_report': exact_loss_report,
        'mask_audit': mask_audit,
        'loss_controls': controls,
        'invalid_pixel_impact': {
            'total_pixels': total_pixels,
            'valid_pixels': valid_pixels,
            'invalid_pixels': invalid_pixels,
            'valid_fraction': valid_fraction,
            'intended_valid_contribution': valid_contribution,
            'invalid_pixels_contribution': invalid_contribution,
        },
        'gradient_comparison': grad_compare,
        'one_step_control': one_step_control,
        'constant_predictor_check': constant_check,
        'loss_scale_audit': loss_scale,
        'spatial_alignment': {
            'mask_shape': list(m4.shape),
            'target_shape': list(y4.shape),
            'prediction_shape': list(pred4.shape),
            'bbox_of_valid_pixels': {
                'x_min': int(np.where(m4[0, 0].cpu().numpy() > 0)[1].min()),
                'x_max': int(np.where(m4[0, 0].cpu().numpy() > 0)[1].max()),
                'y_min': int(np.where(m4[0, 0].cpu().numpy() > 0)[0].min()),
                'y_max': int(np.where(m4[0, 0].cpu().numpy() > 0)[0].max()),
            },
        },
    }

    (OUT / 'LOSS_COMPARISON.json').write_text(json.dumps({
        'implementation_loss': exact_loss_report['implementation_loss'],
        'reference_loss': exact_loss_report['reference_loss'],
        'absolute_difference': exact_loss_report['absolute_difference'],
        'relative_difference': exact_loss_report['relative_difference'],
        'controls': controls,
        'exact_expression': exact_loss_report['exact_expression'],
    }, indent=2), encoding='utf-8')
    (OUT / 'MASK_AUDIT.json').write_text(json.dumps(mask_audit, indent=2), encoding='utf-8')
    (OUT / 'GRADIENT_COMPARISON.json').write_text(json.dumps({'gradient_comparison': grad_compare}, indent=2), encoding='utf-8')
    (OUT / 'ONE_STEP_CONTROL.json').write_text(json.dumps(one_step_control, indent=2), encoding='utf-8')
    (OUT / 'RESULTS.json').write_text(json.dumps(results, indent=2), encoding='utf-8')

    make_visuals(train_crop['rgb'], y4.detach().cpu().numpy()[0, 0], m4.detach().cpu().numpy()[0, 0].astype(bool), actual_loss_map.detach().cpu().numpy()[0, 0], VIS)

    report_lines = [
        '# Phase 81 exact loss and mask forensics',
        '',
        '## 1. Exact Phase 79 loss implementation',
        f"- loss class/function: {exact_loss_report['loss_class']}",
        f"- reduction mode: {exact_loss_report['reduction_mode']}",
        f"- exact expression: {exact_loss_report['exact_expression']}",
        f"- mask shape: {exact_loss_report['mask_shape']}",
        f"- prediction shape: {exact_loss_report['prediction_shape']}",
        f"- target shape: {exact_loss_report['target_shape']}",
        f"- mask dtype: {exact_loss_report['mask_dtype']}",
        f"- prediction dtype: {exact_loss_report['prediction_dtype']}",
        f"- target dtype: {exact_loss_report['target_dtype']}",
        f"- valid-pixel count: {exact_loss_report['valid_pixel_count']}",
        f"- total-pixel count: {exact_loss_report['total_pixel_count']}",
        '',
        '## 2. Manual reference loss vs implementation',
        f"- implementation_loss: {exact_loss_report['implementation_loss']:.12f}",
        f"- reference_loss: {exact_loss_report['reference_loss']:.12f}",
        f"- absolute_difference: {exact_loss_report['absolute_difference']:.12f}",
        f"- relative_difference: {exact_loss_report['relative_difference']:.12e}",
        '',
        '## 3. Alternative mask bug controls',
        f"A correct: {controls['A_correct']:.12f}",
        f"B current implementation: {controls['B_current']:.12f}",
        f"C mask prediction and target before SmoothL1: {controls['C_mask_before_smoothl1']:.12f}",
        f"D unmasked SmoothL1: {controls['D_unmasked']:.12f}",
        f"E valid pixels only: {controls['E_valid_pixels_only']:.12f}",
        '',
        '## 4. Invalid-pixel impact',
        f"- total pixels: {mask_audit['total_pixels']}",
        f"- valid pixels: {mask_audit['valid_pixels']}",
        f"- invalid pixels: {mask_audit['invalid_pixels']}",
        f"- valid fraction: {mask_audit['valid_fraction']:.6f}",
        f"- intended valid contribution: {results['invalid_pixel_impact']['intended_valid_contribution']:.12f}",
        f"- invalid contribution: {results['invalid_pixel_impact']['invalid_pixels_contribution']:.12f}",
        '',
        '## 5. Mask binary audit',
        f"- mask dtype: {mask_audit['mask_dtype']}",
        f"- mask min: {mask_audit['mask_min']}",
        f"- mask max: {mask_audit['mask_max']}",
        f"- unique values: {mask_audit['unique_values']}",
        f"- nonzero count: {mask_audit['nonzero_count']}",
        '',
        '## 6. Spatial alignment',
        f"- mask shape: {results['spatial_alignment']['mask_shape']}",
        f"- target shape: {results['spatial_alignment']['target_shape']}",
        f"- prediction shape: {results['spatial_alignment']['prediction_shape']}",
        f"- bbox of valid pixels: {results['spatial_alignment']['bbox_of_valid_pixels']}",
        '',
        '## 7. Loss scale audit',
        f"- target mean/std: {loss_scale['target_mean']:.6f}/{loss_scale['target_std']:.6f}",
        f"- prediction mean/std: {loss_scale['prediction_mean']:.6f}/{loss_scale['prediction_std']:.6f}",
        f"- absolute error mean/std: {loss_scale['absolute_error_mean']:.6f}/{loss_scale['absolute_error_std']:.6f}",
        f"- SmoothL1 elementwise mean/std/max: {loss_scale['smoothl1_elementwise_mean']:.6f}/{loss_scale['smoothl1_elementwise_std']:.6f}/{loss_scale['smoothl1_elementwise_max']:.6f}",
        '',
        '## 8. Constant predictor comparison',
        f"- loss(constant=0): {constant_check['loss_constant_zero']:.12f}",
        f"- loss(constant=target mean): {constant_check['loss_constant_target_mean']:.12f}",
        f"- loss(current model): {constant_check['loss_current_model']:.12f}",
        f"- delta current vs zero: {constant_check['delta_model_vs_zero']:.12f}",
        f"- delta current vs mean: {constant_check['delta_model_vs_mean']:.12f}",
        '',
        '## 9. Gradient comparison',
        f"- gradient comparison rows: {json.dumps(grad_compare[:5], indent=2)}",
        '',
        '## 10. One-step control',
        f"- MODEL_A loss_before/after: {one_step_control['MODEL_A_current_loss']['loss_before']:.12f} -> {one_step_control['MODEL_A_current_loss']['loss_after']:.12f}",
        f"- MODEL_B loss_before/after: {one_step_control['MODEL_B_reference_loss']['loss_before']:.12f} -> {one_step_control['MODEL_B_reference_loss']['loss_after']:.12f}",
        '',
        '## 11. Final classification',
        final_label,
        '',
        final_label,
    ]
    (OUT / 'REPORT.md').write_text('\n'.join(report_lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
