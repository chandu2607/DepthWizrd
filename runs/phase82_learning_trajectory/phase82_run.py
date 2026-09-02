from __future__ import annotations

import copy
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runs.phase79_spatial_terrain_baseline.phase79_run import (
    PHASE72,
    TRAIN_REGION,
    TerrainDataset,
    TerrainUNet,
    build_crop,
    compute_metrics,
    compute_train_stats,
    deterministic_split,
    gradient_abs_error,
    open_raster,
    save_image,
)

OUT = Path(__file__).resolve().parent
VIS = OUT / 'VISUALS'
VIS.mkdir(parents=True, exist_ok=True)
SEED = 1337
EPOCHS = 5


def relative_param_change(current_state: dict, reference_state: dict) -> float:
    deltas = []
    for name in sorted(current_state):
        cur = current_state[name].detach().float()
        ref = reference_state[name].detach().float()
        denom = ref.abs().clamp_min(1e-8)
        deltas.append((cur - ref).abs().div(denom).mean().item())
    return float(np.mean(deltas)) if deltas else 0.0


def save_epoch_map(epoch: int, pred: np.ndarray, target: np.ndarray, valid: np.ndarray, out_dir: Path):
    pred_valid = pred.copy()
    target_valid = target.copy()
    abs_err = np.abs(pred_valid - target_valid)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(target_valid, cmap='terrain')
    axes[0].set_title(f'Epoch {epoch} true relief')
    axes[0].axis('off')
    im2 = axes[1].imshow(pred_valid, cmap='terrain')
    axes[1].set_title(f'Epoch {epoch} predicted relief')
    axes[1].axis('off')
    im3 = axes[2].imshow(abs_err, cmap='magma')
    axes[2].set_title(f'Epoch {epoch} abs error')
    axes[2].axis('off')
    fig.tight_layout()
    fig.savefig(out_dir / f'epoch_{epoch}_diagnostic.png', dpi=180)
    plt.close(fig)


def evaluate_epoch(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, loss_fn: nn.Module):
    pred_np = pred.detach().cpu().numpy()[0]
    target_np = target.detach().cpu().numpy()[0, 0]
    valid_np = mask.detach().cpu().numpy()[0, 0].astype(bool)
    metrics = compute_metrics(pred_np, target_np, valid_np)
    grad = gradient_abs_error(pred_np, target_np)
    loss_map = loss_fn(pred * mask, target * mask)
    val_loss = (loss_map * mask).sum() / (mask.sum() + 1e-6)
    return {
        'val_loss': float(val_loss.item()),
        'mae': metrics['mae'],
        'rmse': metrics['rmse'],
        'pearson': metrics['pearson'],
        'prediction_mean': metrics['prediction_mean'],
        'prediction_std': metrics['prediction_std'],
        'target_std': metrics['target_std'],
        'mean_bias': metrics['mean_bias'],
        'gradient_x_mae': grad['gradient_x_mae'],
        'gradient_y_mae': grad['gradient_y_mae'],
        'prediction_variance': float(np.var(pred_np[valid_np])),
        'prediction_spatial_gradient_magnitude': float(0.5 * (np.mean(np.abs(np.diff(pred_np, axis=1))) + np.mean(np.abs(np.diff(pred_np, axis=0))))),
        'correlation': metrics['pearson'],
    }


def classify(history: list[dict], baseline_summary: dict) -> str:
    pearson_values = [row['validation_pearson'] for row in history if row['validation_pearson'] is not None and not math.isnan(row['validation_pearson'])]
    std_ratios = [row['prediction_std'] / max(row['target_std'], 1e-8) for row in history]
    max_pearson = max(pearson_values) if pearson_values else 0.0
    max_std_ratio = max(std_ratios) if std_ratios else 0.0
    last_pearson = history[-1]['validation_pearson'] if history and history[-1]['validation_pearson'] is not None else 0.0
    last_std_ratio = history[-1]['prediction_std'] / max(history[-1]['target_std'], 1e-8) if history else 0.0

    if max_pearson > 0.2 and max_std_ratio > 0.3 and last_pearson > 0.1:
        return 'MULTI_EPOCH_SPATIAL_LEARNING_EMERGES'
    if max_pearson <= 0.05 and max_std_ratio < 0.2 and all((row['validation_pearson'] is None or math.isnan(row['validation_pearson']) or row['validation_pearson'] <= 0.05) for row in history):
        return 'MULTI_EPOCH_MODEL_COLLAPSE'
    return 'MULTI_EPOCH_RESULT_INCONCLUSIVE'


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, mask = open_raster(PHASE72 / TRAIN_REGION / 'valid_mask.tif')
    mask = mask[0].astype(bool)
    train_bbox, val_bbox = deterministic_split(mask)
    train_crop = build_crop(TRAIN_REGION, train_bbox)
    val_crop = build_crop(TRAIN_REGION, val_bbox)

    train_target = train_crop['dem'] - train_crop['local_median']
    val_target = val_crop['dem'] - val_crop['local_median']
    train_stats = compute_train_stats(train_target, train_crop['mask'])

    train_ds = TerrainDataset(train_crop, train_stats)
    val_ds = TerrainDataset(val_crop, train_stats)
    loader_train = DataLoader(train_ds, batch_size=1, shuffle=False)
    loader_val = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = TerrainUNet().to(device)
    criterion = nn.SmoothL1Loss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    initial_state = copy.deepcopy(model.state_dict())
    previous_state = copy.deepcopy(model.state_dict())
    previous_epoch_train_loss = None

    # Keep exact Phase 77 baselines, computed once using training mean/median,
    # and the zero baseline on validation set is a trivial constant baseline.
    train_valid = train_target[train_crop['mask']]
    zero_pred = np.zeros_like(val_target, dtype=np.float32)
    mean_pred = np.full_like(val_target, float(train_valid.mean()), dtype=np.float32)
    median_pred = np.full_like(val_target, float(np.median(train_valid)), dtype=np.float32)
    baseline_methods = {
        'zero_baseline': compute_metrics(zero_pred, val_target, val_crop['mask']),
        'training_mean_baseline': compute_metrics(mean_pred, val_target, val_crop['mask']),
        'training_median_baseline': compute_metrics(median_pred, val_target, val_crop['mask']),
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_train_losses = []
        for batch in loader_train:
            x = batch['image'].to(device, dtype=torch.float32)
            y = batch['target'].to(device, dtype=torch.float32)
            m = batch['mask'].to(device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss_map = criterion(pred * m, y * m)
            loss = (loss_map * m).sum() / (m.sum() + 1e-6)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite training loss at epoch {epoch}.')
            epoch_train_losses.append(float(loss.detach().cpu().item()))
            loss.backward()
            optimizer.step()

        train_loss = float(np.mean(epoch_train_losses))

        model.eval()
        with torch.no_grad():
            val_batch = next(iter(loader_val))
            x_val = val_batch['image'].to(device, dtype=torch.float32)
            y_val = val_batch['target'].to(device, dtype=torch.float32)
            m_val = val_batch['mask'].to(device, dtype=torch.float32)
            pred_val = model(x_val)
            eval_data = evaluate_epoch(pred_val, y_val, m_val, criterion)

        param_rel_change = relative_param_change(model.state_dict(), previous_state)
        loss_reduction = None if previous_epoch_train_loss is None else previous_epoch_train_loss - train_loss

        epoch_record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'validation_loss': eval_data['val_loss'],
            'validation_mae': eval_data['mae'],
            'validation_rmse': eval_data['rmse'],
            'validation_pearson': eval_data['pearson'],
            'prediction_mean': eval_data['prediction_mean'],
            'prediction_std': eval_data['prediction_std'],
            'target_std': eval_data['target_std'],
            'mean_bias': eval_data['mean_bias'],
            'gradient_x_mae': eval_data['gradient_x_mae'],
            'gradient_y_mae': eval_data['gradient_y_mae'],
            'prediction_variance': eval_data['prediction_variance'],
            'prediction_spatial_gradient_magnitude': eval_data['prediction_spatial_gradient_magnitude'],
            'correlation': eval_data['correlation'],
            'prediction_std_over_target_std': eval_data['prediction_std'] / max(eval_data['target_std'], 1e-8),
            'loss_reduction': loss_reduction,
            'selected_parameter_relative_change': param_rel_change,
        }
        history.append(epoch_record)

        if epoch in {1, 3, 5}:
            pred_np = pred_val.detach().cpu().numpy()[0]
            target_np = y_val.detach().cpu().numpy()[0, 0]
            valid_np = m_val.detach().cpu().numpy()[0, 0].astype(bool)
            save_epoch_map(epoch, pred_np, target_np, valid_np, VIS)

        previous_state = copy.deepcopy(model.state_dict())
        previous_epoch_train_loss = train_loss

    history_path = OUT / 'EPOCH_HISTORY.csv'
    with history_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'epoch', 'train_loss', 'validation_loss', 'validation_mae', 'validation_rmse', 'validation_pearson',
            'prediction_mean', 'prediction_std', 'target_std', 'mean_bias', 'gradient_x_mae', 'gradient_y_mae',
            'prediction_variance', 'prediction_spatial_gradient_magnitude', 'correlation', 'prediction_std_over_target_std',
            'loss_reduction', 'selected_parameter_relative_change'
        ])
        writer.writeheader()
        for row in history:
            writer.writerow(row)

    final_label = classify(history, baseline_methods)

    # Save outputs
    results = {
        'phase': 'PHASE_82',
        'status': 'CONTROLLED_TERRAIN_LEARNING_TRAJECTORY',
        'constraint': {
            'epochs': 5,
            'same_phase79_setup': True,
            'held_out_uttarakhand_validation': True,
            'local_relief_target': 'DEM - median(valid DEM in crop)',
        },
        'baseline_comparison': baseline_methods,
        'epoch_history': history,
        'final_label': final_label,
    }
    (OUT / 'RESULTS.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    (OUT / 'BASELINE_COMPARISON.json').write_text(json.dumps(baseline_methods, indent=2), encoding='utf-8')

    # Quick summary for report
    report_lines = [
        '# Phase 82 controlled terrain learning trajectory',
        '',
        '## Setup',
        '- Exact same frozen Phase 72 Uttarakhand data, split, local-relief target, TerrainUNet, optimizer, LR, masks, and normalization as Phase 79.',
        '- Only variation: 5 epochs, same fixed Uttarakhand validation region, same valid-pixel-only evaluation.',
        '',
        '## Baselines (fixed, not recomputed from validation)',
    ]
    for name, metrics in baseline_methods.items():
        report_lines.append(
            f"- {name}: MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}, Pearson={metrics['pearson'] if metrics['pearson'] is not None and not math.isnan(metrics['pearson']) else 'nan'}, prediction_mean={metrics['prediction_mean']:.6f}, prediction_std={metrics['prediction_std']:.6f}, target_std={metrics['target_std']:.6f}, mean_bias={metrics['mean_bias']:.6f}"
        )

    report_lines.extend([
        '',
        '## Epoch history',
    ])
    for row in history:
        report_lines.append(
            f"- epoch {row['epoch']}: train_loss={row['train_loss']:.6f}, val_loss={row['validation_loss']:.6f}, val_mae={row['validation_mae']:.6f}, val_rmse={row['validation_rmse']:.6f}, pearson={row['validation_pearson'] if row['validation_pearson'] is not None and not math.isnan(row['validation_pearson']) else 'nan'}, pred_mean={row['prediction_mean']:.6f}, pred_std={row['prediction_std']:.6f}, target_std={row['target_std']:.6f}, mean_bias={row['mean_bias']:.6f}, grad_x_mae={row['gradient_x_mae']:.6f}, grad_y_mae={row['gradient_y_mae']:.6f}, std_ratio={row['prediction_std_over_target_std']:.6f}, param_rel_change={row['selected_parameter_relative_change']:.6f}"
        )

    report_lines.extend([
        '',
        '## Interpretation',
        f'- Max Pearson across 5 epochs: {max(r["validation_pearson"] for r in history if r["validation_pearson"] is not None)}',
        f'- Final Pearson: {history[-1]["validation_pearson"]}',
        f'- Final prediction std / target std: {history[-1]["prediction_std_over_target_std"]:.6f}',
        f'- Final gradient-X MAE: {history[-1]["gradient_x_mae"]:.6f}',
        f'- Final gradient-Y MAE: {history[-1]["gradient_y_mae"]:.6f}',
        '',
        '## Final label',
        final_label,
    ])
    (OUT / 'REPORT.md').write_text('\n'.join(report_lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
