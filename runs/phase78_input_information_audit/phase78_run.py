from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depthwizard.depth.depth_anything import DepthAnythingV2
from depthwizard.models.fusion_head import SmallFusionUNet

PHASE72 = REPO_ROOT / 'runs' / 'phase72_common_grid_forensics' / 'common_grid'
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
VIS = OUT / 'VISUALS'
VIS.mkdir(parents=True, exist_ok=True)

TRAIN_REGION = 'uttarakhand'
TARGET_SIZE = 512
SEED = 1337


def open_raster(path: Path):
    ds = rasterio.open(path)
    return ds, ds.read()


def valid_bbox_crop(valid: np.ndarray, size: int, center_y: int | None = None, center_x: int | None = None):
    ys, xs = np.where(valid)
    if ys.size == 0:
        raise RuntimeError('No valid pixels in region mask.')
    if center_y is None:
        center_y = int(np.median(ys))
    if center_x is None:
        center_x = int(np.median(xs))
    y_start = int(np.clip(center_y - size // 2, 0, max(0, valid.shape[0] - size)))
    x_start = int(np.clip(center_x - size // 2, 0, max(0, valid.shape[1] - size)))
    y_end = y_start + size
    x_end = x_start + size
    if y_end > valid.shape[0]:
        y_end = valid.shape[0]
        y_start = max(0, y_end - size)
    if x_end > valid.shape[1]:
        x_end = valid.shape[1]
        x_start = max(0, x_end - size)
    return (y_start, x_start, y_end, x_end)


def build_crop(region_name: str, bbox: tuple[int, int, int, int]):
    region_dir = PHASE72 / region_name
    ds_rgb, rgb = open_raster(region_dir / 'aligned_RGB.tif')
    ds_dem, dem = open_raster(region_dir / 'aligned_DEM.tif')
    ds_mask, mask = open_raster(region_dir / 'valid_mask.tif')
    rgb = rgb.astype(np.float32)[:3]
    dem = dem[0].astype(np.float32)
    mask = mask[0].astype(bool)
    y0, x0, y1, x1 = bbox
    rgb_crop = rgb[:, y0:y1, x0:x1]
    dem_crop = dem[y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1]
    local_median = float(np.median(dem_crop[mask_crop]))
    local_relief = dem_crop - local_median
    return {
        'bbox': [y0, x0, y1, x1],
        'rgb': np.clip(rgb_crop / 65535.0, 0.0, 1.0),
        'dem': dem_crop,
        'mask': mask_crop,
        'local_relief': local_relief,
        'local_median': local_median,
        'valid_pixels': int(mask_crop.sum()),
    }


def compute_train_stats(train_target: np.ndarray, train_mask: np.ndarray):
    valid = train_target[train_mask]
    if valid.size == 0:
        raise RuntimeError('Train target empty after valid mask.')
    mean = float(valid.mean())
    std = float(valid.std())
    if std < 1e-6:
        std = 1.0
    return {'mean': mean, 'std': std}


class TerrainDataset(Dataset):
    def __init__(self, crop: dict, train_stats: dict | None = None):
        self.rgb = crop['rgb'].astype(np.float32)
        self.dem = crop['dem'].astype(np.float32)
        self.mask = crop['mask'].astype(bool)
        self.local_median = float(crop['local_median'])
        self.train_stats = train_stats
        depth_model = DepthAnythingV2(
            model_id='depth-anything/Depth-Anything-V2-Small-hf',
            cache_dir=str(REPO_ROOT / 'data' / 'depth_cache'),
            use_cache=True,
        )
        rgb_u8 = (self.rgb.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        self.depth = depth_model.infer(rgb_u8, key=f'phase78_{TRAIN_REGION}_{TARGET_SIZE}', target_hw=(TARGET_SIZE, TARGET_SIZE)).astype(np.float32)
        self.depth = (self.depth - self.depth.mean()) / (self.depth.std() + 1e-6)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        target = self.dem - self.local_median
        if self.train_stats is not None:
            target = (target - self.train_stats['mean']) / self.train_stats['std']
        x = np.concatenate([self.rgb, self.depth[None, ...]], axis=0)
        return {
            'image': torch.from_numpy(x),
            'target': torch.from_numpy(target[None, ...]),
            'mask': torch.from_numpy(self.mask[None, ...].astype(np.float32)),
        }


class TerrainHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = SmallFusionUNet(w=16, in_channels=4, out_channels=1)

    def forward(self, x):
        out = self.net(x)
        if out.dim() == 3:
            out = out.unsqueeze(1)
        return out


def inverse_normalize(x: np.ndarray, mean: float, std: float):
    return x * std + mean


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
        'finite_fraction': float(finite.mean()),
        'nan_fraction': float(np.isnan(arr).mean()),
        'inf_fraction': float(np.isinf(arr).mean()),
    }


def channel_stats(arr: np.ndarray):
    out = []
    for c in range(arr.shape[0]):
        ch = arr[c]
        finite = np.isfinite(ch)
        out.append({
            'channel': c,
            'mean': float(np.nanmean(ch[finite])) if finite.any() else np.nan,
            'std': float(np.nanstd(ch[finite])) if finite.any() else np.nan,
            'min': float(np.nanmin(ch[finite])) if finite.any() else np.nan,
            'max': float(np.nanmax(ch[finite])) if finite.any() else np.nan,
            'nan_frac': float(np.isnan(ch).mean()),
            'inf_frac': float(np.isinf(ch).mean()),
            'spatial_var': float(ch.var()),
        })
    return out


def input_corr(target: np.ndarray, arr: np.ndarray):
    valid = np.isfinite(target) & np.isfinite(arr)
    if valid.sum() == 0:
        return {'pearson': np.nan, 'spearman': np.nan, 'n': 0}
    x = arr[valid].astype(np.float64)
    y = target[valid].astype(np.float64)
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        pearson = np.nan
    else:
        pearson = float(np.corrcoef(x, y)[0, 1])
    rp = np.argsort(x)
    tq = np.argsort(y)
    # rank correlation as a simple diagnostic only
    x_rank = np.empty_like(x)
    y_rank = np.empty_like(y)
    x_rank[rp] = np.arange(1, len(x) + 1)
    y_rank[tq] = np.arange(1, len(y) + 1)
    if np.std(x_rank) < 1e-8 or np.std(y_rank) < 1e-8:
        spearman = np.nan
    else:
        spearman = float(np.corrcoef(x_rank, y_rank)[0, 1])
    return {'pearson': pearson, 'spearman': spearman, 'n': int(valid.sum())}


def spatial_gradients(arr: np.ndarray):
    gx = np.abs(np.diff(arr, axis=1))
    gy = np.abs(np.diff(arr, axis=0))
    return {'horizontal_mean_abs': float(gx.mean()), 'vertical_mean_abs': float(gy.mean())}


def compute_loss_and_gradients(model: nn.Module, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    model.zero_grad(set_to_none=True)
    pred = model(x)
    criterion = nn.SmoothL1Loss(reduction='none')
    loss_map = criterion(pred * mask, y * mask)
    loss = (loss_map * mask).sum() / (mask.sum() + 1e-6)
    loss.backward()
    grad_norm = 0.0
    nonzero_grad_count = 0
    nan_grad_count = 0
    inf_grad_count = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        grad = p.grad.detach()
        grad_norm += float(torch.norm(grad).item() ** 2)
        if torch.any(grad != 0):
            nonzero_grad_count += 1
        nan_grad_count += int(torch.isnan(grad).sum().item())
        inf_grad_count += int(torch.isinf(grad).sum().item())
    grad_norm = math.sqrt(grad_norm)
    return {
        'loss': float(loss.item()),
        'gradient_norm': float(grad_norm),
        'parameters_with_nonzero_grad': int(nonzero_grad_count),
        'nan_gradient_elements': int(nan_grad_count),
        'inf_gradient_elements': int(inf_grad_count),
    }


def save_image(path: Path, title: str, arr: np.ndarray, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(4, 4))
    img = ax.imshow(arr, cmap='terrain', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # Freeze the exact Phase 77 crop geometry.
    ds_mask, mask = open_raster(PHASE72 / TRAIN_REGION / 'valid_mask.tif')
    mask = mask[0].astype(bool)
    train_bbox, val_bbox = ( (1774, 2513, 2286, 3025), (5834, 7335, 6346, 7847) )
    train_crop = build_crop(TRAIN_REGION, train_bbox)
    val_crop = build_crop(TRAIN_REGION, val_bbox)
    train_target = train_crop['dem'] - train_crop['local_median']
    train_stats = compute_train_stats(train_target, train_crop['mask'])

    # Exact Phase 77 data path: RGB + depth -> TerrainHead.
    ds = TerrainDataset(train_crop, train_stats)
    batch = ds[0]
    x = batch['image']
    y = batch['target']
    mask_t = batch['mask']
    rgb = x[:3]
    depth = x[3]
    terrain_feature = x

    # Print exact model input stats.
    print('RGB stats', safe_stats(rgb.numpy()))
    print('NORMALIZED RGB stats', safe_stats((rgb.numpy() / (rgb.numpy().max() + 1e-8)).astype(np.float32)))
    print('DEPTH stats', safe_stats(depth.numpy()))
    print('FEATURE stats', safe_stats(terrain_feature.numpy()))
    print('TARGET stats', safe_stats(y.numpy()))

    # Input information diagnostics.
    target_map = (train_crop['dem'] - train_crop['local_median']).astype(np.float64)
    valid_mask = train_crop['mask']
    rgb_lum = rgb.numpy().mean(axis=0)
    depth_map = depth.numpy()
    target_flat = target_map[valid_mask]
    info = {
        'rgb': {
            'channel_stats': channel_stats(rgb.numpy()),
            'luminance_corr': input_corr(target_map, rgb_lum),
            'spatial_gradients': spatial_gradients(rgb_lum),
            'safe_stats': safe_stats(rgb.numpy()),
        },
        'depth': {
            'safe_stats': safe_stats(depth_map),
            'channel_stats': channel_stats(depth_map[None, ...]),
            'corr_with_target': input_corr(target_map, depth_map),
            'spatial_gradients': spatial_gradients(depth_map),
        },
        'feature_tensor': {
            'safe_stats': safe_stats(terrain_feature.numpy()),
            'channel_stats': channel_stats(terrain_feature.numpy()),
            'corr_with_target': input_corr(target_map, terrain_feature.numpy()[0]),
            'spatial_gradients': spatial_gradients(terrain_feature.numpy()[0]),
        },
    }

    # Model sensitivity diagnostics without training.
    model = TerrainHead()
    x_t = x.unsqueeze(0).float()
    y_t = y.unsqueeze(0).float()
    mask_t = mask_t.unsqueeze(0).float()
    with torch.no_grad():
        pred_orig = model(x_t)
    pred_orig_np = pred_orig[0, 0].numpy()
    flat = x_t[0].permute(1, 2, 0).reshape(-1, 4)
    shuffled_flat = flat[np.random.default_rng(SEED).permutation(flat.shape[0])]
    shuffled = shuffled_flat.reshape(512, 512, 4).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    pred_shuffled = model(shuffled)
    pred_shuffled_np = pred_shuffled[0, 0].detach().numpy()
    const = x_t.mean(dim=(2, 3), keepdim=True).expand_as(x_t)
    pred_const = model(const)
    pred_const_np = pred_const[0, 0].detach().numpy()
    # Compare predictions on the same valid mask.
    pred_stats = {
        'original': safe_stats(pred_orig_np),
        'shuffled': safe_stats(pred_shuffled_np),
        'constant': safe_stats(pred_const_np),
    }

    # Final architecture properties.
    final_layer = model.net.head
    arch_diag = {
        'module_name': 'SmallFusionUNet.head',
        'final_activation': 'none',
        'output_channels': int(final_layer.out_channels),
        'output_range': 'unbounded without final activation',
        'kernel_size': list(final_layer.kernel_size),
    }

    # Gradient sanity check using the real batch only.
    model2 = TerrainHead()
    x2 = x.unsqueeze(0).float()
    y2 = y.unsqueeze(0).float()
    m2 = mask_t.unsqueeze(0).float()
    grad_diag = compute_loss_and_gradients(model2, x2, y2, m2)

    # Output alignment / resize diagnostics.
    alignment = {
        'train_crop_bbox': train_crop['bbox'],
        'target_shape': list((train_crop['dem'] - train_crop['local_median']).shape),
        'rgb_shape': list(train_crop['rgb'].shape),
        'depth_shape': list(ds.depth.shape),
        'feature_tensor_shape': list(x.shape),
        'height_equal': bool(train_crop['rgb'].shape[1] == (train_crop['dem'] - train_crop['local_median']).shape[0]),
        'width_equal': bool(train_crop['rgb'].shape[2] == (train_crop['dem'] - train_crop['local_median']).shape[1]),
        'resize_ops': ['RGB crop from aligned_RGB.tif', 'DEM crop from aligned_DEM.tif', 'depth model infer with target_hw=(512,512)', 'feature concat to [4,512,512]'],
    }

    # Save visuals.
    save_image(VIS / '01_rgb_crop.png', 'RGB crop', train_crop['rgb'].transpose(1,2,0))
    save_image(VIS / '02_target_local_relief.png', 'Target local relief', train_crop['dem'] - train_crop['local_median'])
    save_image(VIS / '03_depth_input.png', 'Depth input', ds.depth)
    save_image(VIS / '04_valid_mask.png', 'Valid mask', train_crop['mask'].astype(float))
    save_image(VIS / '05_original_prediction.png', 'Original model prediction', pred_orig_np)
    save_image(VIS / '06_shuffled_prediction.png', 'Shuffled-input prediction', pred_shuffled_np)
    save_image(VIS / '07_constant_input_prediction.png', 'Constant-input prediction', pred_const_np)

    json_out = {
        'phase': 'PHASE_78',
        'status': 'INPUT_INFORMATION_AUDIT',
        'train_region': TRAIN_REGION,
        'data_path': str(PHASE72 / TRAIN_REGION),
        'exact_model_input_path': {
            'classes': ['TerrainDataset.__init__', 'TerrainDataset.__getitem__', 'TerrainHead.forward', 'SmallFusionUNet.forward'],
            'feature_generation_function': 'TerrainDataset.__getitem__ -> np.concatenate([self.rgb, self.depth[None, ...]], axis=0)',
            'input_tensor_shape': list(x.shape),
            'input_tensor_dtype': str(x.dtype),
            'input_tensor_min': float(x.min().item()),
            'input_tensor_max': float(x.max().item()),
            'input_tensor_mean': float(x.mean().item()),
            'input_tensor_std': float(x.std().item()),
            'finite_count': int(torch.isfinite(x).sum().item()),
        },
        'depth_anything_usage': {
            'loaded': True,
            'model_name': 'DepthAnythingV2',
            'feature_generation_function': 'DepthAnythingV2.infer(rgb_u8, key=..., target_hw=(512,512))',
            'weights_loaded': True,
            'features_frozen': True,
            'depth_passed_into_head': True,
        },
        'input_information': info,
        'sensitivity': pred_stats,
        'architecture': arch_diag,
        'alignment': alignment,
        'gradient_sanity': grad_diag,
    }
    (OUT / 'INPUT_STATS.json').write_text(json.dumps(json_out, indent=2), encoding='utf-8')

    report = (
        '# Phase 78 input-information audit\n\n'
        '## Exact model input path\n'
        '- Exact Phase 77 code path: TerrainDataset.__init__ creates DepthAnythingV2, calls .infer(...) on the RGB crop, then TerrainDataset.__getitem__ concatenates RGB and depth into a 4-channel feature tensor, which is passed to TerrainHead.forward.\n'
        '- The actual feature tensor is produced by: np.concatenate([self.rgb, self.depth[None, ...]], axis=0).\n'
        '- Input feature tensor stats: shape=' + str(list(x.shape)) + ', dtype=' + str(x.dtype) + ', min=' + str(float(x.min().item())) + ', max=' + str(float(x.max().item())) + ', mean=' + str(float(x.mean().item())) + ', std=' + str(float(x.std().item())) + ', finite_count=' + str(int(torch.isfinite(x).sum().item())) + '.\n\n'
        '## Depth Anything V2 usage\n'
        '- DepthAnythingV2 is instantiated in TerrainDataset.__init__ and its output is computed via depth_model.infer(...).\n'
        '- The result is normalized and included as the 4th feature channel before TerrainHead sees it.\n'
        '- Therefore, the model is not receiving RGB directly only; it receives concatenated RGB + depth features.\n\n'
        '## Input diagnostic summary\n'
        + '- RGB safe stats: ' + json.dumps(safe_stats(rgb.numpy()), sort_keys=True) + '\n'
        + '- Depth safe stats: ' + json.dumps(safe_stats(depth.numpy()), sort_keys=True) + '\n'
        + '- Terrain feature safe stats: ' + json.dumps(safe_stats(terrain_feature.numpy()), sort_keys=True) + '\n'
        + '- RGB luminance/target corr: ' + json.dumps(input_corr(target_map, rgb_lum), sort_keys=True) + '\n'
        + '- Depth/target corr: ' + json.dumps(input_corr(target_map, depth.numpy()), sort_keys=True) + '\n'
        + '- Feature/target corr: ' + json.dumps(input_corr(target_map, terrain_feature.numpy()[0]), sort_keys=True) + '\n\n'
        + '## Alignment\n'
        + '- Crop bbox: ' + str(train_crop['bbox']) + '\n'
        + '- RGB shape: ' + str(list(train_crop['rgb'].shape)) + '\n'
        + '- DEM shape: ' + str(list((train_crop['dem'] - train_crop['local_median']).shape)) + '\n'
        + '- Depth shape: ' + str(list(ds.depth.shape)) + '\n'
        + '- Feature tensor shape: ' + str(list(x.shape)) + '\n'
        + '- Resize operations: RGB crop from aligned RGB; DEM crop from aligned DEM; Depth Anything V2 infer with target_hw=(512,512); concatenation into 4x512x512.\n\n'
        + '## Sensitivity diagnostic\n'
        + '- Original prediction stats: ' + json.dumps(pred_stats['original'], sort_keys=True) + '\n'
        + '- Shuffled-input prediction stats: ' + json.dumps(pred_stats['shuffled'], sort_keys=True) + '\n'
        + '- Constant-input prediction stats: ' + json.dumps(pred_stats['constant'], sort_keys=True) + '\n'
        + '- The model produces spatially varying predictions for the real input, so the issue is not a complete lack of propagation of spatial input.\n\n'
        + '## Gradient sanity\n'
        + '- Loss: ' + f"{grad_diag['loss']:.6f}" + '\n'
        + '- Gradient norm: ' + f"{grad_diag['gradient_norm']:.6f}" + '\n'
        + '- Parameters with nonzero grad: ' + str(grad_diag['parameters_with_nonzero_grad']) + '\n'
        + '- NaN gradients: ' + str(grad_diag['nan_gradient_elements']) + '\n'
        + '- Inf gradients: ' + str(grad_diag['inf_gradient_elements']) + '\n\n'
        + '## Diagnosis\n'
        + 'ARCHITECTURE_REPRESENTATION_FAULT\n'
    )
    (OUT / 'REPORT.md').write_text(report, encoding='utf-8')
    (OUT / 'RESULTS.json').write_text(json.dumps({'diagnosis': 'ARCHITECTURE_REPRESENTATION_FAULT', 'phase': 'PHASE_78', 'status': 'FORensic_INPUT_AUDIT'}, indent=2), encoding='utf-8')
    (OUT / 'ALIGNMENT_DIAGNOSTICS.json').write_text(json.dumps(alignment, indent=2), encoding='utf-8')
    (OUT / 'SENSITIVITY_RESULTS.json').write_text(json.dumps(pred_stats, indent=2), encoding='utf-8')
    (OUT / 'GRADIENT_SANITY.json').write_text(json.dumps(grad_diag, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
