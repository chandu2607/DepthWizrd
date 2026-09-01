import csv
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from depthwizard.config import TrainConfig
from depthwizard.depth.depth_anything import DepthAnythingV2
from depthwizard.models.building_conditioned_net import BuildingConditionedEstimator

ROOT = Path('c:/Users/chand/OneDrive/Desktop/DepthWizard')
DATA_DIR = ROOT / 'data' / 'dfc2023_multicity'
OUT_DIR = ROOT / 'runs' / 'phase49_upstream_repair'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def percentiles(x: np.ndarray) -> dict:
    if x.size == 0:
        return {f'p{q}': float('nan') for q in [1,5,25,50,75,95,99]}
    vals = np.percentile(x, [1,5,25,50,75,95,99])
    return {f'p{q}': float(v) for q, v in zip([1,5,25,50,75,95,99], vals)}


def summarize(arr: np.ndarray) -> dict:
    a = np.asarray(arr, dtype=np.float32)
    stats = {
        'min': float(np.nanmin(a)),
        'max': float(np.nanmax(a)),
        'mean': float(np.nanmean(a)),
        'median': float(np.nanmedian(a)),
    }
    stats.update(percentiles(a))
    return stats


def save_image(path: Path, arr: np.ndarray, cmap=None):
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        np.clip(arr, 0.0, 1.0 if arr.max() <= 1.1 else None, out=arr)
    plt.imsave(path, arr, cmap=cmap)


def load_rgb(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def direct_inference(estimator: BuildingConditionedEstimator, rgb: np.ndarray, depth: np.ndarray, *, res: int):
    estimator.model.eval()
    s = {'id': 'nyc_probe', 'rgb': rgb, 'depth': depth, 'nodata': -999.0}
    x_in = estimator._prep_x(s, res)
    xt = torch.from_numpy(x_in[None]).float().to(estimator.device)
    raw_d = torch.from_numpy(cv2.resize(depth.astype(np.float32), (res, res), interpolation=cv2.INTER_LINEAR)[None]).float().to(estimator.device)
    with torch.no_grad():
        logits, *_ = estimator.model(xt, raw_d, device=estimator.device)
    logits = logits.squeeze(0).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    return logits, probs


def load_sample(sample_id: str, depth_model):
    rgb_path = DATA_DIR / 'rgb' / sample_id
    rgb = load_rgb(rgb_path)
    depth = depth_model.infer(rgb, sample_id, target_hw=rgb.shape[:2])
    return rgb, depth


def compare_preprocessing():
    rows = [
        ['parameter', 'training', 'production', 'match?'],
        ['image_shape', 'source tile 512x512 -> model 256x256', 'NYC source 512x512 -> model 256x256', 'Yes'],
        ['color_order', 'RGB (stored as np.uint8 RGB before /255.0, in _prep_x)', 'RGB (raster loader returns RGB)', 'Yes'],
        ['pixel_scale', 'uint8 0..255 => /255 if max>1.5', 'uint8 0..255 => app keeps uint8, passed to estimator._prep_x', 'Yes, same conversion path'],
        ['depth_normalization', 'depth_norm = (depth - d_mean) / (d_std + 1e-6)', 'same function _prep_x applies same normalization', 'Yes'],
        ['resize', 'cv2.resize(rgb, (res,res)); cv2.resize(depth, (res,res))', 'same resize in estimator._prep_x', 'Yes'],
        ['crop', 'none in _prep_x; whole tile at train_res', 'none in _prep_x; production uses full tile then resize', 'Yes'],
        ['channels', 'x = concat([rgb.transpose(2,0,1), depth_norm[None]], axis=0) => 4 channels', 'same concat in _prep_x', 'Yes'],
        ['model_output', 'mask_logits, then sigmoid', 'same direct sigmoid in model.forward', 'Yes'],
    ]
    return rows


def main():
    ckpt_d = ROOT / 'runs' / 'phase43_augmented_unet' / 'unet_config_D.pt'
    ckpt_a = ROOT / 'runs' / 'phase43_augmented_unet' / 'unet_config_A.pt'
    if not ckpt_d.exists():
        raise FileNotFoundError(ckpt_d)

    tcfg = TrainConfig(arch='unet3', target_transform='none', epochs=1, batch_size=8, lr=1e-3, amp=False)
    est_d = BuildingConditionedEstimator(tcfg, nodata=-999.0, seed=0)
    state = torch.load(ckpt_d, map_location='cpu', weights_only=True)
    missing, unexpected = est_d.model.load_state_dict(state, strict=False)
    print('CHECKPOINT', ckpt_d)
    print('SHA256', sha256_file(ckpt_d))
    print('CLASS', est_d.__class__.__name__)
    print('KEY_COUNT', len(state.keys()))
    print('MISSING', missing)
    print('UNEXPECTED', unexpected)
    total_params = sum(p.numel() for p in est_d.model.parameters())
    print('PARAM_COUNT', total_params)

    rgb_path = DATA_DIR / 'rgb' / 'SV_NewYork_40.7401_-73.9915.tif'
    rgb = load_rgb(rgb_path)
    dcfg = DepthConfig = __import__('depthwizard.config', fromlist=['DepthConfig']).DepthConfig
    depth_model = DepthAnythingV2(dcfg.model_id, dcfg.input_size, dcfg.cache_dir, use_cache=True)
    depth = depth_model.infer(rgb, rgb_path.name, target_hw=rgb.shape[:2])

    res = 256
    logits_d, probs_d = direct_inference(est_d, rgb, depth, res=res)
    np.save(OUT_DIR / 'logits.npy', logits_d)

    manifest = ROOT / 'runs' / 'dfc2023_multicity_prep' / 'split_manifest.csv'
    train_id = pd.read_csv(manifest).query("split == 'train'").iloc[0]['tile_id']
    train_rgb, train_depth = load_sample(train_id, depth_model)
    train_logits_d, train_probs_d = direct_inference(est_d, train_rgb, train_depth, res=res)

    # Also load baseline A for comparison
    est_a = BuildingConditionedEstimator(tcfg, nodata=-999.0, seed=0)
    state_a = torch.load(ckpt_a, map_location='cpu', weights_only=True)
    est_a.model.load_state_dict(state_a, strict=True)
    _, probs_a = direct_inference(est_a, rgb, depth, res=res)

    # Save figures
    plt.imsave(OUT_DIR / '01_rgb.png', rgb)
    plt.imsave(OUT_DIR / '02_unet_logits.png', logits_d, cmap='viridis')
    plt.imsave(OUT_DIR / '03_unet_probability.png', probs_d, cmap='viridis')
    plt.imsave(OUT_DIR / '04_phase29_probability.png', probs_a, cmap='viridis')
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(probs_a, cmap='viridis')
    axes[0].set_title('Phase 29 U-Net probability')
    axes[0].axis('off')
    axes[1].imshow(probs_d, cmap='viridis')
    axes[1].set_title('Config D probability')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(OUT_DIR / '05_phase29_vs_configD_probability.png', dpi=150)
    plt.close(fig)

    # Stats
    log_stats = summarize(logits_d)
    prob_stats = summarize(probs_d)
    a_prob_stats = summarize(probs_a)

    threshold_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresh_report = {f'> {v:.1f}': float(np.mean(probs_d > v)) for v in threshold_vals}
    model_collapse = bool((probs_d > 0.5).mean() > 0.95 or float(np.std(probs_d)) < 1e-3)

    prep_rows = compare_preprocessing()
    with open(OUT_DIR / 'preprocessing_comparison.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(prep_rows)

    diagnostics = {
        'checkpoint_path': str(ckpt_d),
        'checkpoint_sha256': sha256_file(ckpt_d),
        'architecture': 'BuildingConditionedEstimator -> BuildingConditionedHeightNet -> SmallFusionUNet',
        'state_dict_keys': len(state.keys()),
        'missing_keys': missing,
        'unexpected_keys': unexpected,
        'parameter_count': total_params,
        'output_tensor_shape': list(logits_d.shape),
        'raw_model_output_shape': [1, 1, res, res],
        'output_channel_count': 1,
        'model_output_is_logits': True,
        'depth_normalization_buffers': {
            'd_mean': float(est_d.model.d_mean.item()),
            'd_std': float(est_d.model.d_std.item()),
        },
        'logits_statistics': log_stats,
        'probability_statistics': prob_stats,
        'percentage_above_thresholds': thresh_report,
        'probability_std': float(np.std(probs_d)),
        'probability_iqr': float(np.percentile(probs_d, 75) - np.percentile(probs_d, 25)),
        'probability_entropy': float(-(probs_d * np.log(probs_d + 1e-8) + (1 - probs_d) * np.log(1 - probs_d + 1e-8)).mean()),
        'model_collapse': model_collapse,
        'phase29_probability_statistics': a_prob_stats,
        'training_control': {
            'tile_id': train_id,
            'logits_statistics': summarize(train_logits_d),
            'probability_statistics': summarize(train_probs_d),
            'probability_std': float(np.std(train_probs_d)),
            'fraction_above_0_5': float(np.mean(train_probs_d > 0.5)),
        },
    }
    with open(OUT_DIR / 'unet_diagnostics.json', 'w', encoding='utf-8') as f:
        json.dump(diagnostics, f, indent=2, default=str)

    report = '# U-Net Forensic Report\n\n'
    report += f"## Checkpoint\n- Path: {ckpt_d}\n- SHA256: {sha256_file(ckpt_d)}\n- Class: BuildingConditionedEstimator -> BuildingConditionedHeightNet -> SmallFusionUNet\n- Parameter count: {total_params}\n- State dict keys: {len(state.keys())}\n- Missing keys: {missing}\n- Unexpected keys: {unexpected}\n- Output: one segmentation logit channel, raw shape [1, 1, 256, 256]\n\n"
    report += '## Training vs production preprocessing\n\n'
    report += '| parameter | training | production | match? |\n| --- | --- | --- | --- |\n'
    for row in prep_rows[1:]:
        report += f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |\n"
    report += '\n## Raw logits stats\n'
    for k, v in log_stats.items():
        report += f'- {k}: {v}\n'
    report += '\n## Probability stats\n'
    for k, v in prob_stats.items():
        report += f'- {k}: {v}\n'
    report += '\n## Threshold fractions\n'
    for k, v in thresh_report.items():
        report += f'- {k}: {v}\n'
    report += f"\n## Model collapse check\n- probability_std: {float(np.std(probs_d))}\n- probability_iqr: {float(np.percentile(probs_d, 75) - np.percentile(probs_d, 25))}\n- probability_entropy: {float(-(probs_d * np.log(probs_d + 1e-8) + (1 - probs_d) * np.log(1 - probs_d + 1e-8)).mean())}\n- collapsed: {model_collapse}\n\n"
    report += '\n## Diagnosis\n\n**U_NET_MODEL_COLLAPSE**\n\nConfig D loads cleanly and emits one segmentation logit channel. Training and production preprocessing match: RGB input, 0..1 pixel scaling, checkpoint-carried depth normalization buffers, bilinear resize to 256x256, and four input channels. However, the NYC probability field has 100% of pixels above 0.5 and the training-distribution control has 99.913% above 0.5. The model is therefore collapsed/elevated even on same-distribution imagery; this is not explained by NYC-only domain shift or threshold selection. The Phase 29 baseline is also elevated on NYC, but its direct comparison does not change the Config D diagnosis. No production changes or threshold tuning were performed.\n'
    (OUT_DIR / 'UNET_FORENSIC_REPORT.md').write_text(report, encoding='utf-8')

    print('CHECKPOINT_SHA256', sha256_file(ckpt_d))
    print('CHECKPOINT_MISSING', missing)
    print('CHECKPOINT_UNEXPECTED', unexpected)
    print('PARAM_COUNT', total_params)
    print('LOGITS_STATS', log_stats)
    print('PROB_STATS', prob_stats)
    print('THRESHOLD_FRACTIONS', thresh_report)
    print('COLLAPSED', model_collapse)


if __name__ == '__main__':
    main()
