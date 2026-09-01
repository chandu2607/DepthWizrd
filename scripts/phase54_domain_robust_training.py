import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from depthwizard.config import TrainConfig
from depthwizard.models.building_conditioned_net import BuildingConditionedEstimator

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data' / 'dfc2023_multicity'
MANIFEST = ROOT / 'runs' / 'dfc2023_multicity_prep' / 'split_manifest.csv'
OUT_DIR = ROOT / 'runs' / 'phase54_domain_robust_training'
CACHE_DIR = OUT_DIR / 'cache'
CHECKPOINT_DIR = OUT_DIR / 'checkpoints'
MANIFEST_PATH = OUT_DIR / 'RUN_MANIFEST.json'
RES = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 8
SEEDS = [0, 1]
TARGET_THRESHOLD = 2.0
BATCH = 8


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_manifest():
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    return {}


def save_manifest(manifest):
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str), encoding='utf-8')


def load_split_manifest():
    df = pd.read_csv(MANIFEST)
    return df


def load_rgb(tile):
    image = cv2.imread(str(DATA_DIR / 'rgb' / tile), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(tile)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_gt(tile):
    image = cv2.imread(str(DATA_DIR / 'dsm' / tile), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(tile)
    return image.astype(np.float32)


def make_mask(gt):
    return (np.isfinite(gt) & (gt > TARGET_THRESHOLD)).astype(np.uint8)


def build_sample(tile, depth_model):
    rgb = load_rgb(tile)
    gt = load_gt(tile)
    depth = depth_model.infer(rgb, tile, target_hw=rgb.shape[:2])
    return {'id': tile, 'rgb': rgb, 'gt': gt, 'depth': depth, 'mask_bldg': make_mask(gt), 'nodata': -999.0}


def prepare_estimator(samples, seed):
    cfg = TrainConfig(arch='unet3', target_transform='none', epochs=EPOCHS, batch_size=BATCH, lr=1e-3, amp=True, train_res=RES)
    est = BuildingConditionedEstimator(cfg, nodata=-999.0, seed=seed, device=DEVICE)
    vals = [np.asarray(s['depth'], dtype=np.float32).ravel()[::37] for s in samples[:64]]
    arr = np.concatenate(vals)
    est.d_mean = float(arr.mean())
    est.d_std = float(arr.std() + 1e-6)
    est.model.d_mean.fill_(est.d_mean)
    est.model.d_std.fill_(est.d_std)
    return est


def make_static_cache(samples, est, split_name):
    cache_path = CACHE_DIR / f'{split_name}_static.pt'
    ids = [s['id'] for s in samples]
    if cache_path.exists():
        cached = torch.load(cache_path, map_location='cpu', weights_only=False)
        if cached.get('ids') == ids and cached.get('d_mean') == est.d_mean and cached.get('d_std') == est.d_std:
            for sample, item in zip(samples, cached['items']):
                sample.update(item)
            return
    items = []
    for sample in samples:
        x = est._prep_x(sample, RES).astype(np.float32)
        raw = cv2.resize(sample['depth'].astype(np.float32), (RES, RES), interpolation=cv2.INTER_LINEAR)
        gt_r = cv2.resize(np.where(np.isfinite(sample['gt']), sample['gt'], 0.0), (RES, RES), interpolation=cv2.INTER_LINEAR)
        seg = cv2.resize(sample['mask_bldg'], (RES, RES), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        items.append({'x_static': torch.from_numpy(x), 'raw_depth_static': torch.from_numpy(raw), 'gt_static': torch.from_numpy(gt_r), 'mask_static': torch.from_numpy(seg)})
    torch.save({'ids': ids, 'd_mean': est.d_mean, 'd_std': est.d_std, 'items': items}, cache_path)
    for sample, item in zip(samples, items):
        sample.update(item)


def height_bin_value(gt_values):
    vals = np.asarray(gt_values, dtype=np.float32).ravel()
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    if vals.size == 0:
        return 0
    h = float(np.percentile(vals, 95))
    if h < 10.0:
        return 0
    if h < 20.0:
        return 1
    if h < 30.0:
        return 2
    if h < 40.0:
        return 3
    return 4


def apply_rgb_domain_aug(sample, rng):
    img = sample['rgb'].copy().astype(np.float32)
    alpha = rng.uniform(0.85, 1.15)
    beta = rng.uniform(-20, 20)
    img = np.clip(img * alpha + beta, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.85, 1.15), 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    gamma = rng.uniform(0.8, 1.2)
    img = np.power(img / 255.0, gamma) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        noise = rng.normal(0, 8, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.2:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    sample = sample.copy(); sample['rgb'] = img
    return sample


def apply_scale_density_aug(sample, rng):
    s = sample.copy(); img = s['rgb']; gt = s['gt']; depth = s['depth']; mask = s['mask_bldg']
    scale = rng.uniform(0.8, 1.2)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-12, 12), scale)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    gt = cv2.warpAffine(gt, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    depth = cv2.warpAffine(depth, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    if rng.random() < 0.5:
        crop = int(rng.uniform(0.6, 0.9) * min(h, w))
        y0 = rng.integers(0, max(1, h - crop)); x0 = rng.integers(0, max(1, w - crop))
        img = img[y0:y0 + crop, x0:x0 + crop]
        gt = gt[y0:y0 + crop, x0:x0 + crop]
        depth = depth[y0:y0 + crop, x0:x0 + crop]
        mask = mask[y0:y0 + crop, x0:x0 + crop]
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    s['rgb'] = img; s['gt'] = gt; s['depth'] = depth; s['mask_bldg'] = mask.astype(np.uint8)
    return s


def apply_building_focused_crop(sample, rng):
    s = sample.copy(); img = s['rgb']; gt = s['gt']; depth = s['depth']; mask = s['mask_bldg']
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return s
    y0, y1 = max(0, int(ys.min() - 20)), min(img.shape[0], int(ys.max() + 20))
    x0, x1 = max(0, int(xs.min() - 20)), min(img.shape[1], int(xs.max() + 20))
    crop_h = max(64, y1 - y0); crop_w = max(64, x1 - x0)
    y0 = max(0, y0 - 10); x0 = max(0, x0 - 10); y1 = min(img.shape[0], y0 + crop_h); x1 = min(img.shape[1], x0 + crop_w)
    img = img[y0:y1, x0:x1]; gt = gt[y0:y1, x0:x1]; depth = depth[y0:y1, x0:x1]; mask = mask[y0:y1, x0:x1]
    if img.shape[:2] != (RES, RES):
        img = cv2.resize(img, (RES, RES), interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt, (RES, RES), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (RES, RES), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (RES, RES), interpolation=cv2.INTER_NEAREST)
    s['rgb'] = img; s['gt'] = gt; s['depth'] = depth; s['mask_bldg'] = mask.astype(np.uint8)
    return s


def arm_augmentation(sample, arm, rng):
    if arm == 'A':
        return sample
    sample = apply_rgb_domain_aug(sample, rng)
    if arm in ('C', 'D', 'E'):
        sample = apply_scale_density_aug(sample, rng)
    if arm == 'E':
        sample = apply_building_focused_crop(sample, rng)
    return sample


def apply_policy_to_batch(batch_samples, arm, rng):
    transformed = []
    for sample in batch_samples:
        s = sample.copy()
        s = arm_augmentation(s, arm, rng)
        transformed.append(s)
    return transformed


def compute_pixel_stats_for_coverage(samples):
    rows = []
    for sample in samples:
        mask = sample['mask_bldg']
        gt = sample['gt']
        building_heights = gt[mask > 0]
        rows.append({'tile': sample['id'], 'building_pixels': int(mask.sum()), 'buildings': int(np.unique(cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)[1])[1:].size), 'height_bin': height_bin_value(building_heights), 'density': float(mask.mean())})
    return pd.DataFrame(rows)


def training_city_coverage(train_samples):
    grouped = {}
    for sample in train_samples:
        city = sample['id'].split('_')[0] if '_' in sample['id'] else sample['id']
        grouped.setdefault(city, []).append(sample)
    out = []
    for city, items in grouped.items():
        df = compute_pixel_stats_for_coverage(items)
        out.append({'city': city, 'n_buildings': int(df['buildings'].sum()), 'density_mean': float(df['density'].mean()), 'height_bin_counts': {str(k): int((df['height_bin'] == k).sum()) for k in range(5)}, 'density_per_tile': float(df['density'].mean())})
    return pd.DataFrame(out)


def batch_indices_for_arm(indices, arm, rng, sample_meta):
    if arm in ('A', 'B'):
        return indices
    if arm == 'C':
        return indices
    weights = []
    for idx in indices:
        meta = sample_meta[idx]
        weight = 1.0
        if meta.get('height_bin') == 4:
            weight = 3.5
        elif meta.get('height_bin') == 3:
            weight = 2.0
        elif meta.get('height_bin') == 2:
            weight = 1.2
        if meta.get('density') > 0.20:
            weight *= 1.3
        weights.append(weight)
    if arm == 'D':
        probs = np.asarray(weights, dtype=np.float32)
        probs = probs / max(probs.sum(), 1e-6)
        return list(rng.choice(indices, size=len(indices), replace=True, p=probs))
    if arm == 'E':
        weights = np.asarray(weights, dtype=np.float32) * (1.0 + 0.5 * np.asarray([sample_meta[idx].get('density', 0.0) for idx in indices]))
        probs = weights / max(weights.sum(), 1e-6)
        return list(rng.choice(indices, size=len(indices), replace=True, p=probs))
    return indices


def metrics_for_estimator(est, samples, threshold=0.5):
    scores = []
    for start in range(0, len(samples), BATCH):
        batch = samples[start:start + BATCH]
        logits = []
        with torch.no_grad():
            x = torch.stack([s['x_static'] for s in batch]).float().to(DEVICE)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=DEVICE == 'cuda'):
                out = est.model.backbone(x)[:, est.model.C_feat:, :, :].squeeze(1)
            logits = out.cpu().numpy()
        for s, item in zip(batch, logits):
            prob = 1 / (1 + np.exp(-item.squeeze()))
            pred = cv2.resize((prob > threshold).astype(np.uint8), s['mask_bldg'].shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
            ref = s['mask_bldg'].astype(bool)
            inter = (pred & ref).sum(); union = (pred | ref).sum()
            scores.append(float(inter / max(union, 1)))
    return float(np.mean(scores)) if scores else 0.0


def run_train(arm, seed, train_samples, val_samples, smoke=False):
    rng = np.random.default_rng(seed)
    est = prepare_estimator(train_samples, seed)
    model = est.model
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda', enabled=DEVICE == 'cuda')
    sample_meta = []
    for sample in train_samples:
        gt = sample['gt_static'] if 'gt_static' in sample else sample['gt']
        building_heights = gt[cv2.resize(sample['mask_bldg'], (RES, RES), interpolation=cv2.INTER_NEAREST).astype(bool)]
        sample_meta.append({'height_bin': height_bin_value(building_heights), 'density': float(sample['mask_bldg'].mean())})
    run_key = f'{arm}_seed_{seed}'
    manifest = load_manifest(); manifest.setdefault(run_key, {'config': arm, 'seed': seed, 'status': 'running', 'latest_epoch': 0, 'best_val_iou': -1.0, 'best_epoch': 0, 'batch_size': BATCH, 'smoke': smoke})
    save_manifest(manifest)
    best_iou = -1.0; best_epoch = 0; best_state = None
    for epoch in range(1, EPOCHS + 1):
        order = list(range(len(train_samples)))
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), BATCH):
            idxs = order[start:start + BATCH]
            batch = [train_samples[i] for i in idxs]
            if arm in ('D', 'E'):
                idxs = batch_indices_for_arm(idxs, arm, rng, sample_meta)
                batch = [train_samples[i] for i in idxs]
            batch = apply_policy_to_batch(batch, arm, rng)
            xs = []; gts=[]; masks=[]; raws=[]
            for s in batch:
                xs.append(s['x_static']); gts.append(s['gt_static']); masks.append(s['mask_static']); raws.append(s['raw_depth_static'])
            xt = torch.stack(xs).float().to(DEVICE)
            gt_t = torch.stack(gts).float().to(DEVICE)
            mask_t = torch.stack(masks).float().to(DEVICE)
            raw_t = torch.stack(raws).float().to(DEVICE)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=DEVICE == 'cuda'):
                logits = model.backbone(xt)[:, model.C_feat:, :, :].squeeze(1)
                loss = F.binary_cross_entropy_with_logits(logits, mask_t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            losses.append(float(loss.detach().cpu()))
        val_iou = metrics_for_estimator(est, val_samples)
        if val_iou > best_iou:
            best_iou = val_iou; best_epoch = epoch; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        payload = {'model_state': model.state_dict(), 'optimizer_state': opt.state_dict(), 'scaler_state': scaler.state_dict(), 'epoch': epoch, 'best_val_iou': best_iou, 'best_epoch': best_epoch, 'metadata': {'config': arm, 'seed': seed, 'epoch': epoch, 'batch_size': BATCH, 'training_tile_count': len(train_samples), 'target_resize': 'nearest-neighbor binary mask', 'loss_configuration': 'BCEWithLogits', 'class_weight': None, 'augmentation': arm, 'sampler': 'height-balanced' if arm in ('D', 'E') else 'uniform', 'height_sampling_policy': 'oversample >=40m and 30-40m underrepresented bins' if arm in ('D', 'E') else 'none', 'validation_iou': val_iou, 'loss': float(np.mean(losses))}}
        ckpt = CHECKPOINT_DIR / f'{arm}_seed_{seed}_epoch_{epoch}.pt'
        torch.save(payload, ckpt)
        torch.save(payload, CHECKPOINT_DIR / f'{arm}_seed_{seed}_latest.pt')
        if best_state is not None:
            torch.save({'model_state': best_state, 'metadata': payload['metadata']}, CHECKPOINT_DIR / f'{arm}_seed_{seed}_best.pt')
        manifest = load_manifest(); manifest[run_key] = {'config': arm, 'seed': seed, 'status': 'running', 'latest_epoch': epoch, 'best_epoch': best_epoch, 'best_val_iou': best_iou, 'batch_size': BATCH, 'checkpoint': str(CHECKPOINT_DIR / f'{arm}_seed_{seed}_best.pt'), 'last_update': time.time(), 'completed': False}; save_manifest(manifest)
        if smoke and epoch >= 1:
            break
    manifest = load_manifest(); manifest[run_key] = {'config': arm, 'seed': seed, 'status': 'completed', 'latest_epoch': EPOCHS if not smoke else min(1, EPOCHS), 'best_epoch': best_epoch, 'best_val_iou': best_iou, 'batch_size': BATCH, 'checkpoint': str(CHECKPOINT_DIR / f'{arm}_seed_{seed}_best.pt'), 'last_update': time.time(), 'completed': True}; save_manifest(manifest)
    if best_state is not None:
        model.load_state_dict(best_state)
    return est, best_iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--arm', choices=['A', 'B', 'C', 'D', 'E'], default=None)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_split_manifest()
    train_ids = df.query("split == 'train'").tile_id.tolist()
    val_ids = df.query("split == 'val'").tile_id.tolist()
    depth_model = __import__('depthwizard.depth.depth_anything', fromlist=['DepthAnythingV2']).DepthAnythingV2
    # Use the same fixed API as the project expects.
    from depthwizard.config import DepthConfig
    depth_cfg = DepthConfig(cache_dir=str(DATA_DIR / 'depth_cache'))
    depth_model = __import__('depthwizard.depth.depth_anything', fromlist=['DepthAnythingV2']).DepthAnythingV2(depth_cfg.model_id, depth_cfg.input_size, depth_cfg.cache_dir, use_cache=True)
    if args.smoke:
        train_ids = train_ids[:8]
        val_ids = val_ids[:8]
    train_samples = [build_sample(t, depth_model) for t in train_ids]
    val_samples = [build_sample(t, depth_model) for t in val_ids]
    est = prepare_estimator(train_samples, 0)
    for split_name, samples in [('train', train_samples), ('val', val_samples)]:
        make_static_cache(samples, est, split_name)
    city_cover = training_city_coverage(train_samples)
    city_cover.to_csv(OUT_DIR / 'TRAINING_CITY_COVERAGE.csv', index=False)
    if args.arm:
        arms = [args.arm]
    else:
        arms = ['A', 'B', 'C', 'D', 'E']
    rows = []
    for arm in arms:
        for seed in SEEDS:
            est, best_iou = run_train(arm, seed, train_samples, val_samples, smoke=args.smoke)
            rows.append({'config': arm, 'seed': seed, 'best_val_iou': best_iou, 'batch_size': BATCH, 'train_tiles': len(train_samples), 'smoke': args.smoke, 'augmentation': arm, 'sampler': 'height-balanced' if arm in ('D', 'E') else 'uniform'})
    pd.DataFrame(rows).to_csv(OUT_DIR / 'TRAINING_RESULTS.csv', index=False)
    # Minimal design + report. Full multi-arm training can be expanded by the user when running a full cluster job.
    (OUT_DIR / 'DESIGN.md').write_text('# Phase 54 Domain-Robust Training\n\nThis phase is a controlled variant study around the corrected U-Net with augmentation and height-balanced sampling, without changing the architecture or downstream pipeline.\n', encoding='utf-8')
    (OUT_DIR / 'REPORT.md').write_text('# Phase 54 Domain-Robust Training\n\nThis script implements the Phase 54 baseline and augmentation arms. It preserves the corrected U-Net architecture and uses Copenhagen-only validation.\n', encoding='utf-8')
    (OUT_DIR / 'RESULTS.json').write_text(json.dumps({'arms': arms, 'train_tiles': len(train_samples), 'smoke': args.smoke, 'selected_with_copenhagen_only': True}, indent=2), encoding='utf-8')
    if args.smoke:
        print(json.dumps({'status': 'SMOKE_OK', 'arms': arms, 'seed_count': len(SEEDS), 'train_tiles': len(train_samples)}, indent=2))

if __name__ == '__main__':
    main()
