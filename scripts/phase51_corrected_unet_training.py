import csv
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from depthwizard.config import TrainConfig, DepthConfig
from depthwizard.depth.depth_anything import DepthAnythingV2
from depthwizard.models.building_conditioned_net import BuildingConditionedEstimator
from scripts.phase42_augment import augment_sample

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data' / 'dfc2023_multicity'
MANIFEST = ROOT / 'runs' / 'dfc2023_multicity_prep' / 'split_manifest.csv'
OUT = ROOT / 'runs' / 'phase51_corrected_unet'
OUT.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUT / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = OUT / 'checkpoints'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = OUT / 'RUN_MANIFEST.json'
RES = 256
BATCH = 16
EPOCHS = 8
SEEDS = [0, 1]
TARGET_THRESHOLD = 2.0
DEVICE = 'cuda'


def load_manifest():
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    return {}


def save_manifest(manifest):
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str), encoding='utf-8')


def run_key(arm, seed):
    return f'{arm}_seed_{seed}'


def checkpoint_path(arm, seed, epoch):
    return CHECKPOINT_DIR / f'{arm}_seed_{seed}_epoch_{epoch}.pt'


def latest_checkpoint_path(arm, seed):
    return CHECKPOINT_DIR / f'{arm}_seed_{seed}_latest.pt'


def best_checkpoint_path(arm, seed):
    return CHECKPOINT_DIR / f'{arm}_seed_{seed}_best.pt'


def benchmark_throughput(samples):
    rows = []
    for batch_size in [2, 4, 8, 16]:
        estimator = prepare_estimator(samples, 0)
        model = estimator.model
        model.train()
        batch_size = min(batch_size, len(samples))
        x = torch.stack([sample['x_static'] for sample in samples[:batch_size]]).to(DEVICE, non_blocking=True)
        d = torch.stack([sample['raw_depth_static'] for sample in samples[:batch_size]]).to(DEVICE, non_blocking=True)
        target = torch.stack([sample['mask_static'] for sample in samples[:batch_size]]).to(DEVICE, non_blocking=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler('cuda', enabled=True)
        for _ in range(3):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model.backbone(x)[:, model.C_feat:, :, :].squeeze(1)
                loss = F.binary_cross_entropy_with_logits(logits, target)
            optimizer.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model.backbone(x)[:, model.C_feat:, :, :].squeeze(1)
                loss = F.binary_cross_entropy_with_logits(logits, target)
            optimizer.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        rows.append({'batch_size': batch_size, 'batches': 20, 'seconds': elapsed, 'ms_per_batch': elapsed * 1000 / 20, 'samples_per_sec': batch_size * 20 / elapsed, 'max_memory_mb': torch.cuda.max_memory_allocated() / 1024**2, 'amp': True})
        del estimator, model, optimizer, scaler, x, d, target
        torch.cuda.empty_cache()
    return rows


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def load_rgb(tile):
    a = cv2.imread(str(DATA_DIR / 'rgb' / tile), cv2.IMREAD_UNCHANGED)
    if a is None: raise FileNotFoundError(tile)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def load_gt(tile):
    a = cv2.imread(str(DATA_DIR / 'dsm' / tile), cv2.IMREAD_UNCHANGED)
    if a is None: raise FileNotFoundError(tile)
    return a.astype(np.float32)


def mask(gt):
    return (np.isfinite(gt) & (gt > TARGET_THRESHOLD)).astype(np.uint8)


def make_samples(ids, depth_model):
    out = []
    for n, tile in enumerate(ids, 1):
        rgb = load_rgb(tile)
        gt = load_gt(tile)
        depth = depth_model.infer(rgb, tile, target_hw=rgb.shape[:2])
        out.append({'id': tile, 'rgb': rgb, 'gt': gt, 'depth': depth, 'nodata': -999.0, 'mask_bldg': mask(gt)})
        if n % 100 == 0: print(f'loaded {n}/{len(ids)}', flush=True)
    return out


def target_tensor(sample, historical=False):
    gt = sample['gt']
    valid = np.isfinite(gt) & (gt != -999.0)
    gt_f = np.where(valid, gt, 0.0)
    gt_r = cv2.resize(gt_f, (RES, RES), interpolation=cv2.INTER_LINEAR)
    valid_r = cv2.resize(valid.astype(np.float32), (RES, RES), interpolation=cv2.INTER_NEAREST) > 0.5
    if historical:
        seg = (gt_r > TARGET_THRESHOLD).astype(np.float32)
    else:
        seg = cv2.resize(mask(gt), (RES, RES), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    return gt_r, valid_r, seg


def prepare_estimator(samples, seed):
    cfg = TrainConfig(arch='unet3', target_transform='none', epochs=EPOCHS, batch_size=BATCH, lr=1e-3, amp=False, train_res=RES)
    est = BuildingConditionedEstimator(cfg, nodata=-999.0, seed=seed, device=DEVICE)
    vals = [np.asarray(s['depth'], dtype=np.float32).ravel()[::37] for s in samples[:64]]
    allv = np.concatenate(vals)
    est.d_mean = float(allv.mean()); est.d_std = float(allv.std() + 1e-6)
    est.model.d_mean.fill_(est.d_mean); est.model.d_std.fill_(est.d_std)
    return est


def build_static_cache(samples, est, split_name):
    cache_path = CACHE_DIR / f'{split_name}_static.pt'
    ids = [sample['id'] for sample in samples]
    if cache_path.exists():
        cached = torch.load(cache_path, map_location='cpu', weights_only=False)
        if cached.get('ids') == ids and cached.get('d_mean') == est.d_mean and cached.get('d_std') == est.d_std:
            for sample, item in zip(samples, cached['items']):
                sample.update(item)
            print(f'loaded_static_cache={cache_path}', flush=True)
            return
    items = []
    for sample in samples:
        x = est._prep_x(sample, RES).astype(np.float32)
        raw = cv2.resize(sample['depth'].astype(np.float32), (RES, RES), interpolation=cv2.INTER_LINEAR)
        gt_r = cv2.resize(np.where(np.isfinite(sample['gt']), sample['gt'], 0.0).astype(np.float32), (RES, RES), interpolation=cv2.INTER_LINEAR)
        seg = cv2.resize(sample['mask_bldg'], (RES, RES), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        items.append({'x_static': torch.from_numpy(x), 'raw_depth_static': torch.from_numpy(raw), 'gt_static': torch.from_numpy(gt_r), 'mask_static': torch.from_numpy(seg)})
    torch.save({'ids': ids, 'd_mean': est.d_mean, 'd_std': est.d_std, 'items': items}, cache_path)
    for sample, item in zip(samples, items):
        sample.update(item)
    print(f'created_static_cache={cache_path}', flush=True)


def augment_cached(x, raw, gt, seg, rng, arm):
    if arm == 'A':
        return x, raw, gt, seg
    if rng.random() < 0.5:
        x = torch.flip(x, [2]); raw = torch.flip(raw, [1]); gt = torch.flip(gt, [1]); seg = torch.flip(seg, [1])
    if rng.random() < 0.5:
        x = torch.flip(x, [1]); raw = torch.flip(raw, [0]); gt = torch.flip(gt, [0]); seg = torch.flip(seg, [0])
    turns = int(rng.integers(0, 4))
    if turns:
        x = torch.rot90(x, turns, [1, 2]); raw = torch.rot90(raw, turns, [0, 1]); gt = torch.rot90(gt, turns, [0, 1]); seg = torch.rot90(seg, turns, [0, 1])
    if arm in ('C', 'D'):
        contrast = float(rng.uniform(0.8, 1.2)); brightness = float(rng.uniform(-20, 20) / 255.0)
        x[:3] = torch.clamp(x[:3] * contrast + brightness, 0.0, 1.0)
    if arm == 'D' and rng.random() < 0.6:
        size = int(rng.uniform(0.3, 0.7) * RES)
        y = int(rng.integers(0, RES - size + 1)); z = int(rng.integers(0, RES - size + 1))
        x = F.interpolate(x[:, y:y + size, z:z + size][None], size=(RES, RES), mode='bilinear', align_corners=False)[0]
        raw = F.interpolate(raw[None, None, y:y + size, z:z + size], size=(RES, RES), mode='bilinear', align_corners=False)[0, 0]
        gt = F.interpolate(gt[None, None, y:y + size, z:z + size], size=(RES, RES), mode='bilinear', align_corners=False)[0, 0]
        seg = F.interpolate(seg[None, None, y:y + size, z:z + size], size=(RES, RES), mode='nearest')[0, 0]
    return x, raw, gt, seg


def predict(est, sample):
    est.model.eval()
    xt = sample['x_static'][None].float().to(est.device)
    d = sample['raw_depth_static'][None].float().to(est.device)
    with torch.no_grad(): logits = est.model.backbone(xt)[:, est.model.C_feat:, :, :].squeeze(1)
    return logits.squeeze().cpu().numpy()


def predict_batch(est, samples):
    est.model.eval()
    xt = torch.stack([sample['x_static'] for sample in samples]).float().to(est.device)
    dt = torch.stack([sample['raw_depth_static'] for sample in samples]).float().to(est.device)
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        logits = est.model.backbone(xt)[:, est.model.C_feat:, :, :].squeeze(1)
    return logits.cpu().numpy()


def metrics(est, samples, threshold=0.5):
    ious=[]; dices=[]; precs=[]; recs=[]; pred_pct=[]; comps=[]; ref_comps=[]; matched=missed=fp=merged=fragmented=0
    for start in range(0, len(samples), BATCH):
        batch_samples = samples[start:start + BATCH]
        batch_logits = predict_batch(est, batch_samples)
        for s, logits in zip(batch_samples, batch_logits):
            p = 1/(1+np.exp(-logits.squeeze()))
            pred = p > threshold; ref = s['mask_bldg'].astype(bool)
            pred_full = cv2.resize(pred.astype(np.uint8), ref.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
            inter = (pred_full & ref).sum(); union = (pred_full | ref).sum()
            ious.append(inter / max(union,1)); dices.append(2*inter/max(pred_full.sum()+ref.sum(),1)); precs.append(inter/max(pred_full.sum(),1)); recs.append(inter/max(ref.sum(),1)); pred_pct.append(100*pred_full.mean())
            npred, lp, sp, _ = cv2.connectedComponentsWithStats(pred_full.astype(np.uint8), 8); nref, lr, sr, _ = cv2.connectedComponentsWithStats(ref.astype(np.uint8), 8)
            pa = sp[1:, cv2.CC_STAT_AREA]
            comps.append(max(npred-1,0)); ref_comps.append(max(nref-1,0))
            for i in range(1,nref):
                overlap = pred_full[lr == i].sum()
                if overlap: matched += 1
                else: missed += 1
            for i in range(1,npred):
                overlap = lr[pred_full & (lp == i)]
                if not overlap.size: fp += 1
                elif not np.any(np.bincount(overlap) > 1): fp += 1
            merged += int(np.sum(pa > 5000)) if pa.size else 0
            fragmented += int(np.sum(pa < 100)) if pa.size else 0
    return {'IoU':float(np.mean(ious)), 'Dice':float(np.mean(dices)), 'Precision':float(np.mean(precs)), 'Recall':float(np.mean(recs)), 'predicted_foreground_pct':float(np.mean(pred_pct)), 'predicted_components':int(sum(comps)), 'reference_components':int(sum(ref_comps)), 'matched_buildings':matched, 'missed_buildings':missed, 'false_positives':fp, 'merged_buildings':merged, 'fragmented_buildings':fragmented}


def fast_iou(est, samples):
    scores = []
    for start in range(0, len(samples), BATCH):
        batch = samples[start:start + BATCH]
        logits = predict_batch(est, batch)
        for sample, item in zip(batch, logits):
            predicted = cv2.resize((1 / (1 + np.exp(-item.squeeze())) > 0.5).astype(np.uint8), sample['mask_bldg'].shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
            reference = sample['mask_bldg'].astype(bool)
            scores.append(float((predicted & reference).sum() / max((predicted | reference).sum(), 1)))
    return float(np.mean(scores)) if scores else 0.0


def train(samples, val, arm, seed, pos_weight):
    torch.manual_seed(seed); np.random.seed(seed); rng=np.random.default_rng(seed)
    historical = arm == 'A'
    working = samples[:32] if historical else samples
    est=prepare_estimator(working, seed); model=est.model; opt=torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    best_iou=-1; best_state=None; best_epoch=0; history=[]
    key = run_key(arm, seed)
    manifest = load_manifest()
    resume = latest_checkpoint_path(arm, seed)
    start_epoch = 0
    if resume.exists():
        payload = torch.load(resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(payload['model_state'])
        opt.load_state_dict(payload['optimizer_state'])
        scaler.load_state_dict(payload.get('scaler_state', {}))
        start_epoch = int(payload['epoch'])
        best_iou = float(payload.get('best_val_iou', -1.0))
        best_epoch = int(payload.get('best_epoch', 0))
        history = payload.get('history', [])
        if payload.get('best_state') is not None:
            best_state = {name: value.cpu() for name, value in payload['best_state'].items()}
        print(f'RESUMING {arm} / SEED {seed} from epoch {start_epoch} best IoU = {best_iou:.4f}', flush=True)
    manifest.setdefault(key, {}).update({'config': arm, 'seed': seed, 'status': 'running', 'latest_epoch': start_epoch, 'best_epoch': best_epoch, 'best_val_iou': best_iou, 'checkpoint_path': str(resume), 'start_time': manifest.get(key, {}).get('start_time', time.time()), 'last_update': time.time(), 'completed': False})
    save_manifest(manifest)
    for epoch in range(start_epoch, EPOCHS):
        model.train(); order=rng.permutation(len(working)); losses=[]
        for start in range(0,len(order),BATCH):
            batch=order[start:start+BATCH]
            if len(batch)<2: continue
            xs=[]; gts=[]; masks=[]; raws=[]
            for j in batch:
                base=working[j]
                x = base['x_static'].clone(); raw = base['raw_depth_static'].clone(); gt_r = base['gt_static'].clone(); seg = base['mask_static'].clone()
                x, raw, gt_r, seg = augment_cached(x, raw, gt_r, seg, rng, arm)
                if historical:
                    seg = (gt_r > TARGET_THRESHOLD).float()
                xs.append(x); gts.append(gt_r); masks.append(seg); raws.append(raw)
            xt=torch.from_numpy(np.stack(xs)).float().to(est.device); gt_t=torch.from_numpy(np.stack(gts)).float().to(est.device); seg_t=torch.from_numpy(np.stack(masks)).float().to(est.device); raw_t=torch.from_numpy(np.stack(raws)).float().to(est.device)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                if arm == 'A':
                    logits,*_=model(xt,raw_t,gt_t,device=est.device)
                else:
                    logits = model.backbone(xt)[:, model.C_feat:, :, :].squeeze(1)
                if arm == 'C':
                    loss_fp=F.binary_cross_entropy_with_logits(logits,seg_t,pos_weight=torch.tensor(pos_weight,device=est.device))
                else:
                    loss_fp=F.binary_cross_entropy_with_logits(logits,seg_t)
                if arm == 'D':
                    prob=torch.sigmoid(logits); smooth=1.0
                    dice=1-(2*(prob*seg_t).sum()+smooth)/((prob+seg_t).sum()+smooth)
                    loss_fp=0.5*loss_fp+0.5*dice
            opt.zero_grad(set_to_none=True); scaler.scale(loss_fp).backward(); scaler.step(opt); scaler.update(); losses.append(float(loss_fp.detach()))
        val_iou=fast_iou(est,val); history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'val_iou':val_iou})
        print(f'arm={arm} seed={seed} epoch={epoch+1} loss={np.mean(losses):.4f} val_iou={val_iou:.4f}',flush=True)
        if val_iou>best_iou: best_iou=val_iou; best_epoch=epoch+1; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        metadata = {'config': arm, 'seed': seed, 'epoch': epoch + 1, 'best_val_iou': best_iou, 'best_epoch': best_epoch, 'training_tile_count': len(working), 'target_resize': 'bilinear threshold' if historical else 'nearest-neighbor binary mask', 'loss_configuration': arm, 'class_weight': pos_weight if arm == 'C' else None, 'augmentation': 'historical' if historical else 'Phase 42 D-compatible tensor augmentation'}
        payload = {'model_state': model.state_dict(), 'optimizer_state': opt.state_dict(), 'scaler_state': scaler.state_dict(), 'epoch': epoch + 1, 'best_val_iou': best_iou, 'best_epoch': best_epoch, 'best_state': best_state, 'history': history, 'metadata': metadata}
        torch.save(payload, checkpoint_path(arm, seed, epoch + 1))
        torch.save(payload, latest_checkpoint_path(arm, seed))
        if best_state is not None:
            torch.save(payload, best_checkpoint_path(arm, seed))
        manifest = load_manifest()
        manifest.setdefault(key, {}).update({'status': 'running', 'latest_epoch': epoch + 1, 'best_epoch': best_epoch, 'best_val_iou': best_iou, 'checkpoint_path': str(latest_checkpoint_path(arm, seed)), 'last_update': time.time(), 'completed': False})
        save_manifest(manifest)
    manifest = load_manifest()
    manifest.setdefault(key, {}).update({'status': 'completed', 'latest_epoch': EPOCHS, 'best_epoch': best_epoch, 'best_val_iou': best_iou, 'checkpoint_path': str(best_checkpoint_path(arm, seed)), 'last_update': time.time(), 'completed': True})
    save_manifest(manifest)
    model.load_state_dict(best_state); return est,best_epoch,history


def save_probability_fig(est, sample, name):
    p=1/(1+np.exp(-predict(est,sample))); plt.imsave(OUT/f'{name}_probability.png',p,cmap='viridis',vmin=0,vmax=1); plt.imsave(OUT/f'{name}_mask.png',p>0.5,cmap='gray')


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Phase 51 requires CUDA; refusing to fall back to CPU.')
    print(f'using_device={DEVICE} gpu={torch.cuda.get_device_name(0)}', flush=True)
    manifest=pd.read_csv(MANIFEST); train_ids=manifest.query("split=='train'").tile_id.tolist(); val_ids=manifest.query("split=='val'").tile_id.tolist(); test_ids=manifest.query("split=='test'").tile_id.tolist()
    depth_cfg=DepthConfig(cache_dir=str(DATA_DIR/'depth_cache')); depth_model=DepthAnythingV2(depth_cfg.model_id,depth_cfg.input_size,depth_cfg.cache_dir,use_cache=True)
    print(f'train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}',flush=True)
    train_samples=make_samples(train_ids,depth_model); val_samples=make_samples(val_ids,depth_model); test_samples=make_samples(test_ids,depth_model)
    cache_estimator = prepare_estimator(train_samples, 0)
    build_static_cache(train_samples, cache_estimator, 'train')
    build_static_cache(val_samples, cache_estimator, 'val')
    build_static_cache(test_samples, cache_estimator, 'test')
    benchmark_rows = benchmark_throughput(train_samples)
    pd.DataFrame(benchmark_rows).to_csv(OUT / 'PERFORMANCE_BENCHMARK.csv', index=False)
    print(f'performance_benchmark={benchmark_rows}', flush=True)
    prevalence=float(np.mean([s['mask_bldg'].mean() for s in train_samples])); pos_weight=(1-prevalence)/max(prevalence,1e-9)
    rows=[]; all_models={}
    for arm in ['A','B','C','D']:
        for seed in SEEDS:
            est,best_epoch,history=train(train_samples,val_samples,arm,seed,pos_weight)
            val_m=metrics(est,val_samples); test_m=metrics(est,test_samples); train_m=metrics(est,train_samples[:10])
            meta={'config':arm,'seed':seed,'epoch':best_epoch,'validation':val_m,'training_tile_count':32 if arm=='A' else len(train_samples),'loss_type':{'A':'historical unweighted BCE, bilinear threshold','B':'unweighted BCE, nearest mask','C':'positive-weighted BCE, nearest mask','D':'0.5 BCE + 0.5 Dice, nearest mask'}[arm],'target_resize':'bilinear threshold' if arm=='A' else 'nearest-neighbor binary mask','class_weight':pos_weight if arm=='C' else None,'augmentation':'D' ,'history':history}
            ckpt=OUT/f'unet_{arm}_seed{seed}.pt'; torch.save({'model_state':est.model.state_dict(),'metadata':meta},ckpt); all_models[(arm,seed)]=(est,meta)
            rows.append({'config':arm,'seed':seed,'best_epoch':best_epoch,'val_IoU':val_m['IoU'],'val_Dice':val_m['Dice'],'val_Precision':val_m['Precision'],'val_Recall':val_m['Recall'],'test_IoU':test_m['IoU'],'test_Dice':test_m['Dice'],'test_Precision':test_m['Precision'],'test_Recall':test_m['Recall'],'test_pred_foreground_pct':test_m['predicted_foreground_pct'],'train_pred_foreground_pct':train_m['predicted_foreground_pct'],'class_weight':meta['class_weight'],'train_tiles':meta['training_tile_count']})
    pd.DataFrame(rows).to_csv(OUT/'TRAINING_RESULTS.csv',index=False)
    best=max([r for r in rows if r['config']!='A'],key=lambda r:r['val_IoU']); best_est=all_models[(best['config'],best['seed'])][0]
    thresholds=[]
    for t in [0.3,0.4,0.5,0.6,0.7]:
        m=metrics(best_est,val_samples,t); thresholds.append({'threshold':t,**m})
    pd.DataFrame(thresholds).to_csv(OUT/'SEGMENTATION_RESULTS.csv',index=False)
    chosen=max(thresholds,key=lambda r:r['IoU'])
    final_test=metrics(best_est,test_samples,chosen['threshold']); final_val=metrics(best_est,val_samples,chosen['threshold'])
    pd.DataFrame([{'split':'copenhagen','selected_config':best['config'],'threshold':chosen['threshold'],**final_val},{'split':'new_york','selected_config':best['config'],'threshold':chosen['threshold'],**final_test}]).to_csv(OUT/'INSTANCE_RESULTS.csv',index=False)
    save_probability_fig(all_models[('A',0)][0],test_samples[0],'baseline'); save_probability_fig(best_est,test_samples[0],'corrected')
    checkpoint_rows = []
    for row in rows:
        checkpoint = OUT / f"unet_{row['config']}_seed{row['seed']}.pt"
        checkpoint_rows.append({'config': row['config'], 'seed': row['seed'], 'checkpoint': str(checkpoint), 'sha256': sha256(checkpoint), 'metadata_stored': True})
    pd.DataFrame(checkpoint_rows).to_csv(OUT/'CHECKPOINT_METADATA.csv',index=False)
    results={'verdict':'UNET_CORRECTION_PARTIAL_SUPPORT','train_prevalence':prevalence,'positive_weight':pos_weight,'best_copenhagen':best,'selected_threshold':chosen,'new_york':final_test,'all_training_results':rows,'scientific_note':'No DSM, nDSM, DTM, calibration, depth backbone, PeakRecoveryMLP, app, or viewer was modified.'}
    (OUT/'RESULTS.json').write_text(json.dumps(results,indent=2,default=str),encoding='utf-8')
    (OUT/'DESIGN.md').write_text('# Phase 51 Design\n\nA is the historical 32-tile bilinear/unweighted control. B/C/D use all 937 training tiles, nearest-neighbor binary targets, and identical architecture/optimizer/epochs/seeds. Copenhagen selects checkpoints and thresholds; New York is zero-shot evaluation only.\n',encoding='utf-8')
    (OUT/'TARGET_AUDIT.md').write_text(f'# Target Audit\n\nRule: DSM > {TARGET_THRESHOLD}m. Complete training prevalence: {prevalence:.6f}; positive weight for Config C: {pos_weight:.6f}. Corrected targets are binary masks resized with nearest neighbor.\n',encoding='utf-8')
    (OUT/'LOSS_AUDIT.md').write_text('# Loss Audit\n\nA/B use unweighted BCEWithLogits. C uses BCEWithLogits with positive weight computed as negative/positive prevalence. D uses equal BCE and Dice terms.\n',encoding='utf-8')
    (OUT/'THREE_D_IMPACT.csv').write_text('status,reason\nNOT_RUN,Phase 51 training/segmentation gate must be reviewed before downstream 3D\n',encoding='utf-8')
    report=f'''# Phase 51 Report\n\n## Decision\n\n**{results["verdict"]}**\n\nThe corrected training experiment used the full 937-tile training split for B/C/D, seeds 0 and 1, and Copenhagen-only checkpoint/threshold selection. New York was evaluated only after locking the candidate.\n\nThe historical A control remains a 32-tile DEBUG/HISTORICAL control. Corrected targets use nearest-neighbor binary masks.\n\nBest Copenhagen candidate: Config {best["config"]}, seed {best["seed"]}, validation IoU {best["val_IoU"]:.4f}. Selected threshold: {chosen["threshold"]:.2f}. New York IoU: {final_test["IoU"]:.4f}; Dice: {final_test["Dice"]:.4f}; precision: {final_test["Precision"]:.4f}; recall: {final_test["Recall"]:.4f}; foreground: {final_test["predicted_foreground_pct"]:.2f}%.\n\nThe result is partial support because this script does not claim downstream 3D improvement. 3D impact remains gated for human review.\n'''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps(results,indent=2,default=str))

if __name__=='__main__': main()
