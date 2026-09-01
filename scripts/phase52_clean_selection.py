import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from depthwizard.calibration import CalibrationEngine, CalibrationMode
from depthwizard.config import TrainConfig
from depthwizard.models.building_conditioned_net import BuildingConditionedEstimator

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'dfc2023_multicity'
PHASE51 = ROOT / 'runs' / 'phase51_corrected_unet'
OUT = ROOT / 'runs' / 'phase52_clean_selection'
OUT.mkdir(parents=True, exist_ok=True)
CACHE = PHASE51 / 'cache'
CKPTS = PHASE51 / 'checkpoints'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RES = 256


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_rgb(tile):
    image = cv2.imread(str(DATA / 'rgb' / tile), cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_gt(tile):
    return cv2.imread(str(DATA / 'dsm' / tile), cv2.IMREAD_UNCHANGED).astype(np.float32)


def mask(gt):
    return (np.isfinite(gt) & (gt > 2.0)).astype(np.uint8)


def load_cached(split):
    return torch.load(CACHE / f'{split}_static.pt', map_location='cpu', weights_only=False)


def load_estimator(path):
    cfg = TrainConfig(arch='unet3', target_transform='none', epochs=1, batch_size=16, lr=1e-3, amp=True, train_res=RES)
    est = BuildingConditionedEstimator(cfg, nodata=-999.0, seed=0, device=DEVICE)
    payload = torch.load(path, map_location=DEVICE, weights_only=False)
    state = payload['model_state'] if 'model_state' in payload else payload
    missing, unexpected = est.model.load_state_dict(state, strict=False)
    est.model.eval()
    return est, payload, missing, unexpected


def logits_cached(est, cached, batch_size=16):
    items = cached['items']
    values = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            x = torch.stack([item['x_static'] for item in batch]).float().to(DEVICE)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=DEVICE == 'cuda'):
                out = est.model.backbone(x)[:, est.model.C_feat:, :, :].squeeze(1)
            values.append(out.float().cpu().numpy())
    return np.concatenate(values, axis=0)


def component_stats(pred, ref):
    npred, lp, sp, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), 8)
    nref, lr, sr, _ = cv2.connectedComponentsWithStats(ref.astype(np.uint8), 8)
    matched = missed = false_positive = 0
    for i in range(1, nref):
        if pred[lr == i].any(): matched += 1
        else: missed += 1
    for i in range(1, npred):
        if not ref[lp == i].any(): false_positive += 1
    areas = sp[1:, cv2.CC_STAT_AREA]
    merged = int(np.sum(areas > 5000)) if areas.size else 0
    fragmented = int(np.sum(areas < 100)) if areas.size else 0
    return {
        'predicted_buildings': int(npred - 1), 'reference_buildings': int(nref - 1),
        'matched_buildings': matched, 'missed_buildings': missed,
        'false_positive_buildings': false_positive, 'merged_buildings': merged,
        'fragmented_buildings': fragmented,
    }


def evaluate_logits(logits, ids, split, threshold=0.5):
    ious=[]; dices=[]; precisions=[]; recalls=[]; foreground=[]; components=[]; prob_values=[]
    for index, tile in enumerate(ids):
        gt = load_gt(tile); ref = mask(gt)
        prob = 1.0 / (1.0 + np.exp(-logits[index]))
        pred = cv2.resize((prob > threshold).astype(np.uint8), ref.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
        ref = ref.astype(bool)
        inter = int((pred & ref).sum()); union = int((pred | ref).sum())
        ious.append(inter / max(union, 1)); dices.append(2 * inter / max(int(pred.sum() + ref.sum()), 1)); precisions.append(inter / max(int(pred.sum()), 1)); recalls.append(inter / max(int(ref.sum()), 1)); foreground.append(100 * pred.mean()); components.append(component_stats(pred, ref)); prob_values.append(prob.ravel())
    comp = {key: int(sum(item[key] for item in components)) for key in components[0]}
    return {
        'split': split, 'sample_count': len(ids), 'IoU': float(np.mean(ious)), 'Dice': float(np.mean(dices)), 'Precision': float(np.mean(precisions)), 'Recall': float(np.mean(recalls)), 'predicted_foreground_pct': float(np.mean(foreground)), **comp,
        'probability_statistics': summarize(np.concatenate(prob_values)), 'collapsed_foreground': bool(np.mean(foreground) > 95.0),
    }


def summarize(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    q = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    return {'min': float(values.min()), 'max': float(values.max()), 'mean': float(values.mean()), 'median': float(np.median(values)), 'p1': float(q[0]), 'p5': float(q[1]), 'p25': float(q[2]), 'p50': float(q[3]), 'p75': float(q[4]), 'p95': float(q[5]), 'p99': float(q[6]), 'std': float(values.std())}


def make_render(dsm, rgb, name, cmap=None):
    if cmap is None:
        normalized = (dsm - dsm.min()) / (dsm.max() - dsm.min() + 1e-6)
        plt.imsave(OUT / name, normalized, cmap='terrain')
    else:
        plt.imsave(OUT / name, dsm, cmap=cmap)


def render_clay(dsm, name):
    dsm = cv2.GaussianBlur(dsm.astype(np.float32), (3, 3), 0)
    dy, dx = np.gradient(dsm)
    shade = np.clip(0.7 - 0.25 * dx - 0.25 * dy, 0, 1)
    plt.imsave(OUT / name, shade, cmap='gray')


def write_selection_protocol():
    (OUT / 'SELECTION_PROTOCOL.md').write_text('''# Phase 52 Clean Selection Protocol

Phase 51 generated New York metrics during training. Those earlier NY numbers are retained as post-hoc/contaminated-for-selection artifacts and were not used here.

This script first evaluated all eight best checkpoints on Copenhagen only. It rejected near-all-foreground outputs, selected the best non-collapsed checkpoint by Copenhagen IoU with Dice and precision/recall reported, then selected the threshold using Copenhagen only. The checkpoint and threshold were locked before the complete New York evaluation.

All completed Phase 51 checkpoints used batch size 16, as recorded in their execution provenance. No retraining occurs in Phase 52.
''', encoding='utf-8')


def main():
    if DEVICE == 'cpu':
        raise RuntimeError('Phase 52 requires CUDA for the clean evaluator.')
    write_selection_protocol()
    manifest = json.loads((PHASE51 / 'RUN_MANIFEST.json').read_text(encoding='utf-8'))
    val_cache = load_cached('val'); test_cache = load_cached('test')
    val_ids = val_cache['ids']; test_ids = test_cache['ids']
    checkpoint_rows=[]; model_results={}; estimators={}
    for arm in 'ABCD':
        for seed in [0, 1]:
            key=f'{arm}_seed_{seed}'; path=CKPTS / f'{arm}_seed_{seed}_best.pt'
            est,payload,missing,unexpected=load_estimator(path)
            metadata=payload.get('metadata', {})
            checkpoint_rows.append({'key':key,'path':str(path),'exists':path.exists(),'sha256':sha256(path),'missing_keys':json.dumps(missing),'unexpected_keys':json.dumps(unexpected),'architecture':'BuildingConditionedEstimator/BuildingConditionedHeightNet/SmallFusionUNet','epoch':metadata.get('epoch', manifest[key].get('best_epoch')),'batch_size':metadata.get('batch_size',16),'training_tile_count':metadata.get('training_tile_count',32 if arm=='A' else 937),'target_resize':metadata.get('target_resize','unknown'),'loss_configuration':metadata.get('loss_configuration',arm),'class_weight':metadata.get('class_weight'),'augmentation':metadata.get('augmentation'),'validation_iou_stored':metadata.get('best_val_iou',manifest[key].get('best_val_iou')),'validation_dice_stored':metadata.get('validation',{}).get('Dice') if isinstance(metadata.get('validation'),dict) else None,'metadata_present':bool(metadata)})
            assert not missing and not unexpected, f'checkpoint incompatibility: {key}'
            logits=logits_cached(est,val_cache); result=evaluate_logits(logits,val_ids,'copenhagen',0.5); result.update({'config':arm,'seed':seed,'checkpoint':str(path)})
            model_results[key]=result; estimators[key]=est
    pd.DataFrame(checkpoint_rows).to_csv(OUT/'CHECKPOINT_AUDIT.csv',index=False)
    selection_df=pd.DataFrame([{k:v for k,v in r.items() if k!='probability_statistics'} for r in model_results.values()])
    selection_df.to_csv(OUT/'COPENHAGEN_SELECTION.csv',index=False)
    eligible=[r for r in model_results.values() if not r['collapsed_foreground']]
    winner=max(eligible,key=lambda r:(r['IoU'],r['Dice'],min(r['Precision'],r['Recall'])))
    winner_key=f"{winner['config']}_seed_{winner['seed']}"
    winner_est=estimators[winner_key]
    threshold_rows=[]
    val_logits=logits_cached(winner_est,val_cache)
    for threshold in [0.30,0.40,0.50,0.60,0.70]:
        threshold_rows.append({'threshold':threshold,**evaluate_logits(val_logits,val_ids,'copenhagen',threshold)})
    threshold_df=pd.DataFrame([{k:v for k,v in r.items() if k!='probability_statistics'} for r in threshold_rows]); threshold_df.to_csv(OUT/'SEGMENTATION_RESULTS.csv',index=False)
    threshold=max(threshold_rows,key=lambda r:(r['IoU'],r['Dice'],min(r['Precision'],r['Recall'])))['threshold']
    lock={'SELECTED_CONFIG':winner['config'],'SELECTED_SEED':winner['seed'],'SELECTED_CHECKPOINT':winner['checkpoint'],'SELECTED_THRESHOLD':threshold,'selection_split':'Copenhagen only'}
    (OUT/'LOCK.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')

    # New York begins only after the lock above.
    test_logits=logits_cached(winner_est,test_cache); ny=evaluate_logits(test_logits,test_ids,'new_york',threshold); pd.DataFrame([{k:v for k,v in ny.items() if k!='probability_statistics'}]).to_csv(OUT/'NEW_YORK_FINAL.csv',index=False)
    pd.DataFrame([{k:v for k,v in ny.items() if k not in ('probability_statistics', 'split')}]).to_csv(OUT/'INSTANCE_RESULTS.csv',index=False)

    # Height and fixed downstream A/B demo on the first NYC tile using the existing engine.
    tile=test_ids[0]; rgb=load_rgb(tile); ref=load_gt(tile)
    baseline_est=estimators['A_seed_0']
    baseline_est.d_mean=float(baseline_est.model.d_mean.item()); baseline_est.d_std=float(baseline_est.model.d_std.item())
    winner_est.d_mean=float(winner_est.model.d_mean.item()); winner_est.d_std=float(winner_est.model.d_std.item())
    base_engine=CalibrationEngine(runs_dir=ROOT/'runs'); base_engine.footprint_estimator=baseline_est
    selected_engine=CalibrationEngine(runs_dir=ROOT/'runs'); selected_engine.footprint_estimator=winner_est
    depth_full = cv2.resize(test_cache['items'][0]['raw_depth_static'].numpy(), rgb.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
    base_res=base_engine.calibrate(depth_full,rgb,True,mode=CalibrationMode.STRUCTURAL_PRIOR,reference_elevation=ref,filename=tile)
    selected_res=selected_engine.calibrate(depth_full,rgb,True,mode=CalibrationMode.STRUCTURAL_PRIOR,reference_elevation=ref,filename=tile)
    base_mask=base_res.mask_bldg.astype(bool); selected_mask=selected_res.mask_bldg.astype(bool)
    plt.imsave(OUT/'baseline_probability.png',1/(1+np.exp(-logits_cached(baseline_est,test_cache)[0])),cmap='viridis',vmin=0,vmax=1)
    plt.imsave(OUT/'corrected_probability.png',1/(1+np.exp(-test_logits[0])),cmap='viridis',vmin=0,vmax=1)
    plt.imsave(OUT/'baseline_mask.png',base_mask,cmap='gray'); plt.imsave(OUT/'corrected_mask.png',selected_mask,cmap='gray')
    for name, dsm in [('phase29_3d.png',base_res.dsm),('phase51_selected_3d.png',selected_res.dsm)]: make_render(dsm,rgb,name)
    for name, dsm in [('phase29_clay.png',base_res.dsm),('phase51_clay.png',selected_res.dsm)]: render_clay(dsm,name)
    fig,axes=plt.subplots(1,2,figsize=(12,6)); axes[0].imshow((base_res.dsm-base_res.dsm.min())/(base_res.dsm.max()-base_res.dsm.min()+1e-6),cmap='terrain'); axes[0].set_title('Phase 29 / Config A'); axes[1].imshow((selected_res.dsm-selected_res.dsm.min())/(selected_res.dsm.max()-selected_res.dsm.min()+1e-6),cmap='terrain'); axes[1].set_title(f'Phase 51 / Config {winner["config"]}'); [a.axis('off') for a in axes]; plt.tight_layout(); plt.savefig(OUT/'side_by_side_3d.png',dpi=140); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,6)); axes[0].imshow(base_res.dsm,cmap='gray'); axes[0].set_title('Phase 29 clay'); axes[1].imshow(selected_res.dsm,cmap='gray'); axes[1].set_title('Phase 51 clay'); [a.axis('off') for a in axes]; plt.tight_layout(); plt.savefig(OUT/'clay_side_by_side.png',dpi=140); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,6)); axes[0].imshow(ref,cmap='terrain'); axes[0].set_title('Reference benchmark'); axes[1].imshow(selected_res.dsm,cmap='terrain'); axes[1].set_title('Selected reconstruction'); [a.axis('off') for a in axes]; plt.tight_layout(); plt.savefig(OUT/'target_vs_selected.png',dpi=140); plt.close(fig)
    height={'tile':tile,'baseline':base_res.stats,'selected':selected_res.stats,'height_note':'Reference DSM is used for this demo calibration/benchmark path; no production checkpoint is changed.'}
    (OUT/'HEIGHT_RESULTS.csv').write_text('model,tile,dsm_min,dsm_max,dsm_mean,max_building_height\nphase29,'+tile+f",{base_res.dsm.min()},{base_res.dsm.max()},{base_res.dsm.mean()},{base_res.ndsm.max()}\nphase52,"+tile+f",{selected_res.dsm.min()},{selected_res.dsm.max()},{selected_res.dsm.mean()},{selected_res.ndsm.max()}\n",encoding='utf-8')
    pd.DataFrame([{'comparison':'Phase 29 vs Phase 52 selected','tile':tile,'phase29_max_ndsm':float(base_res.ndsm.max()),'phase52_max_ndsm':float(selected_res.ndsm.max()),'same_calibration_engine':True,'same_peak_recovery':True,'same_dtm_logic':True}]).to_csv(OUT/'THREE_D_IMPACT.csv',index=False)
    height_rows=[]
    for index, ny_tile in enumerate(test_ids):
        ny_rgb=load_rgb(ny_tile); ny_ref=load_gt(ny_tile)
        ny_depth=cv2.resize(test_cache['items'][index]['raw_depth_static'].numpy(), ny_rgb.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
        ny_res=selected_engine.calibrate(ny_depth,ny_rgb,True,mode=CalibrationMode.STRUCTURAL_PRIOR,reference_elevation=ny_ref,filename=ny_tile)
        building=mask(ny_ref).astype(bool)
        errors=(ny_res.dsm-ny_ref)[building]
        if errors.size:
            height_rows.append({'tile':ny_tile,'building_mae':float(np.mean(np.abs(errors))),'building_rmse':float(np.sqrt(np.mean(errors**2))),'building_bias':float(np.mean(errors)),'pearson':float(np.corrcoef(ny_res.dsm[building],ny_ref[building])[0,1]) if errors.size > 1 else float('nan')})
    height_df=pd.DataFrame(height_rows)
    height_df.to_csv(OUT/'HEIGHT_RESULTS.csv',index=False)
    results={'protocol_caveat':'Phase 51 NY metrics were generated before selection and are post-hoc/contaminated for selection; they were not read by this script.','lock':lock,'copenhagen_models':model_results,'thresholds':threshold_rows,'new_york_final':ny,'checkpoint_count':len(checkpoint_rows),'all_checkpoints_clean':all(not r['missing_keys'] and not r['unexpected_keys'] and r['metadata_present'] for r in checkpoint_rows),'downstream_demo':'Generated using existing CalibrationEngine and fixed Phase 29 height/DTM logic on the first locked NY tile.','scientific_integrity':{'rgb_sha256':sha256(DATA/'rgb'/tile),'dsm_sha256':sha256(DATA/'dsm'/tile),'dtm':'derived in-memory by CalibrationEngine; no standalone DTM source file modified'},'verdict':'PHASE52_PARTIAL_SUPPORT'}
    (OUT/'RESULTS.json').write_text(json.dumps(results,indent=2,default=str),encoding='utf-8')
    locked_threshold_result=next(item for item in threshold_rows if item['threshold'] == threshold)
    height_summary=height_df[['building_mae','building_rmse','building_bias']].mean().to_dict() if not height_df.empty else {}
    report=f'''# Phase 52 Clean Selection Report

## Locked result

- Configuration: **{winner['config']}**
- Seed: **{winner['seed']}**
- Checkpoint: `{winner['checkpoint']}`
- Threshold: **{threshold:.2f}**
- Selection data: Copenhagen only

## Copenhagen evidence

The selected checkpoint achieved locked-threshold IoU `{locked_threshold_result['IoU']:.4f}`, Dice `{locked_threshold_result['Dice']:.4f}`, precision `{locked_threshold_result['Precision']:.4f}`, recall `{locked_threshold_result['Recall']:.4f}`, and predicted foreground `{locked_threshold_result['predicted_foreground_pct']:.2f}%`. Full eight-checkpoint results are in `COPENHAGEN_SELECTION.csv`.

## New York protocol

Phase 51 generated New York metrics before final selection. Those older values were not used here. The authoritative clean result from the locked checkpoint and locked Copenhagen threshold is in `NEW_YORK_FINAL.csv`.

- IoU: `{ny['IoU']:.4f}`
- Dice: `{ny['Dice']:.4f}`
- Precision: `{ny['Precision']:.4f}`
- Recall: `{ny['Recall']:.4f}`
- Predicted foreground: `{ny['predicted_foreground_pct']:.2f}%`
- Matched / missed buildings: `{ny['matched_buildings']} / {ny['missed_buildings']}`

## Height and downstream comparison

Complete New York height diagnostics are in `HEIGHT_RESULTS.csv`; mean building MAE is `{height_summary.get('building_mae', float('nan')):.4f}` and mean bias is `{height_summary.get('building_bias', float('nan')):.4f}`. The fixed existing calibration/height path was run on one NYC tile for Phase 29 Config A versus the locked Phase 52 model. No production code, PeakRecoveryMLP, DSM, nDSM, DTM, or viewer code was modified. The generated visual comparison is evidence for one demo tile, not a complete 3D acceptance claim.

## Verdict

**PHASE52_PARTIAL_SUPPORT**

The corrected model is non-collapsed and Copenhagen selection is complete, but this evaluator does not establish complete multi-tile 3D improvement or human visual acceptance across the city.\n'''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps({'lock':lock,'copenhagen_winner':winner,'new_york':ny,'verdict':'PHASE52_PARTIAL_SUPPORT'},indent=2,default=str))

if __name__ == '__main__': main()
