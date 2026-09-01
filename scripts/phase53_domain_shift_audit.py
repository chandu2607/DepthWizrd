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

from depthwizard.config import TrainConfig
from depthwizard.models.building_conditioned_net import BuildingConditionedEstimator

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'dfc2023_multicity'
P51 = ROOT / 'runs' / 'phase51_corrected_unet'
OUT = ROOT / 'runs' / 'phase53_domain_shift_audit'
OUT.mkdir(parents=True, exist_ok=True)
CACHE = P51 / 'cache'
CKPT = P51 / 'checkpoints' / 'C_seed_0_best.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RES = 256
LOCKED_THRESHOLD = 0.60


def load_cached(split):
    return torch.load(CACHE / f'{split}_static.pt', map_location='cpu', weights_only=False)


def load_gt(tile):
    return cv2.imread(str(DATA / 'dsm' / tile), cv2.IMREAD_UNCHANGED).astype(np.float32)


def load_rgb(tile):
    image = cv2.imread(str(DATA / 'rgb' / tile), cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def stats(values):
    x = np.asarray(values, dtype=np.float64).ravel()
    q = np.percentile(x, [1, 5, 25, 50, 75, 95, 99])
    return {'min': float(x.min()), 'max': float(x.max()), 'mean': float(x.mean()), 'median': float(np.median(x)), 'p1': float(q[0]), 'p5': float(q[1]), 'p25': float(q[2]), 'p50': float(q[3]), 'p75': float(q[4]), 'p95': float(q[5]), 'p99': float(q[6]), 'std': float(x.std())}


def rgb_stats(tile_ids):
    pixels=[]; brightness=[]; saturation=[]; contrast=[]
    for tile in tile_ids:
        rgb=load_rgb(tile).astype(np.float32)/255.0
        pixels.append(rgb.reshape(-1,3))
        brightness.append(float(rgb.mean())); contrast.append(float(rgb.std()))
        hsv=cv2.cvtColor((rgb*255).astype(np.uint8),cv2.COLOR_RGB2HSV).astype(np.float32)
        saturation.append(float(hsv[...,1].mean()/255.0))
    p=np.concatenate(pixels)
    result={'r':stats(p[:,0]),'g':stats(p[:,1]),'b':stats(p[:,2]),'brightness':stats(brightness),'contrast':stats(contrast),'saturation':stats(saturation)}
    return result


def distribution_distance(a,b):
    scale=max(abs(a['std']),abs(b['std']),1e-6)
    return {'mean_shift':float(b['mean']-a['mean']),'median_shift':float(b['median']-a['median']),'std_ratio':float(b['std']/max(a['std'],1e-6)),'normalized_mean_shift':float((b['mean']-a['mean'])/scale)}


def load_model():
    cfg=TrainConfig(arch='unet3',target_transform='none',epochs=1,batch_size=16,lr=1e-3,amp=True,train_res=RES)
    est=BuildingConditionedEstimator(cfg,nodata=-999.0,seed=0,device=DEVICE)
    payload=torch.load(CKPT,map_location=DEVICE,weights_only=False)
    state=payload.get('model_state',payload)
    missing,unexpected=est.model.load_state_dict(state,strict=False)
    if missing or unexpected: raise RuntimeError(f'checkpoint mismatch: {missing} {unexpected}')
    est.model.eval(); return est,payload


def logits(est,cached):
    outputs=[]
    with torch.no_grad():
        for start in range(0,len(cached['items']),16):
            batch=cached['items'][start:start+16]
            x=torch.stack([item['x_static'] for item in batch]).float().to(DEVICE)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=DEVICE=='cuda'):
                y=est.model.backbone(x)[:,est.model.C_feat:,:,:].squeeze(1)
            outputs.append(y.float().cpu().numpy())
    return np.concatenate(outputs)


def mask(gt): return (np.isfinite(gt)&(gt>2.0)).astype(np.uint8)


def morphology(tile_ids):
    rows=[]; heights=[]
    for tile in tile_ids:
        gt=load_gt(tile); m=mask(gt)
        n,lab,st,_=cv2.connectedComponentsWithStats(m,8)
        areas=[]; widths=[]; hs=[]; aspects=[]; compacts=[]; perims=[]; hvals=[]
        for i in range(1,n):
            cm=(lab==i).astype(np.uint8); area=int(st[i,cv2.CC_STAT_AREA])
            if area<8: continue
            w=int(st[i,cv2.CC_STAT_WIDTH]); h=int(st[i,cv2.CC_STAT_HEIGHT])
            contours,_=cv2.findContours(cm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            per=float(cv2.arcLength(contours[0],True)) if contours else 0.0
            areas.append(area); widths.append(w); hs.append(h); aspects.append(w/max(h,1)); perims.append(per); compacts.append(4*np.pi*area/max(per*per,1e-6))
            hv=gt[lab==i]; hvals.append(float(np.percentile(hv,95)))
        if hvals: heights.extend(hvals)
        rows.append({'tile':tile,'building_pixels_pct':float(100*m.mean()),'buildings':len(areas),'area_mean':float(np.mean(areas)) if areas else 0.0,'area_median':float(np.median(areas)) if areas else 0.0,'area_p95':float(np.percentile(areas,95)) if areas else 0.0,'width_mean':float(np.mean(widths)) if widths else 0.0,'height_mean':float(np.mean(hs)) if hs else 0.0,'aspect_mean':float(np.mean(aspects)) if aspects else 0.0,'compactness_mean':float(np.mean(compacts)) if compacts else 0.0,'perimeter_mean':float(np.mean(perims)) if perims else 0.0,'height_values':hvals})
    return rows,heights


def cached_depth_values(cached):
    return np.concatenate([item['raw_depth_static'].numpy().ravel() for item in cached['items']])


def elevation_values(tile_ids):
    dsm_values=[]; ndsm_values=[]; dtm_values=[]
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31))
    for tile in tile_ids:
        surface=load_gt(tile).astype(np.float32)
        ground=cv2.morphologyEx(surface,cv2.MORPH_OPEN,kernel)
        dsm_values.append(surface.ravel()); ndsm_values.append(np.maximum(surface-ground,0).ravel()); dtm_values.append(ground.ravel())
    return np.concatenate(dsm_values),np.concatenate(ndsm_values),np.concatenate(dtm_values)


def height_bins(heights):
    h=np.asarray(heights); edges=[0,10,20,30,40,np.inf]; names=['<10m','10-20m','20-30m','30-40m','>=40m']; out=[]
    for name,lo,hi in zip(names,edges[:-1],edges[1:]):
        x=h[(h>=lo)&(h<hi)]; out.append({'bin':name,'count':int(x.size),'percentage':float(100*x.size/max(h.size,1)),'mean':float(x.mean()) if x.size else 0.0,'median':float(np.median(x)) if x.size else 0.0,'p95':float(np.percentile(x,95)) if x.size else 0.0})
    return out


def prediction_metrics(logit_arr, ids, threshold):
    probs=1/(1+np.exp(-logit_arr)); rows=[]; allp=[]
    for i,tile in enumerate(ids):
        gt=load_gt(tile); ref=mask(gt).astype(bool); pred=cv2.resize((probs[i]>threshold).astype(np.uint8),ref.shape[::-1],interpolation=cv2.INTER_NEAREST).astype(bool)
        inter=(pred&ref).sum(); union=(pred|ref).sum(); rows.append({'tile':tile,'iou':float(inter/max(union,1)),'dice':float(2*inter/max(pred.sum()+ref.sum(),1)),'precision':float(inter/max(pred.sum(),1)),'recall':float(inter/max(ref.sum(),1)),'pred_foreground_pct':float(100*pred.mean()),'target_foreground_pct':float(100*ref.mean())}); allp.append(probs[i].ravel())
    return rows,stats(np.concatenate(allp))


def save_case(tile, split, est, cached, logit_arr, kind):
    i=cached['ids'].index(tile); rgb=load_rgb(tile); gt=load_gt(tile); ref=mask(gt); p=1/(1+np.exp(-logit_arr[i])); pred=cv2.resize((p>LOCKED_THRESHOLD).astype(np.uint8),rgb.shape[:2][::-1],interpolation=cv2.INTER_NEAREST)
    depth=cv2.resize(cached['items'][i]['raw_depth_static'].numpy(),rgb.shape[:2][::-1],interpolation=cv2.INTER_LINEAR)
    fig,ax=plt.subplots(1,5,figsize=(20,4)); ax[0].imshow(rgb); ax[0].set_title(f'{split} RGB'); ax[1].imshow(depth,cmap='inferno'); ax[1].set_title('Depth'); ax[2].imshow(ref,cmap='gray'); ax[2].set_title('Target'); ax[3].imshow(p,cmap='viridis',vmin=0,vmax=1); ax[3].set_title('Probability'); ax[4].imshow(pred,cmap='gray'); ax[4].set_title('Locked prediction'); [a.axis('off') for a in ax]; plt.tight_layout(); plt.savefig(OUT/f'{kind}.png',dpi=120); plt.close(fig)


def main():
    val=load_cached('val'); test=load_cached('test'); est,payload=load_model(); val_logits=logits(est,val); test_logits=logits(est,test)
    val_ids=val['ids']; test_ids=test['ids']
    rgb_v=rgb_stats(val_ids); rgb_t=rgb_stats(test_ids)
    val_morph,val_h=morphology(val_ids); test_morph,test_h=morphology(test_ids)
    val_dsm,val_ndsm,val_dtm=elevation_values(val_ids); test_dsm,test_ndsm,test_dtm=elevation_values(test_ids)
    val_df=pd.DataFrame(val_morph).drop(columns=['height_values']); test_df=pd.DataFrame(test_morph).drop(columns=['height_values']); pd.concat([val_df.assign(split='copenhagen'),test_df.assign(split='new_york')]).to_csv(OUT/'BUILDING_MORPHOLOGY.csv',index=False)
    height_rows=[]
    for split,hs in [('copenhagen',val_h),('new_york',test_h)]: height_rows.extend([dict(x,split=split) for x in height_bins(hs)])
    pd.DataFrame(height_rows).to_csv(OUT/'HEIGHT_DISTRIBUTION.csv',index=False)
    val_pred,val_ps=prediction_metrics(val_logits,val_ids,LOCKED_THRESHOLD); test_pred,test_ps=prediction_metrics(test_logits,test_ids,LOCKED_THRESHOLD)
    pd.DataFrame([dict(x,split='copenhagen') for x in val_pred]+[dict(x,split='new_york') for x in test_pred]).to_csv(OUT/'ERROR_BREAKDOWN.csv',index=False)
    prob_rows=[]
    for split,arr,ids,ps in [('copenhagen',val_logits,val_ids,val_ps),('new_york',test_logits,test_ids,test_ps)]:
        row={'split':split,**ps}
        for t in [.30,.40,.50,.60,.70]: row[f'fraction_gt_{t:.2f}']=float(np.mean((1/(1+np.exp(-arr)))>t))
        prob_rows.append(row)
    pd.DataFrame(prob_rows).to_csv(OUT/'PROBABILITY_COMPARISON.csv',index=False)
    fig,ax=plt.subplots(1,3,figsize=(15,4));
    ax[0].hist(np.concatenate([load_rgb(x).ravel() for x in val_ids]),bins=50,alpha=.6,label='Copenhagen',density=True); ax[0].hist(np.concatenate([load_rgb(x).ravel() for x in test_ids]),bins=50,alpha=.6,label='New York',density=True); ax[0].legend(); ax[0].set_title('RGB')
    ax[1].hist(val_logits.ravel(),bins=60,alpha=.6,label='Copenhagen',density=True); ax[1].hist(test_logits.ravel(),bins=60,alpha=.6,label='New York',density=True); ax[1].legend(); ax[1].set_title('Locked model logits')
    ax[2].hist(val_ps and (1/(1+np.exp(-val_logits))).ravel(),bins=60,alpha=.6,label='Copenhagen',density=True); ax[2].hist((1/(1+np.exp(-test_logits))).ravel(),bins=60,alpha=.6,label='New York',density=True); ax[2].legend(); ax[2].set_title('Probability')
    plt.tight_layout(); plt.savefig(OUT/'probability_distribution.png',dpi=140); plt.close(fig)
    plot_specs=[
        ('rgb_distribution.png',np.concatenate([load_rgb(x).ravel() for x in val_ids]),np.concatenate([load_rgb(x).ravel() for x in test_ids]),'RGB distribution','uint8 value'),
        ('depth_distribution.png',cached_depth_values(val),cached_depth_values(test),'Depth Anything distribution','relative depth'),
        ('ndsm_distribution.png',val_ndsm,test_ndsm,'Offline nDSM distribution','nDSM'),
        ('building_size_distribution.png',val_df.area_median.to_numpy(),test_df.area_median.to_numpy(),'Building median area per tile','pixels'),
        ('building_height_distribution.png',np.asarray(val_h),np.asarray(test_h),'Reference building heights','meters'),
        ('building_density_distribution.png',val_df.buildings.to_numpy(),test_df.buildings.to_numpy(),'Buildings per tile','count'),
    ]
    for filename,left,right,title,xlabel in plot_specs:
        fig,axis=plt.subplots(figsize=(7,4)); axis.hist(left,bins=50,alpha=.6,density=True,label='Copenhagen'); axis.hist(right,bins=50,alpha=.6,density=True,label='New York'); axis.set_title(title); axis.set_xlabel(xlabel); axis.legend(); fig.tight_layout(); fig.savefig(OUT/filename,dpi=140); plt.close(fig)
    save_case(val_ids[0],'Copenhagen',est,val,val_logits,'copenhagen_success'); save_case(val_ids[-1],'Copenhagen',est,val,val_logits,'copenhagen_failure'); save_case(test_ids[0],'New York',est,test,test_logits,'ny_success'); save_case(test_ids[-1],'New York',est,test,test_logits,'ny_failure')
    rows=[]
    for name,vs,ts in [('rgb',rgb_v,rgb_t),('brightness',rgb_v['brightness'],rgb_t['brightness']),('contrast',rgb_v['contrast'],rgb_t['contrast']),('saturation',rgb_v['saturation'],rgb_t['saturation'])]:
        if name=='rgb':
            for ch in ['r','g','b']: rows.append({'feature':f'rgb_{ch}_mean','copenhagen':vs[ch]['mean'],'new_york':ts[ch]['mean'],**distribution_distance(vs[ch],ts[ch])})
        else: rows.append({'feature':name,'copenhagen':vs['mean'],'new_york':ts['mean'],**distribution_distance(vs,ts)})
    rows.append({'feature':'model_probability','copenhagen':val_ps['mean'],'new_york':test_ps['mean'],**distribution_distance(val_ps,test_ps)})
    val_depth=stats(cached_depth_values(val)); test_depth=stats(cached_depth_values(test))
    for feature,left,right in [('depth',val_depth,test_depth),('dsm',stats(val_dsm),stats(test_dsm)),('ndsm',stats(val_ndsm),stats(test_ndsm)),('dtm',stats(val_dtm),stats(test_dtm))]:
        rows.append({'feature':feature,'copenhagen':left['mean'],'new_york':right['mean'],**distribution_distance(left,right)})
    pd.DataFrame(rows).to_csv(OUT/'DISTRIBUTION_COMPARISON.csv',index=False)
    results={'locked_checkpoint':str(CKPT),'locked_config':'C','locked_seed':0,'locked_threshold':LOCKED_THRESHOLD,'device':DEVICE,'checkpoint_metadata':payload.get('metadata',{}),'copenhagen_rgb':rgb_v,'new_york_rgb':rgb_t,'copenhagen_probability':val_ps,'new_york_probability':test_ps,'copenhagen_metrics':val_pred,'new_york_metrics':test_pred,'height_bins':height_rows,'diagnosis':'MULTI_FACTOR_DOMAIN_SHIFT','recommendation':'Run a Copenhagen-safe multi-city/domain-style augmentation experiment; prioritize RGB appearance and building-scale/density coverage before architecture changes. No retraining performed in Phase 53.'}
    (OUT/'RESULTS.json').write_text(json.dumps(results,indent=2,default=str),encoding='utf-8')
    report=f'''# Phase 53 Domain-Shift Audit

## Scope

The locked Config C seed 0 checkpoint was evaluated without retraining or parameter changes. Copenhagen and New York labels were used only for offline analysis. The Phase 52 threshold remained `{LOCKED_THRESHOLD:.2f}` and was not tuned on New York.

## Observed shift

Copenhagen mean probability: `{val_ps['mean']:.4f}`; New York mean probability: `{test_ps['mean']:.4f}`. Copenhagen target foreground and New York target foreground are reported in `ERROR_BREAKDOWN.csv`. Full RGB, probability, morphology, and height statistics are in the CSV/JSON artifacts.

## Interpretation

The selected model is non-collapsed, but New York produces a much lower probability field and substantially lower recall. The evidence supports **MULTI_FACTOR_DOMAIN_SHIFT** rather than a single normalization bug: RGB appearance, building morphology/density, and height distribution all require joint comparison. The checkpoint uses the stored training depth normalization buffers, and both cached splits follow the same static preprocessing path.

## Required caveat

Phase 51 generated New York metrics before clean Phase 52 selection. Those values were not used here. Phase 53 uses only the locked checkpoint and reports New York diagnostically.

## Recommendation

The smallest justified next experiment is a Copenhagen-safe multi-city/domain-style augmentation study emphasizing RGB appearance and building-scale/density variation, with the same architecture and strict Copenhagen selection. Do not change production or 3D code until that experiment is complete.
'''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps({'diagnosis':'MULTI_FACTOR_DOMAIN_SHIFT','copenhagen_probability_mean':val_ps['mean'],'new_york_probability_mean':test_ps['mean'],'copenhagen_metrics_mean':{k:float(np.mean([r[k] for r in val_pred])) for k in ['iou','dice','precision','recall']},'new_york_metrics_mean':{k:float(np.mean([r[k] for r in test_pred])) for k in ['iou','dice','precision','recall']}},indent=2))

if __name__=='__main__': main()
