import os
import sys
import json
import time
import pandas as pd
import numpy as np
import cv2
import torch
from pathlib import Path

# Fix for ZoeDepth dependencies
import urllib.request
urllib.request.urlretrieve("https://raw.githubusercontent.com/isl-org/ZoeDepth/main/zoedepth/models/zoedepth/zoedepth_v1.py", "dummy.py")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.depth.depth_anything import DepthAnythingV2
from depthwizard.config import DepthConfig

MANIFEST_PATH = "runs/dfc2023_multicity_prep/split_manifest.csv"
DATA_DIR = Path("data/dfc2023_multicity")
OUT_DIR = Path("runs/phase10_depth_model_probe")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading models...")
    dcfg = DepthConfig(cache_dir=str(DATA_DIR / "depth_cache"))
    da_model = DepthAnythingV2(dcfg.model_id, dcfg.input_size, dcfg.cache_dir, True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load ZoeDepth, trust repo explicitly
    zoe_model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True, trust_repo=True).to(device)
    zoe_model.eval()

    print("Selecting probe tiles...")
    df = pd.read_csv(MANIFEST_PATH)
    # Select 3 cities: Berlin (train), Copenhagen (val), NewYork (test)
    cities = ["Berlin", "Copenhagen", "NewYork"]
    selected_tids = []
    
    for c in cities:
        city_tids = df[df['tile_id'].str.contains(c)]['tile_id'].tolist()
        # Take 5 tiles per city
        selected_tids.extend(city_tids[:5])
        
    print(f"Selected {len(selected_tids)} tiles.")
    
    results = []
    from scipy.stats import pearsonr, spearmanr
    
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    for tid in selected_tids:
        city = next((c for c in cities if c in tid), "Unknown")
        rgb_path = DATA_DIR / "rgb" / tid
        dsm_path = DATA_DIR / "dsm" / tid
        
        rgb = cv2.imread(str(rgb_path))
        if rgb is None: continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        gt = cv2.imread(str(dsm_path), cv2.IMREAD_UNCHANGED)
        if gt is None: continue
        gt = gt.astype(np.float32)
        valid = np.isfinite(gt) & (gt != -999.0)
        
        # Inference DA-V2
        da_depth = da_model.infer(rgb, tid, target_hw=rgb.shape[:2])
        
        # Inference ZoeDepth
        # ZoeDepth infer_pil takes a PIL Image. 
        from PIL import Image
        pil_img = Image.fromarray(rgb)
        with torch.no_grad():
            zoe_depth = zoe_model.infer_pil(pil_img)
            
        zoe_depth = cv2.resize(zoe_depth, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Extract valid pixels
        gt_v = gt[valid]
        da_v = da_depth[valid]
        zoe_v = zoe_depth[valid]
        
        res = {"city": city, "tid": tid}
        
        for t_name, threshold in [("all", -1), (">15m", 15), (">20m", 20), (">30m", 30), (">40m", 40)]:
            mask = gt_v > threshold
            if mask.sum() < 10:
                continue
                
            gt_t = gt_v[mask]
            da_t = da_v[mask]
            zoe_t = zoe_v[mask]
            
            p_da, _ = pearsonr(gt_t, da_t)
            s_da, _ = spearmanr(gt_t, da_t)
            
            p_zoe, _ = pearsonr(gt_t, zoe_t)
            s_zoe, _ = spearmanr(gt_t, zoe_t)
            
            res[f"DA_{t_name}_pearson"] = p_da
            res[f"DA_{t_name}_spearman"] = s_da
            res[f"Zoe_{t_name}_pearson"] = p_zoe
            res[f"Zoe_{t_name}_spearman"] = s_zoe
            
            if t_name == "all":
                # Metric analysis for ZoeDepth
                mae = np.abs(gt_t - zoe_t).mean()
                res["Zoe_MAE"] = mae
                
        results.append(res)
        
        # Generate one scatter plot per city (first tile)
        if tid == [t for t in selected_tids if city in t][0]:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].scatter(gt_v[::100], da_v[::100], alpha=0.1, s=1)
            ax[0].set_title(f"DA-V2 vs GT ({city})")
            ax[0].set_xlabel("GT Height (m)")
            ax[0].set_ylabel("DA-V2 Relative Depth")
            
            ax[1].scatter(gt_v[::100], zoe_v[::100], alpha=0.1, s=1)
            # Add y=x line
            m_max = max(gt_v.max(), zoe_v.max())
            ax[1].plot([0, m_max], [0, m_max], 'r--')
            ax[1].set_title(f"ZoeDepth vs GT ({city})")
            ax[1].set_xlabel("GT Height (m)")
            ax[1].set_ylabel("ZoeDepth Metric Depth (m)")
            
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"scatter_{city}.png", dpi=150)
            plt.close()

    df_res = pd.DataFrame(results)
    print("\nPROBE RESULTS:")
    print(df_res.groupby("city").mean(numeric_only=True).T)
    df_res.to_csv(OUT_DIR / "probe_metrics.csv", index=False)
    
if __name__ == "__main__":
    main()
