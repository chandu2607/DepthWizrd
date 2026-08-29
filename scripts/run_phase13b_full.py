import sys
import json
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.config import TrainConfig
from depthwizard.models.fusion_head import LearnedFusionHead

DATA_DIR = Path("data/dfc2023_multicity")
OUT_DIR = Path("runs/phase13b_ordinal")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_split(manifest_path, split_type):
    df = pd.read_csv(manifest_path)
    if split_type == 'train':
        df = df[df['split'] == 'train']
    else:
        df = df[df['split'] == split_type]
    return df['tile_id'].tolist()

def load_samples(tile_ids):
    from depthwizard.depth.depth_anything import DepthAnythingV2
    from depthwizard.config import DepthConfig
    dcfg = DepthConfig(cache_dir=str(DATA_DIR / "depth_cache"))
    depth_model = DepthAnythingV2(dcfg.model_id, dcfg.input_size, dcfg.cache_dir, True)

    samples = []
    for tid in tile_ids:
        rgb_path = DATA_DIR / "rgb" / tid
        dsm_path = DATA_DIR / "dsm" / tid
        rgb = cv2.imread(str(rgb_path))
        if rgb is None: continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(dsm_path), cv2.IMREAD_UNCHANGED)
        if gt is None: continue
        gt = gt.astype(np.float32)
        depth = depth_model.infer(rgb, tid, target_hw=rgb.shape[:2])
        samples.append({"id": tid, "rgb": rgb, "gt": gt, "depth": depth, "nodata": -999.0})
    return samples

def compute_metrics(true_classes, pred_classes):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_classes, pred_classes, labels=list(range(8)))
    
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    macro_f1 = np.mean(f1)
    
    acc = tp.sum() / (cm.sum() + 1e-8)
    
    def binary_stats(classes_to_include):
        mask_true = np.isin(true_classes, classes_to_include)
        mask_pred = np.isin(pred_classes, classes_to_include)
        tp_b = (mask_true & mask_pred).sum()
        fp_b = (~mask_true & mask_pred).sum()
        fn_b = (mask_true & ~mask_pred).sum()
        p = tp_b / (tp_b + fp_b + 1e-8)
        r = tp_b / (tp_b + fn_b + 1e-8)
        return float(p), float(r)
        
    p15, r15 = binary_stats([4, 5, 6, 7])
    p20, r20 = binary_stats([5, 6, 7])
    p30, r30 = binary_stats([6, 7])
    p40, r40 = binary_stats([7])
    
    return {
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "cm": cm.tolist(),
        "p_15": p15, "r_15": r15,
        "p_20": p20, "r_20": r20,
        "p_30": p30, "r_30": r30,
        "p_40": p40, "r_40": r40
    }

def main():
    print("Loading data...")
    manifest = Path("runs/dfc2023_multicity_prep/split_manifest.csv")
    train_tids = load_split(manifest, "train")
    test_tids = load_split(manifest, "test")
    
    train_samples = load_samples(train_tids)
    test_samples = load_samples(test_tids)
    
    bins = np.array([0, 2, 5, 10, 15, 20, 30, 40, np.inf])
    
    all_results = []
    for seed in [0, 1]:
        print(f"\n--- SEED {seed} ---")
        cfg = TrainConfig(
            arch="unet3",
            target_transform="classification",
            loss_type="standard",
            epochs=15,
            batch_size=4,
            lr=1e-3,
            train_res=256
        )
        cfg.input_mode = "depth"
        
        model = LearnedFusionHead(cfg, nodata=-999.0, seed=seed)
        model.fit(train_samples)
        
        all_true = []
        all_pred = []
        
        for i, s in enumerate(test_samples):
            pred = model.predict(s)
            gt = s["gt"]
            valid = (np.isfinite(gt)) & (gt != -999.0)
            
            gt_v = gt[valid]
            pred_v = pred[valid]
            
            gt_c = np.digitize(np.maximum(gt_v, 0.0), bins) - 1
            gt_c = np.clip(gt_c, 0, 7)
            
            all_true.extend(gt_c.tolist())
            all_pred.extend(pred_v.tolist())
            
            # Save visual for seed 0, first few interesting tiles
            if seed == 0 and "NewYork" in s["id"] and gt_v.max() > 40:
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(s["rgb"])
                plt.title("RGB")
                plt.axis("off")
                
                plt.subplot(1, 3, 2)
                gt_c_img = np.digitize(np.maximum(gt, 0.0), bins) - 1
                gt_c_img = np.clip(gt_c_img, 0, 7)
                gt_c_img = np.where(valid, gt_c_img, -1)
                plt.imshow(gt_c_img, cmap="nipy_spectral", vmin=-1, vmax=7)
                plt.title("GT Classes")
                plt.axis("off")
                
                plt.subplot(1, 3, 3)
                pred_img = np.where(valid, pred, -1)
                plt.imshow(pred_img, cmap="nipy_spectral", vmin=-1, vmax=7)
                plt.title("Pred Classes")
                plt.axis("off")
                
                plt.tight_layout()
                plt.savefig(OUT_DIR / f"vis_{s['id'].replace('.tif', '')}.png")
                plt.close()
                
        mets = compute_metrics(all_true, all_pred)
        all_results.append({"seed": seed, "metrics": mets})
        
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
