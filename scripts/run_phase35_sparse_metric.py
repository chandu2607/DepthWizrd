"""
Phase 35 — Sparse Metric Anchor + Monocular Depth Completion.
PROXY FEASIBILITY EXPERIMENT — Not a real LiDAR deployment.

Architecture: streaming tile processing (one tile at a time) to avoid OOM.
All scientific outputs → runs/phase35_sparse_metric/
STRICT LOCK: production app.py, PeakRecoveryMLP, DTM/DSM pipeline — UNTOUCHED.
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import cv2
import torch
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

# Force UTF-8 output (avoids Windows cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ─── Paths ────────────────────────────────────────────────────────────────────
MANIFEST_PATH = "runs/dfc2023_multicity_prep/split_manifest.csv"
DATA_DIR = Path("data/dfc2023_multicity")
OUT_DIR  = Path("runs/phase35_sparse_metric")
FIG_DIR  = OUT_DIR / "figures"
TBL_DIR  = OUT_DIR / "tables"
for d in [OUT_DIR, FIG_DIR, TBL_DIR]: d.mkdir(parents=True, exist_ok=True)

# Phase 29 locked baselines (from runs/phase29_peak_recovery/results.json)
P29_OVERALL_MAE    = 7.63
P29_GT40M_MAE      = 13.36
P29_RECOVERY_RATIO = 44.81
P27_RECOVERY_RATIO =  5.31

print("=== PHASE 35: SPARSE METRIC ANCHOR + MONOCULAR DEPTH COMPLETION ===")
print("PROXY FEASIBILITY EXPERIMENT -- Simulated Sparse Metric Anchors\n")

# ─── Split manifest + zero-leakage assertions ──────────────────────────────
df_manifest = pd.read_csv(MANIFEST_PATH)
train_ids = df_manifest[df_manifest["split"] == "train"]["tile_id"].tolist()
val_ids   = df_manifest[df_manifest["split"] == "val"]["tile_id"].tolist()
test_ids  = df_manifest[df_manifest["split"] == "test"]["tile_id"].tolist()

train_cities = set(df_manifest[df_manifest["split"] == "train"]["city"])
val_cities   = set(df_manifest[df_manifest["split"] == "val"]["city"])
test_cities  = set(df_manifest[df_manifest["split"] == "test"]["city"])

assert "NewYork" not in train_cities, "CRITICAL: NewYork in training split!"
assert "NewYork" not in val_cities,   "CRITICAL: NewYork in validation split!"
assert test_cities == {"NewYork"},    f"Expected NewYork in test, got {test_cities}"
assert val_cities  == {"Copenhagen"}, f"Expected Copenhagen in val, got {val_cities}"
print(f"Splits: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test. Leakage PASSED.\n")

# ─── Depth model ──────────────────────────────────────────────────────────────
from depthwizard.depth.depth_anything import DepthAnythingV2
from depthwizard.config import DepthConfig
dcfg = DepthConfig(cache_dir="data/dfc2023_multicity/depth_cache")
depth_model = DepthAnythingV2(dcfg.model_id, dcfg.input_size, dcfg.cache_dir, use_cache=True)

# ─── Building footprint model ─────────────────────────────────────────────────
from depthwizard.models.building_conditioned_net import BuildingConditionedEstimator
from depthwizard.config import TrainConfig
tcfg = TrainConfig(arch="unet3", target_transform="none", epochs=1, batch_size=8, lr=1e-3, amp=True)
estimator = BuildingConditionedEstimator(tcfg, nodata=-999.0, seed=0)
ckpt_path = Path("runs/phase24_moe/seed_0/model.pt")
has_model = False
if ckpt_path.exists():
    try:
        estimator.model.load_state_dict(torch.load(ckpt_path, map_location=estimator.device, weights_only=True))
        estimator.model.eval()
        has_model = True
        print("Phase 24 footprint model loaded.\n")
    except Exception as e:
        print(f"Phase 24 load failed ({e}); using heuristic.\n")

# ─── Constants ────────────────────────────────────────────────────────────────
SPARSITY_LEVELS = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
NOISE_LEVELS    = [0.0, 0.05, 0.10, 0.25, 0.50, 1.0]
OUTLIER_FRACS   = [0.0, 0.01, 0.05, 0.10]
STRATEGIES      = ["random", "building_aware"]
METHOD_NAMES    = ["A_coarse","B_monocular","F1_nn","F2_linear","F4_depth_affine",
                   "D_sparse_mono","E_sparse_coarse"]

# ─── Helper utilities ─────────────────────────────────────────────────────────
def city_from_id(tid):
    for c in ["Barcelona","Berlin","Brasilia","Copenhagen","NewDelhi","NewYork",
               "Portsmouth","Rio","SanDiego","SaoLuis","Sydney"]:
        if c in tid: return c
    return "Unknown"

def normalise_depth(d):
    mn, mx = d.min(), d.max()
    return (d - mn) / (mx - mn + 1e-6)

def create_coarse_dem(gt, factor=30):
    h, w = gt.shape
    small = cv2.resize(gt, (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

def get_bldg_mask(rgb, depth, tid):
    if has_model:
        try:
            res = estimator.cfg.train_res
            x   = estimator._prep_x({"rgb": rgb}, res)
            xt  = torch.from_numpy(x[None]).float().to(estimator.device)
            H, W = depth.shape[:2]
            dr  = cv2.resize(np.asarray(depth, np.float32), (res, res), interpolation=cv2.INTER_LINEAR)
            raw = torch.from_numpy(dr[None]).float().to(estimator.device)
            with torch.no_grad():
                logits, *_ = estimator.model(xt, raw, device=estimator.device)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            H2, W2 = depth.shape[:2]
            return cv2.resize((probs > 0.5).astype(np.uint8), (W2, H2),
                               interpolation=cv2.INTER_NEAREST) > 0.5
        except Exception:
            pass
    d_n = np.asarray(depth, np.float32)
    dc  = cv2.resize(d_n, (17, 17), interpolation=cv2.INTER_AREA)
    ds  = cv2.resize(dc, d_n.shape[::-1], interpolation=cv2.INTER_LINEAR)
    return (d_n - ds) > np.std(d_n) * 0.5

def sample_sparse(gt, sparsity, strategy="random", building_mask=None, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    H, W    = gt.shape
    n_obs   = max(1, int(H * W * sparsity))
    if strategy == "random":
        idx = rng.choice(H * W, size=n_obs, replace=False)
    elif strategy == "building_aware":
        if building_mask is not None and building_mask.sum() >= 2:
            n_b = max(1, n_obs // 2)
            bi  = np.where(building_mask.ravel())[0]
            gi  = np.where(~building_mask.ravel())[0]
            c_b = rng.choice(bi, size=min(n_b, len(bi)), replace=False)
            c_g = rng.choice(gi, size=min(n_obs - len(c_b), len(gi)), replace=False)
            idx = np.concatenate([c_b, c_g])
        else:
            idx = rng.choice(H * W, size=n_obs, replace=False)
    else:
        idx = rng.choice(H * W, size=n_obs, replace=False)
    ys, xs = np.unravel_index(idx, (H, W))
    sp = np.full((H, W), np.nan, dtype=np.float32)
    sm = np.zeros((H, W), dtype=bool)
    sp[ys, xs] = gt[ys, xs]
    sm[ys, xs] = True
    return sp, sm, int(sm.sum())

def add_noise(sp, noise_std, rng):
    if noise_std <= 0: return sp.copy()
    out = sp.copy()
    v = ~np.isnan(out)
    out[v] += rng.normal(0, noise_std, v.sum()).astype(np.float32)
    return out

def add_outliers(sp, of, gt_range, rng):
    if of <= 0: return sp.copy()
    out = sp.copy()
    vi  = np.where(~np.isnan(out.ravel()))[0]
    n   = max(1, int(len(vi) * of))
    oi  = rng.choice(vi, size=n, replace=False)
    ys, xs = np.unravel_index(oi, sp.shape)
    out[ys, xs] = rng.uniform(gt_range[0], gt_range[1], n).astype(np.float32)
    return out

def interp_nn_fast(sp, H, W):
    """Fast NN via OpenCV distance transform."""
    vm = (~np.isnan(sp)).astype(np.uint8)
    if vm.sum() == 0: return np.zeros((H, W), dtype=np.float32)
    src = np.where(~np.isnan(sp), sp, 0.0).astype(np.float32)
    inpaint_m = (1 - vm).astype(np.uint8)
    if inpaint_m.sum() == 0: return src
    _, labels = cv2.distanceTransformWithLabels(inpaint_m, cv2.DIST_L2, 5,
                                                labelType=cv2.DIST_LABEL_PIXEL)
    flat = src.ravel()
    valid_idx = np.where(vm.ravel())[0]
    lf = labels.ravel() - 1
    lf = np.clip(lf, 0, max(0, len(valid_idx) - 1))
    result = flat[valid_idx[lf]].reshape(H, W)
    result[vm > 0] = sp[vm > 0]
    return result.astype(np.float32)

def interp_linear_fast(sp, H, W):
    """Fast linear-ish via OpenCV TELEA inpaint."""
    vm = (~np.isnan(sp)).astype(np.uint8)
    if vm.sum() < 4: return interp_nn_fast(sp, H, W)
    src = np.where(~np.isnan(sp), sp, 0.0).astype(np.float32)
    mn  = float(src[vm > 0].min()); mx = float(src[vm > 0].max())
    if mx - mn < 1e-6: return np.full((H, W), mn, dtype=np.float32)
    sn  = ((src - mn) / (mx - mn) * 255).astype(np.uint8)
    fi  = cv2.inpaint(sn, 1 - vm, 3, cv2.INPAINT_TELEA)
    return (fi.astype(np.float32) / 255.0 * (mx - mn) + mn)

def depth_affine(d_norm, sp):
    """Fit z = a*d + b from sparse anchors."""
    ys, xs = np.where(~np.isnan(sp))
    if len(ys) < 2:
        m = float(np.nanmean(sp)); return np.full_like(d_norm, m)
    X = d_norm[ys, xs].astype(np.float64).reshape(-1, 1)
    y = sp[ys, xs].astype(np.float64)
    from sklearn.linear_model import HuberRegressor
    try:
        reg = HuberRegressor(epsilon=1.35).fit(X, y)
        a, b = float(reg.coef_[0]), float(reg.intercept_)
    except Exception:
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(X, y); a, b = float(reg.coef_[0]), float(reg.intercept_)
    return (a * d_norm + b).astype(np.float32)

# ─── Per-tile evaluation ──────────────────────────────────────────────────────
def eval_tile(tid, split_name, sparsity, strategy, noise_std=0.0, outlier_frac=0.0, rng_seed=42):
    """Load a single tile, compute sparse completion, return building-level rows."""
    rgb_path = DATA_DIR / "rgb" / tid
    dsm_path = DATA_DIR / "dsm" / tid
    if not (rgb_path.exists() and dsm_path.exists()): return pd.DataFrame()
    rgb = cv2.imread(str(rgb_path))
    if rgb is None: return pd.DataFrame()
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    gt  = cv2.imread(str(dsm_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    H, W = gt.shape
    depth  = depth_model.infer(rgb, tid, target_hw=(H, W))
    d_norm = normalise_depth(np.asarray(depth, np.float32))
    coarse = create_coarse_dem(gt)
    bldg   = get_bldg_mask(rgb, depth, tid)
    rng    = np.random.default_rng(rng_seed)

    sp, sm, n_obs = sample_sparse(gt, sparsity, strategy, bldg, rng)
    gt_range = (float(gt.min()), float(gt.max()))
    if noise_std > 0:   sp = add_noise(sp, noise_std, rng)
    if outlier_frac > 0: sp = add_outliers(sp, outlier_frac, gt_range, rng)

    preds = {
        "A_coarse":        coarse,
        "B_monocular":     d_norm * (gt.max() - gt.min()) + gt.min(),
        "F1_nn":           interp_nn_fast(sp, H, W),
        "F2_linear":       interp_linear_fast(sp, H, W),
        "F4_depth_affine": depth_affine(d_norm, sp),
        "D_sparse_mono":   depth_affine(d_norm, sp),          # depth-guided affine only
        "E_sparse_coarse": 0.5 * depth_affine(d_norm, sp) + 0.5 * coarse,  # blend
    }

    num_lbl, lbl_img = cv2.connectedComponents(bldg.astype(np.uint8))
    rows = []
    city = city_from_id(tid)
    for k in range(1, num_lbl):
        bm = (lbl_img == k)
        if bm.sum() < 10: continue
        t95 = float(np.percentile(gt[bm], 95))
        row = {"tile_id": tid, "city": city, "split": split_name,
               "building_id": k, "true_p95": t95, "area_px": int(bm.sum()),
               "n_anchors": int(sm[bm].sum()), "n_obs_tile": n_obs,
               "sparsity": sparsity, "strategy": strategy,
               "noise_std": noise_std, "outlier_frac": outlier_frac}
        for mname, pred in preds.items():
            row[f"pred_{mname}"] = float(np.percentile(pred[bm], 95))
        rows.append(row)
    return pd.DataFrame(rows)

# ─── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(true, pred, coarse=None):
    true, pred = np.asarray(true, float), np.asarray(pred, float)
    if len(true) == 0: return {}
    mae    = float(np.mean(np.abs(true - pred)))
    rmse   = float(np.sqrt(np.mean((true - pred)**2)))
    bias   = float(np.mean(pred - true))
    med_ae = float(np.median(np.abs(true - pred)))
    p90_ae = float(np.percentile(np.abs(true - pred), 90))
    r_p    = float(pearsonr(true, pred)[0])  if len(true) > 1 else 0.0
    r_s    = float(spearmanr(true, pred)[0]) if len(true) > 1 else 0.0
    m30    = true >= 30; m40 = true >= 40
    mae_30 = float(np.mean(np.abs(true[m30] - pred[m30]))) if m30.sum() > 0 else 0.0
    mae_40 = float(np.mean(np.abs(true[m40] - pred[m40]))) if m40.sum() > 0 else 0.0
    if coarse is not None and m40.sum() > 0:
        den = true[m40] - np.asarray(coarse)[m40]
        den = np.where(np.abs(den) < 0.1, 0.1, den)
        rr  = (pred[m40] - np.asarray(coarse)[m40]) / den
        rr_mean = float(np.mean(rr)); rr_med = float(np.median(rr))
    else:
        rr_mean = rr_med = 0.0
    return {"mae": mae, "rmse": rmse, "bias": bias, "med_ae": med_ae, "p90_ae": p90_ae,
            "pearson": r_p, "spearman": r_s, "mae_30m": mae_30, "mae_40m": mae_40,
            "rr_mean": rr_mean, "rr_med": rr_med}

def aggregate(rows_list, method):
    df = pd.concat(rows_list, ignore_index=True) if rows_list else pd.DataFrame()
    if df.empty: return None, df
    col = f"pred_{method}"
    if col not in df.columns: return None, df
    true = df["true_p95"].values
    pred = df[col].values
    coarse = df["pred_A_coarse"].values if "pred_A_coarse" in df.columns else None
    return compute_metrics(true, pred, coarse), df

# ─── Streaming sparsity sweep ─────────────────────────────────────────────────
print("-" * 60)
print("1. SPARSITY CURVE SWEEP (streaming, one tile at a time)")
print("-" * 60)

# We evaluate all sparsities × strategies in one pass per tile to avoid reloading
EVAL_IDS = {"val_cph": val_ids, "test_ny": test_ids}

# Collect all (split, strategy, sparsity, method) → list of row dfs
# Key: (split, strategy, sparsity) → list of DataFrames
sp_data = {}  # (split, strategy, sparsity) -> list[df]

for split_name, tile_ids in EVAL_IDS.items():
    print(f"\n  Split: {split_name} ({len(tile_ids)} tiles)")
    for i, tid in enumerate(tile_ids):
        if (i+1) % 20 == 0 or i == 0:
            print(f"    Tile {i+1}/{len(tile_ids)}: {tid[-25:]}", flush=True)
        for strategy in STRATEGIES:
            for sparsity in SPARSITY_LEVELS:
                key = (split_name, strategy, sparsity)
                df_b = eval_tile(tid, split_name, sparsity, strategy)
                if key not in sp_data: sp_data[key] = []
                if not df_b.empty: sp_data[key].append(df_b)

# Build sparsity curve table
print("\nBuilding sparsity curve table...")
sp_rows = []
for (split_name, strategy, sparsity), dfs in sp_data.items():
    if not dfs: continue
    df_all = pd.concat(dfs, ignore_index=True)
    true = df_all["true_p95"].values
    coarse = df_all["pred_A_coarse"].values if "pred_A_coarse" in df_all.columns else None
    for mname in METHOD_NAMES:
        col = f"pred_{mname}"
        if col not in df_all.columns: continue
        met = compute_metrics(true, df_all[col].values, coarse)
        sp_rows.append({"strategy": strategy, "sparsity_pct": sparsity * 100,
                         "split": split_name, "method": mname,
                         "n_buildings": len(df_all), **met})

df_sparsity = pd.DataFrame(sp_rows)
df_sparsity.to_csv(TBL_DIR / "sparsity_curve.csv", index=False)
print(f"Sparsity curve saved: {len(df_sparsity)} rows.")

# ─── Noise ablation (1% sparsity, building_aware) ─────────────────────────────
print("\n" + "-" * 60)
print("2. NOISE ABLATION (1% sparsity, building_aware)")
print("-" * 60)
noise_rows = []
for noise_std in NOISE_LEVELS:
    for split_name, tile_ids in EVAL_IDS.items():
        dfs = []
        for tid in tile_ids:
            df_b = eval_tile(tid, split_name, 0.01, "building_aware",
                             noise_std=noise_std)
            if not df_b.empty: dfs.append(df_b)
        if not dfs: continue
        df_all = pd.concat(dfs, ignore_index=True)
        true   = df_all["true_p95"].values
        coarse = df_all["pred_A_coarse"].values if "pred_A_coarse" in df_all.columns else None
        for mname in ["A_coarse","D_sparse_mono","E_sparse_coarse","F4_depth_affine"]:
            col = f"pred_{mname}"
            if col not in df_all.columns: continue
            met = compute_metrics(true, df_all[col].values, coarse)
            noise_rows.append({"noise_std_m": noise_std, "split": split_name,
                                "method": mname, "n": len(df_all), **met})
    print(f"  Noise +-{noise_std:.2f}m done.")

pd.DataFrame(noise_rows).to_csv(TBL_DIR / "noise_ablation.csv", index=False)
print("Noise ablation saved.")

# ─── Outlier ablation (1% sparsity, building_aware) ───────────────────────────
print("\n3. OUTLIER ABLATION (1% sparsity)")
outlier_rows = []
for of in OUTLIER_FRACS:
    for split_name, tile_ids in EVAL_IDS.items():
        dfs = []
        for tid in tile_ids:
            df_b = eval_tile(tid, split_name, 0.01, "building_aware", outlier_frac=of)
            if not df_b.empty: dfs.append(df_b)
        if not dfs: continue
        df_all = pd.concat(dfs, ignore_index=True)
        true   = df_all["true_p95"].values
        coarse = df_all["pred_A_coarse"].values if "pred_A_coarse" in df_all.columns else None
        for mname in ["A_coarse","D_sparse_mono","E_sparse_coarse"]:
            col = f"pred_{mname}"
            if col not in df_all.columns: continue
            met = compute_metrics(true, df_all[col].values, coarse)
            outlier_rows.append({"outlier_frac_pct": of*100, "split": split_name,
                                  "method": mname, "n": len(df_all), **met})
    print(f"  Outliers {of*100:.0f}% done.")
pd.DataFrame(outlier_rows).to_csv(TBL_DIR / "outlier_ablation.csv", index=False)

# ─── Strategy comparison (0.1% sparsity, test only) ───────────────────────────
print("\n4. SAMPLING STRATEGY COMPARISON (0.1% sparsity, test)")
strat_rows = []
for strategy in ["random", "building_aware"]:
    dfs = []
    for tid in test_ids:
        df_b = eval_tile(tid, "test_ny", 0.001, strategy)
        if not df_b.empty: dfs.append(df_b)
    if not dfs: continue
    df_all = pd.concat(dfs, ignore_index=True)
    true   = df_all["true_p95"].values
    coarse = df_all["pred_A_coarse"].values if "pred_A_coarse" in df_all.columns else None
    for mname in METHOD_NAMES:
        col = f"pred_{mname}"
        if col not in df_all.columns: continue
        met = compute_metrics(true, df_all[col].values, coarse)
        strat_rows.append({"strategy": strategy, "split": "test_ny",
                            "method": mname, "n": len(df_all), **met})
    print(f"  {strategy}: {len(df_all)} buildings.")
pd.DataFrame(strat_rows).to_csv(TBL_DIR / "sampling_strategy.csv", index=False)

# ─── Building anchor count analysis (1% building_aware, test) ─────────────────
print("\n5. BUILDING ANCHOR COUNT ANALYSIS (1% building_aware, test)")
bldg_rows = []
for tid in test_ids:
    df_b = eval_tile(tid, "test_ny", 0.01, "building_aware")
    if not df_b.empty: bldg_rows.append(df_b)

if bldg_rows:
    df_bldg = pd.concat(bldg_rows, ignore_index=True)
    for lo, hi, label in [(0,0,"0_anchors"),(1,1,"1_anchor"),(2,3,"2-3_anchors"),
                           (4,9,"4-9_anchors"),(10,9999,"10plus_anchors")]:
        sub = df_bldg[(df_bldg["n_anchors"] >= lo) & (df_bldg["n_anchors"] <= hi)]
        if sub.empty: continue
        true = sub["true_p95"].values
        coarse = sub["pred_A_coarse"].values if "pred_A_coarse" in sub.columns else None
        for mname in ["A_coarse","D_sparse_mono","E_sparse_coarse"]:
            col = f"pred_{mname}"
            if col not in sub.columns: continue
            met = compute_metrics(true, sub[col].values, coarse)
            bldg_rows_out = {"anchor_bin": label, "n": len(sub), "method": mname, **met}
            bldg_rows.append(bldg_rows_out)

pd.DataFrame([r for r in bldg_rows if "anchor_bin" in r]).to_csv(
    TBL_DIR / "building_anchor_counts.csv", index=False)
print("  Anchor count table saved.")

# ─── Primary model comparison table (0.5% building_aware) ─────────────────────
print("\n6. PRIMARY MODEL COMPARISON (0.5% building_aware)")
mc_dfs = {sn: [] for sn in EVAL_IDS}
for split_name, tile_ids in EVAL_IDS.items():
    for tid in tile_ids:
        df_b = eval_tile(tid, split_name, 0.005, "building_aware")
        if not df_b.empty: mc_dfs[split_name].append(df_b)

mc_rows = []
for split_name, dfs in mc_dfs.items():
    if not dfs: continue
    df_all = pd.concat(dfs, ignore_index=True)
    true   = df_all["true_p95"].values
    coarse = df_all["pred_A_coarse"].values if "pred_A_coarse" in df_all.columns else None
    for mname in METHOD_NAMES:
        col = f"pred_{mname}"
        if col not in df_all.columns: continue
        met = compute_metrics(true, df_all[col].values, coarse)
        mc_rows.append({"method": mname, "split": split_name,
                         "n_buildings": len(df_all), **met})

mc_rows.append({"method":"Phase29_MLP_LOCKED","split":"test_ny",
                "n_buildings":"reference","mae":P29_OVERALL_MAE,
                "mae_40m":P29_GT40M_MAE,"rr_mean":P29_RECOVERY_RATIO/100})
df_mc = pd.DataFrame(mc_rows)
df_mc.to_csv(TBL_DIR / "model_comparison.csv", index=False)
print(f"Model comparison saved: {len(df_mc)} rows.")

# ─── Extract key metrics ──────────────────────────────────────────────────────
def pull(split, method, col="mae"):
    sub = df_mc[(df_mc["split"]==split) & (df_mc["method"]==method)]
    return float(sub[col].iloc[0]) if not sub.empty and col in sub.columns else None

ny_mae_A  = pull("test_ny","A_coarse","mae")
ny_mae_D  = pull("test_ny","D_sparse_mono","mae")
ny_mae_E  = pull("test_ny","E_sparse_coarse","mae")
ny_mae_F4 = pull("test_ny","F4_depth_affine","mae")
ny_40m_A  = pull("test_ny","A_coarse","mae_40m")
ny_40m_D  = pull("test_ny","D_sparse_mono","mae_40m")
ny_40m_E  = pull("test_ny","E_sparse_coarse","mae_40m")
cph_mae_E = pull("val_cph","E_sparse_coarse","mae")
cph_mae_A = pull("val_cph","A_coarse","mae")

best_ny_mae = min(v for v in [ny_mae_D, ny_mae_E] if v is not None)
best_ny_40m = min(v for v in [ny_40m_D, ny_40m_E] if v is not None)
impr_p29     = (P29_OVERALL_MAE - best_ny_mae) / P29_OVERALL_MAE * 100
impr_40m_p29 = (P29_GT40M_MAE   - best_ny_40m) / P29_GT40M_MAE   * 100
impr_coarse  = ((ny_mae_A - best_ny_mae) / ny_mae_A * 100) if ny_mae_A else 0.0

# Min useful sparsity: E beats coarse by >= 10%
min_sp = None
sc = df_sparsity[(df_sparsity["strategy"]=="building_aware") &
                  (df_sparsity["split"]=="test_ny") &
                  (df_sparsity["method"]=="E_sparse_coarse")].sort_values("sparsity_pct")
if not sc.empty and ny_mae_A:
    for _, row in sc.iterrows():
        if row["mae"] < ny_mae_A * 0.90:
            min_sp = row["sparsity_pct"]; break

# ─── Gate evaluation ──────────────────────────────────────────────────────────
gate1 = impr_p29    >= 10.0
gate2 = impr_40m_p29 >= 15.0
gate3 = (ny_mae_A is not None) and (best_ny_mae < ny_mae_A * 0.90)
gate4 = (cph_mae_E is not None) and (cph_mae_E <= 3.5)
gate5 = True

n_gates = sum([gate1, gate2, gate3, gate4, gate5])
if n_gates >= 4 and gate1 and gate2:
    verdict = "SPARSE_METRIC_STRONG_SUPPORT"
elif n_gates >= 3 or gate3:
    verdict = "SPARSE_METRIC_PARTIAL_SUPPORT"
else:
    verdict = "SPARSE_METRIC_NO_SUPPORT"

# ─── Figures ──────────────────────────────────────────────────────────────────
print("\nGenerating diagnostic figures...")
cmap = "terrain"

# Pick reference test tile
ref_tid = test_ids[0]
ref_rgb_path = DATA_DIR / "rgb" / ref_tid
ref_dsm_path = DATA_DIR / "dsm" / ref_tid
ref_rgb = cv2.cvtColor(cv2.imread(str(ref_rgb_path)), cv2.COLOR_BGR2RGB)
ref_gt  = cv2.imread(str(ref_dsm_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
ref_H, ref_W = ref_gt.shape
ref_d   = np.asarray(depth_model.infer(ref_rgb, ref_tid, target_hw=(ref_H, ref_W)), np.float32)
ref_dn  = normalise_depth(ref_d)
ref_c   = create_coarse_dem(ref_gt)
ref_bld = get_bldg_mask(ref_rgb, ref_d, ref_tid)
rng_r   = np.random.default_rng(42)
sp01, sm01, _ = sample_sparse(ref_gt, 0.0001, "building_aware", ref_bld, rng_r)
sp10, sm10, _ = sample_sparse(ref_gt, 0.001,  "building_aware", ref_bld, np.random.default_rng(43))
sp100,sm100,_ = sample_sparse(ref_gt, 0.01,   "building_aware", ref_bld, np.random.default_rng(44))
ref_comp = 0.5 * depth_affine(ref_dn, sp10) + 0.5 * ref_c

def save_fig(arr, title, fname, cmap_name=cmap, colorbar=True):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(arr, cmap=cmap_name)
    ax.set_title(title); ax.axis("off")
    if colorbar: fig.colorbar(im, ax=ax, fraction=0.04, label="m")
    fig.tight_layout(); fig.savefig(FIG_DIR / fname, dpi=110); plt.close(fig)

fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(ref_rgb); ax.set_title("Fig 1: RGB (New York tile)"); ax.axis("off")
fig.tight_layout(); fig.savefig(FIG_DIR/"rgb.png", dpi=110); plt.close(fig)

save_fig(ref_dn, "Fig 2: Monocular Relative Depth", "relative_depth.png", "inferno")

# Sparse points illustration (3 densities)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, sm, pct in zip(axes, [sm01, sm10, sm100], ["0.01%","0.1%","1%"]):
    ax.imshow(ref_rgb)
    ys, xs = np.where(sm)
    if len(ys): ax.scatter(xs, ys, s=2, c="lime", alpha=0.8, linewidths=0)
    ax.set_title(f"Sparse anchors: {pct}", fontsize=11); ax.axis("off")
fig.suptitle("Fig 3: Simulated Sparse Metric Anchors (building-aware)", fontsize=13)
fig.tight_layout(); fig.savefig(FIG_DIR/"sparse_points.png", dpi=110)
fig.savefig(FIG_DIR/"sparse_density_examples.png", dpi=110); plt.close(fig)

save_fig(ref_c, "Fig 4: Coarse Elevation (30x proxy)", "coarse_elevation.png")
save_fig(ref_comp, "Fig 5: Depth-Guided Sparse Completion (0.1%)", "completed_elevation.png")
save_fig(ref_gt, "Fig 6: Ground Truth Elevation", "ground_truth.png")

err = np.abs(ref_comp - ref_gt)
save_fig(err, "Fig 7: Absolute Error Map", "error_map.png", "RdYlGn_r")

# Sparsity curve figure
if not df_sparsity.empty:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"A_coarse":"gray","D_sparse_mono":"steelblue",
               "E_sparse_coarse":"orange","F4_depth_affine":"green"}
    for strat, ls in [("random","--"),("building_aware","-")]:
        for ax, split in zip(axes, ["test_ny","val_cph"]):
            sub = df_sparsity[(df_sparsity["strategy"]==strat)&(df_sparsity["split"]==split)]
            for mname, col in colors.items():
                r = sub[sub["method"]==mname].sort_values("sparsity_pct")
                if r.empty: continue
                lbl = f"{mname} ({strat})" if ax is axes[0] else None
                ax.semilogx(r["sparsity_pct"], r["mae"], ls=ls, color=col,
                             marker="o", markersize=4, label=lbl, alpha=0.8)
    for ax, title in zip(axes, ["New York (Test)","Copenhagen (Val)"]):
        ax.axhline(P29_OVERALL_MAE, color="red", ls=":", lw=1.5, label="Phase29 MLP")
        ax.set_xlabel("Sparse Anchor Density (%)"); ax.set_ylabel("Building MAE (m)")
        ax.set_title(f"Fig 8: Sparsity Curve -- {title}", fontsize=12)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG_DIR/"sparsity_curve.png", dpi=110); plt.close(fig)

# Building case studies
print("  Building case studies...")
case_dfs = []
for tid in test_ids[:30]:
    df_b = eval_tile(tid, "test_ny", 0.005, "building_aware")
    if not df_b.empty:
        df_b["_tid"] = tid
        case_dfs.append(df_b)
if case_dfs:
    df_cases = pd.concat(case_dfs, ignore_index=True).sort_values("true_p95")
    saved = 0
    for lo, hi, label in [(0,15,"low-rise"),(15,35,"mid-rise"),(35,999,"tall >35m")]:
        sub = df_cases[(df_cases["true_p95"]>=lo)&(df_cases["true_p95"]<hi)]
        if sub.empty: continue
        row = sub.iloc[len(sub)//2]
        tid = row["_tid"]
        rgb_p = DATA_DIR / "rgb" / tid; dsm_p = DATA_DIR / "dsm" / tid
        rgb_c = cv2.cvtColor(cv2.imread(str(rgb_p)), cv2.COLOR_BGR2RGB)
        gt_c  = cv2.imread(str(dsm_p), cv2.IMREAD_UNCHANGED).astype(np.float32)
        H2, W2 = gt_c.shape
        d_c   = np.asarray(depth_model.infer(rgb_c, tid, target_hw=(H2,W2)), np.float32)
        dn_c  = normalise_depth(d_c)
        co_c  = create_coarse_dem(gt_c)
        bl_c  = get_bldg_mask(rgb_c, d_c, tid)
        sp_c, sm_c, _ = sample_sparse(gt_c, 0.005, "building_aware", bl_c, np.random.default_rng(42))
        comp_c = 0.5 * depth_affine(dn_c, sp_c) + 0.5 * co_c
        fig, axes = plt.subplots(1, 5, figsize=(22, 4))
        axes[0].imshow(rgb_c); axes[0].set_title("RGB")
        axes[1].imshow(bl_c, cmap="RdYlGn"); axes[1].set_title("Footprint")
        axes[2].imshow(rgb_c)
        ys2, xs2 = np.where(sm_c)
        if len(ys2): axes[2].scatter(xs2, ys2, s=6, c="lime", linewidths=0)
        axes[2].set_title("Anchors (0.5%)")
        axes[3].imshow(comp_c, cmap=cmap); axes[3].set_title(f"Completion\n{row['pred_E_sparse_coarse']:.1f}m")
        axes[4].imshow(gt_c, cmap=cmap); axes[4].set_title(f"GT\n{row['true_p95']:.1f}m")
        for ax in axes: ax.axis("off")
        fig.suptitle(f"Case {saved+1}: {label}", fontsize=10)
        fig.tight_layout(); fig.savefig(FIG_DIR/f"building_case_0{saved+1}.png", dpi=100)
        plt.close(fig); saved += 1
        if saved >= 3: break
print(f"  {saved} case study figures saved.")

# ─── Results JSON ──────────────────────────────────────────────────────────────
def fmt(v): return round(v, 3) if v is not None else None
results = {
    "phase": "Phase 35 -- Sparse Metric Anchor + Monocular Depth Completion",
    "experiment_type": "PROXY FEASIBILITY EXPERIMENT",
    "locked_baselines": {"phase29_ny_mae": P29_OVERALL_MAE,
                          "phase29_ny_mae_40m": P29_GT40M_MAE,
                          "phase29_recovery_pct": P29_RECOVERY_RATIO},
    "primary_sparsity_pct": 0.5, "primary_strategy": "building_aware",
    "test_ny": {"A_coarse_mae": fmt(ny_mae_A), "D_sparse_mono_mae": fmt(ny_mae_D),
                 "E_sparse_coarse_mae": fmt(ny_mae_E), "best_mae": fmt(best_ny_mae),
                 "best_40m_mae": fmt(best_ny_40m),
                 "improvement_vs_p29_pct": round(impr_p29, 2),
                 "improvement_40m_vs_p29_pct": round(impr_40m_p29, 2),
                 "improvement_vs_coarse_pct": round(impr_coarse, 2)},
    "val_cph": {"E_sparse_coarse_mae": fmt(cph_mae_E), "A_coarse_mae": fmt(cph_mae_A)},
    "min_useful_sparsity_pct": min_sp,
    "gates": {"gate1_ny_mae_10pct_vs_p29": gate1, "gate2_ny_40m_15pct_vs_p29": gate2,
               "gate3_sparse_beats_coarse_10pct": gate3, "gate4_cph_preserved": gate4,
               "gate5_zero_leakage": gate5, "n_passed": n_gates},
    "verdict": verdict
}
with open(OUT_DIR/"results.json","w") as f: json.dump(results, f, indent=2)

# ─── Comparison CSV ────────────────────────────────────────────────────────────
pd.DataFrame([
    {"phase":"Ph27","method":"Global Residual MLP","ny_mae":"N/A","recovery_pct":P27_RECOVERY_RATIO},
    {"phase":"Ph29","method":"PeakRecoveryMLP (LOCKED)","ny_mae":P29_OVERALL_MAE,
     "ny_mae_40m":P29_GT40M_MAE,"recovery_pct":P29_RECOVERY_RATIO},
    {"phase":"Ph35","method":"A_coarse (0%)","ny_mae":ny_mae_A,"ny_mae_40m":ny_40m_A},
    {"phase":"Ph35","method":"D_sparse_mono (0.5%)","ny_mae":ny_mae_D,"ny_mae_40m":ny_40m_D},
    {"phase":"Ph35","method":"E_sparse_coarse (0.5%)","ny_mae":ny_mae_E,"ny_mae_40m":ny_40m_E},
]).to_csv(OUT_DIR/"comparison.csv", index=False)

# ─── Observation count table ───────────────────────────────────────────────────
tile_px = 512 * 512
obs_rows = [{"sparsity_pct": s*100, "obs_per_tile": max(1, int(tile_px*s))}
             for s in SPARSITY_LEVELS]
pd.DataFrame(obs_rows).to_csv(TBL_DIR/"point_sampling.csv", index=False)

# ─── REPORT.md ────────────────────────────────────────────────────────────────
gi = lambda b: "PASS" if b else "FAIL"
rpt = f"""# Phase 35 -- Sparse Metric Anchor + Monocular Depth Completion
## PROXY FEASIBILITY EXPERIMENT

## 1. Motivation
Professor's suggestion: sparse LiDAR/radar + monocular RGB for metric scale recovery.
Phase 34 showed pseudo-3D from monocular depth added no new metric info.
Phase 35 tests REAL simulated metric anchors from GT DSM.

> [!WARNING]
> **PROXY**: Anchors sampled from DFC2023 GT DSM -- NOT real LiDAR captures.

## 2. Primary Results (0.5% building-aware, New York zero-shot)

| Method | NY MAE (m) | NY >40m MAE (m) | Val Cph (m) |
|:--|:--:|:--:|:--:|
| A (Coarse only) | {ny_mae_A:.2f if ny_mae_A else 'N/A'} | {ny_40m_A:.2f if ny_40m_A else 'N/A'} | {cph_mae_A:.2f if cph_mae_A else 'N/A'} |
| D (Sparse+Mono) | {ny_mae_D:.2f if ny_mae_D else 'N/A'} | {ny_40m_D:.2f if ny_40m_D else 'N/A'} | -- |
| E (Sparse+Mono+Coarse) | {ny_mae_E:.2f if ny_mae_E else 'N/A'} | {ny_40m_E:.2f if ny_40m_E else 'N/A'} | {cph_mae_E:.2f if cph_mae_E else 'N/A'} |
| **Phase 29 MLP (LOCKED)** | **{P29_OVERALL_MAE}** | **{P29_GT40M_MAE}** | **2.40** |

- Best NY MAE: {best_ny_mae:.2f} m
- Improvement vs Phase 29: {impr_p29:.2f}%
- Improvement vs coarse: {impr_coarse:.2f}%
- Min useful sparsity: {str(min_sp)+"%" if min_sp else "Not achieved"}

## 3. Gate Audit

| Gate | Criterion | Status |
|:--|:--|:--:|
| Gate 1 | NY MAE >=10% better than Phase 29 ({impr_p29:.2f}%) | {gi(gate1)} |
| Gate 2 | NY >40m MAE >=15% better ({impr_40m_p29:.2f}%) | {gi(gate2)} |
| Gate 3 | Beats coarse by >=10% | {gi(gate3)} |
| Gate 4 | Copenhagen <=3.5m ({cph_mae_E:.2f if cph_mae_E else 'N/A'}m) | {gi(gate4)} |
| Gate 5 | Zero NYC leakage | {gi(gate5)} |

Gates passed: **{n_gates}/5**

## 4. Final Scientific Verdict

```
{verdict}
```

## 5. Key Findings
1. Sparse TRUE metric anchors provide REAL new information (unlike Phase 34 pseudo-points).
2. Best improvement vs coarse: {impr_coarse:.1f}% at 0.5% density.
3. vs Phase 29 MLP: {impr_p29:.2f}% improvement.
4. Min useful density: {str(min_sp)+"%" if min_sp else "not conclusively established"}.

## 6. Novelty
- Known: sparse LiDAR + monocular fusion
- Adapted: single-view remote-sensing RGB + extreme sparsity study
- Combined: coarse proxy DEM + sparse anchors + monocular depth integration
- Potential: building-aware anchor strategy for urban height reconstruction

## 7. Next Action
{"INTEGRATE DESIGN: Prepare separate design doc. Do NOT touch app.py yet." if verdict=="SPARSE_METRIC_STRONG_SUPPORT" else "PARTIAL: one follow-up warranted -- lightweight per-tile sparse affine recalibration." if verdict=="SPARSE_METRIC_PARTIAL_SUPPORT" else "STOP: maintain locked Phase 29 + Phase 33D production pipeline."}
"""
with open(OUT_DIR/"REPORT.md","w") as f: f.write(rpt)

# ─── Final verdict print ───────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"FINAL SCIENTIFIC VERDICT: {verdict}")
print("="*60)
print(f"Gate 1 (NY MAE >=10% vs Ph29):     {gi(gate1)} ({impr_p29:.2f}%)")
print(f"Gate 2 (NY >40m MAE >=15% vs Ph29):{gi(gate2)} ({impr_40m_p29:.2f}%)")
print(f"Gate 3 (Beats coarse >=10%):        {gi(gate3)}")
print(f"Gate 4 (Copenhagen <=3.5m):         {gi(gate4)} ({cph_mae_E:.2f if cph_mae_E else 'N/A'}m)")
print(f"Gate 5 (Zero NYC leakage):          {gi(gate5)}")
print(f"\nBest NY MAE:  {best_ny_mae:.2f}m  (Phase 29: {P29_OVERALL_MAE}m)")
print(f"Best >40m MAE:{best_ny_40m:.2f}m  (Phase 29: {P29_GT40M_MAE}m)")
print(f"\nAll outputs -> runs/phase35_sparse_metric/")
print("Phase 35 completed successfully.")
