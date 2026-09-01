"""
Phase 41 — Forensic Pipeline Probe
====================================
Traces the EXACT data that flows through every pipeline stage and
dumps all 18 diagnostic images to runs/phase41_building_trace/.

DO NOT modify scientific rasters. Read-only probe only.
"""
import sys
import os
import csv
import json
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, ".")
from depthwizard.config import DepthConfig
from depthwizard.depth.depth_anything import DepthAnythingV2
from depthwizard.data.raster_loader import load_raster_input
from depthwizard.calibration import CalibrationEngine, CalibrationMode

OUT = Path("runs/phase41_building_trace")
OUT.mkdir(parents=True, exist_ok=True)

SCENE = "SV_NewYork_40.7401_-73.9915.tif"
RGB_PATH  = Path("data/dfc2023_multicity/rgb") / SCENE
DSM_PATH  = Path("data/dfc2023_multicity/dsm") / SCENE

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def save_colormap(arr: np.ndarray, path: str, cmap="inferno", title="", pct_clip=(1,99)):
    """Save a 2D float array as a colorized image with colorbar."""
    vmin, vmax = np.percentile(arr, pct_clip[0]), np.percentile(arr, pct_clip[1])
    fig, ax = plt.subplots(figsize=(8,8), facecolor="#0D1117")
    ax.set_facecolor("#0D1117")
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, color="white", fontsize=12)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)

def rgb_overlay(rgb_bgr, mask, color=(0,255,0), alpha=0.45):
    overlay = np.zeros_like(rgb_bgr)
    overlay[mask > 0] = color
    return cv2.addWeighted(rgb_bgr, 1-alpha, overlay, alpha, 0)

# ─── STAGE 0: Load raster ─────────────────────────────────────────────────────
print("="*70)
print("PHASE 41 — FORENSIC PIPELINE PROBE")
print("="*70)

print(f"\n[STAGE 0] Loading raster: {RGB_PATH}")
raster_in = load_raster_input(RGB_PATH, filename=SCENE)
h, w = raster_in.shape
rgb = raster_in.rgb
print(f"  RGB shape:        {rgb.shape}  dtype={rgb.dtype}")
print(f"  is_georeferenced: {raster_in.is_georeferenced}")
print(f"  GSD:              {raster_in.gsd}")

# 01 — RGB
img_01 = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite(str(OUT / "01_RGB.png"), img_01)
print(f"  [SAVED] 01_RGB.png  ({w}x{h})")

# ─── STAGE 1: Depth inference ─────────────────────────────────────────────────
print(f"\n[STAGE 1] Running Depth Anything V2 …")
dcfg = DepthConfig(cache_dir="data/dfc2023_multicity/depth_cache")
depth_model = DepthAnythingV2(dcfg.model_id, dcfg.input_size, dcfg.cache_dir, use_cache=True)
depth_raw = depth_model.infer(rgb, SCENE, target_hw=(h, w))
print(f"  depth_raw shape: {depth_raw.shape}  dtype={depth_raw.dtype}")
print(f"  depth_raw range: [{depth_raw.min():.3f}, {depth_raw.max():.3f}]")
print(f"  depth_raw P5={np.percentile(depth_raw,5):.3f} P50={np.percentile(depth_raw,50):.3f} P95={np.percentile(depth_raw,95):.3f}")

save_colormap(depth_raw, str(OUT / "02_DEPTH.png"), cmap="magma",
              title=f"02 Depth Anything V2 Raw  [{depth_raw.min():.2f}–{depth_raw.max():.2f}]")
print(f"  [SAVED] 02_DEPTH.png")

# ─── STAGE 2: Reference DSM (ground truth) ────────────────────────────────────
ref_elevation = None
if DSM_PATH.exists():
    ref_elevation = cv2.imread(str(DSM_PATH), cv2.IMREAD_UNCHANGED).astype(np.float32)
    print(f"\n[STAGE 2] DSM ground truth: {ref_elevation.shape}  min={ref_elevation.min():.1f}  max={ref_elevation.max():.1f}")
    save_colormap(ref_elevation, str(OUT / "05_DSM.png"), cmap="inferno",
                  title=f"05 Ground-Truth DSM  [{ref_elevation.min():.1f}–{ref_elevation.max():.1f} m]")
    print(f"  [SAVED] 05_DSM.png")
else:
    print(f"\n[STAGE 2] *** DSM ground truth NOT FOUND at {DSM_PATH} ***")
    print(f"           Calibration will fall back to MONOCULAR_RELATIVE mode!")

# ─── STAGE 3: Calibration ─────────────────────────────────────────────────────
print(f"\n[STAGE 3] Running CalibrationEngine …")
calib_engine = CalibrationEngine(runs_dir=Path("runs"))
calib_res = calib_engine.calibrate(
    depth_raw, rgb,
    is_georeferenced=raster_in.is_georeferenced,
    mode=CalibrationMode.AUTO,
    reference_elevation=ref_elevation,
    filename=SCENE
)
print(f"  Mode used:    {calib_res.mode_used}")
print(f"  is_metric:    {calib_res.is_metric}")
print(f"  units:        {calib_res.units}")
print(f"  DSM: {calib_res.dsm.shape}  min={calib_res.dsm.min():.2f}  max={calib_res.dsm.max():.2f}")
print(f"  DTM: {calib_res.dtm.shape}  min={calib_res.dtm.min():.2f}  max={calib_res.dtm.max():.2f}")
print(f"  nDSM: {calib_res.ndsm.shape}  min={calib_res.ndsm.min():.2f}  max={calib_res.ndsm.max():.2f}")
print(f"  nDSM percentiles: P25={np.percentile(calib_res.ndsm,25):.2f}  P50={np.percentile(calib_res.ndsm,50):.2f}  P75={np.percentile(calib_res.ndsm,75):.2f}  P95={np.percentile(calib_res.ndsm,95):.2f}  P99={np.percentile(calib_res.ndsm,99):.2f}")
print(f"  nDSM >= 1.8m: {100.0*(calib_res.ndsm>=1.8).mean():.1f}% of pixels")
print(f"  mask_bldg: dtype={calib_res.mask_bldg.dtype}  sum={calib_res.mask_bldg.sum()}  pct={100.0*calib_res.mask_bldg.mean():.1f}%")

dsm  = calib_res.dsm
dtm  = calib_res.dtm
ndsm = calib_res.ndsm
mask_bldg = calib_res.mask_bldg.astype(np.uint8)

# Check spatial alignment
print(f"\n  *** ALIGNMENT CHECK ***")
print(f"  RGB.shape  = {rgb.shape[:2]}  => (H={rgb.shape[0]}, W={rgb.shape[1]})")
print(f"  DSM.shape  = {dsm.shape}")
print(f"  DTM.shape  = {dtm.shape}")
print(f"  nDSM.shape = {ndsm.shape}")
print(f"  mask.shape = {mask_bldg.shape}")
if rgb.shape[:2] == dsm.shape[:2] == dtm.shape[:2]:
    print("  ALIGNMENT: ALL MATCH [OK]")
else:
    print("  *** ALIGNMENT MISMATCH! BUILDING CORRESPONDENCE WILL FAIL ***")

# 03 nDSM
save_colormap(ndsm, str(OUT / "03_NDSM.png"), cmap="turbo",
              title=f"03 nDSM (Building Heights)  [{ndsm.min():.1f}–{ndsm.max():.1f}]  P95={np.percentile(ndsm,95):.1f}")
# 04 DTM
save_colormap(dtm, str(OUT / "04_DTM.png"), cmap="terrain",
              title=f"04 DTM (Ground Terrain)  [{dtm.min():.1f}–{dtm.max():.1f}]")
print(f"  [SAVED] 03_NDSM.png, 04_DTM.png")

# ─── STAGE 4: Building mask ────────────────────────────────────────────────────
print(f"\n[STAGE 4] Building mask analysis")
mask_pct = 100.0 * mask_bldg.mean()
num_lab, labels_im, stats_comp, centroids = cv2.connectedComponentsWithStats(mask_bldg)
print(f"  Foreground coverage: {mask_pct:.1f}%")
print(f"  Connected components (all): {num_lab - 1}")

areas = [int(stats_comp[k, cv2.CC_STAT_AREA]) for k in range(1, num_lab)]
print(f"  Component area distribution:")
print(f"    min={min(areas) if areas else 0}  median={int(np.median(areas)) if areas else 0}  max={max(areas) if areas else 0}")
print(f"    # with area>=16: {sum(a>=16 for a in areas)}")
print(f"    # with area>=100: {sum(a>=100 for a in areas)}")
print(f"    # with area>=500: {sum(a>=500 for a in areas)}")

# 06 raw mask
mask_vis = rgb_overlay(img_01, mask_bldg, color=(0,230,80))
cv2.putText(mask_vis, f"06 RAW BUILDING MASK ({mask_pct:.0f}% foreground, {num_lab-1} components)",
            (8,25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,230,80), 2)
cv2.imwrite(str(OUT / "06_RAW_BUILDING_MASK.png"), mask_vis)

# 06_unet — probability
unet_vis = np.zeros_like(img_01)
unet_vis[mask_bldg > 0] = [0,180,60]
unet_vis[mask_bldg == 0] = [30,20,20]
save_colormap(mask_bldg.astype(np.float32), str(OUT / "06_UNET_PROBABILITY.png"),
              cmap="YlOrRd", title="05 U-Net / Building Probability Map (binary)")
print(f"  [SAVED] 06_RAW_BUILDING_MASK.png, 06_UNET_PROBABILITY.png")

# ─── STAGE 5: RGB ↔ DSM alignment visual ──────────────────────────────────────
print(f"\n[STAGE 5] RGB ↔ DSM alignment visual")
rgb_gray = cv2.cvtColor(img_01, cv2.COLOR_BGR2GRAY).astype(np.float32)
rgb_edges = cv2.Canny(cv2.GaussianBlur(img_01, (3,3), 0), 40, 120)

dsm_norm = (dsm - dsm.min()) / (max(dsm.max()-dsm.min(), 1.0))
dsm_uint8 = (dsm_norm * 255).astype(np.uint8)
dsm_edges = cv2.Canny(cv2.GaussianBlur(dsm_uint8, (3,3), 0), 40, 120)

align_vis = img_01.copy()
align_vis[rgb_edges > 0] = [255, 200, 0]   # yellow = RGB edges
align_vis[dsm_edges > 0] = [0, 180, 255]   # cyan = DSM edges
overlap = (rgb_edges > 0) & (dsm_edges > 0)
align_vis[overlap] = [0, 255, 0]            # green = aligned
cv2.putText(align_vis, "11 RGB/DSM ALIGNMENT: Yellow=RGB  Cyan=DSM  Green=MATCH",
            (6,22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
cv2.imwrite(str(OUT / "11_RGB_DSM_ALIGNMENT.png"), align_vis)
overlap_pct = 100.0 * overlap.sum() / max(rgb_edges.sum() + dsm_edges.sum(), 1)
print(f"  Overlap between RGB/DSM edges: {overlap_pct:.1f}%")
print(f"  [SAVED] 11_RGB_DSM_ALIGNMENT.png")

# ─── STAGE 6: Height evidence mask (ndsm >= 1.8m) ─────────────────────────────
print(f"\n[STAGE 6] Height evidence analysis")
height_ev = (ndsm >= 1.8).astype(np.uint8)
height_pct = 100.0 * height_ev.mean()
candidate = cv2.bitwise_and(mask_bldg, height_ev)
cand_pct = 100.0 * candidate.mean()
print(f"  ndsm >= 1.8m: {height_pct:.1f}% of pixels")
print(f"  mask_bldg AND height_evidence: {cand_pct:.1f}% of pixels")

if cand_pct < 0.5:
    print("  *** CRITICAL: After height evidence masking, < 0.5% pixels remain!")
    print(f"  *** nDSM max={ndsm.max():.2f}  → If nDSM is relative (0-10 scale), 1.8m threshold may kill everything!")
    print(f"  *** CHECK: is_metric={calib_res.is_metric}  nDSM max={ndsm.max():.2f}")

# ─── STAGE 7: Detailed component classification ────────────────────────────────
print(f"\n[STAGE 7] Component classification")
kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
candidate_clean = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel_small)
num_cand, lbl_cand, stats_cand, cent_cand = cv2.connectedComponentsWithStats(candidate_clean)

image_area = float(h * w)
rows_comp = []
mega_count = 0
valid_count = 0
frag_count = 0

comp_vis = img_01.copy()
rng = np.random.RandomState(42)
colors_map = rng.randint(80, 255, size=(num_cand+1, 3), dtype=np.uint8)
colors_map[0] = [0,0,0]

for k in range(1, num_cand):
    area = int(stats_cand[k, cv2.CC_STAT_AREA])
    bw   = int(stats_cand[k, cv2.CC_STAT_WIDTH])
    bh_  = int(stats_cand[k, cv2.CC_STAT_HEIGHT])
    cx, cy = float(cent_cand[k][0]), float(cent_cand[k][1])
    
    comp_m = (lbl_cand == k)
    c_ndsm = ndsm[comp_m]
    
    is_mega  = (bw > 0.60*w or bh_ > 0.60*h or area > 0.35*image_area)
    is_frag  = (area < 16)
    
    if is_frag:
        cls = "FRAGMENT"
        frag_count += 1
        col = (80,80,80)
    elif is_mega:
        cls = "MERGED/MEGA"
        mega_count += 1
        col = (0,0,220)
    else:
        cls = "REAL_BUILDING"
        valid_count += 1
        col = (0,220,60)

    rows_comp.append({
        "component_id": k,
        "classification": cls,
        "pixel_area": area,
        "bbox_w": bw, "bbox_h": bh_,
        "centroid_x": round(cx,1), "centroid_y": round(cy,1),
        "ndsm_mean": round(float(np.mean(c_ndsm)),2) if c_ndsm.size else 0,
        "ndsm_p75": round(float(np.percentile(c_ndsm,75)),2) if c_ndsm.size else 0,
        "ndsm_p95": round(float(np.percentile(c_ndsm,95)),2) if c_ndsm.size else 0,
        "ndsm_max": round(float(np.max(c_ndsm)),2) if c_ndsm.size else 0,
    })

    b_mask = (lbl_cand == k).astype(np.uint8)
    contours, _ = cv2.findContours(b_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(comp_vis, contours, -1, [int(c) for c in col], 2)
    cv2.putText(comp_vis, f"{k}:{cls[:3]}", (int(cx)-10, int(cy)+4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, [int(c) for c in col], 1)

print(f"  REAL_BUILDING: {valid_count}")
print(f"  MERGED/MEGA:   {mega_count}")
print(f"  FRAGMENT:      {frag_count}")

cv2.putText(comp_vis, f"08 COMPONENTS: {valid_count} REAL + {mega_count} MEGA + {frag_count} FRAG",
            (6,22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1)
cv2.imwrite(str(OUT / "08_COMPONENTS.png"), comp_vis)

with open(OUT / "component_statistics.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows_comp[0].keys()) if rows_comp else [])
    writer.writeheader()
    writer.writerows(rows_comp)
print(f"  [SAVED] 08_COMPONENTS.png, component_statistics.csv")

# ─── STAGE 8: Human footprint test (Critical Gate) ─────────────────────────────
print(f"\n[STAGE 8] HUMAN FOOTPRINT TEST (Critical Gate)")
fp_vis = img_01.copy()
footprint_count = 0

for row in rows_comp:
    if row["classification"] != "REAL_BUILDING":
        continue
    k = row["component_id"]
    b_mask = (lbl_cand == k).astype(np.uint8)
    contours, _ = cv2.findContours(b_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue
    cnt = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, max(2.0, perimeter/55.0), True)
    cv2.drawContours(fp_vis, [approx], -1, (0,220,60), 2)
    cx, cy = int(row["centroid_x"]), int(row["centroid_y"])
    cv2.putText(fp_vis, f"B{footprint_count+1}", (cx-10, cy+4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,0), 1)
    footprint_count += 1

cv2.putText(fp_vis, f"13 FOOTPRINTS ON RGB ({footprint_count} valid buildings)",
            (6,22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,60), 2)
cv2.imwrite(str(OUT / "10_FOOTPRINTS_OVER_RGB.png"), fp_vis)
cv2.imwrite(str(OUT / "13_FOOTPRINTS_ON_RGB.png"), fp_vis)
print(f"  Valid building footprints drawn: {footprint_count}")
print(f"  [SAVED] 10_FOOTPRINTS_OVER_RGB.png, 13_FOOTPRINTS_ON_RGB.png")

# ─── STAGE 9: Final instances visual ──────────────────────────────────────────
inst_vis = img_01.copy()
for row in rows_comp:
    if row["classification"] != "REAL_BUILDING":
        continue
    k = row["component_id"]
    b_mask = (lbl_cand == k).astype(np.uint8)
    color_fill = rng.randint(80, 255, size=3).tolist()
    overlay_inst = np.zeros_like(img_01)
    overlay_inst[b_mask > 0] = color_fill
    inst_vis = cv2.addWeighted(inst_vis, 0.65, overlay_inst, 0.35, 0)

cv2.putText(inst_vis, f"09 FINAL INSTANCES ({valid_count} buildings  {mega_count} mega  {frag_count} frag)",
            (6,22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1)
cv2.imwrite(str(OUT / "09_FINAL_INSTANCES.png"), inst_vis)
print(f"  [SAVED] 09_FINAL_INSTANCES.png")

# ─── STAGE 10: Mask over RGB ───────────────────────────────────────────────────
mask_over = rgb_overlay(img_01, mask_bldg, color=(0,255,80), alpha=0.5)
cv2.putText(mask_over, f"12 MASK OVER RGB ({mask_pct:.0f}% foreground)",
            (6,22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,80), 2)
cv2.imwrite(str(OUT / "12_MASK_OVER_RGB.png"), mask_over)
print(f"  [SAVED] 12_MASK_OVER_RGB.png")

# ─── STAGE 11: Solid building control ─────────────────────────────────────────
print(f"\n[STAGE 9] Solid building control render (top-down flat box test)")
ctrl_vis = np.zeros_like(img_01)
ctrl_vis[:] = [20,20,35]

gsd_x = float(raster_in.gsd[0]) if isinstance(raster_in.gsd, (list,tuple)) else float(raster_in.gsd)
gsd_y = float(raster_in.gsd[1]) if isinstance(raster_in.gsd, (list,tuple)) else float(raster_in.gsd)
solid_count = 0

z_base = float(np.percentile(dtm, 2))
z_max  = float(np.percentile(dsm, 98))
z_range = max(1.0, z_max - z_base)

for row in rows_comp:
    if row["classification"] != "REAL_BUILDING":
        continue
    k = row["component_id"]
    b_mask_u8 = (lbl_cand == k).astype(np.uint8)
    contours_ctrl, _ = cv2.findContours(b_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_ctrl:
        continue
    
    # Get roof height
    interior_ndsm = ndsm[b_mask_u8 > 0]
    z_ground = float(np.percentile(dtm[b_mask_u8 > 0], 30)) if b_mask_u8.sum() > 0 else z_base
    z_roof = float(np.percentile(ndsm[b_mask_u8 > 0], 75)) + z_ground if b_mask_u8.sum() > 0 else z_ground + 5.0
    bldg_h = max(1.5, z_roof - z_ground)
    
    # Color by height (green=low, yellow=mid, red=high)
    h_norm = float(np.clip(bldg_h / 60.0, 0, 1))
    r_c = int(255 * min(1.0, h_norm * 2))
    g_c = int(255 * min(1.0, (1.0 - h_norm) * 2))
    cv2.drawContours(ctrl_vis, contours_ctrl, -1, (0, g_c, r_c), -1)
    cv2.drawContours(ctrl_vis, contours_ctrl, -1, (200, 200, 200), 1)
    
    cx, cy = int(row["centroid_x"]), int(row["centroid_y"])
    cv2.putText(ctrl_vis, f"{bldg_h:.0f}m", (cx-10, cy+4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255,255,255), 1)
    solid_count += 1

cv2.putText(ctrl_vis, f"14 SOLID BUILDING CONTROL ({solid_count} buildings, colored by height)",
            (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1)
cv2.imwrite(str(OUT / "14_SOLID_BUILDING_CONTROL.png"), ctrl_vis)
print(f"  [SAVED] 14_SOLID_BUILDING_CONTROL.png ({solid_count} solid boxes)")

# ─── FINAL DIAGNOSIS ──────────────────────────────────────────────────────────
print()
print("="*70)
print("FORENSIC DIAGNOSIS SUMMARY")
print("="*70)
print(f"  Input RGB shape:          {rgb.shape}")
print(f"  Calibration mode:         {calib_res.mode_used}")
print(f"  is_metric:                {calib_res.is_metric}")
print(f"  nDSM range:               [{ndsm.min():.2f}, {ndsm.max():.2f}]")
print(f"  nDSM >= 1.8m coverage:    {(ndsm >= 1.8).mean()*100:.1f}%")
print(f"  mask_bldg coverage:       {mask_bldg.mean()*100:.1f}%")
print(f"  candidate (mask&height):  {candidate.mean()*100:.1f}%")
print(f"  Building components:      {valid_count} REAL  {mega_count} MEGA  {frag_count} FRAG")
print(f"  Footprints over RGB:      {footprint_count}")
print()

issues = []
if not calib_res.is_metric:
    issues.append("CRITICAL: Calibration is NON-METRIC (relative). nDSM values are in 0-10 relative scale.")
    issues.append("  -> The 1.8m height evidence threshold destroys most/all pixels!")
    issues.append("  -> Fix: adapt height_evidence threshold to non-metric scale, OR ensure metric calibration.")
if (ndsm >= 1.8).mean() < 0.02:
    issues.append("CRITICAL: Less than 2% of pixels pass height_evidence (ndsm >= 1.8). Almost no buildings survive!")
if candidate.mean() < 0.005:
    issues.append("CRITICAL: After height_evidence masking, < 0.5% of pixels remain as building candidates.")
    issues.append("  -> This is why 3D scene has almost no buildings despite 'N structures detected'.")
if not rgb.shape[:2] == dsm.shape[:2]:
    issues.append("CRITICAL: RGB and DSM shapes do not match — spatial misalignment!")

if issues:
    print("CRITICAL ISSUES FOUND:")
    for iss in issues:
        print(f"  [FAIL] {iss}")
else:
    print("  [OK] No critical issues found at diagnostic level.")
    print("     Building extraction appears functional.")
    print("     Problem may be in roof/wall geometry or Three.js rendering.")

print()
print(f"All diagnostic images saved to: {OUT.resolve()}")
print("="*70)

# Save diagnosis to JSON
diag = {
    "calibration_mode": str(calib_res.mode_used),
    "is_metric": calib_res.is_metric,
    "units": calib_res.units,
    "ndsm_min": round(float(ndsm.min()),3),
    "ndsm_max": round(float(ndsm.max()),3),
    "ndsm_p50": round(float(np.percentile(ndsm,50)),3),
    "ndsm_p95": round(float(np.percentile(ndsm,95)),3),
    "ndsm_gte_1_8_pct": round(float((ndsm>=1.8).mean()*100), 2),
    "mask_bldg_pct": round(float(mask_bldg.mean()*100), 2),
    "candidate_mask_pct": round(float(candidate.mean()*100), 2),
    "components_real": valid_count,
    "components_mega": mega_count,
    "components_fragment": frag_count,
    "footprints_extracted": footprint_count,
    "critical_issues": issues,
}
with open(OUT / "DIAGNOSIS.json", "w", encoding="utf-8") as f:
    json.dump(diag, f, indent=2)
print(f"  [SAVED] DIAGNOSIS.json")
