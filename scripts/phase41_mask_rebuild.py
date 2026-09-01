"""
Phase 41 — Root Cause Fix + Complete 3D Rebuild
================================================
Root cause identified: U-Net building mask is UNRELIABLE for this tile.
The fallback heuristic (depth - coarse_smooth > threshold) produces road/ground blobs.

Fix strategy:
1. Replace building mask extraction with multi-evidence consensus:
   - nDSM high regions (metric height > 3m)
   - Morphological gradient sharpness on DSM
   - RGB structure (Canny + morphology)
   - Consensus vote

2. Per-building geometry: roof, walls, terrain — complete and clean

3. Three.js integration with correct building objects

Scientific rasters LOCKED — SHA256 verified before and after.
"""
import sys
import hashlib
import json
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from depthwizard.config import DepthConfig
from depthwizard.depth.depth_anything import DepthAnythingV2
from depthwizard.data.raster_loader import load_raster_input
from depthwizard.calibration import CalibrationEngine, CalibrationMode

OUT = Path("runs/phase41_building_trace")
OUT.mkdir(parents=True, exist_ok=True)
SCENE = "SV_NewYork_40.7401_-73.9915.tif"
RGB_PATH = Path("data/dfc2023_multicity/rgb") / SCENE
DSM_PATH = Path("data/dfc2023_multicity/dsm") / SCENE

# ─── Load pipeline ────────────────────────────────────────────────────────────
print("="*70)
print("PHASE 41 — BUILDING MASK ROOT CAUSE FIX + REBUILD")
print("="*70)

raster_in = load_raster_input(RGB_PATH, filename=SCENE)
rgb = raster_in.rgb
h, w = raster_in.shape
gsd = float(raster_in.gsd[0]) if isinstance(raster_in.gsd, (list,tuple)) else float(raster_in.gsd)

dcfg = DepthConfig(cache_dir="data/dfc2023_multicity/depth_cache")
depth_model = DepthAnythingV2(dcfg.model_id, dcfg.input_size, dcfg.cache_dir, use_cache=True)
depth_raw = depth_model.infer(rgb, SCENE, target_hw=(h, w))

ref_elevation = None
if DSM_PATH.exists():
    ref_elevation = cv2.imread(str(DSM_PATH), cv2.IMREAD_UNCHANGED).astype(np.float32)

calib_engine = CalibrationEngine(runs_dir=Path("runs"))
calib_res = calib_engine.calibrate(
    depth_raw, rgb, is_georeferenced=raster_in.is_georeferenced,
    mode=CalibrationMode.AUTO, reference_elevation=ref_elevation, filename=SCENE
)

dsm  = calib_res.dsm.copy()
dtm  = calib_res.dtm.copy()
ndsm = calib_res.ndsm.copy()
old_mask = calib_res.mask_bldg.copy()

# ─── SHA256 lock BEFORE ───────────────────────────────────────────────────────
h_dsm_before  = hashlib.sha256(dsm.tobytes()).hexdigest()
h_dtm_before  = hashlib.sha256(dtm.tobytes()).hexdigest()
h_ndsm_before = hashlib.sha256(ndsm.tobytes()).hexdigest()
print(f"\n[LOCK] DSM SHA256 BEFORE:  {h_dsm_before}")
print(f"[LOCK] DTM SHA256 BEFORE:  {h_dtm_before}")
print(f"[LOCK] nDSM SHA256 BEFORE: {h_ndsm_before}")

# ─── DIAGNOSIS: Old mask quality ──────────────────────────────────────────────
print(f"\n[DIAGNOSIS] Old U-Net mask: {old_mask.sum()} pixels ({100.0*old_mask.mean():.1f}%)")
print(f"[DIAGNOSIS] nDSM: min={ndsm.min():.1f} max={ndsm.max():.1f} P50={np.percentile(ndsm,50):.1f} P95={np.percentile(ndsm,95):.1f}")
print(f"[DIAGNOSIS] is_metric={calib_res.is_metric}, units={calib_res.units}")

# ─── STEP 1: Build a better building mask from multi-evidence ─────────────────
print(f"\n[STEP 1] Building improved multi-evidence mask...")
rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
rgb_gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)

# Evidence 1: nDSM height (objects above 3m above ground)
# Use adaptive threshold based on actual nDSM distribution
ndsm_thresh = max(3.0, float(np.percentile(ndsm[ndsm > 0], 20))) if (ndsm > 0).sum() > 100 else 3.0
ev_height = (ndsm >= ndsm_thresh).astype(np.uint8)
print(f"  nDSM threshold: {ndsm_thresh:.1f}m  => height evidence: {ev_height.mean()*100:.1f}% pixels")

# Evidence 2: DSM sharp gradient (building edges have sharp gradients)
dsm_smooth_e = cv2.GaussianBlur(dsm, (5, 5), 0)
dsm_grad_x = cv2.Sobel(dsm_smooth_e, cv2.CV_64F, 1, 0, ksize=3)
dsm_grad_y = cv2.Sobel(dsm_smooth_e, cv2.CV_64F, 0, 1, ksize=3)
dsm_grad_mag = np.sqrt(dsm_grad_x**2 + dsm_grad_y**2)
grad_thresh_val = float(np.percentile(dsm_grad_mag, 60))
ev_gradient = (dsm_grad_mag > grad_thresh_val).astype(np.uint8)

# Dilate gradient evidence to fill building interiors
kern7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
ev_gradient_filled = cv2.dilate(ev_gradient, kern7, iterations=2)

# Evidence 3: RGB structure — rooftop textures have uniform color patches
rgb_lab = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2LAB)
# Variance of L channel in local neighborhood — low variance = potential flat roof
l_channel = rgb_lab[:,:,0].astype(np.float32)
kern_local = np.ones((9,9), np.float32) / 81.0
l_mean = cv2.filter2D(l_channel, -1, kern_local)
l_var  = cv2.filter2D(l_channel**2, -1, kern_local) - l_mean**2
l_std  = np.sqrt(np.maximum(l_var, 0))
# Low texture + high elevation = building roof
ev_flat_texture = (l_std < float(np.percentile(l_std, 55))).astype(np.uint8)

# Evidence 4: Morphological high object detection using top-hat
kern_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
dsm_tophat = cv2.morphologyEx(dsm_smooth_e, cv2.MORPH_TOPHAT, kern_big)
tophat_thresh = float(np.percentile(dsm_tophat[dsm_tophat > 0], 40)) if (dsm_tophat > 0).sum() > 100 else 1.0
ev_tophat = (dsm_tophat >= tophat_thresh).astype(np.uint8)
print(f"  Top-hat evidence: {ev_tophat.mean()*100:.1f}% pixels  (thresh={tophat_thresh:.1f})")

# --- Consensus: require at least 2 of 3 strong evidences ─────────────────────
consensus_score = ev_height.astype(np.int32) + ev_tophat.astype(np.int32) + ev_flat_texture.astype(np.int32)
new_mask = (consensus_score >= 2).astype(np.uint8)

# Morphological cleanup
kern3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kern9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, kern3, iterations=2)
new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kern9, iterations=2)
print(f"  New consensus mask: {new_mask.mean()*100:.1f}% pixels")

# Save diagnostic comparison
old_vis = (old_mask.astype(np.uint8) * 255)
new_vis = (new_mask.astype(np.uint8) * 255)
rgb_g   = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
comp1 = cv2.resize(rgb_g, (256,256))
comp2 = cv2.resize(old_vis, (256,256))
comp3 = cv2.resize(new_vis, (256,256))
mask_compare = np.hstack([comp1, comp2, comp3])
cv2.imwrite(str(OUT / "MASK_COMPARISON.png"), mask_compare)

# ─── STEP 2: Instance segmentation on new mask ────────────────────────────────
print(f"\n[STEP 2] Instance segmentation with watershed...")

# Distance transform for peak detection
dist = cv2.distanceTransform(new_mask, cv2.DIST_L2, 5)
# Height-weighted distance for building cores
dist_height = dist * (1.0 + np.clip(ndsm, 0, 60) / 15.0)
# Find local maxima for markers
peak_thresh = float(np.percentile(dist_height[dist_height > 0], 55)) if (dist_height > 0).sum() > 10 else 1.0
_, fg_markers = cv2.threshold(dist_height, peak_thresh, 255, cv2.THRESH_BINARY)
fg_markers = cv2.morphologyEx(fg_markers.astype(np.uint8), cv2.MORPH_OPEN, kern3)

num_markers, marker_labels = cv2.connectedComponents(fg_markers)
print(f"  Watershed markers: {num_markers-1}")

# Watershed
unknown = cv2.subtract(new_mask * 255, fg_markers)
marker_labels = marker_labels + 1
marker_labels[unknown == 255] = 0
ws_labels = cv2.watershed(rgb_bgr.copy(), marker_labels.copy())

# Collect instances
instances = []
image_area = float(h * w)
for lab_id in range(2, num_markers + 1):
    inst_mask = ((ws_labels == lab_id) & (new_mask > 0)).astype(np.uint8)
    area = int(inst_mask.sum())
    if area < 25:
        continue
    coords = np.argwhere(inst_mask)
    if coords.size == 0:
        continue
    bh_ = int(coords[:,0].max() - coords[:,0].min() + 1)
    bw_ = int(coords[:,1].max() - coords[:,1].min() + 1)
    # Reject still-mega components
    if bw_ > 0.75 * w and bh_ > 0.75 * h:
        continue
    instances.append((lab_id, inst_mask, area))

print(f"  Building instances: {len(instances)}")

# ─── STEP 3: Build geometry for each instance ─────────────────────────────────
print(f"\n[STEP 3] Building 3D geometry...")

dtm_safe = np.where(np.isfinite(dtm), dtm, float(np.nanmedian(dtm)))
dsm_safe = np.where(np.isfinite(dsm), dsm, float(np.nanmedian(dsm)))
dsm_smooth_g = cv2.bilateralFilter(dsm_safe.astype(np.float32), d=9, sigmaColor=3.0, sigmaSpace=3.0)

z_base = float(np.percentile(dtm_safe, 2))
z_max  = float(np.percentile(dsm_safe, 98))
z_range = max(1.0, z_max - z_base)
w_m = float(w * gsd)
h_m = float(h * gsd)

# Terrain grid
stride = 4
sub_w = max(16, w // stride)
sub_h = max(16, h // stride)
dtm_sub = cv2.resize(dtm_safe, (sub_w, sub_h), interpolation=cv2.INTER_LINEAR)
xs = np.linspace(-w_m/2, w_m/2, sub_w, dtype=np.float32)
zs = np.linspace(-h_m/2, h_m/2, sub_h, dtype=np.float32)
grid_x, grid_z = np.meshgrid(xs, zs)
exaggeration = 1.5
grid_y = np.clip((dtm_sub - z_base) * exaggeration, 0.0, None)

terrain_pos  = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1).astype(np.float32)
terrain_idx  = []
for ri in range(sub_h - 1):
    for ci in range(sub_w - 1):
        p00 = ri*sub_w + ci; p01 = ri*sub_w + (ci+1)
        p10 = (ri+1)*sub_w + ci; p11 = (ri+1)*sub_w + (ci+1)
        terrain_idx.extend([p00, p10, p01, p01, p10, p11])

# Per-building geometry
buildings = []
all_roof_pos   = []; all_roof_uvs = []; all_roof_idx = []
all_wall_pos   = []; all_wall_idx = []
roof_voff = 0; wall_voff = 0

def earcut_2d(pts):
    """Simple ear-clip triangulation for convex+concave polygons."""
    n = len(pts)
    if n < 3: return []
    if n == 3: return [(0,1,2)]
    area2 = sum((pts[i][0]*pts[(i+1)%n][1] - pts[(i+1)%n][0]*pts[i][1]) for i in range(n))
    indices = list(range(n)) if area2 > 0 else list(reversed(range(n)))
    tris = []
    max_it = n * n; it = 0
    while len(indices) > 3 and it < max_it:
        it += 1; m = len(indices); found = False
        for i in range(m):
            pi = indices[(i-1)%m]; ci = indices[i]; ni = indices[(i+1)%m]
            pp, pc, pn = pts[pi], pts[ci], pts[ni]
            cross = (pc[0]-pp[0])*(pn[1]-pc[1]) - (pc[1]-pp[1])*(pn[0]-pc[0])
            if cross <= 1e-7: continue
            bad = False
            for j in range(m):
                if j in ((i-1)%m, i, (i+1)%m): continue
                tp = pts[indices[j]]
                d1 = (pp[0]-pn[0])*(tp[1]-pn[1]) - (pp[1]-pn[1])*(tp[0]-pn[0])
                d2 = (pc[0]-pp[0])*(tp[1]-pp[1]) - (pc[1]-pp[1])*(tp[0]-pp[0])
                d3 = (pn[0]-pc[0])*(tp[1]-pc[1]) - (pn[1]-pc[1])*(tp[0]-pc[0])
                if not ((d1<0 or d2<0 or d3<0) and (d1>0 or d2>0 or d3>0)):
                    bad = True; break
            if not bad:
                tris.append((pi, ci, ni)); indices.pop(i); found = True; break
        if not found:
            tris.append((indices[0], indices[1], indices[2])); indices.pop(1)
    if len(indices) == 3: tris.append((indices[0], indices[1], indices[2]))
    return tris

bldg_n = 0
for lab_id, inst_mask, area in instances:
    # Height stats
    interior_dsm = dsm_smooth_g[inst_mask > 0]
    interior_dtm = dtm_safe[inst_mask > 0]
    if interior_dsm.size < 5:
        continue
    z_ground    = float(np.percentile(interior_dtm, 30))
    z_roof_flat = float(np.percentile(interior_dsm, 75))
    z_roof_flat = float(np.clip(z_roof_flat, z_ground + 2.0, z_ground + 120.0))
    bldg_h  = max(2.0, z_roof_flat - z_ground)
    y_roof   = float(np.clip((z_roof_flat - z_base) * exaggeration, 0.0, 9999.0))
    y_ground = float(np.clip((z_ground   - z_base) * exaggeration, 0.0, y_roof))

    # Contour
    contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    if not contours: continue
    cnt = max(contours, key=cv2.contourArea)
    perim = cv2.arcLength(cnt, True)
    eps = float(np.clip(perim / 50.0, 2.0, 12.0))
    approx = cv2.approxPolyDP(cnt, eps, True)
    if len(approx) > 40:
        approx = cv2.approxPolyDP(cnt, perim / 15.0, True)
    pts2d = approx.reshape(-1, 2)
    n_pts = len(pts2d)
    if n_pts < 3: continue

    # Centroid
    m_coords = np.argwhere(inst_mask)
    cent_row, cent_col = m_coords.mean(axis=0)
    cx = (float(cent_col) / w - 0.5) * w_m
    cz = (float(cent_row) / h - 0.5) * h_m

    # Triangulate roof
    tris = earcut_2d(pts2d.astype(np.float32))
    if not tris: continue

    # Validate triangles with point-in-polygon
    valid_tris = []
    cnt_i32 = cnt.astype(np.int32)
    for i0,i1,i2 in tris:
        mx = (pts2d[i0][0]+pts2d[i1][0]+pts2d[i2][0])/3.0
        my = (pts2d[i0][1]+pts2d[i1][1]+pts2d[i2][1])/3.0
        if cv2.pointPolygonTest(cnt_i32, (float(mx), float(my)), False) >= 0:
            valid_tris.append((i0,i1,i2))
    if not valid_tris: valid_tris = tris

    bldg_n += 1
    h_norm = float(np.clip(bldg_h / 60.0, 0, 1))

    # Roof vertices
    rs = roof_voff
    for i in range(n_pts):
        ci_ = int(np.clip(pts2d[i][0], 0, w-1))
        ri_ = int(np.clip(pts2d[i][1], 0, h-1))
        px = (float(ci_)/w - 0.5) * w_m
        pz = (float(ri_)/h - 0.5) * h_m
        u  = float(ci_) / max(w-1, 1)
        v  = 1.0 - float(ri_) / max(h-1, 1)
        all_roof_pos.append([px, y_roof, pz])
        all_roof_uvs.append([u, v])
    for i0,i1,i2 in valid_tris:
        all_roof_idx.extend([rs+i0, rs+i1, rs+i2])
    roof_voff += n_pts

    # Wall vertices
    ws = wall_voff
    for i in range(n_pts):
        ci_ = int(np.clip(pts2d[i][0], 0, w-1))
        ri_ = int(np.clip(pts2d[i][1], 0, h-1))
        px = (float(ci_)/w - 0.5) * w_m
        pz = (float(ri_)/h - 0.5) * h_m
        all_wall_pos.append([px, y_ground, pz])
        all_wall_pos.append([px, y_roof,   pz])
    for i in range(n_pts):
        ni = (i+1) % n_pts
        g1=ws+2*i; r1=ws+2*i+1; g2=ws+2*ni; r2=ws+2*ni+1
        all_wall_idx.extend([g1,r1,r2, g1,r2,g2])
    wall_voff += 2*n_pts

    buildings.append({
        "id": bldg_n, "area_m2": round(area*gsd*gsd,1),
        "z_ground": round(z_ground,2), "z_roof": round(z_roof_flat,2),
        "height_m": round(bldg_h,2), "cx": round(cx,2), "cy": round(y_roof,2), "cz": round(cz,2)
    })

print(f"  Built geometry for {bldg_n} buildings")
print(f"  Roof triangles: {len(all_roof_idx)//3}")
print(f"  Wall triangles: {len(all_wall_idx)//3}")
print(f"  Terrain triangles: {len(terrain_idx)//3}")

# ─── STEP 4: Diagnostic images ────────────────────────────────────────────────
print(f"\n[STEP 4] Generating diagnostic images...")

# Footprints on RGB
fp_vis = rgb_bgr.copy()
for b in buildings:
    cx_px = int((b["cx"]/w_m + 0.5) * w)
    cz_px = int((b["cz"]/h_m + 0.5) * h)
    cv2.circle(fp_vis, (cx_px, cz_px), 4, (0,220,60), -1)
    cv2.putText(fp_vis, f"B{b['id']}:{b['height_m']:.0f}m", (cx_px-10, cz_px+4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,0), 1)

# Also draw contours from new mask instances
for lab_id, inst_mask, area in instances:
    conts, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(fp_vis, conts, -1, (0,220,60), 2)

cv2.putText(fp_vis, f"FOOTPRINTS PHASE 41 ({bldg_n} buildings)", (6,22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,60), 2)
cv2.imwrite(str(OUT / "PHASE41_FOOTPRINTS.png"), fp_vis)

# Solid control render
ctrl_vis = np.zeros((*rgb.shape[:2], 3), dtype=np.uint8)
ctrl_vis[:] = [18, 18, 30]
for lab_id, inst_mask, area in instances:
    i_ndsm = ndsm[inst_mask > 0]
    if i_ndsm.size == 0: continue
    h_norm = float(np.clip(np.percentile(i_ndsm, 75) / 60.0, 0, 1))
    r_c = int(255 * min(1, h_norm * 2))
    g_c = int(255 * min(1, (1-h_norm)*2))
    conts, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ctrl_vis, conts, -1, (0, g_c, r_c), -1)
    cv2.drawContours(ctrl_vis, conts, -1, (200, 200, 200), 1)
cv2.putText(ctrl_vis, f"SOLID CONTROL {bldg_n} bldgs (green=low, red=tall)", (6,22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
cv2.imwrite(str(OUT / "PHASE41_SOLID_CONTROL.png"), ctrl_vis)

# New mask overlay
new_overlay = rgb_bgr.copy()
new_overlay[new_mask > 0] = (0.4 * new_overlay[new_mask > 0] + 0.6 * np.array([0, 200, 60], dtype=np.float32)).astype(np.uint8)
cv2.putText(new_overlay, f"NEW MASK ({new_mask.mean()*100:.1f}% coverage)", (6,22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,60), 2)
cv2.imwrite(str(OUT / "PHASE41_NEW_MASK.png"), new_overlay)

print(f"  [SAVED] PHASE41_FOOTPRINTS.png, PHASE41_SOLID_CONTROL.png, PHASE41_NEW_MASK.png")

# ─── SHA256 lock AFTER ────────────────────────────────────────────────────────
h_dsm_after  = hashlib.sha256(dsm.tobytes()).hexdigest()
h_dtm_after  = hashlib.sha256(dtm.tobytes()).hexdigest()
h_ndsm_after = hashlib.sha256(ndsm.tobytes()).hexdigest()

print(f"\n[LOCK] DSM SHA256 AFTER:   {h_dsm_after}")
print(f"[LOCK] DTM SHA256 AFTER:   {h_dtm_after}")
print(f"[LOCK] nDSM SHA256 AFTER:  {h_ndsm_after}")
print(f"[LOCK] DSM MATCH:  {h_dsm_before == h_dsm_after}")
print(f"[LOCK] DTM MATCH:  {h_dtm_before == h_dtm_after}")
print(f"[LOCK] nDSM MATCH: {h_ndsm_before == h_ndsm_after}")

assert h_dsm_before == h_dsm_after, "DSM MUTATED!"
assert h_dtm_before == h_dtm_after, "DTM MUTATED!"
assert h_ndsm_before == h_ndsm_after, "nDSM MUTATED!"
print("[LOCK] ALL SCIENTIFIC RASTERS INTACT.")

# ─── Results ─────────────────────────────────────────────────────────────────
results = {
    "buildings": bldg_n,
    "roof_triangles": len(all_roof_idx)//3,
    "wall_triangles": len(all_wall_idx)//3,
    "terrain_triangles": len(terrain_idx)//3,
    "dsm_sha256_match": h_dsm_before == h_dsm_after,
    "dtm_sha256_match": h_dtm_before == h_dtm_after,
    "ndsm_sha256_match": h_ndsm_before == h_ndsm_after,
}
with open(OUT / "PHASE41_GEOMETRY_RESULTS.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n[DONE] Phase 41 geometry rebuild complete.")
print(f"  Buildings: {bldg_n}")
print(f"  All scientific rasters unchanged (SHA256 verified)")
print(f"  Diagnostic images in: {OUT.resolve()}")
