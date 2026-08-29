"""
Phase 31C — 3D Reconstruction Integrity Audit
=============================================
Read-only diagnostic: DSM gradient audit, spike detection,
mesh A/B test, texture A/B test, flat-surface control,
resolution downscale test.

DOES NOT modify any scientific DSM data.
"""
import os
import sys
import json
import numpy as np
import cv2
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import median_filter
from pathlib import Path
import pyvista as pv
pv.OFF_SCREEN = True

# ─── Output directory ─────────────────────────────────────────────────────────
OUT_DIR   = Path("runs/phase31c_3d_integrity")
FIG_DIR   = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─── Primary DSM sources (frozen, read-only) ──────────────────────────────────
TILES = {
    "skyscraper-heavy": Path("runs/phase31_3d_prototype/demo/reconstructed_SV_NewYork_40.7401_-73.9915.tif"),
    "dense-highrise":   Path("runs/phase31_3d_prototype/demo/reconstructed_SV_NewYork_40.7372_-73.9901.tif"),
    "lower-rise":       Path("runs/phase31_3d_prototype/demo/reconstructed_SV_NewYork_40.7373_-74.0034.tif"),
}
RGB_DIR = Path("data/dfc2023_multicity/rgb")
RGB_FILES = {
    "skyscraper-heavy": RGB_DIR / "SV_NewYork_40.7401_-73.9915.tif",
    "dense-highrise":   RGB_DIR / "SV_NewYork_40.7372_-73.9901.tif",
    "lower-rise":       RGB_DIR / "SV_NewYork_40.7373_-74.0034.tif",
}

results = {}

# ─── Helpers ──────────────────────────────────────────────────────────────────
def load_dsm(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        transform = src.transform
    return arr, transform


def load_rgb(path):
    with rasterio.open(path) as src:
        if src.count >= 3:
            b = src.read([1, 2, 3])
        else:
            b0 = src.read(1)
            b = np.stack([b0, b0, b0])
    img = np.transpose(b, (1, 2, 0))
    if img.dtype != np.uint8:
        mn, mx = img.min(), img.max()
        img = ((img - mn) / (mx - mn + 1e-6) * 255).astype(np.uint8)
    return img


def geo_coords(transform, h, w):
    cols = np.arange(w, dtype=np.float64)
    rows = np.arange(h, dtype=np.float64)
    c_g, r_g = np.meshgrid(cols, rows)
    x_g = transform.a * c_g + transform.c
    y_g = transform.e * r_g + transform.f
    return x_g, y_g


def build_mesh(Z, transform):
    h, w = Z.shape
    x_g, y_g = geo_coords(transform, h, w)
    pts = np.stack([x_g.ravel(), y_g.ravel(), Z.ravel()], axis=1).astype(np.float64)
    grid = pv.StructuredGrid()
    grid.points = pts
    grid.dimensions = (w, h, 1)
    mesh = grid.extract_surface(algorithm='dataset_surface')
    mesh['Elevation'] = Z.ravel().astype(np.float32)
    mesh.set_active_scalars('Elevation')
    u_g, v_g = np.meshgrid(np.linspace(0, 1, w), np.linspace(1, 0, h))
    mesh.active_texture_coordinates = np.stack([u_g.ravel(), v_g.ravel()], axis=1)
    return mesh


def render_mesh(mesh, path, rgb_img=None, title=""):
    plotter = pv.Plotter(off_screen=True, window_size=(1200, 700))
    if rgb_img is not None:
        tex = pv.numpy_to_texture(rgb_img)
        plotter.add_mesh(mesh, texture=tex, show_edges=False)
    else:
        plotter.add_mesh(mesh, scalars='Elevation', cmap="plasma",
                         show_edges=False,
                         clim=[float(mesh['Elevation'].min()), float(mesh['Elevation'].max())])
        plotter.add_scalar_bar("Elevation (m)")
    plotter.set_background("#111111")
    pts = mesh.points
    cx = float(np.array(pts[:, 0]).mean())
    cy = float(np.array(pts[:, 1]).mean())
    cz = float(np.array(pts[:, 2]).mean())
    rng = max(float(np.array(pts[:, 0]).max() - np.array(pts[:, 0]).min()),
              float(np.array(pts[:, 1]).max() - np.array(pts[:, 1]).min()))
    plotter.camera_position = [
        (cx - rng * 0.6, cy - rng * 0.6, cz + rng * 0.5),
        (cx, cy, cz), (0, 0, 1)]
    plotter.add_title(title, font_size=9)
    plotter.screenshot(str(path))
    plotter.close()


def smooth_viz(Z, sigma_px=2):
    """Gaussian blur for visualization ONLY. Never modifies scientific data."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(Z.astype(np.float32), sigma=sigma_px)


print("=" * 70)
print("PHASE 31C — 3D RECONSTRUCTION INTEGRITY AUDIT")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# 2. DSM LOCAL-GRADIENT AUDIT
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] DSM Local-Gradient Audit")
gradient_results = {}
for tile_name, dsm_path in TILES.items():
    if not dsm_path.exists():
        print(f"  SKIP {tile_name}: {dsm_path}")
        continue
    Z, transform = load_dsm(dsm_path)
    dx = np.abs(Z[:, 1:] - Z[:, :-1]).ravel()
    dy = np.abs(Z[1:, :] - Z[:-1, :]).ravel()
    d_all = np.concatenate([dx, dy])
    gr = {
        "dx_median": float(np.median(dx)), "dx_p95": float(np.percentile(dx, 95)),
        "dx_p99": float(np.percentile(dx, 99)), "dx_max": float(dx.max()),
        "dy_median": float(np.median(dy)), "dy_p95": float(np.percentile(dy, 95)),
        "dy_p99": float(np.percentile(dy, 99)), "dy_max": float(dy.max()),
        "pct_gt_1m":  float((d_all > 1.0).mean() * 100),
        "pct_gt_2m":  float((d_all > 2.0).mean() * 100),
        "pct_gt_5m":  float((d_all > 5.0).mean() * 100),
        "pct_gt_10m": float((d_all > 10.0).mean() * 100),
    }
    gradient_results[tile_name] = gr
    print(f"\n  [{tile_name}]")
    print(f"    dx  median={gr['dx_median']:.3f}m  P95={gr['dx_p95']:.2f}m  P99={gr['dx_p99']:.2f}m  max={gr['dx_max']:.2f}m")
    print(f"    dy  median={gr['dy_median']:.3f}m  P95={gr['dy_p95']:.2f}m  P99={gr['dy_p99']:.2f}m  max={gr['dy_max']:.2f}m")
    print(f"    |ΔZ|>1m:{gr['pct_gt_1m']:.1f}%  >2m:{gr['pct_gt_2m']:.1f}%  >5m:{gr['pct_gt_5m']:.1f}%  >10m:{gr['pct_gt_10m']:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im0 = axes[0].imshow(np.abs(Z[:, 1:] - Z[:, :-1]), cmap='hot', vmax=gr['dx_p99'])
    axes[0].set_title(f"X-gradient |ΔZ| (clamped P99={gr['dx_p99']:.1f}m)")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(np.abs(Z[1:, :] - Z[:-1, :]), cmap='hot', vmax=gr['dy_p99'])
    axes[1].set_title(f"Y-gradient |ΔZ| (clamped P99={gr['dy_p99']:.1f}m)")
    plt.colorbar(im1, ax=axes[1])
    plt.suptitle(f"DSM Gradient Heatmap — {tile_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"01_gradient_{tile_name}.png", dpi=120)
    plt.close()

results["gradient_audit"] = gradient_results

# ══════════════════════════════════════════════════════════════════════════════
# 3. SPIKE / OUTLIER AUDIT
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Spike / Outlier Audit")
spike_results = {}
for tile_name, dsm_path in TILES.items():
    if not dsm_path.exists():
        continue
    Z, transform = load_dsm(dsm_path)
    local_med = median_filter(Z, size=3)
    spike_err = np.abs(Z - local_med).ravel()
    sr = {
        "p95": float(np.percentile(spike_err, 95)),
        "p99": float(np.percentile(spike_err, 99)),
        "max": float(spike_err.max()),
        "frac_gt_2m": float((spike_err > 2.0).mean()),
        "frac_gt_5m": float((spike_err > 5.0).mean()),
        "frac_gt_10m": float((spike_err > 10.0).mean()),
    }
    spike_results[tile_name] = sr
    print(f"\n  [{tile_name}]")
    print(f"    spike_err  P95={sr['p95']:.2f}m  P99={sr['p99']:.2f}m  max={sr['max']:.2f}m")
    print(f"    frac>2m={sr['frac_gt_2m']:.4f}  frac>5m={sr['frac_gt_5m']:.4f}  frac>10m={sr['frac_gt_10m']:.6f}")

    spike_map = np.abs(Z - local_med)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(spike_map, cmap='inferno', vmax=sr['p99'])
    plt.colorbar(im, ax=ax, label='|Z - local_median_3x3| (m)')
    ax.set_title(f"Spike Error Map — {tile_name}\n(P99={sr['p99']:.2f}m  max={sr['max']:.2f}m)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"02_spike_{tile_name}.png", dpi=120)
    plt.close()

results["spike_audit"] = spike_results

# ══════════════════════════════════════════════════════════════════════════════
# 4. WHERE DO LARGE JUMPS OCCUR?
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Building-Scale Transition Analysis")
PRIMARY = "skyscraper-heavy"
jump_results = {}
dsm_path = TILES[PRIMARY]
if dsm_path.exists():
    Z, transform = load_dsm(dsm_path)
    spike_map = np.abs(Z - median_filter(Z, size=3))
    spike_u8 = (np.clip(spike_map / (spike_map.max() + 1e-6), 0, 1) * 255).astype(np.uint8)
    _, spike_mask = cv2.threshold(spike_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    Z_u8 = cv2.normalize(Z, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(Z_u8, 30, 90)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    boundary_zone = cv2.dilate(edges, kernel) > 0
    n_spikes = max(1, spike_mask.sum() / 255)
    n_at_boundary = ((spike_mask > 0) & boundary_zone).sum()
    n_inside = ((spike_mask > 0) & (~boundary_zone)).sum()
    pct_at_boundary = float(n_at_boundary / n_spikes * 100)
    pct_inside = float(n_inside / n_spikes * 100)
    jump_results = {
        "n_spike_pixels": int(n_spikes),
        "pct_at_building_boundary": pct_at_boundary,
        "pct_inside_or_flat_areas": pct_inside,
    }
    print(f"  {PRIMARY}: {int(n_spikes)} spike pixels")
    print(f"    At building boundaries: {pct_at_boundary:.1f}%")
    print(f"    Inside / flat areas:    {pct_inside:.1f}%")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(Z, cmap='terrain'); axes[0].set_title("DSM (Z)")
    axes[1].imshow(spike_mask, cmap='Reds'); axes[1].set_title("Spike pixels (Otsu)")
    axes[2].imshow(boundary_zone, cmap='Blues', alpha=0.5)
    axes[2].imshow(spike_mask, cmap='Reds', alpha=0.4)
    axes[2].set_title(f"Overlap — boundary={pct_at_boundary:.0f}%  inside={pct_inside:.0f}%")
    plt.suptitle(f"Jump Location — {PRIMARY}", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_jump_location.png", dpi=120)
    plt.close()

results["jump_analysis"] = jump_results

# ══════════════════════════════════════════════════════════════════════════════
# 5+6. MESH A/B + TEXTURE A/B TEST
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5+6] Mesh A/B + Texture A/B")
dsm_path = TILES[PRIMARY]
rgb_path  = RGB_FILES[PRIMARY]
if dsm_path.exists():
    Z, transform = load_dsm(dsm_path)
    rgb_img = load_rgb(rgb_path) if rgb_path.exists() else None
    if rgb_img is not None and (rgb_img.shape[0] != Z.shape[0] or rgb_img.shape[1] != Z.shape[1]):
        rgb_img = cv2.resize(rgb_img, (Z.shape[1], Z.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Mesh A — raw
    mesh_A = build_mesh(Z, transform)
    render_mesh(mesh_A, FIG_DIR / "04_meshA_elevation.png", title="MESH A — Raw DSM / Elevation Colour")
    if rgb_img is not None:
        render_mesh(mesh_A, FIG_DIR / "05_meshA_RGB_texture.png", rgb_img=rgb_img, title="MESH A — Raw DSM / RGB Texture")
    print("  Mesh A rendered.")

    # Mesh B — VIZ-ONLY smooth
    Z_viz = smooth_viz(Z, sigma_px=2)
    mesh_B = build_mesh(Z_viz, transform)
    mesh_B['Elevation'] = Z_viz.ravel().astype(np.float32)
    mesh_B.set_active_scalars('Elevation')
    render_mesh(mesh_B, FIG_DIR / "06_meshB_vizsmooth_elevation.png", title="MESH B — VIZ-ONLY SMOOTH / Elevation Colour")
    if rgb_img is not None:
        render_mesh(mesh_B, FIG_DIR / "07_meshB_vizsmooth_RGB_texture.png", rgb_img=rgb_img, title="MESH B — VIZ-ONLY SMOOTH / RGB Texture")
    print("  Mesh B (viz-smooth) rendered.")

# ══════════════════════════════════════════════════════════════════════════════
# 7. FLAT CONTROL
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7] Flat control")
if dsm_path.exists():
    Z_flat = np.full_like(Z, 100.0)
    mesh_flat = build_mesh(Z_flat, transform)
    render_mesh(mesh_flat, FIG_DIR / "08_flat_control_elevation.png", title="FLAT CONTROL Z=100m / Elevation")
    if rgb_img is not None:
        render_mesh(mesh_flat, FIG_DIR / "09_flat_control_RGB_texture.png", rgb_img=rgb_img, title="FLAT CONTROL Z=100m / RGB Texture")
    print("  Flat control rendered.")

# ══════════════════════════════════════════════════════════════════════════════
# 8. RESOLUTION TEST
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8] Resolution downsample test")
if dsm_path.exists():
    from rasterio.transform import Affine
    for factor, label in [(1, "1x_full"), (2, "2x_down"), (4, "4x_down")]:
        h, w = Z.shape
        if factor > 1:
            Z_d = cv2.resize(Z, (w // factor, h // factor), interpolation=cv2.INTER_AREA)
            t = transform
            t_d = Affine(t.a * factor, t.b, t.c, t.d, t.e * factor, t.f)
        else:
            Z_d, t_d = Z, transform
        mesh_d = build_mesh(Z_d, t_d)
        render_mesh(mesh_d, FIG_DIR / f"10_resolution_{label}.png",
                    title=f"Resolution — {label} ({Z_d.shape[0]}x{Z_d.shape[1]})")
        print(f"  {label} rendered ({Z_d.shape[0]}x{Z_d.shape[1]})")

# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[9] Diagnosis")
primary_gr = gradient_results.get(PRIMARY, {})
primary_sp = spike_results.get(PRIMARY, {})
pct_bndry  = jump_results.get("pct_at_building_boundary", 0)
pct_inside = jump_results.get("pct_inside_or_flat_areas", 0)
dx_p99     = primary_gr.get("dx_p99", 0)
spike_p99  = primary_sp.get("p99", 0)
frac_gt_5m = primary_sp.get("frac_gt_5m", 0)
pct_gt_10m = primary_gr.get("pct_gt_10m", 0)

if spike_p99 > 5.0 and frac_gt_5m > 0.01 and pct_inside > 40:
    diagnosis   = "DSM_SURFACE_NOISE"
    evidence    = (f"spike_p99={spike_p99:.2f}m; frac>5m={frac_gt_5m:.3f}; "
                   f"{pct_inside:.0f}% of spike pixels are inside/flat areas (not boundaries). "
                   "DSM carries large random high-frequency noise independent of building structure.")
    next_action = "VISUALIZATION_ONLY_SMOOTHING"
elif spike_p99 > 2.0 and pct_bndry > 40:
    diagnosis   = "MIXED_DSM_AND_MESH"
    evidence    = (f"spike_p99={spike_p99:.2f}m; {pct_bndry:.0f}% of spikes at building boundaries; "
                   f"dx_p99={dx_p99:.1f}m. Abrupt height discontinuities at building edges cause "
                   "StructuredGrid to construct near-vertical faces (curtain artifacts). "
                   "The DSM is not globally noisy but abrupt edges are amplified into visual walls by bilinear mesh construction.")
    next_action = "REPAIR_MESH_PIPELINE"
elif dx_p99 > 10.0:
    diagnosis   = "DSM_SURFACE_NOISE"
    evidence    = f"P99 horizontal gradient={dx_p99:.1f}m; {pct_gt_10m:.1f}% adjacent pairs differ >10m."
    next_action = "VISUALIZATION_ONLY_SMOOTHING"
else:
    diagnosis   = "MESH_TRIANGULATION_ARTIFACT"
    evidence    = "DSM gradients moderate; curtains arise from StructuredGrid face orientation at edges."
    next_action = "REPAIR_MESH_PIPELINE"

diag = {
    "primary_diagnosis": diagnosis,
    "evidence": evidence,
    "recommended_next_action": next_action,
    "key_numbers": {
        "spike_p99_m": spike_p99,
        "frac_spike_gt5m": frac_gt_5m,
        "dx_p99_m": dx_p99,
        "pct_neighbors_gt10m": pct_gt_10m,
        "pct_spikes_at_boundary": pct_bndry,
        "pct_spikes_inside_flat": pct_inside,
    }
}
results["diagnosis"] = diag

print(f"\n  PRIMARY DIAGNOSIS : {diagnosis}")
print(f"  NEXT ACTION       : {next_action}")
print(f"  EVIDENCE          : {evidence}")

# Save outputs
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

report = f"""# Phase 31C — 3D Reconstruction Integrity Audit

## Primary Tile: {PRIMARY}

## 2. DSM Gradient Summary

| Metric | X-direction | Y-direction |
|--------|------------|------------|
| Median | {primary_gr.get('dx_median',0):.3f} m | {primary_gr.get('dy_median',0):.3f} m |
| P95 | {primary_gr.get('dx_p95',0):.2f} m | {primary_gr.get('dy_p95',0):.2f} m |
| P99 | {primary_gr.get('dx_p99',0):.2f} m | {primary_gr.get('dy_p99',0):.2f} m |
| Max | {primary_gr.get('dx_max',0):.2f} m | {primary_gr.get('dy_max',0):.2f} m |

| |ΔZ| Threshold | % Neighboring Pairs |
|---|---|
| > 1 m | {primary_gr.get('pct_gt_1m',0):.1f}% |
| > 2 m | {primary_gr.get('pct_gt_2m',0):.1f}% |
| > 5 m | {primary_gr.get('pct_gt_5m',0):.1f}% |
| > 10 m | {primary_gr.get('pct_gt_10m',0):.1f}% |

## 3. Spike / Outlier Summary

| Metric | Value |
|--------|-------|
| P95 | {primary_sp.get('p95',0):.2f} m |
| P99 | {primary_sp.get('p99',0):.2f} m |
| Max | {primary_sp.get('max',0):.2f} m |
| Fraction > 2m | {primary_sp.get('frac_gt_2m',0):.4f} |
| Fraction > 5m | {primary_sp.get('frac_gt_5m',0):.4f} |
| Fraction > 10m | {primary_sp.get('frac_gt_10m',0):.6f} |

## 4. Jump Location Analysis

| Location | % of Spike Pixels |
|----------|------------------|
| At building boundaries (5px edge zone) | {pct_bndry:.1f}% |
| Inside / flat areas | {pct_inside:.1f}% |

## 5. Primary Diagnosis

**`{diagnosis}`**

### Evidence
{evidence}

## 6. Recommended Next Action

**`{next_action}`**

> STOP. Do NOT implement the next action in this phase.
"""
with open(OUT_DIR / "REPORT.md", "w", encoding="utf-8") as f:
    f.write(report)

print(f"\n✅ Saved: {OUT_DIR}/results.json")
print(f"✅ Saved: {OUT_DIR}/REPORT.md")
print(f"✅ Figures in: {FIG_DIR}/")
print("Phase 31C audit complete.")
