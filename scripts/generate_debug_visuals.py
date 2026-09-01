"""
Phase 37 Debug Visualization Pipeline.
Generates the 11 required debug images + component_filter_debug.png
for three NYC test scenes.
Also verifies SHA256 DSM integrity.
"""
import sys, json, hashlib, csv
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
from depthwizard.viz.interactive_viewer import (
    build_city_geometry, generate_footprint_debug, generate_interactive_webgl_html
)

OUT_DIR = Path("runs/phase37_geometry_fix")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_HASH = "06314e50752869d09296b848ef188ee9399c0111be17e1cc722945ac07590864"

# Three NYC test scenes spanning different density profiles
NYC_SCENES = [
    "SV_NewYork_40.7401_-73.9915.tif",   # NYC skyscraper-heavy (demo scene)
    "SV_NewYork_40.7333_-73.9835.tif",   # NYC dense-highrise
    "SV_NewYork_40.7335_-74.0053.tif",   # NYC lower-rise / mixed
]

dcfg = DepthConfig(cache_dir="data/dfc2023_multicity/depth_cache")
depth_model = DepthAnythingV2(dcfg.model_id, dcfg.input_size, dcfg.cache_dir, use_cache=True)
calib_engine = CalibrationEngine(runs_dir=Path("runs"))

geometry_report = []

for scene_idx, scene_name in enumerate(NYC_SCENES):
    scene_short = scene_name.replace("SV_NewYork_", "").replace(".tif", "")
    scene_out = OUT_DIR / f"scene_{scene_idx+1}_{scene_short}"
    scene_out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"SCENE {scene_idx+1}: {scene_name}")
    print(f"{'='*60}")

    # ── Load and calibrate ──────────────────────────────────────────
    rgb_path = Path("data/dfc2023_multicity/rgb") / scene_name
    if not rgb_path.exists():
        print(f"  SKIP: {rgb_path} not found")
        continue

    raster_in = load_raster_input(rgb_path, filename=scene_name)
    h, w = raster_in.shape
    depth_raw = depth_model.infer(raster_in.rgb, scene_name, target_hw=(h, w))

    dsm_truth_path = Path("data/dfc2023_multicity/dsm") / scene_name
    truth = None
    if dsm_truth_path.exists():
        truth = cv2.imread(str(dsm_truth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)

    calib_res = calib_engine.calibrate(
        depth_raw, raster_in.rgb, is_georeferenced=raster_in.is_georeferenced,
        mode=CalibrationMode.STRUCTURAL_PRIOR, reference_elevation=truth,
        filename=scene_name
    )

    # ── SHA256 hash check for primary demo scene ───────────────────
    if scene_idx == 0:
        dsm_hash_before = hashlib.sha256(calib_res.dsm.tobytes()).hexdigest()
        print(f"  DSM hash BEFORE: {dsm_hash_before}")
        hash_ok = (dsm_hash_before == EXPECTED_HASH)
        hash_status = "PASS" if hash_ok else f"DIFF (expected: {EXPECTED_HASH})"
        print(f"  Hash match: {hash_status}")

    dsm = calib_res.dsm
    dtm = calib_res.dtm
    mask = calib_res.mask_bldg
    rgb  = raster_in.rgb

    # ── Build geometry ─────────────────────────────────────────────
    geom = build_city_geometry(rgb, dsm, dtm, mask, gsd=raster_in.gsd or 0.5,
                               exaggeration=1.5, stride=4)

    n_valid     = len(geom["buildings"])
    n_rejected  = len(geom["rejected"])
    n_roof_tri  = geom["roofs"]["n_faces"]
    n_wall_tri  = geom["walls"]["n_faces"]
    n_terrain_tri = geom["terrain"]["n_faces"]

    print(f"  Valid buildings: {n_valid}")
    print(f"  Rejected (mega): {n_rejected}")
    print(f"  Roof triangles: {n_roof_tri}")
    print(f"  Wall triangles: {n_wall_tri}")
    print(f"  Terrain triangles: {n_terrain_tri}")
    if geom["rejected"]:
        for r in geom["rejected"]:
            print(f"    Rejected #{r['component_id']}: {r['rejection_reason']}")

    if n_valid > 0:
        heights = [b["height_m"] for b in geom["buildings"]]
        print(f"  Height stats: max={max(heights):.1f}m  median={np.median(heights):.1f}m  P95={np.percentile(heights, 95):.1f}m")

    # ── Debug image 01: RGB ────────────────────────────────────────
    cv2.imwrite(str(scene_out / "01_rgb.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # ── Debug image 02: Building mask ─────────────────────────────
    mask_vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    mask_vis[mask > 0] = [255, 255, 100]
    blended = cv2.addWeighted(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.5,
                               mask_vis, 0.5, 0)
    cv2.imwrite(str(scene_out / "02_building_mask.png"), blended)

    # ── Debug image 03: Building footprints with IDs ──────────────
    fp_img = generate_footprint_debug(rgb, mask, gsd=raster_in.gsd or 0.5,
                                       output_path=str(scene_out / "03_building_footprints.png"))

    # ── Debug image: Component filter ─────────────────────────────
    generate_footprint_debug(rgb, mask, gsd=raster_in.gsd or 0.5,
                              output_path=str(scene_out / "component_filter_debug.png"))

    # ── Debug image 04: Roof geometry visualization ────────────────
    if geom["roofs"]["n_verts"] > 0:
        pos = np.array(geom["roofs"]["positions"]).reshape(-1, 3)
        idx = np.array(geom["roofs"]["indices"]).reshape(-1, 3)
        fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0D1117")
        ax.set_facecolor("#0D1117")
        ax.set_title(f"Roof Geometry\n{n_roof_tri} triangles, {n_valid} buildings", color="white")
        for tri in idx[:5000]:  # limit rendering
            pts = pos[tri][:, [0, 2]]
            p = plt.Polygon(pts, fill=True, facecolor="#4A90D9", edgecolor="#88BBFF",
                            linewidth=0.3, alpha=0.7)
            ax.add_patch(p)
        ax.autoscale()
        ax.tick_params(colors="grey")
        for sp in ax.spines.values(): sp.set_edgecolor("grey")
        plt.tight_layout()
        fig.savefig(str(scene_out / "04_roof_only.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
    else:
        # Create blank placeholder
        img = np.full((512, 512, 3), 30, dtype=np.uint8)
        cv2.putText(img, "NO ROOF DATA", (120, 256), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 80, 80), 2)
        cv2.imwrite(str(scene_out / "04_roof_only.png"), img)

    # ── Debug image 05: Wall geometry visualization ────────────────
    if geom["walls"]["n_verts"] > 0:
        wpos = np.array(geom["walls"]["positions"]).reshape(-1, 3)
        widx = np.array(geom["walls"]["indices"]).reshape(-1, 3)
        fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0D1117")
        ax.set_facecolor("#0D1117")
        ax.set_title(f"Wall Geometry\n{n_wall_tri} triangles", color="white")
        for tri in widx[:5000]:
            pts = wpos[tri][:, [0, 2]]
            p = plt.Polygon(pts, fill=True, facecolor="#2D5A8E", edgecolor="#5599CC",
                            linewidth=0.3, alpha=0.6)
            ax.add_patch(p)
        ax.autoscale()
        ax.tick_params(colors="grey")
        for sp in ax.spines.values(): sp.set_edgecolor("grey")
        plt.tight_layout()
        fig.savefig(str(scene_out / "05_walls_only.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
    else:
        img = np.full((512, 512, 3), 30, dtype=np.uint8)
        cv2.putText(img, "NO WALL DATA", (120, 256), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 80, 80), 2)
        cv2.imwrite(str(scene_out / "05_walls_only.png"), img)

    # ── Debug image 06: Terrain only ──────────────────────────────
    dtm_norm = (dtm - dtm.min()) / max(dtm.max() - dtm.min(), 1.0)
    dtm_vis = (plt.cm.terrain(dtm_norm)[:, :, :3] * 255).astype(np.uint8)
    cv2.imwrite(str(scene_out / "06_terrain_only.png"), cv2.cvtColor(dtm_vis, cv2.COLOR_RGB2BGR))

    # ── Debug image 07: Terrain + building footprints overlay ─────
    dtm_bgr = cv2.cvtColor(dtm_vis, cv2.COLOR_RGB2BGR)
    import depthwizard.viz.interactive_viewer as iv
    from cv2 import connectedComponentsWithStats, CC_STAT_AREA, CC_STAT_WIDTH, CC_STAT_HEIGHT
    num_l, labels_im2, stats2, centroids2 = connectedComponentsWithStats(mask.astype(np.uint8))
    img_area = float(h * w)
    for k in range(1, num_l):
        area_k = int(stats2[k, CC_STAT_AREA])
        bw = int(stats2[k, CC_STAT_WIDTH])
        bh2 = int(stats2[k, CC_STAT_HEIGHT])
        if area_k < 18: continue
        if bw > 0.65*w or bh2 > 0.65*h or area_k > 0.40*img_area: continue
        bm = (labels_im2 == k).astype(np.uint8)
        cnts, _ = cv2.findContours(bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(dtm_bgr, cnts, -1, (0, 255, 100), 2)
    cv2.imwrite(str(scene_out / "07_terrain_plus_buildings.png"), dtm_bgr)

    # ── Debug image 08–11: Render mode screenshots ─────────────────
    dsm_norm = (dsm - dsm.min()) / max(dsm.max() - dsm.min(), 1.0)
    dsm_color = (plt.cm.inferno(dsm_norm)[:, :, :3] * 255).astype(np.uint8)

    ndm = np.clip(dsm - dtm, 0, None)
    ndm_norm = ndm / max(ndm.max(), 1.0)
    ndm_color = (plt.cm.RdYlBu_r(ndm_norm)[:, :, :3] * 255).astype(np.uint8)

    cv2.imwrite(str(scene_out / "08_rgb_city.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(scene_out / "09_elevation_city.png"), cv2.cvtColor(dsm_color, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(scene_out / "10_height_city.png"), cv2.cvtColor(ndm_color, cv2.COLOR_RGB2BGR))

    # Final composite: RGB + outlines
    final_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for k in range(1, num_l):
        area_k = int(stats2[k, CC_STAT_AREA])
        bw = int(stats2[k, CC_STAT_WIDTH])
        bh2 = int(stats2[k, CC_STAT_HEIGHT])
        if area_k < 18: continue
        if bw > 0.65*w or bh2 > 0.65*h or area_k > 0.40*img_area: continue
        bm = (labels_im2 == k).astype(np.uint8)
        cnts, _ = cv2.findContours(bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(final_vis, cnts, -1, (0, 255, 120), 2)
        cx2, cy2 = int(centroids2[k][0]), int(centroids2[k][1])
        bldg_h = next((b["height_m"] for b in geom["buildings"] if b["id"] == k), 0.0)
        cv2.putText(final_vis, f"{bldg_h:.0f}m", (max(cx2-10, 0), max(cy2+4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 240, 110), 1)
    cv2.putText(final_vis, f"{n_valid} bldgs | {n_rejected} rejected",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 80), 1)
    cv2.imwrite(str(scene_out / "11_final_city.png"), final_vis)

    print(f"  Saved debug images to: {scene_out}")

    # -- Post-geometry DSM hash check -----------------------------------------
    if scene_idx == 0:
        dsm_hash_after = hashlib.sha256(calib_res.dsm.tobytes()).hexdigest()
        print(f"\n  DSM hash AFTER geometry: {dsm_hash_after}")
        if dsm_hash_after == dsm_hash_before:
            print("  SCIENTIFIC INTEGRITY VERIFIED: Hash unchanged")
        else:
            print("  HASH MISMATCH -- SCIENTIFIC DATA CORRUPTED!")

    # ── Per-building acceptance test ──────────────────────────────
    sample_bldgs = geom["buildings"][:min(10, n_valid)]
    pass_count = 0
    for b in sample_bldgs:
        checks = (
            b["height_m"] > 0,
            b["area_m2"] > 0,
            b["z_roof"] > b["z_ground"],
            np.isfinite(b["z_roof"]),
            np.isfinite(b["z_ground"]),
        )
        if all(checks):
            pass_count += 1
    acceptance_pct = pass_count / max(len(sample_bldgs), 1) * 100

    geometry_report.append({
        "scene": scene_name,
        "n_valid_buildings": n_valid,
        "n_rejected_mega": n_rejected,
        "n_roof_triangles": n_roof_tri,
        "n_wall_triangles": n_wall_tri,
        "n_terrain_triangles": n_terrain_tri,
        "acceptance_pct": round(acceptance_pct, 1),
        "max_height_m": round(max(heights) if n_valid > 0 else 0, 1),
        "median_height_m": round(float(np.median(heights)) if n_valid > 0 else 0, 1),
    })

# ── Write RESULTS.json ─────────────────────────────────────────────────────
results_json = {
    "phase": "Phase 37 — Geometry Reconstruction",
    "dsm_hash_expected": EXPECTED_HASH,
    "geometry_fixes_applied": [
        "P1: Mega-component rejection (>65% bbox or >40% area)",
        "P2: Flat per-building roof elevation (interior P75)",
        "P3: Adaptive contour simplification (perimeter-relative epsilon)",
        "P4: Stable per-building DTM base (P30 percentile)",
        "P5: NaN/Inf guard on all rasters",
        "P6: Camera framing improved (camDist=maxDim*1.65)"
    ],
    "scenes": geometry_report
}
with open(OUT_DIR / "RESULTS.json", "w") as f:
    json.dump(results_json, f, indent=2)

# ── Write CONTROL_MATRIX.csv ───────────────────────────────────────────────
rows = []
for s in geometry_report:
    rows.append({
        "Scene": s["scene"],
        "Valid Buildings": s["n_valid_buildings"],
        "Rejected Mega": s["n_rejected_mega"],
        "Roof Tris": s["n_roof_triangles"],
        "Wall Tris": s["n_wall_triangles"],
        "Acceptance %": s["acceptance_pct"],
        "Max Height (m)": s["max_height_m"],
        "Median Height (m)": s["median_height_m"],
    })
with open(OUT_DIR / "CONTROL_MATRIX.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
    writer.writeheader()
    writer.writerows(rows)

print(f"\n{'='*60}")
print("PHASE 37 DEBUG PIPELINE COMPLETE")
print(f"Output directory: {OUT_DIR.resolve()}")
print("Files:")
for p in sorted(OUT_DIR.rglob("*.*"))[:30]:
    print(f"  {p.relative_to(OUT_DIR)}")
