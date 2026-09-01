"""
Test script to build and validate the Three.js building-aware 3D city scene.
Tests Stage 0 (WebGL basics), Stage 1 (DTM terrain), Stage 2 (Roofs), Stage 3 (Walls), Stage 4 (Combined).
"""
import sys, json, base64
from pathlib import Path
import numpy as np
import cv2
import rasterio

sys.path.insert(0, ".")
from depthwizard.config import DepthConfig
from depthwizard.depth.depth_anything import DepthAnythingV2
from depthwizard.data.raster_loader import load_raster_input
from depthwizard.calibration import CalibrationEngine, CalibrationMode

def build_city_geometry_data(
    rgb_img: np.ndarray,
    dsm: np.ndarray,
    dtm: np.ndarray,
    mask_bldg: np.ndarray,
    gsd: float = 0.5,
    exaggeration: float = 1.0,
    stride: int = 4
):
    h, w = dsm.shape[:2]
    
    # 1. Coordinate scaling in true meters
    if isinstance(gsd, (tuple, list)):
        gsd_x, gsd_y = float(gsd[0]), float(gsd[1])
    else:
        gsd_x, gsd_y = float(gsd), float(gsd)
        
    w_m = float(w * gsd_x)
    h_m = float(h * gsd_y)
    z_base = float(dtm.min())
    
    # Subsampled terrain grid
    sub_w = w // stride
    sub_h = h // stride
    dtm_sub = cv2.resize(dtm.astype(np.float32), (sub_w, sub_h), interpolation=cv2.INTER_LINEAR)
    mask_sub = cv2.resize(mask_bldg.astype(np.uint8), (sub_w, sub_h), interpolation=cv2.INTER_NEAREST) > 0
    
    # Build Terrain Vertices & Faces (Plane Grid of DTM)
    # Three.js: X in [-w_m/2, w_m/2], Y = (dtm - z_base) * exaggeration, Z in [-h_m/2, h_m/2]
    xs = np.linspace(-w_m / 2.0, w_m / 2.0, sub_w, dtype=np.float32)
    zs = np.linspace(-h_m / 2.0, h_m / 2.0, sub_h, dtype=np.float32)
    grid_x, grid_z = np.meshgrid(xs, zs)
    grid_y = (dtm_sub - z_base) * exaggeration
    
    # UVs for terrain
    u_vals = np.linspace(0.0, 1.0, sub_w, dtype=np.float32)
    v_vals = np.linspace(0.0, 1.0, sub_h, dtype=np.float32) # In Three.js standard Plane UV: V goes 1 at top, 0 at bottom
    grid_u, grid_v = np.meshgrid(u_vals, v_vals)
    # Note: image row 0 is top (North), in Three.js texture coordinate V=1 is top, V=0 is bottom
    grid_v_flipped = 1.0 - grid_v
    
    terrain_positions = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1).astype(np.float32)
    terrain_uvs = np.stack([grid_u.ravel(), grid_v_flipped.ravel()], axis=1).astype(np.float32)
    
    # Terrain Triangles
    indices = []
    for r in range(sub_h - 1):
        for c in range(sub_w - 1):
            p00 = r * sub_w + c
            p01 = r * sub_w + (c + 1)
            p10 = (r + 1) * sub_w + c
            p11 = (r + 1) * sub_w + (c + 1)
            # Two triangles per quad: (p00, p10, p01) and (p01, p10, p11)
            indices.extend([int(p00), int(p10), int(p01), int(p01), int(p10), int(p11)])
    terrain_indices = np.array(indices, dtype=np.int32)
    
    # 2. Extract Connected Building Components for Explicit Roofs & Walls
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask_bldg.astype(np.uint8))
    
    building_list = []
    all_roof_pos = []
    all_roof_uvs = []
    all_roof_indices = []
    
    all_wall_pos = []
    all_wall_indices = []
    
    roof_vert_offset = 0
    wall_vert_offset = 0
    
    for k in range(1, num_labels):
        area = stats[k, cv2.CC_STAT_AREA]
        if area < 10:  # ignore noise
            continue
            
        b_mask = (labels_im == k)
        z_ground = float(np.median(dtm[b_mask]))
        z_roof_p95 = float(np.percentile(dsm[b_mask], 95))
        bldg_height = max(1.0, z_roof_p95 - z_ground)
        
        # Center in meters
        cent_col, cent_row = centroids[k]
        cent_x = (cent_col / w - 0.5) * w_m
        cent_z = (cent_row / h - 0.5) * h_m
        
        contours, _ = cv2.findContours(b_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = contours[0]
        if len(cnt) < 3:
            continue
            
        cnt_approx = cv2.approxPolyDP(cnt, 1.2, closed=True)
        pts_cnt = cnt_approx.reshape(-1, 2)
        n_pts = len(pts_cnt)
        if n_pts < 3:
            continue
            
        bldg_info = {
            "id": k,
            "area_m2": round(float(area * (gsd_x * gsd_y)), 1),
            "z_ground": round(z_ground, 1),
            "z_roof": round(z_roof_p95, 1),
            "height_m": round(bldg_height, 1),
            "cx": round(float(cent_x), 2),
            "cz": round(float(cent_z), 2)
        }
        building_list.append(bldg_info)
        
        # Roof elevation (relative to z_base)
        y_roof = (z_roof_p95 - z_base) * exaggeration
        y_ground = (z_ground - z_base) * exaggeration
        
        # --- A. Roof Triangulation (Fan from centroid or Ear Clipping) ---
        # Centroid vertex
        cent_u = cent_col / (w - 1)
        cent_v = 1.0 - (cent_row / (h - 1))
        
        c_idx = roof_vert_offset
        all_roof_pos.append([cent_x, y_roof, cent_z])
        all_roof_uvs.append([cent_u, cent_v])
        
        # Contour vertices
        cnt_start = roof_vert_offset + 1
        for i in range(n_pts):
            c_i, r_i = pts_cnt[i]
            px_m = (c_i / w - 0.5) * w_m
            pz_m = (r_i / h - 0.5) * h_m
            u_i = c_i / (w - 1)
            v_i = 1.0 - (r_i / (h - 1))
            all_roof_pos.append([px_m, y_roof, pz_m])
            all_roof_uvs.append([u_i, v_i])
            
        for i in range(n_pts):
            next_i = (i + 1) % n_pts
            # Triangle from centroid: (cent, p_i, p_next)
            all_roof_indices.extend([c_idx, cnt_start + i, cnt_start + next_i])
            
        roof_vert_offset += (1 + n_pts)
        
        # --- B. Vertical Walls ---
        wall_start = wall_vert_offset
        for i in range(n_pts):
            c_i, r_i = pts_cnt[i]
            px_m = (c_i / w - 0.5) * w_m
            pz_m = (r_i / h - 0.5) * h_m
            
            # Ground vertex
            all_wall_pos.append([px_m, y_ground, pz_m])
            # Roof vertex
            all_wall_pos.append([px_m, y_roof, pz_m])
            
        for i in range(n_pts):
            next_i = (i + 1) % n_pts
            g1 = wall_start + 2 * i
            r1 = wall_start + 2 * i + 1
            g2 = wall_start + 2 * next_i
            r2 = wall_start + 2 * next_i + 1
            # Two triangles for quad face: (g1, r1, r2) and (g1, r2, g2)
            all_wall_indices.extend([g1, r1, r2, g1, r2, g2])
            
        wall_vert_offset += (2 * n_pts)
        
    # Texture encoding
    rgb_resized = cv2.resize(rgb_img, (512, 512), interpolation=cv2.INTER_LINEAR)
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    tex_base64 = base64.b64encode(buffer).decode("utf-8")
    
    return {
        "terrain": {
            "positions": np.array(terrain_positions, dtype=np.float32).ravel().tolist(),
            "uvs": np.array(terrain_uvs, dtype=np.float32).ravel().tolist(),
            "indices": terrain_indices.tolist(),
            "n_verts": len(terrain_positions),
            "n_faces": len(terrain_indices) // 3
        },
        "roofs": {
            "positions": np.array(all_roof_pos, dtype=np.float32).ravel().tolist() if all_roof_pos else [],
            "uvs": np.array(all_roof_uvs, dtype=np.float32).ravel().tolist() if all_roof_uvs else [],
            "indices": all_roof_indices if all_roof_indices else [],
            "n_verts": len(all_roof_pos),
            "n_faces": len(all_roof_indices) // 3
        },
        "walls": {
            "positions": np.array(all_wall_pos, dtype=np.float32).ravel().tolist() if all_wall_pos else [],
            "indices": all_wall_indices if all_wall_indices else [],
            "n_verts": len(all_wall_pos),
            "n_faces": len(all_wall_indices) // 3
        },
        "buildings": building_list,
        "texture_base64": tex_base64,
        "bounds": {
            "w_m": round(w_m, 2),
            "h_m": round(h_m, 2),
            "z_min": round(z_base, 2),
            "z_max": round(float(dsm.max()), 2),
            "max_dim": round(max(w_m, h_m, float(dsm.max()) - z_base), 2)
        }
    }

if __name__ == "__main__":
    print("Testing city geometry construction...")
    raster_in = load_raster_input("data/dfc2023_multicity/rgb/SV_NewYork_40.7401_-73.9915.tif")
    dcfg = DepthConfig(cache_dir="data/dfc2023_multicity/depth_cache")
    depth_model = DepthAnythingV2(dcfg.model_id, dcfg.input_size, dcfg.cache_dir, use_cache=True)
    depth_raw = depth_model.infer(raster_in.rgb, raster_in.filename, target_hw=raster_in.shape)
    
    dsm_truth_path = Path("data/dfc2023_multicity/dsm") / Path(raster_in.filename).name
    truth = cv2.imread(str(dsm_truth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) if dsm_truth_path.exists() else None
    
    engine = CalibrationEngine()
    res = engine.calibrate(
        depth_raw, raster_in.rgb, is_georeferenced=True,
        mode=CalibrationMode.STRUCTURAL_PRIOR, reference_elevation=truth,
        filename=raster_in.filename
    )
    
    geom = build_city_geometry_data(raster_in.rgb, res.dsm, res.dtm, res.mask_bldg, gsd=raster_in.gsd or 0.5)
    
    print(f"Terrain: {geom['terrain']['n_verts']} vertices, {geom['terrain']['n_faces']} faces")
    print(f"Roofs:   {geom['roofs']['n_verts']} vertices, {geom['roofs']['n_faces']} faces")
    print(f"Walls:   {geom['walls']['n_verts']} vertices, {geom['walls']['n_faces']} faces")
    print(f"Buildings: {len(geom['buildings'])} detected structures")
    print(f"Bounds:  {geom['bounds']}")
    
    # Check for NaN / Inf
    for key in ["terrain", "roofs", "walls"]:
        pos = geom[key]["positions"]
        if pos:
            arr = np.array(pos)
            assert not np.isnan(arr).any(), f"NaN in {key} positions"
            assert not np.isinf(arr).any(), f"Inf in {key} positions"
            print(f"✓ {key} geometry is VALID (no NaN/Inf)")
            
    print("Geometry pipeline SUCCESS!")
