"""
Three.js WebGL Building-Aware 3D City Flythrough & Interactive Viewer for DepthWizard.
Reconstructs an explicit 3-layer architectural city model:
  Layer 1: DTM Terrain with satellite RGB orthophoto
  Layer 2: DSM Building Roofs with robust Ear-Clipping triangulation & orthophoto UVs
  Layer 3: Vertical Architectural Facades connecting local DTM ground to DSM roof perimeters

Features:
- Instant client-side Mode Switcher: RGB City, Elevation Colormap, Building Height, Terrain Slope
- True 60fps OrbitControls, Pan, Zoom, WASD/Arrows First-Person Free-Flight, Shift speed boost
- Automated Cinematic Flythrough along sinusoidal orbital trajectory
- Interactive Raycasting Building Picker & Live Inspector HUD
- Responsive canvas resizing and camera framing with scene bounding box calculation
"""

import json
import base64
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2

def is_point_in_triangle(p, a, b, c):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def triangulate_polygon_earcut(pts: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Robust 2D Ear Clipping triangulation for arbitrary simple polygons (convex & concave).
    pts: Nx2 array of (x, y) coordinates.
    Returns: list of (i0, i1, i2) index triplets.
    """
    n = len(pts)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]
        
    # Ensure Counter-Clockwise winding
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += float(pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1])
    
    indices = list(range(n))
    if area < 0:
        indices.reverse()
        
    triangles = []
    max_iters = n * n
    iters = 0
    
    while len(indices) > 3 and iters < max_iters:
        iters += 1
        ear_found = False
        m = len(indices)
        for i in range(m):
            prev_idx = indices[(i - 1 + m) % m]
            curr_idx = indices[i]
            next_idx = indices[(i + 1) % m]
            
            p_prev = pts[prev_idx]
            p_curr = pts[curr_idx]
            p_next = pts[next_idx]
            
            # Convex check (cross product > 0 for CCW)
            cross = float((p_curr[0] - p_prev[0]) * (p_next[1] - p_curr[1]) - (p_curr[1] - p_prev[1]) * (p_next[0] - p_curr[0]))
            if cross <= 1e-7:
                continue
                
            # Check if any other point is inside this triangle
            contains_point = False
            for j in range(m):
                if j in ((i - 1 + m) % m, i, (i + 1) % m):
                    continue
                test_pt = pts[indices[j]]
                if is_point_in_triangle(test_pt, p_prev, p_curr, p_next):
                    contains_point = True
                    break
                    
            if not contains_point:
                triangles.append((prev_idx, curr_idx, next_idx))
                indices.pop(i)
                ear_found = True
                break
                
        if not ear_found:
            # Fallback: pop one triangle to maintain robustness
            triangles.append((indices[0], indices[1], indices[2]))
            indices.pop(1)
            
    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))
        
    return triangles

def turbo_colormap(val: float) -> Tuple[float, float, float]:
    """Smooth perceptual turbo colormap for elevation & height visualization."""
    v = np.clip(val, 0.0, 1.0)
    # Polynomial approximation of Turbo colormap
    r = np.clip(0.1357 + v * (4.5974 + v * (-42.3277 + v * (130.5887 + v * (-150.5668 + v * 58.1375)))), 0.0, 1.0)
    g = np.clip(0.0914 + v * (2.1856 + v * (4.8052 + v * (-14.0195 + v * (4.2109 + v * 2.7747)))), 0.0, 1.0)
    b = np.clip(0.1067 + v * (12.5856 + v * (-67.6534 + v * (161.6405 + v * (-184.4746 + v * 78.4314)))), 0.0, 1.0)
    return float(r), float(g), float(b)

def generate_footprint_debug(
    rgb_img: np.ndarray,
    mask_bldg: np.ndarray,
    gsd: Any = 0.5,
    output_path: str = None
) -> np.ndarray:
    """
    Debug visualizer: RGB + building contours + IDs.
    Green = valid building, Red = rejected mega-component.
    Returns RGB numpy array.
    """
    h, w = mask_bldg.shape[:2]
    rgb_h, rgb_w = rgb_img.shape[:2]
    vis = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR) if rgb_img.ndim == 3 else rgb_img.copy()
    if rgb_h != h or rgb_w != w:
        vis = cv2.resize(vis, (w, h), interpolation=cv2.INTER_LINEAR)
    overlay = vis.copy()

    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
        mask_bldg.astype(np.uint8)
    )
    image_area = float(h * w)
    valid_count = 0
    rejected_count = 0

    for k in range(1, num_labels):
        area = int(stats[k, cv2.CC_STAT_AREA])
        bb_w = int(stats[k, cv2.CC_STAT_WIDTH])
        bb_h = int(stats[k, cv2.CC_STAT_HEIGHT])
        if area < 15:
            continue
        is_mega = (bb_w > int(0.65 * w) or bb_h > int(0.65 * h) or area > int(0.40 * image_area))
        b_mask = (labels_im == k).astype(np.uint8)
        contours, _ = cv2.findContours(b_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = (0, 0, 210) if is_mega else (20, 210, 40)
        cv2.drawContours(overlay, contours, -1, color, 2)
        cx, cy = int(centroids[k][0]), int(centroids[k][1])
        lbl = "X" if is_mega else str(k)
        cv2.putText(overlay, lbl, (max(cx - 8, 0), max(cy + 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        if is_mega:
            rejected_count += 1
        else:
            valid_count += 1

    result = cv2.addWeighted(vis, 0.55, overlay, 0.45, 0)
    cv2.rectangle(result, (4, 4), (210, 58), (15, 15, 15), -1)
    cv2.putText(result, f"VALID: {valid_count}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 210, 40), 1)
    cv2.putText(result, f"REJECTED(mega): {rejected_count}", (8, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 210), 1)
    if output_path:
        cv2.imwrite(output_path, result)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def build_city_geometry(
    rgb_img: np.ndarray,
    dsm: np.ndarray,
    dtm: np.ndarray,
    mask_bldg: np.ndarray,
    gsd: Any = 0.5,
    exaggeration: float = 1.0,
    stride: int = 4
) -> Dict[str, Any]:
    """
    Phase 41 — Multi-Evidence Building Instance Reconstruction.

    Root cause fix: The input mask_bldg from the calibration engine is often
    unreliable (picks up roads/shadows as buildings, misses actual buildings).
    This function rebuilds the building evidence from scratch using:
      Evidence 1: nDSM height above terrain (objects >= threshold above ground)
      Evidence 2: DSM morphological top-hat (sharp convex structures above surroundings)
      Evidence 3: RGB flat texture (rooftops have uniform low-variance color)
    These three evidences are fused by majority vote, then individual building
    instances are separated using watershed seeded by distance-transform peaks.
    """
    h, w = dsm.shape[:2]

    # ── 1. Physical meter scaling ──────────────────────────────────────────────
    if isinstance(gsd, (tuple, list)):
        gsd_x, gsd_y = float(gsd[0]), float(gsd[1])
    else:
        gsd_x, gsd_y = float(gsd), float(gsd)
    w_m = float(w * gsd_x)
    h_m = float(h * gsd_y)

    # Guard NaN/Inf in rasters
    dtm_safe = dtm.astype(np.float32).copy()
    dsm_safe = dsm.astype(np.float32).copy()
    dtm_med = float(np.nanmedian(dtm_safe))
    dsm_med = float(np.nanmedian(dsm_safe))
    dtm_safe = np.where(np.isfinite(dtm_safe), dtm_safe, dtm_med)
    dsm_safe = np.where(np.isfinite(dsm_safe), dsm_safe, dsm_med)

    z_base = float(np.percentile(dtm_safe, 2))
    z_max  = float(np.percentile(dsm_safe, 98))
    z_range = max(1.0, z_max - z_base)

    ndsm_safe = np.maximum(0.0, dsm_safe - dtm_safe)

    # ── 2. DTM Terrain Grid (ground surface only) ──────────────────────────────
    sub_w = max(16, w // stride)
    sub_h = max(16, h // stride)
    dtm_sub = cv2.resize(dtm_safe, (sub_w, sub_h), interpolation=cv2.INTER_LINEAR)

    xs = np.linspace(-w_m / 2.0, w_m / 2.0, sub_w, dtype=np.float32)
    zs = np.linspace(-h_m / 2.0, h_m / 2.0, sub_h, dtype=np.float32)
    grid_x, grid_z = np.meshgrid(xs, zs)
    grid_y = np.clip((dtm_sub - z_base) * exaggeration, 0.0, None)

    u_vals = np.linspace(0.0, 1.0, sub_w, dtype=np.float32)
    v_vals = np.linspace(0.0, 1.0, sub_h, dtype=np.float32)
    grid_u, grid_v = np.meshgrid(u_vals, v_vals)
    grid_v_flipped = 1.0 - grid_v

    terrain_positions = np.stack(
        [grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1
    ).astype(np.float32)
    terrain_uvs = np.stack(
        [grid_u.ravel(), grid_v_flipped.ravel()], axis=1
    ).astype(np.float32)

    terrain_elev_norm = np.clip((dtm_sub - z_base) / z_range, 0.0, 1.0)
    terrain_elev_colors: List[float] = []
    for val in terrain_elev_norm.ravel():
        r, g, b = turbo_colormap(float(val))
        terrain_elev_colors.extend([r, g, b])

    dy_g, dx_g = np.gradient(dtm_sub, h_m / sub_h, w_m / sub_w)
    slope_deg = np.degrees(np.arctan(np.sqrt(dx_g**2 + dy_g**2)))
    terrain_slope_colors: List[float] = []
    for deg in slope_deg.ravel():
        val = float(np.clip(deg / 45.0, 0.0, 1.0))
        r = min(1.0, val * 2.0)
        g = min(1.0, (1.0 - val) * 2.0)
        b = 0.12
        terrain_slope_colors.extend([r, g, b])

    terrain_indices_list: List[int] = []
    for ri in range(sub_h - 1):
        for ci in range(sub_w - 1):
            p00 = ri * sub_w + ci
            p01 = ri * sub_w + (ci + 1)
            p10 = (ri + 1) * sub_w + ci
            p11 = (ri + 1) * sub_w + (ci + 1)
            terrain_indices_list.extend([p00, p10, p01, p01, p10, p11])
    terrain_indices = np.array(terrain_indices_list, dtype=np.int32)

    # ── 3. Phase 41 Multi-Evidence Building Mask ───────────────────────────────
    rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    # Evidence 1: nDSM height (objects significantly above terrain)
    ndsm_thresh = max(3.0, float(np.percentile(ndsm_safe[ndsm_safe > 0], 20))) if (ndsm_safe > 0).sum() > 100 else 3.0
    ev_height = (ndsm_safe >= ndsm_thresh).astype(np.uint8)

    # Evidence 2: DSM morphological top-hat (convex structures above local surroundings)
    dsm_smooth_b = cv2.GaussianBlur(dsm_safe, (5, 5), 0)
    kern_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min(51, w // 10 | 1), min(51, h // 10 | 1)))
    dsm_tophat = cv2.morphologyEx(dsm_smooth_b, cv2.MORPH_TOPHAT, kern_big)
    tophat_vals = dsm_tophat[dsm_tophat > 0]
    tophat_thresh = float(np.percentile(tophat_vals, 40)) if tophat_vals.size > 10 else 1.0
    ev_tophat = (dsm_tophat >= tophat_thresh).astype(np.uint8)

    # Evidence 3: RGB flat-texture (rooftops are uniform color patches)
    l_channel = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    kern_local = np.ones((9, 9), np.float32) / 81.0
    l_mean = cv2.filter2D(l_channel, -1, kern_local)
    l_sq_mean = cv2.filter2D(l_channel ** 2, -1, kern_local)
    l_std = np.sqrt(np.maximum(l_sq_mean - l_mean ** 2, 0.0))
    ev_flat = (l_std < float(np.percentile(l_std, 55))).astype(np.uint8)

    # Majority vote: require >= 2 of 3 evidences
    consensus = ev_height.astype(np.int16) + ev_tophat.astype(np.int16) + ev_flat.astype(np.int16)
    build_candidate = (consensus >= 2).astype(np.uint8)

    # Morphological cleanup
    kern3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kern7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    build_candidate = cv2.morphologyEx(build_candidate, cv2.MORPH_OPEN, kern3, iterations=2)
    build_candidate = cv2.morphologyEx(build_candidate, cv2.MORPH_CLOSE, kern7, iterations=2)

    # ── 4. Watershed Instance Segmentation ─────────────────────────────────────
    dsm_smooth = cv2.bilateralFilter(dsm_safe, d=9, sigmaColor=3.0, sigmaSpace=3.0)
    dist = cv2.distanceTransform(build_candidate, cv2.DIST_L2, 5)
    dist_height = dist * (1.0 + np.clip(ndsm_safe, 0, 60) / 15.0)
    dist_vals = dist_height[dist_height > 0]
    peak_thresh = float(np.percentile(dist_vals, 55)) if dist_vals.size > 0 else 1.0
    _, fg_markers = cv2.threshold(dist_height, peak_thresh, 255, cv2.THRESH_BINARY)
    fg_markers = cv2.morphologyEx(fg_markers.astype(np.uint8), cv2.MORPH_OPEN, kern3)

    num_markers, marker_labels = cv2.connectedComponents(fg_markers)
    unknown = cv2.subtract(build_candidate * 255, fg_markers)
    marker_labels_ws = marker_labels + 1
    marker_labels_ws[unknown == 255] = 0
    ws_labels = cv2.watershed(rgb_bgr.copy(), marker_labels_ws.copy())

    image_area = float(h * w)
    rejected_log: List[Dict] = []
    building_list: List[Dict] = []
    all_roof_pos:       List[List[float]] = []
    all_roof_uvs:       List[List[float]] = []
    all_roof_indices:   List[int] = []
    all_roof_elev_colors:   List[float] = []
    all_roof_height_colors: List[float] = []
    all_wall_pos:       List[List[float]] = []
    all_wall_indices:   List[int] = []
    all_wall_elev_colors:   List[float] = []
    all_wall_height_colors: List[float] = []
    roof_vert_offset = 0
    wall_vert_offset = 0

    bldg_count = 0
    for lab_id in range(2, num_markers + 1):
        b_mask = ((ws_labels == lab_id) & (build_candidate > 0))
        area = int(b_mask.sum())
        if area < 25:
            continue
        coords = np.argwhere(b_mask)
        if coords.size == 0:
            continue
        bh_ = int(coords[:, 0].max() - coords[:, 0].min() + 1)
        bw_ = int(coords[:, 1].max() - coords[:, 1].min() + 1)
        if bw_ > 0.75 * w and bh_ > 0.75 * h:
            rejected_log.append({"lab_id": lab_id, "area": area, "reason": "still_mega"})
            continue

        b_mask_u8 = b_mask.astype(np.uint8)

        # ── P2: FLAT per-building roof elevation (interior P75) ────────────────
        interior_dsm = dsm_smooth[b_mask]
        if interior_dsm.size == 0:
            continue
        z_ground    = float(np.percentile(dtm_safe[b_mask], 30))
        z_roof_flat = float(np.percentile(interior_dsm, 75))
        z_roof_flat = float(np.clip(z_roof_flat, z_ground + 2.0, z_ground + 120.0))
        bldg_height = max(2.0, z_roof_flat - z_ground)

        y_roof   = float(np.clip((z_roof_flat - z_base) * exaggeration, 0.0, 9999.0))
        y_ground = float(np.clip((z_ground    - z_base) * exaggeration, 0.0, y_roof))

        cent_row, cent_col = coords.mean(axis=0)
        cent_x = (float(cent_col) / w - 0.5) * w_m
        cent_z = (float(cent_row) / h - 0.5) * h_m

        bldg_count += 1
        h_norm = float(np.clip(bldg_height / 60.0, 0.0, 1.0))
        hr, hg, hb = turbo_colormap(h_norm)
        er, eg, eb = turbo_colormap(float(np.clip((z_roof_flat - z_base) / z_range, 0.0, 1.0)))
        er_g, eg_g2, eb_g = turbo_colormap(float(np.clip((z_ground - z_base) / z_range, 0.0, 1.0)))

        bldg_info: Dict = {
            "id":       int(bldg_count),
            "orig_id":  str(lab_id),
            "area_m2":  round(float(area * gsd_x * gsd_y), 1),
            "z_ground": round(z_ground, 2),
            "z_roof":   round(z_roof_flat, 2),
            "height_m": round(bldg_height, 2),
            "cx": round(float(cent_x), 2),
            "cy": round(float(y_roof),  2),
            "cz": round(float(cent_z),  2)
        }

        # ── P3: Adaptive contour simplification ───────────────────────────────
        contours, _ = cv2.findContours(b_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)
        approx_eps = float(np.clip(perimeter / 50.0, 2.0, 12.0))
        cnt_approx = cv2.approxPolyDP(cnt, approx_eps, closed=True)
        if len(cnt_approx) > 36:
            cnt_approx = cv2.approxPolyDP(cnt, approx_eps * 2.5, closed=True)
        if len(cnt_approx) > 48:
            cnt_approx = cv2.approxPolyDP(cnt, perimeter / 18.0, closed=True)

        pts_cnt = cnt_approx.reshape(-1, 2)
        n_pts = len(pts_cnt)
        if n_pts < 3:
            continue

        # ── A. Roof triangulation with centroid validation ─────────────────────
        poly_2d = pts_cnt.astype(np.float32)
        tri_indices = triangulate_polygon_earcut(poly_2d)
        if not tri_indices:
            continue

        valid_triangles = []
        cnt_int32 = cnt.astype(np.int32)
        for i0, i1, i2 in tri_indices:
            c_x = (pts_cnt[i0][0] + pts_cnt[i1][0] + pts_cnt[i2][0]) / 3.0
            c_y = (pts_cnt[i0][1] + pts_cnt[i1][1] + pts_cnt[i2][1]) / 3.0
            if cv2.pointPolygonTest(cnt_int32, (c_x, c_y), False) >= 0:
                valid_triangles.append((i0, i1, i2))
        if not valid_triangles:
            valid_triangles = tri_indices

        cnt_start = roof_vert_offset
        for i in range(n_pts):
            c_i = int(np.clip(pts_cnt[i][0], 0, w - 1))
            r_i = int(np.clip(pts_cnt[i][1], 0, h - 1))
            px_m = (float(c_i) / w - 0.5) * w_m
            pz_m = (float(r_i) / h - 0.5) * h_m
            u_i  = float(c_i) / max(w - 1, 1)
            v_i  = 1.0 - float(r_i) / max(h - 1, 1)
            all_roof_pos.append([float(px_m), float(y_roof), float(pz_m)])
            all_roof_uvs.append([u_i, v_i])
            all_roof_elev_colors.extend([er, eg, eb])
            all_roof_height_colors.extend([hr, hg, hb])

        for i0, i1, i2 in valid_triangles:
            all_roof_indices.extend([cnt_start + i0, cnt_start + i1, cnt_start + i2])
        roof_vert_offset += n_pts

        building_list.append(bldg_info)

        # ── B. Walls (stable y_ground, flat y_roof) ───────────────────────────
        wall_start = wall_vert_offset
        for i in range(n_pts):
            c_i = int(np.clip(pts_cnt[i][0], 0, w - 1))
            r_i = int(np.clip(pts_cnt[i][1], 0, h - 1))
            px_m = (float(c_i) / w - 0.5) * w_m
            pz_m = (float(r_i) / h - 0.5) * h_m
            all_wall_pos.append([float(px_m), float(y_ground), float(pz_m)])
            all_wall_pos.append([float(px_m), float(y_roof),   float(pz_m)])
            all_wall_elev_colors.extend([er_g, eg_g2, eb_g, er, eg, eb])
            all_wall_height_colors.extend([hr * 0.6, hg * 0.6, hb * 0.6, hr, hg, hb])

        for i in range(n_pts):
            next_i = (i + 1) % n_pts
            g1 = wall_start + 2 * i;     r1 = wall_start + 2 * i + 1
            g2 = wall_start + 2 * next_i; r2 = wall_start + 2 * next_i + 1
            all_wall_indices.extend([g1, r1, r2, g1, r2, g2])

        wall_vert_offset += (2 * n_pts)

    # ── 5. Texture encoding ────────────────────────────────────────────────────
    rgb_resized = cv2.resize(rgb_img, (512, 512), interpolation=cv2.INTER_LINEAR)
    _, buffer = cv2.imencode(
        ".jpg", cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    )
    tex_base64 = base64.b64encode(buffer).decode("utf-8")

    max_dim = float(max(w_m, h_m, (z_max - z_base) * exaggeration, 1.0))

    return {
        "terrain": {
            "positions": terrain_positions.ravel().tolist(),
            "uvs":       terrain_uvs.ravel().tolist(),
            "indices":   terrain_indices.tolist(),
            "elev_colors":  terrain_elev_colors,
            "slope_colors": terrain_slope_colors,
            "n_verts": int(len(terrain_positions)),
            "n_faces":  int(len(terrain_indices) // 3)
        },
        "roofs": {
            "positions": np.array(all_roof_pos,  dtype=np.float32).ravel().tolist() if all_roof_pos else [],
            "uvs":       np.array(all_roof_uvs,  dtype=np.float32).ravel().tolist() if all_roof_uvs else [],
            "indices":   all_roof_indices,
            "elev_colors":   all_roof_elev_colors,
            "height_colors": all_roof_height_colors,
            "n_verts": int(len(all_roof_pos)),
            "n_faces":  int(len(all_roof_indices) // 3)
        },
        "walls": {
            "positions": np.array(all_wall_pos,  dtype=np.float32).ravel().tolist() if all_wall_pos else [],
            "indices":   all_wall_indices,
            "elev_colors":   all_wall_elev_colors,
            "height_colors": all_wall_height_colors,
            "n_verts": int(len(all_wall_pos)),
            "n_faces":  int(len(all_wall_indices) // 3)
        },
        "buildings": building_list,
        "rejected":  rejected_log,
        "texture_base64": tex_base64,
        "bounds": {
            "w_m":     round(w_m, 1),
            "h_m":     round(h_m, 1),
            "z_min":   round(z_base, 1),
            "z_max":   round(z_max, 1),
            "max_dim": round(max_dim, 1)
        }
    }


def generate_interactive_webgl_html(
    rgb_img: np.ndarray,
    dsm: np.ndarray,
    dtm: np.ndarray,
    mask_bldg: np.ndarray,
    gsd: Any = 0.5,
    exaggeration: float = 1.0,
    stride: int = 4,
    default_preset: str = "overview",
    default_mode: str = "rgb",
    prebuilt_scene: Dict[str, Any] = None
) -> str:
    """
    Constructs a complete self-contained Three.js WebGL application HTML string.
    Renders explicit DTM terrain + Ear-clipped DSM roofs + Slate architectural walls.
    """
    geom = prebuilt_scene if prebuilt_scene is not None else build_city_geometry(
        rgb_img=rgb_img,
        dsm=dsm,
        dtm=dtm,
        mask_bldg=mask_bldg,
        gsd=gsd,
        exaggeration=exaggeration,
        stride=stride
    )
    
    geom_json = json.dumps(geom)
    n_bldgs = len(geom["buildings"])
    z_min_val = geom["bounds"]["z_min"]
    z_max_val = geom["bounds"]["z_max"]
    preset_js = json.dumps(str(default_preset).lower())
    mode_js = json.dumps(str(default_mode).lower())
    
    html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DepthWizard 3D Interactive City Flythrough</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; user-select: none; }}
        body {{
            background: #0D1117;
            color: #C9D1D9;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            overflow: hidden;
            width: 100vw;
            height: 100vh;
        }}
        #viewport-container {{
            position: relative;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at center, #161B22 0%, #0D1117 100%);
        }}
        #webgl-canvas {{
            width: 100%;
            height: 100%;
            display: block;
        }}
        #canny-overlay {{
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            object-fit: fill;
            pointer-events: none;
            opacity: 0.78;
            mix-blend-mode: screen;
            z-index: 5;
        }}
        
        /* ── Top HUD ────────────────────────────── */
        #hud-top {{
            position: absolute;
            top: 14px;
            left: 16px;
            background: rgba(13, 17, 23, 0.88);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(48, 54, 61, 0.9);
            border-radius: 10px;
            padding: 10px 16px;
            font-size: 12px;
            line-height: 1.5;
            color: #C9D1D9;
            box-shadow: 0 8px 24px rgba(0,0,0,0.4);
            pointer-events: none;
            z-index: 20;
        }}
        .hud-badge {{
            display: inline-block;
            background: rgba(63, 185, 80, 0.15);
            color: #3FB950;
            border: 1px solid rgba(63, 185, 80, 0.4);
            border-radius: 4px;
            padding: 2px 6px;
            font-weight: 700;
            font-size: 11px;
            margin-right: 6px;
        }}
        .hud-key {{
            background: #21262D;
            border: 1px solid #30363D;
            border-radius: 4px;
            padding: 1px 5px;
            font-size: 11px;
            font-family: monospace;
            color: #58A6FF;
            margin: 0 1px;
        }}
        
        /* ── Mode Switcher (Top Right) ───────────── */
        #mode-bar {{
            position: absolute;
            top: 14px;
            right: 16px;
            background: rgba(13, 17, 23, 0.9);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(48, 54, 61, 0.9);
            border-radius: 10px;
            padding: 6px 10px;
            display: flex;
            gap: 6px;
            align-items: center;
            z-index: 20;
            box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        }}
        .btn-mode {{
            background: #21262D;
            color: #8B949E;
            border: 1px solid #30363D;
            border-radius: 6px;
            padding: 5px 10px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s;
        }}
        .btn-mode:hover {{
            background: #30363D;
            color: #C9D1D9;
        }}
        .btn-mode.active {{
            background: #1F6FEB;
            color: #FFFFFF;
            border-color: #58A6FF;
        }}

        /* ── Right Inspector Panel ──────────────── */
        #inspector-panel {{
            position: absolute;
            top: 60px;
            right: 16px;
            width: 220px;
            background: rgba(13, 17, 23, 0.92);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(48, 54, 61, 0.9);
            border-radius: 10px;
            padding: 12px 14px;
            font-size: 12px;
            color: #C9D1D9;
            box-shadow: 0 8px 24px rgba(0,0,0,0.5);
            z-index: 20;
            display: none;
        }}
        .insp-title {{
            font-size: 13px;
            font-weight: 700;
            color: #58A6FF;
            border-bottom: 1px solid #30363D;
            padding-bottom: 6px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
        }}
        .insp-close {{
            cursor: pointer;
            color: #8B949E;
            font-weight: bold;
        }}
        .insp-close:hover {{ color: #F0F6FC; }}
        .insp-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }}
        .insp-lbl {{ color: #8B949E; }}
        .insp-val {{ font-weight: 600; color: #F0F6FC; }}

        /* ── Active Legend Indicator ────────────── */
        #legend-box {{
            position: absolute;
            bottom: 68px;
            left: 16px;
            background: rgba(13, 17, 23, 0.88);
            border: 1px solid rgba(48, 54, 61, 0.9);
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 11px;
            color: #C9D1D9;
            z-index: 20;
            display: none;
        }}
        .legend-gradient {{
            width: 140px;
            height: 10px;
            border-radius: 3px;
            margin: 4px 0;
            background: linear-gradient(to right, #30123B, #28BBEC, #A2FC3C, #FB8022, #7A0403);
        }}
        .legend-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 10px;
            color: #8B949E;
        }}
        
        /* ── Bottom Controls Toolbar ────────────── */
        #controls-bar {{
            position: absolute;
            bottom: 16px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(22, 27, 34, 0.92);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(48, 54, 61, 0.9);
            border-radius: 30px;
            padding: 6px 14px;
            display: flex;
            gap: 8px;
            align-items: center;
            box-shadow: 0 12px 32px rgba(0,0,0,0.5);
            z-index: 20;
        }}
        .btn-view {{
            background: #21262D;
            color: #C9D1D9;
            border: 1px solid #30363D;
            border-radius: 18px;
            padding: 6px 14px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            align-items: center;
            gap: 4px;
        }}
        .btn-view:hover {{
            background: #30363D;
            border-color: #58A6FF;
            color: #58A6FF;
            transform: translateY(-1px);
        }}
        .btn-view.active {{
            background: #1F6FEB;
            border-color: #58A6FF;
            color: #FFFFFF;
        }}
        .btn-fly {{
            background: linear-gradient(135deg, #238636 0%, #2EA043 100%);
            color: #FFFFFF;
            border-color: #3FB950;
        }}
        .btn-fly:hover {{
            background: linear-gradient(135deg, #2EA043 0%, #3FB950 100%);
            box-shadow: 0 0 12px rgba(46, 160, 67, 0.4);
        }}
    </style>
    <!-- Standalone Three.js + OrbitControls Bundle -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="viewport-container">
        <canvas id="webgl-canvas"></canvas>
        <img id="canny-overlay" alt="Canny auxiliary structural edge cue" style="display:none;">
        
        <!-- HUD Overlay -->
        <div id="hud-top">
            <div><span class="hud-badge">🌐 3D CITY DIGITAL TWIN</span> <b style="color:#58A6FF;">DepthWizard</b> · 60 FPS</div>
            <div style="margin-top: 4px; color: #8B949E;">
                <b>Mouse:</b> Left Orbit · Right Pan · Scroll Zoom · Click Structure to Inspect<br>
                <b>Flight:</b> <span class="hud-key">W</span><span class="hud-key">A</span><span class="hud-key">S</span><span class="hud-key">D</span> / <span class="hud-key">Arrows</span> Fly · <span class="hud-key">Shift</span> Speed Boost<br>
                <b>City:</b> <span style="color:#3FB950;">{n_bldgs} Ear-Clipped Structures</span> + Reconstructed DTM/DSM
            </div>
        </div>

        <!-- Mode Switcher -->
        <div id="mode-bar">
            <button class="btn-mode active" id="mode-rgb" onclick="setRenderMode('rgb')">🏙️ RGB City</button>
            <button class="btn-mode" id="mode-elev" onclick="setRenderMode('elev')">📈 Elevation</button>
            <button class="btn-mode" id="mode-height" onclick="setRenderMode('height')">🏢 Height</button>
            <button class="btn-mode" id="mode-slope" onclick="setRenderMode('slope')">📐 Slope</button>
        </div>

        <!-- Inspector Panel -->
        <div id="inspector-panel">
            <div class="insp-title">
                <span id="insp-title">🏢 Building Details</span>
                <span class="insp-close" onclick="closeInspector()">×</span>
            </div>
            <div class="insp-row"><span class="insp-lbl">Structure ID:</span><span class="insp-val" id="insp-id">#1</span></div>
            <div class="insp-row"><span class="insp-lbl">Roof Elevation:</span><span class="insp-val" id="insp-roof">0.0 m</span></div>
            <div class="insp-row"><span class="insp-lbl">Ground Elevation:</span><span class="insp-val" id="insp-ground">0.0 m</span></div>
            <div class="insp-row"><span class="insp-lbl">Structure Height:</span><span class="insp-val" style="color:#3FB950;" id="insp-height">0.0 m</span></div>
            <div class="insp-row"><span class="insp-lbl">Footprint Area:</span><span class="insp-val" id="insp-area">0 m²</span></div>
        </div>

        <!-- Legend Box -->
        <div id="legend-box">
            <div id="legend-title" style="font-weight:600; color:#58A6FF; margin-bottom:2px;">Elevation Legend</div>
            <div class="legend-gradient" id="legend-grad"></div>
            <div class="legend-labels">
                <span id="legend-min">{z_min_val:.1f} m</span>
                <span id="legend-max">{z_max_val:.1f} m</span>
            </div>
        </div>

        <!-- Camera Controls -->
        <div id="controls-bar">
            <button class="btn-view active" id="btn-overview" onclick="setPreset('overview')">🏙️ City Overview</button>
            <button class="btn-view" id="btn-urban" onclick="setPreset('urban')">🏢 Urban Oblique</button>
            <button class="btn-view" id="btn-inspection" onclick="setPreset('inspection')">🔍 Inspection</button>
            <button class="btn-view" id="btn-top" onclick="setPreset('top')">⬇️ Top-Down</button>
            <button class="btn-view" id="btn-street" onclick="setPreset('street')">🚶 Pedestrian</button>
            <button class="btn-view btn-fly" id="btn-fly" onclick="toggleFlythrough()">✈️ Cinematic Flythrough</button>
            <button class="btn-view" id="btn-reset" onclick="resetView()">🔄 Fit to Scene</button>
        </div>
    </div>

    <script>
        const cityData = {geom_json};
        const canvas = document.getElementById('webgl-canvas');
        const container = document.getElementById('viewport-container');
        const cannyOverlay = document.getElementById('canny-overlay');
        if (cityData.canny && cityData.canny.enabled && cityData.canny_overlay_base64) {{
            cannyOverlay.src = "data:image/png;base64," + cityData.canny_overlay_base64;
            cannyOverlay.style.display = 'block';
        }}

        // ── 1. Three.js Scene, Camera, Renderer ────────────────────────────────
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0D1117);

        const width = container.clientWidth || window.innerWidth;
        const height = container.clientHeight || window.innerHeight;

        const camera = new THREE.PerspectiveCamera(40, width / height, 0.5, 4000);
        
        const renderer = new THREE.WebGLRenderer({{
            canvas: canvas,
            antialias: true,
            powerPreference: "high-performance",
            alpha: false
        }});
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.15;
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.06;
        controls.maxPolarAngle = Math.PI / 2 - 0.01;
        controls.minDistance = 5;
        controls.maxDistance = 1500;

        // ── 2. Lighting ───────────────────────────────────────────────────────
        const ambientLight = new THREE.AmbientLight(0xFFFFFF, 0.75);
        scene.add(ambientLight);

        const hemiLight = new THREE.HemisphereLight(0xE2E8F0, 0x1E293B, 0.6);
        scene.add(hemiLight);

        const keyLight = new THREE.DirectionalLight(0xFFFFFF, 0.9);
        keyLight.position.set(-150, 220, 120);
        keyLight.castShadow = true;
        keyLight.shadow.mapSize.width = 2048;
        keyLight.shadow.mapSize.height = 2048;
        keyLight.shadow.camera.near = 10;
        keyLight.shadow.camera.far = 800;
        const d = 250;
        keyLight.shadow.camera.left = -d;
        keyLight.shadow.camera.right = d;
        keyLight.shadow.camera.top = d;
        keyLight.shadow.camera.bottom = -d;
        scene.add(keyLight);

        const fillLight = new THREE.DirectionalLight(0x94A3B8, 0.45);
        fillLight.position.set(120, 100, -120);
        scene.add(fillLight);

        // ── 3. Texture Loading (Satellite Orthophoto) ─────────────────────────
        const texLoader = new THREE.TextureLoader();
        const satTexture = texLoader.load("data:image/jpeg;base64," + cityData.texture_base64, function() {{
            renderer.render(scene, camera);
        }});
        satTexture.anisotropy = renderer.capabilities.getMaxAnisotropy();

        // ── 4. Construct Layer 1: DTM Terrain Mesh ────────────────────────────
        const terrainGeom = new THREE.BufferGeometry();
        terrainGeom.setAttribute('position', new THREE.Float32BufferAttribute(cityData.terrain.positions, 3));
        terrainGeom.setAttribute('uv', new THREE.Float32BufferAttribute(cityData.terrain.uvs, 2));
        terrainGeom.setAttribute('colorElev', new THREE.Float32BufferAttribute(cityData.terrain.elev_colors, 3));
        terrainGeom.setAttribute('colorSlope', new THREE.Float32BufferAttribute(cityData.terrain.slope_colors, 3));
        terrainGeom.setIndex(cityData.terrain.indices);
        terrainGeom.computeVertexNormals();

        const terrainMatRGB = new THREE.MeshStandardMaterial({{
            map: satTexture,
            roughness: 0.85,
            metalness: 0.05,
            flatShading: false
        }});
        const terrainMatVertex = new THREE.MeshStandardMaterial({{
            vertexColors: true,
            roughness: 0.85,
            metalness: 0.05,
            flatShading: false
        }});
        const terrainMatSubdued = new THREE.MeshStandardMaterial({{
            color: 0x21262D,
            roughness: 0.9,
            metalness: 0.05,
            flatShading: false
        }});

        const terrainMesh = new THREE.Mesh(terrainGeom, terrainMatRGB);
        terrainMesh.receiveShadow = true;
        scene.add(terrainMesh);

        // Optional XYZ representation of authoritative terrain and building vertices.
        let pointCloud = null;
        if (cityData.point_cloud && cityData.point_cloud.enabled && cityData.point_cloud.positions && cityData.point_cloud.positions.length > 0) {{
            const pointGeom = new THREE.BufferGeometry();
            const finitePointPositions = [];
            for (let i = 0; i < cityData.point_cloud.positions.length; i += 3) {{
                const px = cityData.point_cloud.positions[i];
                const py = cityData.point_cloud.positions[i + 1];
                const pz = cityData.point_cloud.positions[i + 2];
                if (Number.isFinite(px) && Number.isFinite(py) && Number.isFinite(pz)) {{
                    finitePointPositions.push(px, py, pz);
                }}
            }}
            pointGeom.setAttribute('position', new THREE.Float32BufferAttribute(finitePointPositions, 3));
            const pointMat = new THREE.PointsMaterial({{ color: 0xFFD166, size: 2.2, sizeAttenuation: false, transparent: true, opacity: 0.92 }});
            pointCloud = new THREE.Points(pointGeom, pointMat);
            pointCloud.name = 'Authoritative XYZ Point Cloud';
            scene.add(pointCloud);
        }}

        // ── 5. Construct Layer 2: DSM Building Roofs ──────────────────────────
        let roofsMesh = null;
        let roofsMatRGB = null;
        let roofsMatVertex = null;
        if (cityData.roofs.n_verts > 0) {{
            const roofsGeom = new THREE.BufferGeometry();
            roofsGeom.setAttribute('position', new THREE.Float32BufferAttribute(cityData.roofs.positions, 3));
            roofsGeom.setAttribute('uv', new THREE.Float32BufferAttribute(cityData.roofs.uvs, 2));
            roofsGeom.setAttribute('colorElev', new THREE.Float32BufferAttribute(cityData.roofs.elev_colors, 3));
            roofsGeom.setAttribute('colorHeight', new THREE.Float32BufferAttribute(cityData.roofs.height_colors, 3));
            roofsGeom.setIndex(cityData.roofs.indices);
            roofsGeom.computeVertexNormals();

            roofsMatRGB = new THREE.MeshStandardMaterial({{
                map: satTexture,
                roughness: 0.65,
                metalness: 0.1,
                flatShading: false
            }});
            roofsMatVertex = new THREE.MeshStandardMaterial({{
                vertexColors: true,
                roughness: 0.65,
                metalness: 0.1,
                flatShading: false
            }});

            roofsMesh = new THREE.Mesh(roofsGeom, roofsMatRGB);
            roofsMesh.castShadow = true;
            roofsMesh.receiveShadow = true;
            scene.add(roofsMesh);
        }}

        // ── 6. Construct Layer 3: Vertical Architectural Facades ───────────────
        let wallsMesh = null;
        let wallsMatSlate = null;
        let wallsMatVertex = null;
        if (cityData.walls.n_verts > 0) {{
            const wallsGeom = new THREE.BufferGeometry();
            wallsGeom.setAttribute('position', new THREE.Float32BufferAttribute(cityData.walls.positions, 3));
            wallsGeom.setAttribute('colorElev', new THREE.Float32BufferAttribute(cityData.walls.elev_colors, 3));
            wallsGeom.setAttribute('colorHeight', new THREE.Float32BufferAttribute(cityData.walls.height_colors, 3));
            wallsGeom.setIndex(cityData.walls.indices);
            wallsGeom.computeVertexNormals();

            wallsMatSlate = new THREE.MeshStandardMaterial({{
                color: 0x334155, // Clean architectural slate-concrete
                roughness: 0.6,
                metalness: 0.2,
                flatShading: true
            }});
            wallsMatVertex = new THREE.MeshStandardMaterial({{
                vertexColors: true,
                roughness: 0.6,
                metalness: 0.2,
                flatShading: true
            }});

            wallsMesh = new THREE.Mesh(wallsGeom, wallsMatSlate);
            wallsMesh.castShadow = true;
            wallsMesh.receiveShadow = true;
            scene.add(wallsMesh);
        }}

        // ── 7. Render Modes Controller (Instant Client-Side Switching) ─────────
        let currentMode = 'rgb';
        window.setRenderMode = function(mode) {{
            currentMode = mode;
            document.querySelectorAll('.btn-mode').forEach(b => b.classList.remove('active'));
            const modeBtn = document.getElementById('mode-' + mode);
            if (modeBtn) modeBtn.classList.add('active');

            const legendBox = document.getElementById('legend-box');
            const legendTitle = document.getElementById('legend-title');
            const legendGrad = document.getElementById('legend-grad');
            const legendMin = document.getElementById('legend-min');
            const legendMax = document.getElementById('legend-max');

            if (mode === 'rgb') {{
                terrainMesh.material = terrainMatRGB;
                if (roofsMesh) roofsMesh.material = roofsMatRGB;
                if (wallsMesh) wallsMesh.material = wallsMatSlate;
                legendBox.style.display = 'none';
            }} else if (mode === 'elev') {{
                terrainGeom.setAttribute('color', terrainGeom.getAttribute('colorElev'));
                terrainMesh.material = terrainMatVertex;
                if (roofsMesh) {{
                    roofsMesh.geometry.setAttribute('color', roofsMesh.geometry.getAttribute('colorElev'));
                    roofsMesh.material = roofsMatVertex;
                }}
                if (wallsMesh) {{
                    wallsMesh.geometry.setAttribute('color', wallsMesh.geometry.getAttribute('colorElev'));
                    wallsMesh.material = wallsMatVertex;
                }}
                legendTitle.innerText = "Absolute Elevation (m)";
                legendGrad.style.background = "linear-gradient(to right, #30123B, #28BBEC, #A2FC3C, #FB8022, #7A0403)";
                legendMin.innerText = "{z_min_val:.1f} m";
                legendMax.innerText = "{z_max_val:.1f} m";
                legendBox.style.display = 'block';
            }} else if (mode === 'height') {{
                terrainMesh.material = terrainMatSubdued;
                if (roofsMesh) {{
                    roofsMesh.geometry.setAttribute('color', roofsMesh.geometry.getAttribute('colorHeight'));
                    roofsMesh.material = roofsMatVertex;
                }}
                if (wallsMesh) {{
                    wallsMesh.geometry.setAttribute('color', wallsMesh.geometry.getAttribute('colorHeight'));
                    wallsMesh.material = wallsMatVertex;
                }}
                legendTitle.innerText = "Building Height Above Ground (m)";
                legendGrad.style.background = "linear-gradient(to right, #30123B, #28BBEC, #A2FC3C, #FB8022, #7A0403)";
                legendMin.innerText = "0.0 m";
                legendMax.innerText = "60.0+ m";
                legendBox.style.display = 'block';
            }} else if (mode === 'slope') {{
                terrainGeom.setAttribute('color', terrainGeom.getAttribute('colorSlope'));
                terrainMesh.material = terrainMatVertex;
                if (roofsMesh) roofsMesh.material = roofsMatRGB;
                if (wallsMesh) wallsMesh.material = wallsMatSlate;
                legendTitle.innerText = "Terrain Slope (degrees)";
                legendGrad.style.background = "linear-gradient(to right, #22C55E, #EAB308, #EF4444)";
                legendMin.innerText = "0° (Flat)";
                legendMax.innerText = "45°+ (Steep)";
                legendBox.style.display = 'block';
            }}
        }};

        // ── 8. Interactive Raycasting & Building Selection ────────────────────
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();

        window.addEventListener('click', (e) => {{
            if (isFlying) return;
            const rect = canvas.getBoundingClientRect();
            mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

            raycaster.setFromCamera(mouse, camera);
            const targets = [];
            if (roofsMesh) targets.push(roofsMesh);
            if (wallsMesh) targets.push(wallsMesh);

            const intersects = raycaster.intersectObjects(targets);
            if (intersects.length > 0) {{
                const hit = intersects[0];
                const hitPt = hit.point;
                
                let closestBldg = null;
                let minDist = Infinity;
                cityData.buildings.forEach(b => {{
                    const dx = b.cx - hitPt.x;
                    const dz = b.cz - hitPt.z;
                    const d = Math.sqrt(dx*dx + dz*dz);
                    if (d < minDist) {{
                        minDist = d;
                        closestBldg = b;
                    }}
                }});

                if (closestBldg && minDist < 60) {{
                    showBuildingInfo(closestBldg);
                }}
            }}
        }});

        function showBuildingInfo(b) {{
            const panel = document.getElementById('inspector-panel');
            panel.style.display = 'block';
            document.getElementById('insp-id').innerText = '#' + b.id;
            document.getElementById('insp-roof').innerText = b.height_available === false ? 'HEIGHT_UNAVAILABLE' : b.z_roof.toFixed(1) + ' m';
            document.getElementById('insp-ground').innerText = b.height_available === false ? 'HEIGHT_UNAVAILABLE' : b.z_ground.toFixed(1) + ' m';
            document.getElementById('insp-height').innerText = b.height_available === false ? 'HEIGHT_UNAVAILABLE' : b.height_m.toFixed(1) + ' m';
            document.getElementById('insp-area').innerText = Math.round(b.area_m2).toLocaleString() + ' m²';
        }}

        window.closeInspector = function() {{
            document.getElementById('inspector-panel').style.display = 'none';
        }};

        // ── 9. Camera Presets (Bounding-Box Calculated) ────────────────────────
        const maxDim = cityData.bounds.max_dim || 256;
        // Wider base distance so full city block fits with ~20% margin
        const camDist = maxDim * 1.65;
        const sceneTarget = new THREE.Vector3(0, maxDim * 0.12, 0);

        let tallestBldg = cityData.buildings.find(b => b.height_available !== false && Number.isFinite(b.height_m)) || {{ cx: 0, cy: 10, cz: 0, height_m: 20 }};
        cityData.buildings.forEach(b => {{
            if (b.height_available !== false && Number.isFinite(b.height_m) && b.height_m > tallestBldg.height_m) tallestBldg = b;
        }});

        window.setPreset = function(preset) {{
            isFlying = false;
            document.getElementById('btn-fly').innerText = "✈️ Cinematic Flythrough";

            document.querySelectorAll('.btn-view').forEach(btn => btn.classList.remove('active'));
            const btnEl = document.getElementById('btn-' + preset);
            if (btnEl) btnEl.classList.add('active');

            if (preset === 'overview') {{
                // Elevated oblique — entire city block with 20% margin
                animateCameraTo(-camDist * 0.62, maxDim * 0.82, camDist * 0.62,
                                0, maxDim * 0.10, 0);
            }} else if (preset === 'urban') {{
                // Lower oblique — emphasises building facades
                animateCameraTo(-camDist * 0.42, maxDim * 0.30, camDist * 0.42,
                                0, maxDim * 0.10, 0);
            }} else if (preset === 'inspection') {{
                // Close-up on tallest building
                const td = maxDim * 0.30;
                animateCameraTo(tallestBldg.cx - td * 0.5,
                                tallestBldg.cy + maxDim * 0.18,
                                tallestBldg.cz + td * 0.5,
                                tallestBldg.cx, tallestBldg.cy, tallestBldg.cz);
            }} else if (preset === 'top') {{
                // Nadir — footprint / roof inspection
                animateCameraTo(0, camDist * 1.05, 0.5, 0, 0, 0);
            }} else if (preset === 'street') {{
                // Street-level looking up at skyline
                animateCameraTo(0, maxDim * 0.025, maxDim * 0.44,
                                0, maxDim * 0.10, 0);
            }}
        }};

        window.resetView = function() {{
            setPreset('overview');
        }};

        function animateCameraTo(px, py, pz, tx, ty, tz) {{
            const startPos = camera.position.clone();
            const startTarget = controls.target.clone();
            const endPos = new THREE.Vector3(px, py, pz);
            const endTarget = new THREE.Vector3(tx, ty, tz);
            
            let t = 0;
            function step() {{
                t += 0.05;
                if (t <= 1) {{
                    const ease = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
                    camera.position.lerpVectors(startPos, endPos, ease);
                    controls.target.lerpVectors(startTarget, endTarget, ease);
                    controls.update();
                    requestAnimationFrame(step);
                }} else {{
                    camera.position.copy(endPos);
                    controls.target.copy(endTarget);
                    controls.update();
                }}
            }}
            step();
        }}

        // Apply initial preset and initial render mode
        const initPreset = {preset_js};
        const initMode = {mode_js};
        if (initMode && initMode !== 'rgb') {{
            setRenderMode(initMode);
        }}
        if (initPreset && initPreset !== 'overview') {{
            setPreset(initPreset);
        }} else {{
            // Default City Overview: full city block, 20% margin
            camera.position.set(-camDist * 0.62, maxDim * 0.82, camDist * 0.62);
            controls.target.copy(sceneTarget);
            controls.update();
        }}

        // ── 10. Cinematic Flythrough Animation ────────────────────────────────
        let isFlying = false;
        let flyProgress = 0;
        window.toggleFlythrough = function() {{
            isFlying = !isFlying;
            const btn = document.getElementById('btn-fly');
            btn.innerText = isFlying ? "⏸️ Pause Flythrough" : "✈️ Cinematic Flythrough";
            if (isFlying) {{
                document.querySelectorAll('.btn-view').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            }}
        }};

        // ── 11. First-Person Keyboard Navigation (WASD / Arrows) ───────────────
        const keyState = {{}};
        window.addEventListener('keydown', (e) => {{ keyState[e.code] = true; }});
        window.addEventListener('keyup', (e) => {{ keyState[e.code] = false; }});

        function handleFlightControls(delta) {{
            let speed = (keyState['ShiftLeft'] || keyState['ShiftRight']) ? 70 : 30;
            speed *= delta;

            const forward = new THREE.Vector3();
            camera.getWorldDirection(forward);
            forward.y = 0;
            forward.normalize();

            const right = new THREE.Vector3();
            right.crossVectors(forward, camera.up).normalize();

            let moved = false;
            if (keyState['KeyW'] || keyState['ArrowUp'])    {{ camera.position.addScaledVector(forward, speed); controls.target.addScaledVector(forward, speed); moved = true; }}
            if (keyState['KeyS'] || keyState['ArrowDown'])  {{ camera.position.addScaledVector(forward, -speed); controls.target.addScaledVector(forward, -speed); moved = true; }}
            if (keyState['KeyA'] || keyState['ArrowLeft'])  {{ camera.position.addScaledVector(right, -speed); controls.target.addScaledVector(right, -speed); moved = true; }}
            if (keyState['KeyD'] || keyState['ArrowRight']) {{ camera.position.addScaledVector(right, speed); controls.target.addScaledVector(right, speed); moved = true; }}
            if (keyState['KeyE'] || keyState['Space'])      {{ camera.position.y += speed; controls.target.y += speed; moved = true; }}
            if (keyState['KeyQ'])                           {{ camera.position.y -= speed; controls.target.y -= speed; moved = true; }}

            if (moved) controls.update();
        }}

        // ── 12. Render Animation Loop ──────────────────────────────────────────
        const clock = new THREE.Clock();
        function animate() {{
            requestAnimationFrame(animate);
            const delta = clock.getDelta();

            if (isFlying) {{
                flyProgress += delta * 0.22;
                const radius = maxDim * 0.65;
                camera.position.x = Math.sin(flyProgress) * radius;
                camera.position.z = Math.cos(flyProgress) * radius;
                camera.position.y = (maxDim * 0.28) + Math.sin(flyProgress * 2.0) * (maxDim * 0.08);
                controls.target.set(0, maxDim * 0.08, 0);
                controls.update();
            }} else {{
                handleFlightControls(delta);
                controls.update();
            }}

            renderer.render(scene, camera);
        }}
        animate();

        // ── 13. Responsive Canvas Resize ───────────────────────────────────────
        function onResize() {{
            const w = container.clientWidth || window.innerWidth;
            const h = container.clientHeight || window.innerHeight;
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
            renderer.setSize(w, h);
        }}
        window.addEventListener('resize', onResize);
        setTimeout(onResize, 100);
    </script>
</body>
</html>
"""
    return html_code
