import sys
import re

file_path = 'depthwizard/viz/interactive_viewer.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

start_marker = 'def build_city_geometry('
end_marker = 'def generate_interactive_webgl_html('

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx == -1 or end_idx == -1:
    print("Could not find boundaries!")
    sys.exit(1)

new_func = """def build_city_geometry(
    rgb_img: np.ndarray,
    dsm: np.ndarray,
    dtm: np.ndarray,
    mask_bldg: np.ndarray,
    gsd: Any = 0.5,
    exaggeration: float = 1.0,
    stride: int = 4
) -> Dict[str, Any]:
    \"\"\"
    Phase 41 — Multi-Evidence Building Instance Reconstruction.

    Root cause fix: The input mask_bldg from the calibration engine is often
    unreliable (picks up roads/shadows as buildings, misses actual buildings).
    This function rebuilds the building evidence from scratch using:
      Evidence 1: nDSM height above terrain (objects >= threshold above ground)
      Evidence 2: DSM morphological top-hat (sharp convex structures above surroundings)
      Evidence 3: RGB flat texture (rooftops have uniform low-variance color)
    These three evidences are fused by majority vote, then individual building
    instances are separated using watershed seeded by distance-transform peaks.
    \"\"\"
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
"""

new_content = content[:start_idx] + new_func + "\n\n" + content[end_idx:]

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Successfully replaced build_city_geometry function.")
