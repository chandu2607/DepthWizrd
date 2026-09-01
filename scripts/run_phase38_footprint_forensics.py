"""
Phase 38 Footprint Forensics and Diagnostic Pipeline.
Evaluates mask semantics, component statistics, component classification,
morphological ablation, and generates all required forensic visual artifacts (01-05).
"""
import sys
import os
import json
import csv
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any

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

OUT_DIR = Path("runs/phase38_footprint_forensics")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCRIPTS_DIR = Path("screenshots")
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

NYC_SCENES = [
    "SV_NewYork_40.7401_-73.9915.tif",   # NYC skyscraper-heavy (primary demo scene)
    "SV_NewYork_40.7333_-73.9835.tif",   # NYC dense-highrise
    "SV_NewYork_40.7335_-74.0053.tif",   # NYC lower-rise / mixed
]

def compute_raster_hashes(dsm: np.ndarray, dtm: np.ndarray, ndsm: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    stats = {}
    for name, arr in [("dsm", dsm), ("dtm", dtm), ("ndsm", ndsm), ("mask", mask.astype(np.float32))]:
        stats[name] = {
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": float(np.nanmean(arr)),
            "p95": float(np.nanpercentile(arr, 95)),
            "p99": float(np.nanpercentile(arr, 99)),
        }
    return stats

def draw_colored_components(mask: np.ndarray, rgb: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Generates 01_components.png with distinct color per component and bounding boxes."""
    h, w = mask.shape[:2]
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    
    # Generate distinct colors for components
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(num_labels + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0] # Background is black
    
    colored_mask = colors[labels_im]
    vis = cv2.addWeighted(rgb, 0.4, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR), 0.6, 0)
    
    image_area = float(h * w)
    total_mask_pixels = float(np.sum(mask > 0))
    
    comp_list = []
    for k in range(1, num_labels):
        area = int(stats[k, cv2.CC_STAT_AREA])
        x = int(stats[k, cv2.CC_STAT_LEFT])
        y = int(stats[k, cv2.CC_STAT_TOP])
        bw = int(stats[k, cv2.CC_STAT_WIDTH])
        bh = int(stats[k, cv2.CC_STAT_HEIGHT])
        cx, cy = float(centroids[k][0]), float(centroids[k][1])
        
        aspect = float(bw) / max(bh, 1)
        comp_mask = (labels_im == k).astype(np.uint8)
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        
        cnt_area = 0.0
        n_vertices = 0
        perimeter = 0.0
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            cnt_area = float(cv2.contourArea(cnt))
            perimeter = float(cv2.arcLength(cnt, True))
            n_vertices = len(cnt)
            
        img_occ_pct = (area / image_area) * 100.0
        bldg_mask_pct = (area / max(total_mask_pixels, 1.0)) * 100.0
        
        # Classification & suspicious checks
        suspicious_reasons = []
        if bw > int(0.65 * w): suspicious_reasons.append(f"bw={bw} > 65% w")
        if bh > int(0.65 * h): suspicious_reasons.append(f"bh={bh} > 65% h")
        if area > int(0.40 * image_area): suspicious_reasons.append(f"area_pct={img_occ_pct:.1f}% > 40%")
        if aspect > 5.0 or aspect < 0.2: suspicious_reasons.append(f"aspect={aspect:.2f} extreme")
        if perimeter > 0 and (perimeter * perimeter / max(area, 1)) > 120.0: suspicious_reasons.append("high_compactness_ratio")
        
        if not suspicious_reasons:
            status = "Valid"
            classification = "Legitimate Building"
        elif bw > int(0.65 * w) or bh > int(0.65 * h) or area > int(0.40 * image_area):
            status = "Suspicious"
            classification = "Merged/Complex"
        else:
            status = "Suspicious"
            classification = "Irregular Boundary"
            
        comp_info = {
            "component_id": k,
            "area_px": area,
            "area_m2": round(area * 0.25, 2),
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": bw,
            "bbox_h": bh,
            "bbox_aspect": round(aspect, 2),
            "centroid_x": round(cx, 1),
            "centroid_y": round(cy, 1),
            "contour_vertices": n_vertices,
            "contour_area": round(cnt_area, 1),
            "mask_area": area,
            "image_occupancy_pct": round(img_occ_pct, 2),
            "building_mask_pct": round(bldg_mask_pct, 2),
            "perimeter": round(perimeter, 1),
            "status": status,
            "classification": classification,
            "suspicious_reasons": "; ".join(suspicious_reasons) if suspicious_reasons else "None"
        }
        comp_list.append(comp_info)
        
        # Draw bounding box and label on visual
        box_color = (0, 0, 255) if status == "Suspicious" else (0, 255, 0)
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), box_color, 1)
        cv2.putText(vis, str(k), (max(x, 2), max(y - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    return vis, comp_list

def perform_morphology_ablation(rgb: np.ndarray, mask: np.ndarray, ndsm: np.ndarray) -> np.ndarray:
    """Generates mask_ablation.png comparing 4 component separation techniques."""
    h, w = mask.shape[:2]
    
    # Method A: Original Raw Mask
    cA_num, cA_lbl, cA_stats, _ = cv2.connectedComponentsWithStats(mask)
    visA = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(visA, f"Method A: Raw Mask ({cA_num-1} comps)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    for k in range(1, cA_num):
        cnts, _ = cv2.findContours((cA_lbl == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(visA, cnts, -1, (0, 255, 0), 1)
        
    # Method B: Fixed Morphological Opening (7x7)
    kernel7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    maskB = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel7)
    cB_num, cB_lbl, _, _ = cv2.connectedComponentsWithStats(maskB)
    visB = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(visB, f"Method B: Fixed Morph 7x7 ({cB_num-1} comps)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    for k in range(1, cB_num):
        cnts, _ = cv2.findContours((cB_lbl == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(visB, cnts, -1, (255, 200, 0), 1)

    # Method C: Distance Transform Watershed Splitting on Merged Components
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
    fg = np.uint8(fg)
    unknown = cv2.subtract(mask, fg)
    _, markers = cv2.connectedComponents(fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    markers = cv2.watershed(rgb_bgr, markers)
    maskC_num = markers.max() - 1
    visC = rgb_bgr.copy()
    cv2.putText(visC, f"Method C: Dist Watershed ({max(0, maskC_num)} comps)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    visC[markers == -1] = [0, 0, 255]

    # Method D: Adaptive Depth-Guided Splitting (Selective Morph Open + Height Valley)
    visD = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    sub_count = 0
    for k in range(1, cA_num):
        area = cA_stats[k, cv2.CC_STAT_AREA]
        bw = cA_stats[k, cv2.CC_STAT_WIDTH]
        bh = cA_stats[k, cv2.CC_STAT_HEIGHT]
        comp_m = (cA_lbl == k).astype(np.uint8)
        if bw > 0.65 * w or bh > 0.65 * h or area > 0.40 * (h * w):
            # Selective split via morph open & local height thresholding
            kernel_opt = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            sub_m = cv2.morphologyEx(comp_m, cv2.MORPH_OPEN, kernel_opt)
            s_num, s_lbl, _, _ = cv2.connectedComponentsWithStats(sub_m)
            for sk in range(1, s_num):
                cnts, _ = cv2.findContours((s_lbl == sk).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(visD, cnts, -1, (0, 255, 100), 2)
                sub_count += 1
        else:
            cnts, _ = cv2.findContours(comp_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(visD, cnts, -1, (0, 255, 255), 1)
            sub_count += 1
    cv2.putText(visD, f"Method D: Selective Depth-Guided ({sub_count} comps)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Combine into 2x2 grid
    top_row = np.hstack([visA, visB])
    bot_row = np.hstack([visC, visD])
    ablation_grid = np.vstack([top_row, bot_row])
    return ablation_grid

def main():
    print("===============================================================")
    print("DEPTHWIZARD — PHASE 38 FOOTPRINT FORENSICS & MASK AUDIT")
    print("===============================================================")
    
    dcfg = DepthConfig(cache_dir="data/dfc2023_multicity/depth_cache")
    depth_model = DepthAnythingV2(dcfg.model_id, dcfg.input_size, dcfg.cache_dir, use_cache=True)
    calib_engine = CalibrationEngine(runs_dir=Path("runs"))

    all_csv_rows = []
    scene_hash_report = {}

    for idx, scene_name in enumerate(NYC_SCENES):
        print(f"\n--- Processing Scene {idx+1}: {scene_name} ---")
        scene_path = Path("data/dfc2023_multicity/rgb") / scene_name
        if not scene_path.exists():
            print(f"  [ERROR] {scene_path} not found!")
            continue

        raster_in = load_raster_input(scene_path, filename=scene_name)
        h, w = raster_in.shape
        depth_raw = depth_model.infer(raster_in.rgb, scene_name, target_hw=(h, w))

        dsm_truth_path = Path("data/dfc2023_multicity/dsm") / scene_name
        truth = cv2.imread(str(dsm_truth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) if dsm_truth_path.exists() else None

        calib_res = calib_engine.calibrate(
            depth_raw, raster_in.rgb, is_georeferenced=raster_in.is_georeferenced,
            mode=CalibrationMode.STRUCTURAL_PRIOR, reference_elevation=truth,
            filename=scene_name
        )

        dsm = calib_res.dsm
        dtm = calib_res.dtm
        ndsm = calib_res.ndsm
        mask = calib_res.mask_bldg.astype(np.uint8)

        # 1. Hashes & Statistics
        hashes = compute_raster_hashes(dsm, dtm, ndsm, mask)
        scene_hash_report[scene_name] = hashes
        print(f"  DSM SHA256 : {hashes['dsm']['sha256']}")
        print(f"  DTM SHA256 : {hashes['dtm']['sha256']}")
        print(f"  nDSM SHA256: {hashes['ndsm']['sha256']}")
        print(f"  nDSM Range : {hashes['ndsm']['min']:.2f}m to {hashes['ndsm']['max']:.2f}m (Mean: {hashes['ndsm']['mean']:.2f}m)")

        # 2. Forensic Component Overlay (01_components.png)
        vis_comps, comp_info_list = draw_colored_components(mask, raster_in.rgb)
        for c in comp_info_list:
            c["scene"] = scene_name
            all_csv_rows.append(c)

        if idx == 0:  # Primary Demo Scene
            cv2.imwrite(str(OUT_DIR / "01_components.png"), vis_comps)
            cv2.imwrite(str(SCRIPTS_DIR / "01_components.png"), vis_comps)
            print(f"  Saved {OUT_DIR}/01_components.png ({len(comp_info_list)} components)")

            # 3. Valid, Suspicious, Rejected overlays (02, 03, 04)
            rgb_bgr = cv2.cvtColor(raster_in.rgb, cv2.COLOR_RGB2BGR)
            
            # 02_valid_components.png
            v02 = rgb_bgr.copy()
            for c in comp_info_list:
                if c["status"] == "Valid":
                    x, y, bw, bh = c["bbox_x"], c["bbox_y"], c["bbox_w"], c["bbox_h"]
                    cv2.rectangle(v02, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                    cv2.putText(v02, f"#{c['component_id']}", (x, max(y - 2, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(v02, f"VALID COMPONENTS ({sum(1 for c in comp_info_list if c['status']=='Valid')})", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imwrite(str(OUT_DIR / "02_valid_components.png"), v02)
            cv2.imwrite(str(SCRIPTS_DIR / "02_valid_components.png"), v02)

            # 03_suspicious_components.png
            v03 = rgb_bgr.copy()
            for c in comp_info_list:
                if c["status"] == "Suspicious":
                    x, y, bw, bh = c["bbox_x"], c["bbox_y"], c["bbox_w"], c["bbox_h"]
                    cv2.rectangle(v03, (x, y), (x + bw, y + bh), (0, 165, 255), 2)
                    cv2.putText(v03, f"#{c['component_id']}: {c['suspicious_reasons']}", (x, max(y - 2, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1)
            cv2.putText(v03, f"SUSPICIOUS COMPONENTS ({sum(1 for c in comp_info_list if c['status']=='Suspicious')})", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.imwrite(str(OUT_DIR / "03_suspicious_components.png"), v03)
            cv2.imwrite(str(SCRIPTS_DIR / "03_suspicious_components.png"), v03)

            # 04_rejected_components.png
            v04 = rgb_bgr.copy()
            for c in comp_info_list:
                if c["status"] == "Suspicious" and "area_pct" in c["suspicious_reasons"]:
                    x, y, bw, bh = c["bbox_x"], c["bbox_y"], c["bbox_w"], c["bbox_h"]
                    cv2.rectangle(v04, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                    cv2.putText(v04, f"REJECTED #{c['component_id']}: {c['suspicious_reasons']}", (x, max(y - 2, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            cv2.putText(v04, f"REJECTED COMPONENTS (MEGA-SLABS)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imwrite(str(OUT_DIR / "04_rejected_components.png"), v04)
            cv2.imwrite(str(SCRIPTS_DIR / "04_rejected_components.png"), v04)

            # 4. Morphological Ablation (mask_ablation.png)
            ablation_grid = perform_morphology_ablation(raster_in.rgb, mask, ndsm)
            cv2.imwrite(str(OUT_DIR / "mask_ablation.png"), ablation_grid)
            cv2.imwrite(str(SCRIPTS_DIR / "mask_ablation.png"), ablation_grid)
            print(f"  Saved {OUT_DIR}/mask_ablation.png")

            # 5. Clean Final Footprints over RGB (05_final_footprints_over_rgb.png)
            v05 = rgb_bgr.copy()
            # Selective depth-guided footprints
            num_l, labels_im, stats_im, _ = cv2.connectedComponentsWithStats(mask)
            valid_bldg_cnt = 0
            for k in range(1, num_l):
                area_k = stats_im[k, cv2.CC_STAT_AREA]
                bw_k = stats_im[k, cv2.CC_STAT_WIDTH]
                bh_k = stats_im[k, cv2.CC_STAT_HEIGHT]
                comp_m = (labels_im == k).astype(np.uint8)
                if bw_k > 0.65 * w or bh_k > 0.65 * h or area_k > 0.40 * (h * w):
                    # Selective split
                    kernel_opt = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                    sub_m = cv2.morphologyEx(comp_m, cv2.MORPH_OPEN, kernel_opt)
                    s_num, s_lbl, s_st, _ = cv2.connectedComponentsWithStats(sub_m)
                    for sk in range(1, s_num):
                        s_a = s_st[sk, cv2.CC_STAT_AREA]
                        s_bw = s_st[sk, cv2.CC_STAT_WIDTH]
                        s_bh = s_st[sk, cv2.CC_STAT_HEIGHT]
                        if s_a < 20 or s_bw > 0.65 * w or s_bh > 0.65 * h:
                            continue
                        cnts, _ = cv2.findContours((s_lbl == sk).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
                        cv2.drawContours(v05, cnts, -1, (0, 255, 0), 2)
                        valid_bldg_cnt += 1
                        # label centroid
                        m_pts = np.argwhere(s_lbl == sk)
                        cr, cc = m_pts.mean(axis=0)
                        cv2.putText(v05, f"B{valid_bldg_cnt}", (int(cc)-10, int(cr)+4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                else:
                    cnts, _ = cv2.findContours(comp_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
                    cv2.drawContours(v05, cnts, -1, (0, 255, 0), 2)
                    valid_bldg_cnt += 1
                    m_pts = np.argwhere(comp_m > 0)
                    cr, cc = m_pts.mean(axis=0)
                    cv2.putText(v05, f"B{valid_bldg_cnt}", (int(cc)-10, int(cr)+4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            cv2.putText(v05, f"FINAL EXTRACTED FOOTPRINTS ({valid_bldg_cnt} Buildings)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imwrite(str(OUT_DIR / "05_final_footprints_over_rgb.png"), v05)
            cv2.imwrite(str(SCRIPTS_DIR / "05_final_footprints_over_rgb.png"), v05)
            print(f"  Saved {OUT_DIR}/05_final_footprints_over_rgb.png ({valid_bldg_cnt} footprints)")

    # 6. Save component_statistics.csv
    csv_path = OUT_DIR / "component_statistics.csv"
    fieldnames = list(all_csv_rows[0].keys()) if all_csv_rows else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_csv_rows)
    print(f"\nSaved component statistics CSV to: {csv_path} ({len(all_csv_rows)} components total)")

if __name__ == "__main__":
    main()
