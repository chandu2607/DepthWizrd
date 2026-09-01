"""
Phase 39 Multi-Evidence Building Instance Extractor.
Combines nDSM height evidence, DSM depth gradients, RGB edges, distance transform,
and height-valley watershed segmentation to extract clean, individual building instances.
"""
import sys
import os
import hashlib
from pathlib import Path
from typing import Tuple, List, Dict, Any
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

def extract_building_instances(
    rgb: np.ndarray,
    dsm: np.ndarray,
    dtm: np.ndarray,
    ndsm: np.ndarray,
    mask_bldg: np.ndarray,
    gsd: float = 0.5
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Multi-evidence building instance extractor.
    Returns:
      - instances: List of building instance dicts (id, mask, area_px, area_m2, z_ground, z_roof, height_m, bbox, centroid)
      - forensics: Dict with intermediate evidence maps and stats
    """
    h, w = mask_bldg.shape[:2]
    image_area = float(h * w)
    pixel_area_m2 = gsd * gsd if isinstance(gsd, (int, float)) else gsd[0] * gsd[1]

    # ── 1. Structural Building Evidence ──────────────────────────────────────
    # Height evidence: nDSM >= 1.8m
    height_evidence = (ndsm >= 1.8).astype(np.uint8)
    
    # Combined initial building candidate mask (Mask + Height evidence)
    candidate_mask = cv2.bitwise_and(mask_bldg.astype(np.uint8), height_evidence)
    
    # Clean up small noise with morphological opening (3x3 kernel)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    candidate_clean = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel_small)

    # ── 2. Depth Valley & RGB Edge Detection for Instance Splitting ─────────
    # Local depth gradient / valley detector on smoothed nDSM
    ndsm_smooth = cv2.bilateralFilter(ndsm.astype(np.float32), d=7, sigmaColor=3.0, sigmaSpace=3.0)
    sobelx = cv2.Sobel(ndsm_smooth, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(ndsm_smooth, cv2.CV_64F, 0, 1, ksize=3)
    depth_grad = np.sqrt(sobelx**2 + sobely**2)
    depth_valleys = (depth_grad > 4.0).astype(np.uint8) # sharp height transitions between roofs

    # RGB edges
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rgb_edges = cv2.Canny(gray, 40, 120)
    rgb_edges = cv2.dilate(rgb_edges, kernel_small)

    # Combined boundaries
    boundaries = cv2.bitwise_or(depth_valleys, (rgb_edges // 255).astype(np.uint8))

    # ── 3. Connected Component Analysis & Selective Watershed Splitting ──────
    num_l, labels_im, stats, centroids = cv2.connectedComponentsWithStats(candidate_clean)
    
    instances = []
    rejected_components = []
    instance_mask = np.zeros((h, w), dtype=np.int32)
    inst_id_counter = 0

    for k in range(1, num_l):
        area = int(stats[k, cv2.CC_STAT_AREA])
        bw = int(stats[k, cv2.CC_STAT_WIDTH])
        bh = int(stats[k, cv2.CC_STAT_HEIGHT])
        bx = int(stats[k, cv2.CC_STAT_LEFT])
        by = int(stats[k, cv2.CC_STAT_TOP])

        if area < 16:
            continue

        comp_mask = (labels_im == k).astype(np.uint8)
        comp_ndsm = ndsm_smooth[comp_mask > 0]
        mean_h = float(np.mean(comp_ndsm)) if comp_ndsm.size > 0 else 0.0

        # Check if mega/merged component
        is_mega = (bw > int(0.60 * w) or bh > int(0.60 * h) or area > int(0.35 * image_area))

        if is_mega:
            # Multi-evidence Watershed Splitting on Merged Component
            # Distance transform guided by nDSM height peaks
            dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
            # Combine distance transform with local nDSM height map to isolate building cores
            dist_height = dist * (1.0 + np.clip(ndsm_smooth, 0, 40) / 10.0)
            
            # Threshold peaks to find markers
            peak_thresh = 0.35 * dist_height.max()
            _, fg_peaks = cv2.threshold(dist_height, peak_thresh, 255, cv2.THRESH_BINARY)
            fg_peaks = cv2.morphologyEx(fg_peaks.astype(np.uint8), cv2.MORPH_OPEN, kernel_small)
            
            sub_num, sub_markers = cv2.connectedComponents(fg_peaks)
            if sub_num > 2:
                # Run watershed within component bounding box
                unknown = cv2.subtract(comp_mask * 255, fg_peaks)
                sub_markers = sub_markers + 1
                sub_markers[unknown == 255] = 0
                
                # Watershed on RGB gradient
                rgb_crop = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                ws_markers = cv2.watershed(rgb_crop, sub_markers.copy())
                
                split_count = 0
                for sm_id in range(2, sub_num + 1):
                    s_mask = (ws_markers == sm_id) & (comp_mask > 0)
                    s_area = int(np.sum(s_mask))
                    if s_area < 20:
                        continue
                    m_pts = np.argwhere(s_mask)
                    s_r, s_c = m_pts.mean(axis=0)
                    s_bw = int(m_pts[:, 1].max() - m_pts[:, 1].min() + 1)
                    s_bh = int(m_pts[:, 0].max() - m_pts[:, 0].min() + 1)
                    
                    if s_bw <= int(0.60 * w) and s_bh <= int(0.60 * h):
                        inst_id_counter += 1
                        instance_mask[s_mask] = inst_id_counter
                        
                        s_ndsm = ndsm_smooth[s_mask]
                        s_dtm = dtm[s_mask]
                        z_grd = float(np.percentile(s_dtm, 30))
                        z_roof = float(np.percentile(s_ndsm + s_dtm, 75))
                        bldg_h = max(1.5, z_roof - z_grd)
                        
                        instances.append({
                            "id": inst_id_counter,
                            "orig_comp": k,
                            "mask": s_mask,
                            "area_px": s_area,
                            "area_m2": round(s_area * pixel_area_m2, 1),
                            "z_ground": round(z_grd, 2),
                            "z_roof": round(z_roof, 2),
                            "height_m": round(bldg_h, 2),
                            "bbox": [int(m_pts[:, 1].min()), int(m_pts[:, 0].min()), s_bw, s_bh],
                            "centroid": [round(float(s_c), 1), round(float(s_r), 1)],
                            "classification": "REAL_BUILDING (Split Component)"
                        })
                        split_count += 1

                if split_count > 0:
                    continue

            # Fallback for mega component: try morphological opening with 7x7 kernel
            kernel7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            opened = cv2.morphologyEx(comp_mask, cv2.MORPH_OPEN, kernel7)
            sub_num2, sub_lbl2, sub_st2, _ = cv2.connectedComponentsWithStats(opened)
            split_count2 = 0
            for sk in range(1, sub_num2):
                s_area = int(sub_st2[sk, cv2.CC_STAT_AREA])
                s_bw = int(sub_st2[sk, cv2.CC_STAT_WIDTH])
                s_bh = int(sub_st2[sk, cv2.CC_STAT_HEIGHT])
                if s_area < 25 or s_bw > int(0.60 * w) or s_bh > int(0.60 * h):
                    continue
                s_mask = (sub_lbl2 == sk)
                m_pts = np.argwhere(s_mask)
                s_r, s_c = m_pts.mean(axis=0)
                
                inst_id_counter += 1
                instance_mask[s_mask] = inst_id_counter
                s_ndsm = ndsm_smooth[s_mask]
                s_dtm = dtm[s_mask]
                z_grd = float(np.percentile(s_dtm, 30))
                z_roof = float(np.percentile(s_ndsm + s_dtm, 75))
                bldg_h = max(1.5, z_roof - z_grd)
                
                instances.append({
                    "id": inst_id_counter,
                    "orig_comp": k,
                    "mask": s_mask,
                    "area_px": s_area,
                    "area_m2": round(s_area * pixel_area_m2, 1),
                    "z_ground": round(z_grd, 2),
                    "z_roof": round(z_roof, 2),
                    "height_m": round(bldg_h, 2),
                    "bbox": [int(sub_st2[sk, cv2.CC_STAT_LEFT]), int(sub_st2[sk, cv2.CC_STAT_TOP]), s_bw, s_bh],
                    "centroid": [round(float(s_c), 1), round(float(s_r), 1)],
                    "classification": "REAL_BUILDING (Morph Split)"
                })
                split_count2 += 1

            if split_count2 == 0:
                rejected_components.append({
                    "component_id": k,
                    "area": area,
                    "bbox": [bx, by, bw, bh],
                    "reason": f"unsplit_mega: area={area}, bbox={bw}x{bh}"
                })
        else:
            # Single clean building instance
            inst_id_counter += 1
            instance_mask[comp_mask > 0] = inst_id_counter
            
            s_dtm = dtm[comp_mask > 0]
            z_grd = float(np.percentile(s_dtm, 30))
            z_roof = float(np.percentile(comp_ndsm + s_dtm, 75))
            bldg_h = max(1.5, z_roof - z_grd)
            
            instances.append({
                "id": inst_id_counter,
                "orig_comp": k,
                "mask": comp_mask > 0,
                "area_px": area,
                "area_m2": round(area * pixel_area_m2, 1),
                "z_ground": round(z_grd, 2),
                "z_roof": round(z_roof, 2),
                "height_m": round(bldg_h, 2),
                "bbox": [bx, by, bw, bh],
                "centroid": [round(float(centroids[k][0]), 1), round(float(centroids[k][1]), 1)],
                "classification": "REAL_BUILDING"
            })

    forensics = {
        "candidate_mask": candidate_clean,
        "depth_valleys": depth_valleys,
        "rgb_edges": rgb_edges,
        "instance_mask": instance_mask,
        "rejected_components": rejected_components,
        "total_instances": len(instances)
    }

    return instances, forensics

def main():
    scene = "SV_NewYork_40.7401_-73.9915.tif"
    scene_path = Path("data/dfc2023_multicity/rgb") / scene
    dcfg = DepthConfig(cache_dir="data/dfc2023_multicity/depth_cache")
    depth_model = DepthAnythingV2(dcfg.model_id, dcfg.input_size, dcfg.cache_dir, use_cache=True)
    calib_engine = CalibrationEngine(runs_dir=Path("runs"))

    raster_in = load_raster_input(scene_path, filename=scene)
    h, w = raster_in.shape
    depth_raw = depth_model.infer(raster_in.rgb, scene, target_hw=(h, w))

    dsm_truth_path = Path("data/dfc2023_multicity/dsm") / scene
    truth = cv2.imread(str(dsm_truth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) if dsm_truth_path.exists() else None

    calib_res = calib_engine.calibrate(
        depth_raw, raster_in.rgb, is_georeferenced=raster_in.is_georeferenced,
        mode=CalibrationMode.STRUCTURAL_PRIOR, reference_elevation=truth,
        filename=scene
    )

    instances, forensics = extract_building_instances(
        raster_in.rgb, calib_res.dsm, calib_res.dtm, calib_res.ndsm, calib_res.mask_bldg, gsd=0.5
    )

    print(f"Extracted {len(instances)} building instances!")
    print(f"Rejected mega components: {len(forensics['rejected_components'])}")
    for inst in sorted(instances, key=lambda x: -x["height_m"])[:10]:
        print(f"  Instance #{inst['id']:2d}: h={inst['height_m']:5.1f}m, area={inst['area_m2']:6.0f}m2, class={inst['classification']}")

if __name__ == "__main__":
    main()
