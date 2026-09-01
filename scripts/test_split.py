import sys
from pathlib import Path
import numpy as np
import cv2 as cv

sys.path.insert(0, ".")
from depthwizard.data.raster_loader import load_raster_input
from depthwizard.config import DepthConfig
from depthwizard.depth.depth_anything import DepthAnythingV2
from depthwizard.calibration import CalibrationEngine, CalibrationMode

def test_morphological_split(scene_name: str):
    scene_path = Path("data/dfc2023_multicity/rgb") / scene_name
    dcfg = DepthConfig(cache_dir="data/dfc2023_multicity/depth_cache")
    depth_model = DepthAnythingV2(dcfg.model_id, dcfg.input_size, dcfg.cache_dir, use_cache=True)
    calib_engine = CalibrationEngine(runs_dir=Path("runs"))
    
    raster_in = load_raster_input(scene_path, filename=scene_name)
    h, w = raster_in.shape
    depth_raw = depth_model.infer(raster_in.rgb, scene_name, target_hw=(h, w))
    calib_res = calib_engine.calibrate(
        depth_raw, raster_in.rgb, is_georeferenced=raster_in.is_georeferenced,
        mode=CalibrationMode.STRUCTURAL_PRIOR, reference_elevation=None,
        filename=scene_name
    )
    
    mask = calib_res.mask_bldg.astype(np.uint8)
    image_area = float(h * w)

    num_l, labels, stats, centroids = cv.connectedComponentsWithStats(mask)
    print(f"\n--- {scene_name} ---")
    print(f"Original connected components: {num_l-1}")

    split_components = []
    rejected_components = []

    for k in range(1, num_l):
        area = int(stats[k, cv.CC_STAT_AREA])
        bb_w = int(stats[k, cv.CC_STAT_WIDTH])
        bb_h = int(stats[k, cv.CC_STAT_HEIGHT])
        if area < 18:
            continue

        is_mega = (bb_w > int(0.65 * w) or bb_h > int(0.65 * h) or area > int(0.40 * image_area))
        if not is_mega:
            split_components.append((k, area, bb_w, bb_h, (labels == k).astype(np.uint8)))
        else:
            # Try splitting mega component using Distance Transform Watershed or Morphological Opening
            comp_mask = (labels == k).astype(np.uint8)
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
            opened = cv.morphologyEx(comp_mask, cv.MORPH_OPEN, kernel)
            
            sub_num, sub_labels, sub_stats, _ = cv.connectedComponentsWithStats(opened)
            recovered = 0
            for sk in range(1, sub_num):
                s_area = int(sub_stats[sk, cv.CC_STAT_AREA])
                s_bw = int(sub_stats[sk, cv.CC_STAT_WIDTH])
                s_bh = int(sub_stats[sk, cv.CC_STAT_HEIGHT])
                if s_area < 25:
                    continue
                if s_bw <= int(0.65 * w) and s_bh <= int(0.65 * h) and s_area <= int(0.40 * image_area):
                    split_components.append((f"{k}_{sk}", s_area, s_bw, s_bh, (sub_labels == sk).astype(np.uint8)))
                    recovered += 1

            if recovered > 0:
                print(f"  Mega component k={k} (area={area}, bb={bb_w}x{bb_h}) SPLIT into {recovered} valid individual building footprints!")
            else:
                rejected_components.append((k, area, bb_w, bb_h))
                print(f"  Mega component k={k} (area={area}, bb={bb_w}x{bb_h}) REJECTED as non-splittable background mass.")

    print(f"Final valid buildings count: {len(split_components)}")

if __name__ == "__main__":
    test_morphological_split("SV_NewYork_40.7401_-73.9915.tif")
    test_morphological_split("SV_NewYork_40.7333_-73.9835.tif")
