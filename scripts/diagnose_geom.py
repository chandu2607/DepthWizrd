import sys
from pathlib import Path
import numpy as np
import cv2 as cv

sys.path.insert(0, ".")
from depthwizard.data.raster_loader import load_raster_input
from depthwizard.config import DepthConfig
from depthwizard.depth.depth_anything import DepthAnythingV2
from depthwizard.calibration import CalibrationEngine, CalibrationMode
from depthwizard.viz.interactive_viewer import build_city_geometry

def main():
    scene = "data/dfc2023_multicity/rgb/SV_NewYork_40.7333_-73.9835.tif"
    dcfg = DepthConfig(cache_dir="data/dfc2023_multicity/depth_cache")
    depth_model = DepthAnythingV2(dcfg.model_id, dcfg.input_size, dcfg.cache_dir, use_cache=True)
    calib_engine = CalibrationEngine(runs_dir=Path("runs"))
    
    raster_in = load_raster_input(scene, filename="SV_NewYork_40.7401_-73.9915.tif")
    h, w = raster_in.shape
    depth_raw = depth_model.infer(raster_in.rgb, "SV_NewYork_40.7401_-73.9915.tif", target_hw=(h, w))
    calib_res = calib_engine.calibrate(
        depth_raw, raster_in.rgb, is_georeferenced=raster_in.is_georeferenced,
        mode=CalibrationMode.STRUCTURAL_PRIOR, reference_elevation=None,
        filename="SV_NewYork_40.7401_-73.9915.tif"
    )
    
    dsm = calib_res.dsm
    dtm = calib_res.dtm
    mask = calib_res.mask_bldg
    
    print(f"DSM range: {dsm.min():.1f} to {dsm.max():.1f} m")
    print(f"DTM range: {dtm.min():.1f} to {dtm.max():.1f} m")
    print(f"Building mask: {mask.sum()} pixels, {mask.sum()/(h*w)*100:.1f}% coverage")

    h2, w2 = mask.shape
    num_l, labels, stats, centroids = cv.connectedComponentsWithStats(mask.astype(np.uint8))
    img_area = float(h2 * w2)
    print(f"Total connected components: {num_l-1}")
    for k in range(1, min(num_l, 35)):
        area = stats[k, cv.CC_STAT_AREA]
        bb_w = stats[k, cv.CC_STAT_WIDTH]
        bb_h = stats[k, cv.CC_STAT_HEIGHT]
        reject = ""
        if bb_w > 0.65 * w2:
            reject = f"MEGA:bb_w={bb_w}"
        elif bb_h > 0.65 * h2:
            reject = f"MEGA:bb_h={bb_h}"
        elif area > 0.40 * img_area:
            reject = f"MEGA:area={area}"
        print(f"  k={k:2d}: area={area:6d} bb={bb_w:3d}x{bb_h:3d} {reject}")

    geom = build_city_geometry(raster_in.rgb, dsm, dtm, mask, gsd=0.5, exaggeration=1.5, stride=4)
    bldgs = geom["buildings"]
    rej = geom["rejected"]
    print(f"\nValid buildings: {len(bldgs)}")
    print(f"Rejected components: {len(rej)}")
    for r in rej:
        print(f"  REJECTED k={r['component_id']}: {r['rejection_reason']}")
    print("\nTop 10 Tallest Buildings:")
    for b in sorted(bldgs, key=lambda x: -x["height_m"])[:10]:
        print(f"  id={b['id']:2d} h={b['height_m']:5.1f}m area={b['area_m2']:6.0f}m2 z_roof={b['z_roof']:5.1f}m z_grd={b['z_ground']:5.1f}m cx={b['cx']:6.1f} cz={b['cz']:6.1f}")

if __name__ == "__main__":
    main()
