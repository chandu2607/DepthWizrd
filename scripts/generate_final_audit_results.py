"""
Generate RESULTS.json and final audit metrics for runs/phase35_final_audit/
"""
import json, hashlib
import sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
import cv2

from depthwizard.config import DepthConfig
from depthwizard.depth.depth_anything import DepthAnythingV2
from depthwizard.data.raster_loader import load_raster_input
from depthwizard.calibration import CalibrationEngine, CalibrationMode
from depthwizard.viz.interactive_viewer import build_city_geometry

dcfg = DepthConfig(cache_dir="data/dfc2023_multicity/depth_cache")
depth_model = DepthAnythingV2(dcfg.model_id, dcfg.input_size, dcfg.cache_dir, use_cache=True)
calib_engine = CalibrationEngine(runs_dir=Path("runs"))

test_tiles = [
    ("NYC_skyscraper_heavy", "data/dfc2023_multicity/rgb/SV_NewYork_40.7401_-73.9915.tif"),
    ("NYC_dense_highrise", "data/dfc2023_multicity/rgb/SV_NewYork_40.7372_-73.9901.tif"),
    ("NYC_lower_rise", "data/dfc2023_multicity/rgb/SV_NewYork_40.7373_-74.0034.tif"),
]

results = {}

for name, tile_path in test_tiles:
    p = Path(tile_path)
    raster_in = load_raster_input(str(p))
    depth_raw = depth_model.infer(raster_in.rgb, raster_in.filename, target_hw=raster_in.shape)
    dsm_truth_path = Path("data/dfc2023_multicity/dsm") / p.name
    truth = cv2.imread(str(dsm_truth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) if dsm_truth_path.exists() else None
    
    res = calib_engine.calibrate(
        depth_raw, raster_in.rgb, is_georeferenced=raster_in.is_georeferenced,
        mode=CalibrationMode.STRUCTURAL_PRIOR, reference_elevation=truth,
        filename=raster_in.filename
    )
    
    geom = build_city_geometry(
        raster_in.rgb, res.dsm, res.dtm, res.mask_bldg,
        gsd=raster_in.gsd or 0.5, exaggeration=1.0, stride=4
    )
    
    dsm_bytes = res.dsm.tobytes()
    dsm_hash = hashlib.sha256(dsm_bytes).hexdigest()
    
    results[name] = {
        "num_buildings": len(geom["buildings"]),
        "terrain_faces": geom["terrain"]["n_faces"],
        "roof_faces": geom["roofs"]["n_faces"],
        "wall_faces": geom["walls"]["n_faces"],
        "bounds": geom["bounds"],
        "scientific_dsm": {
            "sha256_hash": dsm_hash,
            "min_m": round(float(res.dsm.min()), 2),
            "max_m": round(float(res.dsm.max()), 2),
            "mean_m": round(float(res.dsm.mean()), 2),
            "p95_m": round(float(np.percentile(res.dsm, 95)), 2)
        }
    }
    print(f"Computed {name}: {len(geom['buildings'])} buildings, DSM hash={dsm_hash[:12]}...")

out_file = Path("runs/phase35_final_audit/RESULTS.json")
out_file.parent.mkdir(parents=True, exist_ok=True)
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("Saved RESULTS.json successfully!")
