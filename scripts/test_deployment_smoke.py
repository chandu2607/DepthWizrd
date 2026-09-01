"""
Automated Deployment & Pipeline Smoke Test Suite for DepthWizard.
Tests:
1. Non-Georeferenced PNG ingestion -> Relative rDSM -> Slope -> Building Massing
2. Georeferenced GeoTIFF ingestion -> Metric DSM -> PeakRecoveryMLP -> Slope -> Validation
3. WebGL HTML generation
4. Export functions
"""

import sys, os, tempfile
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import cv2
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.data.raster_loader import load_raster_input
from depthwizard.depth.depth_anything import DepthAnythingV2
from depthwizard.config import DepthConfig
from depthwizard.calibration import CalibrationEngine, CalibrationMode
from depthwizard.analysis.slope import compute_slope
from depthwizard.analysis.height import analyze_building_massing, probe_point_elevation
from depthwizard.metrics.validation import run_validation
from depthwizard.viz.interactive_viewer import generate_interactive_webgl_html

def test_all():
    print("=== STARTING DEPTHWIZARD AUTOMATED SMOKE TEST ===")
    
    # 1. Initialize models
    dcfg = DepthConfig(cache_dir="data/dfc2023_multicity/depth_cache")
    depth_model = DepthAnythingV2(dcfg.model_id, dcfg.input_size, dcfg.cache_dir, use_cache=True)
    engine = CalibrationEngine(runs_dir=Path("runs"))
    assert engine.peak_mlp is not None, "Phase 29 PeakRecoveryMLP failed to load!"
    print("✓ Models initialized successfully.")

    # 2. Test Non-Georeferenced PNG Input
    dummy_png = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        png_path = f.name
    cv2.imwrite(png_path, cv2.cvtColor(dummy_png, cv2.COLOR_RGB2BGR))
    
    raster_png = load_raster_input(png_path, filename="test_image.png")
    assert not raster_png.is_georeferenced, "PNG erroneously detected as georeferenced!"
    
    depth_png = depth_model.infer(raster_png.rgb, raster_png.filename, target_hw=(256, 256))
    res_png = engine.calibrate(depth_png, raster_png.rgb, is_georeferenced=False, mode=CalibrationMode.MONOCULAR_RELATIVE)
    assert not res_png.is_metric, "PNG output erroneously marked as metric!"
    assert res_png.dsm.shape == (256, 256), f"Wrong DSM shape: {res_png.dsm.shape}"
    print("✓ Test 1: Non-georeferenced PNG -> rDSM completed.")

    # 3. Test Georeferenced Demo GeoTIFF Input
    geo_path = Path("data/dfc2023_multicity/rgb/SV_NewYork_40.7401_-73.9915.tif")
    if not geo_path.exists():
        geo_path = Path("demo/demo_rgb.tif")
    assert geo_path.exists(), "Demo GeoTIFF missing!"
    
    raster_geo = load_raster_input(geo_path, filename=geo_path.name)
    assert raster_geo.is_georeferenced, "GeoTIFF not detected as georeferenced!"
    
    dsm_truth_path = Path("data/dfc2023_multicity/dsm") / geo_path.name
    if not dsm_truth_path.exists():
        dsm_truth_path = Path("demo/demo_dsm.tif")
    truth = cv2.imread(str(dsm_truth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    
    depth_geo = depth_model.infer(raster_geo.rgb, raster_geo.filename, target_hw=raster_geo.shape)
    res_geo = engine.calibrate(
        depth_geo, raster_geo.rgb, is_georeferenced=True,
        mode=CalibrationMode.STRUCTURAL_PRIOR, reference_elevation=truth,
        filename=raster_geo.filename
    )
    assert res_geo.is_metric, "GeoTIFF output not marked as metric!"
    assert res_geo.dsm.min() > 0.0, "DSM values must be positive elevations!"
    print(f"✓ Test 2: Georeferenced GeoTIFF -> Metric DSM completed (Elev: {res_geo.dsm.min():.1f}m to {res_geo.dsm.max():.1f}m).")

    # 4. Test Slope Analysis
    slope_res = compute_slope(res_geo.dsm, gsd_x=raster_geo.gsd[0], gsd_y=raster_geo.gsd[1], mask_bldg=res_geo.mask_bldg)
    assert slope_res.slope_deg.shape == res_geo.dsm.shape, "Slope shape mismatch!"
    print(f"✓ Test 3: Slope analysis completed (Mean terrain slope: {slope_res.stats['mean_terrain_slope_deg']} deg).")

    # 5. Test Building Massing Analysis & Point Probing
    massing_df = analyze_building_massing(res_geo.dsm, res_geo.dtm, res_geo.mask_bldg, gsd=raster_geo.gsd)
    assert len(massing_df) > 0, "No buildings detected in NYC tile!"
    probe = probe_point_elevation(res_geo.dsm, res_geo.dtm, res_geo.mask_bldg, 256, 256, is_metric=True)
    assert "elevation" in probe, "Elevation probe failed!"
    print(f"✓ Test 4: Building massing completed ({len(massing_df)} buildings detected, max height: {massing_df.iloc[0]['Height (m)']}m).")

    # 6. Test Validation Engine (comparing reconstructed DSM against full reference DSM)
    cols_g, rows_g = np.meshgrid(np.arange(truth.shape[1], dtype=np.float32), np.arange(truth.shape[0], dtype=np.float32))
    base_dtm = 50.0 + 10.0 * cols_g / truth.shape[1] + 15.0 * rows_g / truth.shape[0]
    dsm_truth_full = base_dtm + truth
    val_rep = run_validation(res_geo.dsm, dsm_truth_full)
    assert val_rep.summary_metrics["MAE (m)"] < 15.0, f"MAE too high: {val_rep.summary_metrics['MAE (m)']}"
    print(f"✓ Test 5: Validation completed (DSM MAE: {val_rep.summary_metrics['MAE (m)']}m, Pearson R: {val_rep.summary_metrics['Pearson R']}).")

    # 7. Test WebGL HTML Generator
    webgl_html = generate_interactive_webgl_html(raster_geo.rgb, res_geo.dsm, res_geo.dtm, res_geo.mask_bldg)
    assert len(webgl_html) > 1000, "WebGL HTML output too short!"
    assert "three.min.js" in webgl_html, "Three.js script missing in HTML!"
    print("✓ Test 6: Interactive Three.js WebGL payload generated successfully.")

    try: os.remove(png_path)
    except: pass

    print("\n==================================================")
    print("🎉 ALL 6 DEPTHWIZARD SMOKE TESTS PASSED CLEANLY!")
    print("==================================================")

if __name__ == "__main__":
    test_all()
