# Module Map — DepthWizard Unified Package

```
depthwizard/
│
├── __init__.py                     # Package entry point
├── config.py                       # Global pipeline & model configurations
│
├── data/                           # Data ingestion & formats
│   ├── __init__.py
│   ├── raster_loader.py            # Unified PNG/JPG/TIFF/GeoTIFF loader + metadata parser
│   └── datasets.py                 # DFC2023 multicity dataset abstractions
│
├── depth/                          # Pretrained backbone & caching
│   ├── __init__.py
│   └── depth_anything.py           # Depth Anything V2 wrapper with MD5 cache
│
├── calibration/                    # Modular metric calibration engine
│   ├── __init__.py
│   ├── engine.py                   # Master calibration dispatcher
│   ├── monocular_relative.py       # Relative 0-10 rDSM calibration
│   ├── dem_anchored.py             # DTM + Coarse DEM anchoring
│   ├── ground_referenced.py        # Local DTM ground plane referencing
│   ├── gcp_anchored.py             # Robust GCP transformation estimator
│   └── structural_prior.py         # Phase 29 PeakRecoveryMLP ensemble
│
├── analysis/                       # Height & slope analytics
│   ├── __init__.py
│   ├── height.py                   # Point-to-point and building massing analysis
│   └── slope.py                    # Gradient slope (deg, %), aspect, terrain separation
│
├── metrics/                        # Quantitative validation
│   ├── __init__.py
│   ├── height_metrics.py           # MAE, RMSE, Pearson, Spearman, Bias, P90
│   └── validation.py               # Comprehensive validation report generator
│
└── viz/                            # 3D reconstruction & interactive viewers
    ├── __init__.py
    ├── mesh_generator.py           # Hybrid DTM + Architectural Walls + Roof mesh builder
    ├── interactive_viewer.py       # Three.js WebGL interactive viewer (Orbit/WASD/Flythrough)
    └── static_renderer.py          # PyVista headless renderer for static exports
```

---

## Script & Experiment Organization
```
scripts/
│
├── run_phase29_peak_recovery.py    # Production PeakRecoveryMLP training & evaluation
├── run_phase30_terrain_dtm.py      # Production DTM extraction pipeline
├── run_phase33d_final_visual_audit.py # Visual regression test suite
├── run_phase34_pseudolidar.py      # Phase 34 pseudo-LiDAR diagnostic
│
└── experiments/professor_methods/  # Consolidated Professor Methods Diagnostic Suite
    ├── 01_pseudo_lidar.py          # Pseudo-3D lifting & geometry probe
    ├── 02_sparse_depth_completion.py # Simulated sparse true-metric depth completion probe
    ├── 03_ground_plane_calibration.py # RANSAC ground-plane metric calibration probe
    └── 04_multiview_geometry_probe.py # Multi-view parallax & scale feasibility report
```
