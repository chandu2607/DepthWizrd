# Phase 72 Common-Grid / DEM Validity Forensic

PHASE 72 STATUS: COMMON_GRID_FORENSIC

UTTARAKHAND:
    RGB_VALID: True
    DEM_VALID: True
    JOINT_VALID: True
    OVERLAP: checked
    CRS: optical=EPSG:32644; DEM=EPSG:4326; common=EPSG:32644
    COMMON_GRID: reprojected DEM onto optical grid
    TARGET_VALID: True

HIMACHAL:
    RGB_VALID: True
    DEM_VALID: True
    JOINT_VALID: True
    OVERLAP: checked
    CRS: optical=EPSG:32643; DEM=EPSG:4326; common=EPSG:32643
    COMMON_GRID: reprojected DEM onto optical grid
    TARGET_VALID: True

SIKKIM:
    RGB_VALID: True
    DEM_VALID: True
    JOINT_VALID: True
    OVERLAP: checked
    CRS: optical=EPSG:32645; DEM=EPSG:4326; common=EPSG:32645
    COMMON_GRID: reprojected DEM onto optical grid
    TARGET_VALID: True

METRIC_IDENTITY_TEST: identity and +1m/+10m checks passed numerically
NORMALIZATION_TEST: train-only normalization computed on Uttarakhand and kept fixed for validation/test

FIRST_FAILURE: phase71 zero valid pixels caused by CRS mismatch and center-crop before common-grid reprojection
ROOT_CAUSE: raw DEM and optical arrays were in different geospatial frames; the target mask was empty until the DEM was projected onto the optical grid

TERRAIN_TRAINING_READY: False
NO_TRAINING_PERFORMED: True
NO_ARCHITECTURE_CHANGE: True
NO_PRODUCTION_CHANGE: True

NEXT_PHASE: validate all common-grid targets for all regions and only then re-enter a minimal training run under a strict lock policy.
