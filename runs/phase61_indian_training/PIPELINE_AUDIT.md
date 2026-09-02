# Phase 61 — Existing Model and Training Pipeline Audit

## Current production path

The active repository implementation remains the same as in prior phases:

Input RGB
 ↓
Raster loading and georeference detection
 ↓
Depth Anything V2 relative-depth prior
 ↓
Depth normalization
 ↓
CalibrationEngine.calibrate()
 ↓
DSM / rDSM generation
 ↓
Slope analysis and structural analysis
 ↓
3D WebGL export / GeoTIFF outputs

## Actual implementation references

- [app.py](../../app.py)
- [depthwizard/depth/depth_anything.py](../../depthwizard/depth/depth_anything.py)
- [depthwizard/calibration/engine.py](../../depthwizard/calibration/engine.py)
- [depthwizard/analysis/slope.py](../../depthwizard/analysis/slope.py)
- [depthwizard/config.py](../../depthwizard/config.py)

## Important conclusions

### Model architecture
The project still uses a frozen Depth Anything V2 backbone and a calibration engine that converts relative depth into a reconstructed surface. It is not a terrain-native architecture yet.

### Current model state
The current model is not proven on Indian mountainous terrain. The repository contains no real Indian benchmark tile yet, so no real Indian training or evaluation evidence exists.

### Training-capable path
The repository contains model training infrastructure and experiment configs, but the Phase 61 objective is to discover an actual Indian benchmark first and only then decide whether a controlled fine-tuning experiment is justified.

### Scientific gate
We are stopping before full training and before any model adaptation. This matches the user instruction: start with dataset discovery and pilot benchmark preparation only.
