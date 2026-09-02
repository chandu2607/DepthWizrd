# Phase 60 — Real Indian Hilly Benchmark + Baseline Inference

## Executive summary

This phase did not reach a real Indian terrain inference result because no valid benchmark was accepted.

The truth is straightforward:

- the repository does contain a real production inference path,
- that path is well-known from the app and calibration engine,
- but no actual Indian mountainous benchmark tile was identified and validated in this workspace,
- therefore no real Indian baseline inference or metric comparison was performed,
- the project remains in the evidence-gate stage.

---

## Provenance from the current repository

The live production path is:

Input
 ↓
RGB preprocessing
 ↓
Depth Anything V2 relative-depth estimation
 ↓
Depth normalization
 ↓
CalibrationEngine.calibrate()
 ↓
DSM / rDSM production
 ↓
Slope map and structural analysis
 ↓
3D WebGL scene export

This is the actual stack present in the project.

---

## What was found

### 1. Actual model/inference path

The current app uses:

- `DepthAnythingV2.infer()` from `depthwizard/depth/depth_anything.py`
- `CalibrationEngine.calibrate()` from `depthwizard/calibration/engine.py`
- `compute_slope()` from `depthwizard/analysis/slope.py`
- WebGL render generation from the viewer utilities

This is the living production path, not a theoretical pipeline.

### 2. Benchmark availability

Public sources exist for Indian mountainous regions, including:

- Copernicus DEM,
- SRTM,
- Bhuvan/NRSC geospatial products,
- OpenTopography where data hosting is available,
- Sentinel-2 optical coverage.

However, none of these were accepted as a real benchmark in this workspace because a valid paired optical + reference DEM/DSM tile was not selected and aligned.

### 3. Scientific condition

No valid benchmark means no valid India result.

This phase therefore concludes:

- real inference on Indian terrain: NO
- real metric evaluation on Indian terrain: NO
- Indian validation claim: NO

---

## Final verdict

### PHASE 60 STATUS:
BENCHMARK: NO_VALID_INDIAN_BENCHMARK_FOUND
REGION: NONE
OPTICAL SOURCE: NONE
ELEVATION SOURCE: NONE
CHECKPOINT: Repository current checkpoint stack only; no real Indian run performed
REAL INFERENCE: NO
REAL METRICS: NO
INDIAN VALIDATION: NO
DOMINANT FAILURE: NO_VALID_BENCHMARK
CANNY ROLE: DIAGNOSTIC ONLY
POINT CLOUD ROLE: DOWNSTREAM GEOMETRY AID ONLY
RETRAINING PERFORMED: NO
ARCHITECTURE CHANGED: NO
FINAL VERDICT: THE PROJECT HAS NOT YET REACHED A VALID INDIAN MOUNTAIN BASELINE EVALUATION.

---

## Why this is the correct final state

The user instruction is explicit: no placeholder values, no fabricated metrics, no disguised success. This is exactly why the phase ends with a benchmark gate rather than a fake result.

The next correct step is to obtain one real Indian mountainous tile with a paired elevation reference, validate the alignment, and only then run the current model unchanged and analyze the genuine error.
