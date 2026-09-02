# Phase 60 — Point Cloud Feasibility Check

## Scope

This is a feasibility check on whether a point-cloud representation improves geometry continuity or visualization, not a claim that point clouds are a replacement for correct estimation.

## Observation from the current repo

The active production implementation is raster-based, calibration-based, and WebGL-rendered. There is no validated point-cloud computation in the current inference path that is used as the primary terrain estimate.

## Distinction required

### Height estimation
This is the core problem. It is driven by the optical prior, calibration, and reference terrain logic.

### 3D representation
This is a downstream geometric representation. It can improve visualization or continuity, but it cannot correct a bad height estimate.

## Conclusion

A point cloud may be useful for:

- geometry visualization
- improving mesh continuity
- smoother 3D presentation
- reducing artifacts in the display layer

But it is not a substitute for actual metric height estimation on steep terrain.

## Decision

Point clouds are a downstream geometry representation aid, not the core solution to the Indian terrain problem.
