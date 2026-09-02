# Phase 62 — Alignment Validation Plan

## Status

The benchmark has not yet been downloaded or aligned. This file records the required alignment process and current status.

## Required checks

For the selected pilot tile, the following must be checked before any baseline inference:

- common CRS between optical and reference layers
- geotransform / affine consistency
- pixel size agreement
- width/height compatibility
- geographic bounds overlap
- orientation consistency
- nodata handling
- overlap proportion after reprojection/resampling

## Allowed preprocessing

- reproject the reference to the optical grid when georeferencing is compatible
- resample continuous elevation with bilinear/cubic interpolation where appropriate
- use nearest-neighbor for masks if binary masks are part of the benchmark
- never silently modify source rasters

## Current outcome

All checks remain pending because no actual benchmark tile has been acquired and aligned in this workspace.

## Scientific conclusion

No real Indian baseline is computed until the benchmark passes alignment checks.
