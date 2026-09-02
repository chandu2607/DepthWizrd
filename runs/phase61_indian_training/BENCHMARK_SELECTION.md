# Phase 61 — Pilot Indian Benchmark Selection

## Selection status

This file marks the current benchmark design as a pilot plan, not a completed dataset selection. No real Indian benchmark has yet been accepted for training.

## Proposed pilot configuration

### Training region(s)
- Uttarakhand (primary candidate)
- Himachal Pradesh (secondary candidate)

### Validation region
- another geographically separate mountain region in Uttarakhand or Himachal

### Held-out test region
- Sikkim or a different Indian mountainous region with a clear geographic separation from train/validation

## Why these regions

These regions are the most defensible because they are representative of Indian mountainous terrain, have public geospatial coverage, and are relevant to hill-state terrain and disaster analysis. They also allow a clear train/validation/test split by region rather than by random tile.

## Data source candidates

- Sentinel-2 optical imagery
- Copernicus DEM or SRTM as reference terrain
- Bhuvan / NRSC products where necessary for local coverage
- OpenTopography if a hosted LiDAR or DSM is available for a selected site

## Benchmark acceptance criteria

A benchmark is accepted only if it satisfies all of the following:

- real optical imagery exists for a mountain region
- real DEM/DSM/DTM exists
- CRS and spatial overlap are compatible
- the tile is small enough for a pilot benchmark
- geographic splits remain independent
- no same-scene leakage across train/val/test

## Current decision

No candidate has yet been accepted as the final real Indian training benchmark. This phase remains data-discovery and benchmark preparation only.

## Known limitations

- Most public datasets are coarse or require manual alignment.
- High-quality Indian mountain benchmarks are not guaranteed to exist in a single public portal.
- We must not fabricate a benchmark or claim success without real alignment and evaluation.

## Required next step

The next step is a single pilot region selection with alignment checks, not full training. The first accepted tile will be used for baseline inference and pilot runtime assessment only.
