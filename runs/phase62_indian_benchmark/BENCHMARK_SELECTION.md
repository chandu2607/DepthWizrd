# Phase 62 — Benchmark Selection

## CURRENT STATUS

The project has not acquired a final valid Indian benchmark yet.

## Candidate region selected for investigation

- Region: Uttarakhand
- Primary interest: Chamoli / Joshimath / nearby hilly terrain
- Why: sustained hill terrain, steep relief, roads, settlements, valleys, and strong disaster relevance

## Candidate data pairing under review

- Optical source: Sentinel-2 Level-2A (or Bhuvan/NRSC orthophoto if a paired regional product is available)
- Elevation reference: Copernicus DEM GLO-30 or an Indian CartoDEM / Bhuvan DEM if accessible

## Why this is not yet accepted

This candidate is a strong scientific candidate, but it is not yet validated as a benchmark because the following remain unverified in this workspace:

- exact paired optical + elevation tile availability
- CRS consistency
- geographic overlap
- pixel grid alignment
- valid data coverage
- deposition and access conditions for local Indian products

## Benchmark acceptance criteria

A benchmark is accepted only if all are true:

1. real optical image exists for a suitable Indian mountainous region
2. real DEM/DSM/DTM exists for the same extent
3. the two layers can be aligned spatially
4. the tile is small enough for a pilot benchmark
5. a geographic train/val/test split is possible
6. no same-scene leakage occurs across splits

## Final decision

No final benchmark is accepted yet. The current status is:

BENCHMARK_STATUS = VALIDATION_PENDING

This is the correct stopping point before a full training phase.
