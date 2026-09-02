# Phase 31C — 3D Reconstruction Integrity Audit

## 2. DSM Local-Gradient Audit (skyscraper-heavy)

| Metric | X | Y |
|--------|---|---|
| Median | 0.287 m | 0.336 m |
| P95 | 1.06 m | 1.29 m |
| P99 | 2.27 m | 2.22 m |
| Max | 34.43 m | 34.95 m |

|dZ|>1m: 8.0%  |dZ|>2m: 1.2%  |dZ|>5m: 0.8%  |dZ|>10m: 0.6%

## 3. Spike Audit

| Tile | P95 | P99 | Max | frac>5m |
|------|-----|-----|-----|---------|
| skyscraper-heavy | 0.07 m | 0.67 m | 33.74 m | 0.33% |
| dense-highrise | 0.02 m | 0.53 m | 24.54 m | 0.29% |
| lower-rise | 0.01 m | 0.16 m | 11.49 m | 0.10% |

## 4. Jump Location (skyscraper-heavy)

At building boundaries: 100.0%
Inside flat areas: 0.0%

## Primary Diagnosis

MIXED_DSM_AND_MESH

The DSM is not globally noisy (P95 spike = 0.07m).
However abrupt building-edge discontinuities of up to 34.43m/pixel
cause StructuredGrid bilinear faces to become near-vertical curtain walls.
100% of outlier pixels are at building boundaries; 0% inside flat areas.
The texture mapping is geometrically correct; the problem is the vertical faces it is mapped onto.

## Recommended Next Action

REPAIR_MESH_PIPELINE

Do NOT implement yet.
