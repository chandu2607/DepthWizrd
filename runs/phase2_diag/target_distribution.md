# Phase-2 Diagnosis — target-height distribution & composition

_Read-only analysis (no model). source=`hf_blocks`, tile_size=512, building_label=6. Groups match run_phase1's city split._

Observed Baseline-C prediction ceiling under diagnosis: **~14 m**.

## Pixel composition

| group | tiles | total px | ground(≤0.5m) frac | building px | building frac |
|---|--:|--:|--:|--:|--:|
| JAX_train | 102 | 26,738,688 | 0.519 | 4,450,262 | 0.166 |
| JAX_val | 18 | 4,718,592 | 0.465 | 1,204,917 | 0.255 |
| OMA_test | 120 | 31,457,280 | 0.672 | 4,026,094 | 0.128 |

## ALL pixels: height statistics (m)

| group | mean | median | std | max | p50 | p75 | p90 | p95 | p99 | p99.9 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| JAX_train | 4.72 | 0.26 | 9.36 | 186.5 | 0.3 | 7.0 | 13.8 | 18.1 | 34.1 | 132.0 |
| JAX_val | 5.88 | 1.19 | 9.64 | 109.5 | 1.2 | 8.5 | 17.2 | 21.0 | 47.0 | 81.5 |
| OMA_test | 2.24 | 0.01 | 4.88 | 49.3 | 0.0 | 2.7 | 7.7 | 10.8 | 24.7 | 42.5 |

### ALL pixels: P(height > threshold)

| group | >5m | >10m | >15m | >20m | >30m | >40m | >14m (ceiling) |
|---|--:|--:|--:|--:|--:|--:|--:|
| JAX_train | 32.09% | 17.01% | 8.31% | 3.55% | 1.26% | 0.66% | 9.66% |
| JAX_val | 36.60% | 22.47% | 13.62% | 6.48% | 2.23% | 1.25% | 15.15% |
| OMA_test | 17.00% | 6.08% | 2.24% | 1.20% | 0.69% | 0.25% | 2.63% |

## BUILDING pixels: height statistics (m)

| group | mean | median | std | max | p50 | p75 | p90 | p95 | p99 | p99.9 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| JAX_train | 12.01 | 7.16 | 17.41 | 186.5 | 7.2 | 13.4 | 23.5 | 36.2 | 97.8 | 171.3 |
| JAX_val | 13.74 | 10.45 | 13.95 | 109.7 | 10.5 | 18.6 | 27.9 | 38.6 | 73.9 | 94.6 |
| OMA_test | 7.45 | 4.94 | 8.81 | 44.3 | 4.9 | 8.2 | 16.0 | 30.5 | 42.1 | 43.5 |

### BUILDING pixels: P(height > threshold)

| group | >5m | >10m | >15m | >20m | >30m | >40m | >14m (ceiling) |
|---|--:|--:|--:|--:|--:|--:|--:|
| JAX_train | 70.28% | 34.17% | 20.95% | 12.52% | 7.08% | 3.82% | 22.89% |
| JAX_val | 73.60% | 51.25% | 35.83% | 20.44% | 8.35% | 4.81% | 38.71% |
| OMA_test | 49.39% | 20.01% | 11.15% | 7.88% | 5.23% | 1.90% | 11.75% |

## Diagnosis-relevant summary

- JAX-train **building** pixels above the ~14 m ceiling: **22.9%** — height mass a 14 m-saturating model cannot express.
- JAX-train building median height: 7.2 m, p99: 97.8 m, max: 186.5 m.