# Phase-4 §10 diagnostic — calibrated tail weight vs aggressive weight

_Read-only, TRAINING split ONLY (['JAX']); the test city (['OMA']) is untouched (§8/§26). source=`hf_blocks`, subsample stride=13._

## JAX-train height distribution → parameter derivation (§8/§9)

| percentile | all-pixel height (m) | building-pixel height (m) |
|---|--:|--:|
| P50 | 0.25 | 7.16 |
| P75 | 6.98 | 13.36 |
| P85 | 10.99 | 18.42 |
| P90 | 13.77 | 23.47 |
| P92 | 15.27 | 28.45 |
| P95 | 18.11 | 36.17 |
| P99 | 34.09 | 97.76 |

- **h_start = 15 m** — onset of the sparse tail: 91.7% of ALL and 79.1% of BUILDING training pixels sit at ≤ 15 m (≈ P79 building), and it coincides with the observed ~14 m learned ceiling. Below it: abundant, well-sampled → ordinary weight (w=1). Building-pixel median = 7.16 m (well inside the protected regime).
- **w_max = 3** — deliberately gentler than the aggressive cap (5); a tall pixel counts at most 3× a ground pixel, so rare extremes cannot dominate the gradient (§11).
- **tail_scale = 12.5 m** — the ramp spans the tail from h_start to the cap at h = h_start+(w_max−1)·scale = 40 m (≈ P99 all-pixel); heights beyond 40 m (the extreme 0.7%) are clamped.

## Weight definition (Phase-4, calibrated tail)

```
w(h) = 1                                          for h <= h_start
     = min(1 + (h - h_start)/tail_scale, w_max)   for h  > h_start
[h_start=15 m, tail_scale=12.5 m, w_max=3]
```
- basis: **physical height (m)**, BEFORE log1p → transform-agnostic, metric-space 'tall matters more'.
- flat w=1 through the protected 0–15 m regime; continuous at h_start (no jump); bounded in [1, 3].

## Weight at representative heights (tail vs aggressive)

| height (m) | 0 | 2 | 5 | 10 | 14 | 15 | 18 | 20 | 25 | 30 | 40 | 50 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| tail w(h) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.24 | 1.40 | 1.80 | 2.20 | 3.00 | 3.00 |
| aggressive w(h) | 1.00 | 1.29 | 1.71 | 2.43 | 3.00 | 3.14 | 3.57 | 3.86 | 4.57 | 5.00 | 5.00 | 5.00 |

## Weight statistics over training valid pixels

| weight · population | n (subsampled) | min w | max w | mean w | median w |
|---|--:|--:|--:|--:|--:|
| tail · ALL | 2,056,830 | 1.000 | 3.000 | 1.047 | 1.000 |
| tail · BUILDING | 342,378 | 1.000 | 3.000 | 1.187 | 1.000 |
| aggressive · ALL | 2,056,830 | 1.000 | 5.000 | 1.620 | 1.036 |
| aggressive · BUILDING | 342,378 | 1.000 | 5.000 | 2.399 | 2.023 |

_Mean tail-w over ALL pixels = **1.047** vs aggressive **1.620** (weighted-MEAN normalizes by Σw, so loss magnitude ≈ unweighted → effective LR unchanged; only relative emphasis moves)._

## Rebalancing check — pixel share vs LOSS-MASS share by height bin (ALL px)

_Loss-mass share = Σw in bin / Σw total. The calibration goal: the 0–15 m mass share should stay ≈ its pixel share (regime protected), while the tall bins rise — a far smaller shift than the aggressive weight imposes._

| GT bin (m) | pixel % | tail mass % | tail Δ | aggressive mass % | aggr Δ |
|---|--:|--:|--:|--:|--:|
| 0–2 | 57.44% | 54.85% | -2.59% | 36.17% | -21.27% |
| 2–5 | 10.47% | 10.00% | -0.47% | 9.79% | -0.68% |
| 5–10 | 15.09% | 14.41% | -0.68% | 18.98% | +3.89% |
| 10–15 | 8.69% | 8.30% | -0.39% | 14.79% | +6.11% |
| 15–20 | 4.76% | 5.36% | +0.59% | 10.17% | +5.41% |
| 20–30 | 2.29% | 3.74% | +1.44% | 6.21% | +3.92% |
| 30–40 | 0.60% | 1.47% | +0.87% | 1.84% | +1.24% |
| 40–inf | 0.66% | 1.89% | +1.23% | 2.04% | +1.38% |

_Total subsampled ALL pixels: 2,056,830._

## §10 must-prove — does it target the tail WITHOUT disturbing 0–15 m?

- **0–15 m regime** (pixel share 91.68%): loss-mass share under the **tail** weight = **87.55%** (Δ -4.14%) vs under the **aggressive** weight = 79.74% (Δ -11.94%).
- The tail weight shifts the protected regime's emphasis by only 4.14% (vs 11.94% for the aggressive weight) → **YES**: it primarily targets the >15 m tail while leaving the 0–15 m optimization emphasis essentially intact.
- tall tail (>15 m) loss-mass share: tail = 12.45% vs pixel share 8.32% (emphasis raised, but bounded).
