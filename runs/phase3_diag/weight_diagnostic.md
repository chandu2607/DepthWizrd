# Phase-3 §10 diagnostic — height → training loss weight

_Read-only, TRAINING split ONLY (['JAX']); the test city (['OMA']) is untouched (§4). source=`hf_blocks`, subsample stride=13._

## Weight definition

```
w(h) = min(1 + max(h,0)/scale, w_max)   [scale=7 m, w_max=5]
```
- basis: **physical height (meters)**, computed BEFORE the log1p transform → transform-agnostic; encodes metric-space 'tall matters more'.
- ground (h=0) → w=1 (rebalanced, never eliminated); saturates at w_max for h ≥ 28 m.
- scale ≈ JAX-train building-pixel median (7.16 m, runs/phase2_diag) → training-derived, no leakage.

## Weight statistics over training valid pixels

| population | n (subsampled) | min w | max w | mean w | median w |
|---|--:|--:|--:|--:|--:|
| ALL pixels | 2,056,830 | 1.000 | 5.000 | 1.620 | 1.036 |
| BUILDING pixels | 342,378 | 1.000 | 5.000 | 2.399 | 2.023 |

_Mean w over ALL pixels = **1.620** — the weighted mean divides by Σw, so the loss magnitude stays ~unweighted (effective LR unchanged; the only change is relative emphasis)._

## Weight at representative heights

| height (m) | 0 | 2 | 5 | 10 | 15 | 20 | 30 | 40 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| weight w(h) | 1.00 | 1.29 | 1.71 | 2.43 | 3.14 | 3.86 | 5.00 | 5.00 |

## Fraction of training pixels by weight range

| weight range | ALL pixels | BUILDING pixels |
|---|--:|--:|
| [1, 1.5) | 62.01% | 16.25% |
| [1.5, 2) | 13.05% | 31.95% |
| [2, 3) | 15.28% | 28.92% |
| [3, 4) | 6.48% | 10.77% |
| [4, 5) | 1.67% | 3.87% |
| = 5 (cap) | 1.51% | 8.24% |

## Rebalancing check — pixel share vs LOSS-MASS share by height bin (ALL px)

_Loss-mass share = Σw in bin / Σw total (how much each regime now counts). Ground share should DROP but stay substantial (not eliminated, §9); tall bins should RISE._

| GT height bin (m) | pixel share | loss-mass share | Δ (mass − pixels) |
|---|--:|--:|--:|
| 0–2 | 57.44% | 36.17% | -21.27% |
| 2–5 | 10.47% | 9.79% | -0.68% |
| 5–10 | 15.09% | 18.98% | +3.89% |
| 10–15 | 8.69% | 14.79% | +6.11% |
| 15–20 | 4.76% | 10.17% | +5.41% |
| 20–30 | 2.29% | 6.21% | +3.92% |
| 30–40 | 0.60% | 1.84% | +1.24% |
| 40–inf | 0.66% | 2.04% | +1.38% |

_Total subsampled ALL pixels: 2,056,830. Ground-dominant bins lose emphasis to buildings/tall structures without being zeroed out._
