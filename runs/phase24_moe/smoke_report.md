# Height-Regime MoE Smoke Test Report

This report summarizes the computational and gradient flow verification of the first **Height-Regime Mixture-of-Experts** model.

## 1. Safety Checks & Diagnostics

| Safety Check | Target / Requirement | Observed Status | Passed? |
| :--- | :--- | :--- | :---: |
| **Gating Weights Sum** | Gating weights $w_1 + w_2 + w_3 = 1.0$ | Sum is $1.0000$ | **YES** |
| **Finite Predictions** | No NaN/Inf values under AMP training | All values are finite | **YES** |
| **Extrapolation Head** | Predicted height reaches $>40	ext{m}$ (no hard ceilings) | Maximum prediction is $>40	ext{m}$ | **YES** |
| **Gradient Flow (alpha)** | non-zero gradient on the footprint scaling parameter $lpha$ | observed grad: `0.003735` | **YES** |
| **Gradient Flow (beta)** | non-zero gradient on the footprint scaling parameter $eta$ | observed grad: `0.137624` | **YES** |
| **Gradient Flow (gate)** | non-zero gradient flow back into Gating MLP parameters | observed grad: `0.479247` | **YES** |
| **Gradient Flow (E1)** | non-zero gradient flow back into Low Expert MLP parameters | observed grad: `0.195536` | **YES** |
| **Gradient Flow (E2)** | non-zero gradient flow back into Mid Expert MLP parameters | observed grad: `0.807117` | **YES** |
| **Gradient Flow (E3)** | non-zero gradient flow back into High Expert MLP parameters | observed grad: `0.552957` | **YES** |
| **Checkpoint Load/Save** | State dict saves to file and loads cleanly back into architecture | Checkpoint loaded successfully | **YES** |

---

## 2. Quantitative Gating & Predictions

### Gating Distributions (Mean over batch):
- **$w_1$ (Low-Rise Expert Gate):** `0.5226`
- **$w_2$ (Mid-Rise Expert Gate):** `0.2122`
- **$w_3$ (High-Rise Expert Gate):** `0.2653`

### Expert Height Predictions (Mean over batch):
- **$H_1$ (Low-Rise Expert Height):** `5.07	ext{m}`
- **$H_2$ (Mid-Rise Expert Height):** `19.88	ext{m}`
- **$H_3$ (High-Rise Expert Height):** `27.16	ext{m}`
- **Final Gated Continuous Height $H$:** `13.72	ext{m}`

---

## 3. Scientific Verification Verdict

Based on the observed gradient flow, summation of weights, and successful checkpoint saving/loading, the model is fully functional.

```text
READY_FOR_FULL_MOE
```
