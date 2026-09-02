# PHASE 20 — TALL-HEIGHT EXTRAPOLATION DESIGN

## 1. Training vs. New York Building Height Distribution

To quantify the geographic transfer and scale extrapolation difficulty, we computed the exact building-level ($P_{95}$ height) distributions:

| Statistic | DFC2023 Training Set | New York Test Set |
| :--- | :---: | :---: |
| **Total Buildings (N)** | 3026 | 2388 |
| **P50 (Median Height)** | 7.8m | 31.5m |
| **P75** | 12.6m | 49.2m |
| **P90** | 23.8m | 63.3m |
| **P95** | 32.5m | 74.2m |
| **P99** | 58.2m | 92.0m |
| **Maximum Height** | 130.9m | 101.5m |
| **Count $\ge 30$m** | 210 | 1248 |
| **Count $\ge 40$m** | 75 | 935 |
| **Count $\ge 60$m** | 28 | 294 |
| **Count $\ge 100$m** | 3 | 8 |

### Extrapolation Difficulty Analysis
The training set is heavily dominated by low-rise and mid-rise structures (P50: **7.8m**, and only **75** buildings exceed 40m). In contrast, New York is a highly dense high-rise city with **935** buildings exceeding 40m (including **8** skyscrapers taller than 100m).
This represents a severe **out-of-distribution extrapolation task** (the target city has buildings more than 5 times taller than the bulk of training examples). Tree-based regressors (RF/GBR) cannot predict above the training max, causing massive bias (underpredicting skyscrapers by $45$m).

---

## 2. Proposed Tall-Height Extrapolation Candidates

We design three candidates that mathematically enable the model to extrapolate beyond the maximum training height:

### Candidate A: Ordinal Regime + Local Scale Residual (OR-LSR)
- **Input:** 21-D building features.
- **Targets:**
  1.  Coarse ordinal height regime classification: $C \in \{0, 1, 2, 3, 4\}$ corresponding to bins $[0, 10), [10, 20), [20, 30), [30, 40), [40, +)$.
  2.  Continuous log-residual scaling factor: $R \in \mathbb{R}$.
- **Decoding Formula:**
  $$H_i = 	ext{base}(C) 	imes (1.0 + 	ext{Softplus}(R_i))$$
  where base values are $[5.0, 15.0, 25.0, 35.0, 45.0]$m.
- **How it extrapolates:** If a skyscraper is classified into the highest regime ($C = 4$), the base anchor is $45$m. The continuous residual $1.0 + 	ext{Softplus}(R_i)$ acts as a multiplier. Since Softplus is unbounded, strong local features (like high `center_edge_diff` or massive footprint area) can scale the prediction up to 80m or 100m, even if no such labels exist in the training set.

### Candidate B: Log-Domain Regression (Log-Reg)
- **Input:** 21-D building features.
- **Target:** $y_{log} = \log_{1p}(H_i)$.
- **Decoding Formula:**
  $$\hat{H}_i = \exp(\hat{y}_{log}) - 1.0$$
- **How it extrapolates:** Logarithmic scaling compresses the target variance during training, preventing gradient collapse. When decoded back via the exponential function, small residual outputs at the high end map to large absolute height variations.

### Candidate C: GSD-Anchored Ratio Predictor (GSD-ARP) [RECOMMENDED]
- **Input:** 21-D building features.
- **Target:** A dimension-free ratio scaling target:
  $$\gamma_i = rac{H_i}{\sqrt{	ext{Area}_{m2}}}$$
- **Decoding Formula:**
  $$\hat{H}_i = \hat{\gamma}_i 	imes \sqrt{	ext{Area}_{m2}}$$
- **How it extrapolates:** Footprint Area ($m^2$) is computed directly from predicted masks at a known constant horizontal GSD (0.5m). Tall buildings physically have larger footprint sizes (often exceeding $5,000m^2$, yielding $\sqrt{	ext{Area}} pprox 70$). Because the target ratio $\gamma$ remains in a narrow, stable regime across all cities (e.g. $[0.3, 0.8]$), predicting a stable ratio of $0.8$ for a skyscraper automatically yields a height of $0.8 	imes 70 = 56$m. Extrapolation is driven physically by the horizontal GSD, bypassing regression clipping!

---

## 3. Information Preservation & Roof Topology

- **Normalized relative topology:** Preserved local relative-depth shape inside each mask component.
- **Center-edge slope (`center_edge_diff`):** Retained as a scale-prediction feature.
- **Raw relative depth range:** Retained as a scale-prediction feature.
- **Reconstruction:** For flat roofs, pixels are set uniformly to $S_i$. For sloped/gabled roofs, pixels scale as:
  $$h_{pixel} = (S_i - 2.0) 	imes N_{norm} + 2.0$$

---

## 4. Minimal Falsification Experiment

- **Setup:** Train the Ridge/MLP regressors on training buildings *strictly below 25m* (excluding all tall training buildings to simulate extreme out-of-distribution testing).
- **Evaluation:** Predict the height of New York buildings $>30$m zero-shot.
- **Falsification Threshold:** If the predicted mean height for buildings $>30$m is **$< 20	ext{m}$** (indicating extrapolation collapse/clipping), the candidate formulation is falsified and must be abandoned.

---
### Final Decision:
```text
PROCEED TO HYBRID BUILDING MODEL
```
We select **Candidate C (GSD-Anchored Ratio Predictor - GSD-ARP)**. It anchors vertical metric height directly to the physical horizontal GSD of predicted footprints, mathematically forcing scale extrapolation.
