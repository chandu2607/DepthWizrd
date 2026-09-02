# Phase 24A — Height-Regime Mixture-of-Experts Design

This document details the architectural design for a **Height-Regime Mixture-of-Experts (MoE)** network. The goal is to resolve the severe regime conflict identified in Phase 21 & Phase 23, where a single estimator fails to generalize to tall skyscrapers without degrading accuracy on the low-rise majority.

---

## 1. Motivation & Scientific Rationale

In Phase 23, the building-conditioned model collapsed on New York skyscrapers, predicting a mean height of **5.76m** (True Mean: 54.3m). This occurred because:
1. **Regime Conflict:** The training set is dominated by low-rises (~83% < 10m). The network's parameters are heavily biased toward predicting the majority class prior to minimize overall L1 loss.
2. **Weak Feature-Height Correlation:** footprint area and relative-depth variations do not share a single global linear or simple non-linear mapping to metric height across all building types (e.g. warehouses vs. towers).

An MoE architecture addresses this by **specializing** network paths. A gating network routes buildings to different experts depending on their localized features, allowing:
- **Low-Rise Expert:** Focused entirely on high-density, low-altitude structures to preserve sub-meter residential accuracy.
- **Mid-Rise Expert:** Focused on urban block structures (10m–30m).
- **High-Rise / Extreme Expert:** Focused exclusively on tall buildings and skyscrapers (30m+), learning tall-scale extrapolation without parameter distortion from low-rises.

---

## 2. Candidate Expert Structures

We evaluate three candidate expert partitions:

| Structure | Configuration | Continuity & Overlap | Training Sample Density | Expert Collapse Risk |
| :--- | :--- | :--- | :--- | :--- |
| **A. Hard Partition** | 3 Experts: Low (<12m) / Mid (12-30m) / High (>30m) | **Poor.** Discontinuous predictions at boundaries cause spatial roof artifacts. | Low expert has 83%, High has 2.4%. High expert lacks data. | **High.** Gate defaults to Low expert due to extreme training bias. |
| **B. Fine Partition** | 4 Experts: Low / Mid / High / Extreme | **Poor.** Sharp boundaries. Low sample count in extreme class makes training unstable. | Extreme class has <0.5% samples. | **High.** Extreme expert parameters never converge. |
| **C. Overlapping Experts (Selected)** | **3 Experts:** Low (0-15m) / Mid (10-35m) / High (30m+) | **Excellent.** Smooth soft-gated transition interpolates boundaries nicely. | Soft membership functions allow high overlaps, increasing sample availability. | **Low.** Gating functions are regularized to distribute weights. |

---

## 3. Gating Network Design

The gating network routes each building component $k$ to the experts.

### Gate Inputs (35-D normalized vector):
1. **Geometry Features (Scaled):** $[Area / 500.0, Area_{m2} / 125.0, W_{box}/20.0, H_{box}/20.0, AspectRatio, Perimeter/50.0, Compactness]$
2. **Normalized Depth Stats:** $[d_{mean}, d_{med}, d_{std}, d_{p90}, d_{p95}, d_{p99}, d_{range}, d_{grad}, center\_edge\_diff]$ (standardized by training-set statistics).
3. **Contextual Piles:** $[Density, AvgBuildingArea / 125.0, N_{buildings} / 10.0]$
4. **CNN Pooled Representation:** Shared backbone representation ($C_{feat} = 16$).

### Gating Output:
A soft weight vector $w = [w_1, w_2, w_3]$ computed via a softmax layer:
$$w_i = \frac{\exp(g_i(f_k))}{\sum_j \exp(g_j(f_k))}, \quad \sum_{i=1}^3 w_i = 1$$
where $g_i$ is a small 2-layer MLP gating head.

---

## 4. Continuous Height Formulation & Tall Extrapolation

To predict heights far outside the training range (extrapolating to 60m–100m for skyscrapers), the **High-Rise Expert ($E_3$)** cannot use simple linear or bounded sigmoid heads. It must use a physically bounded but mathematically open representation.

We select a **Multiplicative Scale Factor on Footprint Dimensions and Depth Range**:
$$H_3 = (\alpha \cdot \text{depth\_range} + \beta \cdot \text{min}(w\_box, h\_box)) \times \exp(r_3)$$
- **Rationale:** A skyscraper's height is physically constrained by its horizontal base (width/height) for structural stability. Multiplying this base by a scale factor derived from normalized relative depth variation and an exponent-scaled residual ($r_3$) allows the model to predict tall, stable metric heights.
- The Low-Rise and Mid-Rise experts will use the standard regime-relative continuous formula:
  $$H_1 = \text{base}_1 \times \exp(r_1), \quad H_2 = \text{base}_2 \times \exp(r_2)$$

---

## 5. Training Strategy & Anti-Collapse

To prevent the gating network from collapsing to the dominant Low-Rise expert, we implement three critical training constraints:

1. **Auxiliary Gate Loss:**
   We train the gating network directly using a multi-class Cross Entropy loss against the true height regimes:
   $$\mathcal{L}_{gate\_aux} = \text{CrossEntropy}(w, \text{true\_regime})$$
   *This acts as a teacher signal forcing the gate to active Mid and High experts on appropriate buildings.*

2. **Soft Membership Weighted Losses:**
   Instead of hard subset training, each expert is trained on all samples but weighted by a soft target membership function $M_i(H_{true})$:
   - $M_1(H) = \exp(-(H - 5)^2 / 50)$ (Low expert: peaks at 5m, active 0-15m)
   - $M_2(H) = \exp(-(H - 20)^2 / 100)$ (Mid expert: peaks at 20m, active 10-35m)
   - $M_3(H) = \exp(-(H - 50)^2 / 400)$ (High expert: peaks at 50m, active 30m+)
   
   The loss for expert $i$ is:
   $$\mathcal{L}_{expert\_i} = \sum_k M_i(H_{true}^{(k)}) \cdot \text{SmoothL1}(H_i^{(k)}, H_{true}^{(k)})$$

3. **Regime-Aware Sampling:**
   We enforce that every training mini-batch contains at least 15% buildings from the mid/high regimes (by oversampling tiles containing tall structures or applying sample-weighted batching).

---

## 6. Novelty & SIH Value

### Novelty Assessment:
Traditional MoE models are applied to dense token routing (Transformers) or image classifications. Applying MoE at the **object/component-level** dynamically to resolve extreme out-of-distribution height domain shifts in remote sensing is highly custom.

### SIH Value:
- **One Unified Model:** The entire system is deployable as a single network. No separate pipelines or city-specific models are needed.
- **Physical Grounding:** Gating and routing happen automatically. Low-rises bypass skyscraper scaling math, preserving sub-meter residential precision, while skyscrapers route to the multiplicative extrapolation head.

---

## 7. Computational Feasibility

We target deployment on a standard **RTX 3050 4GB VRAM**:
- **Backbone:** SmallFusionUNet (~1.5M parameters)
- **Gating Network:** 2-layer MLP (in: 35, hidden: 64, out: 3) ~2.5K parameters
- **Experts:** 3 MLP experts, each with 2 layers (hidden: 64) ~15K parameters total
- **VRAM Estimate:** Peak training VRAM under batch size 8 is **~850 MB** (fully compatible with the 4GB limit).

---

## 8. Proposed Verification Experiment

We define exactly one experiment to compare the baseline against the MoE model:

- **Baseline:** Phase 23 Building-Conditioned Baseline (No MoE)
- **Proposed:** Phase 24 Height-Regime MoE Model
- **Splits:** Train on DFC2023 Arm-B, Val on Copenhagen, Test on New York (Zero-Shot).
- **Seeds:** 2 seeds (0 and 1).
- **Success Criteria:**
  1. Skyscraper (>40m) MAE reduced from 48.5m to **< 25.0m** on New York.
  2. Overall Building MAE on New York reduced to **< 15.0m**.
  3. Low-Rise (<10m) MAE on Copenhagen/New York remains **< 4.5m**.
  4. Stability: Predictions do not show extreme (>150m) spikes.
