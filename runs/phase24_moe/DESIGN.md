# Height-Regime Mixture-of-Experts Design (Phase 24C Aligned)

This document contains the final corrected design and computational graph for the **Height-Regime Mixture-of-Experts (MoE) Model**. It addresses all safety checks, explicitly defines differentiability boundaries, establishes the mathematical basis of tall-tail extrapolation, and defines the training/loss formulation.

---

## 1. Mathematical Extrapolation & Parameterization (No Fictional Physics)

The height prediction for the **High-Rise Expert ($E_3$)** is formulated as a **learned extrapolation parameterization**:
$$H_3 = (\alpha \cdot \text{depth\_range} + \beta \cdot \text{min}(w\_box, h\_box)) \times \exp(r_3)$$

### Rationale & Extrapolation Mechanism:
- **Base Feature Anchors:** Min-dimension of the bounding box ($\min(w\_box, h\_box)$) and relative depth range ($\max(d) - \min(d)$) act as horizontal and relative-vertical anchors. GSD converts pixels to horizontal meters, providing a relative spatial metric scale hint. Pretrained Depth Anything relative depth encodes relative scene disparity, which correlates with building height relative to the ground.
- **Unbounded Scaling Residual:** $r_3 = \text{MLP}_3(f_k)$ is a continuous residual output by a linear head. Since E_3 is trained exclusively on the tall-building tail using soft-membership weighting, the MLP parameters are optimized to scale up the base anchors when tall features are present.
- **Extrapolation:** The exponential function $\exp(r_3)$ acts as an extrapolation multiplier (e.g. $\exp(1.5) \approx 4.48$, scaling a $15\text{m}$ base footprint anchor to a physical height of $67.2\text{m}$). This enables the model to predict tall, continuous metric heights far beyond the training set range.
- **Residual Clamp:** $r_3 \in [-3.0, 3.0]$ enforces numerical stability. This translates to an exponential scaling multiplier range of $[0.05, 20.1]$, which is mathematically sufficient to represent heights up to $150\text{m}$ for realistic learned base values.

---

## 2. Gating Inference Safety

The gating network routes buildings strictly based on inference-available information.

### Computational Graph Input Verification:
- **Inputs to Gate:** Shared U-Net CNN pooled representation, scaled footprint geometry features, normalized relative depth stats, tile building density context.
- **Exclusions:** The gating network does **NOT** receive target-city calibration, true height, true height bins, or ground-truth statistics. 
- *Auxiliary regime classification is used strictly during training as a loss component and has no path to inference.*

---

## 3. Computational Graph & Differentiability Analysis

The end-to-end model is **semi-differentiable / multi-task**. Gradients from the height branch backpropagate to the shared U-Net backbone *only* through the pooled CNN feature path.

```
       [ RGB + Depth Map ]
                ↓
    ======================================
    1. Shared Encoder (SmallFusionUNet)        <-- Fully Differentiable (Trainable)
    ======================================
         ↓                        ↓
    [Feature Map]           [Footprint Logits]
         ↓                        ↓
         ↓                  Threshold & Connected Components (CPU) <-- Non-Differentiable
         ↓                        ↓
         ↓                  [Mask & Footprint Geometry]
         ↓                  [Depth Map Stats Extraction]           <-- Non-Differentiable
         ↓                        ↓
    ======================================
    2. Differentiable CNN Matmul Pooling       <-- Fully Differentiable
    ======================================
         ↓                        ↓
    [cnn_feat]              [feat_geom + feat_depth + feat_context] <-- Input Priors
         └──────────┬─────────────┘
                    ↓
        [35-D Building Vector (f_k)]
                    ↓
         ┌──────────┴──────────┐
         ↓                     ↓
    Gating MLP              Experts (E1, E2, E3 MLPs)             <-- Fully Differentiable (Trainable)
    [Softmax Weights w]     [Expert Heights H1, H2, H3]
         └──────────┬──────────┘
                    ↓
          [Weighted Prediction]
            H = w1*H1 + w2*H2 + w3*H3                             <-- Fully Differentiable
```

---

## 4. Gating & Weak Anti-Collapse Formulation

To prevent the gating network from collapsing to the dominant Low-Rise expert ($E_1$), we implement two lightweight, robust mechanisms:

1. **Auxiliary Gate Supervision:**
   During training, we apply a multi-class Cross Entropy loss on the gate routing weights $w$ against the aligned true regime labels (LOW: $<15\text{m}$, MID: $15-30\text{m}$, HIGH: $\ge 30\text{m}$):
   $$\mathcal{L}_{gate\_aux} = \text{CrossEntropy}(w, \text{true\_regime})$$
   *This acts as a teacher signal forcing the gate to route Mid and High structures to their respective experts.*

2. **Gating Balance Regularization:**
   We penalize the gating network if it concentrates all probability mass on a single expert over the batch:
   $$\mathcal{L}_{balance} = \sum_{i=1}^3 \bar{w}_i^2$$
   where $\bar{w}_i$ is the average gating probability of expert $i$ over the mini-batch. This does not force equal routing but prevents single-expert dominance.

---

## 5. Overlapping Expert Training (Strategy B)

We use **overlapping soft expert membership** during training to avoid sharp height discontinuities.

### Membership Weighting:
Each building $k$ of height $H_{true}$ contributes to the loss of expert $i$ scaled by $M_i(H_{true})$:
- $M_1(H) = \exp(-(H - 7.5)^2 / 100)$ (peaks at 7.5m, active 0-15m)
- $M_2(H) = \exp(-(H - 22.5)^2 / 150)$ (peaks at 22.5m, active 10-35m)
- $M_3(H) = \exp(-(H - 50.0)^2 / 400)$ (peaks at 50m, active 30m+)

### Decoupled Optimization:
Because E_3's loss is weighted by $M_3(H_{true})$, tall buildings (rare in the dataset) dominate E_3's parameters without affecting the low-rise expert E_1. Conversely, the low-rise majority only updates E_1, ensuring low-rise accuracy is preserved.
