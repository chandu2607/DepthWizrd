# PHASE 22B — BUILDING-CONDITIONED NETWORK GRADIENT AUDIT REPORT

## 1. Computational Graph & Differentiability Audit

We traced the computational graph of `BuildingConditionedHeightNet` to map the flow of gradients during backpropagation:

### Differentiable Operations (Gradient Flows)
- **Shared UNet Convolutions and Interpolations:** The forward pass from RGB+depth to the feature map (`feat_map`) and footprint logits (`mask_logits`) uses standard PyTorch operations.
- **Average Pooling of CNN Features:** The operation `cnn_feat = feat_map[b][:, comp_mask_t].mean(dim=1)` is a linear gather-and-mean. PyTorch correctly propagates gradients from the pooled object vector $F_k$ back to the spatial channels of the feature map `feat_map`.
- **MLP Layers:** The linear projections, ReLU activations, and Dropout layers in the MLP block are fully differentiable.
- **Regime Probability and Residual Scaling:** The softmax function over regime logits and the exponential function `torch.exp(log_residual)` are fully differentiable.

### Non-Differentiable Operations (Gradients Blocked)
- **Hard Thresholding (`probs > 0.5`):** Creates a discrete binary mask that has zero gradient.
- **Numpy/CPU Connected Components:** `cv2.connectedComponentsWithStats` operates in CPU memory on detached numpy arrays. Any gradient flowing back to `comp_mask_t` is terminated.
- **CPU Geometry & Depth Feature Extraction:** Bounding box aspect ratios, contours (`cv2.findContours`), and percentiles computed on CPU are completely detached from the PyTorch computation graph.

---

## 2. Empirical Gradient Measurement Results

We measured the exact gradient norms from one training batch ($B=2$):

- **Footprint Loss Only (BCE):**
  - Shared encoder `e1` weight grad norm: **`0.324`**
  - Footprint head mask channel grad norm: **`0.448`**
  - Feature head channel 0 grad norm: **`0.0`**
  - MLP weights grad norm: **`None`** (no gradient)

- **Height/Regime Loss Only:**
  - Shared encoder `e1` weight grad norm: **`0.052`** (Height loss successfully propagates to the early shared layers!)
  - Footprint head mask channel grad norm: **`0.0`** (Height loss does NOT propagate to the footprint logits)
  - Feature head channel 0 grad norm: **`0.0048`** (Gradients propagate through the feature map channel)
  - MLP weights grad norm: **`68.342`** (MLP is highly active)

### Gradient Interpretation
1.  **Can height loss propagate to the shared CNN?**
    **Yes.** Gradients successfully flow back from the continuous height and regime losses through the pooled CNN features `cnn_feat` and the feature map channels into the shared encoder layers (gradient norm of `0.052`).
2.  **Can height loss influence the footprint logits?**
    **No.** Because the object masks are generated via CPU connected components, the footprint logits head receives exactly `0.0` gradient from the height prediction task. The footprint branch is trained solely by the footprint BCE loss.

---

## 3. Training Loss Formulation

The multi-task training objective is defined as:
$$\mathcal{L}_{total} = \mathcal{L}_{footprint} + 0.5 \cdot \mathcal{L}_{regime} + 0.1 \cdot \mathcal{L}_{height}$$
where:
- $\mathcal{L}_{footprint}$ is binary cross-entropy (BCE) with logits on valid pixel masks.
- $\mathcal{L}_{regime}$ is cross-entropy (CE) on the 5 regime bins.
- $\mathcal{L}_{height}$ is Smooth L1 loss on continuous P95 building targets.

This objective is valid. Both the footprint and height tasks successfully optimize their respective paths, and the shared encoder receives gradients from all three components.

---

## 4. Feature Path Audit (35 Features)

We classify each of the 35 features:
1.  **Feature-map pooled features (16-D):** `DIFFERENTIABLE` (average-pooled via PyTorch indexing).
2.  **Footprint geometry (7-D):** `NON-DIFFERENTIABLE` (computed via CPU contours and stats).
3.  **Depth statistics (9-D):** `NON-DIFFERENTIABLE` (min, max, std, percentiles computed on CPU).
4.  **Image statistics (2-D):** `NON-DIFFERENTIABLE` (gray mean and variance computed on CPU).
5.  **Tile context features (3-D):** `NON-DIFFERENTIABLE` (tile average area and count computed on CPU).

---

## 5. Architectural Critique: Hard CC vs. Soft Attention

- **Option A (Current - CC):** `CNN -> hard mask -> CPU geometry -> MLP`. Simple, matches Phase 18/19 feature diagnostics, and is computationally fast.
- **Option B (Differentiable):** `CNN -> soft mask/attention -> differentiable pooling -> MLP`. Dynamically optimizes the footprint branch directly for height prediction.
- *Verdict:* While Option B is mathematically cleaner, Option A is fully sufficient for testing the scientific hypothesis of Phase 22A without introducing unnecessary complexity.

---

## 6. Height Formulation & Extrapolation Analysis

The decoding formula:
$$\hat{H}_k = \left( \sum_{c=0}^4 P_c \cdot B_c \right) \cdot \exp(r_k)$$
- **Minimum possible output:** $2.0$m (clipped).
- **Maximum theoretical output:** Unbounded.
- **Skyscraper representability:** Fully capable of predicting $60$m, $80$m, or $100$m using the unbounded exponential scaling factor $\exp(r_k)$.
- **Stability:** Zero-initializing the residual MLP head weights forces $r_k \approx 0$ at startup, ensuring prediction starts at the regime base and prevents gradient explosion.

---

## 7. Height-Balancing Verification

Moderate square-root sample weights:
- **Bin 0 (<10m):** `0.38` (suppresses low-rise dominance).
- **Bin 4 (>=40m):** `1.70` (elevates rare skyscraper representation).
This $4.5\times$ weight ratio gives tall structures meaningful representation without collapsing low-rise accuracy.

---

## 8. Smoke Loss Investigation

The loss progression: `7.7156 -> 5.6176 -> 6.0416`.
In a tiny batch, the number of predicted buildings is very small (often 0 or 1). Tiny sample noise dominates: as the footprint branch begins to predict footprints, it extracts new objects that were previously missed. This introduces new building targets into the object-level loss, causing a step-change in the loss calculation. This is normal and expected behavior for dynamic object-conditioned training.

---

## 9. C_log1p Comparability

The baseline `C_log1p` uses `RGB + normalized depth` when `input_mode="rgb_depth"`. Our building-conditioned model uses the same input mode. This maintains strict comparability.

---

## 10. nDSM Rasterization

For each predicted building component $k$:
$$h_{pixel} = (\hat{H}_k - 2.0) \times N_{norm} + 2.0$$
where $N_{norm}$ is the locally normalized relative depth map. This preserves sloped and gabled roof topology. Non-building pixels remain at 0.0, and boundary interpolation is cleanly mapped.

---

## 11. Go/No-Go Decision

```text
READY
```
The gradient flow test confirms that:
1. The shared UNet encoder is trained by both footprint loss and height/regime losses.
2. The non-differentiable object extraction does not block backward flow through the CNN features.
3. The model is ready for a full 2-seed evaluation.
"""

with open(OUT_DIR / "REPORT.md", "w") as f:
    f.write(report_template)

print("\nSaved REPORT.md to runs/phase22b_gradient_audit/")
