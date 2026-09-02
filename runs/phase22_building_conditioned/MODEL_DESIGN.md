# MODEL DESIGN: BUILDING-CONDITIONED HEIGHT NETWORK

## 1. Exact Architecture Specification

Our building-conditioned network (`BuildingConditionedHeightNet`) consists of the following components:
- **Shared Backbone:** `SmallFusionUNet` which processes a 4-channel tensor $[RGB(3) + NormalizedDepth(1)]$ and outputs a 17-channel tensor $[FeatMap(16) + MaskLogits(1)]$.
- **Footprint Branch:** Extracts `MaskLogits(1)` from the backbone output, representing building footprint probabilities.
- **Object Segmentation:** Runs connected components dynamically on predicted masks (prob > 0.5) to segment buildings.
- **Differentiable Feature Pooling:** For each building component $M_k$:
  - pools 16-D CNN features from the feature map under $M_k$ using average pooling.
  - Concatenates 7 geometric features (area, aspect ratio, perimeter, compactness).
  - Concatenates 9 local relative depth features (`center_edge_diff`, standard deviation, range).
  - Concatenates 3 spatial context features (tile average area, tile density, tile building count).
  - Resulting pooled representation: 35-D vector $F_k$.
- **MLP Heads:**
  - Passes $F_k$ through a 2-layer MLP (hidden dims: 64, 32).
  - **Height-Regime Head:** Linear layer projecting to 5 logits (probabilities $P_c$ for bins $<10$m, $10-20$m, $20-30$m, $30-40$m, $\ge 40$m).
  - **Continuous Residual Head:** Linear layer projecting to 1 residual log-scale value $r_k \in \mathbb{R}$.

---

## 2. Trainable Parameter Count

- **Backbone parameters:** ~230,000 parameters.
- **MLP Heads:** ~4,500 parameters.
- **Total trainable parameters:** **`234,800`** parameters.

---

## 3. Height-Regime & Continuous Height Formulation

- **Height regimes:** Binned using threshold boundaries: $C \in \{0, 1, 2, 3, 4\}$.
- **Continuous metric-height extrapolation:**
  - regime base heights: $B = [5.0, 15.0, 25.0, 35.0, 45.0]$m.
  - Decoding height formula:
    $$\hat{H}_k = \left( \sum_{c=0}^4 P_c \cdot B_c ight) \cdot \exp(r_k)$$
  - *Extrapolation mechanism:* If a building is classified in the highest regime ($\ge 40$m) with base $45$m, predicting a positive log-residual $r_k = 0.8$ scales the building output height to $45 \cdot \exp(0.8) pprox 100.1$m, allowing mathematically unbounded extrapolation for skyscrapers.

---

## 4. Moderate Height-Balancing Scheme

We apply moderate **square-root sample weighting** based on the natural training building height distribution:
$$w_k = W_{	ext{bin}(k)} = rac{1}{\sqrt{N_{	ext{bin}(k)}}}$$
Normalized bin weights:
- **Bin 0 (<10m):** `0.38` (down-weighted from abundance).
- **Bin 1 (10-20m):** `0.63`.
- **Bin 2 (20-30m):** `1.08`.
- **Bin 3 (30-40m):** `1.24`.
- **Bin 4 (>=40m):** `1.70` (up-weighted to give tall skyscrapers meaningful gradient contribution).

This moderate weighting scheme avoids the aggressive degradation of low-rise structures observed in Phase 21 while protecting tail-supervision.

---

## 5. Dense nDSM Reconstruction & Roof Topology

- Per-building predicted height scale is reconstructed into a dense nDSM map by scaling local relative-depth maps pixel-wise to preserve roof topography:
  $$h_{pixel} = (\hat{H}_k - 2.0) 	imes N_{norm} + 2.0$$
  where $N_{norm}$ is the locally min-max normalized relative depth of the building mask.
