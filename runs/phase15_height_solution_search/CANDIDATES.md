# PHASE 15 — HEIGHT SOLUTION SEARCH (CANDIDATES & RECOMMENDATION)

## 1. Current Failure Diagnosis

Our experiments (Phases 1-14) have proven that while the `C_log1p` + Depth Anything V2 (DA-V2) pipeline successfully extracts relative structural topography (edges, building vs. ground), it suffers from a terminal **metric-scale collapse on the tall tail**. 

**Why did it fail?**
1. **Lack of Physical Scale in DA-V2:** DA-V2 is a foundation model trained heavily on indoor and autonomous driving data, where "depth" is perspective-based. In overhead orthorectified imagery, height must be inferred from orthogonal cues: footprint size, contextual geometry, and shadow length. DA-V2's pre-trained features do not inherently map these specific overhead cues to metric absolute scale.
2. **The Long-Tail Problem:** Building heights follow a severe long-tailed distribution (95%+ of buildings are <15m). Standard regression losses (L1/MSE) overwhelmingly penalize overestimation of the "head" (short buildings), forcing the network to conservatively predict ~15m for anything that looks "tall" to minimize expected penalty.

## 2. Critical Question: Where Does Physical Scale Come From?

In single-view overhead imagery (without multi-view stereo or LiDAR), physical metric height can only be derived from:
1. **Analytical Geometry (Shadows/Facades):**
   - *Shadows:* Height = shadow_length × tan(sun_elevation). 
   - *Facades:* Height = facade_length × GSD / sin(off_nadir).
   - *Limitation:* We do NOT have reliable solar elevation, azimuth, or satellite off-nadir metadata in the DFC2023 dataset. Thus, pure analytical recovery is impossible without reverse-engineering the metadata per tile.
2. **Learned Priors (Footprint & Context anchored by GSD):**
   - Because all DFC2023 tiles share a consistent ~0.5m GSD, the pixel-area of a building footprint has a direct physical size in square meters.
   - The network must learn the statistical correlation between footprint area/shape, shadow extent (relative to the building), and metric height. 
   - *Conclusion:* The model must rely on **learned priors anchored by the dataset's GSD**. The architecture and loss function must explicitly support this mapping without being crushed by the long-tailed distribution.

---

## 3. Candidate Methods

### Candidate A: HTC-DC Net (Classification-Regression for Long-Tailed Height)
- **Concept:** Uses a Head-Tail Cut (HTC) mechanism and Distribution-based Constraints (DCs) to explicitly combat the long-tailed nature of building heights.
- **Metric Scale:** Learned prior, but protected from the "head" dominance by dynamically reweighting and adaptively binning the height distribution.
- **Relevance:** Directly targets our exact failure (tall-building collapse).
- **Hardware Feasibility:** High (standard ViT/CNN backbones).

### Candidate B: HGDNet (Height-Hierarchy Guided Dual-Decoder Network)
- **Concept:** A multi-task network (winner/top-performer at DFC2023) that jointly predicts building footprints (segmentation) and building height. 
- **Metric Scale:** By forcing the network to explicitly extract the footprint, it forces the features to encode the building's geometric area (GSD anchor), which strongly correlates with height. It also uses a discrete height-hierarchy classification branch to guide the continuous regression.
- **Relevance:** Proven on the exact DFC2023 dataset.
- **Hardware Feasibility:** High.

### Candidate C: Depth2Elevation (Scale Modulated Depth Anything)
- **Concept:** Injects a "Scale Modulator" into the Depth Anything Model to adapt its perspective-based depth to overhead elevation.
- **Metric Scale:** Learned prior adapting the DA-V2 latent space.
- **Relevance:** Attempts to fix the exact foundation model we are using.
- **Hardware Feasibility:** Moderate/High (requires joint training of DA-V2).
- **Risk:** We already attempted deep ViT adaptation and found the features highly resistant to scale recovery.

### Candidate D: Analytical Shadow Estimation
- **Concept:** Detect shadows, detect footprints, calculate sun angle, and triangulate height.
- **Metric Scale:** True physical geometry.
- **Relevance:** Conceptually perfect.
- **Risk:** DFC2023 lacks sun metadata. Shadows suffer from "mutual adhesion" in dense cities like New York (our test set). Would require an auxiliary network just to regress the unknown sun angle.

---

## 4. Candidate Scoring

| Candidate | Remote Sensing | Metric Height | Tall-Tail Strategy | Needs Target Labels? | Expected Compute | Main Risk |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **HTC-DC Net** | Yes | Yes (Regr.) | Yes (HTC Loss) | No | Moderate | May still require huge datasets to learn perfect bins |
| **HGDNet** | Yes | Yes (Regr.) | Yes (Multi-task) | No | Moderate | Complexity of balancing dual decoders |
| **Depth2Elevation** | Yes | Yes | Yes (Scale Mod.) | No | High | DA-V2 features may remain stubborn |
| **Shadow Geometry**| Yes | Yes (Physical)| Yes (Geometric) | No | Low | Missing metadata; dense occlusion |

---

## 5. Primary Recommendation

**PRIMARY: HGDNet Paradigm (Multi-task Footprint + Height-Hierarchy Guidance)**

### WHY
The core reason our model fails is that it treats height estimation as a generic pixel-to-pixel translation (like depth). In remote sensing, a building's height is a property of the *entire building instance*. By implementing a dual-decoder that simultaneously predicts the **building footprint mask** and the **building height**, we force the shared encoder to learn the physical extent (GSD scale) and boundaries of the structure. HGDNet's specific use of an auxiliary height-bin classification branch further guides the regression, preventing the tall-tail collapse. It is also the proven state-of-the-art on the exact DFC2023 dataset we are using.

### PHYSICAL SCALE SOURCE
The physical scale is a **learned prior anchored by the constant GSD**. By explicitly segmenting the footprint, the network learns the physical area of the building, which statistically anchors its height.

### EXPECTED ADVANTAGE
- **Over C_log1p:** The multi-task footprint extraction forces the network to treat the building as a discrete physical object rather than a blurry blob of relative depth, pulling the correct scale from the footprint size and explicitly guided by the classification hierarchy.

### TEST
Implement a lightweight dual-decoder on top of our existing frozen/partially frozen DA-V2 (or a fresh ResNet/Swin backbone if DA-V2 is too stubborn):
1. **Decoder A:** Binary building footprint segmentation (BCE Loss).
2. **Decoder B:** Continuous height regression (L1/Huber Loss).
Train on the multi-city split and evaluate if the multi-task footprint guidance lifts the New York >30m prediction ceiling.

### FALLBACK
**HTC-DC Net Paradigm.** If multi-task footprint guidance fails, we abandon structural guidance and implement the Head-Tail Cut (HTC) loss to mathematically force the network to prioritize the rare tall buildings.

### STOP CONDITION
If the dual-decoder footprint-guided model still collapses on the tall tail (< 20m max predictions), it proves that monocular RGB simply lacks sufficient physical cues for extreme heights in this dataset, and we must stop purely monocular metric regression and declare it physically unsolvable without shadows/metadata.

---
*MANDATORY STOP EXECUTED. Awaiting human review before implementation.*
