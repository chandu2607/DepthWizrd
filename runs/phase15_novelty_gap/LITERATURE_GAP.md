# DEPTHWIZARD — RESEARCH GAP + NOVELTY ANALYSIS
## DO NOT IMPLEMENT OR COPY AN EXISTING MODEL

---

## 1. Focused Study of Existing Methods

We review six established methods in monocular remote sensing height estimation to map out the current state-of-the-art.

### Method 1: HGDNet (Height-hierarchy Guided Dual-decoder Network)
*   **Input Modalities:** Single-view RGB optical remote sensing image (optionally SAR in multimodal tracks, but Track 2 uses optical).
*   **Target:** Normalized Digital Surface Model (nDSM) height map + building footprint mask.
*   **Backbone:** ConvNeXt V2 (or Swin/ResNet).
*   **Architecture:** Multi-task dual-decoder network. One decoder branch estimates continuous height; the other extracts building footprints. An auxiliary classifier predicts discrete height-hierarchies.
*   **Height Formulation:** Continuous height regression (Smooth L1 loss) guided by discrete height-hierarchy classification bins (Cross-Entropy loss).
*   **Segmentation/Building Branch:** Yes. Dedicated footprint segmentation decoder branch.
*   **Scale Mechanism:** Learned GSD-anchored spatial priors. By predicting building footprints and enforcing consistency with semantic height hierarchies, the model associates a building's shape/pixel area to its height interval.
*   **Treatment of Long-Tailed Heights:** The auxiliary height-hierarchy branch classifies building pixels into multiple discrete vertical intervals. This guides the continuous regression branch, helping prevent the underestimation of rare, tall buildings.
*   **Direct Metric Supervision:** Yes, supervised via LiDAR-derived ground truth nDSM (meters).
*   **Target-City Labels Required:** No, can be evaluated zero-shot or on multi-city splits (such as DFC2023 validation).
*   **Dataset(s):** DFC2023 Track 2, ISPRS Potsdam, ISPRS Vaihingen.
*   **Cross-City Evaluation:** Evaluated on the DFC2023 test dataset covering 17 cities globally across different continents.
*   **Strengths:** Joint learning of footprints and heights resolves spatial misalignments; discrete hierarchy stabilizes tall building predictions.
*   **Weaknesses:** High architectural complexity; completely dependent on joint semantic footprint annotations during training.
*   **Known Limitations:** If building footprint annotations are noisy, missing, or misaligned with the height maps, height estimation accuracy degrades severely.
*   **Published Results:** Improved Track 2 baseline by over 6% in F1/IoU and reduced height RMSE (~3.5m - 4.5m depending on the split).
*   **Code Availability:** Yes, available under researchers' GitHub repos (usually academic implementations).
*   **License:** Academic / non-commercial.
*   **Where "Metres" Actually Comes From:** Ground-truth LiDAR nDSM labels paired with a constant Ground Sampling Distance (GSD) of 0.5m. The network learns the statistical mapping between building pixel size (area) and physical metric height.

---

### Method 2: HTC-DC Net (Head-Tail Cut Distribution-based Constraints Network)
*   **Input Modalities:** Single-view RGB optical image.
*   **Target:** Height map (nDSM).
*   **Backbone:** EfficientNet-B4/B5.
*   **Architecture:** Encoder-decoder with a Vision Transformer (ViT) bottleneck called HTC-AdaBins.
*   **Height Formulation:** Hybrid classification-regression with adaptive bins. The height range is partitioned into adaptive bins, and the final height is a linear combination of bin centers weighted by classification probabilities.
*   **Segmentation/Building Branch:** No.
*   **Scale Mechanism:** Learned dataset priors. The network uses the HTC-AdaBins transformer module to dynamically compute bin widths and centers based on the global context of the input image.
*   **Treatment of Long-Tailed Heights:** Head-Tail Cut (HTC) loss. Groups height distribution into foreground/tall buildings ("tail") and background/low heights ("head"), dynamically reweighting gradients to prevent the model from underestimating heights of rare, tall structures.
*   **Direct Metric Supervision:** Yes, supervised via LiDAR-derived ground truth nDSM.
*   **Target-City Labels Required:** No.
*   **Dataset(s):** ISPRS Vaihingen, Potsdam, DFC19, Global Building Height (GBH).
*   **Cross-City Evaluation:** Evaluated across datasets (cross-city/cross-sensor transfer) to demonstrate generalization.
*   **Strengths:** Targets the long-tailed height distribution specifically; avoids hard-bound pixel-wise predictions through adaptive bin regression.
*   **Weaknesses:** Highly dependent on the training distribution's max-height settings. The adaptive bins still learn to regress scale directly, which fails when transferring to unseen cities with a completely different scale context (e.g., European suburbs to New York skyscrapers).
*   **Known Limitations:** High memory footprint during training due to the transformer-based binning modules.
*   **Published Results:** RMSE of ~1.6m on Vaihingen and ~3.0m on DFC2019.
*   **Code Availability:** Yes (GitHub: `zhu-xlab/HTC-DC-Net` or similar).
*   **License:** Unspecified academic.
*   **Where "Metres" Actually Comes From:** Ground-truth LiDAR nDSM labels. The scale is inferred via learned priors modulated by adaptive bins (global ViT context) and a Head-Tail Cut loss.

---

### Method 3: DFC2023 Official Baseline
*   **Input Modalities:** Single-view RGB optical image.
*   **Target:** Building footprint mask (semantic segmentation) + height map (nDSM).
*   **Backbone:** ResNet34.
*   **Architecture:** Double-head U-Net or MMDetection-based multi-task FCN.
*   **Height Formulation:** Direct continuous regression (L1 loss).
*   **Segmentation/Building Branch:** Yes. Binary footprint mask decoder branch.
*   **Scale Mechanism:** Direct pixel-wise mapping learned end-to-end.
*   **Treatment of Long-Tailed Heights:** None (standard regression loss).
*   **Direct Metric Supervision:** Yes, supervised via LiDAR-derived ground truth nDSM.
*   **Target-City Labels Required:** No.
*   **Dataset(s):** DFC2023.
*   **Cross-City Evaluation:** Evaluated on the multi-city validation set.
*   **Strengths:** Simple to implement; provides a solid multi-task baseline.
*   **Weaknesses:** Extremely prone to underestimating tall buildings (collapsing towards the mean height of ~10-15m); suffers from boundary blurring.
*   **Known Limitations:** Does not generalize well to unseen cities with tall buildings due to direct L1 regression dominance by flat terrain.
*   **Published Results:** Baseline RMSE ~5.5m - 6.5m on validation set.
*   **Code Availability:** Yes (GitHub: `AICyberTeam/DFC2023-baseline`).
*   **License:** MIT / Apache 2.0.
*   **Where "Metres" Actually Comes From:** Ground-truth LiDAR nDSM labels. Direct supervision forces the network to map RGB pixels to metric heights.

---

### Method 4: IM2HEIGHT (Fully Residual Conv-Deconv Network)
*   **Input Modalities:** Single-view RGB optical image.
*   **Target:** DSM / height map.
*   **Backbone:** ResNet-based Fully Convolutional Network (FCN) or VGG.
*   **Architecture:** Fully residual convolutional-deconvolutional network with skip connections.
*   **Height Formulation:** Direct pixel-wise regression (MSE/L2 loss).
*   **Segmentation/Building Branch:** No.
*   **Scale Mechanism:** Direct statistical regression of pixel intensity to height values (learned dataset prior).
*   **Treatment of Long-Tailed Heights:** None.
*   **Direct Metric Supervision:** Yes, supervised via LiDAR-derived ground truth nDSM/DSM.
*   **Target-City Labels Required:** No.
*   **Dataset(s):** ISPRS Vaihingen, Potsdam.
*   **Cross-City Evaluation:** Limited. Evaluated primarily on train/test splits of the same dataset.
*   **Strengths:** Historical milestone; proved that deep neural networks can estimate height from single monocular images.
*   **Weaknesses:** Predictions are blurry; struggles with fine boundaries; extremely vulnerable to the mean-collapse problem (underestimating tall structures); lacks any explicit scale or object-level representation.
*   **Known Limitations:** Requires direct pixel-to-pixel correspondence; does not scale well to diverse urban structures.
*   **Published Results:** RMSE of ~1.9m - 2.5m on Vaihingen.
*   **Code Availability:** Yes (GitHub implementations).
*   **License:** Unspecified academic.
*   **Where "Metres" Actually Comes From:** Purely learned dataset prior mapping visual representations of buildings to metric height based on the training dataset.

---

### Method 5: HeightFormer (Bilateral Feature Pyramid Fusion)
*   **Input Modalities:** Single-view RGB optical image.
*   **Target:** Height map (nDSM).
*   **Backbone:** Multiscale Vision Transformer (MViT) or Multilevel Interaction Backbone (MIB).
*   **Architecture:** Transformer encoder-decoder with Bilateral Feature Pyramid Fusion (Stepwise Fusion + Multiscale Fusion).
*   **Height Formulation:** Classification-regression hybrid with adaptive height bins (Heightbins module).
*   **Segmentation/Building Branch:** No.
*   **Scale Mechanism:** Learned scale prior using image-adaptive classification bins.
*   **Treatment of Long-Tailed Heights:** Image-adaptive classification-regression (ICG). Reframes regression into coarse bins to avoid direct L1 regression penalties.
*   **Direct Metric Supervision:** Yes, supervised via LiDAR-derived ground truth nDSM.
*   **Target-City Labels Required:** No.
*   **Dataset(s):** ISPRS Vaihingen, Potsdam, DFC2019.
*   **Cross-City Evaluation:** Evaluated on cross-dataset transfers (e.g., train on Potsdam, test on Vaihingen).
*   **Strengths:** Captures long-range spatial context; provides sharp building edges; adaptive bins reduce quantization errors.
*   **Weaknesses:** Extremely heavy computationally (high FLOPs and parameter count); requires high-resolution training data to learn structural features.
*   **Known Limitations:** Difficult to train; requires huge GPU resources.
*   **Published Results:** RMSE ~1.4m on Vaihingen.
*   **Code Availability:** Yes (usually published in official IEEE GRSL/MDPI code links).
*   **License:** Academic/unspecified.
*   **Where "Metres" Actually Comes From:** Ground-truth LiDAR nDSM labels. Scale is inferred via image-adaptive classification height bins trained with ground truth heights.

---

### Method 6: Depth2Elevation (DAM with Scale Modulator)
*   **Input Modalities:** Single-view RGB optical image + relative depth map (from Depth Anything foundation model).
*   **Target:** Height map (nDSM).
*   **Backbone:** Depth Anything Model (DAM) / ViT encoder.
*   **Architecture:** Foundation model encoder with Scale Modulators and a resolution-agnostic decoder.
*   **Height Formulation:** Direct continuous regression modified by learned scale multipliers/shifts (or Scale Modulator layers in the backbone).
*   **Segmentation/Building Branch:** No.
*   **Scale Mechanism:** "Scale Modulator" modules integrated into the encoder to dynamically scale relative features, plus learned transformation parameters.
*   **Treatment of Long-Tailed Heights:** Focuses on fine-tuning DAM to learn scale, but has no explicit mathematical guard against long-tailed regression penalties.
*   **Direct Metric Supervision:** Yes, supervised via LiDAR-derived ground truth nDSM.
*   **Target-City Labels Required:** No.
*   **Dataset(s):** GAMUS, DFC2023 (in derivative studies).
*   **Cross-City Evaluation:** High generalization capability due to the pre-trained features of the foundation model, but scale drift remains a challenge.
*   **Strengths:** Leverages strong spatial representations from foundation models (DAM); handles complex visual patterns robustly.
*   **Weaknesses:** Adapting relative depth to absolute elevation still requires training/fine-tuning. If trained with a standard pixel loss, it still suffers from metric scale collapse on tall buildings in unseen cities.
*   **Known Limitations:** Pre-trained foundation models are computationally expensive to adapt/fine-tune; relative-depth features are highly resistant to scale manipulation.
*   **Published Results:** Relative error reduction of 30-42% over standard CNN-based baselines.
*   **Code Availability:** Yes.
*   **License:** Academic/non-commercial.
*   **Where "Metres" Actually Comes From:** Ground-truth LiDAR nDSM labels. The scale modulator dynamically adapts pre-trained relative features to absolute metric meters using learned scale parameters during training.

---

## 2. Critical Comparison

| Problem | Existing Methods | DepthWizard Current Approach (Phases 1-14) | What Remains Unsolved |
| :--- | :--- | :--- | :--- |
| **Relative Depth** | Scale Modulators (Depth2Elevation) | Pre-trained DA-V2 Backbone | Excellent relative structure, but scale is wrong. |
| **Absolute Scale** | Learned footprint priors (HGDNet) / Shadow metadata | End-to-end pixel-wise regression (`C_log1p`) | Network guesses scale based on dataset mean. |
| **Long-Tail (Tall)**| HTC Loss / Adaptive Bins (HTC-DC, HeightFormer) | Ignored / Simple ordinal classification | Tall buildings permanently capped at ~15-20m. |
| **Cross-City** | Multi-domain datasets | Multi-city training | Scale drifts drastically between unseen cities. |
| **Geospatial Metadata**| Camera-geometry / GSD formulas (e.g. shadow length) | None utilized | GSD is uniform, but solar/camera angle is missing. |
| **Building Footprints**| Joint decoders (HGDNet, DFC2023 baseline) | None (pure height prediction) | Misalignment of height boundaries and footprint. |
| **3D Reconstruction** | nDSM to DSM translation | Height maps converted to DSM | 3D flythrough is distorted by flat building crowns. |

---

## 3. Novelty Protection

The following concepts have been tested in our Phase 1-14 experiments and must **NOT** be claimed as novel contributions by DepthWizard, as they exist in the literature:

*   **[EXISTING IDEA] Multi-city training** for cross-domain generalization (standard practice in DFC2023).
*   **[EXISTING IDEA] Using foundation depth models** (ZoeDepth, DA-V2) as backbones (already solved in Depth2Elevation).
*   **[EXISTING IDEA] Height-bin classification / ordinal bins** to solve the long-tail problem (standard in HTC-DC Net and HeightFormer).
*   **[EXISTING IDEA] Deepest-block / Decoder adaptation** of foundation models (equivalent to Scale Modulators in Depth2Elevation).
*   **[EXISTING IDEA] Logarithmic target transformations (C_log1p)** (standard in DFC2023 baseline optimizations).
*   **[EXISTING IDEA] RGB+depth fusion** architectures (explored in RDAH-Net and general depth estimation).

---

## 4. The Unresolved Research Gap

Our extensive Phase 1-14 experiments uncovered a highly specific, unresolved problem:

**The Metric-Scale Collapse of Foundation Topography**
> A pre-trained relative-depth representation (like DA-V2) provides excellent structural topology (edges, building vs. ground) but completely fails to provide *absolute metric scale* for rare, tall structures under unseen-city transfer. Because standard dense pixel-wise regression losses (L1/MSE) heavily penalize extreme outliers, the network mathematically optimizes by ignoring the structural evidence of tall buildings and collapsing its predictions toward the dataset mean (~15m). 

**The Existing Literature Gap:**
- Methods like *Depth2Elevation* attempt to force the foundation model to learn metric scale pixel-by-pixel, which is still vulnerable to the long-tail penalty in unseen domains.
- Methods like *HTC-DC* use complex binning losses to fight the math, but still demand that the dense pixel-wise decoder outputs metric scale directly.
- *Shadow geometry* provides true physical scale but fails instantly when sun angle metadata is missing (as in DFC2023) or shadows occlude each other.

**Conclusion:** The problem of safely decoupling relative structural topology from absolute metric scale in monocular overhead imagery remains *underexplored*.

---

## 5. Proposed Novel Directions

### Direction A: Scale-Decoupled Normalized Topography (SDNT)
Instead of forcing the dense pixel-wise decoder to output metric meters (which causes the long-tail collapse), we decouple structure from scale. 
1. **Structure Branch:** The network predicts a *Normalized Topology Map* (strictly bounded 0.0 to 1.0, where 1.0 is the tallest local structure). This isolates the foundation model's strength: relative depth.
2. **Scale Branch:** A lightweight global pooling network takes the RGB context and building footprints to predict a single metric scalar per scene: the *Tile Maximum Height (Z_max)*.
3. **Fusion:** Final Metric Height = Normalized Map × Z_max.

### Direction B: Multi-Task Physical Geometry Anchoring
We introduce a triple-decoder network that forces the latent space to encode physical GSD scale by simultaneously predicting:
1. Metric Height Map
2. Binary Building Footprint (anchors spatial GSD area)
3. Shadow Extent Mask (implicitly anchors relative solar scale).
By enforcing consistency between footprint area, shadow length, and height, the network learns a physical geometric prior without requiring actual sun metadata.

### Direction C: Image-Adaptive Scale Modulation (IASM)
Similar to HTC-DC's adaptive bins, but applied directly to the foundation model's relative depth. The network predicts an affine transform (Shift and Scale parameters) dynamically generated by a Vision Transformer looking at the RGB image. These parameters scale the relative depth map into absolute metric height.

---

## 6. Scoring the Directions (1-5)

| Criterion | Dir A (Decoupled Scale) | Dir B (Triple-Task Geometry) | Dir C (Adaptive Affine) |
| :--- | :--- | :--- | :--- |
| **Novelty Potential** | 5 | 3 (Overlaps HGDNet) | 4 |
| **Fit to Failure Evidence**| 5 | 4 | 4 |
| **Metric-Height Potential**| 5 | 4 | 4 |
| **Cross-City Potential** | 5 | 4 | 4 |
| **SIH Feasibility** | 5 | 3 | 4 |
| **Compute Feasibility** | 5 | 3 | 4 |
| **Implementation** | 4 | 2 | 4 |
| **Explainability** | 5 | 4 | 3 |
| **Total Score** | **39** | **27** | **31** |

---

## 7. TOP RECOMMENDATION: Direction A (Scale-Decoupled Normalized Topography)

### WHY IT IS DIFFERENT
Existing state-of-the-art methods (HGDNet, Depth2Elevation) treat metric height as a dense, pixel-wise regression problem. This mathematically forces the network to balance structural accuracy against the massive statistical penalty of overestimating the long tail. We completely abandon dense metric regression. We treat the foundation model purely as a relative topological feature extractor, and extract the metric scale as a separate, scene-level property.

### SIH NOVELTY TEST
*If a jury asks:*
*   **What is existing?** Foundation models for relative depth (DA-V2), and joint multi-task regression networks (HGDNet).
*   **What is your gap?** Dense pixel-wise regression mathematically collapses on rare tall buildings because the loss heavily penalizes scale outliers, forcing models to predict the dataset mean.
*   **What is your contribution?** We explicitly decouple structural prediction from absolute scale prediction.
*   **Why is your approach different?** We do not force the foundation decoder to output metric meters. We generate a 0-1 normalized structural topology map, which is then dynamically scaled by a separate scene-level metric anchor network. 
*   **What exactly did you add?** The Scale-Decoupled architecture (SDNT) that protects the foundation model's relative structural integrity from the destructive gradients of long-tail metric scale regression.

### PHYSICAL SCALE SOURCE
The physical metric scale comes from the **Scene Maximum Height (Z_max) Branch**. This branch acts as a learned statistical prior anchored by the GSD footprint distribution and context of the entire tile, entirely separated from the pixel-wise structural loss.

### PROPOSED MINIMAL EXPERIMENT
1. Modify `SmallFusionUNet` to output a `tanh` or `sigmoid` activated map (0 to 1).
2. Add a simple Global Average Pooling branch to the DA-V2 neck that predicts a single scalar `Z_max`.
3. Loss = `L1(Normalized_Map * Z_max, Ground_Truth)`. 
Train on the multi-city split and evaluate if decoupling scale restores the >40m building heights on New York.

### EXPECTED PROTOTYPE INTEGRATION
Fits perfectly into the SIH pipeline. Input is RGB -> Scale-Decoupled Height -> nDSM -> DSM -> 3D Flythrough. No target-city labels or fabricated metadata are required.

---
*MANDATORY STOP EXECUTED. Awaiting human review before implementation.*
