# PHASE 16 — METRIC-SCALE OBSERVABILITY AUDIT REPORT

## 1. Metadata Findings

We systematically audited the TIFF metadata tags across representative tiles of Berlin, Brasilia, New Delhi, Copenhagen, and New York.

*   **Georeferencing Fields Present:**
    *   `ModelPixelScaleTag (Tag 33550)`: `(0.5, 0.5, 0.0)` for all tiles.
    *   `ModelTiepointTag (Tag 33922)`: Present, defining the absolute UTM coordinates.
    *   `GeoKeyDirectoryTag (Tag 34735)`: Present.
    *   `GeoAsciiParamsTag (Tag 34737)`: Present, specifying the Coordinate Reference System (CRS) e.g., `WGS 84 / UTM zone 18N` for New York.
*   **Datatype:** uint8 for RGB images, float32 for DSM elevations.
*   **Width x Height:** 512 x 512 pixels.
*   **Nodata Value:** `-999.0` (in DSMs).

**Conclusion:** Geo-referencing is fully explicit in the file format, but it is limited *only* to 2D projection and pixel dimensions.

---

## 2. GSD Findings

*   **Varies between cities?** No.
*   **Varies within cities?** No.
*   **Constant value:** Exactly **0.5m** horizontal pixel spacing for all tiles.
*   **Useful for vertical height scaling?** No. While GSD defines the horizontal physical scale (0.5m per pixel), it is a constant across all tiles in the dataset. Because it does not vary, it cannot explain why building heights differ across cities (e.g. Berlin vs. New York). It provides a fixed horizontal anchor, but does not solve the vertical scale collapse problem.

---

## 3. Camera / RPC Parameters

*   `RPCCoefficientTag (Tag 50908)`: **NOT AVAILABLE** (Absent)
*   Viewing angle / incidence angle: **NOT AVAILABLE** (Absent)
*   Camera model: **NOT AVAILABLE** (Absent)
*   Sensor metadata: **NOT AVAILABLE** (Absent)

**Conclusion:** RPC/camera parameters are completely absent. True perspective viewing geometry cannot be recovered from the files.

---

## 4. Sun / Solar Information

*   Sun elevation: **NOT AVAILABLE** (Absent)
*   Sun azimuth: **NOT AVAILABLE** (Absent)
*   Acquisition date/time: **NOT AVAILABLE** (Absent)

**Conclusion:** Solar geometry metadata is completely absent in the files.

---

## 5. Shadow Feasibility Assessment

We computationally evaluated shadow detection in New York.
*   **Are shadows visible?** Yes, building shadows are visible in the RGB images.
*   **Separability:** Extremely poor. A threshold of < 60 grayscale intensity yields only 0.20% - 3.50% of pixels, while increasing the threshold to < 80 jumps to 9.24% - 31.50% as it captures dark asphalt roads, tree shadows, and dark roof textures.
*   **Saturations and Occlusion:** In dense tall building environments like Manhattan (New York), shadows are heavily occluded, overlap with shadows of adjacent buildings, or fall on dark asphalt roads, making them extremely difficult to isolate.
*   **Directionality:** Because sun azimuth is not in the metadata, the shadow orientation cannot be predicted or validated a priori.

**Conclusion:** Shadow-based height calculation is **NOT FEASIBLE** on the DFC2023 dataset due to the complete lack of solar angles and high visual clutter on dark background pixels.

---

## 6. Shadow-Height Diagnostic

Without sun elevation or azimuth, a direct geometric calculation $H \approx L \times \tan(\theta)$ cannot be performed from metadata. Any attempt to infer it would require a learned neural network to estimate the sun's elevation first, introducing a secondary error propagation loop.

---

## 7. Building Footprint Diagnostic

We segmented building footprints (nDSM > 2.0m) into connected components and calculated the correlation between **footprint area (square meters)** and **absolute building height (maximum height of component)** across all cities:

*   **Total buildings analyzed:** 3,468
*   **Pearson Correlation R:** 0.415 (p-value: 3.772e-144)
*   **Spearman Rank Correlation R:** 0.599 (p-value: 0.000e+00)

*Interpretation:* The correlation of **0.599** is moderate. This confirms that **footprint geometry contains a strong statistical prior** (larger building footprints generally correspond to taller buildings). However, this is a learned/statistical correlation rather than a physical metric invariant (a large warehouse can be flat, and a thin skyscraper can be extremely tall).

---

## 8. Depth + Physical Cue Analysis

We tested if combining relative depth variation (Std) and building coverage (footprint fraction) in a multi-cue linear regression improves scene-scale prediction ($P_{98}$):

*   **Depth Std alone** -> $R^2$: 0.034, MAE: 15.54m
*   **Depth Std + Building Coverage** -> $R^2$: 0.109, MAE: 14.65m

*Interpretation:* Combining relative depth and spatial building footprint coverage improves the scale fitting ($R^2$ rises from 0.034 to 0.109, and scale prediction MAE drops to 14.65m). This indicates that **footprint statistics provide a valuable secondary cue for scaling**.

---

## 9. Cross-City Transfer Assessment

*   **GSD:** Generalizes perfectly (it is 0.5m everywhere).
*   **RPC/Camera:** Transfer is impossible (unavailable).
*   **Shadows:** Highly unstable across cities due to varying solar elevation angles and different background road albedos.
*   **Footprint Area:** Represents a domain-dependent prior (e.g. Copenhagen has large, flat low-rise structures, whereas New York has tall, slender skyscrapers). Footprint-to-height scaling coefficients do *not* generalize zero-shot.

---

## 10. Candidate Cue Classification

| Candidate Cue | Classification | Physical Metric Anchor? | Transferable? |
| :--- | :--- | :--- | :--- |
| **GSD (ModelPixelScale)** | **STRONG PHYSICAL ANCHOR (Horizontal)** | Yes (Horizontal only, not vertical) | Yes |
| **GeoTIFF Affine Transform** | **STRONG PHYSICAL ANCHOR (Horizontal)** | Yes (Horizontal only) | Yes |
| **RPC / Camera Parameters** | **UNAVAILABLE** | No | No |
| **Sun / Solar Geometry** | **UNAVAILABLE** | No | No |
| **Shadow Geometry** | **NOT USEFUL** (Due to lack of solar metadata) | No | No |
| **Footprint Area** | **WEAK PRIOR** (Statistical) | No | No (Domain-dependent) |
| **Depth Anything V2** | **WEAK PRIOR** (Relative structure only) | No | Yes (Structure transfers, scale collapses) |

---

## 11. DFC2023 vs. Realistic SIH Inference Availability

| Cue | Available in DFC2023? | Realistically Available in SIH Inference? |
| :--- | :--- | :--- |
| **GeoTIFF GSD (0.5m)** | Yes (Explicit) | Yes (For orthorectified remote sensing data) |
| **Solar Angles (Elevation/Azimuth)** | No (Absent) | Yes (Typically present in raw L1B/L2 satellite metadata) |
| **RPC Coefficients** | No (Absent) | Yes (Standard for raw satellite passes) |
| **Building Footprint Mask** | Yes (Can be predicted) | Yes (Can be predicted via building detector) |

---

## 12. Final Decision

```text
NO RELIABLE PHYSICAL CUE — LEARNED SCALE METHOD REQUIRED
```

### Answers to the 10 Primary Questions:

1.  **Where can metric scale realistically come from?**
    It must come from a **learned scale-anchoring network** that maps relative depth features and visual appearance features to scene scale, regularized by a **multi-task building footprint constraint**.
2.  **Is shadow geometry usable?**
    No. It is completely unavailable in the dataset and highly occluded in dense high-rise urban areas.
3.  **Is GSD useful?**
    Yes, for horizontal scale. But since GSD is constant (0.5m) in the dataset, it provides no vertical variance.
4.  **Are RPC/camera parameters available?**
    No. They are completely absent (`NOT AVAILABLE`).
5.  **Can footprint geometry help?**
    Yes. It provides a weak statistical prior ($R \approx$ 0.599), showing that building footprint size correlates with height.
6.  **Does depth + physical cue improve the scale relationship?**
    Yes. Combining relative depth std and building footprint coverage improves scale regression $R^2$ to 0.109.
7.  **What cue is transferable across cities?**
    GSD is the only physical cue that transfers perfectly.
8.  **What cue is realistically available during SIH inference?**
    GSD (horizontal resolution) and predicted building footprints.
9.  **What should our next height-estimation architecture use?**
    A **hybrid scale prediction module** that:
    *   Predicts normalized height $N$.
    *   Predicts building footprints $F$.
    *   Uses a learned regressor to predict scale $S$, regularized by the spatial area of the predicted building footprint $F$ using a GSD scaling constraint.
10. **What is the smallest experiment that can falsify that design?**
    A scale regressor trained on multi-city data using both depth features and predicted footprint area to predict $P_{99}$ scale, evaluated zero-shot on New York. If the scale error remains $>40$m on New York, the footprint anchor is falsified.

---
*MANDATORY STOP EXECUTED. Awaiting human review.*
