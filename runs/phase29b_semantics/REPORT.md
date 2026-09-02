# Phase 29B — Input Semantics and Physics Audit

This report presents a critical audit of the input elevation semantics used in Phases 26–29, aligning them with the physical realities of satellite remote sensing and the official ISRO/SIH competition constraints.

---

## 1. Trace of current Phase 29 Coarse Input

*   **Origin:** The script loads ground truth data from the `data/dfc2023_multicity/dsm` directory.
*   **Semantic Nature:** Despite the directory name being `dsm`, the ground pixels in these files are normalized to average near **0.0m**. This means these files represent **nDSM (normalized Digital Surface Model)**, which contains building heights above ground level, rather than absolute elevation above sea level.
*   **Downsampling Process:** We downsample this high-resolution nDSM to create `coarse_dem`, which is then upsampled back to `dem_up`.
*   **True Semantics:** Therefore, our input `dem_up` is actually a **coarse nDSM** (coarse above-ground building heights).

---

## 2. Surface Distinctions

| Quantity | Meaning | Contains building height? | Can be obtained directly from SRTM-like DEM? |
| :--- | :--- | :---: | :---: |
| **DEM** | Digital Elevation Model (generic term). | Partially (smoothed) | Yes (direct raw input) |
| **DTM** | Digital Terrain Model (ground bare-earth elevation). | No | No (requires ground filtering) |
| **DSM** | Digital Surface Model (absolute elevation including buildings). | Yes (full sharp height) | Yes (high-resolution output) |
| **nDSM** | Normalized DSM (building height above local ground). | Yes (isolated height) | No (requires DTM estimation) |

---

## 3. SIH-Faithfulness Classification

### Current Formulation Class: `APPROXIMATE PROXY`

### Why:
*   In the intended SIH deployment, we are given a raw satellite **absolute DEM** (e.g. SRTM 30m), which includes sea-level terrain elevation. 
*   Our current code downsamples clean ground-truth **nDSM** to create `coarse_dem`. This gives the model a **coarse building-height estimate** with a perfect 0.0m ground plane, bypassing the absolute terrain elevations and the errors introduced by terrain filtering (DTM estimation).
*   However, it remains a valid proxy because a coarse nDSM is physically derivable from SRTM by subtracting a filtered coarse DTM. It allows us to isolate the building height refinement problem from the terrain ground estimation problem.

---

## 4. Correct Reconstruction Physics for SIH Deployment

In a fully deployed system, the absolute elevation DSM must be reconstructed as follows:

```
                  RGB Image ────────┐
                                    ▼
  Coarse DEM ──► DTM Extraction ──► DTM (Terrain Ground)
       │                            │
       └───────► nDSM Extraction ───┴─► Peak Recovery ──► Reconstructed DSM
```

### Relationship:
$$DSM(x,y) = DTM(x,y) + nDSM(x,y)$$
1.  **DTM (Terrain ground):** Extracted from coarse DEM using morphological ground filtering (e.g. progressive morphological filter).
2.  **nDSM (Building height):** Refined by PeakRecoveryMLP using RGB, relative depth, and geometry.
3.  **DSM (Final output):** Sum of ground terrain DTM and the refined above-ground building height nDSM.

---

## 5. Required Semantic Adjustments

To transition from the current proxy to a fully SIH-faithful deployment:
1.  **Lock nomenclature:** Rename variables from `coarse_dem` and `dem_up` to `coarse_ndsm` and `ndsm_up` to prevent confusing ground terrain with building heights.
2.  **For Phase 29 execution:** The current formulation of $\Delta H = H_{true} - H_{coarse\_ndsm}$ remains **semantically valid** as a building-level correction because both terms are above-ground heights.
3.  **Deployment extension:** To produce the final georeferenced absolute elevation DSM (required by ISRO), we must add the predicted nDSM back to the DTM derived from the absolute georeferenced DEM.
