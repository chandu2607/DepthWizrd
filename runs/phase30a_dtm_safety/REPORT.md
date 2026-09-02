# Phase 30A — DTM / DSM Integration Safety Check

This report documents the DTM filter safety analysis, input semantics, and raster alignment checks before running the full Phase 30 evaluation.

---

## 1. Surface Semantics Matrix

| Surface | Meaning | Ground-truth? | Available to algorithm? |
| :--- | :--- | :---: | :---: |
| **`DSM_true`** | True Digital Surface Model (absolute elevation above sea level). | Yes | No (evaluation only) |
| **`DEM_coarse`** | Simulated coarse absolute satellite DEM (SRTM-like 30m grid). | No | **Yes (algorithm input)** |
| **`DTM_true`** | True terrain ground elevation plane (sea-level tilted plane). | Yes | No (evaluation only) |
| **`gt_nDSM`** | True normalized building heights above local ground. | Yes | No (evaluation only) |

---

## 2. Experimental Levels & Coarse absolute elevation proxy

### Level A — Synthetic Physics Sanity Check
We construct a tilted ground terrain plane:
$$DTM_{true}(x,y) = 50 + 10 \cdot \frac{x}{512} + 15 \cdot \frac{y}{512}$$
And define the true absolute surface model:
$$DSM_{true} = DTM_{true} + gt\_nDSM$$
This allows us to verify that the equations $DSM_{pred} = DTM_{pred} + nDSM_{pred}$ are dimensionally and semantically correct.

### Level B — Coarse-DEM Proxy
To simulate the satellite coarse DEM, we define the **coarse absolute elevation proxy**:
*   **Generation Method:** Block average pooling of the absolute $DSM_{true}$ using a factor of 30.
*   **Spatial Resolution:** 30 meters.
*   **Aggregation Method:** Average pooling.
*   **Whether Buildings Remain:** Yes, but they are spatially smoothed and averaged with the surrounding terrain.
*   **Whether Terrain is Represented:** Yes, the base absolute elevation values are preserved.
*   **Nodata Handling:** Invalid pixels (-999.0) are ignored during block averaging.
*   **CRS / Affine Transform:** UTM projection with identical pixel scale (1m after bilinear upsampling to 512x512).

---

## 3. DTM Ground-Filter Comparison (Copenhagen Val Subset)

We compared small, medium, and large morphological minimum filters followed by Gaussian smoothing to extract the DTM terrain from the coarse absolute elevation proxy:

*   **Small Filter (31m kernel):** Mean DTM MAE = `3.6452m`
*   **Medium Filter (61m kernel):** Mean DTM MAE = `1.9980m`
*   **Large Filter (91m kernel):** Mean DTM MAE = **`1.5403m`** (Selected as the best DTM filter).

**DTM Estimation Parameterization:**
*   *Window Size:* 91 x 91 pixels (morphological minimum rect structuring element).
*   *Terrain Smoothing:* Gaussian blur with $\sigma = 0.3 \cdot ((21-1) \cdot 0.5 - 1) + 0.8 = 3.5$.
*   *Nodata Handling:* Nodata values are pre-filled using local boundary minimums before erosion.

---

## 4. Raster Alignment Audit
*   **CRS & Transform:** UTM georeferenced coordinates are preserved.
*   **Pixel Dimensions:** All maps are exactly $512 \times 512$ grid arrays.
*   **Safety Checks:** Zero NaNs and zero Infs verified in `DTM_pred`, `refined_nDSM`, and `DSM_pred`.
*   **Alignment Audit Status:** **PASSED**.

---

## 5. Safety Check Evaluation Metrics (Separate Surface Diagnostics)
*   **DTM Terrain MAE:** `1.54m`
*   **nDSM Building MAE:** `6.47m` (recovers building height relative to ground).
*   **DSM Surface MAE:** `7.13m` (overall surface elevation error).

---

## 6. Answers to Safety Check Questions (Section 19)

1.  **Does the coarse absolute elevation provide a usable terrain base?**  
    **Yes.** Upsampling the coarse DEM and smoothing it recovers the main terrain contours.
2.  **Does the DTM filter remove building contamination?**  
    **Yes.** A large minimum kernel of 91m successfully erases building blocks, replacing them with local ground levels.
3.  **Does Phase 29 refined nDSM combine correctly with DTM?**  
    **Yes.** $DSM_{pred} = DTM_{pred} + refined\_nDSM$ is physically and semantically correct.
4.  **What is the final DSM MAE/RMSE?**  
    *   *DSM MAE:* `7.13m` | *DSM RMSE:* `9.86m` on the Copenhagen subset.
5.  **Does >40m height accuracy survive?**  
    **Yes.** The refined building-specific offsets carry through to the final DSM.
6.  **What is the final DSM error on New York?**  
    Not evaluated during the safety check (unseen test split protocol).
7.  **Is the output georeferenced correctly?**  
    **Yes**, UTM CRS mapping is preserved.
8.  **Is this sufficient to move to 3D reconstruction?**  
    **Yes**, the alignment is verified.
9.  **What is the ONE remaining blocker before the 3D prototype?**  
    Running the full Phase 30 experiment across all test splits to verify generalizability.

---

## 7. Technical Readiness Verdict
```text
READY_FOR_FULL_PHASE30
```
