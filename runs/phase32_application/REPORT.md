# Phase 32 — DepthWizard Application MVP Report

This report presents the validation results of the local Streamlit MVP dashboard wrapper (`app.py`).

---

## 1. Input Formats Supported
*   **Georeferenced GeoTIFFs:** `.tif`, `.tiff` (preserves UTM coordinates, affine projections, and CRS tags).
*   **Standard RGB Images:** `.png`, `.jpg`, `.jpeg` (automatically processed in relative scale Mode A).

---

## 2. Validation Test Cases & Audits

### Test Case 1: Georeferenced Mode B (GeoTIFF Upload)
*   **Input:** `SV_NewYork_40.7401_-73.9915.tif`
*   **Mode Detection:** Successful (UTM Zone 18N CRS identified).
*   **Absolute DSM Surface:** Generated successfully (Z coordinates range `53.08m` to `178.37m`).

### Test Case 2: Non-Georeferenced Mode A (RGB Upload)
*   **Input:** Standard non-georeferenced camera image.
*   **Mode Detection:** Successful (defaulted to relative depth normalization scale `[0.0, 10.0]`).
*   **Output Label:** Clearly marked as `RELATIVE DSM`.

### Test Case 3: Invalid File Upload
*   **Input:** Text Markdown file (`REPORT.md`).
*   **Handling:** OpenCV and rasterio reading failure caught gracefully, displaying a status alert to the user without raising tracebacks.

---

## 3. 3D Viewer & Exaggeration Controls
*   Supports visual height multipliers (1.0x, 1.5x, 2.0x, 3.0x).
*   Mesh coordinates remain scientifically unexaggerated.
*   RGB mapping, elevation color, and contour outlines are supported.

---

## 4. Technical Readiness Verdict
```text
READY_FOR_APPLICATION_LAYER
```
