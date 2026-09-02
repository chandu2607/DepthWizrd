# Professor Method 4: Multi-View Geometry & Epipolar Parallax Feasibility Probe

## 1. Problem Statement Scope & Boundary
- **SIH Problem Statement 26175**: *"DepthWizard - Single-View Height Estimation and 3D Flythrough"*
- **Core Mission**: Reconstruct 3D elevation from a **single** optical image where stereo pairs or multi-pass constellations are unavailable.

## 2. Multi-View Geometry Formulation
When multiple overlapping satellite views $I_1, I_2$ are available with baseline $B$ and convergence angle $\theta$:
$$\text{Parallax: } p = x_1 - x_2 = \frac{B \cdot f}{Z}$$
$$\Delta H = \frac{p \cdot H_{\text{orbit}}}{B + p}$$

Triangulation requires:
1. Known epipolar geometry / rational polynomial coefficients (RPCs).
2. Dense sub-pixel stereo feature matching.
3. Multi-temporal radiometric consistency.

## 3. Comparative Feasibility Analysis

| Attribute | Single-View (DepthWizard Production) | Multi-View / Stereo |
|:--|:--|:--|
| **Input Requirement** | Single optical RGB image (PNG/JPG/GeoTIFF) | >= 2 overlapping frames with baseline |
| **Availability** | High (any single satellite pass or drone shot) | Low / Expensive (requires specialized agile passes) |
| **Computation Time** | ~0.5s per tile (Vision Transformer) | ~15-60s per tile (dense disparity matching) |
| **Scale Ambiguity** | Requires DEM / GCP / Prior calibration | Resolved geometrically if baseline is known |
| **Occlusion Handling** | Monocular prior predicts behind facades | Stereo matching fails in shadow / occlusions |

## 4. Scientific Verdict
```
SINGLE_VIEW_CORE_WITH_MULTIVIEW_OPTIONAL_EXTENSION
```
- Multi-view geometry is mathematically sound and unassailable when stereo pairs exist, but violates the single-view operational constraint of Problem Statement 26175.
- DepthWizard keeps the **single-view pipeline** as the primary production engine, while architecting the modular calibration engine to ingest multi-view parallax constraints whenever multi-pass metadata is available.
