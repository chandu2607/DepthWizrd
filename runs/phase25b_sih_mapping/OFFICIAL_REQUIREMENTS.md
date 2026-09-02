# Phase 25B — Official SIH/ISRO Requirement Mapping

This document provides a source-locked analysis of the official **Smart India Hackathon (SIH) 2026 / Indian Space Research Organisation (ISRO) Problem Statement (ID: 26175)** titled **"DepthWizard - Single-View Height Estimation and 3D Flythrough"**. It compares the official requirements against the current `DepthWizard` repository and defines the technical route forward.

---

## 1. Official Input Matrix

Based strictly on the official problem description and dataset guidelines, here is the availability of candidate inputs:

| Input Candidate | Officially Available? | Exact Description / Source Citation | Metric Information? | Required / Optional |
| :--- | :--- | :--- | :--- | :--- |
| **Single RGB Image** | **EXPLICITLY PROVIDED** | "transforms single-view optical RGB remote-sensing images" (Description) | No direct vertical scale | **REQUIRED** |
| **Georeferenced / Non-georeferenced support** | **EXPLICITLY PROVIDED** | "The framework must support both non-georeferenced and georeferenced imagery." (Description) | Horizontal coordinate reference | **REQUIRED** |
| **Low-Resolution DEM (e.g. SRTM 30m)** | **OPTIONAL** | "A lower-resolution DEM source such as SRTM 30 m may be used to map scale-agnostic depth features to absolute metric elevations." (Dataset Link) | Direct coarse vertical metric elevations | **OPTIONAL (Recommended)** |
| **Stereo / Multi-View** | **EXPLICITLY NOT PROVIDED** | "Traditionally, elevation data is acquired through stereo-imaging pairs... These approaches can be cost-prohibitive... Single-view height estimation offers an agile alternative." (Description) | N/A | **ABSENT** |
| **LiDAR / Point Cloud** | **EXPLICITLY NOT PROVIDED** | "Traditionally... LiDAR... cost-prohibitive." (Description) | N/A | **ABSENT** |
| **SAR / InSAR** | **EXPLICITLY NOT PROVIDED** | "Traditionally... Synthetic Aperture Radar (InSAR)... cost-prohibitive." (Description) | N/A | **ABSENT** |
| **RPC Metadata** | **UNCLEAR FROM SOURCE** | *NOT SPECIFIED BY OFFICIAL DOCUMENT* | N/A | **UNKNOWN** |
| **Camera Calibration / Viewing Angle** | **UNCLEAR FROM SOURCE** | *NOT SPECIFIED BY OFFICIAL DOCUMENT* | N/A | **UNKNOWN** |
| **Sun Geometry / Acquisition Time** | **UNCLEAR FROM SOURCE** | *NOT SPECIFIED BY OFFICIAL DOCUMENT* | N/A | **UNKNOWN** |
| **GSD (Ground Sample Distance)** | **UNCLEAR FROM SOURCE** | *NOT SPECIFIED BY OFFICIAL DOCUMENT* | N/A | **UNKNOWN** |

---

## 2. Official Output Matrix

Below are the deliverables expected for the competition:

| Output Candidate | Officially Required? | Description / Source Citation |
| :--- | :--- | :--- |
| **High-Precision Elevation Maps** | **EXPLICITLY REQUIRED** | "transforms single-view optical RGB remote-sensing images into high-precision elevation maps." (Description) |
| **3D Flythrough / Interactive Asset** | **EXPLICITLY REQUIRED** | "transform static elevation profiles into interactive 3D assets that can be navigated in real time." (Description) |
| **Building Heights / Footprints** | **UNCLEAR FROM SOURCE** | *NOT SPECIFIED BY OFFICIAL DOCUMENT* (Implied as sub-tasks to generate high-precision DSM/nDSM). |
| **GeoTIFF / Georeferenced outputs** | **EXPLICITLY REQUIRED** | Implied by "must support both non-georeferenced and georeferenced imagery." (Description) |

---

## 3. Metric-Scale Audit

To resolve the vertical scale collapse identified in Phases 1–24, we classify the available inputs:

*   **Low-Resolution DEM (SRTM 30m):** **A. Direct vertical metric information.** It provides coarse ground elevations relative to sea level. By spatially aligning the high-res image and the SRTM DEM, we can calculate a robust scale and shift parameter to map relative depth features to absolute elevations.
*   **GSD (Ground Sample Distance):** **C. Only horizontal metric information.** It dictates pixel-to-meter conversions horizontally, but does not constrain vertical building heights.
*   **Single RGB Image:** **D. No useful metric vertical information.** Monocular depth features (texture, shading, perspective) only encode relative ordinal depth, not absolute heights.

---

## 4. Compare Against Current DepthWizard

| Official Requirement | Current DepthWizard Implementation | Status |
| :--- | :--- | :---: |
| **Single RGB Input** | RGB + Depth Anything V2 monocular inference | **ALREADY SATISFIED** |
| **SRTM 30m DEM Integration** | None. Relies purely on relative depth and heuristics | **NOT SATISFIED** |
| **High-Precision Elevation Map** | Outputs static nDSM predictions | **PARTIALLY SATISFIED** |
| **Interactive 3D Flythrough** | None. No 3D asset generation or real-time visualization | **NOT SATISFIED** |
| **Georeferenced GeoTIFF** | Reads and saves standard local files without projection logic | **PARTIALLY SATISFIED** |

---

## 5. Technical Route & Survival Test

### Decision Tree Route:
```text
F — HYBRID AI + GEOMETRY
```
*Rationale:* The official task provides single-view RGB and allows a lower-resolution DEM source (such as SRTM 30m) to map scale-agnostic features to absolute elevations. This requires a hybrid approach combining monocular depth models (AI) with geographical scale registration and terrain subtraction (Geometry).

### Project Survival Test:
```text
YES, WITH MAJOR CHANGES
```
*Explanation:* monocular relative-depth AI alone is mathematically incapable of solving absolute vertical height on unseen cities. However, because the official statement permits using SRTM 30m DEMs, the project can survive by implementing a **DEM-guided scale calibration layer** to scale relative predictions into absolute elevations. Additionally, we must build an **interactive 3D flythrough** rendering pipeline to satisfy the primary visualization deliverable.

---

## 6. Novelty & Minimum Viable Prototype (MVP)

### Genuinely Open Novelty Opportunity:
- **Coarse-to-Fine Elevation Registration:** Designing an algorithm that aligns a high-resolution, scale-agnostic relative depth map with a low-resolution absolute DEM (SRTM 30m), preserving fine building structures (from the RGB) while anchoring absolute scale and ground terrain (from the DEM).

### Minimum Viable Prototype Flow:
```
  [Single RGB Image] ────> [Depth Anything V2] ───> [Relative Depth Map]
                                                           │
                                                           ▼
  [SRTM 30m DEM] ────────> [Bicubic Upsampling] ──> [DEM-guided Calibration]
                                                           │
                                                           ▼
  [Interactive 3D Flythrough] <── [3D Mesh / PyVista] <── [Absolute DSM]
```

### Research vs Prototype:
The **Competition Demo / Prototype** must be solved first, as the interactive 3D flythrough and the low-resolution DEM scaling are the defining judging criteria.
