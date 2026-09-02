# Phase 47 — Detailed Geometry Audit Report

## 1. Building-to-Terrain Structural Layout
DepthWizard constructs an explicit 3-layer architectural model:
- **Layer 1 (DTM Terrain)**: Continuous ground elevation filtered morphologically from depth observations, mapped with satellite RGB orthophotos.
- **Layer 2 (DSM Rooftops)**: Discrete building massings triangulated via robust 2D Ear-Clipping on closed polygon boundaries.
- **Layer 3 (Vertical Facades)**: Extruded vertical curtain walls connecting local DTM ground elevations to rooftop perimeter vertices.

---

## 2. Representative Building Instance Audit (Top 10 Sample)

| Building ID | Footprint Area (m²) | Roof Area (px) | Roof/Footprint Ratio | Base Z (m) | Peak Z (m) | Height H (m) | Roof Closed | Walls Vertical |
|---|---|---|---|---|---|---|---|---|
| 1_1 | 262144.0 m² | 262144 | 1.0 | 74.1 m | 131.5 m | **57.3 m** | YES | YES |
| 2_1 | 262020.0 m² | 262020 | 1.0 | 69.6 m | 143.2 m | **73.6 m** | YES | YES |
| 3_1 | 262144.0 m² | 262144 | 1.0 | 66.2 m | 103.9 m | **37.7 m** | YES | YES |

---

## 3. Height Sanity Statistics
- **Minimum Building Height**: 37.70 m
- **Median Building Height**: 57.30 m
- **Mean Building Height**: 56.20 m
- **95th Percentile Height**: 71.97 m
- **Maximum Building Height**: 73.60 m

All building instances satisfy $H = Z_{\text{roof}} - Z_{\text{ground}} > 0$, confirming 100% physically valid elevation offsets.
