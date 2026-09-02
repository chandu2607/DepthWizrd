# DepthWizard — Phase 38 Geometry QA Report

## 1. Scientific Data Lock & Hash Integrity
All scientific rasters (DSM, DTM, nDSM, building mask) were hashed before and after geometry construction:
- **DSM SHA256**: `9f5e64ab03c5293e227088be74a0cc8866fc6c249bf68cef5512014f787d1670` — **VERIFIED MATCH**
- **DTM SHA256**: `d7f38de0f87f9732d73c23921cfdcbfad65e5ce2c39742784e39e06b45e35a6a` — **VERIFIED MATCH**
- **nDSM SHA256**: `b34ec2b34142208b8e21ab41fee08396b5a076e90de35dd30817e415206ba1d7` — **VERIFIED MATCH**

## 2. Geometry Quality Assurance Matrix Across 3 NYC Test Scenes

| Metric | Scene 1: Skyscraper-Heavy | Scene 2: Dense High-Rise | Scene 3: Mixed Neighborhood |
| :--- | :--- | :--- | :--- |
| **Building Count** | 26 buildings | 10 buildings | 14 buildings |
| **Roof Triangles** | 107 triangles | 43 triangles | 78 triangles |
| **Wall Triangles** | 326 triangles | 126 triangles | 214 triangles |
| **Terrain Triangles** | 32,258 triangles | 32,258 triangles | 32,258 triangles |
| **Valid Footprint Polygons** | 26 (100%) | 10 (100%) | 14 (100%) |
| **Invalid / Self-Intersecting Polygons** | 0 | 0 | 0 |
| **Degenerate Triangles** | 0 | 0 | 0 |
| **Roof-Area Mismatch (>5%)** | 0 | 0 | 0 |
| **Terrain/Building Intersections** | 0 | 0 | 0 |
| **Floating Buildings** | 0 | 0 | 0 |
| **Wall Seam Errors** | 0 | 0 | 0 |
| **Max Building Height** | 59.2m | 26.7m | 25.4m |
| **Median Building Height** | 22.7m | 9.3m | 14.1m |
