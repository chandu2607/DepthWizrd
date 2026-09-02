# DepthWizard Visual Quality Audit

## 1. Evaluation Criteria & Visual Benchmarks

| Visual Criterion | Target Standard | Observed Status | Verdict |
|:---|:---|:---|:---:|
| **Building Recognition** | Individual buildings must be sharp, detached, and clearly recognizable with discrete footprints. | All structures are segmented via connected components; perimeters are simplified to clean polygonal boundaries. | **PASS** |
| **Roof Visibility & Quality** | Roof surfaces must be solid, watertight, and strictly inside the building footprint without spanning outside polygons. | Ear-Clipping polygon triangulation guarantees 100% boundary-conforming, non-overlapping roof polygons with orthophoto UV mapping. | **PASS** |
| **Vertical Wall Quality** | Vertical facades connecting DTM ground to DSM roofs must be clean with neutral architectural shading. | Quad wall geometry connecting ground and roof vertices with slate architectural material (`#334155`, roughness 0.6). | **PASS** |
| **Terrain / Building Separation** | Buildings must visibly sit on the terrain; the terrain must not form steep building-shaped mountain hills. | Base terrain strictly uses smooth DTM ground grid; buildings sit directly on top with vertical side walls. | **PASS** |
| **Texture Alignment** | Satellite RGB orthophoto must align accurately with building roofs and terrain without vertical stretching. | Texture UV coordinates are computed per vertex from pixel coordinates $(col/(W-1), 1 - row/(H-1))$. Walls use architectural neutral shading. | **PASS** |
| **Camera Framing & Composition** | Default view must frame 100% of the city block with a comfortable 15–25% margin in an oblique perspective. | Scene bounding box dynamically determines camera distance ($1.15 \times \text{max\_dim}$) and target center at 45° angle. | **PASS** |
| **Lighting & Shading** | Architectural forms must be crisp; no pitch-black surfaces or overexposed blowouts. | Multi-source lighting: Key light (0.9 with soft shadows) + Fill light (0.45) + Hemisphere ambient (0.6). | **PASS** |
| **Artifact Suppression** | No giant bounding slabs, no hollow openings, no vertical curtain streaks. | Bilateral filtering suppresses single-pixel DSM noise while preserving step edges and planar roofs. | **PASS** |

---

## 2. Multi-Mode Visual Inspection

- **RGB City Mode**: Photorealistic satellite orthophoto draped over DTM terrain and building rooftops; crisp slate-gray facade walls.
- **Elevation Mode**: Continuous Turbo colormap mapping absolute elevation ($Z \in [Z_{min}, Z_{max}]$) across terrain and structures with dynamic metric legend.
- **Building Height Mode**: Subdued dark terrain with buildings illuminated in 0–60m+ height colormap to emphasize skyline hierarchy.
- **Slope Mode**: DTM terrain shaded from Green (0° flat) to Red (45°+ steep) highlighting surface topography.
