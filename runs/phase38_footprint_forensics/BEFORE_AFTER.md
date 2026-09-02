# DepthWizard — Phase 38 Before vs After Comparison

| Category | Previous (Phase 37 / Earlier) | Fixed Phase 38 Reconstruction |
| :--- | :--- | :--- |
| **Building Extraction** | Connected components merged adjacent buildings into mega-blobs covering 70% of tile | Selective depth-guided morphological splitting isolates **26 individual building footprints** |
| **Roof Profile** | Per-vertex DSM boundary sampling caused spiky, noisy, mountain-like roofs | Robust interior P75 roof elevation produces **perfectly flat, solid, horizontal roofs** |
| **Roof Triangulation** | Ear-clipping on concave footprints created triangles spilling into courtyards | Point polygon test centroid validation guarantees **all roof triangles lie inside footprint** |
| **Walls** | Per-vertex DTM sampling produced wavy, curtain-like facade walls | Building-wide P30 ground elevation ensures **perfectly vertical, straight walls** |
| **Terrain Connection** | Terrain used DSM, causing z-fighting and terrain bleeding through roofs | Terrain uses pure DTM; buildings sit cleanly on ground surface with **zero z-fighting** |
| **Camera Framing** | Fixed `camDist = maxDim * 1.15` clipped tall skyscrapers | Dynamic `camDist = maxDim * 1.65` frames full city block comfortably with **~20% margin** |
