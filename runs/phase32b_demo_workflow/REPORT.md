# Phase 32B — SIH Demonstration Workflow

## Demo Flow
| Step | Action | Status |
|------|--------|--------|
| 1 | Load Demo Scene (NYC) button | ✅ |
| 2 | RUN DEPTHWIZARD (primary button) | ✅ |
| 3 | DSM + depth outputs shown | ✅ |
| 4 | Interactive 3D City Mesh | ✅ |
| 5 | Export 4 assets | ✅ |

## Upload Support
PNG, JPEG, TIFF, GeoTIFF via sidebar uploader.

## Relative Mode
Non-georeferenced: orange RELATIVE ELEVATION MODE banner, 0-10m normalised DSM.

## Absolute Mode
Georeferenced: green ABSOLUTE DSM MODE banner, CRS/GSD/bounds table.

## DSM Output
Side-by-side: Relative Depth map + Reconstructed DSM. Six metric stat cards.

## 3D Viewer
Phase 31D edge-aware mesh. RGB/Elevation/Contour modes. 1x/1.5x/2x/3x exaggeration. Oblique/Overhead/Perspective cameras. Edge-Aware badge.

## Mesh Integration
build_edge_aware_mesh() active. Sidebar shows timing + filtered quad count.

## Export
4 buttons: DSM GeoTIFF | nDSM GeoTIFF | 3D Mesh .vtp | Preview PNG.

## Performance
Models cached (@st.cache_resource). Render cached by render_key (camera+mode+exag). No pipeline rerun on view changes.

## Error Handling
Friendly st.error() for bad files, failed inference. Relative DSM fallback for missing geo. Per-component model status in sidebar.

## New UX Elements Added
- Hero section (gradient title, tagline, description, tech pills)
- Landing feature cards (when no image loaded)
- Input metadata cards (mode banner + meta table)
- Animated step progress tracker
- Processing timing banner
- Edge-Aware 3D Mesh badge
- 4th export button (Preview PNG)
- How DepthWizard works expander
- Why This Matters 5-card section
- Sidebar: select_slider for exaggeration, demo scene label

## Verdict
**DEMO_WORKFLOW_READY**

## Next Action
PRESENT_TO_JURY
