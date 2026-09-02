# DepthWizard — Phase 39 Footprint Extraction Report

## 1. Multi-Evidence Instance Extraction Strategy
Rather than relying on a single binary threshold, the presentation building instance mask is derived from 5 structural evidence layers:
1. **nDSM Height Evidence**: `ndsm >= 1.8m` filtering ground clutter.
2. **Depth Gradient Valleys**: Sobel gradient analysis on bilateral-filtered nDSM to find height drops between adjacent roofs.
3. **RGB Edge Boundaries**: Canny edge detection highlighting architectural roof boundaries.
4. **Distance Transform Peak Cores**: Local distance transform peaks identifying individual building centroids.
5. **Selective Watershed Instance Segmentation**: Watershed segmentation separating merged building complexes into true constituent building footprints.

## 2. Visual Forensic Artifacts Generated
- `01_rgb.png`: Original satellite RGB orthophoto.
- `02_mask.png`: Overlay of building candidate mask.
- `03_components.png`: Connected components colored with random RGB colors and bounding boxes.
- `04_component_classification.png`: Real building instance classification map.
- `05_final_footprints.png`: Final clean building footprint outlines.
