# Minimal Terrain Head Design

## Goal
Train a small dense terrain regression head on RGB to predict a terrain elevation map (DTM / terrain elevation), without building segmentation, Canny, point cloud, or hazard branches.

## Minimal architecture
RGB -> shared encoder -> terrain regression head -> dense terrain elevation map

- Input: RGB tiles, e.g. 512x512x3
- Backbone: frozen or lightweight encoder (Depth Anything features can be kept as a feature source; no building proposal branch is required)
- Head: 1x1 conv or small CNN decoder to a single-channel elevation map
- Activation: linear output, no bounded activation on the final layer
- Loss: SmoothL1 or L1 on DEM pixels
- Optional small gradient-consistency term: L1(|dy_pred - dy_gt| + |dx_pred - dx_gt|)

## Why this is minimal
It isolates the actual terrain learning objective and avoids mixing building height, segmentation, and disaster layers in the first experiment.
