# Phase 73 terrain regression architecture

- Model: lightweight dense terrain regression head
- Backbone: frozen Depth Anything V2 relative-depth prior
- Fusion: SmallFusionUNet, 4 input channels (RGB + normalized relative depth)
- Output: 1-channel dense terrain map on the aligned common grid
- Loss: SmoothL1 (Huber-like) over valid pixels only
- No Canny, no point cloud, no building branch
