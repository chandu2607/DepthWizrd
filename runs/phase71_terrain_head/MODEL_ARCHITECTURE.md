# Minimal Terrain Regression Architecture

Input: RGB tile, 3 channels
Network: small CNN with 3 convolutional blocks and a final 1x1 regression head
Output shape: [B,1,H,W]
Target semantics: dense terrain elevation in meters on a common geospatial grid
Loss: SmoothL1 / Huber
No building mask, Canny edge branch, point cloud, or hazard head included in this phase.
