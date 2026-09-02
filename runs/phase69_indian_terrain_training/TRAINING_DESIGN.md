# Phase 69 Training Design

## Purpose
Keep the experiment small and honest. We are not claiming building-height readiness.

## Data split
- Train: Uttarakhand
- Validation: Himachal Pradesh
- Test: Sikkim (locked unseen)

## Pilot design
- Fixed seed = 0
- One epoch only
- CNN: small 3-layer terrain regressor on RGB input
- Target: DEM normalized to [0, 1]
- Loss: MSE
- Device: CPU (no CUDA available in this environment)

## Why this pilot is valid
This pilot checks whether a terrain-elevation head can learn anything from real Indian terrain data, without improperly using the building-conditioned architecture or leaking Sikkim into model selection.
