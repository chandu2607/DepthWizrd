# Phase 69 Terrain Target Definition

This phase is terrain-first and does not claim building-height accuracy.

## Target
The model is trained to predict a terrain elevation / DTM-like surface from RGB imagery.

## Why this is the right first target
- The public DEMs are real terrain references and can be read directly with rasterio.
- The current project architecture is explicitly building-conditioned and object-level.
- The existing building-conditioned head expects a building mask and component-wise height target, not a continuous terrain surface.

## Why the current model is not directly compatible
The code in `depthwizard/models/building_conditioned_net.py` is built around connected components, building roof masks, and object-level height regime prediction. It models building structure and roof-geometries, not DTM terrain. It is therefore not a terrain-elevation regressor without a new target head.

## Minimum required architecture change
A terrain branch must replace the building-conditioned head with a dense regression head that maps RGB to DEM, e.g. a small U-Net or CNN outputting a single-channel terrain elevation map. The output is continuous and should be supervised with DEM values, not with building-object heuristics.

This is the minimum viable architecture change required for a terrain-first Indian pilot.
