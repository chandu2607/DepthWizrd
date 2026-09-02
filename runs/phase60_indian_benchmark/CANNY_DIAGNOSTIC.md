# Phase 60 — Canny Edge Diagnostic

## Scope

This is a diagnostic-only check. Canny is not integrated into the current model path.

## Scientific role of Canny

Canny is an edge detector. It highlights local intensity transitions, not terrain semantics or building classes.

In mountainous terrain, Canny edges can help highlight:

- ridgelines
- valleys
- road boundaries
- building edges
- vegetation texture boundaries

But it cannot, by itself, unambiguously distinguish between:

- terrain structure and man-made edges,
- building boundaries and slope discontinuities,
- roads and physical terrain fractures,
- vegetation texture and object boundaries.

## Current conclusion

Canny can be useful as a diagnostic or post-processing aid only. It is not a replacement for terrain-aware geometry or a building/terrain segmentation model.

## Decision

Canny role for this project: diagnostic boundary analysis only; not an accepted part of the current pipeline.
