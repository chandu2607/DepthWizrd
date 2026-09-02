# Phase 58: Canny Edge Detector Assessment

## Core principle

Canny edge detection is a low-level image operator. It is useful for finding local intensity gradients and boundaries, but it is not equivalent to semantic building detection.

This distinction matters critically for Indian hilly terrain: many natural scene boundaries are strong edges but not buildings.

---

## What Canny can help with

Canny may be valuable as a supporting operator in a post-processing pipeline, for example:

1. Building boundary refinement
   - In urban areas with crisp roof edges and strong contrast, Canny can help refine a rough roof boundary.
   - It can sharpen the boundary of a building mask when the mask is already approximately correct.

2. Roof-edge refinement
   - Useful when a model finds a coarse building footprint but the boundary needs geometric cleanup.

3. Post-processing cleanup
   - Canny can help smooth or refine edges after semantic segmentation or contour extraction.

4. Mesh boundary extraction
   - If the goal is a final geometry front-end for a mesh or polygonal building footprint, Canny may help accentuate boundaries before polygon fitting.

5. Visual structural edge detection
   - It can highlight visually strong edges for human inspection or visualization, especially in urban roofline scenes.

---

## What Canny cannot do reliably

Canny does not tell us whether an edge is a building, a road boundary, a rock face, a vegetation edge, a shadow edge, or a terrain scar.

In hilly terrain, the following are all likely to produce strong edge responses:

- vegetation boundaries,
- shadow boundaries,
- roads and tracks,
- gullies and ravines,
- ridge lines,
- rock outcrops,
- landslide scarps,
- bare-earth transitions,
- slope discontinuities.

These are not semantically equivalent to building edges.

---

## Specific limitations in Indian hilly terrain

### Vegetation
Vegetation creates complex texture and strong local edges. Canny will detect them, but they are not building rooftops.

### Natural terrain
Hills, ravines, terraces, ridges, and cut slopes all produce strong edge structure. This is terrain geometry, not building geometry.

### Shadows
Shadow edges are often strong but misleading. They can create false building boundaries and false roof-like structures.

### Roads
Road edges and transform boundaries often appear as strong straight edges and can be confused with building outlines.

### Rock boundaries
Rock outcrops and natural stone surfaces create highly irregular, sharply contrasted boundaries that Canny will detect even though they are not structures.

### Landslides
A landslide scarp or terrain failure often produces a very high-contrast edge, but that edge is a geomorphic feature, not a building outline. Canny alone would confuse hazard boundaries with building structure.

---

## Correct role for Canny

The correct role for Canny is as a supporting operator, not as a primary semantic detector.

Best use cases:

- refine an already good building mask,
- improve roof boundary polygons,
- extract geometry cues after a building or terrain segmentation stage,
- support visualization of strong discontinuities.

Not appropriate as:

- the main terrain model,
- a substitute for building detection,
- a landslide detector,
- the main hazard-analysis input,
- a terrain semantic model.

---

## Conclusion

Canny is useful for edge-based refinement, but it is not a substitute for semantic understanding.

The correct scientific distinction is:

- Edge detection: local visual boundary extraction
- Semantic building detection: understanding what the boundary belongs to

For Indian hilly disaster monitoring, Canny may be helpful only in a low-level post-processing stage. It should not be used as the core model for terrain or building detection.
