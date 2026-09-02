# Phase 59: Minimal Benchmark Selection for Indian Hilly Terrain

## Summary

A single all-India benchmark would be scientifically premature and unrealistic for the current hardware and project scope. The smallest defensible benchmark should instead be a focused Indian hilly region with openly accessible optical imagery and a real elevation reference.

The most practical baseline candidate is:

- a small region in Uttarakhand or Himachal Pradesh,
- paired with a DEM / DSM source such as Copernicus DEM, SRTM, or a local higher-resolution DEM if available,
- a manageable region size suitable for inference testing,
- held-out evaluation by region so that we do not leak by random tile split.

This is not yet a final production benchmark; it is the smallest region that can answer whether the current model fails under Indian terrain conditions.

---

## Why this region strategy?

We want a region that has:

- mountainous terrain,
- major elevation variation,
- slopes and valleys,
- vegetation and shadow,
- roads and settlement patches,
- a plausible disaster-related terrain regime.

The Indian Himalayan belt is the most relevant and scientifically defensible choice.

### Candidate priority order

1. Uttarakhand
2. Himachal Pradesh
3. Sikkim
4. Jammu & Kashmir / Ladakh (only where suitable and available)
5. Arunachal / Northeast India only if public data is verifiable and manageable

These regions are plausible because they contain hilly terrain, slope-driven risk, and disaster-relevant terrain structure. They are also the regions that matter for the SIH question.

---

## Why not choose a huge benchmark?

A broad Indian benchmark would force us to answer too many questions at once:

- varying sensors,
- unknown image quality,
- different CRS and DEM products,
- mixed resolutions,
- mismatch between optical and elevation sources,
- broad label sparsity.

That is not the minimal scientifically valid next step.

---

## Minimal benchmark design

The minimal benchmark should contain:

- 1–2 moderate-size regions in a hilly Indian landscape,
- optical imagery at a resolution that is compatible with the current model,
- DEM or DSM reference at a similar or coarser spatial grid,
- slope-based stratification,
- a separate test region for evaluation,
- no random tile mixing across the whole region.

This keeps the benchmark small enough to be tractable while still testing the actual domain shift.

---

## What ground truth exists?

The ground truth available in a realistic open benchmark will usually be one of the following:

- DEM from SRTM / Copernicus DEM / local regional product,
- DSM if available from local LiDAR or higher-resolution altimetry,
- slopes derived from DEM,
- building footprint masks from OpenStreetMap or local vector layers,
- landslide / flood maps only as secondary hazard labels.

Important scientific limitation:

- the open benchmark may not have true landslide labels paired with every optical image,
- slope maps and DEMs are far easier to obtain than exact disaster-event labels,
- a model cannot claim landslide prediction unless hazard labels exist.

---

## What can we measure?

In a minimal benchmark we can measure:

- elevation MAE / RMSE,
- correlation with DEM / DSM reference,
- slope error by slope class,
- terrain behavior by elevation bin,
- building height / footprint where building labels are available,
- whether the failure is dominated by terrain complexity rather than building confusion.

---

## What cannot we measure yet?

We cannot yet reliably measure:

- true landslide onset prediction,
- flood-depth estimation from monocular RGB alone,
- disaster impact severity without event-specific labels,
- terrain deformation caused by a specific recent event without pre/post DEM pairs.

These remain out of scope unless a suitable event dataset is found.

---

## Recommended benchmark approach

We should choose the smallest benchmark that contains both:

1. mountainous terrain, and
2. a valid elevation reference,

not necessarily a perfect disaster dataset.

This will answer whether the current pipeline fails due to Indian terrain domain shift before we decide whether a different model branch is warranted.

---

## Final selection statement

The selected benchmark is not yet a production-grade disaster dataset. It is a small Indian hilly terrain benchmark designed to test whether the current model fails under realistic topographic complexity.

That is the scientifically appropriate first step.
