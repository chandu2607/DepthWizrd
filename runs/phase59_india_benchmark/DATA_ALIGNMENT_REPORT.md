# Phase 59: Data Alignment Report

## Scope

This report covers the spatial and geometric sanity checks required before any Indian benchmark can be used for baseline evaluation.

The purpose is not to modify data. The purpose is to check whether the optical image and reference elevation layers share compatible geometry.

---

## Required alignment checks

For any candidate Indian benchmark, the following checks must pass before using it:

- CRS consistency between RGB and DEM / DSM
- overlapping bounds between image and reference layers
- similar or compatible resolution
- nodata handling strategy
- image orientation sanity check
- vertical units sanity check
- missing-pixel handling
- elevation range plausibility

---

## Expected issues in Indian hilly data

Indian mountainous regions commonly show the following alignment challenges:

1. CRS mismatches between WGS84 geographic and local UTM or state grids
2. DEMs at 30 m resolution while RGB may be 10 m or finer
3. missing or masked terrain in cloud/shadow areas
4. different radiometric or image orientation conventions
5. elevation products that are demarcated by nodata or masked water pixels
6. local terrain that differs strongly from urban assumptions used in training

These do not invalidate the benchmark, but they require careful pre-checking.

---

## Conditions required for valid use

The Indian benchmark is valid for baseline evaluation only if:

- the RGB and DEM align on the same projected coordinate system,
- the DEM and image overlap sufficiently,
- the image and DEM are not grossly misregistered,
- no obvious nodata or orientation failure exists,
- the reference height values are meaningful in meters,
- the benchmark is not dominated by missing pixels or broken georegistration.

If any of these fail, we must record the benchmark as unsuitable for current baseline evaluation.

---

## Alignment decision rule

We do not proceed to metric evaluation until the benchmark passes:

- spatial overlap test,
- CRS compatibility test,
- resolution sanity test,
- nodata and missing-pixel test,
- elevation plausibility test.

Otherwise the benchmark is not scientifically usable for model evaluation.

---

## Recommendation

Use the smallest public DEM-optical pairing that passes alignment checks. Do not attempt to evaluate a benchmark that fails simple geometry sanity checks.
