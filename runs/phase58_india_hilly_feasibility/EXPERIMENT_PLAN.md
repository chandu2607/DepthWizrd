# Phase 58: Minimum Scientifically Defensible Experiment Plan

## Principle

No metrics should be produced until a real Indian terrain benchmark exists. This experiment plan is designed to be the smallest valid next step, not a broad all-India study.

---

## 1. Baseline experiment

### Baseline dataset
Use the currently validated non-Indian dataset already in the project, preserving the historical separation between training, validation, and test data.

### Purpose
This establishes the current “best known urban baseline” before domain shift.

### Required outputs
- elevation MAE
- elevation RMSE
- Pearson / Spearman correlation
- building IoU where labels exist
- building height error where labels exist

---

## 2. India benchmark experiment

### Benchmark design
Select a small Indian hilly region with available public DEM / optical data and a held-out test region separate from tuning and selection.

### Benchmark minimum contents
- georeferenced Indian optical imagery,
- DEM / DSM ground truth,
- slope mask or slope map,
- building footprint labels where available,
- vegetation or land-cover labels where possible,
- at least one hazard-relevant mask such as landslide scarp or flood extent.

### Splits
- train on one Indian subregion or a small set of representative tiles,
- validate on a second region,
- test on a held-out region with no leakage.

---

## 3. Metrics to compare

### Standard geometry metrics
- elevation MAE
- elevation RMSE
- correlation with reference elevation
- building height MAE (where labels exist)
- building IoU / F1 (where labels exist)

### Terrain-specific metrics
- DTM MAE
- DTM RMSE
- nDSM MAE
- terrain slope error by slope bin
- ridge / valley elevation error
- vegetation-surface confusion penalty where relevant

### Hazard-related metrics
- landslide scar localization quality (if labeled)
- flood-relevant elevation error (e.g., relative error in low-lying / flood-prone regions)
- terrain slope error in steep zones

The key point is that disaster-focused metrics must be computed on disaster-relevant terrain regimes, not just on an overall average scene.

---

## 4. Evaluation protocol

### Minimum test protocol
1. Run the current model on the baseline urban benchmark.
2. Run the same inference on the Indian hilly benchmark.
3. Compare basin-level, building-level, and terrain-level metrics separately.
4. Stratify by slope class: gentle, moderate, steep.
5. Report performance separately for urban, peri-urban, and natural terrain.
6. Evaluate disaster-relevant subsets separately.

---

## 5. What this experiment will answer

This experiment will tell us:

- whether the current model transfers to Indian terrain at all,
- whether error is concentrated in terrain and slope regimes,
- whether building performance is still acceptable while terrain is failing,
- whether the next scientific step is terrain-aware calibration or a new terrain branch.

---

## 6. What this experiment will not do

It will not prove operational success and will not justify India deployment without the actual benchmark data. It will also not justify modifying the production architecture before the failure modes are measured.

---

## 7. Safe next step

The next step is not model retraining.

The next step is: collect or acquire the minimal benchmark, run the current model, measure real Indian terrain error, and then decide whether the model can be extended with a terrain branch or whether a different architecture is required.

---

## Final recommendation

The minimum defensible path is:

- benchmark first,
- separate terrain and building metrics,
- stratify by slope / elevation / vegetation,
- then decide architecture extensions only after real Indian evidence exists.
