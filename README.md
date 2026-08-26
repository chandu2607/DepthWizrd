# DepthWizard — Phase 1: Feasibility Before We Build

Single-view optical RGB → **object height (nDSM)** → (later) absolute DSM → (later)
3D flythrough. This repository is **Phase 1 only**: a disciplined experiment to
decide whether the core idea is worth building, **before** writing a full system.

> **The one question Phase 1 answers:** *Can a small learned RGB + relative-depth
> fusion head estimate height (nDSM) meaningfully better than trivial affine
> calibration of the depth prior — especially on a **city it never saw in
> training**?* The answer gates everything else via a **GO / MODIFY / ABANDON**
> checkpoint.

---

## The hypothesis under test

```
RGB image
   │
   ├─► Depth Anything V2 (FROZEN)  ──►  relative depth  (a structural prior, NOT meters)
   │                                        │
   └────────────────────────────┬──────────┘
                                 ▼
              small learned fusion head (RGB + depth → nDSM, ~few 100k params)
                                 ▼
                       nDSM  (object height above ground, meters)
                                 │
                     (Phase 3+)  ├─► + DTM (from a DEM)  ──►  DSM
                                 └─► 3D flythrough
```

Phase 1 builds and measures **only the left half** (→ nDSM). The `+ DTM → DSM` and
3D steps are deliberately deferred until the checkpoint says GO.

### Baselines (each is a swappable `HeightEstimator`)

| ID | Method | Metric? | Why it exists |
|----|--------|:------:|---------------|
| **A** | Raw relative depth | no | Ceiling of any monotonic scaling. If Pearson *r* ≈ 0 here, nothing downstream can help → the premise fails. |
| **B** | Global affine `h = a·d + b` (fit on train city) | yes | The "just calibrate the depth" strawman. **The bar C must beat.** |
| **C** | Small learned RGB+depth fusion U-Net → nDSM | yes | **The hypothesis.** Tiny on purpose, so any win is from *learned fusion*, not capacity. |
| **D** | RDAH-Net (published reference) | yes | *Optional*, bounded-effort. We report only numbers **we** measured — never RDAH-Net's published accuracy as ours. |

Also reported: a per-image **oracle affine** upper bound (peeks at each tile's GT;
not deployable) to separate *"the signal is there but scale drifts per scene"*
from *"there is no signal."*

---

## Key concepts

- **nDSM** (normalized DSM) = height of objects **above the local ground** (a
  building's height, a tree's height). This is what C predicts, in meters. Also
  called **AGL** (above-ground level) in DFC2019.
- **DTM** = bare-earth terrain elevation (no objects).
- **DSM** = **DTM + nDSM** = full surface elevation. Producing an accurate DSM
  needs an accurate DTM (from a DEM), whose error compounds with nDSM error. Phase
  1 evaluates **nDSM only** and does not claim DSM accuracy.
- **Relative vs metric depth:** Depth Anything V2 outputs *dimensionless,
  scale/shift-ambiguous* depth. For near-nadir overhead imagery, true camera-depth
  variation across a scene is negligible, so the model expresses **learned
  appearance priors**, not measured geometry. We treat it strictly as a frozen
  prior — never as a height sensor.

---

## Why the split is city-held-out (and not random)

Random same-city tile splits leak context (same buildings, same sun angle, same
sensor) and **overstate generalization**. The mandatory protocol here trains on
one city (`JAX`), keeps a small in-domain validation set of held-out `JAX` tiles,
and reports the **headline number on an entirely unseen city (`OMA`)**. Even that
is only a *within-DFC2019* proxy; the real deployment risk is **cross-sensor /
cross-country shift to Indian ISRO imagery**, which Phase 1 cannot measure and
does not claim.

---

## Run it

### Colab / Kaggle GPU (recommended)

Open [`notebooks/phase1_feasibility_colab.ipynb`](notebooks/phase1_feasibility_colab.ipynb)
and run top to bottom. It does a fast **self-check** on synthetic data first, then
the real run, then displays the report + error maps and stops at the checkpoint.

### Local

```bash
pip install -r requirements.txt

# offline plumbing self-check (synthetic + fake depth; NOT evidence)
python scripts/run_phase1.py --config configs/smoke.yaml --allow-fake-depth

# confirm your real tiles are discoverable
python scripts/00_fetch_dataset.py configs/phase1.yaml

# the real experiment (needs DFC2019 tiles under data/dfc2019/ and torch)
python scripts/run_phase1.py --config configs/phase1.yaml
```

Outputs land in `runs/phase1/`: `results.json`, `EXPERIMENT_RESULTS.md` (also
copied to the repo root), and `figures/` (RGB | reference | prediction | error,
plus a pred-vs-GT scatter). Depth-prior outputs are cached, so re-runs are fast.

### Get the data

DFC2019 Track-1 (US3D) needs a free IEEE DataPort login + EULA (or a Kaggle/HF
mirror). Place `*_RGB.tif` / `*_AGL.tif` / `*_CLS.tif` under `data/dfc2019/`.
See `depthwizard/data/fetch.py` and notebook §4.

---

## Reading the decision

`depthwizard/eval/decision.py` maps **measured** metrics to a recommendation via
transparent, editable criteria (`DecisionThresholds`). Thresholds are either
signal sanity checks or the **user-supplied reference points** (learned methods
sit in a rough **2–4 m RMSE** band; a shadow-based reference was **~3.84 m RMSE**)
— **context, not pass/fail targets**. The verdict is a decision *aid*; a human
makes the final call using the numbers **and** the error maps.

- **GO** — C beats B cross-city, within/near the reference band, gap acceptable.
- **MODIFY** — signal exists but C doesn't beat B cross-city, or a guardrail fails.
- **ABANDON** — the frozen depth prior carries ~no monotone height signal.
- **INCONCLUSIVE** — synthetic/fake run, or C missing (no torch). Not a verdict.

---

## What Phase 1 does **not** prove

- Accuracy on **Indian ISRO optical imagery** (domain shift — the top open risk).
- Full **DSM** accuracy (nDSM only; DTM error is not included).
- That a bigger model wouldn't do better (C is intentionally tiny).

---

## Project layout

```
depthwizard/
  config.py              dataclass config tree (YAML-loaded, reproducible)
  data/datasets.py       DFC2019 canonical loader, city grouping, synthetic gen
  data/fetch.py          bounded-effort acquisition + synthetic fallback
  depth/depth_anything.py frozen Depth Anything V2 wrapper (disk-cached)
  depth/fake.py          offline fake-depth stub (self-check only, NOT evidence)
  models/base.py         HeightEstimator interface (swappable methods)
  models/affine.py       Baseline A (raw) + B (global affine) + oracle
  models/fusion_head.py  Baseline C (small learned RGB+depth U-Net)
  models/rdah_net.py     Baseline D adapter (optional, TODO-gated)
  metrics/height_metrics.py  MAE/RMSE/Pearson, class split, aggregation (numpy-only)
  eval/evaluate.py       in-domain vs cross-city eval loops
  eval/decision.py       GO/MODIFY/ABANDON from measured metrics
  eval/report.py         renders EXPERIMENT_RESULTS.md
  viz/plots.py           qualitative maps + scatter
scripts/                 00_fetch_dataset, run_phase1, 03_make_report
configs/                 phase1.yaml (real), smoke.yaml (synthetic)
tests/                   metrics + decision unit tests
notebooks/               phase1_feasibility_colab.ipynb
```

Run the tests with `python tests/test_metrics.py && python tests/test_decision.py`
(or `pytest`).

---

## Hardware

Baselines A/B and all metrics are numpy-only and run anywhere. Baseline C and the
Depth Anything V2 prior want a GPU (free Colab/Kaggle is plenty for Phase 1);
CPU works but is slow. Nothing here trains a large model from scratch.
