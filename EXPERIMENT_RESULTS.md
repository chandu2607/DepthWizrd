# DepthWizard — Phase-1 Feasibility Results

> **STATUS: TEMPLATE — NOT YET RUN ON REAL DATA.**
> This file is a placeholder. `scripts/run_phase1.py` **overwrites it** with the
> measured numbers from an actual *evidence-valid* run (a synthetic smoke run
> writes only to `runs/…/` and never touches this file). Every value below reads
> `TO BE FILLED BY RUN` until then. Nothing here is fabricated, and no verdict has
> been reached.

To produce the real version:

```bash
python scripts/run_phase1.py --config configs/phase1.yaml
```

(or run `notebooks/phase1_feasibility_colab.ipynb` on a GPU).

---

## Run metadata

- data source: _TO BE FILLED BY RUN_ | evidence valid: _TO BE FILLED BY RUN_
- depth prior: `depth-anything/Depth-Anything-V2-Small-hf` (frozen)
- device / torch: _TO BE FILLED BY RUN_
- train city: `JAX` | in-domain val: held-out `JAX` tiles | **held-out test city: `OMA`**
- tiles (train/val/test): _TO BE FILLED BY RUN_ | tile_size: _TO BE FILLED BY RUN_
- seeds / head params: _TO BE FILLED BY RUN_

## Cross-city generalization (held-out TEST city) — headline

_The number that matters: trained on the train city, evaluated on a city never
seen in training._

| method | MAE (m) | RMSE (m) | Pearson r | scenes |
|---|---|---|---|---|
| A · raw depth (scale-free) | —* | —* | _TBF_ |  |
| B · global affine | _TBF_ | _TBF_ | _TBF_ |  |
| C · learned fusion (hypothesis) | _TBF_ | _TBF_ | _TBF_ |  |
| oracle affine (upper bound) | _TBF_ | _TBF_ | _TBF_ |  |

(Report also breaks this down **buildings-only** and **non-building-only**, and
repeats an **in-domain** table for the held-out train-city tiles.)

`—*` = MAE/RMSE omitted for the scale-free raw-depth baseline (Pearson only).

## Reproducibility (learned head across seeds)

- cross-city MAE across seeds: _TO BE FILLED BY RUN_ (mean ± std)
- cross-city RMSE across seeds: _TO BE FILLED BY RUN_

## Reference points (context, NOT targets)

- Learned monocular height on DFC2019-class data: rough **2–4 m RMSE** band.
- Researched shadow-based reference: **~3.84 m RMSE**.
- External context only; not tuned to our data, not pass/fail thresholds.

## Phase-2 checkpoint decision

### Verdict: _TO BE FILLED BY RUN_

Computed from the measured metrics by `depthwizard/eval/decision.py` (GO /
MODIFY / ABANDON / INCONCLUSIVE), with a per-criterion pass/fail table. A human
makes the final call using these numbers **and** the error maps in
`runs/phase1/figures/`.

## What this run does and does NOT prove

- Measures whether a small learned RGB+depth fusion head beats trivial affine
  calibration **on a fully held-out city**, on DFC2019 (US WorldView-3) imagery.
- Does **not** establish accuracy on **Indian ISRO optical imagery** (domain
  shift — the top open risk), nor full **DSM** accuracy (nDSM only; DTM error is
  not included here).
