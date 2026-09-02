# DFC2023 Track 2 — Dataset Validation Probe

**Date:** 2026-08-27 · **Type:** dataset validation (NOT a scientific experiment — not appended to canonical history)
**Verdict:** ✅ **APPROVED WITH CAVEATS** (metadata) · **File-probe status:** 🚧 **BLOCKED — human-provided sample NOT FOUND (2026-08-27 attempt; see §9)**

> One-line: DFC2023 Track 2 is the right *kind* of dataset (real RGB + measured nDSM height + COCO building masks, 17 cities/6 continents, 512×512 tiles — near-drop-in for our nDSM pipeline), but the actual bytes are behind a Codalab + geovisearth registration gate with **no open mirror anywhere**, so the deciding file-level checks (in-data units, tall-building distribution, RGB↔height alignment, per-city public-label counts) **could not be run here** and must be done by the human on a tiny sample after registering.

---

## 1. What was verified (primary sources)

| Requirement | Verified? | Evidence |
|---|---|---|
| Identity / track | ✅ | 2023 IEEE GRSS DFC, **Track 2 = Building Height Estimation** ("reconstruct building heights and extract building footprints") — baseline repo + official page |
| Cities / diversity | ✅ | **17 cities, 6 continents** (baseline repo: "seventeen cities worldwide across six continents") |
| Real optical RGB | ✅ | `rgb/` present; SuperView-1 0.5 m + Gaofen-2 0.8 m (official page, Exp-8) |
| Height target type | ✅ | **nDSM** (normalized DSM), per-pixel elevation TIFF in `height/` |
| Height GT is *measured*, not predicted | ✅ (low circular risk) | nDSM derived from stereo (Gaofen-7/WorldView per official page, Exp-8) — **not** a model output → non-circular; reconfirm in data-description doc |
| Building masks | ✅ | **MS COCO instance segmentation JSON** (`buildings_only_{train,val,test}.json`) → supports building/tall MAE-RMSE eval |
| Tile size | ✅ | **512×512** for all images (matches our current `tile_size=512`) |
| Directory structure | ✅ | `train/ val/ test/` each with `rgb/ sar/ height/` + `annotations/` |
| Access mechanism | ✅ | **Gated**: Codalab account + register at `https://dfc.geovisearth.com/en/register` (Participate tab) |
| Open mirror exists? | ✅ (answered: **NO**) | HF `search=DFC2023` → empty; HF `search=data fusion contest` → empty; Zenodo → only DFC2018/2019/2022 + unrelated, **no DFC2023**; baseline repo bundles no samples |

## 2. What could NOT be verified (requires the actual files / human-enabled access)

These are the **deciding file-probe questions** — blocked because there is no open source to pull a probe from and the gate must not be bypassed (§17):

- **Height *units* confirmed in-data** — baseline README says only "pixel by pixel elevation value" (no units). nDSM convention ⇒ metres, but §12 forbids inferring units from magnitude; the data-description doc / a real tile must confirm.
- **Tall-building content** — fractions of pixels >15 / >20 / >30 / >40 m (the exact regime DepthWizard currently fails). Unknown until a sample is read.
- **RGB↔height pixel alignment** — dimensions, offset, orientation, scaling, nodata (§10, the main point of the probe).
- **Exact count of cities with *public* metric-height labels** — the contest withholds val/test references; the README gives no per-city counts or city names. Could be far fewer than 17 usable for training labels.
- **Verified dataset size (GB)** — no file listing is reachable without registration.
- **Exact license text** — contest terms apply via registration; not re-verified this session.

## 3. City coverage (§6) — honest granularity

Per-city Yes/No cannot be truthfully filled without the gated data, so no fabricated 17-row table is given. What is verified:

| Scope | RGB | nDSM height | Building masks | Publicly-usable training labels |
|---|---|---|---|---|
| DFC2023 Track 2 (17 cities, 6 continents; names not captured) | ✅ (contest structure) | ✅ (contest structure) | ✅ COCO | ⚠️ **UNVERIFIED** — val/test refs withheld; train-partition per-city count unknown |

**Feasibility of 4-train / 1-val / 1-test (§7):** *plausible* given 17 cities, but **contingent** on how many carry public training labels — the single most important thing the human-enabled probe must confirm.

## 4. Compatibility with the existing DepthWizard pipeline (§20)

- **Strong.** 512×512 tiles = our current tile size; target is nDSM height (same concept as our DFC2019 mirror). RGB-only training is feasible by **ignoring the SAR channel** (we drive depth from RGB via DA-V2).
- **Adapter needed (NOT built):** a DFC2023 loader for the `train/val/test → rgb/height` layout + COCO→building-mask decode, analogous to the existing `hf_blocks.py`. Keep **DA-V2 frozen, C_log1p, loss, log1p transform, city-held-out eval all unchanged**.

## 5. Backup — HighBuild-1M (§22, six critical facts)

| Fact | Finding |
|---|---|
| Cities | 26 groups / 12 countries / 6 continents |
| Height units | **not stated** |
| GT source (LiDAR/stereo/predicted) | **not stated** → circular-pipeline risk unresolved |
| Real vs synthetic | **not stated** |
| GSD | **not stated** |
| License | `other` (unspecified) |
| Paper / code | **none** (arXiv 0, GitHub 0) |

→ **Backup only.** Openly downloadable, best-designed-on-paper (pre-defined cross-city/cross-country splits), but its unverified provenance is disqualifying for a *scale* study until the authors confirm real imagery + measured metric GT. **M4Heights (§23):** not revisited (gated 401, Sentinel-multimodal); no evidence it beats DFC2023.

## 6. Recommendation & next download (§24, §25)

**Classification: APPROVED WITH CAVEATS.** Not "approved for full download" (deciding file checks unrun); not "rejected" (nothing verified disqualifies it).

**Do NOT full-download.** After you register (Codalab + `dfc.geovisearth.com/en/register`), the smallest decision-changing pull is:

1. **Training partition only**, modalities **`rgb/` + `height/` + `annotations/buildings_only_train.json`** — **skip `sar/`** (unused).
2. **First probe:** 2–3 cities × a handful of 512×512 tiles → run the local checks (units-in-data, height min/median/P90/P95/P99/max, tall fractions >15/20/30/40 m, RGB↔height alignment + a visualization).
3. **Then STOP again** for a go/no-go on the full training download.

**Size:** per-tile ≈ 1.3–1.8 MB (512² RGB + float32 height, no SAR) — **total is unknown until the post-registration file listing is seen; do not assume.** Re-check RTX-3050 (~4 GB VRAM) / ~88 GB free before any bulk pull.

## 7. Future experiment (§26 — designed, NOT run)

Train ≥4 DFC2023 cities (≥3 continents) → val 1 unseen city → test 1 unseen city. Arms: **JAX-only C_log1p baseline** vs **multi-city**. Fixed: DA-V2, C_log1p, log1p transform, loss, optimizer, resolution, seeds {0,1}, city-held-out eval. **Only variable = training-city diversity.** Success (§27) = meaningful gains in unseen-city building MAE **and** tall-building MAE **and** RMSE **and** high-height bias **and** cross-city stability — *without* mean-shift/false-positive/ground-exploit/edge-artifact/overfit shortcuts.

## 8. Final answers (§29)

- **What we verified:** identity, 17 cities/6 continents, real RGB, nDSM height target, measured (non-circular) GT provenance, COCO building masks, 512×512 tiles, folder structure, gated access, and that **no open mirror exists**.
- **What we could not verify:** in-data height units, tall-building distribution, RGB↔height alignment, exact publicly-labelled city count, verified GB size, exact license text — all require the gated files.
- **Suitable?** Yes, in principle — it is the closest verified relative of our DFC2019 nDSM pipeline with real multi-city metric height.
- **Provides the missing metric-height supervision?** Very likely yes (measured nDSM across 17 cities) — pending in-data unit + distribution confirmation.
- **Enough city diversity?** Yes at the contest level (17/6); usable-label diversity to be confirmed.
- **Did the small probe pass?** The **metadata/provenance probe passed; the file probe is BLOCKED** (gated, no open mirror) — it did not run here.
- **What to download next:** the training `rgb/`+`height/`+`buildings_only_train.json`, 2–3 cities × a few tiles first (skip SAR).
- **Estimated download size:** unknown until the file listing is seen; per-tile ≈1.3–1.8 MB.
- **What the future experiment tests:** whether broader multi-city training teaches one model to convert DA-V2 relative depth into reliable **metric** height on **unseen** cities.

## 9. File-probe attempt — 2026-08-27 — 🚧 BLOCKED: sample not found

The follow-up task stated a small real DFC2023 sample had been provided and instructed a file-level probe. **No such sample is present at any location I can access.** Search performed (reproducible):

- **Project tree** `DepthWizard/**` for `*.tif/*.tiff/*.png/*.jp2/*.jpg`, `*.json`, and any `*dfc2023*` / `track2` / `rgb` / `height` / `sar` directory → found only the existing **DFC2019** mirror (`data/dfc2019/**`, ~999 `.tif`) + prior-phase PNG **figures** under `runs/*/figures/`. No DFC2023 rasters, no COCO `buildings_only_*.json`.
- **`C:/Users/chand/Downloads`** for `*.tif*`, `*.json`, `*DFC2023*`, and archives → none (newest item is a `.md`, 2026-08-27 17:53).
- **`C:/Users/chand/OneDrive/Desktop`** for a new sample folder → none (only the existing project folders).
- **`Downloads/04 Archives/*.zip`** listings (no extraction): `archive.zip` = a lone 2020 `annotations.json`; `Farm (2).zip` = a Flask web app — both unrelated.

**Consequence:** the file-level checks (§4–§14 of the follow-up task: RGB inspection, height units/semantics, RGB↔height alignment, COCO categories, tall-building fractions, tile size, GSD, multi-city label availability) **could not be run** — there are no bytes to inspect. **No numbers were fabricated; the task forbids self-downloading, so I did not fetch the data.**

**What the human must do to unblock:** place the sample where I can read it — e.g. `DepthWizard/data/dfc2023/<city>/{rgb,height,annotations}/…` — **or** tell me its exact path. If it lives on another drive or outside my working tree, grant access to that folder. A minimal useful sample = **2–3 cities × a few tiles each of RGB + height TIFFs + the matching `buildings_only_*.json`** (SAR not needed).

---

### Final decision (§23): 🚧 **BLOCKED** — sample not found; the file-probe is not yet possible.
The earlier **metadata/provenance** verdict (APPROVED WITH CAVEATS) is unchanged, but the **file-level probe cannot proceed** until the actual sample is provided.

**STOP — no download, no training, no model changes performed. Awaiting the sample path from the human.**
