# Phase 61 — Alignment Validation Plan

## Status

No real Indian benchmark tile has yet been accepted. This means the alignment work is a formal pilot plan, not a completed dataset validation.

## Required checks before any benchmark
after selection

- CRS compatibility between optical imagery and DEM/DSM
- affine transform checking
- pixel size and projection consistency
- overlap proportion between imagery and reference
- orientation sanity
- nodata / invalid pixel handling
- geographic bounds audit
- reference elevation plausibility
- no silent data modification

## Current outcome

No benchmark tile was selected and therefore no alignment computation was run.

## Required next step

The first accepted pilot tile must pass all checks before any model baseline or fine-tuning begins.
