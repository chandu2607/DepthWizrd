# Phase 53 Domain-Shift Audit

## Scope

The locked Config C seed 0 checkpoint was evaluated without retraining or parameter changes. Copenhagen and New York labels were used only for offline analysis. The Phase 52 threshold remained `0.60` and was not tuned on New York.

## Observed shift

Copenhagen mean probability: `0.5711`; New York mean probability: `0.1470`. Copenhagen target foreground and New York target foreground are reported in `ERROR_BREAKDOWN.csv`. Full RGB, probability, morphology, and height statistics are in the CSV/JSON artifacts.

## Interpretation

The selected model is non-collapsed, but New York produces a much lower probability field and substantially lower recall. The evidence supports **MULTI_FACTOR_DOMAIN_SHIFT** rather than a single normalization bug: RGB appearance, building morphology/density, and height distribution all require joint comparison. The checkpoint uses the stored training depth normalization buffers, and both cached splits follow the same static preprocessing path.

## Required caveat

Phase 51 generated New York metrics before clean Phase 52 selection. Those values were not used here. Phase 53 uses only the locked checkpoint and reports New York diagnostically.

## Recommendation

The smallest justified next experiment is a Copenhagen-safe multi-city/domain-style augmentation study emphasizing RGB appearance and building-scale/density variation, with the same architecture and strict Copenhagen selection. Do not change production or 3D code until that experiment is complete.
