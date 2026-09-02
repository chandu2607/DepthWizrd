# Phase 60 — Failure Analysis

## Status

No real Indian mountainous benchmark was selected or evaluated in this phase. Therefore, the current failure analysis is a formal evidence gate, not a model diagnosis.

## Constrained evidence

The project has established the following from repository inspection and prior audit work:

- Depth Anything V2 is a frozen relative-depth prior, not a metric height sensor.
- The current calibration engine converts relative depth into either relative surfaces or metric outputs using heuristics and optional reference elevation.
- The system is building-heavy and urban-geometry oriented rather than terrain-native.
- The repo contains no verified Indian mountainous benchmark tile for inference.

## Dominant failure mode currently supported by evidence

The currently supported dominant failure mode is:

`NO_VALID_BENCHMARK`

This is stronger than guessing at a model failure. It means we do not yet have the actual terrain data needed to decide whether the failure is terrain domain shift, scale mismatch, building confusion, or a mixed failure.

## What is not supported yet

We do not have evidence for:

- building detection failure on Indian slopes,
- terrain estimation failure on Indian mountains,
- metric-scale failure in Himalayan terrain,
- relative-depth domain failure under steep slopes,
- multi-factor failure in the Indian disaster context.

Those remain hypotheses until a benchmark is actually selected and evaluated.

## Scientific conclusion

This phase does not claim a model-level Indian failure. It asserts that the current evidence is insufficient to evaluate the model on Indian mountainous terrain.

That is the correct evidence-first position.
