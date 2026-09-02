# Phase 63 — Alignment Validation Plan

## Status

No real Indian pilot tile has yet been physically acquired. Therefore the alignment validation is not complete.

## Required alignment verification

Before a real benchmark can be accepted, the following must be verified for the acquired optical image and elevation reference:

- CRS compatibility
- transform / affine match
- pixel-size compatibility
- bounds overlap
- dimension compatibility
- orientation sanity
- nodata handling
- valid-data percentage
- resampling method trace

## Rules

- Use the optical grid as the default target if the reference is safely reprojected/resampled.
- Use bilinear / cubic interpolation for continuous elevation values only where justified.
- Use nearest-neighbor for masks and categorical fields.
- Never silently alter source files.
- Only derived aligned copies can be written to `ALIGNED_DATA`.

## Current outcome

No actual aligned files exist yet; the phase remains at the acquisition gate.
