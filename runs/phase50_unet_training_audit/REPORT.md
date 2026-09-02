# Phase 50 U-Net Training / Target / Loss / Checkpoint Audit

## Verdict

**LOSS_OR_CLASS_BALANCE_FAULT**

## Target pipeline

Training code loads the DSM raster, constructs `mask_bldg = (gt > 2.0)`, resizes the DSM-derived target with `cv2.INTER_LINEAR`, then constructs the loss target as `gt_fp = (yt > 2.0).float()`. The model therefore receives a thresholded target after bilinear interpolation, rather than a nearest-neighbor-resized binary mask.

Complete-split target statistics are in `target_statistics.csv`. Mean training foreground prevalence is **29.1445%**. The unweighted-BCE constant-positive logit baseline is `log(p/(1-p)) = -0.8884`.

## Loss audit

The segmentation term is `torch.nn.functional.binary_cross_entropy_with_logits(mask_logits, gt_fp, reduction="none")`, averaged over valid pixels. There is no `pos_weight`, no class weighting, no Dice term, no focal term, and no smoothing. The total training loss is segmentation loss plus `0.5 * regime_loss` plus `0.1 * height_loss`.

At the source level, the segmentation target and unweighted loss permit a globally positive solution if the learned image/depth features do not separate classes. The target resize is also inconsistent with the expected nearest-neighbor mask resize.

## Checkpoint audit

Config D SHA256: `93ebf2f2da89f57125ba954ad1dfdc7c9abbf3f8330f62bff5d1f2edd3d7ea1e`. It loads with zero missing and unexpected keys. The saved checkpoint contains model `state_dict` only; optimizer state, epoch, and validation metric are absent. Phase 43 selects the highest Copenhagen IoU in memory, then saves that model, but the checkpoint itself cannot independently prove which seed/epoch produced it. See `checkpoint_provenance.csv`.

## Prediction forensic result

Config D is elevated across all audited splits. The exact pooled statistics are in `split_prediction_statistics.csv`. The NYC direct result remains 100% above probability 0.5. The training control is also globally positive, so the failure is not NYC-only domain shift.

## Phase 29 comparison

The Config A/Phase 29 baseline is also elevated on the NYC tile. Therefore Config D augmentation did not introduce the only failure; both models share the same broad training/output pathology.

## Controlled experiment

The tiny synthetic experiment uses known RGB masks and the same `BCEWithLogits` loss. Its loss decreases and logits move across zero, demonstrating that the loss/model plumbing can learn a separable toy mask. This does not exonerate the real target/loss distribution; it narrows the issue toward the real target prevalence, target resize, class balance, and checkpoint/training selection behavior.

## Required interpretation

The evidence supports **LOSS_OR_CLASS_BALANCE_FAULT**, with a confirmed target-resize defect and unweighted segmentation loss as concrete risk factors. This is not a threshold problem, calibration problem, DSM problem, or 3D viewer problem.

## Smallest corrective change for review

Before retraining a production model, run one controlled ablation on the existing Phase 43 training code: resize the binary mask with nearest-neighbor and report target prevalence, per-pixel loss contributions, and validation predictions with an explicitly measured class-balance treatment. Preserve the current checkpoint and compare against it. Do not change thresholds or downstream geometry.
