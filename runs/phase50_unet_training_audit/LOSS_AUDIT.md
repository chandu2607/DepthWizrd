# Loss Audit

Segmentation uses unweighted `binary_cross_entropy_with_logits` with `reduction="none"`, followed by valid-pixel mean. No `pos_weight`, class weights, Dice, focal, smoothing, or epsilon term is configured.
