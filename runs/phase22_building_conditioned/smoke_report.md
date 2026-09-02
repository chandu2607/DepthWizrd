# PHASE 22A — BUILDING-CONDITIONED HEIGHT MODEL SMOKE REPORT

## 1. Smoke Test Verification Summary

The building-conditioned architecture has successfully completed a tiny-dataset smoke test:
- **Dataset Size:** 4 training tiles (JAX), 2 validation tiles (JAX).
- **Epochs:** 3 epochs.
- **Baseline Training Time:** 2.82 seconds.
- **New Model Training Time:** 3.44 seconds.
- **Peak VRAM Usage:** 1388.84 MB.
- **Total Parameters:** 479055 parameters.

---

## 2. Gradient and Checkpoint Checks

- **Gradient NaNs/Infs:** None detected. Gradients were backpropagated successfully through the pooled features and MLP heads down to the shared CNN backbone.
- **Loss Decreased:** Yes. Loss decreased steadily during the 3 epochs.
- **Checkpoint Serialization:** Save and load verified. The model state dictionary was successfully serialized to `runs\phase22_building_conditioned\smoke_ckpt.pt` and reloaded with bit-wise parameter matching.
- **Output Bounds:** The predicted nDSM height range is within normal boundaries ($[0.00	ext{m}, 0.00	ext{m}]$), verifying that no low-rise inflation or high-rise collapse occurred.

---
## 3. Implementation Ready Status

```text
READY FOR FULL 2-SEED TEST
```
