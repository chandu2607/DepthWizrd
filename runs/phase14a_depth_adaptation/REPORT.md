# PHASE 14A SMOKE TEST - DECODER-ONLY ADAPTATION

This report details the feasibility and mechanics of unfreezing the Depth Anything V2 decoder (DPT neck and head) to fine-tune it directly on metric height data, while keeping the ViT encoder backbone frozen.

## 1. Modules & Parameter Counts
- **Exact Trainable Modules:** `['head', 'neck']`
- **Exact Frozen Modules:** `['backbone']`
- **Frozen Parameter Count:** 22,056,576 (~22M)
- **Trainable Parameter Count:** 2,728,513 (~2.7M)
- **Total Parameter Count:** 24,785,089 (~24.7M)

## 2. Gradient Verification
- **Requires Grad:** 
  - Backbone parameters: `requires_grad=False` (Verified)
  - Neck/Head parameters: `requires_grad=True` (Verified)
- **Gradient Computation:** 
  - After `loss.backward()`, the backbone gradients remained `None`.
  - The neck/head gradients populated correctly (mean absolute gradient: `0.000038`).
  - The optimizer successfully updated the weights.

## 3. Loss Behavior
The model was subjected to a 5-step overfit test using a batch size of 2, standard masked-L1 loss, and the `log1p` target transform. The loss strictly monotonically decreased without any `NaN/Inf` errors:
- Step 1: `1.3950`
- Step 2: `1.2110`
- Step 3: `1.1554`
- Step 4: `1.0269`
- Step 5: `0.9432`

## 4. System & Memory Constraints
- **Peak VRAM:** `596.7 MB` (measured using `torch.cuda.max_memory_allocated()`).
- **Memory Safety:** The peak memory is exceptionally safe. A batch size of 2 utilizes less than 15% of the 4GB budget on an RTX 3050 Laptop GPU. Even larger batch sizes will comfortably fit.

## 5. Checkpoint Verification
- **Save/Load:** `torch.save` and `load_state_dict` successfully wrote and restored the hybrid frozen/unfrozen model state.

## 6. Implementation Issues
- **Output Resizing:** The raw output tensor from `DepthAnythingForDepthEstimation` (`out.predicted_depth`) is slightly smaller than the 518x518 input (e.g. downsampled based on the patch size constraint). A standard `F.interpolate` with bilinear mode was added in the forward pass immediately before the loss calculation to exactly match the ground truth tensor shape. This is a normal requirement for segmentation/depth transformers.

## 7. Conclusion: Technical Feasibility
**Decoder-only adaptation is technically feasible.**
The code path correctly prevents backward passes through the massive ViT backbone, drastically cutting VRAM requirements while still allowing full differentiable updates to the ~2.7M parameters in the DPT neck and head. 

We can now proceed to test if adjusting these 2.7M parameters is sufficient to extract the required metric scale from the frozen representations.
