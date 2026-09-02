# Phase 80 learning-dynamics forensic

## 1. Exact target space
Target units: meters in raw local relief; model target is z-score normalized local relief based on the train crop only.
target_normalized = (raw_local_relief - mean_train_target) / std_train_target

Train raw DEM min/max/mean/std: 4437.036621, 6121.617188, 5208.455078, 380.598511
Train raw local relief min/max/mean/std: -803.090332, 881.490234, -31.671883, 380.598511
Train model target min/max/mean/std: -2.026856, 2.399279, 0.000000, 1.000000
Validation raw local relief min/max/mean/std: -1327.232178, 1185.815430, -53.076824, 561.360291
Validation model target min/max/mean/std: -3.404008, 3.198876, -0.056240, 1.474941
Inverse transform max reconstruction error: 0.000061035156
Inverse transform mean reconstruction error: 0.000002872686

## 2. Crop distribution audit
Train valid pixels: 262144; valid fraction: 1.000000; target mean/std/min/max: -31.671883/380.598511/-803.090332/881.490234
Validation valid pixels: 262144; valid fraction: 1.000000; target mean/std/min/max: -53.076824/561.360291/-1327.232178/1185.815430
Across crops summary: valid_pixel_count={'minimum': 262144.0, 'maximum': 262144.0, 'mean': 262144.0, 'median': 262144.0}; valid_fraction={'minimum': 1.0, 'maximum': 1.0, 'mean': 1.0, 'median': 1.0}; target_mean={'minimum': -53.07682418823242, 'maximum': -31.67188262939453, 'mean': -42.37435340881348, 'median': -42.37435340881348}; target_std={'minimum': 380.5985107421875, 'maximum': 561.3602905273438, 'mean': 470.9794006347656, 'median': 470.9794006347656}; target_min={'minimum': -1327.232177734375, 'maximum': -803.09033203125, 'mean': -1065.1612548828125, 'median': -1065.1612548828125}; target_max={'minimum': 881.490234375, 'maximum': 1185.8154296875, 'mean': 1033.65283203125, 'median': 1033.65283203125}

## 3. Optimization-step audit
train_crops=1; batch_size=1; number_of_batches=1; optimizer_step_calls=1; gradient_accumulation_steps=1; effective_optimization_steps_per_epoch=1

## 4. Loss sanity
loss=0.443539023399; prediction mean/std=-0.032225/0.002123; target mean/std=0.000000/1.000000
loss_if_prediction_zero=0.838048815727
loss_if_prediction_target=0.000000000000
loss_if_prediction_target_mean=0.838048815727

## 5. Mask / loss-formula audit
exact_loss_expression=loss = ( SmoothL1Loss(pred * mask, target * mask) * mask ).sum() / (mask.sum() + 1e-6)
valid_pixel_count=262144; numerator=116271.093750000000; denominator=262144.000000000000; implementation_loss=0.443539023399; reference_loss=0.839534759521; match_error=3.959957361221e-01

## 6. Gradient flow
any_nan=False; any_inf=False; mean_gradient_norm=0.009309; earliest_nonzero=e1.0.weight; largest_norm=d1.0.weight

## 7. Parameter update check
loss_before_step=0.445725500584; loss_after_step=0.440685302019
parameter change summary: [
  {
    "name": "e1.0.weight",
    "abs_parameter_change_mean": 0.0004816426371689886,
    "relative_parameter_change_mean": 0.0058817836569372926
  },
  {
    "name": "e2.0.weight",
    "abs_parameter_change_mean": 0.0004967778222635388,
    "relative_parameter_change_mean": 0.016776917274171802
  },
  {
    "name": "bottleneck.0.weight",
    "abs_parameter_change_mean": 0.00038535287603735924,
    "relative_parameter_change_mean": 0.018456068836407498
  },
  {
    "name": "head.weight",
    "abs_parameter_change_mean": 0.0008417353965342045,
    "relative_parameter_change_mean": 0.008687789091539916
  }
]

## 8. Output response before/after one step
before min/max/mean/std: -0.059189/-0.041051/-0.045939/0.002963
after min/max/mean/std: -0.058562/-0.032514/-0.044421/0.007769
output_change_mean=0.001518; output_change_std=0.010361; output_change_max=0.025423

## 9. Output activation check
final_layer=Conv2d(32, 1, kernel_size=(1,1)); activation_after_final_layer=none (linear output, no sigmoid/tanh/clamp); output_range=unbounded real-valued output; fresh_output_range={'min': 0.18626655638217926, 'max': 0.209677591919899, 'mean': 0.19697746634483337, 'std': 0.000661789090372622}

## 10. Baseline comparison
zero_baseline={'mae': 1.2157599925994873, 'rmse': 1.4760127067565918, 'pearson': nan, 'mean_bias': 0.056240230798721313, 'prediction_mean': 0.0, 'prediction_std': 0.0, 'target_std': 1.4749408960342407, 'n': 262144}
mean_training_target_baseline={'mae': 1.2190628051757812, 'rmse': 1.4749408960342407, 'pearson': nan, 'mean_bias': 0.0, 'prediction_mean': -0.05624021589756012, 'prediction_std': 1.4901161193847656e-08, 'target_std': 1.4749408960342407, 'n': 262144}
median_training_target_baseline={'mae': 1.2139289379119873, 'rmse': 1.4815189838409424, 'pearson': nan, 'mean_bias': 0.13945558667182922, 'prediction_mean': 0.08321535587310791, 'prediction_std': 0.0, 'target_std': 1.4749408960342407, 'n': 262144}
stored_phase79_model={'mae': 1.2236868143081665, 'rmse': 1.4768003225326538, 'pearson': -0.057983660671387625, 'mean_bias': -0.057679273188114166, 'prediction_mean': -0.11391951888799667, 'prediction_std': 0.011821982450783253, 'target_std': 1.4749408960342407, 'n': 262144}

Diagnosis: LOSS_OR_MASKING_FAULT

LOSS_OR_MASKING_FAULT
