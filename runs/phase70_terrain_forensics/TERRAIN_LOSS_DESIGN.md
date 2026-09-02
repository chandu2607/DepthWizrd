# Terrain Loss Design

We compare:

A = L1 / SmoothL1

B = L1 + small gradient consistency term

## A: L1 / SmoothL1
For prediction p and target y:

L1 = |p - y|
SmoothL1(x) = 0.5 x^2 if |x| < 1 else |x| - 0.5

This is robust and simple for terrain regression.

## B: L1 + gradient consistency
L_grad = |
abla_x p - 
abla_x y| + |
abla_y p - 
abla_y y|

Total = L1 + lambda * L_grad

with a small lambda (e.g. 0.01 or 0.05) so the model learns terrain structure without large instability.

## Recommendation
Start with SmoothL1 only for the first terrain pilot, then add only a very small gradient term if the training is numerically stable.
