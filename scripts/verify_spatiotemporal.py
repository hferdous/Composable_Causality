import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath("src"))

from dtcd.spatial import (
    SpatioTemporalGenerator, SpatioTemporalConfig,
    grid_2d, GaussianKernel,
    SpatioTemporalCounterfactualSuite, HardIntervention,
)

d = 2
topology = grid_2d(n_rows=4, n_cols=4)
kernel = GaussianKernel(sigma=1.5, normalize=True)
config = SpatioTemporalConfig(
    topology=topology,
    kernel=kernel,
    spatial_coupling=np.array([0.3, 0.3]),
    spatial_lag=1,
)

rng = np.random.default_rng(42)
W = np.zeros((d, d))
A = np.zeros((2, d, d))

gen = SpatioTemporalGenerator(config=config, W=W, A=A, rng=rng)
dataset = gen.generate(T=200)

print("=" * 60)
print("SPATIO-TEMPORAL VERIFICATION")
print("=" * 60)
print(f"\n[Test 1] Generated trajectory shape: {dataset.X.shape}")
print(f"  Expected (T, L, d) = (200, 16, 2)")
print(f"  Match: {dataset.X.shape == (200, 16, 2)}")
print(f"  Mean: {dataset.X.mean():.4f}, Std: {dataset.X.std():.4f}")
print(f"  Any NaNs? {np.isnan(dataset.X).any()}")

print(f"\n[Test 2] Cached noise shape")
if hasattr(dataset, 'N') and dataset.N is not None:
    print(f"  Noise shape: {dataset.N.shape}")
    print(f"  Match (T, L, d): {dataset.N.shape == (200, 16, 2)}")
else:
    print("  No cached noise attribute found (check API)")

print(f"\n[Test 3] Spatial kernel matrix")
K = kernel.build(topology)
print(f"  Kernel shape: {K.shape}")
print(f"  Diagonal is zero (no self-influence): {np.all(np.diag(K) == 0)}")
print(f"  Row sums (should be ~1.0 if normalized): {K.sum(axis=1)[:3]}")

print(f"\n[Test 4] Counterfactual twin")
cf_suite = SpatioTemporalCounterfactualSuite(dataset)
intervention = HardIntervention(
    times=[50],
    locations=[5],
    variables=[0],
    values=[10.0],
)
cf_pair = cf_suite.apply(intervention)

X_factual = cf_pair.factual
X_counter = cf_pair.counterfactual

# Pre-intervention: should be identical
pre_diff = np.abs(X_factual[:50] - X_counter[:50]).max()
print(f"  Pre-intervention max diff (should be 0): {pre_diff:.2e}")

print(f"  Intervened value at (t=50, loc=5, var=0): "
      f"{X_counter[50, 5, 0]:.4f} (expected 10.0)")

post_diff = np.abs(X_factual[51:] - X_counter[51:]).sum()
print(f"  Post-intervention total absolute diff: {post_diff:.4f}")
print(f"  Effect propagated (post_diff > 0): {post_diff > 0}")

print("\n" + "=" * 60)
print("SPATIO-TEMPORAL: ALL TESTS COMPLETED")
print("=" * 60)
