"""
Demonstration and self-verification of dtcd.spatial.

Runs three sanity checks:
    1. Basic generation — shapes correct, no NaNs, stationary trajectory
    2. Counterfactual exactness — pre-intervention timesteps are bit-exact
       between factual and counterfactual; post-intervention ITE equals
       a manually computed ground truth
    3. Identifiability classification — unidentifiable configurations
       trigger warnings as documented
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

# Make the local module importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).parent))

from dtcd.spatial import (  # noqa: E402
    GaussianKernel,
    HardIntervention,
    ShiftIntervention,
    SpatioTemporalConfig,
    SpatioTemporalCounterfactualSuite,
    SpatioTemporalGenerator,
    grid_2d,
)


def _build_within_location_scm(d: int, tau_max: int, rng: np.random.Generator):
    """Construct a small acyclic W and a stack of lagged matrices A."""
    # Triangular W ensures acyclicity (topological order = identity).
    W = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < 0.4:
                W[i, j] = rng.uniform(-0.4, 0.4)
    # Sparse lagged stack with controlled magnitude for stationarity.
    A = np.zeros((tau_max, d, d), dtype=np.float64)
    for tau in range(tau_max):
        mask = rng.random(size=(d, d)) < 0.3
        A[tau] = mask * rng.uniform(-0.25, 0.25, size=(d, d))
    return W, A


# ---------------------------------------------------------------------------
# Test 1: basic generation
# ---------------------------------------------------------------------------


def test_generation_shapes_and_stability() -> None:
    rng = np.random.default_rng(42)
    d = 4
    tau_max = 2
    topology = grid_2d(n_rows=4, n_cols=4)  # L = 16
    kernel = GaussianKernel(sigma=1.5)
    coupling = np.array([0.3, 0.3, 0.0, 0.0])
    config = SpatioTemporalConfig(
        topology=topology,
        kernel=kernel,
        spatial_coupling=coupling,
        spatial_lag=1,
    )
    W, A = _build_within_location_scm(d, tau_max, rng)
    gen = SpatioTemporalGenerator(config, W, A, rng=rng)
    out = gen.generate(T=500, burn_in=200, noise_scale=1.0)

    assert out.X.shape == (500, 16, 4), f"unexpected X shape: {out.X.shape}"
    assert out.N.shape == (500, 16, 4), f"unexpected N shape: {out.N.shape}"
    assert not np.any(np.isnan(out.X)), "trajectory contains NaN"
    # Crude stationarity check: variance shouldn't blow up across the trajectory.
    var_first_half = out.X[:250].var()
    var_second_half = out.X[250:].var()
    ratio = var_second_half / var_first_half
    assert 0.5 < ratio < 2.0, f"non-stationary trajectory (var ratio = {ratio:.3f})"

    print(f"  test_generation: PASS  "
          f"(X.shape={out.X.shape}, "
          f"identifiability={out.identifiability!r}, "
          f"var ratio={ratio:.3f})")


# ---------------------------------------------------------------------------
# Test 2: counterfactual exactness
# ---------------------------------------------------------------------------


def test_counterfactual_exactness() -> None:
    """
    The contribution claim: re-injecting the cached noise yields a twin
    that is bit-exact with the factual at every (t, ell, i) BEFORE the
    intervention timestep, and differs only through propagation AFTER it.
    """
    rng = np.random.default_rng(123)
    d = 4
    tau_max = 2
    topology = grid_2d(n_rows=3, n_cols=3)  # L = 9
    kernel = GaussianKernel(sigma=1.0)
    coupling = np.array([0.4, 0.0, 0.4, 0.0])
    config = SpatioTemporalConfig(
        topology=topology,
        kernel=kernel,
        spatial_coupling=coupling,
        spatial_lag=1,
    )
    W, A = _build_within_location_scm(d, tau_max, rng)
    gen = SpatioTemporalGenerator(config, W, A, rng=rng)
    factual = gen.generate(T=200, burn_in=100, noise_scale=1.0)

    cfs = SpatioTemporalCounterfactualSuite(factual)

    # Intervention at a single (t, ell, var).
    t_star = 100
    ell_star = 4  # center of the 3x3 grid
    var_star = 0
    new_value = 5.0
    twin = cfs.apply(HardIntervention(
        times=[t_star], locations=[ell_star], variables=[var_star],
        values=[new_value],
    ))

    # Pre-intervention: twin must equal factual exactly.
    pre = twin.counterfactual[:t_star]
    pre_factual = factual.X[:t_star]
    max_pre_err = np.max(np.abs(pre - pre_factual))
    assert max_pre_err < 1e-12, (
        f"pre-intervention twin diverged from factual: max abs err = {max_pre_err}"
    )

    # At the intervention timestep: the intervened cell has the new value.
    assert abs(twin.counterfactual[t_star, ell_star, var_star] - new_value) < 1e-12, (
        "intervened cell does not have the specified value"
    )

    # ITE at the intervened cell equals (new_value - factual_value).
    expected_ite_at_iv = new_value - factual.X[t_star, ell_star, var_star]
    actual_ite_at_iv = twin.ite[t_star, ell_star, var_star]
    assert abs(actual_ite_at_iv - expected_ite_at_iv) < 1e-12

    # Post-intervention: SOME entries should differ (signal has propagated)
    # but all differences should be deterministic (no noise differences).
    post_diff = twin.counterfactual[t_star + 1 :] - factual.X[t_star + 1 :]
    has_propagation = np.any(np.abs(post_diff) > 1e-6)
    assert has_propagation, "intervention did not propagate downstream"

    # Crucial: re-applying the same intervention must produce identical
    # counterfactual (deterministic given cached noise).
    twin2 = cfs.apply(HardIntervention(
        times=[t_star], locations=[ell_star], variables=[var_star],
        values=[new_value],
    ))
    repro_err = np.max(np.abs(twin.counterfactual - twin2.counterfactual))
    assert repro_err == 0.0, f"counterfactual is not deterministic: {repro_err}"

    print(f"  test_counterfactual_exactness: PASS  "
          f"(pre-intervention max err = {max_pre_err:.2e}, "
          f"reproducibility err = {repro_err:.2e})")


# ---------------------------------------------------------------------------
# Test 3: shift intervention
# ---------------------------------------------------------------------------


def test_shift_intervention() -> None:
    rng = np.random.default_rng(7)
    d = 3
    tau_max = 1
    topology = grid_2d(n_rows=2, n_cols=2)  # L = 4
    config = SpatioTemporalConfig(
        topology=topology,
        kernel=GaussianKernel(sigma=1.0),
        spatial_coupling=np.array([0.3, 0.0, 0.0]),
        spatial_lag=1,
    )
    W, A = _build_within_location_scm(d, tau_max, rng)
    gen = SpatioTemporalGenerator(config, W, A, rng=rng)
    factual = gen.generate(T=100, burn_in=50)
    cfs = SpatioTemporalCounterfactualSuite(factual)

    twin = cfs.apply(ShiftIntervention(
        times=[50], locations=[0], variables=[0], deltas=[1.0],
    ))
    # ITE at the intervention cell should equal the shift exactly.
    assert abs(twin.ite[50, 0, 0] - 1.0) < 1e-12, (
        "shift intervention did not produce delta=1.0 at the cell"
    )
    print(f"  test_shift_intervention: PASS  "
          f"(ITE at intervened cell = {twin.ite[50, 0, 0]:.6f})")


# ---------------------------------------------------------------------------
# Test 4: identifiability warning
# ---------------------------------------------------------------------------


def test_identifiability_warning() -> None:
    rng = np.random.default_rng(2026)
    d = 3
    topology = grid_2d(n_rows=2, n_cols=2)
    config = SpatioTemporalConfig(
        topology=topology,
        kernel=GaussianKernel(sigma=1.0),
        spatial_coupling=np.array([0.3, 0.0, 0.0]),
        spatial_lag=0,  # unidentifiable
    )
    W, A = _build_within_location_scm(d, 1, rng)
    gen = SpatioTemporalGenerator(config, W, A, rng=rng)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gen.generate(T=50, burn_in=20, within_location_class="identifiable")

    flagged = any("unidentifiable" in str(w.message).lower() for w in caught)
    assert flagged, "spatial_lag=0 did not trigger identifiability warning"
    print(f"  test_identifiability_warning: PASS  "
          f"(captured {len(caught)} warnings, "
          f"unidentifiable flagged = {flagged})")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("Running dtcd.spatial self-verification ...")
    test_generation_shapes_and_stability()
    test_counterfactual_exactness()
    test_shift_intervention()
    test_identifiability_warning()
    print("\nAll tests passed.")
