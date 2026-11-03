"""
Spatio-temporal counterfactual twin generation.

Generalizes dtcd's CounterfactualSuite from (T, d) to (T, L, d). The core
mechanism is unchanged: cache the noise tensor N during the factual
rollout, then re-inject the same N during the interventional rollout
while overwriting the intervened entries. The intervention can target any
(location, variable, time) triple, and the resulting twin is exact at
every (t, ell, i) up to floating-point precision.

Use cases:
    - Spatial CATE evaluation: how does an intervention at one location
      propagate to neighboring locations through the spatial kernel?
    - Spillover quantification: ground-truth direct vs. indirect effects
      under the SUTVA-violating regime that spatial coupling represents.
    - Robustness: PEHE on spatially correlated outcomes is a strictly
      harder target than IID PEHE; this module exposes that target.

Causal assumption:
    - The intervention does not change W, A, K, or s. Soft interventions
      that modify mechanisms (rather than fixing values) require a
      different module not implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np

from .generator import SpatioTemporalDataset


# ---------------------------------------------------------------------------
# Intervention specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HardIntervention:
    """
    do(X[t_i, ell_i, var_i] = value) for one or more (t, ell, var) cells.

    Parameters
    ----------
    times : sequence of int
        Timesteps at which to fix the intervened variables. Indices are
        relative to the post-burn-in trajectory.
    locations : sequence of int
        Locations at which the intervention applies, indexed 0..L-1.
    variables : sequence of int
        Variable indices at the intervened locations.
    values : sequence of float
        The fixed values. Must have the same length as `times`,
        `locations`, and `variables` (one value per (t, ell, var) tuple).

    All four sequences are zipped: the i-th intervention is
    do(X[times[i], locations[i], variables[i]] = values[i]).
    """

    times: Sequence[int]
    locations: Sequence[int]
    variables: Sequence[int]
    values: Sequence[float]

    def __post_init__(self) -> None:
        n = len(self.times)
        if not (len(self.locations) == len(self.variables) == len(self.values) == n):
            raise ValueError(
                "times, locations, variables, and values must all have "
                "the same length."
            )
        if n == 0:
            raise ValueError("HardIntervention requires at least one cell.")


@dataclass(frozen=True)
class ShiftIntervention:
    """
    do(X[t_i, ell_i, var_i] = X[t_i, ell_i, var_i] + delta_i).

    A shift intervention adds a fixed offset rather than overwriting
    the value, preserving the noise contribution at the intervened cell.
    Equivalent to a hard intervention with `values = factual + delta`,
    but more convenient when the user wants to express a "what if this
    was 0.5 standard deviations higher" question.
    """

    times: Sequence[int]
    locations: Sequence[int]
    variables: Sequence[int]
    deltas: Sequence[float]

    def __post_init__(self) -> None:
        n = len(self.times)
        if not (len(self.locations) == len(self.variables) == len(self.deltas) == n):
            raise ValueError(
                "times, locations, variables, and deltas must all have "
                "the same length."
            )
        if n == 0:
            raise ValueError("ShiftIntervention requires at least one cell.")


Intervention = Union[HardIntervention, ShiftIntervention]


# ---------------------------------------------------------------------------
# Counterfactual output
# ---------------------------------------------------------------------------


@dataclass
class CounterfactualPair:
    """
    A factual / counterfactual twin pair.

    Attributes
    ----------
    factual : np.ndarray, shape (T, L, d)
        The original trajectory (passed in unchanged).
    counterfactual : np.ndarray, shape (T, L, d)
        The trajectory under the specified intervention.
    ite : np.ndarray, shape (T, L, d)
        Individual treatment effect: counterfactual - factual. Exact up
        to floating-point precision because both rollouts use identical
        cached noise.
    """

    factual: np.ndarray
    counterfactual: np.ndarray
    ite: np.ndarray


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class SpatioTemporalCounterfactualSuite:
    """
    Generate exact counterfactual twins for a precomputed factual
    `SpatioTemporalDataset`.

    Usage:
        gen = SpatioTemporalGenerator(config, W, A, rng=rng)
        factual = gen.generate(T=500)
        cfs = SpatioTemporalCounterfactualSuite(factual)

        twin = cfs.apply(HardIntervention(
            times=[100], locations=[5], variables=[0], values=[2.0]
        ))
        # twin.ite is exact ground-truth effect of do(X[100, 5, 0] = 2.0)
        # at every (t >= 100, ell, i) downstream.

    The suite holds W, A, K, and the cached noise N from the factual
    dataset, so applying many interventions to the same factual scenario
    is cheap (no re-sampling).
    """

    def __init__(self, factual: SpatioTemporalDataset) -> None:
        self._factual = factual
        self._W = factual.W
        self._A = factual.A
        self._K = factual.K
        self._s = factual.spatial_coupling
        self._spatial_lag = factual.spatial_lag
        self._tau_max = factual.A.shape[0]
        self._d = factual.X.shape[2]

        # Reduced-form projection.
        self._M = np.linalg.inv(np.eye(self._d) - self._W.T)

    def apply(self, intervention: Intervention) -> CounterfactualPair:
        """
        Generate the counterfactual twin under the supplied intervention.

        Parameters
        ----------
        intervention : HardIntervention or ShiftIntervention

        Returns
        -------
        CounterfactualPair
        """
        if isinstance(intervention, HardIntervention):
            return self._apply_hard(intervention)
        if isinstance(intervention, ShiftIntervention):
            return self._apply_shift(intervention)
        raise TypeError(f"Unsupported intervention type: {type(intervention)!r}")

    # ------------------------------------------------------------------
    # Internal: shared rollout with intervention hook
    # ------------------------------------------------------------------

    def _rollout(
        self,
        intervention_hook,  # callable: (X_cf, t_buffer, max_lag_used) -> overrides at t
        earliest_intervention_t: int,
    ) -> np.ndarray:
        """
        Run an interventional rollout using the cached noise.

        Strategy: pre-intervention, the counterfactual must equal the factual
        exactly (same noise, same dynamics, no intervention applied yet), so
        we seed the buffer with the factual trajectory + saved pre-history
        and only recompute from `earliest_intervention_t` onward. This makes
        pre-intervention exactness a property of construction, not numerical
        luck.

        The `intervention_hook` is called immediately after computing the
        natural value at each post-intervention timestep, with the chance to
        override any (ell, var) entries.

        Returns the counterfactual trajectory of shape (T, L, d).
        """
        T, L, d = self._factual.X.shape
        N = self._factual.N
        spatial_lag = self._spatial_lag
        tau_max = self._tau_max
        max_lag_used = max(tau_max, spatial_lag)

        # Buffer with pre-history at the front, then T post-burn-in steps.
        X_cf = np.zeros((T + max_lag_used, L, d), dtype=np.float64)
        X_cf[:max_lag_used] = self._factual.prehistory
        X_cf[max_lag_used:] = self._factual.X

        # Recompute only from the earliest intervention timestep onward.
        start = max_lag_used + earliest_intervention_t
        end = max_lag_used + T

        for t in range(start, end):
            lagged = np.zeros((L, d), dtype=np.float64)
            for tau in range(1, tau_max + 1):
                lagged += X_cf[t - tau] @ self._A[tau - 1]

            if spatial_lag > 0:
                spatial = (self._K.T @ X_cf[t - spatial_lag]) * self._s[None, :]
            else:
                spatial = np.zeros((L, d), dtype=np.float64)

            n_idx = t - max_lag_used
            X_cf[t] = (lagged + spatial + N[n_idx]) @ self._M.T

            # Apply intervention overrides for this timestep.
            intervention_hook(X_cf, t, max_lag_used)

        return X_cf[max_lag_used:]

    # ------------------------------------------------------------------
    # Internal: hard intervention
    # ------------------------------------------------------------------

    def _apply_hard(self, iv: HardIntervention) -> CounterfactualPair:
        # Pre-organize overrides by post-burn-in timestep.
        T = self._factual.X.shape[0]
        overrides_by_t: dict[int, list[tuple[int, int, float]]] = {}
        for ti, ell, var, val in zip(
            iv.times, iv.locations, iv.variables, iv.values
        ):
            if not (0 <= ti < T):
                raise ValueError(
                    f"Intervention time {ti} out of range [0, {T})."
                )
            overrides_by_t.setdefault(int(ti), []).append((int(ell), int(var), float(val)))

        def hook(X_cf: np.ndarray, t: int, max_lag_used: int) -> None:
            t_post = t - max_lag_used
            if t_post in overrides_by_t:
                for ell, var, val in overrides_by_t[t_post]:
                    X_cf[t, ell, var] = val

        earliest = min(overrides_by_t.keys())
        cf = self._rollout(hook, earliest_intervention_t=earliest)
        ite = cf - self._factual.X
        return CounterfactualPair(
            factual=self._factual.X.copy(), counterfactual=cf, ite=ite
        )

    # ------------------------------------------------------------------
    # Internal: shift intervention
    # ------------------------------------------------------------------

    def _apply_shift(self, iv: ShiftIntervention) -> CounterfactualPair:
        T = self._factual.X.shape[0]
        overrides_by_t: dict[int, list[tuple[int, int, float]]] = {}
        for ti, ell, var, delta in zip(
            iv.times, iv.locations, iv.variables, iv.deltas
        ):
            if not (0 <= ti < T):
                raise ValueError(
                    f"Intervention time {ti} out of range [0, {T})."
                )
            overrides_by_t.setdefault(int(ti), []).append(
                (int(ell), int(var), float(delta))
            )

        def hook(X_cf: np.ndarray, t: int, max_lag_used: int) -> None:
            t_post = t - max_lag_used
            if t_post in overrides_by_t:
                for ell, var, delta in overrides_by_t[t_post]:
                    X_cf[t, ell, var] = X_cf[t, ell, var] + delta

        earliest = min(overrides_by_t.keys())
        cf = self._rollout(hook, earliest_intervention_t=earliest)
        ite = cf - self._factual.X
        return CounterfactualPair(
            factual=self._factual.X.copy(), counterfactual=cf, ite=ite
        )
