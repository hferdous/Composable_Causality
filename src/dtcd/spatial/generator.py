"""
Spatio-temporal data generator with cached-noise rollouts.

Produces trajectories of shape (T, L, d) along with the cached noise tensor
N of identical shape, the within-location structural matrices (W, A_tau),
and the spatial kernel K. The cached noise is the central object that
enables exact counterfactual twin generation (`SpatioTemporalCounterfactualSuite`).

Structural equation:
    X[t, ell, i] = sum_j W[j, i] * X[t, ell, j]
                 + sum_{tau=1..tau_max} sum_j A_tau[j, i] * X[t-tau, ell, j]
                 + s[i] * sum_{ell' != ell} K[ell', ell] * X[t - spatial_lag, ell', i]
                 + N[t, ell, i]

Causal assumptions (also documented per-method):
    - Markov: each variable is independent of non-descendants given parents
      in the time-and-space-unrolled DAG.
    - Faithfulness: no fortuitous parameter cancellations (the user is
      responsible for not constructing pathological coefficient sets).
    - Causal sufficiency: the simulator's variable set X[t, ell, :] for all
      t, ell is closed under causation. No hidden common causes are
      introduced. Spatial confounding via location features (lat-lon
      affecting variables exogenously) is NOT modeled here; users who
      want this should add the location features as explicit variables.
    - Stationarity: structural matrices and the spatial kernel do not vary
      with t. For drifting structural coefficients, use the drift mechanism
      in the underlying within-location SCM.

Notes on shapes:
    X : (T, L, d)       trajectory
    N : (T, L, d)       cached noise; identical to X minus structural part
    W : (d, d)          contemporaneous adjacency (zero diagonal, acyclic)
    A : (tau_max, d, d) lagged adjacency stack
    K : (L, L)          spatial kernel (zero diagonal)
    s : (d,)            per-variable spatial coupling
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import (
    SpatioTemporalConfig,
    classify_identifiability,
    warn_if_unidentifiable,
)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------


@dataclass
class SpatioTemporalDataset:
    """
    Output of `SpatioTemporalGenerator.generate`.

    Attributes
    ----------
    X : np.ndarray, shape (T, L, d)
        The generated trajectory.
    N : np.ndarray, shape (T, L, d)
        Cached noise realization. Required input to
        `SpatioTemporalCounterfactualSuite` for exact twin generation.
    prehistory : np.ndarray, shape (max_lag_used, L, d)
        The trajectory values immediately preceding t=0, needed to seed
        counterfactual rollouts whose intervention timestep is small.
        This is the tail of the burn-in, not zeros.
    W : np.ndarray, shape (d, d)
        Contemporaneous adjacency / coefficients (the within-location
        instantaneous DAG). Same at every location when
        `same_within_location_graph=True`.
    A : np.ndarray, shape (tau_max, d, d)
        Lagged adjacency / coefficients. A[tau-1] is the matrix for lag tau.
    K : np.ndarray, shape (L, L)
        Spatial kernel.
    spatial_coupling : np.ndarray, shape (d,)
        Per-variable spatial coupling strength s.
    spatial_lag : int
        The lag at which spatial coupling acts.
    identifiability : str
        Classification result; see `config.classify_identifiability`.
    """

    X: np.ndarray
    N: np.ndarray
    prehistory: np.ndarray
    W: np.ndarray
    A: np.ndarray
    K: np.ndarray
    spatial_coupling: np.ndarray
    spatial_lag: int
    identifiability: str

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.X.shape)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class SpatioTemporalGenerator:
    """
    Generate spatio-temporal trajectories under a configured DGP.

    The within-location SCM (W, A_tau, noise distribution) is supplied
    as already-sampled numpy arrays. This keeps the spatial generator
    decoupled from the structural-mechanism implementation in dtcd.core,
    so any within-location mechanism (linear SVAR, PNL, Fourier) that
    eventually produces W and A can be wrapped here.

    Parameters
    ----------
    config : SpatioTemporalConfig
        Spatial layer configuration.
    W : np.ndarray, shape (d, d)
        Within-location contemporaneous coefficient matrix. Diagonal must
        be zero and the implied directed graph must be acyclic.
    A : np.ndarray, shape (tau_max, d, d)
        Within-location lagged coefficient stack.
    rng : np.random.Generator, optional
        Random number generator. Defaults to a fresh default_rng().

    Causal assumption: W and A encode the true within-location SCM. They
    must be consistent with whatever mechanism class the user claims;
    e.g. PNL outer distortions are applied separately and are NOT folded
    into W. (For the spatial layer, distortions complicate identifiability
    and are applied per-location via the `outer_distortion` callable.)
    """

    def __init__(
        self,
        config: SpatioTemporalConfig,
        W: np.ndarray,
        A: np.ndarray,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._config = config
        self._W = np.asarray(W, dtype=np.float64)
        self._A = np.asarray(A, dtype=np.float64)
        self._rng = np.random.default_rng() if rng is None else rng

        d = config.n_variables
        if self._W.shape != (d, d):
            raise ValueError(
                f"W shape {self._W.shape} inconsistent with d={d}."
            )
        if self._A.ndim != 3 or self._A.shape[1:] != (d, d):
            raise ValueError(
                f"A shape {self._A.shape} expected (tau_max, {d}, {d})."
            )
        if not np.allclose(np.diag(self._W), 0.0):
            raise ValueError("W must have zero diagonal (no self-loops).")
        if not _is_acyclic(self._W):
            raise ValueError(
                "W induces a cycle in the contemporaneous DAG; "
                "rejection-sample W until acyclic before passing in."
            )

        self._tau_max = int(self._A.shape[0])
        self._K = config.kernel.build(config.topology)

    # ------------------------------------------------------------------
    # Identifiability surface
    # ------------------------------------------------------------------

    def identifiability_class(self, within_location_class: str) -> str:
        """Classify the joint (within-location, spatial) identifiability."""
        return classify_identifiability(
            self._config, within_location_class=within_location_class
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        T: int,
        *,
        burn_in: int = 200,
        noise_scale: float = 1.0,
        within_location_class: str = "identifiable",
        allow_unidentifiable: bool = False,
        precomputed_noise: Optional[np.ndarray] = None,
    ) -> SpatioTemporalDataset:
        """
        Roll out a trajectory of length T.

        Parameters
        ----------
        T : int
            Length of the returned trajectory (after burn-in).
        burn_in : int
            Number of initial timesteps to discard for stationarity. The
            actual rollout has length T + burn_in.
        noise_scale : float
            Standard deviation of the (Gaussian) noise. For non-Gaussian
            noise, pass `precomputed_noise` directly instead — this lets
            the caller use the existing dtcd noise generators.
        within_location_class : str
            Identifiability class of the within-location SCM, used only
            for emitting warnings about the joint configuration.
        allow_unidentifiable : bool
            If True, raises on unidentifiable configurations rather than
            warning. Default False (warn and continue).
        precomputed_noise : np.ndarray, optional, shape (T+burn_in, L, d)
            If supplied, used directly instead of generating Gaussian
            noise. This is the integration point for the existing dtcd
            noise generators (Cauchy, drift, etc.).

        Returns
        -------
        SpatioTemporalDataset

        Notes on the structural recursion
        ---------------------------------
        Define the contemporaneous-projection matrix M = (I - W^T)^{-1}.
        At each timestep t we compute
            lagged[ell, i] = sum_{tau, j} A_tau[j, i] X[t-tau, ell, j]
            spatial[ell, i] = s[i] * sum_{ell'} K[ell', ell] X[t-spatial_lag, ell', i]
            X[t, ell, :] = M (lagged + spatial + N[t, ell, :])
        The matrix M reduces from the structural form to the reduced form;
        we precompute it once.
        """
        if T < 1:
            raise ValueError("T must be >= 1.")
        if burn_in < 0:
            raise ValueError("burn_in must be non-negative.")

        warn_if_unidentifiable(
            self._config,
            within_location_class=within_location_class,
            allow_unidentifiable=allow_unidentifiable,
        )

        L = self._config.n_locations
        d = self._config.n_variables
        tau_max = self._tau_max
        spatial_lag = self._config.spatial_lag
        s = self._config.spatial_coupling
        K = self._K

        T_total = T + burn_in
        # Need extra history slots for the maximum of (tau_max, spatial_lag).
        max_lag_used = max(tau_max, spatial_lag)

        # Allocate full trajectory including pre-history (zero-initialized).
        X = np.zeros((T_total + max_lag_used, L, d), dtype=np.float64)
        if precomputed_noise is not None:
            noise = np.asarray(precomputed_noise, dtype=np.float64)
            if noise.shape != (T_total, L, d):
                raise ValueError(
                    f"precomputed_noise shape {noise.shape} expected "
                    f"({T_total}, {L}, {d})."
                )
        else:
            noise = self._rng.standard_normal((T_total, L, d)) * noise_scale

        # Reduced-form contemporaneous projection: M = (I - W^T)^-1.
        # Acyclicity of W guarantees invertibility.
        M = np.linalg.inv(np.eye(d) - self._W.T)

        # Roll out.
        for t in range(max_lag_used, max_lag_used + T_total):
            # Lagged within-location term: shape (L, d).
            lagged = np.zeros((L, d), dtype=np.float64)
            for tau in range(1, tau_max + 1):
                # A[tau-1] is (d, d); X[t-tau] is (L, d).
                lagged += X[t - tau] @ self._A[tau - 1]

            # Spatial term at lag `spatial_lag`: shape (L, d).
            if spatial_lag > 0:
                # K.T @ X[t-spatial_lag] performs the location-summation:
                # for each ell, summing over ell' of K[ell', ell] * X[t-spatial_lag, ell', :].
                spatial = (K.T @ X[t - spatial_lag]) * s[None, :]
            else:
                # Lag-0 spatial coupling (unidentifiable configuration).
                # We skip it here; the warning was already emitted above.
                spatial = np.zeros((L, d), dtype=np.float64)

            n_idx = t - max_lag_used
            X[t] = (lagged + spatial + noise[n_idx]) @ M.T

        # Strip pre-history and burn-in.
        X_out = X[max_lag_used + burn_in :]
        N_out = noise[burn_in:]
        # The pre-history for the post-burn-in trajectory is the
        # max_lag_used steps immediately before the post-burn-in start.
        prehistory = X[burn_in : burn_in + max_lag_used].copy()

        return SpatioTemporalDataset(
            X=X_out,
            N=N_out,
            prehistory=prehistory,
            W=self._W.copy(),
            A=self._A.copy(),
            K=K.copy(),
            spatial_coupling=s.copy(),
            spatial_lag=spatial_lag,
            identifiability=self.identifiability_class(within_location_class),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_acyclic(W: np.ndarray, tol: float = 1e-9) -> bool:
    """
    Check that the directed graph induced by W (nonzero entries as edges)
    is acyclic. Uses the trace-of-matrix-exponential criterion from
    NOTEARS: tr(exp(|W|)) - d == 0 iff acyclic. Robust for small d; for
    very large d a topological-sort check would be cheaper, but this
    module's intended scale (d in tens) makes the matrix-exponential test
    fine.
    """
    A = np.abs(W).astype(np.float64)
    d = A.shape[0]
    h = np.trace(np.linalg.matrix_power(np.eye(d) + A / d, d)) - d
    return bool(h < tol)
