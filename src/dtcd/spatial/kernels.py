"""
Spatial influence kernels.

A spatial kernel K is an L x L matrix where K[ell', ell] quantifies the
strength with which location ell' influences location ell at one temporal
lag. We expose four parametric families:

    - GaussianKernel:   K(ell, ell') = exp(-||p_ell - p_ell'||^2 / (2 sigma^2))
    - PowerLawKernel:   K(ell, ell') = (||p_ell - p_ell'|| + epsilon)^(-alpha)
    - AdjacencyKernel:  K(ell, ell') = adjacency[ell, ell'] (binary or weighted)
    - StepKernel:       K(ell, ell') = 1 if distance <= radius else 0

All kernels zero out the diagonal (no self-influence through the spatial
channel; self-effects belong to the within-location lagged graph A_tau).

Identifiability note:
    For the spatial kernel to be identifiable from observational data, it
    must be constrained to a finite-parameter family. The kernels exposed
    here are 1- or 2-parameter; each is identifiable up to the parameter(s)
    given sufficient location diversity. Free-form L x L kernels (the
    `from_matrix` constructor) are unidentifiable in general and emit a
    warning when used with default settings. See `config.py`.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .topologies import SpatialTopology


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class SpatialKernel(ABC):
    """
    Abstract spatial kernel. Subclasses implement `_build_matrix` returning
    an L x L numpy array; the base class handles diagonal zeroing,
    optional row-normalization, and shape validation.

    Causal assumption: K is constant in time. Time-varying spatial
    structure (e.g. a moving weather front shifting the diffusion kernel)
    is not supported by this base class; users would need to subclass and
    override the construction at each timestep.
    """

    def __init__(self, *, normalize: bool = True) -> None:
        """
        Parameters
        ----------
        normalize : bool
            If True, each row of K is rescaled to sum to 1, so the spatial
            term in the structural equation behaves as a weighted average
            of neighbors rather than an unbounded sum. Strongly recommended
            for stability; setting False is supported only for users who
            want raw kernel values for diagnostic purposes.
        """
        self._normalize = bool(normalize)

    @abstractmethod
    def _build_matrix(self, topology: SpatialTopology) -> np.ndarray:
        """Subclass implementation. Returns raw L x L kernel before
        diagonal zeroing and normalization."""

    def build(self, topology: SpatialTopology) -> np.ndarray:
        """
        Construct the L x L spatial kernel matrix.

        Returns
        -------
        np.ndarray, shape (L, L)
            Kernel with zero diagonal. Row-normalized if `normalize=True`
            (default). Each row sums to at most 1 even when normalization
            is off (kernels here are always non-negative).
        """
        K = np.asarray(self._build_matrix(topology), dtype=np.float64)
        L = topology.n_locations
        if K.shape != (L, L):
            raise ValueError(f"Kernel shape {K.shape} inconsistent with L={L}.")
        np.fill_diagonal(K, 0.0)

        if self._normalize:
            row_sums = K.sum(axis=1, keepdims=True)
            # Avoid division by zero for isolated rows (no neighbors).
            row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
            K = K / row_sums
        return K


# ---------------------------------------------------------------------------
# Gaussian kernel
# ---------------------------------------------------------------------------


class GaussianKernel(SpatialKernel):
    """
    K(ell, ell') = exp(-||p_ell - p_ell'||^2 / (2 sigma^2))

    Standard choice when influence falls off smoothly with distance and the
    decay scale `sigma` is the natural parameter of interest. This is the
    spatial analog of the squared-exponential GP kernel.

    Identifiable from observational data given enough location pairs at
    different distances and a sufficiently long time series. The single
    parameter `sigma` is identifiable; the kernel value at a given
    distance is fully determined by it.
    """

    def __init__(self, sigma: float, *, normalize: bool = True) -> None:
        super().__init__(normalize=normalize)
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        self._sigma = float(sigma)

    def _build_matrix(self, topology: SpatialTopology) -> np.ndarray:
        diffs = topology.coords[:, None, :] - topology.coords[None, :, :]
        sq_dist = np.sum(diffs**2, axis=-1)
        return np.exp(-sq_dist / (2.0 * self._sigma**2))


# ---------------------------------------------------------------------------
# Power-law kernel
# ---------------------------------------------------------------------------


class PowerLawKernel(SpatialKernel):
    """
    K(ell, ell') = (||p_ell - p_ell'|| + epsilon)^(-alpha)

    Heavy-tailed alternative to GaussianKernel. Appropriate when influence
    decays polynomially rather than exponentially (gravitational analogies,
    epidemic spread on long-range networks). The epsilon offset prevents
    singularity at zero distance and should be set comparable to the
    minimum distance between locations.

    Note: this kernel does not naturally vanish at moderate distances, so
    long-range entries can dominate when L is large. Row-normalization
    (default) is essential for numerical stability.
    """

    def __init__(
        self, alpha: float, *, epsilon: float = 1e-3, normalize: bool = True
    ) -> None:
        super().__init__(normalize=normalize)
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        self._alpha = float(alpha)
        self._epsilon = float(epsilon)

    def _build_matrix(self, topology: SpatialTopology) -> np.ndarray:
        diffs = topology.coords[:, None, :] - topology.coords[None, :, :]
        dist = np.sqrt(np.sum(diffs**2, axis=-1))
        return (dist + self._epsilon) ** (-self._alpha)


# ---------------------------------------------------------------------------
# Step (radius) kernel
# ---------------------------------------------------------------------------


class StepKernel(SpatialKernel):
    """
    K(ell, ell') = 1 if ||p_ell - p_ell'|| <= radius else 0

    Hard cutoff. Models scenarios with a well-defined interaction range
    (e.g. radio sensors with finite communication radius, or grid cells
    where only immediate neighbors interact). Equivalent to building a
    geometric graph at threshold `radius` and using AdjacencyKernel.

    Identifiable: the discrete neighborhood structure is fully determined
    by `radius` given the coordinates.
    """

    def __init__(self, radius: float, *, normalize: bool = True) -> None:
        super().__init__(normalize=normalize)
        if radius <= 0:
            raise ValueError("radius must be positive.")
        self._radius = float(radius)

    def _build_matrix(self, topology: SpatialTopology) -> np.ndarray:
        diffs = topology.coords[:, None, :] - topology.coords[None, :, :]
        dist = np.sqrt(np.sum(diffs**2, axis=-1))
        return (dist <= self._radius).astype(np.float64)


# ---------------------------------------------------------------------------
# Adjacency-based kernel (for non-metric topologies)
# ---------------------------------------------------------------------------


class AdjacencyKernel(SpatialKernel):
    """
    K(ell, ell') = topology.adjacency[ell, ell']

    Uses the explicit adjacency matrix supplied with the topology. This is
    the only correct choice for non-metric structures (road networks, brain
    connectivity, social networks) where Euclidean distance is misleading.

    Requires `topology.adjacency is not None`; raises otherwise.
    """

    def _build_matrix(self, topology: SpatialTopology) -> np.ndarray:
        if topology.adjacency is None:
            raise ValueError(
                "AdjacencyKernel requires a topology with an explicit "
                "adjacency matrix; the supplied topology has adjacency=None. "
                "Use a metric kernel (Gaussian, PowerLaw, Step) instead, or "
                "construct the topology via topologies.from_coords with an "
                "adjacency argument."
            )
        return topology.adjacency.astype(np.float64).copy()


# ---------------------------------------------------------------------------
# Free-form kernel (escape hatch — emits identifiability warning)
# ---------------------------------------------------------------------------


class FromMatrixKernel(SpatialKernel):
    """
    User-supplied L x L kernel matrix. Provides full flexibility but
    sacrifices identifiability: an arbitrary L x L matrix has L^2 free
    parameters, which is not consistently estimable from a single time
    series of any finite length.

    Emits a warning at construction time. The `acknowledge_unidentifiable`
    flag silences the warning; use only when the kernel structure is
    known a priori from physical theory.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        *,
        normalize: bool = True,
        acknowledge_unidentifiable: bool = False,
    ) -> None:
        super().__init__(normalize=normalize)
        if not acknowledge_unidentifiable:
            warnings.warn(
                "FromMatrixKernel uses an unconstrained L x L kernel which "
                "is generally unidentifiable from observational data. "
                "Pass acknowledge_unidentifiable=True to suppress this "
                "warning if the kernel is known a priori.",
                UserWarning,
                stacklevel=2,
            )
        self._matrix = np.asarray(matrix, dtype=np.float64).copy()

    def _build_matrix(self, topology: SpatialTopology) -> np.ndarray:
        L = topology.n_locations
        if self._matrix.shape != (L, L):
            raise ValueError(
                f"FromMatrixKernel matrix shape {self._matrix.shape} "
                f"inconsistent with topology (L={L})."
            )
        return self._matrix
