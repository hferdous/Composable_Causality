"""
Spatio-temporal configuration objects and identifiability checks.

The SpatioTemporalConfig encodes every choice that affects the data-
generating process: spatial topology, kernel, per-variable spatial
coupling strengths, lag at which spatial influence applies, and the
within-location SCM specified by an existing dtcd `DGPConfig`.

Identifiability tags follow the same conventions as the core dtcd
library:
    - "identifiable":  point-identified DAG including all spatial parameters
    - "cpdag":         identified up to Markov equivalence at each location
    - "spatial_only":  spatial kernel identifiable but within-location DAG is not
    - "unidentifiable": cannot recover causal structure even in principle
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .kernels import SpatialKernel
from .topologies import SpatialTopology


# ---------------------------------------------------------------------------
# Config object
# ---------------------------------------------------------------------------


@dataclass
class SpatioTemporalConfig:
    """
    Specification of a spatio-temporal data-generating process.

    The within-location structural causal model is provided externally
    (typically a `dtcd.DGPConfig` already used for non-spatial benchmarks).
    This class adds the spatial layer on top.

    Attributes
    ----------
    topology : SpatialTopology
        Spatial layout of the L locations.
    kernel : SpatialKernel
        Influence kernel that builds the L x L matrix K from the topology.
    spatial_coupling : np.ndarray, shape (d,)
        Per-variable scaling s_i of the spatial influence term. Variables
        with s_i=0 are "non-diffusive" — their values at one location are
        not directly affected by neighbors. Defaults to all-ones if not
        specified, but realistic configurations typically set only a
        subset of variables to nonzero (e.g. temperature diffuses, but
        a categorical regime indicator does not).
    spatial_lag : int
        Temporal lag at which spatial coupling acts. Default 1. Setting
        to 0 enables instantaneous spatial diffusion, which is confounded
        with within-location contemporaneous causal effects and emits an
        identifiability warning.
    same_within_location_graph : bool
        If True (default), every location uses the same within-location
        W and A matrices (drawn once and replicated). If False, each
        location draws its own — only realistic when the user models
        regionally heterogeneous physics, and substantially harder to
        identify.
    """

    topology: SpatialTopology
    kernel: SpatialKernel
    spatial_coupling: np.ndarray
    spatial_lag: int = 1
    same_within_location_graph: bool = True

    def __post_init__(self) -> None:
        coupling = np.asarray(self.spatial_coupling, dtype=np.float64)
        if coupling.ndim != 1:
            raise ValueError("spatial_coupling must be 1D of shape (d,).")
        if np.any(coupling < 0):
            warnings.warn(
                "Negative spatial_coupling values produce anti-diffusive "
                "dynamics; this is supported but rarely physically motivated.",
                UserWarning,
                stacklevel=2,
            )
        self.spatial_coupling = coupling

        if self.spatial_lag < 0:
            raise ValueError("spatial_lag must be non-negative.")

    @property
    def n_locations(self) -> int:
        return self.topology.n_locations

    @property
    def n_variables(self) -> int:
        return int(self.spatial_coupling.shape[0])


# ---------------------------------------------------------------------------
# Identifiability classification
# ---------------------------------------------------------------------------


def classify_identifiability(
    config: SpatioTemporalConfig,
    *,
    within_location_class: str,
) -> str:
    """
    Classify the identifiability status of a (within-location SCM, spatial
    layer) combination.

    Parameters
    ----------
    config : SpatioTemporalConfig
        The spatial portion of the DGP.
    within_location_class : str
        Identifiability class of the within-location SCM, one of
        {"identifiable", "cpdag", "unidentifiable"}. Typically obtained
        from `dtcd.identifiability.classify(within_location_dgp)`.

    Returns
    -------
    str
        One of {"identifiable", "cpdag", "spatial_only", "unidentifiable"}.

    Notes
    -----
    Rules encoded:
        - If spatial_lag == 0, instantaneous spatial coupling is confounded
          with the contemporaneous within-location DAG. The combined model
          is then unidentifiable regardless of the within-location class.
        - If same_within_location_graph is False, each of L within-location
          DAGs has only T samples to identify it (rather than L*T), so
          identifiability degrades to the within-location class only when
          T is large enough; we mark it as "cpdag" pessimistically.
        - Otherwise the joint identifiability follows the within-location
          class, since the spatial kernel is point-identified separately.
    """
    if within_location_class not in {"identifiable", "cpdag", "unidentifiable"}:
        raise ValueError(
            f"Unknown within_location_class: {within_location_class!r}."
        )

    if config.spatial_lag == 0:
        return "unidentifiable"

    if not config.same_within_location_graph:
        # Pessimistic: each location's graph has fewer effective samples.
        if within_location_class == "identifiable":
            return "cpdag"
        return within_location_class

    return within_location_class


def warn_if_unidentifiable(
    config: SpatioTemporalConfig,
    *,
    within_location_class: str,
    allow_unidentifiable: bool = False,
) -> None:
    """
    Emit a warning (or raise) for unidentifiable configurations.

    Default: warns and continues, matching the existing dtcd convention.
    Pass `allow_unidentifiable=False` only if the caller wants strict
    enforcement (raises ValueError instead).
    """
    cls = classify_identifiability(
        config, within_location_class=within_location_class
    )
    if cls == "unidentifiable":
        msg = (
            "Configuration is unidentifiable in principle: "
            f"spatial_lag={config.spatial_lag}, "
            f"within_location_class={within_location_class}. "
            "Methods evaluated on this configuration cannot be expected to "
            "recover the true graph regardless of sample size."
        )
        if not allow_unidentifiable:
            warnings.warn(msg, UserWarning, stacklevel=2)
        else:
            raise ValueError(msg)
