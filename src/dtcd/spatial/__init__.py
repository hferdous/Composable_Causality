"""
dtcd.spatial — spatio-temporal extension of the dtcd data-generating library.

Composes with the existing dtcd within-location SCM to produce
trajectories of shape (T, L, d) on a configurable spatial topology.
Supports exact counterfactual twins for spatial CATE evaluation.

Public API:
    Topologies:   grid_2d, grid_3d, random_geometric, from_coords,
                  SpatialTopology
    Kernels:      GaussianKernel, PowerLawKernel, StepKernel,
                  AdjacencyKernel, FromMatrixKernel, SpatialKernel
    Config:       SpatioTemporalConfig, classify_identifiability,
                  warn_if_unidentifiable
    Generation:   SpatioTemporalGenerator, SpatioTemporalDataset
    Counterfactuals: SpatioTemporalCounterfactualSuite,
                  HardIntervention, ShiftIntervention,
                  CounterfactualPair

Quick start:

    from dtcd.spatial import (
        grid_2d, GaussianKernel, SpatioTemporalConfig,
        SpatioTemporalGenerator, SpatioTemporalCounterfactualSuite,
        HardIntervention,
    )
    import numpy as np

    rng = np.random.default_rng(42)
    d = 4
    topology = grid_2d(n_rows=4, n_cols=4)            # L = 16
    kernel = GaussianKernel(sigma=1.5)
    config = SpatioTemporalConfig(
        topology=topology, kernel=kernel,
        spatial_coupling=np.array([0.3, 0.3, 0.0, 0.0]),  # only first 2 vars diffuse
        spatial_lag=1,
    )

    # Sample a within-location DAG and lagged stack (caller's responsibility,
    # typically using dtcd.core utilities).
    W = np.zeros((d, d))
    W[0, 1] = 0.5; W[1, 2] = 0.4
    A = rng.normal(0, 0.2, size=(2, d, d))            # tau_max=2

    gen = SpatioTemporalGenerator(config, W, A, rng=rng)
    factual = gen.generate(T=500)

    cfs = SpatioTemporalCounterfactualSuite(factual)
    twin = cfs.apply(HardIntervention(
        times=[100], locations=[5], variables=[0], values=[2.0]
    ))
    # twin.ite[t, ell, i] is exact ground-truth treatment effect.
"""

from .config import (
    SpatioTemporalConfig,
    classify_identifiability,
    warn_if_unidentifiable,
)
from .counterfactual import (
    CounterfactualPair,
    HardIntervention,
    ShiftIntervention,
    SpatioTemporalCounterfactualSuite,
)
from .generator import SpatioTemporalDataset, SpatioTemporalGenerator
from .kernels import (
    AdjacencyKernel,
    FromMatrixKernel,
    GaussianKernel,
    PowerLawKernel,
    SpatialKernel,
    StepKernel,
)
from .topologies import (
    SpatialTopology,
    from_coords,
    grid_2d,
    grid_3d,
    random_geometric,
)

__all__ = [
    # topologies
    "SpatialTopology",
    "grid_2d",
    "grid_3d",
    "random_geometric",
    "from_coords",
    # kernels
    "SpatialKernel",
    "GaussianKernel",
    "PowerLawKernel",
    "StepKernel",
    "AdjacencyKernel",
    "FromMatrixKernel",
    # config
    "SpatioTemporalConfig",
    "classify_identifiability",
    "warn_if_unidentifiable",
    # generation
    "SpatioTemporalGenerator",
    "SpatioTemporalDataset",
    # counterfactuals
    "SpatioTemporalCounterfactualSuite",
    "HardIntervention",
    "ShiftIntervention",
    "CounterfactualPair",
]
