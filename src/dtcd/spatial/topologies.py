"""
Spatial topology generators.

A spatial topology specifies (i) the coordinates of L locations in some
ambient space (R^2 for lat-lon grids, R^3 for atmospheric columns, or an
abstract embedding for sensor networks), and (ii) optionally an explicit
adjacency matrix when the topology is non-metric (e.g. a road network where
"distance" is graph distance, not Euclidean).

Causal assumptions encoded here:
    - Locations themselves are not causally affected by the variables. The
      spatial structure is an exogenous design feature, not part of the
      structural causal model.
    - The coordinate system is fixed across time. Moving sensors require an
      extension not implemented in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Common output container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpatialTopology:
    """
    Container describing the spatial layout of L locations.

    Attributes
    ----------
    coords : np.ndarray, shape (L, dim_space)
        Coordinates of each location in the ambient space.
    adjacency : np.ndarray or None, shape (L, L)
        Optional explicit adjacency matrix. When None, kernels compute
        influence directly from `coords` via a metric distance. When
        provided, kernels may use `adjacency` instead (e.g. for road
        networks where Euclidean distance is misleading).
    name : str
        Human-readable identifier of the generator that produced this
        topology. Recorded for reproducibility metadata.
    """

    coords: np.ndarray
    adjacency: Optional[np.ndarray]
    name: str

    @property
    def n_locations(self) -> int:
        """Number of locations L."""
        return int(self.coords.shape[0])

    @property
    def dim_space(self) -> int:
        """Dimensionality of the ambient coordinate space."""
        return int(self.coords.shape[1])


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def grid_2d(n_rows: int, n_cols: int, spacing: float = 1.0) -> SpatialTopology:
    """
    Regular 2D rectangular grid.

    Parameters
    ----------
    n_rows, n_cols : int
        Grid dimensions. Total locations L = n_rows * n_cols.
    spacing : float
        Distance between adjacent grid cells. Defaults to 1.0; users
        modeling a real lat-lon grid should pass the cell size in km
        (or whatever unit they want kernel bandwidths expressed in).

    Returns
    -------
    SpatialTopology
        Coordinates on a regular lattice; adjacency=None (kernels use
        Euclidean distance directly).

    Notes
    -----
    Causal assumption: Grid spacing is uniform and known. Real-world
    lat-lon grids near the poles violate this; users should preproject
    coordinates to an equal-area projection before passing them in via
    `from_coords` rather than relying on this generator.
    """
    if n_rows < 1 or n_cols < 1:
        raise ValueError("n_rows and n_cols must be >= 1.")
    if spacing <= 0:
        raise ValueError("spacing must be positive.")

    rows, cols = np.meshgrid(
        np.arange(n_rows, dtype=np.float64),
        np.arange(n_cols, dtype=np.float64),
        indexing="ij",
    )
    coords = np.stack([rows.ravel(), cols.ravel()], axis=1) * spacing
    return SpatialTopology(coords=coords, adjacency=None, name="grid_2d")


def grid_3d(
    n_x: int, n_y: int, n_z: int, spacing: float = 1.0
) -> SpatialTopology:
    """
    Regular 3D grid (e.g. atmospheric column, voxelized brain region).

    Parameters
    ----------
    n_x, n_y, n_z : int
        Grid dimensions; total locations L = n_x * n_y * n_z.
    spacing : float
        Distance between adjacent grid cells along each axis.

    Causal assumption: same as `grid_2d`. Anisotropic spacing (different
    physical distances along each axis, common in atmospheric science
    where vertical resolution differs from horizontal) is not supported
    here; pass coords directly via `from_coords`.
    """
    if min(n_x, n_y, n_z) < 1:
        raise ValueError("All grid dimensions must be >= 1.")
    if spacing <= 0:
        raise ValueError("spacing must be positive.")

    xs, ys, zs = np.meshgrid(
        np.arange(n_x, dtype=np.float64),
        np.arange(n_y, dtype=np.float64),
        np.arange(n_z, dtype=np.float64),
        indexing="ij",
    )
    coords = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1) * spacing
    return SpatialTopology(coords=coords, adjacency=None, name="grid_3d")


def random_geometric(
    n_locations: int,
    dim_space: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> SpatialTopology:
    """
    Locations drawn uniformly at random in the unit hypercube.

    Useful for sensor-network simulations where placement is irregular but
    the metric is meaningful. Not appropriate for road networks or any
    scenario where graph distance differs from Euclidean distance — use
    `from_adjacency` with explicit edges in those cases.

    Causal assumption: Spatial positions are exogenous (not influenced by
    the variables being simulated). For real sensor networks where
    placement was optimized based on prior signal observations, this
    assumption is violated and the resulting benchmark will understate
    real-world identifiability difficulty.
    """
    if n_locations < 1:
        raise ValueError("n_locations must be >= 1.")
    if dim_space < 1:
        raise ValueError("dim_space must be >= 1.")

    rng = np.random.default_rng() if rng is None else rng
    coords = rng.uniform(0.0, 1.0, size=(n_locations, dim_space))
    return SpatialTopology(coords=coords, adjacency=None, name="random_geometric")


def from_coords(
    coords: np.ndarray, adjacency: Optional[np.ndarray] = None
) -> SpatialTopology:
    """
    Build a topology from user-provided coordinates and (optionally)
    adjacency. This is the entry point for real geographic data.

    Parameters
    ----------
    coords : np.ndarray, shape (L, dim_space)
        User-supplied coordinates. For lat-lon data, the user is responsible
        for projecting to an equal-distance representation if Euclidean
        kernels are to be applied.
    adjacency : np.ndarray, optional, shape (L, L)
        Explicit adjacency matrix. Required for non-metric topologies
        (e.g. road networks) where graph structure differs from spatial
        proximity. Should be symmetric for undirected adjacency.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2:
        raise ValueError("coords must be a 2D array of shape (L, dim_space).")
    if adjacency is not None:
        adjacency = np.asarray(adjacency, dtype=np.float64)
        L = coords.shape[0]
        if adjacency.shape != (L, L):
            raise ValueError(
                f"adjacency shape {adjacency.shape} inconsistent with "
                f"coords shape (L={L})."
            )
    return SpatialTopology(coords=coords, adjacency=adjacency, name="from_coords")
