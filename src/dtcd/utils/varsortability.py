"""Varsortability diagnostics for generated benchmark data.

Varsortability (Reisach, Seiler & Weichwald, "Beware of the Simulated DAG!",
NeurIPS 2021) measures how well the marginal variance of each variable encodes
its position in the causal order. A value of 0.5 is uninformative. A value near
1.0 means a baseline that simply sorts variables by increasing marginal variance
and regresses each on its predecessors will recover the graph, so reported
performance on such data does not evidence causal reasoning.

Synthetic additive-noise generators with homoscedastic noise and coefficients
bounded away from zero produce high varsortability by construction, because
variance accumulates along every structural path. This module makes the
resulting risk measurable rather than implicit.

Time-series note
----------------
In a window graph, lagged edges are already direction-resolved by time ordering,
so the sorting shortcut is only informative about the contemporaneous
sub-graph. `contemporaneous_varsortability` is therefore the primary diagnostic.
`summary_varsortability` collapses lags and is reported alongside it because a
window-graph learner such as DYNOTEARS scores both edge types jointly.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "varsortability",
    "contemporaneous_varsortability",
    "summary_varsortability",
    "standardize",
]


def varsortability(X: np.ndarray, W: np.ndarray, tol: float = 1e-9) -> float:
    """Fraction of directed paths running from lower to higher marginal variance.

    Parameters
    ----------
    X : np.ndarray, shape (T, d)
        Observed trajectory.
    W : np.ndarray, shape (d, d)
        Adjacency in DTCD convention: ``W[i, j] != 0`` means ``i -> j``.
    tol : float
        Tolerance for treating a variance ratio as exactly 1.

    Returns
    -------
    float
        Value in [0, 1]. 0.5 is uninformative, 1.0 is perfectly sortable.
        ``np.nan`` when the graph has no edges.
    """
    E = (np.abs(W) > tol).astype(int)
    if E.sum() == 0:
        return float("nan")
    Ek = E.copy()
    var = np.var(X, axis=0, keepdims=True)
    n_paths = 0.0
    n_ordered = 0.0
    for _ in range(E.shape[0] - 1):
        n_paths += Ek.sum()
        ratio = var / var.T
        n_ordered += (Ek * (ratio > 1 + tol)).sum()
        n_ordered += 0.5 * (Ek * (np.abs(ratio - 1) <= tol)).sum()
        Ek = Ek.dot(E)
        if Ek.sum() == 0:
            break
    return float(n_ordered / n_paths)


def contemporaneous_varsortability(X: np.ndarray, W0: np.ndarray) -> float:
    """Varsortability of the contemporaneous sub-graph. Primary diagnostic."""
    return varsortability(X, W0)


def summary_varsortability(X: np.ndarray, W0: np.ndarray, W_lags: np.ndarray) -> float:
    """Varsortability of the summary graph (contemporaneous plus all lags)."""
    W_sum = (np.abs(W0) > 0).astype(float)
    for k in range(W_lags.shape[0]):
        W_sum = W_sum + (np.abs(W_lags[k]) > 0).astype(float)
    return varsortability(X, W_sum)


def standardize(X: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Z-score each column, removing the marginal-variance shortcut.

    Standardizing drives varsortability to exactly 0.5 by construction: all
    variances become equal, so every path contributes a tie. Any performance
    difference between raw and standardized input is attributable to the
    shortcut rather than to method capability.

    Missing entries (``mask == False`` or NaN) are excluded from the moments.
    """
    Xc = np.asarray(X, dtype=float)
    if mask is not None:
        Xc = np.where(mask, Xc, np.nan)
    mu = np.nanmean(Xc, axis=0, keepdims=True)
    sd = np.nanstd(Xc, axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (Xc - mu) / sd
