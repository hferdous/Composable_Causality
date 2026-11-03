"""Variance-sorting control baselines (Reisach, Seiler & Weichwald 2021).

`SortnRegress` is not a causal discovery method. It is a diagnostic: it orders
the contemporaneous variables by increasing marginal variance and fits an
adaptive lasso of each variable on its predecessors. If it scores close to a
real discovery method on a benchmark, that benchmark's difficulty is largely an
artifact of variance accumulation rather than a test of causal reasoning.

`RandomRegress` is the matched control. It is identical except the
contemporaneous order is a random permutation, so it cannot exploit variance.
The gap between the two isolates the varsortability channel; the gap between a
real method and RandomRegress is what that method contributes beyond the
shortcut.

Both use the time index for free, as any time-series method does: all lagged
variables are candidate parents regardless of ordering. Only the
contemporaneous ordering is at issue.
"""
from __future__ import annotations

import time
import tracemalloc
from typing import Any, Literal

import numpy as np
from sklearn.linear_model import LassoLarsIC, LinearRegression

from dtcd.baselines.base import AdjacencyResult, BaseDiscoveryMethod


def _adaptive_lasso(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Adaptive lasso with BIC model selection, as used by sortnregress."""
    lr = LinearRegression().fit(A, y)
    weights = 1.0 / (np.abs(lr.coef_) + 1e-10)
    ll = LassoLarsIC(criterion="bic").fit(A / weights, y)
    return ll.coef_ / weights


def _order_regress(X: np.ndarray, max_lag: int, order):
    """Fit each target on its contemporaneous predecessors (per `order`) and all lags."""
    T, d = X.shape
    W_inst = np.zeros((d, d))
    W_lag = np.zeros((max_lag, d, d))
    X_t = X[max_lag:]
    lagged = [X[max_lag - k: T - k] for k in range(1, max_lag + 1)]

    for pos, i in enumerate(order):
        keys, cols = [], []
        for j in order[:pos]:
            keys.append(("inst", 0, j))
            cols.append(X_t[:, j])
        for k in range(1, max_lag + 1):
            for j in range(d):
                keys.append(("lag", k, j))
                cols.append(lagged[k - 1][:, j])
        if not cols:
            continue
        coef = _adaptive_lasso(np.column_stack(cols), X_t[:, i])
        for (kind, k, j), c in zip(keys, coef):
            if c != 0:
                if kind == "inst":
                    W_inst[j, i] = c
                else:
                    W_lag[k - 1, j, i] = c
    return W_inst, W_lag


class _BaseRegressControl(BaseDiscoveryMethod):
    paradigm = "control"
    compute_class = "cpu"
    supports_instantaneous = True
    supports_missing = False

    def _order(self, X: np.ndarray):
        raise NotImplementedError

    def fit(self, X: np.ndarray, mask: np.ndarray | None = None) -> AdjacencyResult:
        T, d = X.shape
        tracemalloc.start()
        start = time.perf_counter()
        status: Literal["ok", "error"] = "ok"
        error_msg = None
        W_inst = np.zeros((d, d))
        W_lag = np.zeros((self.max_lag, d, d))
        try:
            W_inst, W_lag = _order_regress(X, self.max_lag, self._order(X))
        except Exception as e:  # noqa: BLE001
            status = "error"
            error_msg = str(e)
        runtime = time.perf_counter() - start
        _, peak_kb = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return AdjacencyResult(
            W_inst=W_inst,
            W_lag=W_lag,
            W_inst_score=np.abs(W_inst),
            W_lag_score=np.abs(W_lag),
            hp=self.hp,
            runtime_s=runtime,
            peak_memory_mb=peak_kb / 1024.0,
            status=status,
            error_msg=error_msg,
        )


class SortnRegress(_BaseRegressControl):
    """Contemporaneous order taken from increasing marginal variance."""

    name = "SortnRegress"

    def _order(self, X: np.ndarray):
        return list(np.argsort(np.var(X, axis=0)))


class RandomRegress(_BaseRegressControl):
    """Matched control with a random contemporaneous order."""

    name = "RandomRegress"

    def __init__(self, max_lag: int, seed: int = 0, **hp: Any):
        super().__init__(max_lag, seed, **hp)

    def _order(self, X: np.ndarray):
        return list(np.random.RandomState(self.seed).permutation(X.shape[1]))
