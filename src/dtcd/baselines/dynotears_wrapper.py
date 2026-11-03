from __future__ import annotations
import time
import tracemalloc
import numpy as np
import pandas as pd
from dtcd.baselines.dynotears_core import from_numpy_dynamic
from typing import Literal, Dict, Any, Optional
from dtcd.baselines.base import BaseDiscoveryMethod, AdjacencyResult

class DYNOTEARS(BaseDiscoveryMethod):
    """
    DYNOTEARS baseline (Pamfil et al., 2020). The solver is vendored from
    CausalNex (Apache-2.0) in dynotears_core.py, so CausalNex is not a
    dependency. Learns a dynamic Bayesian network by continuous optimization.
    """
    name = "DYNOTEARS"
    paradigm = "score"
    compute_class = "cpu"
    supports_instantaneous = True
    supports_missing = False

    def __init__(self, max_lag: int, seed: int = 0, **hp: Any):
        super().__init__(max_lag, seed, **hp)
        self.lambda_w = hp.get("lambda_w", 0.1)
        self.lambda_a = hp.get("lambda_a", 0.1)
        self.w_threshold = hp.get("w_threshold", 0.1)

    def fit(self, X: np.ndarray, mask: np.ndarray | None = None) -> AdjacencyResult:
        """
        Assumptions: Linearity, No Latent Confounders, Causal Sufficiency.
        Complexity: O(T * d^3) per iteration.
        Input: (T, d), Output: AdjacencyResult
        """
        T, d = X.shape
        tracemalloc.start()
        start_time = time.perf_counter()
        
        status: Literal["ok", "error"] = "ok"
        error_msg = None
        
        W_inst = np.zeros((d, d))
        W_lag = np.zeros((self.max_lag, d, d))
        W_inst_score = np.full((d, d), np.nan)
        W_lag_score = np.full((self.max_lag, d, d), np.nan)

        try:
            p = self.max_lag
            if T <= p:
                raise ValueError("Time series length must be greater than max_lag.")
            
            X_target = X[p:, :]
            Xlags = np.hstack([X[p-k : T-k, :] for k in range(1, p + 1)])
            
            w_est, a_est = from_numpy_dynamic(
                X=X_target, 
                Xlags=Xlags, 
                lambda_w=self.lambda_w, 
                lambda_a=self.lambda_a, 
                w_threshold=self.w_threshold
            )
            
            # Extract weights
            W_inst = w_est
            W_inst_score = np.abs(w_est)
            
            for i in range(a_est.shape[0]):
                for j in range(a_est.shape[1]):
                    weight = a_est[i, j]
                    if weight != 0:
                        u = i % d
                        k = (i // d) + 1
                        W_lag[k-1, u, j] = weight
                        W_lag_score[k-1, u, j] = abs(weight)
                            
        except Exception as e:
            status = "error"
            error_msg = str(e)

        runtime_s = time.perf_counter() - start_time
        _, peak_kb = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return AdjacencyResult(
            W_inst=W_inst,
            W_lag=W_lag,
            W_inst_score=W_inst_score,
            W_lag_score=W_lag_score,
            hp=self.hp,
            runtime_s=runtime_s,
            peak_memory_mb=peak_kb / 1024.0,
            status=status,
            error_msg=error_msg
        )
