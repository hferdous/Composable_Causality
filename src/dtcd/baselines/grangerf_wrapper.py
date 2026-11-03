from __future__ import annotations
import time
import tracemalloc
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from typing import Literal, Dict, Any
from dtcd.baselines.base import BaseDiscoveryMethod, AdjacencyResult

class GrangerF(BaseDiscoveryMethod):
    """
    Classical Granger Causality baseline using F-tests on VAR models.
    """
    name = "GrangerF"
    paradigm = "granger"
    compute_class = "cpu"
    supports_instantaneous = False
    supports_missing = False

    def __init__(self, max_lag: int, seed: int = 0, **hp: Any):
        super().__init__(max_lag, seed, **hp)
        self.alpha = hp.get("alpha", 0.05)

    def fit(self, X: np.ndarray, mask: np.ndarray | None = None) -> AdjacencyResult:
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
            model = VAR(X)
            results = model.fit(maxlags=self.max_lag, ic=None)
            
            # Bonferroni correction for d*(d-1) tests
            adj_alpha = self.alpha / (d * (d - 1)) if d > 1 else self.alpha
            
            for j in range(d):
                for i in range(d):
                    if i == j: continue
                    test_res = results.test_causality(caused=j, causing=i, kind='f')
                    p_val = test_res.pvalue
                    
                    if p_val < adj_alpha:
                        # Mark ALL lags for this pair as causal (standard Granger)
                        W_lag[:, i, j] = 1.0
                            
        except Exception as e:
            status = "error"
            error_msg = str(e)

        runtime_s = time.perf_counter() - start_time
        _, peak_kb = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return AdjacencyResult(
            W_inst=W_inst, W_lag=W_lag,
            W_inst_score=W_inst_score, W_lag_score=W_lag_score,
            hp=self.hp, runtime_s=runtime_s, peak_memory_mb=peak_kb / 1024.0, 
            status=status, error_msg=error_msg
        )
