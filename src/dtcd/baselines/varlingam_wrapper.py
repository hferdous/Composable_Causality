from __future__ import annotations
import time
import tracemalloc
import numpy as np
import lingam
from typing import Literal, Dict, Any
from dtcd.baselines.base import BaseDiscoveryMethod, AdjacencyResult

class VARLiNGAM(BaseDiscoveryMethod):
    """
    VARLiNGAM baseline (Hyvarinen et al. 2010).
    Identifies SVAR models with non-Gaussian noise.
    """
    name = "VARLiNGAM"
    paradigm = "granger"
    compute_class = "cpu"
    supports_instantaneous = True
    supports_missing = False

    def __init__(self, max_lag: int, seed: int = 0, **hp: Any):
        super().__init__(max_lag, seed, **hp)
        self.lags = hp.get("lags", max_lag)
        self.criterion = hp.get("criterion", "bic")

    def fit(self, X: np.ndarray, mask: np.ndarray | None = None) -> AdjacencyResult:
        """
        Assumptions: Causal Sufficiency, Non-Gaussian Noise, Linearity, Stationarity.
        Complexity: O(p * d^3) where p is lags.
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
            # Numerical Stability Fix: Standardize inputs
            from sklearn.preprocessing import StandardScaler
            X_scaled = StandardScaler().fit_transform(X)
            
            model = lingam.VARLiNGAM(lags=self.lags, criterion=self.criterion, random_state=self.seed)
            model.fit(X_scaled)
            
            # Extract matrices
            # lingam returns [B0, B1, ..., Bp]
            # B0 is instantaneous (d, d)
            # Bi are lagged (d, d)
            adj_matrices = model.adjacency_matrices_
            
            # Note: lingam adjacency is (target, source)? Let's check or assume source -> target
            # Standard lingam convention: W[i, j] means j -> i. 
            # DTCD convention: W[i, j] means i -> j.
            
            # Instantaneous
            W_inst = adj_matrices[0].T # Transpose to match DTCD i -> j
            
            # Lags
            for k in range(1, min(len(adj_matrices), self.max_lag + 1)):
                W_lag[k-1] = adj_matrices[k].T
                
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
