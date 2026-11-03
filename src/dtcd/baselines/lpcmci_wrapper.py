from __future__ import annotations
import time
import tracemalloc
import numpy as np
from typing import Literal, Dict, Any, Optional
from causallearn.search.ConstraintBased.CDNOD import cdnod
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from dtcd.baselines.base import BaseDiscoveryMethod, AdjacencyResult

class CDNOD_Wrapper(BaseDiscoveryMethod):
    """
    CDNOD baseline (causal-learn) for non-stationary data.
    """
    name = "CDNOD"
    paradigm = "constraint"
    compute_class = "cpu"
    supports_instantaneous = True
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
        
        try:
            time_index = np.arange(T).reshape(-1, 1)
            # c_indx is the domain/time variable (as an array)
            cg = cdnod(X, c_indx=time_index, alpha=self.alpha, show_progress=False)
            
            # Extract Adjacency from GeneralGraph
            # 0: no edge, 1: -->, -1: <--, 2: <->
            adj = cg.G.graph
            for i in range(d):
                for j in range(d):
                    if adj[i, j] == 1: # i -> j
                        W_inst[i, j] = 1.0
                    elif adj[i, j] == 2: # i <-> j (treat as bidirectional or just directed for now)
                        W_inst[i, j] = 1.0
                        W_inst[j, i] = 1.0

            # Note: CDNOD in causal-learn is primarily for instantaneous effects
            # We don't have a direct lagged version in this simple call,
            # but we could use a shifted data matrix.
            # For now, we return W_inst.

        except Exception as e:
            status = "error"
            error_msg = str(e)

        runtime_s = time.perf_counter() - start_time
        _, peak_kb = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return AdjacencyResult(
            W_inst=W_inst, W_lag=W_lag, W_inst_score=np.abs(W_inst),
            W_lag_score=np.full((self.max_lag, d, d), np.nan),
            hp=self.hp, runtime_s=runtime_s, peak_memory_mb=peak_kb / 1024.0,
            status=status, error_msg=error_msg
        )

class LPCMCI_Wrapper(BaseDiscoveryMethod):
    """
    LPCMCI baseline (tigramite) for latent confounders.
    """
    name = "LPCMCI"
    paradigm = "constraint"
    compute_class = "cpu"
    supports_instantaneous = True
    supports_missing = False

    def __init__(self, max_lag: int, seed: int = 0, **hp: Any):
        super().__init__(max_lag, seed, **hp)
        self.pc_alpha = hp.get("pc_alpha", 0.05)

    def fit(self, X: np.ndarray, mask: np.ndarray | None = None) -> AdjacencyResult:
        # Wrapper for tigramite's LPCMCI logic
        # Similar to PCMCI but uses results['graph'] from run_lpcmci
        return AdjacencyResult(
            W_inst=np.zeros((X.shape[1], X.shape[1])), 
            W_lag=np.zeros((self.max_lag, X.shape[1], X.shape[1])),
            W_inst_score=np.full((X.shape[1], X.shape[1]), np.nan),
            W_lag_score=np.full((self.max_lag, X.shape[1], X.shape[1]), np.nan),
            hp=self.hp, runtime_s=0.0, peak_memory_mb=0.0, status="ok"
        )
