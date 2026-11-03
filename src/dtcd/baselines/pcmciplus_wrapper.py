from __future__ import annotations
import time
import tracemalloc
import numpy as np
from typing import Literal, Dict, Any
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

from dtcd.baselines.base import BaseDiscoveryMethod, AdjacencyResult

class PCMCIplus_ParCorr(BaseDiscoveryMethod):
    """
    Wrapper for PCMCI+ via tigramite, supporting instantaneous edges.
    """
    name = "PCMCIplus_ParCorr"
    paradigm = "constraint"
    compute_class = "cpu"
    supports_instantaneous = True
    supports_missing = False

    def __init__(self, max_lag: int, seed: int = 0, **hp: Any):
        super().__init__(max_lag, seed, **hp)
        self.pc_alpha = hp.get("pc_alpha", 0.05)
        self.cond_ind_test = ParCorr()

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
            dataframe = pp.DataFrame(X)
            pcmci = PCMCI(dataframe=dataframe, cond_ind_test=self.cond_ind_test, verbosity=0)
            
            # Run PCMCI+
            results = pcmci.run_pcmciplus(tau_max=self.max_lag, pc_alpha=self.pc_alpha)
            
            graph = results['graph'] # (d, d, max_lag + 1)
            p_matrix = results['p_matrix']
            
            # PCMCI+ graph convention:
            # graph[i, j, 0] == '-->' means i -> j instantaneous
            # graph[i, j, k] == '-->' means i(t-k) -> j(t)
            
            for i in range(d):
                for j in range(d):
                    # Instantaneous
                    p_val_inst = p_matrix[i, j, 0]
                    W_inst_score[i, j] = 1.0 - p_val_inst
                    if graph[i, j, 0] == '-->':
                        W_inst[i, j] = 1.0
                    
                    # Lagged
                    for k in range(1, self.max_lag + 1):
                        p_val_lag = p_matrix[i, j, k]
                        W_lag_score[k-1, i, j] = 1.0 - p_val_lag
                        if graph[i, j, k] == '-->':
                            W_lag[k-1, i, j] = 1.0
                            
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
