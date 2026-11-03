from __future__ import annotations
import time
import tracemalloc
import numpy as np
import pandas as pd
from typing import Literal, Dict, Any, Optional
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.cmiknn import CMIknn

from dtcd.baselines.base import BaseDiscoveryMethod, AdjacencyResult

class PCMCI_Base(BaseDiscoveryMethod):
    """
    Wrapper for PCMCI algorithms via tigramite.
    """
    paradigm = "constraint"
    supports_instantaneous = False # Standard PCMCI is lagged only
    supports_missing = False

    def __init__(self, max_lag: int, seed: int = 0, ci_test: str = "ParCorr", **hp: Any):
        super().__init__(max_lag, seed, **hp)
        self.ci_test_name = ci_test
        self.pc_alpha = hp.get("pc_alpha", 0.05)
        
        # Select CI test
        if ci_test == "ParCorr":
            self.cond_ind_test = ParCorr()
            self.compute_class = "cpu"
        elif ci_test == "GPDC":
            self.cond_ind_test = GPDC()
            self.compute_class = "cpu"
        elif ci_test == "CMIknn":
            self.cond_ind_test = CMIknn()
            self.compute_class = "gpu" # Tagged as heavy
        else:
            raise ValueError(f"Unknown CI test: {ci_test}")

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
            # Tigramite dataframe
            dataframe = pp.DataFrame(X)
            pcmci = PCMCI(dataframe=dataframe, cond_ind_test=self.cond_ind_test, verbosity=0)
            
            # Run PCMCI
            results = pcmci.run_pcmci(tau_max=self.max_lag, pc_alpha=self.pc_alpha)
            
            # results['graph'] contains the orientation
            # results['p_matrix'] contains p-values
            p_matrix = results['p_matrix'] # (d, d, max_lag + 1)
            
            # Tigramite index convention: (source, target, lag)
            # p_matrix[i, j, k] is p-val for X_{i, t-k} -> X_{j, t}
            
            for i in range(d):
                for j in range(d):
                    for k in range(1, self.max_lag + 1):
                        p_val = p_matrix[i, j, k]
                        W_lag_score[k-1, i, j] = 1.0 - p_val
                        if p_val < self.pc_alpha:
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
