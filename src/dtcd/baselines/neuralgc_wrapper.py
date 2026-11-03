from __future__ import annotations
import time
import tracemalloc
import numpy as np
import torch
from typing import Literal, Dict, Any, Optional
from dtcd.baselines.base import BaseDiscoveryMethod, AdjacencyResult

class NeuralGC(BaseDiscoveryMethod):
    """
    Neural Granger causality baseline (Tank et al., 2022), cMLP and cLSTM.
    Interface stub; no implementation is bundled in this release.
    """
    paradigm = "functional"
    compute_class = "gpu"
    supports_instantaneous = False
    supports_missing = False

    def __init__(self, max_lag: int, seed: int = 0, **hp: Any):
        super().__init__(max_lag, seed, **hp)
        self.model_type = hp.get("model_type", "cMLP")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def fit(self, X: np.ndarray, mask: np.ndarray | None = None) -> AdjacencyResult:
        """
        Assumptions: Non-linear Granger causality.
        Complexity: O(Epochs * T * d^2).
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

        # Interface stub: cMLP/cLSTM training is not implemented in this
        # release, so report an error rather than an empty graph.
        status = "error"
        error_msg = ("{name} is an interface stub in this release: no implementation is bundled, so it returns no graph. It was not used in the paper.").format(name="NeuralGC")

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
