from __future__ import annotations
import time
import tracemalloc
import numpy as np
import torch
from typing import Literal, Dict, Any, Optional
from dtcd.baselines.base import BaseDiscoveryMethod, BaseCATEMethod, AdjacencyResult, CATEResult

# These wrappers are interface stubs: the authors' implementations are not
# bundled, so fit()/predict() return status="error" instead of a fabricated
# result. None of them is used in the paper. To add one, vendor the original
# code and fill in fit(); see docs/MANUAL.md, Section 8.4.
STUB_MSG = ("{name} is an interface stub in this release: no implementation is bundled, so it returns no graph. It was not used in the paper.")

class TCDF_Wrapper(BaseDiscoveryMethod):
    """
    TCDF baseline (Nauta et al. 2019): attention-based dilated causal CNNs.
    Interface stub; no implementation is bundled in this release.
    """
    name = "TCDF"
    paradigm = "functional"
    compute_class = "gpu"
    supports_instantaneous = False
    supports_missing = False

    def __init__(self, max_lag: int, seed: int = 0, **hp: Any):
        super().__init__(max_lag, seed, **hp)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, X: np.ndarray, mask: np.ndarray | None = None) -> AdjacencyResult:
        T, d = X.shape
        tracemalloc.start()
        start_time = time.perf_counter()
        
        W_inst = np.zeros((d, d))
        W_lag = np.zeros((self.max_lag, d, d))
        

        runtime_s = time.perf_counter() - start_time
        _, peak_kb = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return AdjacencyResult(
            W_inst=W_inst, W_lag=W_lag, W_inst_score=np.full((d, d), np.nan),
            W_lag_score=np.full((self.max_lag, d, d), np.nan),
            hp=self.hp, runtime_s=runtime_s, peak_memory_mb=peak_kb / 1024.0,
            status="error", error_msg=STUB_MSG.format(name=self.name)
        )

class ACD_Wrapper(BaseDiscoveryMethod):
    """
    ACD baseline (Löwe et al. 2022): amortized causal discovery with GNNs.
    Interface stub; no implementation is bundled in this release.
    """
    name = "ACD"
    paradigm = "amortized"
    compute_class = "gpu"
    supports_instantaneous = True
    supports_missing = False

    def __init__(self, max_lag: int, seed: int = 0, **hp: Any):
        super().__init__(max_lag, seed, **hp)

    def fit(self, X: np.ndarray, mask: np.ndarray | None = None) -> AdjacencyResult:
        return AdjacencyResult(
            W_inst=np.zeros((X.shape[1], X.shape[1])),
            W_lag=np.zeros((self.max_lag, X.shape[1], X.shape[1])),
            W_inst_score=np.full((X.shape[1], X.shape[1]), np.nan),
            W_lag_score=np.full((self.max_lag, X.shape[1], X.shape[1]), np.nan),
            hp=self.hp, runtime_s=0.0, peak_memory_mb=0.0,
            status="error", error_msg=STUB_MSG.format(name=self.name)
        )

class CausalTransformerWrapper(BaseCATEMethod):
    """
    Causal Transformer baseline (Melnychuk et al. 2022) for time-varying treatments.
    Interface stub; no implementation is bundled in this release.
    """
    name = "CausalTransformer"
    compute_class = "gpu"
    supports_missing = False

    def __init__(self, seed: int = 0, **hp: Any):
        super().__init__(seed, **hp)

    def fit(self, X: np.ndarray, W: np.ndarray, Y: np.ndarray) -> None:
        return None

    def predict(self, X: np.ndarray) -> CATEResult:
        return CATEResult(
            cate_pred=np.full(X.shape[0], np.nan),
            hp=self.hp, runtime_s=0.0, peak_memory_mb=0.0,
            status="error", error_msg=STUB_MSG.format(name=self.name)
        )
