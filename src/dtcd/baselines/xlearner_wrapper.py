from __future__ import annotations
import time
import tracemalloc
import numpy as np
from econml.metalearners import XLearner
from sklearn.ensemble import RandomForestRegressor
from typing import Literal, Dict, Any, Optional
from dtcd.baselines.base import BaseCATEMethod, CATEResult

class XLearnerWrapper(BaseCATEMethod):
    """
    Wrapper for X-Learner via EconML.
    """
    name = "XLearner"
    compute_class = "cpu"
    supports_missing = False

    def __init__(self, seed: int = 0, **hp: Any):
        super().__init__(seed, **hp)
        self.model = XLearner(
            models=RandomForestRegressor(random_state=seed),
            **hp
        )

    def fit(self, X: np.ndarray, W: np.ndarray, Y: np.ndarray) -> None:
        self.model.fit(Y, W, X=X)

    def predict(self, X: np.ndarray) -> CATEResult:
        tracemalloc.start()
        start_time = time.perf_counter()
        
        status: Literal["ok", "error"] = "ok"
        error_msg = None
        cate_pred = np.zeros(X.shape[0])

        try:
            cate_pred = self.model.const_marginal_effect(X)
        except Exception as e:
            status = "error"
            error_msg = str(e)

        runtime_s = time.perf_counter() - start_time
        _, peak_kb = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return CATEResult(
            cate_pred=cate_pred.flatten(),
            hp=self.hp,
            runtime_s=runtime_s,
            peak_memory_mb=peak_kb / 1024.0,
            status=status,
            error_msg=error_msg
        )
