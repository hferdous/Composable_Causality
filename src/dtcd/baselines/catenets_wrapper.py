from __future__ import annotations
import time
import tracemalloc
import numpy as np
from typing import Literal, Dict, Any, Optional
from catenets.models.jax import TARNet, SNet, FlexTENet
from dtcd.baselines.base import BaseCATEMethod, CATEResult

class CATENets_Wrapper(BaseCATEMethod):
    """
    Wrapper for catenets (TARNet, SNet, FlexTENet).
    """
    compute_class = "cpu"
    supports_missing = False

    def __init__(self, seed: int = 0, model_type: str = "TARNet", **hp: Any):
        super().__init__(seed, **hp)
        self.model_type = model_type
        self.name = model_type
        
        # catenets models take their own `seed`; without this every evaluation seed
        # used catenets' fixed default. An explicit `seed` in hp still wins.
        hp = {"seed": seed, **hp}

        # Initialize model
        if model_type == "TARNet":
            self.model = TARNet(**hp)
        elif model_type == "SNet":
            self.model = SNet(**hp)
        elif model_type == "FlexTENet":
            self.model = FlexTENet(**hp)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def fit(self, X: np.ndarray, W: np.ndarray, Y: np.ndarray) -> None:
        # catenets uses (X, y, w) order
        self.model.fit(X, Y, W)

    def predict(self, X: np.ndarray) -> CATEResult:
        tracemalloc.start()
        start_time = time.perf_counter()
        
        status: Literal["ok", "error"] = "ok"
        error_msg = None
        cate_pred = np.zeros(X.shape[0])

        try:
            cate_pred = self.model.predict(X)
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
