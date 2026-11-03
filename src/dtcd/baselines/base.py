from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Dict, Any, Optional
import numpy as np

@dataclass
class AdjacencyResult:
    """
    Unified result object for causal discovery methods.
    """
    W_inst: np.ndarray         # (d, d), binary or weighted; W[i,j] != 0 means i -> j
    W_lag: np.ndarray          # (max_lag, d, d); W[k-1,i,j] != 0 means X_{t-k,i} -> X_{t,j}
    W_inst_score: np.ndarray   # (d, d) real-valued confidence; NaN if unsupported
    W_lag_score: np.ndarray    # (max_lag, d, d); NaN if unsupported
    hp: Dict[str, Any]         # final hyperparameters used
    runtime_s: float
    peak_memory_mb: float
    status: Literal["ok", "timeout", "oom", "error"]
    error_msg: str | None = None

class BaseDiscoveryMethod(ABC):
    """
    Abstract base class for Causal Discovery Baselines.
    """
    name: str
    paradigm: Literal["granger", "constraint", "score", "functional", "amortized", "control"]
    compute_class: Literal["cpu", "gpu"]
    supports_instantaneous: bool
    supports_missing: bool

    def __init__(self, max_lag: int, seed: int = 0, **hp: Any):
        self.max_lag = max_lag
        self.seed = seed
        self.hp = hp

    @abstractmethod
    def fit(self, X: np.ndarray, mask: np.ndarray | None = None) -> AdjacencyResult:
        """
        Fits the causal discovery method to the data.

        Args:
            X: Input tensor of shape (T, d).
            mask: Optional binary mask of shape (T, d) where 0 indicates missing.

        Returns:
            AdjacencyResult containing predicted graphs and metadata.
        """
        pass

@dataclass
class CATEResult:
    """
    Unified result object for CATE estimation methods.
    """
    cate_pred: np.ndarray      # (N,) or (T, d) predicted effects
    hp: Dict[str, Any]
    runtime_s: float
    peak_memory_mb: float
    status: Literal["ok", "timeout", "oom", "error"]
    error_msg: str | None = None

class BaseCATEMethod(ABC):
    """
    Abstract base class for CATE Estimation Baselines.
    """
    name: str
    compute_class: Literal["cpu", "gpu"]
    supports_missing: bool

    def __init__(self, seed: int = 0, **hp: Any):
        self.seed = seed
        self.hp = hp

    @abstractmethod
    def fit(self, X: np.ndarray, W: np.ndarray, Y: np.ndarray) -> None:
        """
        Fits the CATE model.

        Args:
            X: Covariates / History.
            W: Treatment assignment.
            Y: Observed outcomes.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> CATEResult:
        """
        Predicts CATE for given covariates.

        Args:
            X: Covariates / History.

        Returns:
            CATEResult containing predictions and metadata.
        """
        pass
