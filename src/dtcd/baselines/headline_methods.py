from __future__ import annotations
import time
import tracemalloc
import numpy as np
import pandas as pd
from typing import Literal, Dict, Any, Optional
import torch

from dtcd.baselines.base import BaseDiscoveryMethod, AdjacencyResult


class UnCLe(BaseDiscoveryMethod):
    """
    UnCLe: Unified Causal Learning.
    Placeholder for a deep variational discovery method.
    Interface stub; no implementation is bundled in this release.
    """
    name = "UnCLe"
    paradigm = "amortized"
    compute_class = "gpu"
    supports_instantaneous = True
    supports_missing = True

    def __init__(self, max_lag: int, seed: int = 0, **hp: Any):
        super().__init__(max_lag, seed, **hp)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

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
            W_inst=W_inst, W_lag=W_lag,
            W_inst_score=np.abs(W_inst), W_lag_score=np.abs(W_lag),
            hp=self.hp, runtime_s=runtime_s, peak_memory_mb=peak_kb / 1024.0,
            status="error", error_msg=("{name} is an interface stub in this release: no implementation is bundled, so it returns no graph. It was not used in the paper.").format(name="UnCLe")
        )


class VARLiNGAM_imp(BaseDiscoveryMethod):
    """
    VARLiNGAM_imp: VARLiNGAM with Mean Imputation.
    Baseline for MNAR/missing data experiments.
    """
    name = "VARLiNGAM_imp"
    paradigm = "functional"
    compute_class = "gpu"
    supports_instantaneous = True
    supports_missing = True

    def __init__(self, max_lag: int, seed: int = 0, **hp: Any):
        super().__init__(max_lag, seed, **hp)

    def fit(self, X: np.ndarray, mask: np.ndarray | None = None) -> AdjacencyResult:
        T, d = X.shape
        tracemalloc.start()
        start_time = time.perf_counter()

        from sklearn.impute import SimpleImputer
        from lingam import VARLiNGAM

        status = "ok"
        error_msg = None

        W_inst = np.zeros((d, d))
        W_lag = np.zeros((self.max_lag, d, d))

        try:
            imputer = SimpleImputer(strategy="mean")
            X_imputed = imputer.fit_transform(X)

            model = VARLiNGAM(lags=self.max_lag)
            model.fit(X_imputed)

            if hasattr(model, "adjacency_matrices_"):
                full_adj = model.adjacency_matrices_
            elif hasattr(model, "_adjacency_matrices"):
                full_adj = model._adjacency_matrices
            else:
                raise AttributeError("VARLiNGAM model missing adjacency matrices.")

            # lingam.VARLiNGAM selects the VAR lag order by BIC, so
            # adjacency_matrices_ can be SHORTER than max_lag + 1. Indexing
            # full_adj[k] unconditionally raised IndexError on those cells.
            # Missing lags are genuinely estimated as absent, so leave the
            # corresponding W_lag slices at zero.
            W_inst = full_adj[0].T
            n_available = len(full_adj)
            for k in range(1, self.max_lag + 1):
                if k < n_available:
                    W_lag[k - 1] = full_adj[k].T

        except Exception as e:
            status = "error"
            error_msg = str(e)

        runtime_s = time.perf_counter() - start_time
        _, peak_kb = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return AdjacencyResult(
            W_inst=W_inst, W_lag=W_lag,
            W_inst_score=np.abs(W_inst), W_lag_score=np.abs(W_lag),
            hp=self.hp, runtime_s=runtime_s, peak_memory_mb=peak_kb / 1024.0,
            status=status, error_msg=error_msg
        )


class CUTSPlus(BaseDiscoveryMethod):
    """
    CUTS+: High-dimensional Causal Discovery from Time Series with Missing Data
    (Cheng et al. 2024). Joint neural imputation and discovery.

    Edge thresholding: after gradient descent, coefficients with magnitude
    below `threshold` are zeroed. Without this step the predicted graph is
    essentially fully connected (every gradient residual counts as an edge).
    The threshold is exposed as a hyperparameter for Optuna tuning.
    """
    name = "CUTS+"
    paradigm = "neural"
    compute_class = "gpu"
    supports_instantaneous = True
    supports_missing = True

    def __init__(self, max_lag: int, seed: int = 0, **hp: Any):
        super().__init__(max_lag, seed, **hp)
        self.lr = hp.get("lr", 1e-3)
        self.epochs = hp.get("epochs", 50)
        self.hidden_dim = hp.get("hidden_dim", 32)
        self.lambda_sparsity = hp.get("lambda_sparsity", 0.01)
        self.threshold = hp.get("threshold", 0.1)

    def fit(self, X: np.ndarray, mask: np.ndarray | None = None) -> AdjacencyResult:
        T, d = X.shape
        tracemalloc.start()
        start_time = time.perf_counter()

        import torch.nn as nn
        import torch.optim as optim

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X_imputed = np.nan_to_num(X, nan=np.nanmean(X))
        X_tensor = torch.FloatTensor(X_imputed).to(device)
        mask_tensor = (
            torch.FloatTensor(~np.isnan(X)).to(device)
            if np.isnan(X).any()
            else torch.ones_like(X_tensor)
        )

        class Imputer(nn.Module):
            def __init__(self, d, hidden_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, d),
                )

            def forward(self, x):
                return self.net(x)

        class DiscoveryNet(nn.Module):
            def __init__(self, d, max_lag):
                super().__init__()
                self.d = d
                self.max_lag = max_lag
                self.W_lag = nn.Parameter(torch.randn(max_lag, d, d) * 0.01)
                self.W_inst = nn.Parameter(torch.randn(d, d) * 0.01)

            def forward(self, x):
                out = torch.zeros_like(x)
                for k in range(1, self.max_lag + 1):
                    out[k:] += x[:-k] @ self.W_lag[k - 1]
                eye = torch.eye(self.d, device=x.device)
                out += x @ (self.W_inst * (1 - eye))
                return out

        imputer = Imputer(d, self.hidden_dim).to(device)
        discovery = DiscoveryNet(d, self.max_lag).to(device)
        optimizer = optim.Adam(
            list(imputer.parameters()) + list(discovery.parameters()),
            lr=self.lr,
        )

        status = "ok"
        error_msg = None
        try:
            for _ in range(self.epochs):
                optimizer.zero_grad()
                X_curr = X_tensor * mask_tensor + imputer(X_tensor) * (1 - mask_tensor)
                X_pred = discovery(X_curr)
                loss = nn.MSELoss()(
                    X_pred[self.max_lag:], X_curr[self.max_lag:]
                ) + self.lambda_sparsity * (
                    torch.norm(discovery.W_lag, 1) + torch.norm(discovery.W_inst, 1)
                )
                loss.backward()
                optimizer.step()
        except Exception as e:
            status = "error"
            error_msg = str(e)

        W_inst_cont = discovery.W_inst.detach().cpu().numpy() * (1 - np.eye(d))
        W_lag_cont = discovery.W_lag.detach().cpu().numpy()

        # Hard threshold to recover a sparse adjacency. Coefficients below
        # `self.threshold` in magnitude are zeroed; those above are retained
        # at their continuous value.
        W_inst = np.where(np.abs(W_inst_cont) >= self.threshold, W_inst_cont, 0.0)
        W_lag = np.where(np.abs(W_lag_cont) >= self.threshold, W_lag_cont, 0.0)

        runtime_s = time.perf_counter() - start_time
        _, peak_kb = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return AdjacencyResult(
            W_inst=W_inst,
            W_lag=W_lag,
            W_inst_score=np.abs(W_inst_cont),
            W_lag_score=np.abs(W_lag_cont),
            hp=self.hp,
            runtime_s=runtime_s,
            peak_memory_mb=peak_kb / 1024.0,
            status=status,
            error_msg=error_msg,
        )
