from __future__ import annotations
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score, mean_squared_error
from typing import Tuple, Optional

def _unroll_adj(W_inst: np.ndarray, W_lag: np.ndarray) -> np.ndarray:
    """
    Unrolls a (d, d) instantaneous and (max_lag, d, d) lagged adjacency 
    into a joint (d * (max_lag + 1), d) representation.
    """
    d = W_inst.shape[0]
    max_lag = W_lag.shape[0]
    # W_joint has parents (rows) from t, t-1, ..., t-max_lag and targets (cols) at t
    # Shape: (d * (max_lag + 1), d)
    # Rows 0:d are t (instantaneous)
    # Rows d:2d are t-1
    # ...
    joint = np.zeros((d * (max_lag + 1), d))
    joint[:d, :] = W_inst
    for k in range(max_lag):
        joint[(k+1)*d:(k+2)*d, :] = W_lag[k]
    return joint

def shd(W_pred: np.ndarray, W_true: np.ndarray) -> float:
    """
    Structural Hamming Distance.
    """
    return float(np.sum(np.abs((W_pred != 0).astype(int) - (W_true != 0).astype(int))))

def normalized_shd(W_pred: np.ndarray, W_true: np.ndarray) -> float:
    """
    SHD normalized by the number of true edges.
    """
    n_edges = np.sum(W_true != 0)
    if n_edges == 0:
        return float(np.sum(W_pred != 0))
    return shd(W_pred, W_true) / n_edges

def get_n_edges(W: np.ndarray) -> int:
    return int(np.sum(W != 0))

def sid(W_pred: np.ndarray, W_true: np.ndarray) -> int:
    """
    Structural Intervention Distance. 
    Range: [0, d(d-1)] ↓.
    Requires 'cdt' package or custom implementation.
    """
    try:
        from cdt.metrics import SID
        return int(SID(W_true, W_pred))
    except ImportError:
        return -1 # Fallback or warning

def f1_skeleton(W_pred: np.ndarray, W_true: np.ndarray) -> float:
    """
    F1 score on the undirected skeleton.
    Range: [0, 1] ↑.
    """
    if W_true.shape[0] != W_true.shape[1]:
        # For non-square (unrolled) graphs, skeleton is less standard.
        # We'll treat it as oriented or return NaN.
        return f1_oriented(W_pred, W_true)
        
    y_true = ((W_true != 0) | (W_true.T != 0)).flatten().astype(int)
    y_pred = ((W_pred != 0) | (W_pred.T != 0)).flatten().astype(int)
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return float(f1)

def f1_oriented(W_pred: np.ndarray, W_true: np.ndarray) -> float:
    """
    F1 score on directed edges.
    Range: [0, 1] ↑.
    """
    y_true = (W_true != 0).flatten().astype(int)
    y_pred = (W_pred != 0).flatten().astype(int)
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return float(f1)

def precision_recall(W_pred: np.ndarray, W_true: np.ndarray) -> Tuple[float, float]:
    """
    Precision and Recall for directed edges.
    """
    y_true = (W_true != 0).flatten().astype(int)
    y_pred = (W_pred != 0).flatten().astype(int)
    p, r, _, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return float(p), float(r)

def auroc_edges(W_score: np.ndarray, W_true: np.ndarray) -> float:
    """
    AUROC for edge prediction using confidence scores.
    Range: [0, 1] ↑.
    """
    y_true = (W_true != 0).flatten().astype(int)
    y_score = W_score.flatten()
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))

def auprc(W_score: np.ndarray, W_true: np.ndarray) -> float:
    """
    AUPRC for edge prediction.
    """
    y_true = (W_true != 0).flatten().astype(int)
    y_score = W_score.flatten()
    
    # Handle NaNs (missing scores)
    y_score = np.nan_to_num(y_score, nan=0.0)
    
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))

def rmse_weights(W_pred: np.ndarray, W_true: np.ndarray) -> float:
    """
    Weight RMSE computed only on true non-zero entries.
    Range: [0, inf) ↓.
    """
    mask = W_true != 0
    if not np.any(mask):
        return 0.0
    return float(np.sqrt(np.mean((W_pred[mask] - W_true[mask])**2)))

def pehe(tau_pred: np.ndarray, tau_true: np.ndarray) -> float:
    """
    Precision in Estimation of Heterogeneous Effect (PEHE).
    Range: [0, inf) ↓.
    """
    return float(np.sqrt(np.mean((tau_pred - tau_true)**2)))

def ate_abs_error(tau_pred: np.ndarray, tau_true: np.ndarray) -> float:
    """
    Absolute error on Average Treatment Effect (ATE).
    Range: [0, inf) ↓.
    """
    return float(np.abs(np.mean(tau_pred) - np.mean(tau_true)))
