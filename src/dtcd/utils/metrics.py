import numpy as np

def compute_shd(true_adj: np.ndarray, pred_adj: np.ndarray) -> int:
    """
    Computes a simple Structural Hamming Distance (SHD) between two graph adjacency matrices.
    If given tensors (lags, vars, vars), it squashes them or computes element-wise.
    """
    assert true_adj.shape == pred_adj.shape, "Shapes of true and predicted adjacencies must match."
    
    true_bin = (np.abs(true_adj) > 1e-5).astype(int)
    pred_bin = (np.abs(pred_adj) > 1e-5).astype(int)
    
    # Basic structural mismath: missing edges + extra edges
    # For full DAG SHD, one typically accounts for undirected/reversed edges.
    # Because time-series DAGs are fully directed (by time), element-wise difference is adequate.
    return int(np.sum(np.abs(true_bin - pred_bin)))

def compute_f1_score(true_adj: np.ndarray, pred_adj: np.ndarray) -> float:
    """
    Computes the F1-score for edge identification.
    """
    assert true_adj.shape == pred_adj.shape, "Shapes of true and predicted adjacencies must match."
    
    true_bin = (np.abs(true_adj) > 1e-5).astype(int)
    pred_bin = (np.abs(pred_adj) > 1e-5).astype(int)
    
    tp = np.sum((true_bin == 1) & (pred_bin == 1))
    fp = np.sum((true_bin == 0) & (pred_bin == 1))
    fn = np.sum((true_bin == 1) & (pred_bin == 0))
    
    if tp == 0:
        return 0.0
        
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    return 2 * (precision * recall) / (precision + recall)
