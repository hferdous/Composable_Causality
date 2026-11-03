import numpy as np

def extract_adjacency_matrix(links: dict, n_vars: int, max_lag: int, format_type: str = 'tensor') -> np.ndarray:
    """
    Extracts an adjacency matrix from the links dictionary.
    
    links: dictionary {(source, lag, target): weight} or {(source, lag, target, power): weight}
    
    format_type: 
    - 'tensor': Returns shape (max_lag + 1, n_vars, n_vars) where A[lag, source, target] = weight
    - 'flat': Returns flattened (max_lag + 1) * n_vars x n_vars, typically used in unrolled DAGs
              where columns are targets at time t, rows are sources at t, t-1 ... t-max_lag.
    """
    tensor_adj = np.zeros((max_lag + 1, n_vars, n_vars))
    
    for key, weight in links.items():
        if len(key) == 3:
            src, lag, tgt = key
        else:
            src, lag, tgt, _ = key
            
        if isinstance(src, str) and src.startswith('X'):
            src_idx = int(src[1:]) - 1
        elif isinstance(src, int):
            src_idx = src
        else:
            continue
            
        if isinstance(tgt, str) and tgt.startswith('X'):
            tgt_idx = int(tgt[1:]) - 1
        elif isinstance(tgt, int):
            tgt_idx = tgt
        else:
            continue
            
        tensor_adj[abs(lag), src_idx, tgt_idx] = weight
        
    if format_type == 'tensor':
        return tensor_adj
    elif format_type == 'flat':
        # Unrolled DAG
        flat_adj = np.zeros((n_vars * (max_lag + 1), n_vars))
        for lag in range(max_lag + 1):
            flat_adj[lag * n_vars : (lag + 1) * n_vars, :] = tensor_adj[lag]
        return flat_adj
    else:
        raise ValueError(f"Invalid format_type: {format_type}")

def to_tigramite_links(links: dict, n_vars: int, max_lag: int) -> dict:
    """
    Converts DTCD links to Tigramite's link dictionary format.
    Tigramite format is typ. val_matrix[i, j, lag] which can be derived from the tensor.
    """
    tensor_adj = extract_adjacency_matrix(links, n_vars, max_lag, format_type='tensor')
    # Can populate Tigramite val_matrix directly from this tensor_adj shape
    return {'val_matrix': tensor_adj.transpose(1, 2, 0)} # shape (n_vars, n_vars, max_lag+1)

def to_causal_learn_matrix(links: dict, n_vars: int, max_lag: int) -> np.ndarray:
    """
    Converts DTCD links to a 2D adjacency matrix for causal-learn.
    Often causal-learn uses an unrolled adjacency matrix or summary graph.
    Returns the flattened unrolled DAG adjacency.
    """
    return extract_adjacency_matrix(links, n_vars, max_lag, format_type='flat')
