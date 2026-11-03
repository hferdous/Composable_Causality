import numpy as np
import random

def generate_random_dag(n_vars, n_edges, max_lag, random_state=None):
    """
    Generate a random DAG for time series causal discovery.
    
    Returns:
        links: dict of {(source, lag, target): weight}
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    links = {}
    possible_links = []
    
    # Generate all possible links
    for target in range(n_vars):
        for source in range(n_vars):
            for lag in range(max_lag + 1):
                # No self-loops at lag 0
                if lag == 0 and source == target:
                    continue
                # Avoid reverse causality at lag 0 to ensure DAG
                if lag == 0 and source > target:
                    continue
                
                possible_links.append((source, lag, target))
    
    # Randomly select edges
    if n_edges > len(possible_links):
        selected_links = possible_links # Take all if requested more than possible
    else:
        selected_links = random.sample(possible_links, n_edges)
        
    # Assign random weights and powers
    for source, lag, target in selected_links:
        # Random weight between 0.2 and 0.8 (positive or negative)
        weight = np.random.uniform(0.2, 0.8)
        if np.random.random() < 0.5:
            weight = -weight
            
        source_name = f"X{source+1}"
        target_name = f"X{target+1}"
        
        # Default power is 1 (linear)
        power = 1
        links[(source_name, -lag, target_name, power)] = weight

    return links

def generate_confounded_links(n_vars, n_confounder_edges, links, random_state=None):
    """
    Add confounder links to existing links.
    U -> X relationships.
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
        
    # U affects multiple variables.
    # We want U to be a common cause, so it should affect at least 2 variables.
    targets = [f"X{i+1}" for i in range(n_vars)]
    selected_targets = random.sample(targets, min(len(targets), max(2, n_confounder_edges)))
    
    for target in selected_targets:
        weight = np.random.uniform(0.2, 0.8)
        if np.random.random() < 0.5:
            weight = -weight
        
        # U is contemporaneous (lag 0) usually in these models
        # (source, lag, target, power)
        links[('U', 0, target, 1)] = weight
        
    return links

def _assign_random_weights(selected_links):
    links = {}
    for source, lag, target in selected_links:
        weight = np.random.uniform(0.2, 0.8)
        if np.random.random() < 0.5:
            weight = -weight
            
        source_name = f"X{source+1}"
        target_name = f"X{target+1}"
        power = 1
        links[(source_name, -lag, target_name, power)] = weight
    return links

def generate_er_dag(n_vars, edge_prob, max_lag, random_state=None):
    """Erdős-Rényi (ER) graph with given edge probability."""
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
        
    possible_links = []
    for target in range(n_vars):
        for source in range(n_vars):
            for lag in range(max_lag + 1):
                if lag == 0 and source == target:
                    continue
                if lag == 0 and source > target:
                    continue
                possible_links.append((source, lag, target))
                
    selected_links = [link for link in possible_links if np.random.random() < edge_prob]
    return _assign_random_weights(selected_links)

def generate_ba_dag(n_vars, m_edges, max_lag, random_state=None):
    """Barabási-Albert (BA) scale-free graph using preferential attachment."""
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
        
    if n_vars < 2: return {}
        
    summary_edges = []
    degrees = np.zeros(n_vars)
    
    m0 = max(2, min(m_edges + 1, n_vars))
    # Initial clique
    for i in range(m0):
        for j in range(i + 1, m0):
            summary_edges.append((i, j))
            degrees[i] += 1
            degrees[j] += 1
            
    # Preferential attachment
    for i in range(m0, n_vars):
        prob = degrees[:i] / np.sum(degrees[:i])
        targets = np.random.choice(i, size=min(m_edges, i), replace=False, p=prob)
        for target in targets:
            summary_edges.append((target, i))
            degrees[target] += 1
            degrees[i] += 1
            
    selected_links = []
    for u, v in summary_edges:
        lag = np.random.randint(0, max_lag + 1)
        if lag == 0:
            source, target = min(u, v), max(u, v)
        else:
            source, target = (u, v) if np.random.random() < 0.5 else (v, u)
        selected_links.append((source, lag, target))
        
    return _assign_random_weights(selected_links)

def generate_ws_dag(n_vars, k_neighbors, rewire_prob, max_lag, random_state=None):
    """Watts-Strogatz (WS) small-world graph."""
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
        
    summary_edges = set()
    k = min(k_neighbors, n_vars - 1)
    if k % 2 != 0: k -= 1 # k must be even
    if k < 2: k = 2
    
    # Ring lattice
    for i in range(n_vars):
        for j in range(1, k // 2 + 1):
            u = i
            v = (i + j) % n_vars
            summary_edges.add((min(u, v), max(u, v)))
            
    # Rewire
    final_edges = []
    existing_edges = set(summary_edges)
    for u, v in list(summary_edges):
        if np.random.random() < rewire_prob:
            potential_targets = [n for n in range(n_vars) if n != u and (min(u, n), max(u, n)) not in existing_edges]
            if potential_targets:
                new_v = np.random.choice(potential_targets)
                final_edges.append((u, new_v))
                existing_edges.add((min(u, new_v), max(u, new_v)))
            else:
                final_edges.append((u, v))
        else:
            final_edges.append((u, v))
            
    selected_links = []
    for u, v in final_edges:
        lag = np.random.randint(0, max_lag + 1)
        if lag == 0:
            source, target = min(u, v), max(u, v)
        else:
            source, target = (u, v) if np.random.random() < 0.5 else (v, u)
        selected_links.append((source, lag, target))
        
    return _assign_random_weights(selected_links)

