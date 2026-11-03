import networkx as nx
from typing import Dict, List, Tuple, Any

# d=4 graphs (4 edges each), τ_max varies
GRAPH_4_2 = {  # d=4, τ_max=2
    "instantaneous": [(0, 1), (1, 2)],          # contemporaneous
    "lagged": [(0, 2, 1), (1, 3, 2)],           # (src, dst, lag)
    "treatment_node": 0, "outcome_node": 3,
    "d": 4, "tau_max": 2
}
GRAPH_4_3 = {
    "instantaneous": [(0, 1)],
    "lagged": [(0, 2, 1), (1, 3, 2), (2, 3, 3)],
    "treatment_node": 0, "outcome_node": 3,
    "d": 4, "tau_max": 3
}
GRAPH_4_4 = {
    "instantaneous": [(0, 1)],
    "lagged": [(0, 2, 2), (1, 3, 3), (0, 3, 4)],
    "treatment_node": 0, "outcome_node": 3,
    "d": 4, "tau_max": 4
}

# d=6 graphs (6 edges each)
GRAPH_6_2 = {
    "instantaneous": [(0, 1), (2, 3)],
    "lagged": [(0, 2, 1), (1, 4, 1), (3, 5, 2), (2, 5, 2)],
    "treatment_node": 0, "outcome_node": 5,
    "d": 6, "tau_max": 2
}
GRAPH_6_3 = {
    "instantaneous": [(0, 1), (2, 3)],
    "lagged": [(0, 4, 1), (1, 5, 2), (3, 5, 3), (2, 4, 3)],
    "treatment_node": 0, "outcome_node": 5,
    "d": 6, "tau_max": 3
}
GRAPH_6_4 = {
    "instantaneous": [(0, 1), (2, 3)],
    "lagged": [(0, 4, 2), (1, 5, 3), (3, 5, 4), (2, 4, 4)],
    "treatment_node": 0, "outcome_node": 5,
    "d": 6, "tau_max": 4
}

# d=8 graphs (8 edges each)
GRAPH_8_2 = {
    "instantaneous": [(0, 1), (2, 3), (4, 5)],
    "lagged": [(0, 4, 1), (1, 6, 1), (3, 7, 2), (5, 7, 2), (2, 6, 1)],
    "treatment_node": 0, "outcome_node": 7,
    "d": 8, "tau_max": 2
}
GRAPH_8_3 = {
    "instantaneous": [(0, 1), (2, 3), (4, 5)],
    "lagged": [(0, 6, 1), (1, 7, 2), (3, 7, 3), (5, 6, 3), (2, 4, 2)],
    "treatment_node": 0, "outcome_node": 7,
    "d": 8, "tau_max": 3
}
GRAPH_8_4 = {
    "instantaneous": [(0, 1), (2, 3), (4, 5)],
    "lagged": [(0, 6, 2), (1, 7, 3), (3, 7, 4), (5, 6, 4), (2, 4, 3)],
    "treatment_node": 0, "outcome_node": 7,
    "d": 8, "tau_max": 4
}

PREDEFINED_GRAPHS = {
    "d4_tau2": GRAPH_4_2,
    "d4_tau3": GRAPH_4_3,
    "d4_tau4": GRAPH_4_4,
    "d6_tau2": GRAPH_6_2,
    "d6_tau3": GRAPH_6_3,
    "d6_tau4": GRAPH_6_4,
    "d8_tau2": GRAPH_8_2,
    "d8_tau3": GRAPH_8_3,
    "d8_tau4": GRAPH_8_4,
}

def validate_acyclicity(graph_dict: Dict[str, Any]):
    G = nx.DiGraph()
    G.add_nodes_from(range(graph_dict["d"]))
    G.add_edges_from(graph_dict["instantaneous"])
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError(f"Graph {graph_dict.get('d')}_tau{graph_dict.get('tau_max')} contains an instantaneous cycle!")

def check_all_graphs():
    for name, g in PREDEFINED_GRAPHS.items():
        validate_acyclicity(g)
    print("✅ All predefined graphs verified acyclic (contemporaneous component).")

if __name__ == "__main__":
    check_all_graphs()
