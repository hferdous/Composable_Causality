import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from dtcd.viz.graph_viz import plot_ground_truth_graph
from dtcd.benchmarks.predefined_graphs import PREDEFINED_GRAPHS

def plot_recovered_graph(method_name, W_inst_pred, W_lag_pred, graph_dict, metrics, output_path):
    d = graph_dict["d"]
    W_inst_true = np.zeros((d, d))
    for s, t in graph_dict["instantaneous"]: W_inst_true[s, t] = 1.0
    
    W_lag_true = np.zeros((graph_dict["tau_max"], d, d))
    for s, t, lag in graph_dict["lagged"]: W_lag_true[lag-1, s, t] = 1.0

    plt.figure(figsize=(3.5, 3.5))
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(d))
    pos = nx.circular_layout(G)

    # Node coloring same as GT
    tr, out = graph_dict["treatment_node"], graph_dict["outcome_node"]
    node_colors = ["#e6ffed" if i==tr else "#fff1f0" if i==out else "#f0f0f0" for i in range(d)]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, edgecolors="#8c8c8c", linewidths=1.2, node_size=600)
    nx.draw_networkx_labels(G, pos, font_size=9)

    def get_edge_type(s, t, is_inst, lag=0):
        pred = W_inst_pred[s, t] if is_inst else W_lag_pred[lag-1, s, t]
        true = W_inst_true[s, t] if is_inst else W_lag_true[lag-1, s, t]
        if pred > 0 and true > 0: return "tp"
        if pred > 0 and true == 0: return "fp"
        if pred == 0 and true > 0: return "fn"
        return None

    # Collect edges
    edges_to_draw = []
    # Inst
    for i in range(d):
        for j in range(d):
            etype = get_edge_type(i, j, True)
            if etype: edges_to_draw.append((i, j, 0, etype))
    # Lags
    for k in range(1, graph_dict["tau_max"] + 1):
        for i in range(d):
            for j in range(d):
                etype = get_edge_type(i, j, False, k)
                if etype: edges_to_draw.append((i, j, k, etype))

    for src, dst, lag, etype in edges_to_draw:
        color = "black" if etype=="tp" else "red" if etype=="fp" else "lightgray"
        style = "solid" if etype in ["tp", "fp"] else "dashed"
        width = 1.2 if etype in ["tp", "fp"] else 0.8
        alpha = 1.0 if etype in ["tp", "fp"] else 0.5
        
        nx.draw_networkx_edges(G, pos, edgelist=[(src, dst)], arrowstyle='-|>', 
                               arrowsize=10, width=width, edge_color=color, 
                               style=style, alpha=alpha, connectionstyle=f'arc3,rad={0.1*lag}')

    plt.title(f"{method_name}\nF1={metrics['f1_orient']:.2f} SHD={metrics['shd']:.1f}", fontsize=9)
    plt.axis('off')
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # This will be called by the main benchmark runner when the cell is done
    pass
