import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def _tigramite_plotting():
    """Import tigramite lazily. It is an optional [discovery] extra, so a
    core-only install must still be able to import dtcd and generate data."""
    try:
        from tigramite import plotting as tp
        return tp
    except ImportError as e:
        raise ImportError(
            "Graph plotting requires the optional 'discovery' extra: "
            "pip install 'dtcd[discovery]'"
        ) from e


def save_dataset_and_graph(df, n_vars, max_lag, sample_size, noise_type, true_links, equations, output_dir="output"):
    """Save dataset and create causal graph using Tigramite"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save dataset
    filename = f'{output_dir}/ts_n{sample_size}_vars{n_vars}_lag{max_lag}_{noise_type}.csv'
    df.to_csv(filename, index=False)

    # Create matrices for tigramite plotting
    # Assuming variables are named X1, X2, ... or provided in df columns
    var_names = list(df.columns)
    if 'time' in var_names:
        var_names.remove('time')
    
    n_total_vars = len(var_names)
    val_matrix = np.zeros((n_total_vars, n_total_vars, max_lag + 1))
    graph_matrix = np.zeros((n_total_vars, n_total_vars, max_lag + 1), dtype='bool')

    # Map variable names to indices
    var_to_idx = {name: i for i, name in enumerate(var_names)}

    # Fill matrices based on true links
    # Link format: (source_name, lag, target_name, power) -> weight
    # Or (source_name, lag, target_name) -> weight (backward compatibility)
    for link_key, weight in true_links.items():
        if len(link_key) == 4:
            source, lag, target, power = link_key
        else:
            source, lag, target = link_key
            power = 1
            
        if source not in var_to_idx or target not in var_to_idx:
            continue
            
        source_idx = var_to_idx[source]
        target_idx = var_to_idx[target]
        lag_idx = abs(lag)

        # Add the link to the matrices
        # For visualization, we might just sum weights if there are multiple terms (e.g. linear + quadratic)
        # or just take the max/first.
        # Tigramite graph matrix is boolean. Val matrix is float.
        # Let's accumulate weight.
        val_matrix[source_idx, target_idx, lag_idx] += weight
        graph_matrix[source_idx, target_idx, lag_idx] = True

        # For contemporaneous links, make val_matrix symmetric for visualization if needed
        if lag == 0:
            val_matrix[target_idx, source_idx, lag_idx] += weight


    # Plot and save causal graph
    plt.figure(figsize=(12, 12))
    try:
        _tigramite_plotting().plot_time_series_graph(
            val_matrix=val_matrix,
            graph=graph_matrix,
            var_names=var_names,
            link_colorbar_label='Effect Strength',
            node_size=0.05
        )
        plt.title(f'Causal Graph (n={sample_size}, vars={n_vars}, lag={max_lag}, {noise_type})')
        plt.savefig(f'{output_dir}/causal_graph_n{sample_size}_vars{n_vars}_lag{max_lag}_{noise_type}.png')
    except Exception as e:
        print(f"Error plotting graph: {e}")
    finally:
        plt.close()

    # Plot time series
    plt.figure(figsize=(15, 10))
    for col in var_names:
        plt.plot(df['time'], df[col], label=col, alpha=0.7)
    plt.title(f'Time Series (n={sample_size}, vars={n_vars}, lag={max_lag})')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/ts_plot_n{sample_size}_vars{n_vars}_lag{max_lag}_{noise_type}.png')
    plt.close()

    # Save causal structure description
    with open(f'{output_dir}/causal_structure_vars{n_vars}_lag{max_lag}_{noise_type}.txt', 'w') as f:
        f.write(f"True Causal Structure ({noise_type} noise):\n")
        f.write("Format: (source, lag, target) => coefficient\n")
        f.write("\nEquations:\n")
        for eq in equations:
            f.write(f"{eq}\n")
        f.write("\nCausal Links:\n")
        for link_key, coef in true_links.items():
            if len(link_key) == 4:
                source, lag, target, power = link_key
                power_str = f"^{power}" if power != 1 else ""
            else:
                source, lag, target = link_key
                power_str = ""
            f.write(f"({source}, {lag}, {target}){power_str} => {coef}\n")

