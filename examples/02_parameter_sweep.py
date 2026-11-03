"""
02_parameter_sweep.py

Demonstrates how to sweep across data-generation parameters 
(sparsity levels and noise types) and generate multiple trajectories,
observing how these choices impact the cross-correlation structure.
"""

import os
import sys
import numpy as np
import pandas as pd

try:
    from dtcd.generators.linear import LinearTimeSeriesGenerator
except ImportError:
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'src')
    ))
    from dtcd.generators.linear import LinearTimeSeriesGenerator

def main():
    print("=" * 60)
    print("02: PARAMETER SWEEP")
    print("=" * 60)
    
    n_vars = 4
    max_lag = 2
    n_points = 500
    seed = 42
    
    sparsities = [2, 4, 6]
    noise_types = ['gaussian', 'laplace', 'student_t']
    
    out_dir = os.path.join(os.path.dirname(__file__), 'output', '02_sweep')
    os.makedirs(out_dir, exist_ok=True)
    
    results = []
    
    print("Starting parameter sweep...")
    print(f"{'Edges':<8} | {'Noise Type':<12} | {'Max Off-Diag Corr':<18}")
    print("-" * 45)
    
    for n_edges in sparsities:
        for noise in noise_types:
            # 1. Initialize generator
            generator = LinearTimeSeriesGenerator(
                n_vars=n_vars,
                max_lag=max_lag,
                n_edges=n_edges,
                noise_type=noise,
                noise_scale=0.1,
                random_state=42
            )
            
            # 2. Generate
            df = generator.generate_multivariate_ts(n_points=n_points)
            
            # 3. Analyze correlation (excluding 'time' column)
            data_cols = [c for c in df.columns if c != 'time']
            corr_matrix = df[data_cols].corr().abs().to_numpy(copy=True)
            np.fill_diagonal(corr_matrix, 0)
            max_corr = np.max(corr_matrix)
            
            # 4. Save dataset
            out_path = os.path.join(out_dir, f"{n_edges}_{noise}.csv")
            df.to_csv(out_path, index=False)
            
            print(f"{n_edges:<8} | {noise:<12} | {max_corr:.4f}")
            results.append((n_edges, noise, max_corr))

    print(f"\nSaved {len(results)} trajectories to {out_dir}/")

if __name__ == "__main__":
    main()
