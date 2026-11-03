"""
05_export_to_causallearn.py

Demonstrates generating a dataset and exporting its ground-truth
links into a 2D adjacency matrix suitable for causal-learn.
"""

import os
import sys
import numpy as np

try:
    from dtcd.generators.linear import LinearTimeSeriesGenerator
    from dtcd.utils.export import to_causal_learn_matrix
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    from dtcd.generators.linear import LinearTimeSeriesGenerator
    from dtcd.utils.export import to_causal_learn_matrix

def main():
    print("=" * 60)
    print("05: EXPORT TO CAUSAL-LEARN")
    print("=" * 60)
    
    generator = LinearTimeSeriesGenerator(n_vars=4, max_lag=2, n_edges=4, random_state=42)
    df = generator.generate_multivariate_ts(n_points=500)
    
    non_data_cols = [c for c in ['time'] if c in df.columns]
    X_base = df.drop(columns=non_data_cols).values
    
    print("Exporting links to causal-learn format...")
    causal_learn_matrix = to_causal_learn_matrix(generator.links, n_vars=4, max_lag=2)
    print("\nMatrix Shape:", causal_learn_matrix.shape)
    
    # Run PC if installed
    try:
        from causallearn.search.ConstraintBased.PC import pc
        print("\ncausal-learn is installed. Running PC algorithm...")
        cg = pc(X_base, alpha=0.05)
        print("Discovered Graph:\n", cg.G.graph)
    except ImportError:
        print("\nInstall causallearn to run PC: pip install causal-learn")
        print("Skipping PC step.")
        
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, '05_causallearn.npy')
    np.save(out_path, causal_learn_matrix)
    print(f"\nSaved export matrix to {out_path}")

if __name__ == "__main__":
    main()
