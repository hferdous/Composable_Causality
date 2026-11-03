"""
08_full_pipeline_minimal.py

A minimal end-to-end demonstration showing the full user journey:
Generate -> Inspect Ground Truth -> Export -> (Handoff to Baseline).
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
    print("08: FULL PIPELINE MINIMAL")
    print("=" * 60)
    
    # Step 1: Generate
    print("Step 1: Generating dataset (n_vars=4, max_lag=2)...")
    generator = LinearTimeSeriesGenerator(n_vars=4, max_lag=2, n_edges=4, random_state=42)
    df = generator.generate_multivariate_ts(n_points=500)
    
    # Step 2: Inspect ground truth
    print("\nStep 2: Inspecting ground truth links...")
    print("  ", list(generator.links.keys())[:3], "...")
    
    # Step 3: Export
    print("\nStep 3: Exporting to causal-learn format...")
    cl_matrix = to_causal_learn_matrix(generator.links, n_vars=4, max_lag=2)
    print("Matrix shape:", cl_matrix.shape)
    
    # Step 4: Handoff
    print("\nStep 4: Pass this matrix to any causal-learn algorithm.")
    print("See examples/05_export_to_causallearn.py for an end-to-end example including the PC algorithm.")
    
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, '08_full_pipeline.npz')
    np.savez(out_path, matrix=cl_matrix)
    print(f"\nSaved export to {out_path}")

if __name__ == "__main__":
    main()
