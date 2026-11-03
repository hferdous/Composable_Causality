"""
01_basic_generation.py

Demonstrates how to generate a simple multivariate time series dataset
using the LinearTimeSeriesGenerator with default settings.
Outputs summary statistics and the ground-truth adjacency.
"""

import os
import sys
import numpy as np

try:
    from dtcd.generators.linear import LinearTimeSeriesGenerator
except ImportError:
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'src')
    ))
    from dtcd.generators.linear import LinearTimeSeriesGenerator

def main():
    print("=" * 60)
    print("01: BASIC DATA GENERATION")
    print("=" * 60)
    
    # 1. Initialize the generator
    print("Initializing LinearTimeSeriesGenerator...")
    generator = LinearTimeSeriesGenerator(
        n_vars=4, 
        max_lag=2, 
        n_edges=4, 
        noise_type='gaussian', 
        noise_scale=0.1, 
        random_state=42
    )
    
    # 2. Generate the trajectory
    print("Generating 1000-step trajectory...")
    df = generator.generate_multivariate_ts(n_points=1000)
    
    # 3. Print summaries
    print(f"\nData Shape: {df.shape}")
    print("\nData Head:")
    print(df.head())
    print("\nSummary Statistics:")
    print(df.describe().loc[['mean', 'std', 'min', 'max']])
    
    print("\nGround-Truth Adjacency (links):")
    print(f"DEBUG: links type = {type(generator.links)}, "
          f"first key = {list(generator.links.keys())[0] if generator.links else 'empty'}")
    for link, weight in generator.links.items():
        print(f"  {link} -> {weight:.4f}")
        
    # 4. Save to output directory
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, '01_basic.npz')
    
    # We save DataFrame and links
    np.savez(
        out_path, 
        data=df.values, 
        columns=df.columns.values,
        links=np.array(list(generator.links.items()), dtype=object)
    )
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
