"""
03_missingness_variants.py

Demonstrates how to apply MCAR (Missing Completely At Random) and 
MNAR (Missing Not At Random) masks to a generated trajectory.
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
    print("03: MISSINGNESS VARIANTS")
    print("=" * 60)
    
    # 1. Generate base trajectory
    print("Generating base trajectory...")
    generator = LinearTimeSeriesGenerator(
        n_vars=4, max_lag=2, n_edges=4, random_state=42
    )
    df = generator.generate_multivariate_ts(n_points=500)
    non_data_cols = [c for c in ['time'] if c in df.columns]
    X_base = df.drop(columns=non_data_cols).values
    
    # 2. Apply MCAR Mask (20% random missing)
    print("Applying MCAR mask (20%)...")
    rng_mcar = np.random.default_rng(42)
    mcar_mask = rng_mcar.random(X_base.shape) > 0.20
    X_mcar = X_base.copy()
    X_mcar[~mcar_mask] = np.nan
    
    # 3. Apply MNAR Mask (top 20% self-thresholding)
    print("Applying MNAR mask (top 20% thresholding)...")
    mnar_mask = np.ones_like(X_base, dtype=bool)
    for i in range(X_base.shape[1]):
        thresh = np.percentile(X_base[:, i], 80)
        mnar_mask[:, i] = X_base[:, i] <= thresh
    X_mnar = X_base.copy()
    X_mnar[~mnar_mask] = np.nan
    
    # 4. Print Missing Fractions
    print("\nMissing Fractions:")
    print(f"  MCAR: {np.isnan(X_mcar).mean():.4f}")
    print(f"  MNAR: {np.isnan(X_mnar).mean():.4f}")
    
    # 5. Save outputs
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_path_npz = os.path.join(out_dir, '03_missingness.npz')
    np.savez(out_path_npz, X_base=X_base, X_mcar=X_mcar, X_mnar=X_mnar)
    print(f"\nSaved data to {out_path_npz}")
    
    # 6. Optional Plotting
    try:
        import matplotlib.pyplot as plt
        print("Plotting missingness masks...")
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        t_plot = np.arange(100) # Plot first 100 steps
        
        axes[0].plot(t_plot, X_base[:100, 0], 'k-', label='Base (X1)')
        axes[0].set_title('Base Trajectory')
        
        axes[1].plot(t_plot, X_mcar[:100, 0], 'r-', label='MCAR (X1)')
        axes[1].set_title('MCAR (Missing Completely At Random)')
        
        axes[2].plot(t_plot, X_mnar[:100, 0], 'b-', label='MNAR (X1)')
        axes[2].set_title('MNAR (Missing Not At Random - Peak thresholding)')
        
        for ax in axes:
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        out_path_png = os.path.join(out_dir, '03_missingness.png')
        plt.savefig(out_path_png)
        print(f"Saved plot to {out_path_png}")
        
    except ImportError:
        print("matplotlib not installed. Skipping plot generation.")

if __name__ == "__main__":
    main()
