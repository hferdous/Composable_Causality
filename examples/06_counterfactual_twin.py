"""
06_counterfactual_twin.py

Demonstrates DTCD's ability to generate perfectly aligned counterfactual 
twins by caching underlying noise. Evaluates a hard "do-intervention".
"""

import os
import sys
import numpy as np

try:
    from dtcd.generators.linear import LinearTimeSeriesGenerator
    from dtcd.interventions.base import InterventionManager, CounterfactualSuite
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    from dtcd.generators.linear import LinearTimeSeriesGenerator
    from dtcd.interventions.base import InterventionManager, CounterfactualSuite

def main():
    print("=" * 60)
    print("06: COUNTERFACTUAL TWIN")
    print("=" * 60)
    
    print("Initializing generator...")
    generator = LinearTimeSeriesGenerator(n_vars=3, max_lag=2, n_edges=3, random_state=42)
    
    print("Building InterventionManager for do(X1 = 5.0) at t=100...")
    ivm = InterventionManager()
    ivm.add_intervention(target='X1', intervention_type='do', value=5.0, start=100, end=101)
    
    print("Generating twins via CounterfactualSuite...")
    suite = CounterfactualSuite(generator)
    obs_df, int_df = suite.generate_twins(n_points=200, intervention_manager_for_twin=ivm)
    
    non_data_cols = [c for c in ['time'] if c in obs_df.columns]
    X_obs = obs_df.drop(columns=non_data_cols).values
    X_int = int_df.drop(columns=non_data_cols).values
    
    diff = np.abs(X_int - X_obs)
    pre_diff = diff[:100].max()
    post_diff = diff[100:].max()
    
    print(f"\nObservational Shape: {X_obs.shape}")
    print(f"Interventional Shape: {X_int.shape}")
    print(f"Max Difference Before Intervention (t < 100): {pre_diff:.6f} (Expected ~0)")
    print(f"Value at X1[t=100] (intervened): {X_int[100, 0]:.4f}")
    print(f"Max Difference After Intervention (t >= 100): {post_diff:.6f}")
    
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_path_npz = os.path.join(out_dir, '06_counterfactual.npz')
    np.savez(out_path_npz, X_obs=X_obs, X_int=X_int)
    print(f"\nSaved data to {out_path_npz}")
    
    try:
        import matplotlib.pyplot as plt
        print("Plotting counterfactuals...")
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        t_plot = np.arange(150)
        
        for i in range(3):
            axes[i].plot(t_plot, X_obs[:150, i], 'k-', label='Factual', alpha=0.7)
            axes[i].plot(t_plot, X_int[:150, i], 'r--', label='Counterfactual', alpha=0.7)
            axes[i].axvline(100, color='b', linestyle=':', label='Intervention (t=100)')
            axes[i].set_ylabel(f'X{i+1}')
            axes[i].legend(loc='upper left')
            
        plt.xlabel('Time')
        plt.tight_layout()
        out_path_png = os.path.join(out_dir, '06_counterfactual.png')
        plt.savefig(out_path_png)
        print(f"Saved plot to {out_path_png}")
    except ImportError:
        print("matplotlib not installed. Skipping plot.")

if __name__ == "__main__":
    main()
