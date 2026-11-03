"""
07_spatiotemporal.py

Demonstrates generating spatio-temporal data across a 2D grid,
applying a hard intervention, and computing the counterfactual twin.
"""

import os
import sys
import numpy as np

try:
    from dtcd.spatial import (
        grid_2d, 
        GaussianKernel, 
        SpatioTemporalConfig, 
        SpatioTemporalGenerator,
        HardIntervention,
        SpatioTemporalCounterfactualSuite
    )
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    from dtcd.spatial import (
        grid_2d, 
        GaussianKernel, 
        SpatioTemporalConfig, 
        SpatioTemporalGenerator,
        HardIntervention,
        SpatioTemporalCounterfactualSuite
    )

def main():
    print("=" * 60)
    print("07: SPATIO-TEMPORAL GENERATION")
    print("=" * 60)
    
    print("Setting up grid topology and spatial kernel...")
    topology = grid_2d(n_rows=4, n_cols=4)
    kernel = GaussianKernel(sigma=1.5, normalize=True)
    
    config = SpatioTemporalConfig(
        topology=topology, 
        kernel=kernel, 
        spatial_coupling=np.array([0.3, 0.3]), 
        spatial_lag=1
    )
    
    W = np.zeros((2, 2))
    A = np.zeros((2, 2, 2))
    rng = np.random.default_rng(42)
    
    generator = SpatioTemporalGenerator(config=config, W=W, A=A, rng=rng)
    dataset = generator.generate(T=200)
    
    print("Generating counterfactual twins with HardIntervention...")
    suite = SpatioTemporalCounterfactualSuite(dataset)
    
    intervention = HardIntervention(times=[100], locations=[0], variables=[0], values=[10.0])
    pair = suite.apply(intervention)
    
    factual = pair.factual
    counterfactual = pair.counterfactual
    
    diff = np.abs(counterfactual - factual)
    
    print(f"\nData Shape: {factual.shape}")
    print(f"Kernel Matrix Shape: {kernel.build(topology).shape}")
    print(f"Max pre-intervention diff (t<100): {diff[:100].max():.6f}")
    print(f"Max post-intervention diff (t>=100): {diff[100:].max():.6f}")
    
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, '07_spatial_factual.npz')
    np.savez(out_path, factual=factual, counterfactual=counterfactual)
    print(f"\nSaved arrays to {out_path}")
    
    try:
        import matplotlib.pyplot as plt
        print("Plotting spatial heatmaps...")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Heatmap of factual data at t=99 (pre-intervention)
        img1 = factual[99, :, 0].reshape(4, 4)
        im1 = axes[0].imshow(img1, cmap='viridis')
        axes[0].set_title('Factual Data (t=99, Var=0)')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot 2: Heatmap of difference at t=110 (post-intervention)
        img2 = diff[110, :, 0].reshape(4, 4)
        im2 = axes[1].imshow(img2, cmap='magma')
        axes[1].set_title('Counterfactual Diff (t=110, Var=0)')
        plt.colorbar(im2, ax=axes[1])
        
        out_path_png = os.path.join(out_dir, '07_spatial_heatmaps.png')
        plt.savefig(out_path_png)
        print(f"Saved plots to {out_path_png}")
    except ImportError:
        print("matplotlib not installed. Skipping plots.")

if __name__ == "__main__":
    main()
