"""
04_export_to_tigramite.py

Demonstrates generating a dataset and exporting its ground-truth
links into the format expected by the Tigramite package.
"""

import os
import sys
import pickle

try:
    from dtcd.generators.linear import LinearTimeSeriesGenerator
    from dtcd.utils.export import to_tigramite_links
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    from dtcd.generators.linear import LinearTimeSeriesGenerator
    from dtcd.utils.export import to_tigramite_links

def main():
    print("=" * 60)
    print("04: EXPORT TO TIGRAMITE")
    print("=" * 60)
    
    generator = LinearTimeSeriesGenerator(n_vars=4, max_lag=2, n_edges=4, random_state=42)
    df = generator.generate_multivariate_ts(n_points=500)
    
    non_data_cols = [c for c in ['time'] if c in df.columns]
    X_base = df.drop(columns=non_data_cols).values
    
    print("Exporting links to Tigramite format...")
    tigramite_links = to_tigramite_links(generator.links, n_vars=4, max_lag=2)
    
    print("\nTigramite val_matrix shape:", tigramite_links['val_matrix'].shape)
    
    # Run PCMCI if installed
    try:
        import tigramite
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite import data_processing as pp
        
        print("\nTigramite is installed. Running PCMCI...")
        dataframe = pp.DataFrame(X_base)
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr())
        results = pcmci.run_pcmci(tau_max=2, pc_alpha=0.05)
        print("PCMCI complete. Resulting p-value matrix shape:", results['p_matrix'].shape)
        
    except ImportError:
        print("\nInstall tigramite to run PCMCI: pip install tigramite")
        print("Skipping PCMCI step.")
        
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, '04_tigramite.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump({'data': X_base, 'links': tigramite_links}, f)
    print(f"\nSaved export to {out_path}")

if __name__ == "__main__":
    main()
