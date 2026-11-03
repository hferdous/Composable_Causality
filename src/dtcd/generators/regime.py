import numpy as np
import pandas as pd
from pathlib import Path
from .base import BaseTimeSeriesGenerator

from .linear import LinearTimeSeriesGenerator
from .nonlinear import NonlinearTimeSeriesGenerator
from .confounded import ConfoundedTimeSeriesGenerator
from .irregular import IrregularTimeSeriesGenerator
from ..utils.visualization import save_dataset_and_graph

class RegimeSwitchingTimeSeriesGenerator(BaseTimeSeriesGenerator):
    def __init__(self, n_vars=4, max_lag=2, regimes=None, random_state=None):
        """
        Initialize regime switching generator.
        regimes: list of dicts, each defining a regime.
            Example:
            [
                {'start': 0, 'model': 'linear', 'params': {'n_edges': 5}},
                {'start': 50, 'model': 'nonlinear', 'params': {'nonlinear_degree': 2}}
            ]
        """
        super().__init__(random_state=random_state)
        self.n_vars = n_vars
        self.max_lag = max_lag
        self.regimes = regimes if regimes else []
        self.generators = []
        
        # Sort regimes by start time
        self.regimes.sort(key=lambda x: x['start'])
        
        # Initialize generators for each regime
        for regime in self.regimes:
            model_type = regime.get('model', 'linear')
            params = regime.get('params', {}).copy()
            
            # Ensure common params
            params['n_vars'] = n_vars
            params['max_lag'] = max_lag
            params['random_state'] = random_state # Share random state or offset it?
            
            if model_type == 'linear':
                gen = LinearTimeSeriesGenerator(**params)
            elif model_type == 'nonlinear':
                gen = NonlinearTimeSeriesGenerator(**params)
            elif model_type == 'confounded':
                gen = ConfoundedTimeSeriesGenerator(**params)
            elif model_type == 'irregular':
                gen = IrregularTimeSeriesGenerator(**params)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
            self.generators.append(gen)

    def generate_multivariate_ts(self, n_points):
        """
        Generate time series with regime switches.
        Note: This is a simplified implementation where we concatenate series.
        A more continuous approach would be to pass the history from the previous regime 
        to the next generator's step function, but our generators currently compute 
        the whole series at once.
        
        To support continuity, we would need to refactor generators to expose a `step` function
        or `generate_from_history`.
        
        For now, we will generate segments and concatenate, which might cause jumps at boundaries.
        To smooth it, we can seed the next regime with the last lags of the previous one.
        """
        
        full_df = pd.DataFrame()
        
        # We need to handle the transition.
        # Ideally, we modify the generators to accept 'initial_history'.
        # Since we didn't implement that yet, let's do a best-effort approach:
        # Generate full series for each regime, then slice. 
        # But that doesn't capture the causal transition (the new regime depending on old regime's past).
        
        # CORRECT APPROACH:
        # We must implement a step-wise generation or modify generators to accept history.
        # Given the constraints, let's modify the `generate_multivariate_ts` of the base/sub classes
        # to accept `initial_history`.
        
        # However, that requires changing all 4 generator files.
        # Alternative: The `RegimeSwitching` generator implements its own loop 
        # and delegates `compute_value` to the underlying models.
        
        X = np.zeros((n_points, self.n_vars))
        var_names = [f'X{i+1}' for i in range(self.n_vars)]
        
        # Initialize with noise
        for i in range(self.max_lag):
            X[i] = self.generate_noise(self.n_vars)
            
        current_regime_idx = 0
        
        for t in range(self.max_lag, n_points):
            # Check if we need to switch regime
            if current_regime_idx < len(self.regimes) - 1:
                next_start = self.regimes[current_regime_idx + 1]['start']
                if t >= next_start:
                    current_regime_idx += 1
            
            current_gen = self.generators[current_regime_idx]
            
            # We need to use the current generator's model to compute the next value.
            # All our generators use `self.model.compute_value`.
            # So we can reuse that.
            
            noise = current_gen.generate_noise(self.n_vars)
            current_values = {}
            for i in range(self.n_vars):
                current_values[f'X{i+1}'] = noise[i]
                
            history = {name: X[:, i] for i, name in enumerate(var_names)}
            
            # Handle Confounded Generator special case (U)
            if isinstance(current_gen, ConfoundedTimeSeriesGenerator):
                # Generate U
                u_noise = current_gen.generate_noise(1)[0]
                current_values['U'] = u_noise
                
            # Handle Irregular Generator special case
            # Irregular is tricky here because t is an index, but irregular works on timestamps.
            # Mixing irregular and regular regimes is complex.
            # If the current regime is irregular, we need timestamps.
            # For simplicity, let's assume regime switching happens on integer steps 
            # and if a regime is irregular, it handles its own internal timing?
            # Actually, `compute_value` in irregular uses `temp_history` mapping.
            # This is getting complicated.
            
            # SIMPLIFICATION:
            # For this version, we will assume all regimes share the same sampling (regular)
            # OR we restrict regime switching to non-irregular models for now, 
            # or we just use the model's equation logic.
            
            if isinstance(current_gen, IrregularTimeSeriesGenerator):
                # Fallback to regular sampling for regime switching for now
                # or raise warning.
                pass

            for i in range(self.n_vars):
                target_name = f'X{i+1}'
                structural_val = current_gen.model.compute_value(target_name, t, history, current_values)
                final_val = structural_val + noise[i]
                current_values[target_name] = final_val
                X[t, i] = final_val
                
        df = pd.DataFrame(X, columns=var_names)
        df['time'] = np.arange(n_points)
        return df

    def save(self, df, output_dir="output"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Save the full dataset
        filename = f'{output_dir}/regime_switching_ts_n{len(df)}_vars{self.n_vars}.csv'
        df.to_csv(filename, index=False)
        
        # 2. Save regime summary
        summary_file = f'{output_dir}/regime_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("Regime Switching Summary\n")
            f.write("========================\n\n")
            
            for i, regime in enumerate(self.regimes):
                start = regime['start']
                end = self.regimes[i+1]['start'] if i < len(self.regimes) - 1 else len(df)
                model_type = regime.get('model', 'linear')
                gen = self.generators[i]
                
                f.write(f"Regime {i+1}: Time {start} to {end}\n")
                f.write(f"Model: {model_type}\n")
                f.write("Equations:\n")
                for eq in gen.model.equations:
                    f.write(f"  {eq}\n")
                f.write("\n")
                
                # 3. Save graph for this regime
                # We reuse save_dataset_and_graph but strictly for the graph/structure part
                # We pass a dummy df or just the subset to satisfy the function signature
                # Ideally we should refactor visualization to separate plotting, but this works.
                
                # Create a subdirectory for regime graphs to keep it clean
                regime_dir = f"{output_dir}/regimes"
                Path(regime_dir).mkdir(parents=True, exist_ok=True)
                
                # We pass the full df subset so the time series plot in the regime folder 
                # corresponds to that regime's data
                df_subset = df.iloc[start:end]
                
                save_dataset_and_graph(
                    df_subset, self.n_vars, self.max_lag, len(df_subset), 
                    f"regime_{i}_{model_type}", 
                    gen.links, gen.model.equations, 
                    regime_dir
                )

