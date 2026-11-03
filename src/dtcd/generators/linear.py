import numpy as np
import pandas as pd
from ..structure.graph import generate_random_dag
from ..structure.equations import StructuralModel
from .base import BaseTimeSeriesGenerator
from ..utils.visualization import save_dataset_and_graph

class LinearTimeSeriesGenerator(BaseTimeSeriesGenerator):
    def __init__(self, n_vars=4, max_lag=2, n_edges=None, links=None, 
                 noise_type='gaussian', noise_scale=0.1, df=3, random_state=None):
        """
        Initialize generator.
        """
        super().__init__(noise_type, noise_scale, df, random_state)
        self.n_vars = n_vars
        self.max_lag = max_lag
        
        if links is not None:
            self.links = links
        elif n_edges is not None:
            self.links = generate_random_dag(n_vars, n_edges, max_lag, random_state)
        else:
            default_edges = int(n_vars * 1.5)
            self.links = generate_random_dag(n_vars, default_edges, max_lag, random_state)
            
        self.model = StructuralModel(self.links, n_vars=self.n_vars)
        self.intervention_manager = None

    def generate_multivariate_ts(self, n_points, noise_data=None):
        """Generate multivariate time series with optional frozen noise.
        
        Parameters
        ----------
        n_points : int
            Number of timesteps to generate.
        noise_data : np.ndarray, optional
            If provided, reuses this noise array instead of sampling 
            new noise. Used by CounterfactualSuite for exact-twin 
            generation.
        
        Notes
        -----
        Hard ('do') interventions set the observed value at the 
        intervention timestep but do NOT propagate to contemporaneous 
        children at the same timestep — contemporaneous children read 
        the natural (pre-intervention) value of the intervened variable. 
        Lagged children are unaffected and observe the intervened value 
        at subsequent timesteps. This convention is shared with 
        UniversalTimeSeriesGenerator. Soft interventions are added 
        after the structural value is computed and propagate normally.
        """
        X = np.zeros((n_points, self.n_vars))
        var_names = [f'X{i+1}' for i in range(self.n_vars)]
        
        if noise_data is None:
            noise_history = np.zeros((n_points, self.n_vars))
        else:
            noise_history = noise_data

        # Initialize first steps with noise
        for i in range(self.max_lag):
            if noise_data is None:
                noise_history[i] = self.generate_noise(self.n_vars)
            X[i] = noise_history[i]

        # Generate time series
        for t in range(self.max_lag, n_points):
            if noise_data is None:
                noise = self.generate_noise(self.n_vars)
                noise_history[t] = noise
            else:
                noise = noise_history[t]
            
            current_values = {}
            # Initialize with noise
            for i in range(self.n_vars):
                current_values[f'X{i+1}'] = noise[i]
                
            history = {name: X[:, i] for i, name in enumerate(var_names)}
            
            for i in range(self.n_vars):
                target_name = f'X{i+1}'
                
                # Check for Hard Interventions (Do)
                if hasattr(self, 'intervention_manager') and self.intervention_manager is not None:
                    SENTINEL = object()
                    ival = self.intervention_manager.get_intervention_value(
                        target_name, t, SENTINEL, history=history, current_values=current_values, only_type='do'
                    )
                    if ival is not SENTINEL:
                        X[t, i] = ival
                        continue
                        
                # Add structural component
                structural_val = self.model.compute_value(target_name, t, history, current_values)
                
                final_val = structural_val + noise[i]
                
                # Apply Soft/Noise interventions via manager
                if hasattr(self, 'intervention_manager') and self.intervention_manager is not None:
                    final_val = self.intervention_manager.get_intervention_value(
                        target_name, t, final_val, history=history, current_values=current_values, only_type='soft'
                    )
                
                current_values[target_name] = final_val
                X[t, i] = final_val

        # Create DataFrame
        df = pd.DataFrame(X, columns=var_names)
        df['time'] = np.arange(n_points)
        
        self.last_noise = noise_history
        
        return df

    def save(self, df, output_dir="output"):
        save_dataset_and_graph(
            df, self.n_vars, self.max_lag, len(df), 
            self.noise_type, self.links, self.model.equations, 
            output_dir
        )
