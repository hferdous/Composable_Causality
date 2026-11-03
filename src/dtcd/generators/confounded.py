import numpy as np
import pandas as pd
from ..structure.graph import generate_random_dag, generate_confounded_links
from ..structure.equations import StructuralModel
from .base import BaseTimeSeriesGenerator
from ..utils.visualization import save_dataset_and_graph

class ConfoundedTimeSeriesGenerator(BaseTimeSeriesGenerator):
    def __init__(self, n_vars=4, max_lag=2, n_edges=None, n_confounder_edges=2, links=None, 
                 nonlinear_degree=1,
                 noise_type='gaussian', noise_scale=0.1, df=3, random_state=None):
        """
        Initialize confounded generator.
        Adds a latent variable U that affects multiple observed variables.
        """
        super().__init__(noise_type, noise_scale, df, random_state)
        self.n_vars = n_vars
        self.max_lag = max_lag
        
        if links is not None:
            self.links = links
        else:
            if n_edges is None:
                n_edges = int(n_vars * 1.5)
            
            # Generate base DAG
            base_links = generate_random_dag(n_vars, n_edges, max_lag, random_state)
            
            # Add nonlinear terms if requested
            if nonlinear_degree > 1:
                # Reuse logic from nonlinear generator or duplicate simple version here
                # For simplicity, let's just modify the base links
                temp_links = {}
                for (source, lag, target, _), weight in base_links.items():
                    power = np.random.randint(1, nonlinear_degree + 1)
                    temp_links[(source, lag, target, power)] = weight
                base_links = temp_links
            
            # Add confounder links
            self.links = generate_confounded_links(n_vars, n_confounder_edges, base_links, random_state)
            
        self.model = StructuralModel(self.links, n_vars=self.n_vars)


    def generate_multivariate_ts(self, n_points):
        """Generate multivariate time series with confounder"""
        X = np.zeros((n_points, self.n_vars))
        U = np.zeros(n_points)
        var_names = [f'X{i+1}' for i in range(self.n_vars)]
        
        # Initialize first steps with noise
        for i in range(self.max_lag):
            X[i] = self.generate_noise(self.n_vars)
            U[i] = self.generate_noise(1)[0]

        # Generate time series
        for t in range(self.max_lag, n_points):
            noise = self.generate_noise(self.n_vars)
            u_noise = self.generate_noise(1)[0]
            
            # Generate U first (it's a root cause usually, or has its own dynamics)
            # For simplicity, let's say U is just noise (random walk or i.i.d)
            # The original code had U[t] = eU, so i.i.d noise.
            U[t] = u_noise
            
            current_values = {'U': U[t]}
            for i in range(self.n_vars):
                current_values[f'X{i+1}'] = noise[i]
                
            history = {name: X[:, i] for i, name in enumerate(var_names)}
            # U history not strictly needed if U is i.i.d and only lag 0 effects from U
            
            for i in range(self.n_vars):
                target_name = f'X{i+1}'
                structural_val = self.model.compute_value(target_name, t, history, current_values)
                final_val = structural_val + noise[i]
                current_values[target_name] = final_val
                X[t, i] = final_val

        # Create DataFrame
        df = pd.DataFrame(X, columns=var_names)
        df['U'] = U
        df['time'] = np.arange(n_points)
        
        return df

    def save(self, df, output_dir="output"):
        save_dataset_and_graph(
            df, self.n_vars, self.max_lag, len(df), 
            self.noise_type, self.links, self.model.equations, 
            output_dir
        )
