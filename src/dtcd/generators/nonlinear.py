import numpy as np
import pandas as pd
import random
from ..structure.graph import generate_random_dag
from ..structure.equations import StructuralModel
from .base import BaseTimeSeriesGenerator
from ..utils.visualization import save_dataset_and_graph

class NonlinearTimeSeriesGenerator(BaseTimeSeriesGenerator):
    def __init__(self, n_vars=4, max_lag=2, n_edges=None, links=None, 
                 nonlinear_degree=3,
                 noise_type='gaussian', noise_scale=0.1, df=3, random_state=None):
        """
        Initialize nonlinear generator.
        nonlinear_degree: max degree of polynomial terms (e.g. 2 for quadratic, 3 for cubic)
        """
        super().__init__(noise_type, noise_scale, df, random_state)
        self.n_vars = n_vars
        self.max_lag = max_lag
        self.nonlinear_degree = nonlinear_degree
        
        if links is not None:
            self.links = links
        else:
            if n_edges is None:
                n_edges = int(n_vars * 1.5)
            
            # Generate base DAG
            base_links = generate_random_dag(n_vars, n_edges, max_lag, random_state)
            
            # Add nonlinear terms
            self.links = self._add_nonlinear_terms(base_links, nonlinear_degree, random_state)
            
        self.model = StructuralModel(self.links, n_vars=self.n_vars)


    def _add_nonlinear_terms(self, base_links, degree, random_state):
        """
        Convert linear links to nonlinear ones by assigning random powers.
        """
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
            
        new_links = {}
        for (source, lag, target, _), weight in base_links.items():
            # For each link, decide if it's linear or polynomial
            # Let's say 50% chance of being nonlinear
            if np.random.random() < 0.5 and degree > 1:
                power = np.random.randint(2, degree + 1)
            else:
                power = 1
                
            new_links[(source, lag, target, power)] = weight
            
        return new_links

    def generate_multivariate_ts(self, n_points):
        """Generate multivariate time series"""
        X = np.zeros((n_points, self.n_vars))
        var_names = [f'X{i+1}' for i in range(self.n_vars)]
        
        # Initialize first steps with noise
        for i in range(self.max_lag):
            X[i] = self.generate_noise(self.n_vars)

        # Generate time series
        for t in range(self.max_lag, n_points):
            noise = self.generate_noise(self.n_vars)
            current_values = {}
            for i in range(self.n_vars):
                current_values[f'X{i+1}'] = noise[i]
                
            history = {name: X[:, i] for i, name in enumerate(var_names)}
            
            for i in range(self.n_vars):
                target_name = f'X{i+1}'
                structural_val = self.model.compute_value(target_name, t, history, current_values)
                final_val = structural_val + noise[i]
                current_values[target_name] = final_val
                X[t, i] = final_val

        # Create DataFrame
        df = pd.DataFrame(X, columns=var_names)
        df['time'] = np.arange(n_points)
        
        return df

    def save(self, df, output_dir="output"):
        save_dataset_and_graph(
            df, self.n_vars, self.max_lag, len(df), 
            self.noise_type, self.links, self.model.equations, 
            output_dir
        )
