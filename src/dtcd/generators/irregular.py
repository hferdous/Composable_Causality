import numpy as np
import pandas as pd
from ..structure.graph import generate_random_dag
from ..structure.equations import StructuralModel
from .base import BaseTimeSeriesGenerator
from ..utils.visualization import save_dataset_and_graph

class IrregularTimeSeriesGenerator(BaseTimeSeriesGenerator):
    def __init__(self, n_vars=4, max_lag=2, n_edges=None, links=None, 
                 noise_type='gaussian', noise_scale=0.1, df=3, random_state=None):
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


    def generate_irregular_timestamps(self, n_points, total_time, min_gap=0.1):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        times = np.zeros(n_points)
        times[0] = np.random.uniform(0, min_gap)

        for i in range(1, n_points):
            # Exponential gap distribution
            gap = np.random.exponential(scale=(total_time-times[i-1])/(n_points-i))
            times[i] = times[i-1] + max(gap, min_gap)

        # Normalize to total_time if exceeded (simple scaling)
        if times[-1] > total_time:
             times = times * (total_time / times[-1])
             
        return times

    def find_nearest_lag_idx(self, timestamps, current_idx, lag_time):
        target_time = timestamps[current_idx] - lag_time
        past_timestamps = timestamps[:current_idx]
        if len(past_timestamps) == 0:
            return 0
        return (np.abs(past_timestamps - target_time)).argmin()

    def generate_multivariate_ts(self, n_points, total_time=100):
        """Generate multivariate time series with irregular sampling"""
        X = np.zeros((n_points, self.n_vars))
        var_names = [f'X{i+1}' for i in range(self.n_vars)]
        
        timestamps = self.generate_irregular_timestamps(n_points, total_time)
        mean_diff = total_time / n_points # Approximation for lag unit

        # Initialize first steps with noise
        for i in range(self.max_lag):
            X[i] = self.generate_noise(self.n_vars)

        # Generate time series
        for t in range(self.max_lag, n_points):
            noise = self.generate_noise(self.n_vars)
            current_values = {}
            for i in range(self.n_vars):
                current_values[f'X{i+1}'] = noise[i]
                
            # For irregular sampling, we can't just index [t-lag].
            # We need to find the index corresponding to time t - lag*mean_diff
            
            # We need to construct a "virtual" history for the compute_value function
            # or modify compute_value to accept indices.
            # Let's construct a history where history[var][t-lag] maps to the nearest neighbor value.
            # But compute_value expects a fixed array.
            
            # Hack: We can pre-calculate the indices for this time step t
            # and create a temporary history array that `compute_value` can index into using standard lags.
            # e.g. temp_history[var][t-lag] = X[nearest_idx, var_idx]
            
            temp_history = {name: np.zeros(n_points) for name in var_names}
            
            # Populate temp history for relevant lags
            for lag in range(1, self.max_lag + 1):
                lag_time = lag * mean_diff
                nearest_idx = self.find_nearest_lag_idx(timestamps, t, lag_time)
                # We map the "virtual" index (t-lag) to the actual value at nearest_idx
                virtual_idx = t - lag
                if virtual_idx >= 0:
                    for i, name in enumerate(var_names):
                        temp_history[name][virtual_idx] = X[nearest_idx, i]
            
            # Now compute
            for i in range(self.n_vars):
                target_name = f'X{i+1}'
                structural_val = self.model.compute_value(target_name, t, temp_history, current_values)
                final_val = structural_val + noise[i]
                current_values[target_name] = final_val
                X[t, i] = final_val

        # Create DataFrame
        df = pd.DataFrame(X, columns=var_names)
        df['time'] = timestamps
        
        return df

    def save(self, df, output_dir="output"):
        save_dataset_and_graph(
            df, self.n_vars, self.max_lag, len(df), 
            self.noise_type, self.links, self.model.equations, 
            output_dir
        )
