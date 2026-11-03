import numpy as np
import pandas as pd
from typing import Dict, Union, List

from .base import BaseTimeSeriesGenerator
from ..structure.scheduler import AdjacencyScheduler, RegimeScheduler
from ..mechanisms.base import Mechanism, LinearMechanism
from ..noise.models import NoiseModel, GaussianNoise
from ..interventions.base import InterventionManager
from ..observation.process import ObservationProcess

class UniversalTimeSeriesGenerator(BaseTimeSeriesGenerator):
    """
    A comprehensive generator that supports:
    - Time-varying graphs (AdjacencyScheduler)
    - Arbitrary mechanisms (Linear, MLP, etc.)
    - Flexible noise models (Gaussian, Heteroskedastic, etc.)
    - Interventions (Do, Soft)
    - Observation processes (Censoring, Quantization)
    """
    def __init__(self, 
                 n_vars: int, 
                 max_lag: int,
                 scheduler: AdjacencyScheduler,
                 mechanisms: Dict[str, Mechanism] = None, # Map variable -> Mechanism
                 noise_model: NoiseModel = None,
                 intervention_manager: InterventionManager = None,
                 observation_process: ObservationProcess = None,
                 random_state=None):
        
        super().__init__(random_state=random_state)
        self.n_vars = n_vars
        self.max_lag = max_lag
        self.scheduler = scheduler
        self.mechanisms = mechanisms if mechanisms else {}
        self.noise_model = noise_model if noise_model else GaussianNoise(random_state=random_state)
        self.intervention_manager = intervention_manager if intervention_manager else InterventionManager()
        self.observation_process = observation_process if observation_process else ObservationProcess(random_state=random_state)
        
        # Default mechanism if not provided
        if not self.mechanisms:
            # We need a default mechanism factory. 
            # For now, let's assume the user MUST provide mechanisms or we default to Linear
            # But LinearMechanism needs weights.
            # This generator assumes the Scheduler provides the weights (adjacency).
            # If Scheduler provides weights, do we need Mechanism weights?
            # YES. Adjacency usually implies existence of edge. Mechanism defines functional form.
            # In Linear, Adjacency weight IS the mechanism parameter.
            # In Nonlinear, Adjacency might just be binary mask, and Mechanism has its own parameters.
            
            # HYBRID APPROACH:
            # 1. Scheduler returns `links` dict: {(src, lag, tgt): weight}
            # 2. If Mechanism is Linear, it uses these weights.
            # 3. If Mechanism is Nonlinear, it might ignore weights or use them as scaling factors.
            pass

    def generate_multivariate_ts(self, n_points: int, noise_data: np.ndarray = None,
                                 obs_noise_data: np.ndarray = None) -> pd.DataFrame:
        """
        Generate time series.
        noise_data: Optional pre-generated structural noise for counterfactuals.
        obs_noise_data: Optional pre-generated measurement noise for counterfactuals.
        Both realizations of the latest call are stored in ``last_noise`` and
        ``last_obs_noise``.
        """
        X = np.zeros((n_points, self.n_vars))
        var_names = [f'X{i+1}' for i in range(self.n_vars)]
        
        # Generate or use provided noise
        if noise_data is None:
            # We need to generate noise step-by-step or all at once?
            # NoiseModel.sample takes size.
            # Simple noise models (Gaussian) are i.i.d.
            # Heteroskedastic depends on parents -> must be step-by-step.
            # So we generally need step-by-step noise generation.
            # But for counterfactuals, we need to save it.
            # Let's store generated noise.
            noise_history = np.zeros((n_points, self.n_vars))
        else:
            noise_history = noise_data
            
        # Initialization (first max_lag steps)
        # We treat them as noise or zero
        for t in range(self.max_lag):
            if noise_data is None:
                n_val = self.noise_model.sample(size=1)[0] # Assuming sample returns (1, n_vars) or similar?
                # My NoiseModel.sample returns (size,) or (size, n_vars)?
                # Let's check NoiseModel implementation. It returns (size,). 
                # Wait, GaussianNoise.sample returns (size,).
                # We need (n_vars,) for one time step.
                # So we call it n_vars times or update NoiseModel to return vector.
                # Let's assume we call it with size=n_vars for now.
                n_val = self.noise_model.sample(size=self.n_vars)
                noise_history[t] = n_val
            else:
                n_val = noise_data[t]
            X[t] = n_val

        # Main loop
        for t in range(self.max_lag, n_points):
            # 1. Get current graph structure
            links = self.scheduler.get_adjacency_at(t)
            
            # 2. Get noise for this step
            if noise_data is None:
                # For heteroskedastic, we need parents values.
                # But parents are not fully known if we have contemporaneous links.
                # Let's assume lagged parents for noise scaling for now to avoid cycle.
                # Or pass None and let NoiseModel handle it (e.g. constant variance).
                n_val = self.noise_model.sample(size=self.n_vars)
                noise_history[t] = n_val
            else:
                n_val = noise_data[t]
                
            current_values = {}
            # Initialize with noise (this acts as the exogenous variable U)
            for i in range(self.n_vars):
                current_values[f'X{i+1}'] = n_val[i]
                
            # 3. Compute structural equations
            # We iterate variables. Topological order is needed for lag 0.
            # Assuming X1..Xn order is valid or we use fixed order.
            
            history = {name: X[:, i] for i, name in enumerate(var_names)}
            
            for i in range(self.n_vars):
                target_name = f'X{i+1}'
                
                # Identify parents from links
                # links: {(src, lag, tgt): weight}
                parents_indices = []
                parents_vals = []
                
                # We need to collect all parents for this target
                # This search is inefficient inside the loop (O(Edges)).
                # Optimization: Pre-process links into adjacency list for this step?
                # Since links change per step (Scheduler), we might have to pay this cost 
                # or Scheduler returns optimized structure.
                
                # Let's do simple search for now.
                term_value = 0
                
                # Check for Hard Interventions (Do)
                SENTINEL = object()
                ival = self.intervention_manager.get_intervention_value(
                    target_name, t, SENTINEL, history=history, current_values=current_values, only_type='do'
                )
                if ival is not SENTINEL:
                    X[t, i] = ival
                    continue

                # Calculate structural part
                # We support arbitrary mechanisms.
                # If mechanism is provided for this var, use it.
                # Else default to Linear using link weights.
                
                mech = self.mechanisms.get(target_name)
                
                relevant_links = []
                for link_key, weight in links.items():
                    if len(link_key) == 4:
                        src, lag, tgt, power = link_key
                    else:
                        src, lag, tgt = link_key
                        
                    if tgt == target_name:
                        relevant_links.append((src, lag, weight))
                        
                # Sort to ensure consistent input order for Mechanism
                relevant_links.sort()
                
                my_parents = []
                for src, lag, weight in relevant_links:
                    if lag == 0:
                        val = current_values[src]
                    else:
                        val = history[src][t - abs(lag)]
                    my_parents.append(val)
                
                if mech:
                    term_value = mech(np.array(my_parents))
                    final_val = mech.combine_noise(term_value, n_val[i])
                    final_val = mech.post_nonlinear(final_val)
                else:
                    # Default linear fallback
                    term_value = 0.0
                    for (src, lag, weight), pval in zip(relevant_links, my_parents):
                        term_value += weight * pval
                    final_val = term_value + n_val[i]
                
                # Apply Soft/Noise interventions via manager
                final_val = self.intervention_manager.get_intervention_value(
                    target_name, t, final_val, history=history, current_values=current_values, only_type='soft'
                )
                
                current_values[target_name] = final_val
                X[t, i] = final_val
                
        # 4. Observation Process (Censoring/Quantization)
        X_obs = self.observation_process.apply_censoring(X)
        X_obs = self.observation_process.apply_quantization(X_obs)
        X_obs = self.observation_process.apply_observation_noise(X_obs, noise=obs_noise_data)
        self.last_obs_noise = self.observation_process.last_noise
        
        df = pd.DataFrame(X_obs, columns=var_names)
        df['time'] = np.arange(n_points)
        
        # Return noise history for counterfactuals
        # We can attach it to df metadata or return tuple.
        # Let's attach to the object for now or return tuple if requested.
        # For API consistency, let's just return df, but maybe store noise in self.last_noise
        self.last_noise = noise_history
        
        return df

    def save(self, df, output_dir="output"):
        """
        Save dataset and metadata.
        """
        import json
        from pathlib import Path
        from ..utils.visualization import save_dataset_and_graph
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Save Data
        filename = f'{output_dir}/universal_ts_n{len(df)}_vars{self.n_vars}.csv'
        df.to_csv(filename, index=False)
        
        # 2. Save Metadata (JSON sidecar)
        metadata = {
            'n_vars': self.n_vars,
            'max_lag': self.max_lag,
            'n_points': len(df),
            'scheduler': self.scheduler.__class__.__name__,
            'noise_model': self.noise_model.__class__.__name__,
            'mechanisms': {k: v.__class__.__name__ for k, v in self.mechanisms.items()},
            'interventions': self.intervention_manager.interventions,
            # We can also save the graph structure if it's static, or a summary if dynamic
        }
        
        with open(f'{output_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
            
        # 3. Save Graph (Snapshot at t=0 or representative)
        # Since graph is dynamic, we might want to save a few snapshots
        # or just t=max_lag
        t_snapshot = self.max_lag
        links = self.scheduler.get_adjacency_at(t_snapshot)
        
        # Convert links to expected format for visualization if needed
        # Visualization expects {(src, lag, tgt): weight}
        # We have that.
        
        # We need equations list. 
        # Universal generator doesn't maintain a list of string equations easily 
        # because mechanisms are callables.
        # We can generate dummy strings.
        equations = [f"X{i+1} = f(parents)" for i in range(self.n_vars)]
        
        save_dataset_and_graph(
            df, self.n_vars, self.max_lag, len(df), 
            "universal", links, equations, 
            output_dir
        )
