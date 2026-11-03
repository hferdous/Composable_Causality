import numpy as np
import pandas as pd
from typing import List, Dict, Union
from ..generators.base import BaseTimeSeriesGenerator
from ..generators.linear import LinearTimeSeriesGenerator

class PanelGenerator:
    """
    Generates panel data (multiple entities/units) sharing a common causal structure.
    """
    def __init__(self, n_units: int, n_vars: int, max_lag: int, 
                 base_generator_class=LinearTimeSeriesGenerator,
                 base_params: Dict = None,
                 heterogeneity: str = 'noise', # 'noise', 'params', 'none'
                 random_state=None):
        
        self.n_units = n_units
        self.n_vars = n_vars
        self.max_lag = max_lag
        self.base_generator_class = base_generator_class
        self.base_params = base_params if base_params else {}
        self.heterogeneity = heterogeneity
        self.rng = np.random.RandomState(random_state)
        
        # Ensure base params has n_vars and max_lag
        self.base_params['n_vars'] = n_vars
        self.base_params['max_lag'] = max_lag
        
        # Initialize generators for each unit
        self.generators = self._initialize_generators()

    def _initialize_generators(self) -> List[BaseTimeSeriesGenerator]:
        generators = []
        
        # Master generator to define shared structure
        master_gen = self.base_generator_class(random_state=self.rng.randint(0, 2**31), **self.base_params)
        shared_links = getattr(master_gen, 'links', {})
        
        for i in range(self.n_units):
            unit_seed = self.rng.randint(0, 2**31)
            
            if self.heterogeneity == 'params':
                # Independent structure
                gen = self.base_generator_class(random_state=unit_seed, **self.base_params)
                generators.append(gen)
                
            elif self.heterogeneity == 'noise':
                # Shared structure, different noise
                # We pass the shared links to the new generator
                # Note: This requires the generator class to accept 'links' in __init__
                # We just updated LinearTimeSeriesGenerator to support this.
                # For others, we might need similar updates or kwargs handling.
                
                params = self.base_params.copy()
                params['links'] = shared_links
                
                gen = self.base_generator_class(random_state=unit_seed, **params)
                generators.append(gen)
                
            else: # 'none'
                # Identical generator (same seed would mean identical data if we re-instantiated)
                # But we want same structure, same parameters, just different noise realization *during generation*
                # If we use the same generator instance, it will generate new noise each time called.
                # So we can just reuse the master_gen? 
                # Yes, if generate_multivariate_ts is stateless regarding the model (it is).
                generators.append(master_gen)
        
        return generators

    def generate_panel(self, n_points: int) -> pd.DataFrame:
        """
        Generate panel data.
        Returns DataFrame with 'unit_id', 'time', 'X1', ...
        """
        # Placeholder implementation until we refactor generators to accept external graphs
        # For now, we will just generate independent units if heterogeneity='params'
        # or just warn about shared structure limitation.
        
        dfs = []
        for i, gen in enumerate(self.generators):
            df = gen.generate_multivariate_ts(n_points)
            df.insert(0, 'unit_id', i)
            dfs.append(df)
            
        return pd.concat(dfs, ignore_index=True)
