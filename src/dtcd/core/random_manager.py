import numpy as np
from scipy import stats

class RandomStateManager:
    """
    Centralized manager for random states to ensure reproducibility across
    different components (graph, mechanism, noise, observation).
    """
    def __init__(self, seed=None):
        self.master_seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Generate sub-seeds for different components
        # We generate a large number of seeds to avoid collisions
        self.seeds = {
            'graph': self.rng.randint(0, 2**31),
            'mechanism': self.rng.randint(0, 2**31),
            'noise': self.rng.randint(0, 2**31),
            'observation': self.rng.randint(0, 2**31),
            'intervention': self.rng.randint(0, 2**31)
        }
        
    def get_rng(self, component):
        """Get a numpy RandomState for a specific component."""
        if component not in self.seeds:
            # If a new component is requested, generate a deterministic seed for it
            # based on the master seed and component name hash
            if self.master_seed is not None:
                component_hash = int(hashlib.md5(str(component).encode()).hexdigest()[:8], 16) % (2**31)
                seed = (self.master_seed + component_hash) % (2**31)
            else:
                seed = None
            self.seeds[component] = seed
            
        return np.random.RandomState(self.seeds[component])

    def get_seed(self, component):
        """Get the integer seed for a specific component."""
        if component not in self.seeds:
            self.get_rng(component) # This will generate and store the seed
        return self.seeds[component]
