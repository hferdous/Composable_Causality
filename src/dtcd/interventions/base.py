import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class Intervention(ABC):
    """
    Abstract base class for interventions.
    """
    @abstractmethod
    def apply(self, target_node: str, current_val: float, t: int) -> float:
        pass

class DoIntervention(Intervention):
    """
    Hard intervention: do(X = c)
    """
    def __init__(self, value: float):
        self.value = value
        
    def apply(self, target_node: str, current_val: float, t: int) -> float:
        return self.value

class SoftIntervention(Intervention):
    """
    Soft intervention: Change mechanism or add shift.
    X = f(parents) + shift
    """
    def __init__(self, shift: float):
        self.shift = shift
        
    def apply(self, target_node: str, current_val: float, t: int) -> float:
        return current_val + self.shift

class NoiseIntervention(Intervention):
    """
    Change noise level.
    Note: This is usually applied during noise generation, not post-hoc.
    But if we view it as X = f(parents) + N_new, we can simulate it if we have access to N_old.
    Here we assume this intervention is handled by the generator logic swapping the noise model.
    So this class might be a marker.
    """
    def __init__(self, scale_multiplier: float):
        self.scale_multiplier = scale_multiplier
        
    def apply(self, target_node: str, current_val: float, t: int) -> float:
        # This is tricky post-hoc without separating noise.
        # Ideally, the generator checks for this intervention before adding noise.
        return current_val

class InterventionManager:
    """
    Manages interventions for a data generation process.
    """
    def __init__(self):
        self.interventions: Dict[str, List[Dict]] = {} 
        # Structure: {'X1': [{'type': 'do', 'value': 5, 'start': 10, 'end': 20}]}

    def add_intervention(self, target: str, intervention_type: str, value: float = None, start: int = 0, end: int = None, condition=None, policy=None):
        """
        condition: Callable[[int, Dict[str, np.ndarray], Dict[str, float]], bool]
        policy: Callable[[int, Dict[str, np.ndarray], Dict[str, float]], float]
        """
        if target not in self.interventions:
            self.interventions[target] = []
            
        self.interventions[target].append({
            'type': intervention_type,
            'value': value,
            'start': start,
            'end': end,
            'condition': condition,
            'policy': policy
        })

    def get_intervention_value(self, target: str, t: int, current_val: float, history: Dict[str, np.ndarray] = None, current_values: Dict[str, float] = None, only_type: str = None) -> float:
        """
        Check if any intervention is active for target at time t and apply it.
        """
        if target not in self.interventions:
            return current_val
            
        final_val = current_val
        
        for rule in self.interventions[target]:
            if only_type is not None and rule['type'] != only_type:
                continue
                
            start = rule['start']
            end = rule['end'] if rule['end'] is not None else float('inf')
            
            # Check custom condition
            if rule['condition'] is not None:
                if not rule['condition'](t, history, current_values):
                    continue
            
            if start <= t < end:
                itype = rule['type']
                
                # Resolve value
                if rule['policy'] is not None:
                    val = rule['policy'](t, history, current_values)
                else:
                    val = rule['value']
                
                if itype == 'do':
                    final_val = val
                    return final_val
                elif itype == 'soft':
                    final_val += val
                elif itype == 'noise':
                    pass
                    
        return final_val

class CounterfactualSuite:
    """
    Generates observational and interventional twin datasets utilizing the identically sampled noise realizations.
    """
    def __init__(self, generator):
        self.generator = generator
        
    def generate_twins(self, n_points: int, intervention_manager_for_twin: InterventionManager):
        # 1. Generate observational baseline
        obs_df = self.generator.generate_multivariate_ts(n_points=n_points)
        noise = getattr(self.generator, 'last_noise', None)
        if noise is None:
            raise ValueError("Generator must store its noise in .last_noise for counterfactuals to be perfectly aligned.")
            
        # 2. Swap intervention manager
        original_im = self.generator.intervention_manager
        self.generator.intervention_manager = intervention_manager_for_twin
        
        # 3. Generate interventional twin with same frozen noise (and the same
        #    measurement noise, for generators that add it)
        extra = {}
        obs_noise = getattr(self.generator, 'last_obs_noise', None)
        if obs_noise is not None:
            extra['obs_noise_data'] = obs_noise
        int_df = self.generator.generate_multivariate_ts(n_points=n_points, noise_data=noise, **extra)
        
        # 4. Restore original state
        self.generator.intervention_manager = original_im
        
        return obs_df, int_df
