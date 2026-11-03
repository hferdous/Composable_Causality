import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

class AdjacencyScheduler(ABC):
    """
    Abstract base class for scheduling adjacency matrix changes over time.
    """
    def __init__(self, n_vars: int, max_lag: int, random_state=None):
        self.n_vars = n_vars
        self.max_lag = max_lag
        self.rng = np.random.RandomState(random_state)

    @abstractmethod
    def get_adjacency_at(self, t: int) -> Dict[Tuple[str, int, str], float]:
        """
        Return the adjacency list (links) at time t.
        Format: {(source, lag, target): weight}
        """
        pass

class RegimeScheduler(AdjacencyScheduler):
    """
    Switches between pre-defined adjacency structures at specific time points.
    """
    def __init__(self, n_vars, max_lag, regimes: List[Dict], random_state=None):
        super().__init__(n_vars, max_lag, random_state)
        # regimes: [{'start': 0, 'links': {...}}, {'start': 100, 'links': {...}}]
        self.regimes = sorted(regimes, key=lambda x: x['start'])
        
    def get_adjacency_at(self, t: int):
        # Find the active regime
        active_regime = self.regimes[0]
        for regime in self.regimes:
            if t >= regime['start']:
                active_regime = regime
            else:
                break
        return active_regime['links']

class SmoothDriftScheduler(AdjacencyScheduler):
    """
    Edge weights drift over time using a random walk.
    """
    def __init__(self, n_vars, max_lag, initial_links: Dict, drift_scale=0.01, random_state=None):
        super().__init__(n_vars, max_lag, random_state)
        self.current_links = initial_links.copy()
        self.drift_scale = drift_scale
        # Weights already drawn for each time step. A second rollout (for
        # example the counterfactual twin) reuses them, so both rollouts see
        # the same time-varying graph.
        self._history: Dict[int, Dict] = {}

    def get_adjacency_at(self, t: int):
        if t in self._history:
            return self._history[t]
        # Advance the random walk by one step for each new time index.
        new_links = {}
        for key, weight in self.current_links.items():
            drift = self.rng.normal(0, self.drift_scale)
            new_links[key] = weight + drift
        self.current_links = new_links
        self._history[t] = new_links
        return new_links

class PiecewiseConstantScheduler(AdjacencyScheduler):
    """
    Similar to RegimeScheduler but generated automatically.
    Changes graph structure every `period` steps.
    """
    def __init__(self, n_vars, max_lag, period=100, n_changes=5, base_graph_generator=None, random_state=None):
        super().__init__(n_vars, max_lag, random_state)
        self.period = period
        self.base_graph_generator = base_graph_generator
        self.schedule = {}
        
        # Pre-generate regimes
        for i in range(n_changes):
            start_time = i * period
            if base_graph_generator:
                links = base_graph_generator(n_vars, max_lag) # Assuming this signature
            else:
                links = {} # Empty or default
            self.schedule[start_time] = links
            
    def get_adjacency_at(self, t: int):
        # Determine regime index
        regime_idx = t // self.period
        start_time = regime_idx * self.period
        
        if start_time in self.schedule:
            return self.schedule[start_time]
        else:
            # Return last known or generate new if we want infinite
            keys = sorted(self.schedule.keys())
            if keys:
                return self.schedule[keys[-1]]
            return {}
