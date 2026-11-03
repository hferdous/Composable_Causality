import numpy as np
import pandas as pd

class ObservationProcess:
    """
    Handles observation artifacts like censoring and quantization.
    """
    def __init__(self, random_state=None):
        self.rng = np.random.RandomState(random_state)

    def apply_censoring(self, data: np.ndarray, lower: float = None, upper: float = None) -> np.ndarray:
        """
        Apply censoring (clipping) to the data.
        """
        if lower is None and upper is None:
            return data
        return np.clip(data, lower, upper)

    def apply_quantization(self, data: np.ndarray, decimals: int = None, bins: int = None) -> np.ndarray:
        """
        Apply quantization (rounding or binning).
        """
        if decimals is not None:
            return np.round(data, decimals)
        elif bins is not None:
            # Bin into 'bins' number of intervals
            # This is a bit complex for continuous data without range.
            # Let's assume we digitize based on min/max of current data
            min_val = np.min(data)
            max_val = np.max(data)
            bins_edges = np.linspace(min_val, max_val, bins + 1)
            # We return the midpoints of bins? or the bin index?
            # Usually quantization implies mapping to discrete values.
            # Let's return bin midpoints.
            indices = np.digitize(data, bins_edges) - 1
            indices = np.clip(indices, 0, bins - 1)
            midpoints = (bins_edges[:-1] + bins_edges[1:]) / 2
            return midpoints[indices]
        else:
            return data

    def apply_observation_noise(self, data: np.ndarray, scale: float = 0.01,
                                noise: np.ndarray = None) -> np.ndarray:
        """
        Add independent measurement noise.

        Pass ``noise`` to replay a previous realization (the counterfactual twin
        of a trajectory must see the same measurement noise as the factual one).
        The realization used is stored in ``self.last_noise``.
        """
        if noise is None:
            noise = self.rng.normal(0, scale, size=data.shape)
        self.last_noise = noise
        return data + noise

    def apply_mnar(self, data: np.ndarray, threshold: float = None, above: bool = True, prob: float = 1.0) -> np.ndarray:
        """
        Apply Missing Not At Random (MNAR).
        Values beyond the threshold become NaN with a certain probability.
        """
        if threshold is None:
            return data
            
        data_mnar = data.astype(float).copy()
        
        if above:
            condition_mask = data_mnar > threshold
        else:
            condition_mask = data_mnar < threshold
            
        if prob < 1.0:
            rand_mask = self.rng.random(size=data.shape) < prob
            mask = condition_mask & rand_mask
        else:
            mask = condition_mask
            
        data_mnar[mask] = np.nan
        return data_mnar
