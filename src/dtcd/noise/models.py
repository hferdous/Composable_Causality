import numpy as np
from scipy import stats
from abc import ABC, abstractmethod

class NoiseModel(ABC):
    """
    Abstract base class for noise models.
    """
    def __init__(self, random_state=None):
        self.rng = np.random.RandomState(random_state)

    @abstractmethod
    def sample(self, size: int = 1, parents_values: np.ndarray = None) -> np.ndarray:
        """
        Sample noise. 
        parents_values: Optional, for heteroskedastic noise dependent on parents.
        """
        pass

class GaussianNoise(NoiseModel):
    def __init__(self, scale=0.1, random_state=None):
        super().__init__(random_state)
        self.scale = scale
        
    def sample(self, size=1, parents_values=None):
        return self.rng.normal(0, self.scale, size=size)

class StudentTNoise(NoiseModel):
    def __init__(self, df=3, scale=0.1, random_state=None):
        super().__init__(random_state)
        self.df = df
        self.scale = scale
        # Use scipy's t distribution but seed it
        stats.t.random_state = self.rng
        
    def sample(self, size=1, parents_values=None):
        return stats.t.rvs(df=self.df, loc=0, scale=self.scale, size=size)

class LaplaceNoise(NoiseModel):
    def __init__(self, scale=0.1, random_state=None):
        super().__init__(random_state)
        self.scale = scale
        
    def sample(self, size=1, parents_values=None):
        return self.rng.laplace(0, self.scale, size=size)

class HeteroskedasticNoise(NoiseModel):
    """
    Variance depends on the magnitude of parents.
    sigma = base_scale + alpha * |sum(parents)|
    """
    def __init__(self, base_scale=0.1, alpha=0.05, random_state=None):
        super().__init__(random_state)
        self.base_scale = base_scale
        self.alpha = alpha
        
    def sample(self, size=1, parents_values=None):
        if parents_values is None:
            scale = self.base_scale
        else:
            # Simple dependency: scale increases with magnitude of parents
            magnitude = np.abs(np.sum(parents_values))
            scale = self.base_scale + self.alpha * magnitude
            
        return self.rng.normal(0, scale, size=size)

class MixedNoise(NoiseModel):
    """
    Mixture of two noise distributions (e.g. Gaussian + Laplace).
    """
    def __init__(self, model1: NoiseModel, model2: NoiseModel, weight1=0.5, random_state=None):
        super().__init__(random_state)
        self.model1 = model1
        self.model2 = model2
        self.weight1 = weight1
        
    def sample(self, size=1, parents_values=None):
        # Mixture: sample from both and combine? Or select one?
        # Usually additive noise is N1 + N2 or mixture distribution (select N1 with prob p).
        # Let's do additive for "Mixed Noise" as per user request "Additive + Multiplicative" 
        # Wait, user said "Mixed (additive + multiplicative noise)".
        # That usually means Y = f(X) + N_add + N_mult * f(X).
        # But here we are just sampling the noise term E.
        # If user meant Mixture Distribution (e.g. outliers), we select.
        # Let's implement Additive combination of two noise sources for now.
        
        n1 = self.model1.sample(size, parents_values)
        n2 = self.model2.sample(size, parents_values)
        return self.weight1 * n1 + (1 - self.weight1) * n2

class ExponentialNoise(NoiseModel):
    def __init__(self, scale=1.0, random_state=None):
        super().__init__(random_state)
        self.scale = scale
        
    def sample(self, size=1, parents_values=None):
        # Center the exponential noise so its mean is 0
        return self.rng.exponential(self.scale, size=size) - self.scale

class UniformNoise(NoiseModel):
    def __init__(self, low=-0.1, high=0.1, random_state=None):
        super().__init__(random_state)
        self.low = low
        self.high = high
        
    def sample(self, size=1, parents_values=None):
        return self.rng.uniform(self.low, self.high, size=size)

class CauchyNoise(NoiseModel):
    def __init__(self, location=0.0, scale=0.1, random_state=None):
        super().__init__(random_state)
        self.location = location
        self.scale = scale
        
    def sample(self, size=1, parents_values=None):
        return self.location + self.scale * self.rng.standard_cauchy(size=size)

class NonStationaryNoise(NoiseModel):
    """
    Noise model where the scale/variance changes over time.
    Internal counter `t` is advanced each time `sample()` is called.
    """
    def __init__(self, initial_scale=0.1, drift=0.001, random_state=None):
        super().__init__(random_state)
        self.initial_scale = initial_scale
        self.drift = drift
        self.t = 0
        
    def sample(self, size=1, parents_values=None):
        current_scale = max(0.0001, self.initial_scale + self.drift * self.t)
        out = self.rng.normal(0, current_scale, size=size)
        self.t += 1
        return out
