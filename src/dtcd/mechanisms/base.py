import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Callable

class Mechanism(ABC):
    """
    Abstract base class for structural mechanisms.
    f(parents) -> value
    """
    @abstractmethod
    def __call__(self, parents_values: np.ndarray) -> float:
        pass

    def combine_noise(self, structural_val: float, noise_val: float) -> float:
        """Combine structural value and noise. Default is additive."""
        return structural_val + noise_val

    def post_nonlinear(self, value_with_noise: float) -> float:
        """Apply function after noise addition (PNL). Default is identity."""
        return value_with_noise

class LinearMechanism(Mechanism):
    def __init__(self, weights: np.ndarray):
        self.weights = weights
        
    def __call__(self, parents_values: np.ndarray) -> float:
        return np.dot(self.weights[:len(parents_values)], parents_values)

class PolynomialMechanism(Mechanism):
    """
    f(x) = sum(w_i * x_i^p_i)
    """
    def __init__(self, weights: np.ndarray, powers: np.ndarray):
        self.weights = weights
        self.powers = powers
        
    def __call__(self, parents_values: np.ndarray) -> float:
        return np.dot(self.weights, np.power(parents_values, self.powers))

class TanhMechanism(Mechanism):
    """
    f(x) = tanh(sum(w_i * x_i))
    """
    def __init__(self, weights: np.ndarray):
        self.weights = weights
        
    def __call__(self, parents_values: np.ndarray) -> float:
        return np.tanh(np.dot(self.weights, parents_values))

class MLPMechanism(Mechanism):
    """
    Simple MLP mechanism: f(x) = w_out * relu(W_in * x + b)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 5, random_state=None):
        rng = np.random.RandomState(random_state)
        self.W_in = rng.randn(hidden_dim, input_dim) * 0.1
        self.b_in = rng.randn(hidden_dim) * 0.1
        self.w_out = rng.randn(hidden_dim) * 0.1
        
    def __call__(self, parents_values: np.ndarray) -> float:
        hidden = np.maximum(0, np.dot(self.W_in, parents_values) + self.b_in)
        return np.dot(self.w_out, hidden)

class InteractionMechanism(Mechanism):
    """
    Includes pairwise interactions: f(x) = sum(w_i * x_i) + sum(v_ij * x_i * x_j)
    """
    def __init__(self, linear_weights: np.ndarray, interaction_matrix: np.ndarray):
        self.linear_weights = linear_weights
        self.interaction_matrix = interaction_matrix # Upper triangular
        
    def __call__(self, parents_values: np.ndarray) -> float:
        linear_term = np.dot(self.linear_weights, parents_values)
        
        interaction_term = 0
        n = len(parents_values)
        for i in range(n):
            for j in range(i+1, n):
                interaction_term += self.interaction_matrix[i, j] * parents_values[i] * parents_values[j]
                
        return linear_term + interaction_term

class SplineMechanism(Mechanism):
    """
    Fourier basis expansion (trigonometric spline equivalent). 
    f(x) = sum_i sum_h [w_sin * sin(h*x_i) + w_cos * cos(h*x_i)]
    """
    def __init__(self, weights: np.ndarray, n_harmonics: int = 2):
        self.weights = weights
        self.n_harmonics = n_harmonics
        
    def __call__(self, parents_values: np.ndarray) -> float:
        val = 0.0
        for i, x in enumerate(parents_values):
            for h in range(1, self.n_harmonics + 1):
                w_sin = self.weights[i, (h-1)*2]
                w_cos = self.weights[i, (h-1)*2 + 1]
                val += w_sin * np.sin(h * x) + w_cos * np.cos(h * x)
        return val

class MultiplicativeWrapper(Mechanism):
    """Wraps a mechanism to use multiplicative noise: Y = f(X) * E"""
    def __init__(self, base_mechanism: Mechanism):
        self.base_mechanism = base_mechanism
        
    def __call__(self, parents_values: np.ndarray) -> float:
        return self.base_mechanism(parents_values)
        
    def combine_noise(self, structural_val: float, noise_val: float) -> float:
        return structural_val * noise_val
        
    def post_nonlinear(self, value_with_noise: float) -> float:
        return self.base_mechanism.post_nonlinear(value_with_noise)

def signed_power(x: float, p: float = 1.5) -> float:
    """Strictly monotonic signed power: sign(x) * |x|^p"""
    return np.sign(x) * np.power(np.abs(x), p)

class PNLWrapper(Mechanism):
    """
    Wraps a mechanism to apply a post-nonlinear function: Y = g(f(X) + E).
    g must be strictly monotonic for identifiability.
    """
    def __init__(self, base_mechanism: Mechanism, g_type: str = "signed_power"):
        self.base_mechanism = base_mechanism
        self.g_type = g_type
        
        if g_type == "signed_power":
            self.g_func = lambda x: signed_power(x, 1.5)
        elif g_type == "tanh":
            self.g_func = np.tanh
        elif g_type == "cubic_perturb":
            self.g_func = lambda x: x + 0.1 * x**3
        else:
            self.g_func = lambda x: x # Identity
            
    def __call__(self, parents_values: np.ndarray) -> float:
        return self.base_mechanism(parents_values)
        
    def combine_noise(self, structural_val: float, noise_val: float) -> float:
        return self.base_mechanism.combine_noise(structural_val, noise_val)
        
    def post_nonlinear(self, value_with_noise: float) -> float:
        return self.g_func(self.base_mechanism.post_nonlinear(value_with_noise))
