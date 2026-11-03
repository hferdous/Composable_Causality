import numpy as np
from .base import Mechanism

class SoftmaxMechanism(Mechanism):
    """
    Generates categorical data from parents.
    f(parents) -> class_index
    
    The output is the index of the class sampled from softmax(W * parents + b).
    """
    def __init__(self, n_classes: int, input_dim: int, random_state=None):
        self.n_classes = n_classes
        self.rng = np.random.RandomState(random_state)
        # Weights: (n_classes, input_dim)
        self.W = self.rng.randn(n_classes, input_dim) * 0.5
        self.b = self.rng.randn(n_classes) * 0.1
        
    def __call__(self, parents_values: np.ndarray) -> float:
        # logits = Wx + b
        logits = np.dot(self.W, parents_values) + self.b
        # softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        # sample
        return float(self.rng.choice(self.n_classes, p=probs))

class PoissonMechanism(Mechanism):
    """
    Generates count data from parents.
    f(parents) -> count
    
    lambda = exp(w * parents + b)
    val ~ Poisson(lambda)
    """
    def __init__(self, input_dim: int, random_state=None):
        self.rng = np.random.RandomState(random_state)
        self.w = self.rng.randn(input_dim) * 0.1
        self.b = self.rng.randn() * 0.0
        
    def __call__(self, parents_values: np.ndarray) -> float:
        log_lambda = np.dot(self.w, parents_values) + self.b
        # Clip log_lambda to avoid overflow
        log_lambda = np.clip(log_lambda, -5, 5)
        lam = np.exp(log_lambda)
        return float(self.rng.poisson(lam))
