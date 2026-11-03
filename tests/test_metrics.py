import numpy as np
import pytest
from dtcd.evaluators.metrics import shd, f1_skeleton, f1_oriented, rmse_weights, pehe

def test_shd():
    # Empty vs 3-edge truth
    W_true = np.zeros((5, 5))
    W_true[0, 1] = 1
    W_true[1, 2] = 1
    W_true[2, 3] = 1
    W_pred = np.zeros((5, 5))
    assert shd(W_pred, W_true) == 3
    
    # One extra edge
    W_pred[4, 0] = 1
    assert shd(W_pred, W_true) == 4
    
    # Perfect match
    W_pred = W_true.copy()
    assert shd(W_pred, W_true) == 0

def test_f1_oriented_vs_skeleton():
    # Directed edge 0 -> 1
    W_true = np.zeros((3, 3))
    W_true[0, 1] = 1
    
    # Predict reversed edge 1 -> 0
    W_pred = np.zeros((3, 3))
    W_pred[1, 0] = 1
    
    # Skeleton should be perfect (F1=1.0)
    assert f1_skeleton(W_pred, W_true) == 1.0
    # Oriented should be zero (F1=0.0)
    assert f1_oriented(W_pred, W_true) == 0.0

def test_rmse_weights():
    W_true = np.array([[0, 2.0], [0, 0]])
    W_pred = np.array([[0, 1.5], [0, 0]])
    # (1.5 - 2.0)^2 = 0.25 -> sqrt(0.25) = 0.5
    assert pytest.approx(rmse_weights(W_pred, W_true)) == 0.5

def test_pehe():
    tau_true = np.array([1.0, 2.0, 3.0])
    tau_pred = np.array([1.1, 1.9, 3.2])
    # diffs: 0.1, -0.1, 0.2
    # squares: 0.01, 0.01, 0.04
    # mean: 0.06 / 3 = 0.02
    # sqrt: sqrt(0.02)
    assert pytest.approx(pehe(tau_pred, tau_true)) == np.sqrt(0.02)
