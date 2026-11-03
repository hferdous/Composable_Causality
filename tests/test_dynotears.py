"""DYNOTEARS recovers the d4_tau2 reference graph on raw linear data."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dtcd.benchmarks.predefined_graphs import PREDEFINED_GRAPHS  # noqa: E402
from dtcd.evaluators.runner import run_single_experiment  # noqa: E402
from generate_datasets import generate_ts  # noqa: E402


def test_dynotears_recovers_reference_cell():
    gid = "d4_tau2"
    g = PREDEFINED_GRAPHS[gid]
    X, _mask, W0, W_lags, *_ = generate_ts(g, gid, "linear", 1000, 42)
    res = run_single_experiment("DYNOTEARS", X, W0, W_lags, g["tau_max"], 42,
                                timeout_s=300, lambda_w=0.0044, lambda_a=0.0043,
                                w_threshold=0.1)
    assert res["status"] == "ok"
    assert res["f1_orient"] >= 0.9
    assert np.isfinite(res["shd_joint"])
