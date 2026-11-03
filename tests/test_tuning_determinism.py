"""Tuning must give the same setting for the same cell on every run, and must not
reuse a cache entry made under a different tuning seed or trial budget."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("optuna")
from dtcd.baselines.tuning import TuningManager  # noqa: E402


def _cell():
    gen = pytest.importorskip("generate_datasets")
    from dtcd.benchmarks.predefined_graphs import PREDEFINED_GRAPHS
    g = PREDEFINED_GRAPHS["d4_tau2"]
    W0, Wl, _ = gen.get_stable_weights(g, gen.stable_weight_seed("d4_tau2"))
    X = gen.generate_ts(g, "d4_tau2", "linear", 500, 41)[0]
    return X, W0, Wl, g["tau_max"]


@pytest.mark.parametrize("method", ["DYNOTEARS", "GrangerF", "PCMCI_ParCorr"])
def test_same_cell_same_setting(method, tmp_path):
    X, W0, Wl, tau = _cell()
    meta = {"track": "d4_tau2_linear", "d": 4, "T": 500, "tune_seed": 41}
    hp_a, src_a = TuningManager(cache_dir=str(tmp_path / "a"), n_trials=6).get_best_hp(method, X, W0, Wl, tau, meta)
    hp_b, src_b = TuningManager(cache_dir=str(tmp_path / "b"), n_trials=6).get_best_hp(method, X, W0, Wl, tau, meta)
    assert src_a == src_b == "tuned"
    assert hp_a == hp_b


def test_cache_key_covers_tuning_seed_and_budget(tmp_path):
    X, W0, Wl, tau = _cell()
    base = {"track": "d4_tau2_linear", "d": 4, "T": 500, "tune_seed": 41}
    mgr = TuningManager(cache_dir=str(tmp_path / "c"), n_trials=6)
    mgr.get_best_hp("GrangerF", X, W0, Wl, tau, base)
    _, src_same = mgr.get_best_hp("GrangerF", X, W0, Wl, tau, base)
    _, src_other_seed = mgr.get_best_hp("GrangerF", X, W0, Wl, tau, {**base, "tune_seed": 42})
    other_budget = TuningManager(cache_dir=str(tmp_path / "c"), n_trials=9)
    _, src_other_budget = other_budget.get_best_hp("GrangerF", X, W0, Wl, tau, base)
    assert src_same == "cached"
    assert src_other_seed == "tuned"
    assert src_other_budget == "tuned"
