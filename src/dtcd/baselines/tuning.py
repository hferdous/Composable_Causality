from __future__ import annotations
import optuna
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Callable, Optional, Tuple
import numpy as np

from dtcd.baselines.registry import get_discovery_method
from dtcd.evaluators.metrics import f1_oriented, _unroll_adj

logger = logging.getLogger(__name__)

# Bump when a search space in get_tuning_space changes.
SPACE_VERSION = 2


class TuningManager:
    def __init__(self, cache_dir: str = "tuning_cache", n_trials: int = 50):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n_trials = n_trials

    def _get_cell_hash(self, method_name: str, cell_meta: Dict[str, Any]) -> str:
        relevant_meta = {k: v for k, v in cell_meta.items() if isinstance(v, (str, int, float, bool))}
        meta_str = f"{method_name}_{json.dumps(relevant_meta, sort_keys=True)}"
        return hashlib.md5(meta_str.encode()).hexdigest()

    def get_best_hp(
        self,
        method_name: str,
        X_val: np.ndarray,
        W_inst_true: np.ndarray,
        W_lag_true: np.ndarray,
        max_lag: int,
        cell_meta: Dict[str, Any],
        val_seed: int = 0,
    ) -> Tuple[Dict[str, Any], str]:
        n_trials_to_run = self.n_trials
        if method_name in ["PCMCI_GPDC", "DYNOTEARS"]:
            n_trials_to_run = 5

        # The cache key covers everything that changes the selected setting:
        # the tuning trajectory (cell_meta["tune_seed"]), the trial budget, and
        # the search-space version. Entries tuned under a different budget or
        # tuning seed are therefore never reused.
        keyed_meta = {**cell_meta, "n_trials": n_trials_to_run,
                      "space_version": SPACE_VERSION, "val_seed": val_seed}
        cell_hash = self._get_cell_hash(method_name, keyed_meta)
        cache_path = self.cache_dir / method_name / f"{cell_hash}.json"

        if cache_path.exists():
            with open(cache_path, "r") as f:
                return json.load(f), "cached"

        def objective(trial: optuna.Trial) -> float:
            hp = get_tuning_space(method_name, trial)
            try:
                method = get_discovery_method(method_name, max_lag=max_lag, seed=val_seed, **hp)
                res = method.fit(X_val)
                if res.status != "ok":
                    return 0.0
                W_pred_joint = _unroll_adj(res.W_inst, res.W_lag)
                W_true_joint = _unroll_adj(W_inst_true, W_lag_true)
                return f1_oriented(W_pred_joint, W_true_joint)
            except Exception:
                return 0.0

        # Seeded TPE sampler, derived from the cache key, so the same cell is
        # tuned identically on every run. optimize() is single-threaded.
        sampler_seed = int(hashlib.md5(f"{method_name}|{cell_hash}".encode()).hexdigest()[:8], 16)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=sampler_seed),
        )
        study.optimize(objective, n_trials=n_trials_to_run)

        best_hp = study.best_params

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(best_hp, f)

        return best_hp, "tuned"


def get_tuning_space(method_name: str, trial: optuna.Trial) -> Dict[str, Any]:
    if method_name in ["PCMCI_ParCorr", "PCMCI_GPDC", "PCMCI_CMIknn"]:
        return {"pc_alpha": trial.suggest_categorical("pc_alpha", [0.001, 0.01, 0.05])}
    elif method_name == "CDNOD":
        return {"alpha": trial.suggest_categorical("alpha", [0.001, 0.01, 0.05])}
    elif method_name == "VARLiNGAM":
        return {}
    elif method_name == "DYNOTEARS":
        return {
            "lambda_w": trial.suggest_float("lambda_w", 0.001, 1.0, log=True),
            "lambda_a": trial.suggest_float("lambda_a", 0.001, 1.0, log=True),
            "w_threshold": 0.1,
        }
    elif method_name == "GrangerF":
        return {"alpha": trial.suggest_categorical("alpha", [0.001, 0.01, 0.05])}
    elif method_name == "CUTS+":
        # Added `threshold` so the binarization cutoff is tuned per cell.
        # Without this the wrapper kept every continuous coefficient as an
        # edge, producing near-fully-connected predictions.
        return {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "lambda_sparsity": trial.suggest_float("lambda_sparsity", 1e-4, 1e-1, log=True),
            "threshold": trial.suggest_float("threshold", 0.05, 0.5),
        }
    return {}


def get_default_hp(method_name: str) -> Dict[str, Any]:
    defaults = {
        "VARLiNGAM": {},
        "PCMCI_ParCorr": {"pc_alpha": 0.05},
        "PCMCI_GPDC": {"pc_alpha": 0.05},
        "DYNOTEARS": {"lambda_w": 0.1, "lambda_a": 0.1, "w_threshold": 0.1},
        "CDNOD": {"alpha": 0.05},
        "GrangerF": {"alpha": 0.05},
    }
    return defaults.get(method_name, {})
