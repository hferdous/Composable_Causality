from __future__ import annotations
import time
import logging
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Any, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import multiprocessing

from dtcd.baselines.registry import get_discovery_method, get_cate_method
from dtcd.baselines.tuning import TuningManager
from dtcd.evaluators.metrics import (
    shd, normalized_shd, get_n_edges,
    f1_skeleton, f1_oriented, precision_recall,
    auprc, _unroll_adj,
)

logger = logging.getLogger(__name__)


def run_single_experiment(
    method_name: str,
    X: np.ndarray,
    W_inst_true: np.ndarray,
    W_lag_true: np.ndarray,
    max_lag: int,
    seed: int,
    timeout_s: int = 86400,
    **hp: Any,
) -> Dict[str, Any]:
    """
    Runs a single discovery experiment.

    Methods declaring supports_missing=False are skipped with a structured
    error rather than receiving NaN inputs (which previously caused
    DYNOTEARS to silently return an empty graph with status="ok").
    """
    method = get_discovery_method(method_name, max_lag=max_lag, seed=seed, **hp)

    has_nan = bool(np.isnan(X).any())
    if has_nan and not method.supports_missing:
        logger.warning(
            f"{method_name} declares supports_missing=False but received "
            f"NaN input. Skipping with structured error."
        )
        return {
            "method": method_name,
            "seed": seed,
            "status": "skipped_no_missing_support",
            "runtime_s": 0.0,
            "mem_mb": 0.0,
            "shd_joint": np.nan,
            "shd_norm": np.nan,
            "n_true_edges": int(np.sum(_unroll_adj(W_inst_true, W_lag_true) != 0)),
            "n_pred_edges": 0,
            "f1_skel": np.nan,
            "f1_orient": np.nan,
            "prec": np.nan,
            "rec": np.nan,
            "auprc": np.nan,
            "hp": json.dumps(hp),
            "error_msg": (
                f"Method does not support missing values "
                f"(supports_missing={method.supports_missing}). "
                f"X contains {int(np.isnan(X).sum())} NaN entries."
            ),
        }

    try:
        res = method.fit(X)
    except Exception as e:
        logger.error(f"Error in {method_name}: {e}")
        return {"status": "error", "error_msg": str(e), "method": method_name, "seed": seed}

    W_pred_joint = _unroll_adj(res.W_inst, res.W_lag)
    W_true_joint = _unroll_adj(W_inst_true, W_lag_true)

    n_true = get_n_edges(W_true_joint)
    n_pred = get_n_edges(W_pred_joint)
    s_joint = shd(W_pred_joint, W_true_joint)
    s_norm = normalized_shd(W_pred_joint, W_true_joint)
    f1_s = f1_skeleton(W_pred_joint, W_true_joint)
    f1_o = f1_oriented(W_pred_joint, W_true_joint)
    prec, rec = precision_recall(W_pred_joint, W_true_joint)

    W_score_joint = _unroll_adj(res.W_inst_score, res.W_lag_score)
    aup = auprc(W_score_joint, W_true_joint)

    return {
        "method": method_name,
        "seed": seed,
        "status": res.status,
        "runtime_s": res.runtime_s,
        "mem_mb": res.peak_memory_mb,
        "shd_joint": s_joint,
        "shd_norm": s_norm,
        "n_true_edges": n_true,
        "n_pred_edges": n_pred,
        "f1_skel": f1_s,
        "f1_orient": f1_o,
        "prec": prec,
        "rec": rec,
        "auprc": aup,
        "hp": json.dumps(hp),
        "error_msg": res.error_msg,
    }


class BenchmarkRunner:
    def __init__(self, output_dir: str, tuning_trials: int = 50):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.output_dir / "results.parquet"
        self.tuning_mgr = TuningManager(n_trials=tuning_trials)

    def run(
        self,
        methods: List[str],
        cells: List[Dict[str, Any]],
        seeds: List[int],
        val_seed: int = 0,
        timeout_s: int = 86400,
    ):
        all_results = []
        if self.results_path.exists():
            try:
                all_results = pd.read_parquet(self.results_path).to_dict("records")
                logger.info(f"Resuming from {len(all_results)} existing results.")
            except Exception:
                pass

        for cell in cells:
            X = cell["data"]
            W_inst_true = cell["W_inst_true"]
            W_lag_true = cell["W_lag_true"]
            max_lag = cell["max_lag"]

            X_val = X

            for method_name in methods:
                logger.info(f"Tuning {method_name}...")
                best_hp, hp_source = self.tuning_mgr.get_best_hp(
                    method_name, X_val, W_inst_true, W_lag_true, max_lag, cell, val_seed
                )

                for seed in seeds:
                    if any(
                        r["method"] == method_name
                        and r["seed"] == seed
                        and r.get("track") == cell.get("track")
                        for r in all_results
                    ):
                        continue

                    logger.info(f"Evaluating {method_name} (Seed {seed})")
                    res = run_single_experiment(
                        method_name, X, W_inst_true, W_lag_true, max_lag, seed, timeout_s, **best_hp
                    )

                    actual_hp = json.loads(res["hp"])
                    for k, v in best_hp.items():
                        assert actual_hp[k] == v, f"HP injection broken for {k}: {actual_hp[k]} != {v}"

                    res.update({k: v for k, v in cell.items() if k not in ["data", "W_inst_true", "W_lag_true"]})
                    res["hp_source"] = hp_source
                    all_results.append(res)

                    if len(all_results) % 10 == 0:
                        pd.DataFrame(all_results).to_parquet(self.results_path)

        df = pd.DataFrame(all_results)
        df.to_parquet(self.results_path)
        return df
