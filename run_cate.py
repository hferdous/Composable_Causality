import pandas as pd
import numpy as np
import os
import logging
import time
import tracemalloc
from pathlib import Path
from joblib import Parallel, delayed
from dtcd.baselines.registry import get_cate_method
from dtcd.benchmarks.predefined_graphs import PREDEFINED_GRAPHS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/datasets")
OUT_DIR = Path("benchmark_results")
CATE_RESULTS_PATH = OUT_DIR / "cate_results.csv"

def load_cate_data(dataset_id, seed):
    prefix = f"{dataset_id}_s{seed}"
    X_path = DATA_DIR / f"{prefix}_X.npy"
    cate_path = DATA_DIR / f"{prefix}_cate.npy"
    if not X_path.exists(): return None
    X = np.load(X_path)
    true_cate = np.load(cate_path)
    return X, true_cate

def run_cate_seed(method_name, dataset_id, seed):
    parts = dataset_id.split("_")
    gid = f"{parts[0]}_{parts[1]}"
    g_info = PREDEFINED_GRAPHS[gid]
    tr_node = g_info["treatment_node"]
    out_node = g_info["outcome_node"]
    variant = "_".join(parts[2:]) # e.g. cate_linear
    
    X_all, true_cate = load_cate_data(dataset_id, seed)
    if X_all is None: return None
    
    # Pre-treatment covariates (all except outcome, but including treatment)
    # Most CATE methods take X, W, Y.
    # We will use all nodes EXCEPT treatment and outcome as features X_feat.
    # W is the treatment node. Y is the outcome node.
    d = X_all.shape[1]
    feature_indices = [i for i in range(d) if i not in [tr_node, out_node]]
    X_feat = X_all[:, feature_indices]
    W = X_all[:, tr_node].astype(int)
    Y = X_all[:, out_node]
    
    try:
        tracemalloc.start()
        start_time = time.perf_counter()
        
        # FlexTENet, SNet, TARNet require different HPs? 
        # We use defaults for now.
        method = get_cate_method(method_name, seed=seed)
        method.fit(X_feat, W, Y)
        res = method.predict(X_feat)
        
        runtime_s = time.perf_counter() - start_time
        _, peak_kb = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Metrics
        cate_pred = res.cate_pred
        pehe = np.sqrt(np.mean((true_cate - cate_pred)**2))
        ate_true = np.mean(true_cate)
        ate_pred = np.mean(cate_pred)
        ate_abs_error = np.abs(ate_true - ate_pred)
        
        return {
            "method": method_name, "dataset_id": dataset_id, "seed": seed,
            "pehe": pehe, "ate_abs_error": ate_abs_error,
            "runtime_s": runtime_s, "mem_mb": peak_kb / 1024.0,
            "status": "ok", "d": d, "tau": int(parts[1][3:]), "dgp_variant": variant
        }
    except Exception as e:
        if tracemalloc.is_tracing(): tracemalloc.stop()
        return {
            "method": method_name, "dataset_id": dataset_id, "seed": seed,
            "status": "error", "error_msg": str(e),
            "d": d, "tau": int(parts[1][3:]), "dgp_variant": variant
        }

def process_task(m, did, seeds):
    """Top-level worker so it can run in its own process."""
    res_list = []
    for s in seeds:
        r = run_cate_seed(m, did, s)
        if r:
            res_list.append(r)
    return res_list


def main():
    methods = ["TARNet", "SNet", "FlexTENet", "XLearner"]
    graph_ids = list(PREDEFINED_GRAPHS.keys())
    variants = ["cate_linear", "cate_nonlinear"]
    seeds = list(range(42, 52))
    
    tasks = []
    for m in methods:
        for gid in graph_ids:
            for v in variants:
                tasks.append((m, f"{gid}_{v}"))

    all_results = []
    if CATE_RESULTS_PATH.exists():
        df_exist = pd.read_csv(CATE_RESULTS_PATH)
        all_results = df_exist.to_dict('records')

    completed = set([(r['method'], r['dataset_id'], r['seed']) for r in all_results])

    def pending(m, did):
        return [s for s in seeds if (m, did, s) not in completed]

    logger.info(f"Starting CATE Benchmark (10 seeds)...")
    from joblib import Parallel, delayed
    # Separate processes, not threads: JAX, numpy and tracemalloc state is then
    # per task, so a cell gives the same answer however it is scheduled.
    n_jobs = int(os.environ.get("DTCD_JOBS", "4"))
    for i in range(0, len(tasks), n_jobs):
        batch = [(m, did) for m, did in tasks[i:i+n_jobs] if pending(m, did)]
        if not batch:
            continue
        batch_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(process_task)(m, did, pending(m, did)) for m, did in batch
        )
        new_rows = [item for sublist in batch_results for item in sublist]
        if new_rows:
            all_results.extend(new_rows)
            pd.DataFrame(all_results).to_csv(CATE_RESULTS_PATH, index=False)
            logger.info(f"Progress: {len(all_results)} CATE rows saved.")

if __name__ == "__main__":
    main()
