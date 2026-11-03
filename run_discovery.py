import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from joblib import Parallel, delayed
from dtcd.evaluators.runner import BenchmarkRunner, run_single_experiment
from dtcd.benchmarks.predefined_graphs import PREDEFINED_GRAPHS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/datasets")
OUT_DIR = Path("benchmark_results")
RESULTS_PATH = OUT_DIR / "results.csv"

# Hyperparameter tuning uses a dedicated trajectory that is NOT part of the
# reported evaluation set. Previously tuning ran on seeds[0] (= 42), which was
# then also included in the reported mean, so one tenth of the evaluation data
# had been seen by the oracle-F1 tuning objective.
TUNING_SEED = 41
EVAL_SEEDS = list(range(42, 52))

def load_data(dataset_id, seed):
    prefix = f"{dataset_id}_s{seed}"
    X_path = DATA_DIR / f"{prefix}_X.npy"
    gt_path = DATA_DIR / f"{prefix}_gt.npy"
    if not X_path.exists(): return None
    X = np.load(X_path)
    gt = np.load(gt_path)
    return X, gt[0], gt[1:]

def run_cell_method(method_name, dataset_id, seeds, runner=None):
    if runner is None:
        runner = BenchmarkRunner(output_dir=str(OUT_DIR), tuning_trials=30)
    # 1. Tuning on a held-out trajectory (TUNING_SEED), never evaluated.
    tuning_data = load_data(dataset_id, TUNING_SEED)
    if tuning_data is None:
        raise FileNotFoundError(
            f"Tuning trajectory for {dataset_id} at seed {TUNING_SEED} is missing. "
            f"Run generate_datasets.py (which now emits seed {TUNING_SEED} "
            f"alongside the evaluation seeds) before running discovery."
        )
    X0, W0, Wl = tuning_data
    # Extract d and tau from dataset_id (e.g. d4_tau2_linear)
    parts = dataset_id.split("_")
    d = int(parts[0][1:])
    tau = int(parts[1][3:])
    variant = "_".join(parts[2:]) # e.g. linear or cate_linear
    
    if method_name == "PCMCI_GPDC" and d > 4:
        return [{
            "method": method_name, "seed": s, "status": "skipped_compute_budget",
            "error_msg": "GPDC excluded for d>4 due to compute scaling",
            "dataset_id": dataset_id, "d": d, "tau": tau, "dgp_variant": variant,
            "runtime_s": np.nan, "mem_mb": np.nan, "shd_joint": np.nan, "shd_norm": np.nan,
            "n_true_edges": np.nan, "n_pred_edges": np.nan, "f1_skel": np.nan, "f1_orient": np.nan,
            "prec": np.nan, "rec": np.nan, "auprc": np.nan, "hp": {}, "hp_source": "skipped"
        } for s in seeds]
    
    cell_meta = {"track": f"{dataset_id}", "d": d, "T": 1000, "tune_seed": TUNING_SEED}
    best_hp, source = runner.tuning_mgr.get_best_hp(method_name, X0, W0, Wl, tau, cell_meta)
    
    results = []
    for s in seeds:
        # Check completion
        X, W0, Wl = load_data(dataset_id, s)
        try:
            res = run_single_experiment(method_name, X, W0, Wl, tau, s, timeout_s=3600, **best_hp)
            res.update({
                "dataset_id": dataset_id,
                "d": d, "tau": tau, "dgp_variant": variant,
                "hp_source": source
            })
            results.append(res)
        except Exception as e:
            results.append({
                "method": method_name, "seed": s, "status": "error", "error_msg": str(e),
                "dataset_id": dataset_id, "d": d, "tau": tau, "dgp_variant": variant
            })
    return results

def process_task(method_name, dataset_id, pending_seeds):
    """Top-level worker so it can run in a separate process. Each process has its
    own global RNG state, JAX/torch state and tracemalloc, which keeps results
    independent of how tasks are scheduled."""
    return run_cell_method(method_name, dataset_id, pending_seeds)


def write_environment_record(path):
    """Record the interpreter and every installed package version used for this run."""
    import json, platform, sys
    from importlib import metadata
    pkgs = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()
                  if d.metadata and d.metadata["Name"])
    path.write_text(json.dumps({"python": sys.version, "platform": platform.platform(),
                                "machine": platform.machine(), "packages": pkgs}, indent=1))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_environment_record(OUT_DIR / "environment.json")
    methods = ["CDNOD", "VARLiNGAM_imp", "GrangerF", "VARLiNGAM", "PCMCI_ParCorr", "DYNOTEARS", "CUTS+", "PCMCI_GPDC"]
    variants = ["linear", "nonlinear", "drift", "mnar", "mcar", "trend", "seasonal", "trend_seasonal", "nonlinear_trend_seasonal"]
    graph_ids = list(PREDEFINED_GRAPHS.keys())
    seeds = list(EVAL_SEEDS)
    
    # Generate all (method, dataset_id) pairs
    tasks = []
    for m in methods:
        for gid in graph_ids:
            for v in variants:
                dataset_id = f"{gid}_{v}"
                tasks.append((m, dataset_id))

    # Load existing results
    all_results = []
    completed_pairs = set()
    if RESULTS_PATH.exists():
        df_exist = pd.read_csv(RESULTS_PATH)
        all_results = df_exist.to_dict('records')
        for r in all_results:
            completed_pairs.add((r['method'], r['dataset_id'], r['seed']))
    
    # We will run (method, dataset) blocks in parallel
    # But check if some seeds in the block are already done
    
    def pending(m, dataset_id):
        return [s for s in seeds if (m, dataset_id, s) not in completed_pairs]

    # Separate processes, not threads. Wrappers seed global RNGs (numpy, torch),
    # which threads would share and race on.
    n_jobs = int(os.environ.get("DTCD_JOBS", "4"))
    for i in range(0, len(tasks), n_jobs):
        batch = [(m, did) for m, did in tasks[i:i+n_jobs] if pending(m, did)]
        if batch:
            results_batch = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(process_task)(m, did, pending(m, did)) for m, did in batch)
            
            # Flatten and append
            new_rows = [item for sublist in results_batch for item in sublist]
            if new_rows:
                # Append to current list, deduplicate, and save
                all_results.extend(new_rows)
                df_to_save = pd.DataFrame(all_results).drop_duplicates(subset=['method', 'dataset_id', 'seed'])
                df_to_save.to_csv(RESULTS_PATH, index=False)
                logger.info(f"Progress: {len(df_to_save)} unique rows saved.")

    print("\n✅ Discovery Benchmark Step Complete!")

if __name__ == "__main__":
    main()
