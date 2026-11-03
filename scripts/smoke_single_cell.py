"""Smoke run of every discovery baseline on one reference cell (d4_tau2, linear).

Tunes each method on the held-out seed and evaluates it on seed 42, printing the
status and wall-clock time per method. Needs the datasets from
generate_datasets.py and the [discovery,deep] extras.

    python scripts/smoke_single_cell.py
"""
import time
import numpy as np
from pathlib import Path
from dtcd.evaluators.runner import BenchmarkRunner
import logging

logging.basicConfig(level=logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]


def load_data(dataset_id, seed):
    prefix = f"{dataset_id}_s{seed}"
    DATA_DIR = ROOT / "data" / "datasets"
    X_path = DATA_DIR / f"{prefix}_X.npy"
    gt_path = DATA_DIR / f"{prefix}_gt.npy"
    if not X_path.exists(): return None, None, None
    X = np.load(X_path)
    gt = np.load(gt_path)
    return X, gt[0], gt[1:]

def main():
    dataset_id = "d4_tau2_linear"
    seed = 42
    methods = ["GrangerF", "VARLiNGAM", "VARLiNGAM_imp", "PCMCI_ParCorr", "DYNOTEARS", "CUTS+", "PCMCI_GPDC", "CDNOD"]
    
    runner = BenchmarkRunner(output_dir=str(ROOT / "benchmark_results_test"), tuning_trials=30)
    
    X0, W0, Wl = load_data(dataset_id, seed)
    if X0 is None:
        raise SystemExit("datasets not found; run python generate_datasets.py first")
    cell_meta = {"track": dataset_id, "d": 4, "tau": 2, "T": 1000}
    
    start_all = time.time()
    times = {}
    
    for m in methods:
        start_m = time.time()
        print(f"Running {m}...")
        try:
            # Tuning
            best_hp, source = runner.tuning_mgr.get_best_hp(m, X0, W0, Wl, 2, cell_meta)
            # Evaluation
            from dtcd.evaluators.runner import run_single_experiment
            res = run_single_experiment(m, X0, W0, Wl, 2, seed, timeout_s=3600, **best_hp)
            status = res.get("status", "unknown")
            times[m] = time.time() - start_m
            print(f"  {m} took {times[m]:.2f}s (Status: {status})")
        except Exception as e:
            times[m] = time.time() - start_m
            print(f"  {m} failed after {times[m]:.2f}s: {e}")
        
    total_time = time.time() - start_all
    print(f"\nTotal single-cell time: {total_time:.2f}s")
    for m, t in sorted(times.items(), key=lambda x: x[1], reverse=True):
        print(f"{m}: {t:.2f}s")

if __name__ == "__main__":
    main()
