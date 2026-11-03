"""Time one PCMCI-GPDC fit on the largest reference cell (d8_tau4, linear, seed 42).

This is the measurement behind restricting PCMCI-GPDC to d = 4 in the reference
benchmark. Needs the datasets from generate_datasets.py and the [discovery]
extra. A single fit can take several minutes.

    python scripts/time_pcmci_gpdc.py
"""
import time
import numpy as np
from pathlib import Path
from dtcd.evaluators.runner import run_single_experiment



def load_data(dataset_id, seed):
    prefix = f"{dataset_id}_s{seed}"
    DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "datasets"
    X_path = DATA_DIR / f"{prefix}_X.npy"
    gt_path = DATA_DIR / f"{prefix}_gt.npy"
    X = np.load(X_path)
    gt = np.load(gt_path)
    return X, gt[0], gt[1:]

dataset_id = "d8_tau4_linear"
seed = 42

X, W0, Wl = load_data(dataset_id, seed)

start = time.time()
res = run_single_experiment(
    method_name="PCMCI_GPDC",
    X=X,
    W_inst_true=W0,
    W_lag_true=Wl,
    max_lag=4,
    seed=seed,
    timeout_s=600
)
runtime = time.time() - start

print(f"Status: {res.get('status')}")
if res.get('error_msg'):
    print(f"Error: {res.get('error_msg')}")
print(f"Runtime (measured): {runtime:.2f}s")
