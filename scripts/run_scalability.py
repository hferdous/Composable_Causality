"""Scalability sweep over graph dimension and trajectory length.

Answers: how do accuracy and wall-clock cost move as d and T grow beyond the
reference benchmark's d in {4,6,8}, T = 1000?

Only methods whose implementation is vendored in this repository or depends on
core requirements are run here (DYNOTEARS, SortnRegress, RandomRegress), so the
sweep reproduces without the optional [discovery] extra. Extending it to the
wrapped third-party baselines is a matter of adding names to --methods.

    PYTHONPATH=src python scripts/run_scalability.py --out benchmark_results/scalability.csv
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dtcd.baselines.registry import (  # noqa: E402
    DISCOVERY_REGISTRY,
    get_discovery_method,
)
from dtcd.utils.varsortability import standardize  # noqa: E402


class _FitTimeout(Exception):
    pass


@contextmanager
def time_budget(seconds):
    """Abort a fit that exceeds `seconds`. POSIX only; no-op elsewhere.

    PCMCI-family methods at d=32 with a long lag window can run for hours, so
    an unbounded sweep silently stalls on one cell. A cell that exceeds the
    budget is recorded as status='timeout' rather than dropped, which keeps the
    runtime story honest instead of reporting only the cells that finished.

    Note: several wrappers catch all exceptions inside fit(), so the alarm is
    usually absorbed there and returned as status='error'. The caller relabels
    those cells by elapsed time, so a genuine error and a budget abort remain
    distinguishable in the output.
    """
    if seconds is None or seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise(signum, frame):
        raise _FitTimeout(f"exceeded {seconds:.0f}s budget")

    old = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def random_window_graph(d, tau, n_edges, rng):
    """Sparse DAG: contemporaneous edges respect index order, lagged are free."""
    W0 = np.zeros((d, d))
    WL = np.zeros((tau, d, d))
    n_inst = max(1, n_edges // 3)
    placed = 0
    while placed < n_inst:
        i, j = sorted(rng.choice(d, 2, replace=False))
        if W0[i, j] == 0:
            W0[i, j] = rng.uniform(0.3, 0.8) * rng.choice([-1, 1])
            placed += 1
    placed = 0
    while placed < n_edges - n_inst:
        i, j, k = rng.randint(d), rng.randint(d), rng.randint(1, tau + 1)
        if WL[k - 1, i, j] == 0:
            WL[k - 1, i, j] = rng.uniform(0.3, 0.8) * rng.choice([-1, 1])
            placed += 1
    inv = np.linalg.inv(np.eye(d) - W0)
    comp = np.zeros((d * tau, d * tau))
    for k in range(tau):
        comp[:d, k * d:(k + 1) * d] = inv @ WL[k]
    if tau > 1:
        comp[d:, :(tau - 1) * d] = np.eye((tau - 1) * d)
    rho = np.max(np.abs(np.linalg.eigvals(comp)))
    if rho >= 0.95:
        WL *= (0.9 / rho)
    return W0, WL


def simulate(W0, WL, T, seed):
    d, tau = W0.shape[0], WL.shape[0]
    rng = np.random.RandomState(seed)
    noises = rng.randn(T, d)
    X = np.zeros((T, d))
    for t in range(T):
        v = np.zeros(d)
        for i in range(d):
            pa = np.dot(W0[:, i], v)
            for k in range(1, tau + 1):
                if t - k >= 0:
                    pa += np.dot(WL[k - 1, :, i], X[t - k])
            v[i] = pa + noises[t, i]
        X[t] = v
    return X


def unroll(W0, WL):
    d, tau = W0.shape[0], WL.shape[0]
    J = np.zeros((d * (tau + 1), d))
    J[:d] = W0
    for k in range(tau):
        J[(k + 1) * d:(k + 2) * d] = WL[k]
    return J


def f1(Wp, Wt):
    _, _, f, _ = precision_recall_fscore_support(
        (Wt != 0).flatten().astype(int), (Wp != 0).flatten().astype(int),
        average="binary", zero_division=0)
    return float(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--lengths", type=int, nargs="+", default=[1000, 5000, 10000])
    ap.add_argument("--tau", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--standardized", action="store_true",
                    help="run on z-scored input (varsortability-free)")
    ap.add_argument("--methods", nargs="+",
                    default=["DYNOTEARS", "PCMCI_ParCorr", "VARLiNGAM",
                             "VARLiNGAM_imp", "GrangerF", "CDNOD", "CUTS+",
                             "SortnRegress", "RandomRegress"],
                    help="Methods to sweep. Unavailable ones are skipped with "
                         "a warning rather than aborting the run.")
    ap.add_argument("--hp", default=None,
                    help="Method-keyed JSON of hyperparameters, or a path to a "
                         "file containing it. Example: "
                         "'{\"CUTS+\": {\"lr\": 0.00966, \"threshold\": 0.286}}'. "
                         "Methods not listed run at library defaults. Any "
                         "override is recorded in the output CSV so the "
                         "provenance of a non-default run is never lost.")
    ap.add_argument("--timeout", type=float, default=600.0,
                    help="Per-fit budget in seconds; slower fits are recorded "
                         "as status=timeout rather than silently omitted.")
    ap.add_argument("--out", type=Path, default=Path("benchmark_results/scalability.csv"))
    args = ap.parse_args()

    hp_by_method: dict[str, dict] = {}
    if args.hp:
        raw_hp = args.hp
        if Path(raw_hp).exists():
            raw_hp = Path(raw_hp).read_text()
        hp_by_method = json.loads(raw_hp)
        if not all(isinstance(v, dict) for v in hp_by_method.values()):
            raise SystemExit(
                "--hp must be method-keyed, e.g. '{\"CUTS+\": {\"lr\": 0.01}}'")
        unknown = [m for m in hp_by_method if m not in args.methods]
        if unknown:
            raise SystemExit(f"--hp names methods not in --methods: {unknown}")
        for m, hp in hp_by_method.items():
            print(f"[hp] {m} overridden: {hp}")

    methods = [m for m in args.methods if DISCOVERY_REGISTRY.get(m) is not None]
    skipped = [m for m in args.methods if m not in methods]
    if skipped:
        print(f"[warn] unavailable, skipping: {skipped}")
    if not methods:
        raise SystemExit("no runnable methods requested")
    print(f"sweeping {len(methods)} methods: {methods}")

    rows = []
    for d in args.dims:
        for T in args.lengths:
            for seed in args.seeds:
                rng = np.random.RandomState(1000 + d)
                W0, WL = random_window_graph(d, args.tau, n_edges=d, rng=rng)
                Wt = unroll(W0, WL)
                X = simulate(W0, WL, T, seed)
                if args.standardized:
                    X = standardize(X)

                rec = dict(d=d, T=T, tau=args.tau, seed=seed,
                           standardized=args.standardized)
                for m in methods:
                    t0 = time.perf_counter()
                    try:
                        with time_budget(args.timeout):
                            res = get_discovery_method(
                                m, max_lag=args.tau, seed=seed,
                                **hp_by_method.get(m, {})).fit(X)
                        elapsed = time.perf_counter() - t0
                        rec[f"f1_{m}"] = (
                            f1(unroll(res.W_inst, res.W_lag), Wt)
                            if res.status == "ok" else np.nan)
                        rec[f"runtime_{m}"] = elapsed
                        # Most wrappers catch every exception internally, so an
                        # alarm raised inside fit() surfaces as status='error'
                        # rather than propagating. A fit that errored after
                        # consuming essentially the whole budget was aborted by
                        # the alarm, not by a genuine failure; relabel it so the
                        # runtime story stays readable.
                        st = res.status
                        if (st != "ok" and args.timeout
                                and elapsed >= 0.95 * args.timeout):
                            st = "timeout"
                        rec[f"status_{m}"] = st
                    except _FitTimeout as e:
                        rec[f"f1_{m}"] = np.nan
                        rec[f"runtime_{m}"] = time.perf_counter() - t0
                        rec[f"status_{m}"] = f"timeout: {e}"
                        print(f"    [timeout] {m} d={d} T={T} tau={args.tau}",
                              flush=True)
                    except Exception as e:  # noqa: BLE001
                        rec[f"f1_{m}"] = np.nan
                        rec[f"runtime_{m}"] = time.perf_counter() - t0
                        rec[f"status_{m}"] = f"error: {str(e)[:80]}"
                    rec[f"hp_{m}"] = (json.dumps(hp_by_method[m])
                                      if m in hp_by_method else "default")
                rows.append(rec)
            print(f"  d={d} T={T} done", flush=True)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    cols = [f"f1_{m}" for m in methods] + [f"runtime_{m}" for m in methods]
    print("\n" + df.groupby(["d", "T"])[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
