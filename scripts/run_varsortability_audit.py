"""Varsortability audit for the DTCD reference benchmark.

Reproduces Table (varsortability) and the raw-vs-standardized contrast reported
in the paper. Answers the question: how much of the reported discovery
performance on the linear DGP is carried by marginal variance alone?

Design
------
The DGP, the graph, the seeds and the hyperparameters are held fixed. The only
thing that changes is whether the input is z-scored. Because standardization
drives varsortability to exactly 0.5 while leaving the ground-truth graph
untouched, any F1 difference is attributable to the variance shortcut.

Two controls bracket the result:
  SortnRegress   orders contemporaneous variables by increasing marginal
                 variance, then adaptive-lasso regression. Upper bound on what
                 the shortcut alone buys.
  RandomRegress  identical but with a random contemporaneous order. Lower
                 bound: what the same estimator gets with no variance signal.

Usage
-----
    PYTHONPATH=src python scripts/run_varsortability_audit.py \
        --out benchmark_results/varsortability_audit.csv

Requires the optional [discovery] extra only if --methods includes wrapped
third-party baselines. The default method set runs on core dependencies.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dtcd.benchmarks.predefined_graphs import PREDEFINED_GRAPHS
from dtcd.evaluators.metrics import _unroll_adj, f1_oriented
from dtcd.utils.varsortability import (
    contemporaneous_varsortability,
    standardize,
    summary_varsortability,
)
from generate_datasets import generate_ts, get_stable_weights, stable_weight_seed


def load_tuned_hp(results_csv: Path, method: str, variant: str) -> dict:
    """Reuse the hyperparameters from the reported benchmark run, if available."""
    if not results_csv.exists():
        return {}
    df = pd.read_csv(results_csv)
    df = df[(df.method == method) & (df.dgp_variant == variant) & (df.status == "ok")]
    return {
        r.dataset_id.replace(f"_{variant}", ""): json.loads(r.hp)
        for r in df.itertuples()
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="linear")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--methods", nargs="+",
                    default=["DYNOTEARS", "SortnRegress", "RandomRegress"])
    ap.add_argument("--results-csv", type=Path,
                    default=Path("benchmark_results/results.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("benchmark_results/varsortability_audit.csv"))
    ap.add_argument("--retune", action="store_true",
                    help="Re-run Optuna separately for each condition on the "
                         "held-out tuning trajectory, instead of reusing the "
                         "hyperparameters from the published run. Use this to "
                         "rule out 'the drop is just stale hyperparameters'.")
    ap.add_argument("--tuning-seed", type=int, default=41,
                    help="Held-out trajectory used for --retune. Must not be "
                         "in --seeds.")
    ap.add_argument("--tuning-trials", type=int, default=30)
    args = ap.parse_args()

    if args.retune and args.tuning_seed in args.seeds:
        raise SystemExit(
            f"--tuning-seed {args.tuning_seed} is also an evaluation seed. "
            f"Tuning must be held out.")

    from dtcd.baselines.registry import DISCOVERY_REGISTRY, get_discovery_method

    missing = [m for m in args.methods if DISCOVERY_REGISTRY.get(m) is None]
    if missing:
        print(f"[warn] unavailable on this install, skipping: {missing}")
        print("       install with: pip install -e '.[discovery,deep,cate]'")
        args.methods = [m for m in args.methods if m not in missing]
    if not args.methods:
        raise SystemExit("no runnable methods requested")

    tuner = None
    if args.retune:
        from dtcd.baselines.tuning import TuningManager
        tuner = TuningManager(cache_dir="tuning_cache_audit",
                              n_trials=args.tuning_trials)
        print(f"[retune] tuning per condition on held-out seed "
              f"{args.tuning_seed}, {args.tuning_trials} trials")

    hp_by_method = ({} if args.retune else
                    {m: load_tuned_hp(args.results_csv, m, args.variant)
                     for m in args.methods})

    rows = []
    for gid, g in PREDEFINED_GRAPHS.items():
        tau = g["tau_max"]
        W0_true, W_lags_true, _ = get_stable_weights(g, stable_weight_seed(gid))
        W_joint_true = _unroll_adj(W0_true, W_lags_true)

        for seed in args.seeds:
            X_raw, mask, _, _, _, _, _ = generate_ts(
                g, gid, args.variant, args.T, seed)
            conditions = {"raw": X_raw, "standardized": standardize(X_raw)}

            for cond, X in conditions.items():
                if tuner is not None and seed == args.seeds[0]:
                    # tune once per (graph, condition) on the held-out seed
                    Xt_raw, *_ = generate_ts(
                        g, gid, args.variant, args.T, args.tuning_seed)
                    Xt = Xt_raw if cond == "raw" else standardize(Xt_raw)
                    for m in args.methods:
                        cell_meta = {"track": f"{gid}_{args.variant}",
                                     "d": g["d"], "T": args.T,
                                     "condition": cond}
                        try:
                            best, _src = tuner.get_best_hp(
                                m, Xt, W0_true, W_lags_true, tau, cell_meta,
                                val_seed=args.tuning_seed)
                        except Exception as e:  # noqa: BLE001
                            print(f"    [warn] tuning {m} {gid} {cond}: "
                                  f"{str(e)[:80]}")
                            best = {}
                        hp_by_method.setdefault(m, {})[f"{gid}|{cond}"] = best
                rec = {
                    "graph": gid, "d": g["d"], "tau_max": tau,
                    "variant": args.variant, "seed": seed, "condition": cond,
                    "v_contemporaneous": contemporaneous_varsortability(X, W0_true),
                    "v_summary": summary_varsortability(X, W0_true, W_lags_true),
                }
                for m in args.methods:
                    key = f"{gid}|{cond}" if args.retune else gid
                    hp = hp_by_method.get(m, {}).get(key, {})
                    try:
                        est = get_discovery_method(m, max_lag=tau, seed=seed, **hp)
                        res = est.fit(X)
                        rec[f"f1_{m}"] = (
                            f1_oriented(_unroll_adj(res.W_inst, res.W_lag), W_joint_true)
                            if res.status == "ok" else np.nan
                        )
                    except Exception as e:  # noqa: BLE001
                        rec[f"f1_{m}"] = np.nan
                        rec[f"err_{m}"] = str(e)[:120]
                rows.append(rec)
        print(f"  audited {gid}", flush=True)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    cols = ["v_contemporaneous", "v_summary"] + [f"f1_{m}" for m in args.methods]
    print("\n" + "=" * 78)
    summary = df.groupby("condition")[cols].mean().round(3)
    print(summary.to_string())
    print("=" * 78)

    # Correctness checks. Scale-invariant methods must not move between
    # conditions; if they do, the harness is wrong and the rest is untrustworthy.
    INVARIANT = ["PCMCI_ParCorr", "VARLiNGAM", "VARLiNGAM_imp", "GrangerF"]
    if not args.retune and "raw" in summary.index and "standardized" in summary.index:
        print("\nsanity checks (scale-invariant methods should not move):")
        for m in INVARIANT:
            c = f"f1_{m}"
            if c in summary.columns:
                delta = abs(summary.loc["raw", c] - summary.loc["standardized", c])
                flag = "OK" if delta < 0.01 else "*** UNEXPECTED ***"
                print(f"  {m:<16} delta = {delta:.4f}  {flag}")
        print("  (PCMCI_ParCorr uses partial correlation; VARLiNGAM wrappers")
        print("   already call StandardScaler internally)")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
