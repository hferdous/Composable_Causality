"""Measure how much the tuning protocol changes reported results.

The original pipeline tuned hyperparameters on seed 42 and then included
seed 42 in the reported mean. The current pipeline tunes on a held-out
trajectory (seed 41) that is never evaluated. This script tunes both ways on
the same cells and reports the difference in mean oriented F1 over the
evaluation seeds, so the effect of the protocol is a measured number.

Runs a spot check on three cells rather than the full grid.

    PYTHONPATH=src python scripts/check_tuning_protocol.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dtcd.baselines.registry import DISCOVERY_REGISTRY, get_discovery_method
from dtcd.baselines.tuning import TuningManager
from dtcd.benchmarks.predefined_graphs import PREDEFINED_GRAPHS
from dtcd.evaluators.metrics import _unroll_adj, f1_oriented
from generate_datasets import generate_ts, get_stable_weights, stable_weight_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", nargs="+", default=["d4_tau2", "d6_tau3", "d8_tau4"])
    ap.add_argument("--variant", default="linear")
    ap.add_argument("--methods", nargs="+",
                    default=["DYNOTEARS", "PCMCI_ParCorr", "VARLiNGAM"])
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=list(range(42, 52)))
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--out", type=Path,
                    default=Path("benchmark_results/tuning_protocol_check.csv"))
    args = ap.parse_args()

    methods = [m for m in args.methods if DISCOVERY_REGISTRY.get(m) is not None]
    if not methods:
        raise SystemExit("none of the requested methods are installed")

    rows = []
    for gid in args.graphs:
        g = PREDEFINED_GRAPHS[gid]
        tau = g["tau_max"]
        W0, WL, _ = get_stable_weights(g, stable_weight_seed(gid))
        W_true = _unroll_adj(W0, WL)

        # two tuning trajectories: the old in-sample seed, and the held-out one
        protocols = {"old_seed42_in_sample": 42, "new_seed41_held_out": 41}
        for label, tseed in protocols.items():
            Xt, *_ = generate_ts(g, gid, args.variant, args.T, tseed)
            tuner = TuningManager(cache_dir=f"tuning_cache_check/{label}",
                                  n_trials=args.trials)
            for m in methods:
                meta = {"track": f"{gid}_{args.variant}", "d": g["d"],
                        "T": args.T, "protocol": label}
                try:
                    hp, _ = tuner.get_best_hp(m, Xt, W0, WL, tau, meta,
                                              val_seed=tseed)
                except Exception as e:  # noqa: BLE001
                    print(f"  [warn] tuning {m} {gid} {label}: {str(e)[:80]}")
                    hp = {}

                # evaluate on 42-51; under the new protocol seed 42 is honest
                for s in args.eval_seeds:
                    X, *_ = generate_ts(g, gid, args.variant, args.T, s)
                    try:
                        res = get_discovery_method(m, max_lag=tau, seed=s, **hp).fit(X)
                        f1 = (f1_oriented(_unroll_adj(res.W_inst, res.W_lag), W_true)
                              if res.status == "ok" else np.nan)
                    except Exception:  # noqa: BLE001
                        f1 = np.nan
                    rows.append(dict(graph=gid, variant=args.variant, method=m,
                                     protocol=label, tuning_seed=tseed,
                                     eval_seed=s, f1=f1,
                                     tuned_in_sample=(s == tseed)))
        print(f"  checked {gid}", flush=True)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print("\n" + "=" * 78)
    print("Mean F1 by method and tuning protocol (evaluated on seeds 42-51)")
    print("=" * 78)
    piv = df.pivot_table(index="method", columns="protocol", values="f1")
    piv["delta"] = piv.get("old_seed42_in_sample", np.nan) - piv.get(
        "new_seed41_held_out", np.nan)
    print(piv.round(4).to_string())

    print("\nOptimism attributable to evaluating on the tuning seed:")
    old = df[df.protocol == "old_seed42_in_sample"]
    for m, sub in old.groupby("method"):
        ins = sub[sub.tuned_in_sample].f1.mean()
        out = sub[~sub.tuned_in_sample].f1.mean()
        print(f"  {m:<16} seed 42 = {ins:.3f}   seeds 43-51 = {out:.3f}   "
              f"gap = {ins - out:+.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
