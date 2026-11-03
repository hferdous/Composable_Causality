"""DYNOTEARS under three recursive structural mechanisms (paper Appendix G).

Graph, weights, noise realization and hyperparameters are held fixed; only
the mechanism applied at every node and time step changes:

  linear          identity
  nonlinear       sign(x)|x|^0.5, strictly monotone and saturating
                  (the reference benchmark's Nonlinear variant)
  nonlinear_sin   sin(x), non-monotone and bounded

DYNOTEARS hyperparameters are fixed at the values tuned for each graph's
Nonlinear cell, read from benchmark_results/results.csv. The design is the
reference grid: nine graphs x ten evaluation seeds (42 to 51), so 90 fits per
mechanism.

The script also records the peak amplitude of the amplifying mechanism x|x|
("nonlinear_quad"). That mechanism is left out of the table because its
recursive composition makes trajectories explode, which turns structure
recovery into a numerical-conditioning test.

    python scripts/run_nonlinear_mechanisms.py

Writes benchmark_results/nonlinear_mechanisms.csv and
benchmark_results/nonlinear_quad_amplitude.csv. Takes a few minutes on two
CPU cores. Build the paper table with
python paper/tables/build_nonlin_table.py.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dtcd.benchmarks.predefined_graphs import PREDEFINED_GRAPHS  # noqa: E402
from dtcd.evaluators.runner import run_single_experiment  # noqa: E402
from generate_datasets import generate_ts  # noqa: E402

MECHANISMS = ("linear", "nonlinear", "nonlinear_sin")
SEEDS = tuple(range(42, 52))
T = 1000


def nonlinear_cell_hp(results: pd.DataFrame, gid: str) -> dict:
    rows = results[(results.method == "DYNOTEARS")
                   & (results.dataset_id == f"{gid}_nonlinear")]
    if rows.empty:
        raise KeyError(f"no DYNOTEARS row for {gid}_nonlinear in results.csv")
    return json.loads(rows.hp.iloc[0])


def fit_one(gid: str, mechanism: str, seed: int, hp: dict) -> dict:
    warnings.filterwarnings("ignore")
    g = PREDEFINED_GRAPHS[gid]
    X, _mask, W0, W_lags, *_ = generate_ts(g, gid, mechanism, T, seed)
    out = run_single_experiment("DYNOTEARS", X, W0, W_lags, g["tau_max"], seed, **hp)
    return {"graph": gid, "d": g["d"], "tau_max": g["tau_max"],
            "mechanism": mechanism, "seed": seed, "status": out["status"],
            "shd": out["shd_joint"], "f1_oriented": out["f1_orient"],
            "hp": json.dumps(hp)}


def quad_amplitude(gid: str, seed: int) -> dict:
    warnings.filterwarnings("ignore")
    g = PREDEFINED_GRAPHS[gid]
    X, *_ = generate_ts(g, gid, "nonlinear_quad", T, seed)
    return {"graph": gid, "d": g["d"], "tau_max": g["tau_max"], "seed": seed,
            "max_abs_x": float(np.nanmax(np.abs(X)))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path,
                    default=ROOT / "benchmark_results" / "results.csv",
                    help="main benchmark results holding the tuned hyperparameters")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "benchmark_results")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    results = pd.read_csv(args.results)
    hps = {gid: nonlinear_cell_hp(results, gid) for gid in PREDEFINED_GRAPHS}

    jobs = [(gid, m, s, hps[gid]) for gid in PREDEFINED_GRAPHS
            for m in MECHANISMS for s in SEEDS]
    rows = Parallel(n_jobs=args.jobs)(delayed(fit_one)(*j) for j in jobs)
    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "nonlinear_mechanisms.csv", index=False)

    amp = pd.DataFrame(Parallel(n_jobs=args.jobs)(
        delayed(quad_amplitude)(gid, s) for gid in PREDEFINED_GRAPHS for s in SEEDS))
    amp.to_csv(args.out_dir / "nonlinear_quad_amplitude.csv", index=False)

    summary = df.groupby("mechanism")[["shd", "f1_oriented"]].agg(["mean", "std", "count"])
    print(summary.round(3).to_string())
    peak = amp.groupby("graph").max_abs_x.max()
    print("\nx|x| peak |x| per graph (max over seeds):")
    print(peak.map(lambda v: f"{v:.3g}").to_string())


if __name__ == "__main__":
    main()
