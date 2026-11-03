"""Annotate scalability results with the varsortability of each cell's DGP.

Why this exists
---------------
The scalability sweep and the varsortability audit use different generators.
The audit runs the nine hand-built reference graphs; the sweep draws random
Erdos-Renyi window graphs with ~d edges, only a third of which are
contemporaneous. Those two populations do NOT have the same varsortability, so
the F1 cost of standardizing differs between them, and a reader comparing the
two tables without this column will see an apparent contradiction.

The sweep's DGP is fully determined by (d, tau, seed): random_window_graph
seeds itself with RandomState(1000 + d), and simulate() takes the cell seed. So
varsortability can be recovered exactly after the fact, with no re-run.

    PYTHONPATH=src python scripts/annotate_scalability_varsortability.py
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dtcd.utils.varsortability import (  # noqa: E402
    contemporaneous_varsortability,
    standardize,
    summary_varsortability,
)
from run_scalability import random_window_graph, simulate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("benchmark_results"))
    ap.add_argument("--pattern", default="scal_tau*.csv")
    ap.add_argument("--out", type=Path,
                    default=Path("benchmark_results/scalability_annotated.csv"))
    args = ap.parse_args()

    files = sorted(glob.glob(str(args.results_dir / args.pattern)))
    if not files:
        raise SystemExit(f"no files matching {args.pattern} in {args.results_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    cache: dict[tuple, tuple[float, float, int, int]] = {}
    vc, vs, ni, nl = [], [], [], []
    for r in df.itertuples():
        # T MUST be in the key: varsortability is estimated from the sampled
        # trajectory, so a T=1000 cell and a T=5000 cell with the same graph
        # and seed give different estimates. Omitting T silently reused the
        # first-seen T's value for every other length.
        key = (int(r.d), int(r.tau), int(r.T), int(r.seed), bool(r.standardized))
        if key not in cache:
            d, tau, T, seed, std = key
            rng = np.random.RandomState(1000 + d)
            W0, WL = random_window_graph(d, tau, n_edges=d, rng=rng)
            X = simulate(W0, WL, T, seed)
            if std:
                X = standardize(X)
            cache[key] = (
                contemporaneous_varsortability(X, W0),
                summary_varsortability(X, W0, WL),
                int((np.abs(W0) > 0).sum()),
                int((np.abs(WL) > 0).sum()),
            )
        a, b, c, e = cache[key]
        vc.append(a); vs.append(b); ni.append(c); nl.append(e)

    df["v_contemporaneous"] = vc
    df["v_summary"] = vs
    df["n_contemp_edges"] = ni
    df["n_lagged_edges"] = nl
    df["contemp_fraction"] = df["n_contemp_edges"] / (
        df["n_contemp_edges"] + df["n_lagged_edges"])
    df.to_csv(args.out, index=False)

    print("Varsortability of the scalability DGP, raw condition")
    print("=" * 78)
    raw = df[~df["standardized"]]
    piv = raw.pivot_table(index=["tau", "d"],
                          values=["v_contemporaneous", "v_summary",
                                  "n_contemp_edges", "contemp_fraction"])
    print(piv.round(3).to_string())
    print("=" * 78)
    print("\nReference benchmark (the audit's generators): v_contemporaneous ~0.95")
    print("A cell whose v is already near 0.5 has no shortcut left to remove, so")
    print("standardizing it should cost little. Read the F1 deltas against this")
    print("column, not against the audit's.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
