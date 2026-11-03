"""Aggregate per-variant varsortability audits into one table.

For every DGP variant this reports the varsortability of the generated data,
each method's F1 on raw and standardized inputs, and the difference. It also
re-runs the scale-invariance sanity checks across the whole set.

    PYTHONPATH=src python scripts/summarize_audit.py
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

# Methods whose results are invariant to per-variable rescaling as wrapped here.
# Their raw and standardized F1 must agree; a disagreement means the harness is
# wrong. VARLiNGAM is exactly invariant because its wrapper z-scores the input
# before fitting. VARLiNGAM_imp is deliberately absent: it fits on the raw scale
# after mean imputation, so on the trend, seasonal and drift variants (large
# exogenous components) its raw and standardized results can differ.
SCALE_INVARIANT = ["PCMCI_ParCorr", "VARLiNGAM", "GrangerF"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("benchmark_results"))
    ap.add_argument("--out", type=Path,
                    default=Path("benchmark_results/audit_summary.csv"))
    args = ap.parse_args()

    files = sorted(glob.glob(str(args.results_dir / "audit_*.csv")))
    # the re-tuned run covers a subset of methods on one variant; keep it separate
    files = [f for f in files if "summary" not in f and "retuned" not in f]
    if not files:
        raise SystemExit(f"no audit_*.csv found in {args.results_dir}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    methods = sorted({c[3:] for c in df.columns if c.startswith("f1_")})

    rows = []
    for variant, sub in df.groupby("variant"):
        for cond, s2 in sub.groupby("condition"):
            rec = {"variant": variant, "condition": cond,
                   "v_contemporaneous": s2["v_contemporaneous"].mean(),
                   "n_cells": len(s2)}
            for m in methods:
                c = f"f1_{m}"
                if c in s2:
                    rec[m] = s2[c].mean()
            rows.append(rec)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out, index=False)

    print("=" * 100)
    print("VARSORTABILITY AUDIT SUMMARY")
    print("=" * 100)
    piv = summary.pivot_table(index="variant", columns="condition",
                              values=["v_contemporaneous"] + methods)
    print(piv.round(3).to_string())

    print("\n" + "=" * 100)
    print("F1 LOST TO STANDARDIZATION (raw minus standardized)")
    print("=" * 100)
    deltas = []
    for variant, sub in summary.groupby("variant"):
        r = sub[sub.condition == "raw"]
        s = sub[sub.condition == "standardized"]
        if r.empty or s.empty:
            continue
        rec = {"variant": variant}
        for m in methods:
            if m in r and m in s:
                rec[m] = float(r[m].iloc[0]) - float(s[m].iloc[0])
        deltas.append(rec)
    d = pd.DataFrame(deltas).set_index("variant")
    print(d.round(3).to_string())

    print("\n" + "=" * 100)
    print("SANITY CHECKS")
    print("=" * 100)
    failed = False
    for m in SCALE_INVARIANT:
        if m not in d.columns:
            continue
        worst = d[m].abs().max()
        ok = worst < 0.01
        failed |= not ok
        print(f"  {m:<16} max |delta| = {worst:.4f}  "
              f"{'OK' if ok else '*** UNEXPECTED, INVESTIGATE ***'}")
    if failed:
        print("\n  A scale-invariant method moved between conditions. Do not")
        print("  report these numbers until that is explained.")
    else:
        print("\n  All scale-invariant methods held steady. Harness looks correct.")

    print("\n" + "=" * 100)
    print("RANKING CHANGE ON THE LINEAR VARIANT")
    print("=" * 100)
    lin = summary[summary.variant == "linear"]
    if not lin.empty:
        for cond in ("raw", "standardized"):
            row = lin[lin.condition == cond]
            if row.empty:
                continue
            scores = {m: float(row[m].iloc[0]) for m in methods
                      if m in row and pd.notna(row[m].iloc[0])}
            order = sorted(scores.items(), key=lambda kv: -kv[1])
            print(f"\n  {cond}:")
            for i, (m, v) in enumerate(order, 1):
                print(f"    {i}. {m:<16} {v:.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
