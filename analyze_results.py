"""Summarize the reference benchmark results.

Reads benchmark_results/results.csv (causal discovery) and
benchmark_results/cate_results.csv (CATE estimation), writes per-method,
per-variant means and standard deviations to benchmark_results/reports/, and
prints the discovery summary. The paper tables themselves are built by the
scripts in paper/tables/ (see README.md).

    python analyze_results.py
"""
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("benchmark_results")
REPORT_DIR = RESULTS_DIR / "reports"


def summarize_discovery() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "results.csv")
    status = (df.groupby(["method", "dgp_variant"]).status
                .value_counts().unstack(fill_value=0))
    ok = df[df.status == "ok"]
    metrics = (ok.groupby(["method", "dgp_variant"])
                 [["f1_orient", "shd_norm", "prec", "rec", "auprc", "runtime_s"]]
                 .agg(["mean", "std"]))
    metrics.columns = [f"{m}_{s}" for m, s in metrics.columns]
    summary = metrics.join(status, how="outer")
    summary.to_csv(REPORT_DIR / "discovery_summary.csv")
    return summary


def summarize_cate() -> pd.DataFrame | None:
    path = RESULTS_DIR / "cate_results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    ok = df[df.status == "ok"]
    summary = (ok.groupby(["method", "dgp_variant"])[["pehe", "ate_abs_error", "runtime_s"]]
                 .agg(["mean", "std"]))
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.to_csv(REPORT_DIR / "cate_summary.csv")
    return summary


if __name__ == "__main__":
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    disc = summarize_discovery()
    cate = summarize_cate()
    pd.set_option("display.width", 160)
    print("Oriented F1 by method and DGP variant (mean over successful runs):")
    print(disc["f1_orient_mean"].unstack().round(3).to_string())
    if cate is not None:
        print("\nPEHE by method and CATE variant:")
        print(cate["pehe_mean"].unstack().round(3).to_string())
    print(f"\nWrote {REPORT_DIR / 'discovery_summary.csv'} and {REPORT_DIR / 'cate_summary.csv'}")
