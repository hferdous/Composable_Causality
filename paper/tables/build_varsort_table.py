"""Build paper/tables/varsort_audit.tex (paper Table 5) from the linear-variant
varsortability audit, benchmark_results/audit_linear.csv. Run from the
repository root:

    python paper/tables/build_varsort_table.py [benchmark_results/audit_linear.csv]
"""
import pathlib
import sys

import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/audit_linear.csv"
df = pd.read_csv(src)
cols = ["v_contemporaneous", "f1_DYNOTEARS", "f1_SortnRegress", "f1_RandomRegress"]
raw = df[df.condition == "raw"][cols].mean()
std = df[df.condition == "standardized"][cols].mean()


def delta(x: float) -> str:
    return "$0.00$" if abs(x) < 0.005 else f"${x:.2f}$"


lines = [
    r"\begin{tabular}{lcccc}",
    r"\toprule",
    r"Condition & $v$ & DYNOTEARS & \textsc{SortnRegress} & \textsc{RandomRegress} \\",
    r"\midrule",
    (f"Raw (as shipped) & {raw.iloc[0]:.2f} & \\textbf{{{raw.iloc[1]:.2f}}} & "
     f"{raw.iloc[2]:.2f} & {raw.iloc[3]:.2f} \\\\"),
    (f"Standardized     & {std.iloc[0]:.2f} & {std.iloc[1]:.2f} & "
     f"{std.iloc[2]:.2f} & {std.iloc[3]:.2f} \\\\"),
    r"\midrule",
    r"$\Delta$         & " + " & ".join(delta(std.iloc[i] - raw.iloc[i]) for i in range(4)) + r" \\",
    r"\bottomrule",
    r"\end{tabular}",
]
out = pathlib.Path(__file__).parent / "varsort_audit.tex"
out.write_text("\n".join(lines) + "\n")
print("wrote", out)
print("cells per condition:", df.groupby("condition").size().to_dict())
