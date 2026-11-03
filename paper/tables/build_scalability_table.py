"""Build paper/tables/scalability.tex (the tau_max = 3 sweep up to T = 10,000)
from benchmark_results/scalability.csv. Run from the repository root:

    python paper/tables/build_scalability_table.py
"""
import pathlib
import sys

import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/scalability.csv"
df = pd.read_csv(src)
g = df.groupby(["d", "T", "standardized"]).mean(numeric_only=True)

rows = [
    "\\begin{tabular}{ccccccc}",
    "\\toprule",
    "& & \\multicolumn{2}{c}{DYNOTEARS F1} & \\multicolumn{2}{c}{\\textsc{SortnRegress} F1} & DYNOTEARS \\\\",
    "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
    "$d$ & $T$ & raw & std. & raw & std. & runtime (s) \\\\",
    "\\midrule",
]
for d in sorted(df.d.unique()):
    for T in sorted(df["T"].unique()):
        r, s = g.loc[(d, T, False)], g.loc[(d, T, True)]
        rows.append(f"{d} & {T} & {r.f1_DYNOTEARS:.2f} & {s.f1_DYNOTEARS:.2f} & "
                    f"{r.f1_SortnRegress:.2f} & {s.f1_SortnRegress:.2f} & "
                    f"{r.runtime_DYNOTEARS:.1f} \\\\")
rows += ["\\bottomrule", "\\end{tabular}"]
out = pathlib.Path(__file__).parent / "scalability.tex"
out.write_text("\n".join(rows) + "\n")
print("wrote", out)
