"""Build paper/tables/nonlin_mechanisms.tex (paper Table 17) from
benchmark_results/nonlinear_mechanisms.csv. Run from the repository root:

    python paper/tables/build_nonlin_table.py [benchmark_results/nonlinear_mechanisms.csv]
"""
import pathlib
import sys

import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/nonlinear_mechanisms.csv"
df = pd.read_csv(src)
df = df[df.status == "ok"]

ROWS = [
    ("linear", r"Linear ($x$)", "identity"),
    ("nonlinear", r"Nonlinear ($\text{sign}(x)|x|^{0.5}$)", "monotonic, saturating"),
    ("nonlinear_sin", r"Non-monotonic ($\sin x$)", "non-monotonic, bounded"),
]

lines = [r"\begin{tabular}{lccc}", r"\toprule",
         r"Mechanism & Property & SHD ($\downarrow$) & F1 ($\uparrow$) \\", r"\midrule"]
for key, label, prop in ROWS:
    x = df[df.mechanism == key]
    lines.append(f"{label} & {prop} & ${x.shd.mean():.2f} \\pm {x.shd.std():.2f}$ & "
                 f"${x.f1_oriented.mean():.2f} \\pm {x.f1_oriented.std():.2f}$ \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]

out = pathlib.Path(__file__).parent / "nonlin_mechanisms.tex"
out.write_text("\n".join(lines) + "\n")
print("wrote", out)
print("cells per mechanism:", df.groupby("mechanism").size().to_dict())
