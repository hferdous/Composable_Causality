"""Regenerate paper/tables/scal_f1_grid.tex and paper/tables/scal_runtime.tex
from the scalability sweep CSVs (scal_tau{3,6,12}_{raw,std}.csv). Run from the
repository root:

    python paper/tables/build_scal_tables.py benchmark_results
"""
import glob
import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).parent

d = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results"
files = {}
for f in glob.glob(f"{d}/*scal_tau*.csv"):
    tau = int(f.split("tau")[1].split("_")[0])
    files.setdefault(tau, {})["std" if "_std" in f else "raw"] = pd.read_csv(f)

rows = [(t, [files[t]["std"].groupby("d")["f1_DYNOTEARS"].mean()[k]
             for k in (4, 8, 16, 32)]) for t in (3, 6, 12)]
t1 = ["\\begin{tabular}{lcccc}", "\\toprule",
      "DYNOTEARS $F_1$ (std.) & $d{=}4$ & $d{=}8$ & $d{=}16$ & $d{=}32$ \\\\",
      "\\midrule"]
t1 += [f"$\\tau_{{\\max}}{{=}}{t}$ & " + " & ".join(f"{v:.2f}" for v in vs) + " \\\\"
       for t, vs in rows]
open(HERE / "scal_f1_grid.tex", "w").write("\n".join(t1 + ["\\bottomrule", "\\end{tabular}"]) + "\n")

t2 = ["\\begin{tabular}{lccc}", "\\toprule",
      "Seconds per fit, $d{=}32$ & $\\tau_{\\max}{=}3$ & $\\tau_{\\max}{=}6$ & $\\tau_{\\max}{=}12$ \\\\",
      "\\midrule"]
for m, lab in [("DYNOTEARS", "DYNOTEARS"), ("VARLiNGAM", "VARLiNGAM"),
               ("PCMCI_ParCorr", "PCMCI-ParCorr"), ("GrangerF", "GrangerF")]:
    vs = [files[t]["raw"].query("d==32")[f"runtime_{m}"].mean() for t in (3, 6, 12)]
    t2.append(f"{lab} & " + " & ".join(f"{v:.0f}" for v in vs) + " \\\\")
open(HERE / "scal_runtime.tex", "w").write("\n".join(t2 + ["\\bottomrule", "\\end{tabular}"]) + "\n")
print("wrote scal_f1_grid.tex, scal_runtime.tex")
