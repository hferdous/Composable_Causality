"""Build tables/tg_c1_results.tex from the TimeGraph C1 sweep CSVs
(tg_full.csv, tg_seasonal.csv, tg_mech.csv produced by run_timegraph_c1.py).
Usage: python tables/build_tg_c1_table.py <dir-with-csvs>
"""
import sys
import pathlib

import pandas as pd

d = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results"
methods = [
    ("DYNOTEARS", "DYNOTEARS$^{\\dagger}$"),
    ("SortnRegress", "\\textsc{SortnRegress}"),
    ("RandomRegress", "\\textsc{RandomRegress}"),
    ("VARLiNGAM", "VARLiNGAM"),
    ("GrangerF", "GrangerF"),
    ("PCMCI_ParCorr", "PCMCI-ParCorr"),
]
variants = ["tg_full", "tg_seasonal", "tg_mech"]
data = {v: pd.read_csv(f"{d}/{v}.csv") for v in variants}
vs = {v: data[v][data[v].cond == "raw"].v_contemporaneous.mean() for v in variants}

rows = []
for key, label in methods:
    cells = []
    for v in variants:
        g = data[v].groupby("cond")[f"f1_{key}"].mean()
        cells += [f"{g['raw']:.2f}", f"{g['std']:.2f}"]
    rows.append(label + " & " + " & ".join(cells) + " \\\\")

tex = [
    "\\begin{tabular}{lcccccc}",
    "\\toprule",
    f" & \\multicolumn{{2}}{{c}}{{full ($v{{=}}{vs['tg_full']:.2f}$)}}"
    f" & \\multicolumn{{2}}{{c}}{{seasonal ($v{{=}}{vs['tg_seasonal']:.2f}$)}}"
    f" & \\multicolumn{{2}}{{c}}{{mechanism ($v{{=}}{vs['tg_mech']:.2f}$)}} \\\\",
    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
    "Method & raw & std. & raw & std. & raw & std. \\\\",
    "\\midrule",
] + rows + ["\\bottomrule", "\\end{tabular}"]
out = pathlib.Path(__file__).parent / "tg_c1_results.tex"
out.write_text("\n".join(tex) + "\n")
print("wrote", out)
