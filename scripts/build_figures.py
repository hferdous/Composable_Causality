"""Build the figures used in the paper.

Writes vector PDFs into figures/. Run from the repository root:

    python scripts/build_figures.py

Figure 1 (the architecture diagram) is drawn in TikZ and compiles with the
paper; its standalone source is figures/fig_architecture_tikz.tex. Every other
figure is produced here:

    figures/fig_ground_truth_grid.pdf   nine reference window graphs
    figures/ground_truth/gt_*.pdf       the same graphs, one file each
    figures/fig_sample_data.pdf         one graph under four DGP variants
    figures/fig_recovered_vs_gt.pdf     DYNOTEARS output beside ground truth
    figures/fig_varsortability.pdf      varsortability audit (needs the audit CSV)

All widths equal the NeurIPS text width (5.5 in), so font sizes in the PDF are
the sizes that print.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# A fixed timestamp makes the PDFs byte-identical from run to run.
os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
TEXTWIDTH = 5.5  # inches, NeurIPS \textwidth

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})
BLUE, ORANGE, GREEN, GREY, RED = "#3B6EA5", "#D1832F", "#4F8A5B", "#8A8A8A", "#B03030"


def _despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ------------------------------------------------------- ground-truth graphs
def fig_ground_truth() -> None:
    """The nine reference templates: one aligned grid plus one file per graph."""
    from dtcd.benchmarks.predefined_graphs import PREDEFINED_GRAPHS
    from dtcd.viz.graph_viz import plot_ground_truth_graph, plot_ground_truth_grid

    plot_ground_truth_grid(PREDEFINED_GRAPHS, str(FIG / "fig_ground_truth_grid.pdf"),
                           width_in=TEXTWIDTH)
    for name, g in PREDEFINED_GRAPHS.items():
        plot_ground_truth_graph(g, str(FIG / "ground_truth" / f"gt_{name}.pdf"))
    print("  figures/fig_ground_truth_grid.pdf, figures/ground_truth/gt_*.pdf")


# ------------------------------------------------------------- sample data
def fig_sample_data() -> None:
    """Same graph, weights and noise seed under four DGP variants."""
    from dtcd.benchmarks.predefined_graphs import PREDEFINED_GRAPHS
    from generate_datasets import generate_ts

    gid, g = "d4_tau2", PREDEFINED_GRAPHS["d4_tau2"]
    variants = [("linear", "Linear"), ("seasonal", "Seasonal"),
                ("drift", "Non-stationary drift"), ("mnar", "MNAR mask")]
    colors = [BLUE, ORANGE, GREEN, GREY]
    fig, axes = plt.subplots(1, 4, figsize=(TEXTWIDTH, 1.85), sharex=True, sharey=True)
    for ax, (v, title) in zip(axes, variants):
        X, mask, *_ = generate_ts(g, gid, v, 400, 42)
        for j in range(g["d"]):
            ax.plot(X[:, j], lw=0.45, color=colors[j], alpha=0.9)
        if v == "mnar":
            miss = np.where(~mask[:, 0])[0]
            trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.vlines(miss, 0.0, 0.045, transform=trans, color=RED, lw=0.5)
        ax.set_title(title, pad=3)
        ax.set_xlabel("$t$", labelpad=1)
        ax.set_xticks([0, 200, 400])
        _despine(ax)
    axes[0].set_ylabel("value")
    handles = [Line2D([0], [0], color=c, lw=1.2) for c in colors]
    labels = [f"$X_{{{j + 1}}}$" for j in range(g["d"])]
    handles.append(Line2D([0], [0], color=RED, lw=1.2))
    labels.append("censored ($X_1$, MNAR)")
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02), handlelength=1.4, columnspacing=1.4)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.88, bottom=0.30, wspace=0.12)
    fig.savefig(FIG / "fig_sample_data.pdf")
    plt.close(fig)
    print("  figures/fig_sample_data.pdf")


# --------------------------------------------------------- varsortability
def fig_varsortability(csv: Path,
                       methods=("DYNOTEARS", "SortnRegress", "RandomRegress")) -> None:
    """Raw versus standardized: how much F1 the variance shortcut carries.

    Reads the linear-variant audit (benchmark_results/audit_linear.csv, stage 2
    of scripts/run_paper_experiments.sh), the same file behind Table 5.
    """
    if not csv.exists():
        print(f"  [skip] {csv} not found; run scripts/run_paper_experiments.sh --stage 2")
        return
    df = pd.read_csv(csv)
    methods = [m for m in methods if f"f1_{m}" in df.columns]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.25),
                                 gridspec_kw={"width_ratios": [1, 1.45]})

    gs = sorted(df.graph.unique())
    raw = [df[(df.graph == g) & (df.condition == "raw")].v_contemporaneous.mean() for g in gs]
    std = [df[(df.graph == g) & (df.condition == "standardized")].v_contemporaneous.mean()
           for g in gs]
    y = np.arange(len(gs))
    a1.barh(y + 0.19, raw, 0.36, color=ORANGE)
    a1.barh(y - 0.19, std, 0.36, color=BLUE)
    a1.axvline(0.5, color="#333333", ls=":", lw=0.8)
    a1.text(0.5, 1.0, "uninformative", transform=mtransforms.blended_transform_factory(
        a1.transData, a1.transAxes), ha="center", va="bottom", fontsize=6.5)
    a1.set_yticks(y)
    a1.set_yticklabels([g.replace("_", " ") for g in gs], fontsize=6.5)
    a1.set_ylim(-0.6, len(gs) - 0.4)
    a1.set_xlim(0, 1.0)
    a1.set_xlabel("contemporaneous varsortability $v$")
    _despine(a1)

    x = np.arange(len(methods))
    m_raw = [df[df.condition == "raw"][f"f1_{m}"].mean() for m in methods]
    m_std = [df[df.condition == "standardized"][f"f1_{m}"].mean() for m in methods]
    e_raw = [df[df.condition == "raw"][f"f1_{m}"].std() for m in methods]
    e_std = [df[df.condition == "standardized"][f"f1_{m}"].std() for m in methods]
    ekw = {"lw": 0.6, "capthick": 0.6}
    a2.bar(x - 0.19, m_raw, 0.36, yerr=e_raw, capsize=2, color=ORANGE, error_kw=ekw)
    a2.bar(x + 0.19, m_std, 0.36, yerr=e_std, capsize=2, color=BLUE, error_kw=ekw)
    for xi, (r, s, er, es) in enumerate(zip(m_raw, m_std, e_raw, e_std)):
        top = max(r + er, s + es)
        txt = f"$-${r - s:.2f}" if r - s > 0.005 else "0.00"
        a2.text(xi, min(top, 1.08) + 0.03, txt, ha="center", va="bottom",
                fontsize=6.8, color=RED if r - s > 0.005 else "#333333")
    a2.set_xticks(x)
    a2.set_xticklabels(methods)
    a2.set_ylabel("F1 (oriented)")
    a2.set_ylim(0, 1.22)
    a2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    _despine(a2)

    fig.legend([Patch(color=ORANGE), Patch(color=BLUE)],
               ["raw ($v \\approx 0.93$)", "standardized ($v = 0.5$)"],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.10, right=0.995, top=0.83, bottom=0.19, wspace=0.32)
    fig.savefig(FIG / "fig_varsortability.pdf")
    plt.close(fig)
    print("  figures/fig_varsortability.pdf")


# ------------------------------------------------- recovered vs ground truth
def fig_recovered_vs_gt() -> None:
    """DYNOTEARS window graph beside the ground truth, raw and standardized."""
    import importlib.util as il

    from dtcd.benchmarks.predefined_graphs import PREDEFINED_GRAPHS
    from dtcd.evaluators.metrics import _unroll_adj, f1_oriented
    from dtcd.utils.varsortability import standardize
    from generate_datasets import generate_ts, get_stable_weights, stable_weight_seed

    spec = il.spec_from_file_location("dyn", ROOT / "src/dtcd/baselines/dynotears_core.py")
    dyn = il.module_from_spec(spec)
    spec.loader.exec_module(dyn)

    gid, g = "d4_tau2", PREDEFINED_GRAPHS["d4_tau2"]
    d, tau = g["d"], g["tau_max"]
    W0t, WLt, _ = get_stable_weights(g, stable_weight_seed(gid))
    Jt = _unroll_adj(W0t, WLt)
    truth = np.abs(Jt) > 0
    X_raw, *_ = generate_ts(g, gid, "linear", 1000, 42)

    panels = [("Ground truth", truth.astype(float), None)]
    for cond, X in (("raw", X_raw), ("standardized", standardize(X_raw))):
        T = X.shape[0]
        w, a = dyn.from_numpy_dynamic(
            X=X[tau:, :], Xlags=np.hstack([X[tau - k:T - k, :] for k in range(1, tau + 1)]),
            lambda_w=0.0044, lambda_a=0.0043, w_threshold=0.1)
        WL = np.zeros((tau, d, d))
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if a[i, j] != 0:
                    WL[i // d, i % d, j] = a[i, j]
        J = _unroll_adj(w, WL)
        panels.append((f"DYNOTEARS, {cond}", (np.abs(J) > 0).astype(float),
                       f1_oriented(J, Jt)))

    n_rows = d * (tau + 1)
    rows = [f"$X_{{{j + 1}}}^{{t}}$" for j in range(d)]
    for k in range(tau):
        rows += [f"$X_{{{j + 1}}}^{{t-{k + 1}}}$" for j in range(d)]

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.75), sharey=True)
    for ax, (title, M, f1) in zip(axes, panels):
        # Vector cells (not a raster), so cell edges stay sharp in the PDF.
        ax.pcolormesh(np.arange(d + 1) - 0.5, np.arange(n_rows + 1) - 0.5, M,
                      cmap="Blues", vmin=0, vmax=1.35, shading="flat",
                      edgecolors="white", linewidth=0.8)
        ax.set_xlim(-0.5, d - 0.5)
        ax.set_ylim(n_rows - 0.5, -0.5)
        for b in range(1, tau + 1):
            ax.axhline(b * d - 0.5, color="#555555", lw=0.6)
        if f1 is not None:
            fp = np.argwhere((M > 0) & ~truth)
            fn = np.argwhere((M == 0) & truth)
            ax.scatter(fp[:, 1], fp[:, 0], marker="x", s=22, color=RED, lw=1.0, zorder=3)
            for r, c in fn:
                ax.add_patch(Rectangle((c - 0.38, r - 0.38), 0.76, 0.76, fill=False,
                                       ec=RED, lw=1.0, zorder=3))
            ax.set_title(f"{title}\nF1 = {f1:.2f}")
        else:
            ax.set_title(f"{title}\n")
        ax.set_xticks(range(d))
        ax.set_xticklabels([f"$X_{{{j + 1}}}^{{t}}$" for j in range(d)])
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(rows, fontsize=6.5)
        ax.set_xlabel("target")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
    axes[0].set_ylabel("parent (lagged window)")
    fig.legend([Line2D([0], [0], marker="x", color=RED, lw=0, ms=5, mew=1.0),
                Line2D([0], [0], marker="s", mfc="none", mec=RED, lw=0, ms=6, mew=1.0)],
               ["spurious edge", "missed edge"], loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.115, right=0.995, top=0.86, bottom=0.21, wspace=0.08)
    fig.savefig(FIG / "fig_recovered_vs_gt.pdf")
    plt.close(fig)
    print("  figures/fig_recovered_vs_gt.pdf  "
          + "  ".join(f"{t}: F1={f:.2f}" for t, _, f in panels[1:]))


if __name__ == "__main__":
    print("building figures:")
    fig_ground_truth()
    fig_sample_data()
    fig_varsortability(ROOT / "benchmark_results" / "audit_linear.csv")
    fig_recovered_vs_gt()
