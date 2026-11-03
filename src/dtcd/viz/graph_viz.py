"""Rendering of ground-truth window graphs.

Each graph is drawn on a circular layout. Contemporaneous edges are solid
black; lagged edges are dashed, coloured by lag, and labelled with their lag
on the arc itself. Edges that share a node pair receive distinct curvatures
so that neither the arcs nor their labels overlap.

Usage
-----
    from dtcd.benchmarks.predefined_graphs import PREDEFINED_GRAPHS
    from dtcd.viz.graph_viz import plot_ground_truth_graph, plot_ground_truth_grid

    plot_ground_truth_graph(PREDEFINED_GRAPHS["d4_tau2"], "gt_d4_tau2.pdf")
    plot_ground_truth_grid(PREDEFINED_GRAPHS, "fig_ground_truth_grid.pdf")
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch  # noqa: E402

# Okabe-Ito colours, distinguishable under the common forms of colour blindness.
LAG_COLORS = {1: "#0072B2", 2: "#CC79A7", 3: "#009E73", 4: "#E69F00"}
TREATMENT_EDGE, OUTCOME_EDGE, NODE_EDGE = "#2E8B57", "#C0392B", "#7F7F7F"
NODE_RADIUS = 0.15
BULGE = 0.17  # arc midpoint offset from its chord, in layout units
LAYOUT_RADIUS = 1.0
# STIX ships with matplotlib and matches the Times body text of the paper.
STYLE = {"font.family": "STIXGeneral", "mathtext.fontset": "stix",
         "pdf.fonttype": 42, "ps.fonttype": 42}


def _layout(d: int) -> np.ndarray:
    """Nodes on a circle, node 0 at the top, proceeding clockwise."""
    angles = np.pi / 2 - 2 * np.pi * np.arange(d) / d
    return LAYOUT_RADIUS * np.column_stack([np.cos(angles), np.sin(angles)])


def _bezier(p0: np.ndarray, p2: np.ndarray, rad: float, t: float) -> np.ndarray:
    """Point on the quadratic Bezier curve matplotlib draws for arc3,rad=rad."""
    dx, dy = p2 - p0
    ctrl = (p0 + p2) / 2 + rad * np.array([dy, -dx])
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p2


def _arc_samples(p0: np.ndarray, p2: np.ndarray, rad: float, n: int = 17) -> np.ndarray:
    return np.array([_bezier(p0, p2, rad, t) for t in np.linspace(0.12, 0.88, n)])


def _choose_rad(src: int, dst: int, pos: np.ndarray, taken_mids: List[np.ndarray],
                has_straight: bool) -> float:
    """Pick the curvature for one lagged edge by maximising clearance.

    Candidates bow to either side of the chord with three bulge sizes. Each is
    scored by its clearance from every node other than its endpoints and from
    the midpoints of arcs already drawn on the same node pair; ties favour a
    small bulge and, for neighbouring nodes, the outward side.
    """
    d = len(pos)
    p0, p2 = pos[src], pos[dst]
    chord = p2 - p0
    length = float(np.linalg.norm(chord))
    normal = np.array([chord[1], -chord[0]])
    mid = (p0 + p2) / 2
    outward = 1.0 if (np.dot(normal, mid) >= 0 or np.linalg.norm(mid) < 1e-6) else -1.0
    adjacent = (abs(src - dst) % d) in (1, d - 1)
    others = np.array([pos[k] for k in range(d) if k not in (src, dst)])
    best_rad, best_score = 0.0, -np.inf
    for bulge in (BULGE, 1.8 * BULGE, 2.7 * BULGE):
        for side in (outward, -outward):
            rad = side * min(2.0 * bulge / max(length, 1e-6), 1.2)
            pts = _arc_samples(p0, p2, rad)
            clear = (np.min(np.linalg.norm(pts[:, None, :] - others[None], axis=2))
                     - NODE_RADIUS) if len(others) else 1.0
            q = _bezier(p0, p2, rad, 0.5)
            sep = min((np.linalg.norm(q - m) for m in taken_mids), default=1.0)
            if has_straight:
                sep = min(sep, bulge)
            score = 4.0 * min(clear, 0.2) + 2.0 * min(sep, 0.15) - 0.6 * (bulge - BULGE)
            if adjacent and side == outward:
                score += 0.05
            if score > best_score:
                best_rad, best_score = rad, score
    return best_rad


def _place_label(p0, p2, rad, taken: List[np.ndarray], nodes: np.ndarray,
                 lines: np.ndarray):
    """Choose a point along the arc that keeps clear of labels, nodes and other arcs."""
    best, best_score = None, -np.inf
    for t in (0.5, 0.44, 0.56, 0.38, 0.62, 0.32, 0.68):
        q = _bezier(p0, p2, rad, t)
        d_lab = min((np.linalg.norm(q - o) for o in taken), default=9.0)
        d_node = np.min(np.linalg.norm(nodes - q, axis=1)) - NODE_RADIUS
        d_line = np.min(np.linalg.norm(lines - q, axis=1)) if len(lines) else 9.0
        score = (min(d_lab, 0.35) + 1.5 * min(d_node, 0.18) + 0.6 * min(d_line, 0.12)
                 - 0.15 * abs(t - 0.5))
        if score > best_score:
            best, best_score = q, score
    return best


def draw_window_graph(ax, graph_dict: Dict[str, Any], title: Optional[str] = None,
                      fontsize: float = 7.0) -> None:
    """Draw one ground-truth window graph onto an existing Axes."""
    d = graph_dict["d"]
    pos = _layout(d)
    treatment = graph_dict.get("treatment_node")
    outcome = graph_dict.get("outcome_node")

    circles = []
    for i in range(d):
        ec = (TREATMENT_EDGE if i == treatment else
              OUTCOME_EDGE if i == outcome else NODE_EDGE)
        c = Circle(pos[i], NODE_RADIUS, facecolor="white", edgecolor=ec,
                   linewidth=1.1, zorder=3)
        ax.add_patch(c)
        circles.append(c)
        ax.text(*pos[i], f"$X_{{{i + 1}}}$", ha="center", va="center",
                fontsize=fontsize, zorder=4)

    inst = [(s, t) for s, t in graph_dict["instantaneous"]]
    lagged = sorted([(s, t, lag) for s, t, lag in graph_dict["lagged"]],
                    key=lambda e: -np.linalg.norm(pos[e[0]] - pos[e[1]]))
    straight_pairs = {(min(s, t), max(s, t)) for s, t in inst}
    mids: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)
    drawn = []  # (src, dst, lag, rad)
    for s_, t_ in inst:
        drawn.append((s_, t_, 0, 0.0))
    for s_, t_, lag in lagged:
        pair = (min(s_, t_), max(s_, t_))
        rad = _choose_rad(s_, t_, pos, mids[pair], pair in straight_pairs)
        mids[pair].append(_bezier(pos[s_], pos[t_], rad, 0.5))
        drawn.append((s_, t_, lag, rad))

    taken: List[np.ndarray] = []
    for k, (src, dst, lag, rad) in enumerate(drawn):
        color = "black" if lag == 0 else LAG_COLORS.get(lag, "#555555")
        ax.add_patch(FancyArrowPatch(
            pos[src], pos[dst], patchA=circles[src], patchB=circles[dst],
            connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
            mutation_scale=8, linewidth=1.0, color=color,
            linestyle="-" if lag == 0 else (0, (3.0, 1.6)), shrinkA=1, shrinkB=1,
            zorder=2))
    for k, (src, dst, lag, rad) in enumerate(drawn):
        if lag == 0:
            continue
        others = [e for j, e in enumerate(drawn) if j != k]
        lines = (np.vstack([_arc_samples(pos[a], pos[b], r, 25) for a, b, _, r in others])
                 if others else np.empty((0, 2)))
        q = _place_label(pos[src], pos[dst], rad, taken, pos, lines)
        taken.append(q)
        color = LAG_COLORS.get(lag, "#555555")
        ax.text(q[0], q[1], f"$\\tau$={lag}", ha="center", va="center",
                fontsize=fontsize - 1.0, color=color, zorder=5,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))

    lim = LAYOUT_RADIUS + NODE_RADIUS + 0.12
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=fontsize + 1, pad=2)


def plot_ground_truth_graph(graph_dict: Dict[str, Any],
                            output_path: Optional[str] = None,
                            title: Optional[str] = None):
    """Render a single window graph to ``output_path`` (PDF or PNG)."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(2.0, 2.0))
        draw_window_graph(ax, graph_dict, title=title)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, bbox_inches="tight")
            plt.close(fig)
    return fig


def plot_ground_truth_grid(graphs: Dict[str, Dict[str, Any]],
                           output_path: str,
                           dims: Iterable[int] = (4, 6, 8),
                           lags: Iterable[int] = (2, 3, 4),
                           width_in: float = 5.5) -> None:
    """Render the reference templates as one aligned grid (rows d, columns tau_max)."""
    dims, lags = list(dims), list(lags)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(dims), len(lags),
                                 figsize=(width_in, width_in * len(dims) / len(lags) * 1.04))
        for r, d in enumerate(dims):
            for c, tau in enumerate(lags):
                g = graphs[f"d{d}_tau{tau}"]
                draw_window_graph(axes[r, c], g,
                                  title=f"$d{{=}}{d},\\ \\tau_{{\\max}}{{=}}{tau}$")
        fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01,
                            wspace=0.04, hspace=0.12)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)


if __name__ == "__main__":
    import sys

    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "src"))
    from dtcd.benchmarks.predefined_graphs import PREDEFINED_GRAPHS

    out = root / "figures" / "ground_truth"
    for name, g in PREDEFINED_GRAPHS.items():
        plot_ground_truth_graph(g, str(out / f"gt_{name}.pdf"))
    plot_ground_truth_grid(PREDEFINED_GRAPHS, str(root / "figures" / "fig_ground_truth_grid.pdf"))
    print("wrote", out, "and figures/fig_ground_truth_grid.pdf")
