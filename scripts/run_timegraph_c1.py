"""TimeGraph C1 equations evaluated through the DTCD harness.

Generates data with the published TimeGraph C1 generator (imported from
scripts/external/timegraph_c1.py, not transcribed), builds the ground-truth
window graph by parsing the generator's equation strings, and runs six
DTCD-registered discovery methods under the audit protocol: raw and
standardized conditions, per-cell varsortability, and oriented F1 on the
unrolled window graph (the same metric as the reference benchmark).

Three variants separate the exogenous components:
  tg_full     : TimeGraph defaults (trend strength 0.01, seasonal strength 0.5, period 12)
  tg_seasonal : trend set to 0, seasonality kept
  tg_mech     : trend and seasonality set to 0, isolating the mechanism family

DYNOTEARS receives an oracle-tuned advantage: a 3 x 2 grid over
(lambda_w, lambda_a) in {0.01, 0.1, 0.3} x {0.01, 0.1}, selected on a held-out
seed (41) separately for every configuration and condition. Every other method
runs at library defaults. The asymmetry is deliberate: if the tuned method
still collapses, the conclusion is about the equation family, not the tuning.
Pass --no-tune to run DYNOTEARS at defaults too (as done for tg_mech).

    python scripts/run_timegraph_c1.py --variant tg_seasonal
"""
import argparse
import re
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "external"))

from timegraph_c1 import NonlinearTimeSeriesGenerator, get_nonlinear_equations  # noqa: E402
from dtcd.baselines.registry import DISCOVERY_REGISTRY as REG  # noqa: E402
from dtcd.evaluators.metrics import _unroll_adj, f1_oriented  # noqa: E402
from dtcd.utils.varsortability import contemporaneous_varsortability, standardize  # noqa: E402

TERM = re.compile(r"X(\d+)\[t(?:-(\d+))?\]")


class FitTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise FitTimeout()


def fit_with_timeout(method, X, seconds=180):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)
    try:
        return method.fit(X)
    finally:
        signal.alarm(0)


def truth_from_equations(n_vars: int, max_lag: int):
    """Parse the C1 equation strings into (W0, W_lags) binary adjacencies.

    Convention matches _unroll_adj: W[i, j] = 1 means variable i is a parent
    of variable j (i -> j); W_lags[k] holds lag k+1 parents.
    """
    W0 = np.zeros((n_vars, n_vars))
    W_lags = np.zeros((max_lag, n_vars, n_vars))
    for eq in get_nonlinear_equations(n_vars, max_lag):
        left, right = eq.split("=", 1)
        target = int(left.split("[")[0].strip()[1:]) - 1
        for m in TERM.finditer(right):
            parent = int(m.group(1)) - 1
            lag = int(m.group(2)) if m.group(2) else 0
            if lag == 0:
                W0[parent, target] = 1
            else:
                W_lags[lag - 1, parent, target] = 1
    return W0, W_lags


def generate(n_vars, max_lag, seed, variant, n_points):
    kw = dict(noise_scale=0.1, seasonal_period=12, random_state=seed)
    if variant == "tg_full":
        gen = NonlinearTimeSeriesGenerator(trend_strength=0.01,
                                           seasonal_strength=0.5, **kw)
    elif variant == "tg_mech":
        gen = NonlinearTimeSeriesGenerator(trend_strength=0.0,
                                           seasonal_strength=0.0, **kw)
    elif variant == "tg_seasonal":
        gen = NonlinearTimeSeriesGenerator(trend_strength=0.0,
                                           seasonal_strength=0.5, **kw)
    else:
        raise ValueError(variant)
    df = gen.generate_multivariate_ts(n_points, n_vars, max_lag)
    X = df[[f"X{i+1}" for i in range(n_vars)]].to_numpy(dtype=float)
    assert np.isfinite(X).all(), "generator produced non-finite values"
    return X


def tune_dynotears(Xtune, W_true, max_lag, grid):
    best, best_hp = -1.0, None
    for lw in grid:
        for la in grid[:2]:
            m = REG["DYNOTEARS"](max_lag=max_lag, seed=0,
                                 lambda_w=lw, lambda_a=la, w_threshold=0.1)
            try:
                r = fit_with_timeout(m, Xtune)
            except FitTimeout:
                continue
            f1 = f1_oriented(_unroll_adj(r.W_inst, r.W_lag), W_true)
            if f1 > best:
                best, best_hp = f1, {"lambda_w": lw, "lambda_a": la,
                                     "w_threshold": 0.1}
    return best_hp, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["tg_full", "tg_mech", "tg_seasonal"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--tune-seed", type=int, default=41)
    ap.add_argument("--methods", nargs="+", default=[
        "DYNOTEARS", "SortnRegress", "RandomRegress",
        "VARLiNGAM", "GrangerF", "PCMCI_ParCorr"])
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--out", default=None,
                    help="output CSV (default: benchmark_results/<variant>.csv)")
    args = ap.parse_args()
    if args.out is None:
        args.out = str(ROOT / "benchmark_results" / f"{args.variant}.csv")

    grid = [0.01, 0.1, 0.3]  # lambda_w over all three; lambda_a over the first two
    rows, tuned = [], {}
    for n_vars in (4, 6, 8):
        for max_lag in (2, 3, 4):
            W0, W_lags = truth_from_equations(n_vars, max_lag)
            W_true = _unroll_adj(W0, W_lags)
            n_edges = int(W_true.sum())
            for cond in ("raw", "std"):
                key = (n_vars, max_lag, cond)
                Xt = generate(n_vars, max_lag, args.tune_seed,
                              args.variant, args.n)
                if cond == "std":
                    Xt = standardize(Xt)
                if "DYNOTEARS" in args.methods and not args.no_tune:
                    hp, tune_f1 = tune_dynotears(Xt, W_true, max_lag, grid)
                    tuned[key] = hp
                    print(f"[tune] d={n_vars} lag={max_lag} {cond}: "
                          f"{hp} (held-out F1 {tune_f1:.3f})", flush=True)
                for seed in args.seeds:
                    X = generate(n_vars, max_lag, seed, args.variant, args.n)
                    v = contemporaneous_varsortability(X, W0)
                    Xc = standardize(X) if cond == "std" else X
                    rec = dict(variant=args.variant, d=n_vars,
                               max_lag=max_lag, cond=cond, seed=seed,
                               n_true_edges=n_edges, v_contemporaneous=v)
                    for name in args.methods:
                        hp = tuned.get(key, {}) if name == "DYNOTEARS" else {}
                        t0 = time.perf_counter()
                        try:
                            m = REG[name](max_lag=max_lag, seed=seed, **hp)
                            r = fit_with_timeout(m, Xc)
                            rec[f"f1_{name}"] = f1_oriented(
                                _unroll_adj(r.W_inst, r.W_lag), W_true)
                            rec[f"status_{name}"] = getattr(r, "status", "ok")
                        except Exception as e:  # noqa: BLE001
                            rec[f"f1_{name}"] = np.nan
                            rec[f"status_{name}"] = ("timeout" if isinstance(e, FitTimeout) else f"error:{type(e).__name__}")
                        rec[f"rt_{name}"] = round(time.perf_counter() - t0, 2)
                    rows.append(rec)
                    print(f"  d={n_vars} lag={max_lag} {cond} seed={seed} "
                          + " ".join(f"{m}={rec[f'f1_{m}']:.2f}"
                                     for m in args.methods
                                     if rec[f"f1_{m}"] == rec[f"f1_{m}"]),
                          flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
