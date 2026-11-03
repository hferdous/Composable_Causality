"""
DTCD dataset generator.

Variants supported:
    linear, nonlinear, drift,
    mnar, mcar,
    trend, seasonal, trend_seasonal, nonlinear_trend_seasonal,
    cate_linear, cate_nonlinear

DGP fixes (relative to earlier versions):
    A. "pnl" renamed to "nonlinear"; exponent changed from 1.5 to 0.5
       (saturating, prevents overflow on chain SCMs).
    B. MCAR variant added (uniform 20% missingness, decoupled RNG).
    C. Weight realization decoupled from noise seed (fixed per graph).
    D. Counterfactual rollouts share an explicitly-cached noise tensor.

Trend and seasonality are additive exogenous components; they do NOT
enter the structural equations and the ground-truth graph is unchanged.
Seasonality is generated via dtcd.utils.seasonality.SeasonalityGenerator.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np

from dtcd.benchmarks.predefined_graphs import PREDEFINED_GRAPHS
from dtcd.utils.varsortability import (
    contemporaneous_varsortability,
    summary_varsortability,
)
from dtcd.utils.seasonality import SeasonalityGenerator


def stable_weight_seed(gid):
    """Deterministic per-graph weight seed.

    Uses md5 rather than the builtin hash(), which is salted per interpreter
    process unless PYTHONHASHSEED is set and therefore does NOT produce the
    same weights across runs.
    """
    return int(hashlib.md5(gid.encode("utf-8")).hexdigest()[:8], 16) % (2 ** 31)


def get_stable_weights(graph_dict, weight_seed):
    """Sample structural weights using a fixed weight_seed.

    The weight_seed is derived deterministically from the graph identifier
    in generate_ts, so all noise seeds for the same graph use the same
    weights. Reported variance across noise seeds therefore reflects noise
    realization only.
    """
    rng = np.random.RandomState(weight_seed)
    d = graph_dict["d"]
    tau_max = graph_dict["tau_max"]
    inst_edges = graph_dict["instantaneous"]
    lagged_edges = graph_dict["lagged"]

    while True:
        W0 = np.zeros((d, d))
        for src, dst in inst_edges:
            W0[src, dst] = rng.uniform(0.3, 0.8) * rng.choice([-1, 1])

        W_lags = np.zeros((tau_max, d, d))
        for src, dst, lag in lagged_edges:
            W_lags[lag - 1, src, dst] = rng.uniform(0.3, 0.8) * rng.choice([-1, 1])

        try:
            I_W0_inv = np.linalg.inv(np.eye(d) - W0)
            companion = np.zeros((d * tau_max, d * tau_max))
            for k in range(tau_max):
                Ak = I_W0_inv @ W_lags[k]
                companion[:d, k * d : (k + 1) * d] = Ak
            if tau_max > 1:
                companion[d:, : (tau_max - 1) * d] = np.eye((tau_max - 1) * d)
            eigvals = np.linalg.eigvals(companion)
            rho = np.max(np.abs(eigvals))
            if rho < 0.95:
                return W0, W_lags, rho
        except np.linalg.LinAlgError:
            continue


def get_trend_seasonal_components(d, T, seed, *, want_trend, want_seasonal):
    """Per-variable additive trend and seasonality.

    Calls dtcd.utils.seasonality.SeasonalityGenerator for the seasonal
    component. Both components are zero-padded if not requested.
    """
    rng_trend = np.random.RandomState(seed + 90000)
    rng_seasonal = np.random.RandomState(seed + 95000)

    trend = np.zeros((T, d))
    trend_amplitudes = np.zeros(d)
    if want_trend:
        active_trend = rng_trend.random(d) < 0.5
        amps = rng_trend.uniform(0.5, 2.0, size=d) * rng_trend.choice([-1, 1], size=d)
        trend_amplitudes = amps * active_trend
        t_axis = np.arange(T) / T
        trend = np.outer(t_axis, trend_amplitudes)

    seasonal = np.zeros((T, d))
    seasonal_metadata_per_var = [None] * d
    if want_seasonal:
        all_candidate_periods = np.array([12, 24, 50, 168], dtype=int)
        feasible_periods = [P for P in all_candidate_periods if T >= 3 * P]
        if not feasible_periods:
            feasible_periods = [max(4, T // 10)]

        sg = SeasonalityGenerator(random_state=int(seed + 95000))

        for i in range(d):
            n_active = rng_seasonal.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
            if n_active == 0:
                seasonal_metadata_per_var[i] = {"periods": [], "strengths": []}
                continue
            n_active = min(n_active, len(feasible_periods))
            chosen = rng_seasonal.choice(
                feasible_periods, size=n_active, replace=False
            ).tolist()
            strengths = rng_seasonal.uniform(0.3, 1.0, size=n_active).tolist()

            seasonal[:, i] = sg.generate_seasonality(
                n_points=T,
                periods=chosen,
                strengths=strengths,
                phases="random",
                combine_sin_cos=True,
            )
            seasonal_metadata_per_var[i] = {
                "periods": chosen,
                "strengths": strengths,
            }

    metadata = {
        "trend_amplitudes": trend_amplitudes.tolist(),
        "seasonal_per_variable": seasonal_metadata_per_var,
    }
    return trend, seasonal, metadata


def generate_ts(graph_dict, gid, variant, T, seed):
    """Generate one trajectory.

    Parameters
    ----------
    graph_dict : dict
        Predefined graph specification.
    gid : str
        Graph identifier; used to derive the weight_seed deterministically
        per graph (decoupling weights from noise).
    variant : str
        One of: linear, nonlinear, drift, mnar, mcar, trend, seasonal,
        trend_seasonal, nonlinear_trend_seasonal, cate_linear,
        cate_nonlinear, nonlinear_sin, nonlinear_quad.
    T : int
        Trajectory length.
    seed : int
        Per-cell seed for the NOISE realization only.

    Returns
    -------
    X_obs : np.ndarray (T, d)
    mask : np.ndarray (T, d) of bool
    W0 : np.ndarray (d, d)
    W_lags : np.ndarray (tau_max, d, d)
    true_cate : np.ndarray (T,)
    exo_meta : dict
    rho : float
    """
    rng = np.random.RandomState(seed)
    d = graph_dict["d"]
    tau_max = graph_dict["tau_max"]
    tr_node = graph_dict["treatment_node"]
    out_node = graph_dict["outcome_node"]

    weight_seed = stable_weight_seed(gid)
    W0, W_lags, rho = get_stable_weights(graph_dict, weight_seed)

    # A "_std" suffix z-scores every column after simulation, driving
    # varsortability to 0.5 while leaving the ground-truth graph unchanged.
    # This is the Reisach et al. (2021) control condition.
    standardize_out = variant.endswith("_std")
    if standardize_out:
        variant = variant[:-len("_std")]

    is_mnar = variant == "mnar"
    is_mcar = variant == "mcar"

    want_trend = variant in ("trend", "trend_seasonal", "nonlinear_trend_seasonal")
    want_seasonal = variant in ("seasonal", "trend_seasonal", "nonlinear_trend_seasonal")
    trend_arr, seasonal_arr, exo_meta = get_trend_seasonal_components(
        d=d, T=T, seed=seed, want_trend=want_trend, want_seasonal=want_seasonal,
    )

    noises = rng.randn(T, d)

    G0 = nx.DiGraph()
    G0.add_nodes_from(range(d))
    G0.add_edges_from(graph_dict["instantaneous"])
    order = list(nx.topological_sort(G0))

    apply_nonlinearity = variant in (
        "nonlinear", "cate_nonlinear", "nonlinear_trend_seasonal",
        "nonlinear_sin", "nonlinear_quad"
    )
    drift_beta = 0.5 if variant == "drift" else 0.0

    def simulate(noises_in, tr_shift=0.0, is_binary_tr=False, tr_mask=None):
        X_sim = np.zeros((T, d))
        for t in range(T):
            val_t = np.zeros(d)
            for i in order:
                pa_inst = np.dot(W0[:, i], val_t)
                pa_lag = 0
                for k in range(1, tau_max + 1):
                    if t - k >= 0:
                        weight_factor = (
                            1.0 + drift_beta * (t / T) if drift_beta else 1.0
                        )
                        pa_lag += np.dot(W_lags[k - 1, :, i], X_sim[t - k]) * weight_factor

                exogenous = trend_arr[t, i] + seasonal_arr[t, i]
                total_val = pa_inst + pa_lag + exogenous + noises_in[t, i]
                if i == tr_node:
                    if is_binary_tr:
                        total_val = float(tr_mask[t])
                    else:
                        total_val += tr_shift
                if apply_nonlinearity:
                    if variant == "nonlinear_sin":
                        total_val = np.sin(total_val)
                    elif variant == "nonlinear_quad":
                        total_val = total_val * np.abs(total_val)
                    else:
                        # p=0.5 (saturating). p=1.5 was used previously but
                        # overflowed float64 along chain SCMs after ~10 levels.
                        total_val = np.sign(total_val) * np.abs(total_val) ** 0.5
                val_t[i] = total_val
            X_sim[t] = val_t
        return X_sim

    if variant.startswith("cate"):
        # cached_noise must be sampled ONCE and reused across all three
        # simulate() calls. This makes counterfactual twins exact by
        # construction, not by accidental RNG state alignment.
        cached_noise = noises
        tr_mask_obs = rng.binomial(1, 0.5, size=T)

        X_obs = simulate(cached_noise, is_binary_tr=True, tr_mask=tr_mask_obs)
        X_cf1 = simulate(cached_noise, is_binary_tr=True, tr_mask=np.ones(T))
        X_cf0 = simulate(cached_noise, is_binary_tr=True, tr_mask=np.zeros(T))
        true_cate = X_cf1[:, out_node] - X_cf0[:, out_node]
    else:
        X_obs = simulate(noises)
        true_cate = np.zeros(T)

    mask = np.ones_like(X_obs, dtype=bool)
    if is_mnar:
        for i in range(d):
            thresh = np.percentile(X_obs[:, i], 80)
            mask[:, i] = X_obs[:, i] <= thresh
    elif is_mcar:
        mcar_rng = np.random.RandomState(seed + 50000)
        mask = mcar_rng.random(X_obs.shape) > 0.20

    if standardize_out:
        from dtcd.utils.varsortability import standardize
        X_obs = standardize(X_obs, mask if (is_mnar or is_mcar) else None)

    return X_obs, mask, W0, W_lags, true_cate, exo_meta, rho


def main():
    out_dir = Path("data/datasets")
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        "linear", "nonlinear", "drift",
        "mnar", "mcar",
        "trend", "seasonal", "trend_seasonal",
        "nonlinear_trend_seasonal",
        "cate_linear", "cate_nonlinear",
        # standardized controls for the varsortability audit
        "linear_std", "nonlinear_std", "trend_seasonal_std",
    ]
    # Seed 41 is the held-out hyperparameter-tuning trajectory; 42-51 are the
    # reported evaluation seeds. Tuning must never touch an evaluation seed.
    TUNING_SEED = 41
    seeds = [TUNING_SEED] + list(range(42, 52))

    for gid, g in PREDEFINED_GRAPHS.items():
        for var in variants:
            print(f"Generating {gid} | {var}...")
            for s in seeds:
                X, mask, W0, W_lags, true_cate, exo_meta, rho = generate_ts(
                    g, gid, var, 1000, s
                )

                X_obs_for_v = X.copy()
                if var in ("mnar", "mcar"):
                    X = X.copy()
                    X[~mask] = np.nan

                prefix = f"{gid}_{var}_s{s}"
                np.save(out_dir / f"{prefix}_X.npy", X)
                np.save(out_dir / f"{prefix}_mask.npy", mask)
                np.save(out_dir / f"{prefix}_cate.npy", true_cate)
                gt = np.concatenate([W0[None, ...], W_lags], axis=0)
                np.save(out_dir / f"{prefix}_gt.npy", gt)

                # Varsortability of the emitted cell (Reisach et al. 2021).
                # Recorded so that any result produced from this dataset can be
                # read against the marginal-variance shortcut it might exploit.
                # Computed on the pre-mask array: NaNs would make the variances
                # undefined, and the mask is an observation process, not part of
                # the generating mechanism whose sortability is at issue.
                v_contemp = contemporaneous_varsortability(X_obs_for_v, W0)
                v_summary = summary_varsortability(X_obs_for_v, W0, W_lags)

                meta = {
                    "graph_id": gid,
                    "variant": var,
                    "seed": s,
                    "varsortability_contemporaneous": v_contemp,
                    "varsortability_summary": v_summary,
                    "weight_seed": stable_weight_seed(gid),
                    "T": 1000,
                    "d": int(g["d"]),
                    "tau_max": int(g["tau_max"]),
                    "spectral_radius": float(rho),
                    "treatment_node": int(g["treatment_node"]),
                    "outcome_node": int(g["outcome_node"]),
                    "exogenous": exo_meta if (
                        "trend" in var or "seasonal" in var
                    ) else None,
                }
                with open(out_dir / f"{prefix}_meta.json", "w") as f:
                    json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
