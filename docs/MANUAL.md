# DTCD Manual

This manual covers the compositional API of DTCD in more depth than the README. Every Python block below runs as written, in order, after `pip install -e .` from the repository root; later blocks reuse names defined in earlier ones.

## Contents

1. [Graph topologies](#1-graph-topologies)
2. [Structural mechanisms](#2-structural-mechanisms)
3. [Noise and non-stationarity](#3-noise-and-non-stationarity)
4. [Composing a generator](#4-composing-a-generator)
5. [Interventions and counterfactual twins](#5-interventions-and-counterfactual-twins)
6. [Missing data](#6-missing-data)
7. [Metrics and exporters](#7-metrics-and-exporters)
8. [Extending DTCD](#8-extending-dtcd)

---

## 1. Graph topologies

Graph generators return a links dictionary that maps `(parent, -lag, child, power)` to an edge weight. Lag 0 marks a contemporaneous edge; contemporaneous edges always run from a lower to a higher variable index, so the index order is a valid causal order.

```python
import numpy as np
from dtcd.structure.graph import (generate_ba_dag, generate_er_dag,
                                  generate_random_dag, generate_ws_dag)

# Erdos-Renyi: every admissible edge is present with probability 0.3
links_er = generate_er_dag(n_vars=10, edge_prob=0.3, max_lag=3, random_state=42)

# Barabasi-Albert (scale-free): preferential attachment, a few hubs and many leaves
links_ba = generate_ba_dag(n_vars=15, m_edges=2, max_lag=2, random_state=42)

# Watts-Strogatz (small-world): a ring lattice rewired with probability 0.1
links_ws = generate_ws_dag(n_vars=20, k_neighbors=4, rewire_prob=0.1, max_lag=1, random_state=42)

# A fixed number of edges, sampled uniformly over lags 0..max_lag
links = generate_random_dag(n_vars=5, n_edges=8, max_lag=2, random_state=42)
print(len(links), "edges, for example", next(iter(links.items())))
```

The nine reference templates of the paper's benchmark are in `dtcd.benchmarks.predefined_graphs.PREDEFINED_GRAPHS`.

---

## 2. Structural mechanisms

A mechanism maps the values of a node's parents to its structural value. Besides the linear default, the library provides polynomial, tanh, pairwise-interaction, Fourier-basis (`SplineMechanism`), and multilayer-perceptron mechanisms, a post-nonlinear wrapper `PNLWrapper` that applies a strictly monotone distortion after the noise is added, and a `MultiplicativeWrapper` in which the noise enters multiplicatively. `SoftmaxMechanism` and `PoissonMechanism` in `dtcd.mechanisms.discrete` cover discrete variables.

```python
from dtcd.mechanisms.base import MultiplicativeWrapper, PNLWrapper, SplineMechanism

rng = np.random.default_rng(0)

# Fourier basis with two harmonics; weights have shape (n_parents, 2 * n_harmonics)
fourier = SplineMechanism(weights=rng.normal(scale=0.5, size=(3, 4)), n_harmonics=2)

# Y = g(f(parents) + noise) with a strictly monotone g ("signed_power", "tanh", or "cubic_perturb")
pnl = PNLWrapper(base_mechanism=fourier, g_type="tanh")

# Y = f(parents) * noise
multiplicative = MultiplicativeWrapper(base_mechanism=fourier)

print(fourier(np.array([0.1, -0.4, 1.2])))
```

---

## 3. Noise and non-stationarity

Noise models implement `sample(size, parents_values=None)`. The library provides Gaussian, Laplace, Student-t, exponential, uniform, Cauchy, heteroscedastic (scale grows with the parents' magnitude), drifting-scale, and mixture noise.

```python
from dtcd.noise.models import CauchyNoise, NonStationaryNoise, StudentTNoise

heavy_tails = CauchyNoise(location=0.0, scale=0.1, random_state=0)
drifting = NonStationaryNoise(initial_scale=0.1, drift=0.005, random_state=0)  # scale grows with t
student = StudentTNoise(df=3, scale=0.1, random_state=0)
print(heavy_tails.sample(size=3))
```

---

## 4. Composing a generator

`UniversalTimeSeriesGenerator` combines a graph schedule, per-variable mechanisms, a noise model, an intervention manager, and an observation process. A `RegimeScheduler` holds one or more graphs and the time step at which each becomes active, so a single regime gives a static graph and several regimes give structural change.

```python
from dtcd.generators.universal import UniversalTimeSeriesGenerator
from dtcd.noise.models import LaplaceNoise
from dtcd.structure.scheduler import RegimeScheduler

links_a = generate_random_dag(n_vars=4, n_edges=6, max_lag=2, random_state=1)
links_b = generate_random_dag(n_vars=4, n_edges=6, max_lag=2, random_state=2)

scheduler = RegimeScheduler(n_vars=4, max_lag=2, regimes=[
    {"start": 0, "links": links_a},
    {"start": 500, "links": links_b},   # the graph switches at t = 500
])

generator = UniversalTimeSeriesGenerator(
    n_vars=4, max_lag=2, scheduler=scheduler,
    noise_model=LaplaceNoise(scale=0.1, random_state=42),   # non-Gaussian noise
    random_state=42,
)
df = generator.generate_multivariate_ts(n_points=1000)
print(df.shape, list(df.columns))
```

Variables without an entry in `mechanisms` use the edge weights linearly. To give a variable its own functional form, pass `mechanisms={"X3": some_mechanism}`; the mechanism receives that variable's parent values sorted by `(parent, lag)`.

---

## 5. Interventions and counterfactual twins

`InterventionManager` holds hard (`do`) and soft (additive) interventions, each with an optional time window, a condition, or a policy that computes the intervention value from the history. `CounterfactualSuite` generates a factual trajectory, stores its noise realization, and replays the same noise under a second intervention manager, so the two trajectories differ only through the intervention.

```python
from dtcd.interventions.base import CounterfactualSuite, InterventionManager

im = InterventionManager()

# Soft policy: add 5.0 to X2 whenever X1 is strongly negative at time t
im.add_intervention(
    target="X2",
    intervention_type="soft",
    policy=lambda t, history, current: 5.0 if current.get("X1", 0.0) < -0.2 else 0.0,
)

suite = CounterfactualSuite(generator)
df_obs, df_twin = suite.generate_twins(n_points=1000, intervention_manager_for_twin=im)

effect = (df_twin["X2"] - df_obs["X2"]).to_numpy()
print("twins identical until t =", int(np.argmax(effect != 0)),
      "| X2 differs at", int((effect != 0).sum()), "of", len(effect), "steps")
```

---

## 6. Missing data

The reference benchmark uses two masks: MCAR, where each entry is missing with a fixed probability, and MNAR, where a variable's large values are the ones that go missing. Both are available as standalone functions:

```python
from dtcd.observation.process import ObservationProcess
from dtcd.utils.missing_data import MissingDataHandler

X = df[["X1", "X2", "X3", "X4"]].to_numpy()

# MNAR: values above 0.2 go missing with probability 0.9
X_mnar = ObservationProcess(random_state=0).apply_mnar(data=X, threshold=0.2, above=True, prob=0.9)

# MCAR: each entry missing with probability 0.2
mcar_mask = MissingDataHandler(random_state=0).generate_mcar_mask(X.shape, prob=0.2)  # True = missing
X_mcar = np.where(mcar_mask, np.nan, X)
print("MNAR missing fraction:", round(float(np.isnan(X_mnar).mean()), 3),
      "MCAR missing fraction:", round(float(np.isnan(X_mcar).mean()), 3))
```

Discovery wrappers declare `supports_missing`; the benchmark harness skips methods that do not support missing values on MCAR and MNAR cells and records the skip.

---

## 7. Metrics and exporters

```python
from dtcd.utils.export import extract_adjacency_matrix, to_causal_learn_matrix, to_tigramite_links
from dtcd.utils.metrics import compute_f1_score, compute_shd

# Ground truth of the first regime above, in three layouts; the tensor is indexed [lag, source, target]
tensor_truth = extract_adjacency_matrix(links_a, n_vars=4, max_lag=2, format_type="tensor")
tigramite_links = to_tigramite_links(links_a, n_vars=4, max_lag=2)
causal_learn_adj = to_causal_learn_matrix(links_a, n_vars=4, max_lag=2)

# Score an estimate with one spurious edge (X4 -> X1 at lag 0) against the truth
truth = (tensor_truth != 0).astype(float)
estimate = truth.copy()
estimate[0, 3, 0] = 1.0
print("SHD:", compute_shd(true_adj=truth, pred_adj=estimate),
      "F1:", round(compute_f1_score(true_adj=truth, pred_adj=estimate), 2))
```

The paper's metrics (oriented F1 and SHD on the unrolled window graph, AUPRC, PEHE) are implemented in `dtcd.evaluators.metrics`.

---

## 8. Extending DTCD

Every axis of the generator is a small interface, so adding a mechanism, a noise model, a missingness process, or a discovery baseline means writing one class or function. Nothing in the generator core changes, and the new component composes with every existing axis.

### 8.1 A new structural mechanism

Subclass `Mechanism` and implement `__call__`. The optional `combine_noise` and `post_nonlinear` hooks let a mechanism depart from the additive-noise model.

```python
from dtcd.mechanisms.base import Mechanism


class SaturatingMechanism(Mechanism):
    """f(pa) = scale * tanh(w . pa / scale): linear near 0, saturating far out.

    Locally close to a linear SVAR, so linear learners look adequate on short
    trajectories and degrade as T grows and the tails are sampled.
    """

    def __init__(self, weights, scale: float = 2.0):
        self.weights = np.asarray(weights, dtype=float)
        self.scale = float(scale)

    def __call__(self, parents_values: np.ndarray) -> float:
        z = float(np.dot(self.weights[:len(parents_values)], parents_values))
        return self.scale * np.tanh(z / self.scale)


class MultiplicativeSaturating(SaturatingMechanism):
    """Same structural form, with noise entering multiplicatively."""

    def combine_noise(self, structural_val: float, noise_val: float) -> float:
        return structural_val * (1.0 + 0.3 * noise_val)


custom = UniversalTimeSeriesGenerator(
    n_vars=4, max_lag=2,
    scheduler=RegimeScheduler(n_vars=4, max_lag=2, regimes=[{"start": 0, "links": links_a}]),
    mechanisms={"X4": SaturatingMechanism(weights=[0.6, -0.4, 0.5]),
                "X3": MultiplicativeSaturating(weights=[0.5, 0.5])},
    random_state=42,
)
print(custom.generate_multivariate_ts(n_points=500).describe().loc[["mean", "std"]].round(3))
```

Identifiability is the author's responsibility to state. If a new mechanism is not provably identifiable, say so in its docstring, as the built-in recursive power transform does.

### 8.2 A new noise distribution

Subclass `NoiseModel` and implement `sample`:

```python
from dtcd.noise.models import NoiseModel


class GaussianMixtureNoise(NoiseModel):
    """90% N(0, s^2), 10% N(0, (3s)^2): heavy-tailed with finite variance."""

    def __init__(self, scale=0.1, random_state=None):
        super().__init__(random_state)
        self.scale = scale

    def sample(self, size=1, parents_values=None):
        heavy = self.rng.random_sample(size) < 0.10
        return np.where(heavy, self.rng.normal(0, 3 * self.scale, size),
                        self.rng.normal(0, self.scale, size))


mixture_gen = UniversalTimeSeriesGenerator(n_vars=4, max_lag=2, scheduler=scheduler,
                                           noise_model=GaussianMixtureNoise(random_state=0),
                                           random_state=0)
print(mixture_gen.generate_multivariate_ts(n_points=300).shape)
```

### 8.3 A new missingness mechanism

A missingness process maps a complete trajectory to a mask. Because it sees the true values, it can be informative:

```python
def burst_mnar(X: np.ndarray, rng, burst_len: int = 20, threshold: float = 0.3) -> np.ndarray:
    """A sensor that fails for a stretch after reading an extreme value.

    Returns a boolean mask in which False marks an unobserved entry. Informative
    in both value and duration, unlike the built-in per-entry MNAR.
    """
    T, d = X.shape
    observed = np.ones((T, d), dtype=bool)
    for j in range(d):
        t = 0
        while t < T:
            if abs(X[t, j]) > threshold and rng.random() < 0.5:
                observed[t:t + burst_len, j] = False
                t += burst_len
            else:
                t += 1
    return observed


observed = burst_mnar(X, np.random.default_rng(0))
X_burst = np.where(observed, X, np.nan)
print("burst MNAR missing fraction:", round(float(np.isnan(X_burst).mean()), 3))
```

Compute the varsortability diagnostic (Section 8.5) on the complete array, since a mask is an observation process rather than part of the generating mechanism.

### 8.4 A new discovery baseline

Subclass `BaseDiscoveryMethod`, implement `fit`, and register the class. The class attributes tell the benchmark harness whether a method is eligible for a cell, for example whether it runs on the MCAR and MNAR variants.

```python
import time

from dtcd.baselines.base import AdjacencyResult, BaseDiscoveryMethod
from dtcd.baselines.registry import DISCOVERY_REGISTRY


class LaggedCorrelation(BaseDiscoveryMethod):
    """Toy baseline: a lagged edge wherever |corr(X_j[t-k], X_i[t])| > threshold."""

    name = "LaggedCorrelation"
    paradigm = "granger"          # granger | constraint | score | functional | amortized | control
    compute_class = "cpu"
    supports_instantaneous = False
    supports_missing = False      # the harness records skipped_no_missing_support

    def fit(self, X, mask=None) -> AdjacencyResult:
        T, d = X.shape
        t0 = time.perf_counter()
        threshold = self.hp.get("threshold", 0.3)
        W_inst = np.zeros((d, d))
        W_lag = np.zeros((self.max_lag, d, d))
        for k in range(1, self.max_lag + 1):
            C = np.corrcoef(X[:-k].T, X[k:].T)[:d, d:]   # C[j, i] = corr(X_j[t-k], X_i[t])
            W_lag[k - 1] = np.where(np.abs(C) > threshold, C, 0.0)
        return AdjacencyResult(
            W_inst=W_inst, W_lag=W_lag,
            W_inst_score=np.abs(W_inst), W_lag_score=np.abs(W_lag),
            hp=self.hp, runtime_s=time.perf_counter() - t0,
            peak_memory_mb=0.0, status="ok", error_msg=None,
        )


DISCOVERY_REGISTRY["LaggedCorrelation"] = LaggedCorrelation   # or add it in registry.py
result = DISCOVERY_REGISTRY["LaggedCorrelation"](max_lag=2, seed=0, threshold=0.3).fit(X)
print(result.status, int((result.W_lag != 0).sum()), "lagged edges")
```

To make the method permanent, add it to `DISCOVERY_REGISTRY` in `src/dtcd/baselines/registry.py`. If it needs an optional package, guard the import there so a core-only install still works:

```python
try:
    from dtcd.baselines.my_method import MyMethod
except ImportError:
    MyMethod = None
```

A tuning space is optional. If `get_tuning_space` in `src/dtcd/baselines/tuning.py` has no entry for the method, it runs at its defaults and the harness records `hp_source='default'`. The method is then available to every runner, including the varsortability audit and the scalability sweep.

### 8.5 Reporting varsortability for a new component

A new mechanism or noise model changes how variance accumulates along causal paths, and with it varsortability (Reisach et al., 2021). The diagnostic is one call:

```python
from dtcd.utils.varsortability import contemporaneous_varsortability, standardize

W0 = extract_adjacency_matrix(links_a, n_vars=4, max_lag=2, format_type="tensor")[0]   # [lag, source, target]
X_a = df[["X1", "X2", "X3", "X4"]].to_numpy()[:500]   # regime A only

v = contemporaneous_varsortability(X_a, W0)                       # 0.5 uninformative, 1.0 sortable
v_std = contemporaneous_varsortability(standardize(X_a), W0)      # exactly 0.5
print("v =", round(v, 3) if v == v else "undefined (no contemporaneous edges)", " v after standardizing =", v_std)
```

`scripts/run_varsortability_audit.py` runs any registered method on raw and standardized versions of the reference benchmark and reports both, which separates a method's capability from the marginal-variance shortcut.
