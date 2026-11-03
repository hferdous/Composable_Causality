# Composable Causality (DTCD)

Code, data, and results for **Composable Causality: A Toolkit for Systematic Time-Series Causal Discovery and Treatment-Effect Benchmarking** (Muhammad Hasan Ferdous and Md Osman Gani, NeurIPS 2026, Track on Evaluations and Datasets).

DTCD (Datasets and Toolkit for Causal Discovery) generates multivariate time series together with their ground-truth contemporaneous and lagged causal graphs. Graph topology, structural mechanism, noise distribution, observation mask, and intervention policy are independent components, so one difficulty axis can be varied while every other axis stays fixed. Paired counterfactual twins are produced by caching the noise realization during the factual rollout and re-injecting it during the interventional rollout, which gives individual treatment effects that are exact to floating-point precision.

![DTCD architecture](figures/fig_architecture.png)

The repository contains the library (`src/dtcd/`), the reference benchmark and every additional experiment reported in the paper, the raw result files behind every table and figure, and one command that rebuilds all of them.

## Installation

Python 3.10 or newer. For the reference benchmark use Python 3.12.9 or newer. Earlier 3.11 releases crashed with a segmentation fault in `tracemalloc` during a full discovery run.

```bash
git clone https://github.com/hferdous/Composable_Causality.git
cd Composable_Causality
pip install -e .                               # core library
pip install -e ".[discovery,deep,cate,dev]"    # all baselines and the tests
```

The core install (`numpy`, `pandas`, `scipy`, `networkx`, `matplotlib`, `scikit-learn`, `optuna`) is enough to generate data, run the examples, run DYNOTEARS and the varsortability controls, and rebuild every paper table and figure from the released results. The extras add the third-party baselines:

| Extra | Packages | Needed for |
| :--- | :--- | :--- |
| `discovery` | tigramite, dcor, causal-learn, lingam, statsmodels | PCMCI (ParCorr, GPDC), VARLiNGAM, CD-NOD, GrangerF |
| `deep` | torch | CUTS+, and VARLiNGAM_imp (its module also hosts CUTS+) |
| `cate` | catenets, econml, autorank | TARNet, SNet, FlexTENet, X-Learner |
| `dev` | pytest | the test suite |

Every wrapper that needs an optional package is imported behind a guard, so a missing package only matters when you request that method.

## Quick start

Generate a linear vector autoregressive series with its ground-truth graph:

```python
from dtcd.generators import LinearTimeSeriesGenerator

gen = LinearTimeSeriesGenerator(n_vars=4, max_lag=2, random_state=42)
df = gen.generate_multivariate_ts(n_points=1000)

# df             DataFrame with columns X1..X4 and 'time'
# gen.links      ground-truth edges, {(parent, -lag, child, power): weight}
# gen.last_noise cached noise realization, reused for counterfactual twins
```

Generate a factual and an interventional trajectory that share one noise realization:

```python
from dtcd.interventions.base import InterventionManager, CounterfactualSuite

im = InterventionManager()
im.add_intervention(target="X1", intervention_type="do", value=1.0, start=500)

suite = CounterfactualSuite(gen)
obs_df, int_df = suite.generate_twins(n_points=1000, intervention_manager_for_twin=im)

# The two trajectories agree exactly before t = 500. Afterwards, int_df - obs_df
# at a downstream node is that node's individual treatment effect.
```

Export a ground-truth graph for other libraries:

```python
from dtcd.utils.export import to_tigramite_links, to_causal_learn_matrix

links = to_tigramite_links(gen.links, n_vars=4, max_lag=2)
adj = to_causal_learn_matrix(gen.links, n_vars=4, max_lag=2)
```

Eight short scripts in [`examples/`](examples/) walk through the API, and [`docs/MANUAL.md`](docs/MANUAL.md) covers topologies, mechanisms, noise, interventions, missingness, metrics, and exporters in detail.

## Reproducing the paper

Every table and figure in the paper is rebuilt from the released results with one command:

```bash
bash reproduce.sh
```

This takes a few seconds and needs only the core install. The rebuilt tables in `paper/tables/` are byte-identical to the ones in the paper.

To rerun every experiment from scratch first:

```bash
bash reproduce.sh --full
```

This moves the released results to `benchmark_results_released/`, runs the reference benchmark (`run_pipeline.sh`) and every additional experiment (`scripts/run_paper_experiments.sh`), and then rebuilds the tables and figures. It needs the `discovery`, `deep`, and `cate` extras. The authors ran the reference benchmark in about 2.6 hours of wall-clock time on an Apple M4 Mac mini (24 GB RAM, no GPU, 4 worker processes); the stage notes at the top of `scripts/run_paper_experiments.sh` give approximate times for the rest. Both scripts resume where they stopped, and each stage of `run_paper_experiments.sh` can be run alone with `--stage N`.

| Paper content | Produced by | Result file(s) | Table or figure builder |
| :--- | :--- | :--- | :--- |
| Tables 1, 2 (discovery), 7, 8, 11, 12, 13 | `run_pipeline.sh` | `results.csv` | `paper/tables/build_main_table.py`, `build_precision_recall.py`, `build_appendix_tables.py` |
| Tables 3 and 9 (CATE) | `run_pipeline.sh` | `cate_results.csv` | `build_cate_table.py`, `build_cate_by_d.py` |
| Table 5, Figure 5, Section 4.3 (varsortability audit) | stages 1 and 2 | `audit_linear.csv`, `audit_linear_retuned.csv`, `audit_*.csv`, `audit_summary.csv` | `build_varsort_table.py`; `scripts/build_figures.py` |
| Tuning-protocol spot check (seed 42 versus held-out seed 41) | stage 3 | `tuning_protocol_check.csv` | summarized in `benchmark_results/README.md` |
| Tables 14 to 16 (scalability) | stages 4 and 5 | `scalability.csv`, `scal_tau{3,6,12}_{raw,std}.csv`, `scalability_annotated.csv` | `build_scalability_table.py`, `build_scal_tables.py` |
| Table 17 (TimeGraph C1) | stage 6 | `tg_full.csv`, `tg_seasonal.csv`, `tg_mech.csv` | `build_tg_c1_table.py` |
| Table 10 (DYNOTEARS under three mechanisms) | stage 7 | `nonlinear_mechanisms.csv`, `nonlinear_quad_amplitude.csv` | `build_nonlin_table.py` |
| Figures 2 to 4 | generated directly | none | `scripts/build_figures.py` |
| Figure 1 | TikZ source | none | `figures/fig_architecture_tikz.tex` (compile with pdflatex) |

Tables 4 and 6 list fixed DGP parameters and hyperparameter search spaces; they are written by hand in the paper and match `generate_datasets.py` and `src/dtcd/baselines/tuning.py`. Result files live in [`benchmark_results/`](benchmark_results/); its README documents every file and column.

### Protocol notes

- The reference grid has 9 graph templates (d in {4, 6, 8}, tau_max in {2, 3, 4}), 9 discovery DGP variants (linear, nonlinear, drift, MCAR, MNAR, trend, seasonal, trend+seasonal, nonlinear+trend+seasonal), 2 CATE variants, and evaluation seeds 42 to 51.
- Hyperparameters are tuned with Optuna (30 trials per method and cell, 5 for PCMCI-GPDC and DYNOTEARS) on a separate trajectory at seed 41, which is never evaluated. Selections are cached in `tuning_cache/` (created on first run, not shipped). The cache key includes the tuning seed, the trial budget and the search-space version, and the TPE sampler is seeded from the key, so a cell tunes to the same setting on every run. Delete `tuning_cache/` to retune from scratch.
- The tuning objective is oracle F1 against the true graph, so numbers for tuned methods are optimistic relative to what is attainable without the graph. VARLiNGAM and VARLiNGAM_imp have no tunable parameters.
- `run_discovery.py` and `run_cate.py` run tasks in separate processes (`DTCD_JOBS`, default 4), because the wrappers seed global random number generators. Each run writes `benchmark_results/environment.json` with the Python version and every installed package version. CATE estimators receive the evaluation seed.
- Structural weights are drawn once per graph from a seed derived from `md5(graph_id)`, so the same DGP is produced on any machine and Python version.
- The varsortability audit evaluates every DGP variant both as generated and after z-scoring each variable, with hyperparameters, weights, and noise held fixed. Standardizing drives varsortability (Reisach et al., 2021) to 0.5 without changing the true graph, which separates method capability from the marginal-variance shortcut.

### Notes on the result files

- `results.csv` and `cate_results.csv` come from a full regeneration with the fixed VARLiNGAM_imp wrapper (it previously indexed past the adjacency array when BIC chose a lag order below tau_max), the held-out tuning seed 41, a seeded Optuna sampler, and evaluation seeds passed to the CATE estimators. An earlier run of the benchmark had 171 VARLiNGAM_imp errors from that defect. The tuned CUTS+ setting predicts no edges on all ten seeds in 3 of 81 cells (`d4_tau4_linear`, `d4_tau4_drift`, `d6_tau3_seasonal`), which scores F1 = 0 for those cells. `varlingam_imp_correction.csv` and the `vlimp_fixed_*.csv` files document the wrapper change.
- The files behind Table 10 (`nonlinear_mechanisms.csv`, `nonlinear_quad_amplitude.csv`) are produced by the current code. Stage 7 takes about 30 seconds and reproduces byte-identical output.
- `scalability.csv` was produced by an earlier version of `scripts/run_scalability.py` that wrote raw and standardized runs to one file. Stage 4 now runs the two conditions separately and concatenates them into the same format.
- `varsortability_audit.csv` is the first, three-method version of the linear audit, from an earlier version of the audit script. Its condition means agree with `audit_linear.csv` to within 0.001 F1 for DYNOTEARS and 0.007 for SortnRegress, and its `v_summary` column used an earlier definition. Table 5 and Figure 5 are built from `audit_linear.csv`.

## Repository layout

```
.
├── src/dtcd/              the library
│   ├── structure/         graph topologies (ER, BA, WS), edge schedulers
│   ├── mechanisms/        linear, polynomial, tanh, interaction, Fourier, MLP, PNL, discrete
│   ├── noise/             Gaussian, Laplace, Student-t, exponential, uniform, Cauchy,
│   │                      heteroscedastic, drifting-scale, mixtures
│   ├── generators/        linear, nonlinear, confounded, irregular, regime-switching,
│   │                      and the universal compositional generator
│   ├── observation/       censoring, quantization, measurement noise, MNAR masks
│   ├── interventions/     InterventionManager, CounterfactualSuite
│   ├── spatial/           spatio-temporal extension
│   ├── baselines/         discovery and CATE wrappers, registry, tuning
│   ├── evaluators/        metrics and experiment runner
│   ├── benchmarks/        the nine predefined reference graphs
│   ├── utils/             exporters, varsortability, MCAR and block masks, seasonality
│   └── viz/               ground-truth and recovered-graph plots
├── generate_datasets.py   reference benchmark datasets (all variants, seeds 41 to 51)
├── run_discovery.py       discovery baselines with per-cell tuning
├── run_cate.py            CATE baselines
├── analyze_results.py     per-method summaries of the two result files
├── run_pipeline.sh        reference benchmark end to end
├── reproduce.sh           every table and figure (optionally every experiment)
├── scripts/               additional experiments, figure builder, external TimeGraph generator
├── paper/tables/          table builders and the LaTeX tables they produce
├── benchmark_results/     released results (see its README)
├── figures/               paper figures and the TikZ source of Figure 1
├── sample_dataset/        a small example trajectory
├── examples/              eight tutorial scripts
├── tests/                 unit and integration tests
├── docs/                  manual and datasheet
└── croissant.json         Croissant metadata with Responsible AI fields
```

## Baselines

| Method | Paradigm | Contemporaneous edges | Missing values | Notes |
| :--- | :--- | :--- | :--- | :--- |
| GrangerF | Granger | No | No | Linear VAR F-test |
| VARLiNGAM | Functional (LiNGAM) | Yes | No | Requires non-Gaussian noise; standardizes its input |
| VARLiNGAM_imp | Functional (LiNGAM) | Yes | Yes | Mean imputation, then VARLiNGAM on the raw scale |
| PCMCI-ParCorr | Constraint-based | No | No | Partial correlation test |
| PCMCI-GPDC | Constraint-based | No | No | GP regression and distance correlation; run at d = 4 only |
| DYNOTEARS | Score-based | Yes | No | Solver vendored from CausalNex |
| CD-NOD | Constraint-based | Yes | No | Designed for nonstationary data |
| CUTS+ | Neural | Yes | Yes | Joint imputation and discovery |
| SortnRegress | Control | Yes | No | Orders by marginal variance; not a discovery method |
| RandomRegress | Control | Yes | No | Random order; not a discovery method |

| CATE estimator | Package | Approach |
| :--- | :--- | :--- |
| TARNet | catenets | Shared representation with one head per treatment arm |
| SNet | catenets | Representation split into shared and arm-specific parts |
| FlexTENet | catenets | Flexible sharing between arm-specific and shared layers |
| X-Learner | econml | Meta-learner with imputed treatment effects and propensity weighting |

Discovery wrappers expose `fit(X, mask=None)` and CATE wrappers `fit(X, A, mask=None)`, each with a declared `supports_missing` flag. A method that does not support missing values is skipped on MCAR and MNAR cells with a structured status record instead of returning an empty graph.

The registry also contains interface stubs for TCDF, ACD, NeuralGC (cMLP and cLSTM), Rhino, UnCLe, and the Causal Transformer. No implementation of these is bundled; they return `status="error"` with an explanatory message and were not used in the paper. Section 8.4 of the manual shows how to add a working baseline.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

Tests that need an optional baseline package, or the benchmark datasets written by `python generate_datasets.py` (about a minute, 100 MB in `data/`), skip themselves when those are missing. `scripts/smoke_single_cell.py` tunes and runs every paper baseline on one cell, and `scripts/time_pcmci_gpdc.py` times one PCMCI-GPDC fit at d = 8.

## Citation

```bibtex
@inproceedings{ferdous2026composable,
  title     = {Composable Causality: A Toolkit for Systematic Time-Series Causal Discovery and Treatment-Effect Benchmarking},
  author    = {Ferdous, Muhammad Hasan and Gani, Md Osman},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Track on Evaluations and Datasets},
  year      = {2026}
}
```

`CITATION.cff` carries the same information for GitHub's citation button.

## License

- Code: [Apache License 2.0](LICENSE).
- Data and result files (`benchmark_results/`, `sample_dataset/`, and datasets generated by the scripts): [CC BY 4.0](LICENSE-CC-BY-4.0.txt).
- `src/dtcd/baselines/dynotears_core.py` is derived from CausalNex (Apache-2.0) and `scripts/external/timegraph_c1.py` from TimeGraph (MIT); both keep their original notices. See [NOTICE](NOTICE).
- Optional dependencies are installed from their own distributions under their own licenses and are not redistributed here. Note that tigramite is GPL-3.0.

## Contact

Muhammad Hasan Ferdous (h.ferdous@umbc.edu) and Md Osman Gani (mogani@umbc.edu), Causal AI Lab, Department of Information Systems, University of Maryland, Baltimore County. Please report problems through GitHub issues.

## Acknowledgments

Research reported in this publication was supported by the National Institute On Minority Health And Health Disparities of the National Institutes of Health under Award Number R01MD019814. The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health.
