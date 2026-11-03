# Released results

Every number, table, and figure in the paper is computed from the CSV files in this directory. `bash reproduce.sh` (repository root) rebuilds all paper tables and figures from them. The files are released under CC BY 4.0.

Conventions used throughout:

- **Graphs.** The nine reference templates `d{4,6,8}_tau{2,3,4}` (`src/dtcd/benchmarks/predefined_graphs.py`).
- **Seeds.** Evaluation seeds 42 to 51. Seed 41 is the held-out tuning trajectory and never appears as an evaluation seed.
- **Conditions.** `raw` is the data as generated; `standardized` (or `std`) z-scores each variable after generation, which drives contemporaneous varsortability `v` to 0.5 without changing the true graph.
- **F1.** Oriented F1 on the unrolled window graph (contemporaneous plus lagged edges), as in the paper. SHD is computed on the same graph.
- **Status.** `ok`, `error`, `timeout`, `skipped_no_missing_support` (the method declares `supports_missing=False` and the cell has missing values), or `skipped_compute_budget` (PCMCI-GPDC at d > 4).

## Reference benchmark

Produced by `run_pipeline.sh`.

| File | Rows | Content |
| :--- | ---: | :--- |
| `results.csv` | 6,480 | Causal discovery: 8 methods x 9 graphs x 9 DGP variants x 10 seeds. Columns: `method`, `seed`, `status`, `runtime_s`, `mem_mb`, `shd_joint`, `shd_norm`, `n_true_edges`, `n_pred_edges`, `f1_skel`, `f1_orient`, `prec`, `rec`, `auprc`, `hp` (selected hyperparameters as JSON), `error_msg`, `dataset_id`, `d`, `tau`, `dgp_variant`, `hp_source`. Paper Tables 1, 2, 7, 8, 11, 12, 13. |
| `cate_results.csv` | 720 | CATE estimation: 4 estimators x 9 graphs x 2 variants x 10 seeds. Columns: `method`, `dataset_id`, `seed`, `pehe`, `ate_abs_error`, `runtime_s`, `mem_mb`, `status`, `d`, `tau`, `dgp_variant`. Paper Tables 3 and 9. |

Both files come from a full regeneration with the current code: hyperparameters are tuned on the held-out seed 41 with a seeded Optuna sampler, the VARLiNGAM_imp wrapper is fixed, and the CATE estimators receive the evaluation seed. All 6,480 discovery rows were attempted: 4,980 completed, 960 were skipped because the method does not support missing inputs, 540 were skipped by the PCMCI-GPDC compute-budget guard, and none raised an exception. All 720 CATE rows completed. The regeneration ran under Python 3.12.13 (`environment.json`, `environment_pip_freeze.txt`) with 4 worker processes and took about 2.6 hours. Rerunning selected cells in separate processes reproduced identical rows. It has not been tested on other machines. An earlier run, before the wrapper fix and the held-out tuning protocol, had 171 VARLiNGAM_imp errors; the two files below quantify those changes.

## Varsortability audit

Produced by stages 1 to 3 of `scripts/run_paper_experiments.sh`. Each audit file has 180 rows (9 graphs x 10 seeds x 2 conditions) with hyperparameters, weights, and noise held fixed across conditions.

| File | Content |
| :--- | :--- |
| `audit_<variant>.csv` (9 files) | One per DGP variant. Columns: `graph`, `d`, `tau_max`, `variant`, `seed`, `condition`, `v_contemporaneous`, `v_summary`, and `f1_<method>` for DYNOTEARS, PCMCI_ParCorr, VARLiNGAM, VARLiNGAM_imp, GrangerF, CDNOD, CUTS+, SortnRegress, RandomRegress. Hyperparameters come from `results.csv`. The `f1_VARLiNGAM_imp` column holds the rerun with the fixed wrapper (from `vlimp_fixed_<variant>.csv`). `audit_linear.csv` is the source of Table 5 and Figure 5. |
| `audit_linear_retuned.csv` | Linear variant with DYNOTEARS and both controls retuned separately for each condition on seed 41 (30 Optuna trials). Retuning raises standardized DYNOTEARS from 0.815 to 0.822, so stale hyperparameters do not explain the drop. |
| `audit_summary.csv` | Mean F1 per variant and condition for every method (`scripts/summarize_audit.py`). Source of the numbers in Section 4.3, for example DYNOTEARS 0.988 raw and 0.815 standardized on the linear variant, and 0.788 and 0.461 on the seasonal variant. |
| `vlimp_fixed_<variant>.csv` (9 files) | VARLiNGAM_imp alone, rerun with the fixed wrapper; merged into the audit files above. |
| `varlingam_imp_correction.csv` | Per variant: `old` (mean F1 with the defective wrapper), `new_raw` and `new_std` (fixed wrapper), `nans` (failed cells), `inv_delta` (`new_raw` minus `new_std`), `delta` (`new_raw` minus `old`). On the nonlinear variant the corrected mean is 0.758 against 0.849 before. |
| `tuning_protocol_check.csv` | DYNOTEARS, PCMCI_ParCorr, and VARLiNGAM on three cells (d4_tau2, d6_tau3, d8_tau4, linear), tuned on seed 42 (`old_seed42_in_sample`) and on held-out seed 41 (`new_seed41_held_out`), evaluated on seeds 42 to 51. Moving tuning to the held-out seed changes the mean F1 of a cell by at most 0.014 (DYNOTEARS on d6_tau3, upward); PCMCI_ParCorr and VARLiNGAM are unchanged. |
| `varsortability_audit.csv` | The first, three-method version of the linear audit (DYNOTEARS, SortnRegress, RandomRegress), from an earlier version of `scripts/run_varsortability_audit.py`. Its condition means agree with `audit_linear.csv` to within 0.001 F1 for DYNOTEARS and 0.007 for SortnRegress; its `v_summary` used an earlier definition. Kept for provenance; no table or figure reads it. |

Rerunning the audit reproduces these files exactly for every method except the SortnRegress control on standardized data. After standardizing, all marginal variances are equal up to floating-point rounding, so the order SortnRegress sorts by is decided by rounding noise and can change with the platform and the numerical libraries. In a check of 36 cells (linear variant, seeds 42 and 43, six methods) on Linux with Python 3.12, 34 matched exactly and 2 standardized SortnRegress cells differed. This is the uninformative regime the control is meant to show, so the conclusion does not depend on it.

## Scalability

Produced by stages 4 and 5. All methods run at library defaults with a per-fit time budget.

| File | Rows | Content |
| :--- | ---: | :--- |
| `scal_tau{3,6,12}_{raw,std}.csv` | 24 each | d in {4, 8, 16, 32} x T in {1000, 5000} x seeds 42 to 44, for all nine methods: `f1_<method>`, `runtime_<method>`, `status_<method>`. Paper Tables 15 and 16. |
| `scalability_annotated.csv` | 144 | The six files above, concatenated, with per-cell `v_contemporaneous`, `v_summary`, `n_contemp_edges`, `n_lagged_edges`, `contemp_fraction`. Source of the varsortability dose-response in the scalability appendix. |
| `scalability.csv` | 70 | Long trajectories at `tau_max` = 3: d in {4, 8, 16, 32} x T in {1000, 5000, 10000} x seeds 42 to 44, raw and standardized (one standardized seed at d = 32, T = 10000), for DYNOTEARS and SortnRegress. Written by an earlier version of `scripts/run_scalability.py`; stage 4 now regenerates it in the same format. Paper Table 14. |

## External equation family (TimeGraph C1)

Produced by stage 6 with `scripts/run_timegraph_c1.py`, which imports the published TimeGraph C1 generator from `scripts/external/timegraph_c1.py`.

| File | Rows | Content |
| :--- | ---: | :--- |
| `tg_full.csv`, `tg_seasonal.csv`, `tg_mech.csv` | 36 each | d in {4, 6, 8} x max lag in {2, 3, 4} x 2 conditions (`cond`) x seeds 42 and 43, for DYNOTEARS, SortnRegress, RandomRegress, VARLiNGAM, GrangerF, PCMCI_ParCorr: `f1_<method>`, `status_<method>`, `rt_<method>`, plus `n_true_edges` and `v_contemporaneous`. `tg_full` keeps TimeGraph's trend and seasonality, `tg_seasonal` removes the trend, `tg_mech` removes both. DYNOTEARS is tuned on seed 41 in `tg_full` and `tg_seasonal` and runs at defaults in `tg_mech`. Paper Table 17. |

## DYNOTEARS under three recursive mechanisms

Produced by stage 7 with `scripts/run_nonlinear_mechanisms.py`.

| File | Rows | Content |
| :--- | ---: | :--- |
| `nonlinear_mechanisms.csv` | 270 | 9 graphs x 3 mechanisms (`linear`, `nonlinear` = sign(x)\|x\|^0.5, `nonlinear_sin` = sin x) x 10 seeds, with DYNOTEARS hyperparameters fixed at each graph's Nonlinear-cell values from `results.csv`: `shd`, `f1_oriented`, `status`, `hp`. Paper Table 10. |
| `nonlinear_quad_amplitude.csv` | 90 | Peak absolute value (`max_abs_x`) of trajectories under the amplifying mechanism x\|x\| for every graph and seed, which supports the statement that this mechanism explodes under recursion. |
