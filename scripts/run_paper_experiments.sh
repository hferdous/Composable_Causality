#!/usr/bin/env bash
# Reproduce every experiment reported in the NeurIPS 2026 paper beyond the
# reference benchmark itself (which run_pipeline.sh regenerates).
#
# Each stage writes its own CSV into benchmark_results/ and skips if that CSV
# already exists, so the script can be interrupted and re-run without losing
# work. Delete a CSV to force its stage to run again.
#
#   bash scripts/run_paper_experiments.sh               # stages 0 to 8
#   bash scripts/run_paper_experiments.sh --stage 4     # one stage only
#   bash scripts/run_paper_experiments.sh --with-gpdc   # adds PCMCI-GPDC to stage 2 (slow)
#
# Stages and the paper content they produce:
#   0  datasets for the reference grid, incl. held-out tuning seed 41  (data/datasets/)
#   1  re-tuned varsortability audit, linear variant      varsortability section (re-tuning control)
#   2  varsortability audit, all nine DGP variants        varsortability section, audit table and figure
#   3  tuning-protocol check (seed 42 vs held-out 41)     hyperparameter tuning paragraph
#   4  scalability grids: d to 32, tau_max to 12; T to 10000   scalability appendix
#   5  varsortability annotation of the scalability grid  scalability appendix
#   6  TimeGraph C1 external equation family              TimeGraph C1 appendix
#   7  DYNOTEARS under three recursive mechanisms         nonlinear-mechanism appendix
#   8  rebuild paper tables and figures from the CSVs     paper/tables/, figures/
#
# Approximate wall clock on one modern CPU core: stage 0 about 5 min, stage 1
# about 15 min, stage 2 about 4 h (about 20 h with --with-gpdc), stage 3 about
# 20 min, stage 4 about 3 h, stages 5 to 8 under 30 min together.
#
# Stages 2 and 7 read the tuned hyperparameters in
# benchmark_results/results.csv, which run_pipeline.sh produces.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH:-}:src"
# The DGP weight seed is md5-derived and does not depend on PYTHONHASHSEED;
# pinning it keeps any incidental hash use deterministic as well.
export PYTHONHASHSEED=0

RESULTS_DIR="benchmark_results"
LOG_DIR="logs"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

WITH_GPDC=0
ONLY_STAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-gpdc) WITH_GPDC=1; shift ;;
    --stage) ONLY_STAGE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

METHODS=(DYNOTEARS PCMCI_ParCorr VARLiNGAM VARLiNGAM_imp GrangerF CDNOD "CUTS+" SortnRegress RandomRegress)
VARIANTS=(linear nonlinear drift mnar mcar trend seasonal trend_seasonal nonlinear_trend_seasonal)

run_stage() { [[ -z "$ONLY_STAGE" || "$ONLY_STAGE" == "$1" ]]; }
banner() { printf '\n=== %s ===\n' "$1"; }

# Stage 0: datasets. Stages 1 to 3 read these from disk.
if run_stage 0; then
  if [[ ! -f "data/datasets/d4_tau2_linear_s41_X.npy" ]]; then
    banner "Stage 0: generating benchmark datasets (includes held-out seed 41)"
    python generate_datasets.py 2>&1 | tee "$LOG_DIR/stage0_generate.log"
  else
    echo "Stage 0: datasets already present, skipping"
  fi
fi

# Stage 1: does DYNOTEARS recover if hyperparameters are re-tuned on
# standardized data? Answered first because it is the strongest objection to
# the varsortability finding.
if run_stage 1; then
  OUT="$RESULTS_DIR/audit_linear_retuned.csv"
  if [[ -f "$OUT" ]]; then
    echo "Stage 1: $OUT exists, skipping"
  else
    banner "Stage 1: re-tuned audit on the linear variant"
    python scripts/run_varsortability_audit.py \
      --variant linear --retune --tuning-seed 41 --tuning-trials 30 \
      --methods DYNOTEARS SortnRegress RandomRegress \
      --out "$OUT" 2>&1 | tee "$LOG_DIR/stage1_retune.log"
  fi
fi

# Stage 2: raw versus standardized across all nine variants, hyperparameters
# held at the benchmark values (the controlled contrast).
if run_stage 2; then
  M=("${METHODS[@]}")
  if [[ "$WITH_GPDC" == "1" ]]; then M+=(PCMCI_GPDC); fi
  banner "Stage 2: varsortability audit across ${#VARIANTS[@]} DGP variants"
  for v in "${VARIANTS[@]}"; do
    OUT="$RESULTS_DIR/audit_${v}.csv"
    if [[ -f "$OUT" ]]; then
      echo "  $v: already done, skipping"
      continue
    fi
    echo "  --- variant: $v ---"
    python scripts/run_varsortability_audit.py \
      --variant "$v" --methods "${M[@]}" \
      --out "$OUT" 2>&1 | tee "$LOG_DIR/stage2_${v}.log"
  done
  python scripts/summarize_audit.py --results-dir "$RESULTS_DIR" \
    --out "$RESULTS_DIR/audit_summary.csv" 2>&1 | tee "$LOG_DIR/stage2_summary.log"
fi

# Stage 3: how much did tuning on an evaluated seed inflate results?
if run_stage 3; then
  OUT="$RESULTS_DIR/tuning_protocol_check.csv"
  if [[ -f "$OUT" ]]; then
    echo "Stage 3: $OUT exists, skipping"
  else
    banner "Stage 3: tuning-protocol check (seed 42 vs held-out seed 41)"
    python scripts/check_tuning_protocol.py --out "$OUT" \
      2>&1 | tee "$LOG_DIR/stage3_tuning.log"
  fi
fi

# Stage 4: scalability. All nine methods at library defaults, d up to 32,
# T in {1000, 5000}, tau_max in {3, 6, 12}, raw and standardized, three seeds.
if run_stage 4; then
  banner "Stage 4: scalability grid"
  for tau in 3 6 12; do
    for cond in raw std; do
      OUT="$RESULTS_DIR/scal_tau${tau}_${cond}.csv"
      if [[ -f "$OUT" ]]; then echo "  $OUT exists, skipping"; continue; fi
      FLAG=""; [[ "$cond" == "std" ]] && FLAG="--standardized"
      python scripts/run_scalability.py --dims 4 8 16 32 --lengths 1000 5000 \
        --tau "$tau" --seeds 42 43 44 --timeout 900 $FLAG \
        --out "$OUT" 2>&1 | tee "$LOG_DIR/stage4_tau${tau}_${cond}.log"
    done
  done
  # Long-trajectory sweep at tau_max = 3 (T up to 10000) for DYNOTEARS and
  # the SortnRegress control, raw and standardized in one file.
  OUT="$RESULTS_DIR/scalability.csv"
  if [[ -f "$OUT" ]]; then
    echo "  $OUT exists, skipping"
  else
    for cond in raw std; do
      FLAG=""; [[ "$cond" == "std" ]] && FLAG="--standardized"
      python scripts/run_scalability.py --methods DYNOTEARS SortnRegress \
        --dims 4 8 16 32 --lengths 1000 5000 10000 --tau 3 --seeds 42 43 44 \
        $FLAG --out "$RESULTS_DIR/scalability_long_${cond}.csv" \
        2>&1 | tee "$LOG_DIR/stage4_long_${cond}.log"
    done
    python -c "import pandas as pd, sys; pd.concat([pd.read_csv(f) for f in sys.argv[2:]]).to_csv(sys.argv[1], index=False)" \
      "$OUT" "$RESULTS_DIR/scalability_long_raw.csv" "$RESULTS_DIR/scalability_long_std.csv"
  fi
fi

# Stage 5: per-cell varsortability of the scalability grid.
if run_stage 5; then
  OUT="$RESULTS_DIR/scalability_annotated.csv"
  if [[ -f "$OUT" ]]; then
    echo "Stage 5: $OUT exists, skipping"
  else
    banner "Stage 5: varsortability annotation of the scalability grid"
    python scripts/annotate_scalability_varsortability.py --results-dir "$RESULTS_DIR" \
      --out "$OUT" 2>&1 | tee "$LOG_DIR/stage5_annotate.log"
  fi
fi

# Stage 6: an independently authored equation family (TimeGraph C1).
if run_stage 6; then
  banner "Stage 6: TimeGraph C1 external family"
  for v in tg_full tg_seasonal tg_mech; do
    OUT="$RESULTS_DIR/${v}.csv"
    if [[ -f "$OUT" ]]; then echo "  $OUT exists, skipping"; continue; fi
    EXTRA=""; [[ "$v" == "tg_mech" ]] && EXTRA="--no-tune"
    python scripts/run_timegraph_c1.py --variant "$v" $EXTRA \
      --out "$OUT" 2>&1 | tee "$LOG_DIR/stage6_${v}.log"
  done
fi

# Stage 7: DYNOTEARS with the mechanism varied and everything else fixed.
if run_stage 7; then
  OUT="$RESULTS_DIR/nonlinear_mechanisms.csv"
  if [[ -f "$OUT" ]]; then
    echo "Stage 7: $OUT exists, skipping"
  else
    banner "Stage 7: DYNOTEARS under three recursive mechanisms"
    python scripts/run_nonlinear_mechanisms.py --out-dir "$RESULTS_DIR" \
      2>&1 | tee "$LOG_DIR/stage7_mechanisms.log"
  fi
fi

# Stage 8: rebuild every paper table and figure from the CSVs above.
if run_stage 8; then
  banner "Stage 8: tables and figures"
  python paper/tables/build_main_table.py
  python paper/tables/build_cate_table.py
  python paper/tables/build_cate_by_d.py
  python paper/tables/build_precision_recall.py
  python paper/tables/build_appendix_tables.py
  python paper/tables/build_varsort_table.py "$RESULTS_DIR/audit_linear.csv"
  python paper/tables/build_scalability_table.py "$RESULTS_DIR/scalability.csv"
  python paper/tables/build_scal_tables.py "$RESULTS_DIR"
  python paper/tables/build_tg_c1_table.py "$RESULTS_DIR"
  python paper/tables/build_nonlin_table.py "$RESULTS_DIR/nonlinear_mechanisms.csv"
  python scripts/build_figures.py
fi

banner "COMPLETE"
echo "Results in $RESULTS_DIR/, logs in $LOG_DIR/"
