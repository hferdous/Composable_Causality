#!/usr/bin/env bash
# Reference benchmark: datasets, causal discovery, CATE estimation, summaries,
# and the paper tables built from them (Tables 1 to 3 and 7 to 13).
#
# Discovery and CATE runs resume from benchmark_results/results.csv and
# benchmark_results/cate_results.csv, skipping cells already present. To
# rerun from scratch, move the released CSVs out of benchmark_results/ first
# (bash reproduce.sh --full does this for you).
#
# Runtime: dataset generation takes minutes; discovery with per-cell tuning
# takes many hours, dominated by PCMCI-GPDC and CUTS+.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:-}:src"
export PYTHONHASHSEED=0

echo "--- DTCD reference benchmark ---"
echo "Step 0: datasets (evaluation seeds 42 to 51, held-out tuning seed 41)"
python generate_datasets.py

echo "Step 1: causal discovery"
python run_discovery.py

echo "Step 2: CATE estimation"
python run_cate.py

echo "Step 3: summaries"
python analyze_results.py

echo "Step 4: paper tables"
python paper/tables/build_main_table.py
python paper/tables/build_cate_table.py
python paper/tables/build_cate_by_d.py
python paper/tables/build_precision_recall.py
python paper/tables/build_appendix_tables.py

echo "--- done: results in benchmark_results/, tables in paper/tables/ ---"
