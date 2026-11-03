#!/usr/bin/env bash
# One command for every table and figure in the paper.
#
#   bash reproduce.sh          rebuild all paper tables (paper/tables/) and
#                              figures (figures/) from the released results in
#                              benchmark_results/. Takes a few minutes and
#                              needs only the core install.
#
#   bash reproduce.sh --full   rerun every experiment first, then rebuild.
#                              The released results are moved aside to
#                              benchmark_results_released/ so that nothing is
#                              skipped. Needs the [discovery,deep,cate] extras
#                              and many hours of CPU time; PCMCI-GPDC and CUTS+
#                              dominate. Interrupted runs resume where they
#                              stopped.
#
# Figure 1 is TikZ source (figures/fig_architecture_tikz.tex) and is compiled
# with the paper, not by this script.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:-}:src"
export PYTHONHASHSEED=0

if [[ "${1:-}" == "--full" ]]; then
  if [[ -d benchmark_results && ! -d benchmark_results_released ]]; then
    mv benchmark_results benchmark_results_released
    echo "released results moved to benchmark_results_released/"
  fi
  mkdir -p benchmark_results
  bash run_pipeline.sh
  bash scripts/run_paper_experiments.sh
else
  bash scripts/run_paper_experiments.sh --stage 8
fi
