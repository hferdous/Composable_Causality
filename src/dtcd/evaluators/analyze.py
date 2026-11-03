from __future__ import annotations
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from autorank import autorank, plot_stats
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ResultAnalyzer:
    def __init__(self, results_df: pd.DataFrame):
        self.df = results_df

    def compute_aggregates(self, groupby_cols: List[str]) -> pd.DataFrame:
        metrics = [c for c in self.df.columns if c.startswith(("shd", "f1", "prec", "rec", "runtime", "mem", "auprc"))]
        agg = self.df.groupby(groupby_cols)[metrics].agg(["mean", "std"]).reset_index()
        return agg

    def perform_stat_tests(self, metric: str, method_a: str, method_b: str) -> pd.DataFrame:
        results = []
        # Structural metadata that defines a 'cell'
        grid_cols = ['track', 'd', 'T', 'pnl_g_function']
        # max_lag might be in some versions
        if 'max_lag' in self.df.columns: grid_cols.append('max_lag')
        
        for name, group in self.df.groupby(grid_cols):
            # Filtering for methods
            sub_a = group[group['method'] == method_a].sort_values('seed')
            sub_b = group[group['method'] == method_b].sort_values('seed')
            
            # Intersection of seeds to ensure paired test
            common_seeds = set(sub_a['seed']) & set(sub_b['seed'])
            if len(common_seeds) < 2:
                continue
            
            a_scores = sub_a[sub_a['seed'].isin(common_seeds)][metric].values
            b_scores = sub_b[sub_b['seed'].isin(common_seeds)][metric].values
            
            if np.array_equal(a_scores, b_scores):
                p = 1.0
                stat = 0.0
            else:
                try:
                    stat, p = wilcoxon(a_scores, b_scores)
                except Exception:
                    p = np.nan
                    stat = np.nan
            
            results.append({
                'cell': str(name),
                'method_a': method_a,
                'method_b': method_b,
                'n_seeds': len(common_seeds),
                'statistic': stat,
                'p_val': p
            })
        
        test_df = pd.DataFrame(results)
        if test_df.empty:
            return pd.DataFrame(columns=['cell', 'method_a', 'method_b', 'n_seeds', 'statistic', 'p_val', 'p_adj', 'significant'])
            
        valid_mask = test_df['p_val'].notna()
        if valid_mask.any():
            _, p_adj, _, _ = multipletests(test_df.loc[valid_mask, 'p_val'], method='holm')
            test_df.loc[valid_mask, 'p_adj'] = p_adj
            test_df['significant'] = test_df['p_adj'] < 0.05
        else:
            test_df['p_adj'] = np.nan
            test_df['significant'] = False
        return test_df

    def generate_cd_diagram(self, metric: str, output_path: str):
        # We need a pivot table where methods are columns
        # We use 'track' + 'seed' as the index to provide multiple samples per method
        try:
            pivot_df = self.df.pivot_table(index=['track', 'seed'], columns='method', values=metric)
            if pivot_df.shape[1] < 2:
                logger.warning("Not enough methods for CD diagram.")
                return
            res = autorank(pivot_df, alpha=0.05, verbose=False)
            plot_stats(res)
            plt.title(f"CD Diagram ({metric})")
            plt.savefig(output_path)
            plt.close()
        except Exception as e:
            logger.error(f"CD Diagram failed: {e}")

    def plot_scaling(self, output_path: str):
        if 'd' not in self.df.columns: return
        plt.figure(figsize=(8, 6))
        for m in self.df['method'].unique():
            sub = self.df[self.df['method'] == m]
            avg = sub.groupby('d')['runtime_s'].mean()
            if not avg.empty:
                plt.loglog(avg.index, avg.values, marker='o', label=m)
        plt.xlabel("d"); plt.ylabel("Runtime (s)"); plt.legend(); plt.grid(True)
        plt.savefig(output_path); plt.close()

    def to_latex_table(self, agg_df: pd.DataFrame, experiment_name: str) -> str:
        latex = f"\\begin{{table}}[h]\n\\caption{{Results: {experiment_name}}}\n\\centering\n"
        latex += "\\begin{tabular}{lcccc}\n\\toprule\n"
        latex += "Method & SHD (Norm) $\\downarrow$ & F1 (Orient) $\\uparrow$ & AUPRC $\\uparrow$ & Runtime (s) \\\\\n\\midrule\n"
        for _, row in agg_df.iterrows():
            m = row["method"]
            shd_v = row[('shd_norm', 'mean')]; shd_s = row[('shd_norm', 'std')]
            f1_v = row[('f1_orient', 'mean')]; f1_s = row[('f1_orient', 'std')]
            aup_v = row[('auprc', 'mean')]; aup_s = row[('auprc', 'std')]
            rt_v = row[('runtime_s', 'mean')]
            latex += f"{m} & {shd_v:.2f} \\pm {shd_s:.2f} & {f1_v:.2f} \\pm {f1_s:.2f} & {aup_v:.2f} \\pm {aup_s:.2f} & {rt_v:.2f} \\\\\n"
        latex += "\\bottomrule\n\\end{tabular}\n\\end{table}"
        return latex
