import pandas as pd
import numpy as np

def format_mean_std(mean, std, is_best=False):
    s = f"{mean:.2f} $\\pm$ {std:.2f}"
    if is_best:
        return f"\\textbf{{{s}}}"
    return s

def build_appendix_tables():
    try:
        df_raw = pd.read_csv("benchmark_results/results.csv")
    except FileNotFoundError:
        print("Results file not found.")
        return

    # --- (a) Per-DGP-variant tables for trend, seasonal, trend_seasonal, nonlinear_trend_seasonal ---
    df_ok = df_raw[df_raw["status"] == "ok"].copy()
    if not df_ok.empty:
        df_ok["fdr"] = 1.0 - df_ok["prec"]
        target_variants = ["trend", "seasonal", "trend_seasonal", "nonlinear_trend_seasonal"]
        df_a = df_ok[df_ok["dgp_variant"].isin(target_variants)]

        metrics = [
            ("shd_norm", "SHD", "min"),
            ("rec", "TPR", "max"),
            ("fdr", "FDR", "min"),
            ("f1_orient", "F1", "max")
        ]

        agg_funcs = {m[0]: ["mean", "std"] for m in metrics}
        grouped = df_a.groupby(["method", "dgp_variant"]).agg(agg_funcs).reset_index()

        methods_ordered = df_a["method"].unique().tolist()
        out_cols = pd.MultiIndex.from_product([target_variants, [m[1] for m in metrics]])
        table_df = pd.DataFrame(index=methods_ordered, columns=out_cols)
        plain_df = pd.DataFrame(index=methods_ordered, columns=out_cols)

        for variant in target_variants:
            var_data = grouped[grouped["dgp_variant"] == variant]
            
            for metric, metric_name, opt_dir in metrics:
                if not var_data.empty:
                    means = var_data[(metric, "mean")]
                    best_val = means.max() if opt_dir == "max" else means.min()
                else:
                    best_val = None

                for _, row in var_data.iterrows():
                    m = row["method"].values[0] if isinstance(row["method"], pd.Series) else row["method"]
                    if type(m) != str: m = m[0]
                    
                    mean_val = row[(metric, "mean")]
                    std_val = row[(metric, "std")]
                    
                    is_best = False
                    if best_val is not None and not np.isnan(mean_val):
                        if abs(mean_val - best_val) < 1e-5:
                            is_best = True

                    table_df.loc[m, (variant, metric_name)] = format_mean_std(mean_val, std_val, is_best)
                    s_plain = f"{mean_val:.2f} ± {std_val:.2f}"
                    if is_best: s_plain += "*"
                    plain_df.loc[m, (variant, metric_name)] = s_plain

        table_df = table_df.fillna("-")
        plain_df = plain_df.fillna("-")
        
        variant_map = {
            "trend": "Trend",
            "seasonal": "Seasonal",
            "trend_seasonal": "Trend+Seasonal",
            "nonlinear_trend_seasonal": "Nonlinear+Trend+Seasonal"
        }
        table_df = table_df.rename(columns=variant_map, level=0)
        
        print("=== PLAIN TEXT APPENDIX TABLE (A) ===")
        print(plain_df.to_string())
        print("\n")
        
        table_df.index = [str(x).replace("_", "\\_") for x in table_df.index]
        with open("paper/tables/appendix_a.tex", "w") as f:
            f.write(table_df.to_latex(escape=False, column_format="l" + "c"*len(out_cols), multicolumn_format="c"))

    # --- (b1) Failure-modes summary table ---
    status_counts_full = df_raw.groupby(["method", "status"]).size().unstack(fill_value=0)
    for col in ["ok", "error", "skipped_compute_budget", "skipped_no_missing_support"]:
        if col not in status_counts_full.columns:
            status_counts_full[col] = 0
    status_counts_full = status_counts_full[["ok", "skipped_no_missing_support", "skipped_compute_budget", "error"]]
    
    col_map_summary = {
        "ok": "Completed",
        "skipped_no_missing_support": "Skipped (no missing support)",
        "skipped_compute_budget": "Skipped (compute budget)",
        "error": "Errored"
    }
    status_counts_full = status_counts_full.rename(columns=col_map_summary)
    status_counts_full.index.name = "Method"
    status_counts_full = status_counts_full.sort_index()
    
    print("=== PLAIN TEXT: failure_modes_summary.tex ===")
    print(status_counts_full.to_string())
    print("\n")
    
    status_counts_full.index = [str(x).replace("_", "\\_") for x in status_counts_full.index]
    status_counts_full.columns = [str(x).replace("_", "\\_") for x in status_counts_full.columns]
    
    with open("paper/tables/failure_modes_summary.tex", "w") as f:
        f.write(status_counts_full.to_latex(escape=False, column_format="l" + "r"*len(status_counts_full.columns)))

    # --- (b2) VARLiNGAM errors by variant ---
    var_df = df_raw[df_raw["method"] == "VARLiNGAM_imp"]
    var_status = var_df.groupby(["dgp_variant", "status"]).size().unstack(fill_value=0)
    for col in ["ok", "error"]:
        if col not in var_status.columns:
            var_status[col] = 0
            
    var_status["Total attempted"] = 90
    var_status = var_status[["ok", "error", "Total attempted"]]
    var_status = var_status.rename(columns={"ok": "Completed", "error": "Errored"})
    
    variant_order = [
        "linear", "nonlinear", "drift", "trend", "seasonal", 
        "trend_seasonal", "nonlinear_trend_seasonal", "mcar", "mnar"
    ]
    
    var_status = var_status.reindex(variant_order, fill_value=0)
    
    variant_map = {
        "linear": "Linear",
        "nonlinear": "Nonlinear",
        "drift": "Drift",
        "mnar": "MNAR",
        "mcar": "MCAR",
        "trend": "Trend",
        "seasonal": "Seasonal",
        "trend_seasonal": "Trend+Seasonal",
        "nonlinear_trend_seasonal": "Nonlinear+Trend+Seasonal"
    }
    var_status.index = [variant_map.get(x, x) for x in var_status.index]
    var_status.index.name = "DGP Variant"
    
    print("=== PLAIN TEXT: varlingam_errors_by_variant.tex ===")
    print(var_status.to_string())
    print("\n")
    
    var_status.index = [str(x).replace("_", "\\_") for x in var_status.index]
    with open("paper/tables/varlingam_errors_by_variant.tex", "w") as f:
        f.write(var_status.to_latex(escape=False, column_format="lrrr"))

    # --- (c) Scaling table: SHD vs d (4, 6, 8) for each method on linear DGP ---
    if not df_ok.empty:
        df_lin = df_ok[df_ok["dgp_variant"] == "linear"]
        scaling_grouped = df_lin.groupby(["method", "d"])[["shd_norm", "runtime_s"]].mean().unstack()
        
        print("=== PLAIN TEXT APPENDIX TABLE (C) SCALING (linear) ===")
        print(scaling_grouped.to_string())
        print("\n")
        
        scaling_grouped.index = [str(x).replace("_", "\\_") for x in scaling_grouped.index]
        scaling_grouped = scaling_grouped.rename(columns={"shd_norm": "SHD", "runtime_s": "Runtime (s)"}, level=0)
        
        # Format strings manually to handle NaN easily
        scaling_grouped = scaling_grouped.map(lambda x: "-" if pd.isna(x) else f"{x:.2f}")
        
        with open("paper/tables/appendix_c.tex", "w") as f:
            f.write(scaling_grouped.to_latex(escape=False, multicolumn_format="c"))

if __name__ == "__main__":
    build_appendix_tables()
