import pandas as pd
import numpy as np

def format_mean_std(mean, std, is_best=False):
    s = f"{mean:.2f} $\\pm$ {std:.2f}"
    if is_best:
        return f"\\textbf{{{s}}}"
    return s

def build_tables(df_raw, methods_ordered, target_variants, out_file):
    df = df_raw[df_raw["dgp_variant"].isin(target_variants)].copy()
    
    metrics = [
        ("shd_norm", "SHD", "min"),
        ("rec", "TPR", "max"),
        ("fdr", "FDR", "min")
    ]

    agg_funcs = {m[0]: ["mean", "std"] for m in metrics}
    grouped = df.groupby(["method", "dgp_variant"]).agg(agg_funcs).reset_index()

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
                
                if m not in methods_ordered:
                    continue
                
                mean_val = row[(metric, "mean")]
                std_val = row[(metric, "std")]
                
                is_best = False
                if best_val is not None and not np.isnan(mean_val):
                    if abs(mean_val - best_val) < 1e-5:
                        is_best = True

                s_tex = format_mean_std(mean_val, std_val, is_best)
                s_plain = f"{mean_val:.2f} ± {std_val:.2f}"
                if is_best: s_plain += "*"
                
                table_df.loc[m, (variant, metric_name)] = s_tex
                plain_df.loc[m, (variant, metric_name)] = s_plain

    table_df = table_df.fillna("-")
    plain_df = plain_df.fillna("-")
    
    variant_map = {
        "linear": "Linear",
        "nonlinear": "Nonlinear",
        "drift": "Drift",
        "mnar": "MNAR",
        "mcar": "MCAR"
    }
    table_df = table_df.rename(columns=variant_map, level=0)
    plain_df = plain_df.rename(columns=variant_map, level=0)

    print(f"=== PLAIN TEXT: {out_file} ===")
    print(plain_df.to_string())
    print("\n")

    table_df.index = [str(x).replace("_", "\\_") for x in table_df.index]
    latex_str = table_df.to_latex(escape=False, column_format="l" + "c"*len(out_cols), multicolumn_format="c")
    with open(f"paper/tables/{out_file}", "w") as f:
        f.write(latex_str)

def build_main_table():
    try:
        df_raw = pd.read_csv("benchmark_results/results.csv")
    except FileNotFoundError:
        print("Results file not found.")
        return

    # Filter OK
    df_raw = df_raw[df_raw["status"] == "ok"].copy()
    if df_raw.empty:
        print("No 'ok' rows found.")
        return

    # Create FDR
    df_raw["fdr"] = 1.0 - df_raw["prec"]

    # Ordering by F1 on linear
    lin_f1 = df_raw[df_raw["dgp_variant"] == "linear"].groupby("method")["f1_orient"].mean().sort_values(ascending=False)
    methods_ordered_all = lin_f1.index.tolist()
    for m in df_raw["method"].unique():
        if m not in methods_ordered_all:
            methods_ordered_all.append(m)

    # Obs table
    build_tables(df_raw, methods_ordered_all, ["linear", "nonlinear"], "main_results_obs.tex")
    
    # Drift table
    build_tables(df_raw, methods_ordered_all, ["drift"], "drift_results.tex")
    
    # Miss table
    methods_miss = [m for m in methods_ordered_all if m in ["CUTS+", "VARLiNGAM_imp"]]
    build_tables(df_raw, methods_miss, ["mnar", "mcar"], "main_results_miss.tex")

if __name__ == "__main__":
    build_main_table()
