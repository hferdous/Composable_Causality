import pandas as pd
import numpy as np

def build_cate_table():
    try:
        df = pd.read_csv("benchmark_results/cate_results.csv")
    except FileNotFoundError:
        print("CATE results file not found.")
        return

    df_ok = df[df["status"] == "ok"].copy()
    if df_ok.empty:
        print("No 'ok' rows in CATE results.")
        return

    metrics = [("pehe", "PEHE", "min"), ("runtime_s", "Runtime (s)", "min")]
    agg_funcs = {m[0]: ["mean", "std"] for m in metrics}
    
    grouped = df_ok.groupby(["method", "dgp_variant"]).agg(agg_funcs).reset_index()

    variants = ["cate_linear", "cate_nonlinear"]
    methods = ["TARNet", "SNet", "FlexTENet", "XLearner"]
    
    # filter methods that exist
    methods = [m for m in methods if m in df_ok["method"].unique()]
    
    out_cols = pd.MultiIndex.from_product([variants, [m[1] for m in metrics]])
    table_df = pd.DataFrame(index=methods, columns=out_cols)
    plain_df = pd.DataFrame(index=methods, columns=out_cols)

    for variant in variants:
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
                
                if m not in methods:
                    continue

                mean_val = row[(metric, "mean")]
                std_val = row[(metric, "std")]
                
                is_best = False
                # User asked for \textbf for best PEHE per column
                if metric == "pehe" and best_val is not None and not np.isnan(mean_val):
                    if abs(mean_val - best_val) < 1e-5:
                        is_best = True

                s_tex = f"{mean_val:.3f} $\\pm$ {std_val:.3f}"
                if is_best:
                    s_tex = f"\\textbf{{{s_tex}}}"
                    
                table_df.loc[m, (variant, metric_name)] = s_tex
                
                s_plain = f"{mean_val:.3f} ± {std_val:.3f}"
                if is_best: s_plain += "*"
                plain_df.loc[m, (variant, metric_name)] = s_plain

    table_df = table_df.fillna("-")
    plain_df = plain_df.fillna("-")
    
    variant_map = {
        "cate_linear": "CATE Linear",
        "cate_nonlinear": "CATE Nonlinear"
    }
    table_df = table_df.rename(columns=variant_map, level=0)
    plain_df = plain_df.rename(columns=variant_map, level=0)
    
    method_name_map = {"XLearner": "X-Learner"}
    table_df = table_df.rename(index=method_name_map)
    plain_df = plain_df.rename(index=method_name_map)

    print("=== PLAIN TEXT CATE TABLE ===")
    print(plain_df.to_string())
    print("\n")

    table_df.index = [str(x).replace("_", "\\_") for x in table_df.index]
    with open("paper/tables/cate_results.tex", "w") as f:
        f.write(table_df.to_latex(escape=False, column_format="l" + "c"*len(out_cols), multicolumn_format="c"))

if __name__ == "__main__":
    build_cate_table()
