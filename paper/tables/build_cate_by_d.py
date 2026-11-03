import pandas as pd
import numpy as np
import re

def format_mean_std(mean, std, is_best=False):
    s = f"{mean:.3f} $\\pm$ {std:.3f}"
    if is_best:
        return f"\\textbf{{{s}}}"
    return s

def extract_d(dataset_id):
    # parse d from dataset_id like 'd4_tau2_cate_linear_seed...'
    match = re.search(r'd(\d+)', str(dataset_id))
    if match:
        return int(match.group(1))
    return None

def build_cate_by_d():
    try:
        df = pd.read_csv("benchmark_results/cate_results.csv")
    except FileNotFoundError:
        print("CATE results file not found.")
        return

    df_ok = df[(df["status"] == "ok") & (df["dgp_variant"] == "cate_linear")].copy()
    if df_ok.empty:
        print("No 'ok' rows in CATE results for cate_linear.")
        return

    df_ok["d"] = df_ok["dataset_id"].apply(extract_d)
    
    # group by method and d
    grouped = df_ok.groupby(["method", "d"])["pehe"].agg(["mean", "std"]).reset_index()

    methods_order = ["TARNet", "SNet", "FlexTENet", "XLearner"]
    d_values = [4, 6, 8]
    
    out_cols = pd.MultiIndex.from_product([["PEHE"], [f"d={d}" for d in d_values]])
    
    # ensure methods exist in the data and keep desired order
    methods_internal = [m for m in methods_order if m in grouped["method"].unique()]
    # add any extra methods not in the requested list just in case
    for m in grouped["method"].unique():
        if m not in methods_internal:
            methods_internal.append(m)
            
    table_df = pd.DataFrame(index=methods_internal, columns=out_cols)
    plain_df = pd.DataFrame(index=methods_internal, columns=out_cols)

    for d in d_values:
        d_data = grouped[grouped["d"] == d]
        
        best_val = None
        if not d_data.empty:
            best_val = d_data["mean"].min()

        for m in methods_internal:
            row = d_data[d_data["method"] == m]
            if not row.empty:
                mean_val = row["mean"].values[0]
                std_val = row["std"].values[0]
                
                is_best = False
                if best_val is not None and not np.isnan(mean_val):
                    if abs(mean_val - best_val) < 1e-5:
                        is_best = True

                s_tex = format_mean_std(mean_val, std_val, is_best)
                s_plain = f"{mean_val:.3f} ± {std_val:.3f}"
                if is_best: s_plain += "*"
                
                table_df.loc[m, ("PEHE", f"d={d}")] = s_tex
                plain_df.loc[m, ("PEHE", f"d={d}")] = s_plain

    table_df = table_df.fillna("-")
    plain_df = plain_df.fillna("-")
    
    # Rename XLearner to X-Learner in the final table index
    method_name_map = {"XLearner": "X-Learner"}
    table_df = table_df.rename(index=method_name_map)
    plain_df = plain_df.rename(index=method_name_map)

    # Escape underscores in index for safety
    table_df.index = [str(x).replace("_", "\\_") for x in table_df.index]

    print("=== PLAIN TEXT CATE BY D TABLE ===")
    print(plain_df.to_string())
    print("\n")

    with open("paper/tables/cate_by_d.tex", "w") as f:
        f.write(table_df.to_latex(escape=False, column_format="l" + "c"*len(out_cols), multicolumn_format="c"))

if __name__ == "__main__":
    build_cate_by_d()
