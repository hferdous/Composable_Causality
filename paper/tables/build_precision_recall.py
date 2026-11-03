import pandas as pd
import numpy as np

def format_val(val, is_best=False):
    s = f"{val:.2f}"
    if is_best:
        return f"\\textbf{{{s}}}"
    return s

def build_pr_table():
    try:
        df = pd.read_csv("benchmark_results/results.csv")
    except FileNotFoundError:
        print("Results file not found.")
        return

    df_ok = df[(df["status"] == "ok") & (df["dgp_variant"] == "linear")].copy()
    if df_ok.empty:
        print("No 'ok' rows for linear variant.")
        return

    # Group by method and calculate means
    grouped = df_ok.groupby("method")[["prec", "rec", "f1_orient"]].mean()
    
    # Sort by F1 descending
    grouped = grouped.sort_values("f1_orient", ascending=False)
    
    metrics = [("prec", "Precision"), ("rec", "Recall"), ("f1_orient", "F1")]
    
    methods = grouped.index.tolist()
    out_cols = [m[1] for m in metrics]
    
    table_df = pd.DataFrame(index=methods, columns=out_cols)
    plain_df = pd.DataFrame(index=methods, columns=out_cols)
    
    for metric_col, metric_name in metrics:
        best_val = grouped[metric_col].max()
        for m in methods:
            mean_val = grouped.loc[m, metric_col]
            
            is_best = False
            if not np.isnan(mean_val) and abs(mean_val - best_val) < 1e-5:
                is_best = True
                
            table_df.loc[m, metric_name] = format_val(mean_val, is_best)
            
            s_plain = f"{mean_val:.2f}"
            if is_best: s_plain += "*"
            plain_df.loc[m, metric_name] = s_plain
            
    table_df = table_df.fillna("-")
    plain_df = plain_df.fillna("-")
    
    # Escape underscores in method names for LaTeX
    table_df.index = [str(x).replace("_", "\\_") for x in table_df.index]
    
    print("=== PLAIN TEXT PRECISION/RECALL TABLE ===")
    print(plain_df.to_string())
    print("\n")
    
    with open("paper/tables/precision_recall.tex", "w") as f:
        f.write(table_df.to_latex(escape=False, column_format="lccc"))

if __name__ == "__main__":
    build_pr_table()
