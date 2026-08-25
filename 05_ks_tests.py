"""
==========================================================
Astro Project
File : 05_ks_tests.py
Purpose
-------
Perform unbinned Kolmogorov-Smirnov (K-S) goodness-of-fit 
tests on symmetrized error residuals. Programmatically 
optimizes the scale factor (S*) to determine which continuous 
distribution (Gaussian, Laplace, Cauchy, or Student's-t) 
best describes the actual scatter in the literature.
==========================================================
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import norm, laplace, cauchy, t as student_t, kstest
from scipy.optimize import minimize_scalar

# Ensure output folders exist
os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/processed", exist_ok=True)

print("="*70)
print("Loading Symmetrized Residuals for K-S Testing")
print("="*70)

# Loaded directly from the Step 3 output for perfect pipeline consistency
df = pd.read_csv("outputs/processed/symmetrized_residuals.csv")
print(df.head())

# Map the estimators to our symmetrized columns
residual_columns = {
    "mean": "Sym_Residual_Mean",
    "wmean": "Sym_Residual_Weighted",
    "median": "Sym_Residual_Median"
}

# Define CDF functions for the K-S comparisons
def gaussian_cdf(x): return norm.cdf(x)
def laplace_cdf(x): return laplace.cdf(x)
def cauchy_cdf(x): return cauchy.cdf(x)
def student_cdf(x, nu): return student_t.cdf(x, df=nu)

def ks_test_distribution(r, dist_name, S, df_t=None):
    """Calculate the K-S D-statistic and p-value for a given scale factor S."""
    scaled_r = r / S
    if dist_name == "Gaussian":
        cdf = gaussian_cdf
    elif dist_name == "Laplace":
        cdf = laplace_cdf
    elif dist_name == "Cauchy":
        cdf = cauchy_cdf
    elif dist_name == "Student's-t":
        if df_t is None:
            raise ValueError("Degrees of freedom (nu) required for Student's-t.")
        cdf = lambda x: student_cdf(x, df_t)
    else:
        raise ValueError(f"Unknown distribution: {dist_name}")
    
    return kstest(scaled_r, cdf)

def optimize_scale(r, dist_name, df_t=None, S_min=0.05, S_max=5.0):
    """Find the scale S* that maximizes the K-S p-value (minimizes -p)."""
    # Objective function: we want to maximize the p-value, so we minimize -p
    objective = lambda S: -ks_test_distribution(r, dist_name, S, df_t)[1]
    
    # Run the bounded scalar minimizer
    res = minimize_scalar(objective, bounds=(S_min, S_max), method="bounded")
    S_opt = res.x
    
    # Recalculate K-S stats at the optimized scale
    D, p = ks_test_distribution(r, dist_name, S_opt, df_t)
    return S_opt, D, p

def optimize_student_t(r, df_min=2, df_max=50):
    """
    Find the optimal degrees of freedom (nu) and scale (S*) for Student's-t.
    Capped at df_max=50 for massive speedups (as nu > 50 converges to Gaussian).
    """
    best = None
    for nu in range(df_min, df_max + 1):
        S, D, p = optimize_scale(r, "Student's-t", nu)
        if best is None or p > best["p"]:
            best = {"S": S, "D": D, "p": p, "df": nu}
    return best

def analyze_residuals(r):
    """Run scale optimization across all 4 candidate distributions."""
    out = {}
    # Run standard 2-parameter distributions
    for dist in ["Gaussian", "Laplace", "Cauchy"]:
        S, D, p = optimize_scale(r, dist)
        out[dist] = {"S": float(S), "D": float(D), "p": float(p)}
        
    # Run Student's-t (which requires nested df loop)
    st_opt = optimize_student_t(r)
    out["Student's-t"] = {
        "S": float(st_opt["S"]),
        "D": float(st_opt["D"]),
        "p": float(st_opt["p"]),
        "df": int(st_opt["df"])
    }
    return out


# --- Execute the K-S Pipeline ---
all_results = {}

print("\n" + "="*70)
print("Running Kolmogorov-Smirnov Goodness-of-Fit Tests")
print("="*70)

for estimator, column in residual_columns.items():
    residuals = df[column].dropna().to_numpy()
    all_results[estimator] = analyze_residuals(residuals)
    
    print(f"\nResults for Estimator: {estimator.upper()}")
    print("-" * 50)
    for dist, v in all_results[estimator].items():
        df_str = f", df={v['df']}" if dist == "Student's-t" else ""
        print(f"  {dist:<13} -> Optimized S* = {v['S']:.4f} | D = {v['D']:.4f} | p-value = {v['p']:.6f}{df_str}")

# Save JSON results for plotting (07_final_project_plots.py) and discussion
with open("outputs/processed/ks_results.json", "w") as f:
    json.dump(all_results, f, indent=4)

# Save individual and combined CSV summary tables
for estimator, results in all_results.items():
    rows = []
    for dist, v in results.items():
        rows.append({
            "Distribution": dist,
            "S* (Scale)": v["S"],
            "D (K-S Stat)": v["D"],
            "p-value": v["p"],
            "df (nu)": v.get("df", np.nan)
        })
    pd.DataFrame(rows).to_csv(f"outputs/tables/KS_{estimator}.csv", index=False)

combined_rows = []
for e, results in all_results.items():
    for dist, v in results.items():
        combined_rows.append({
            "Estimator": e,
            "Distribution": dist,
            "S* (Scale)": v["S"],
            "D (K-S Stat)": v["D"],
            "p-value": v["p"],
            "df (nu)": v.get("df", np.nan)
        })
pd.DataFrame(combined_rows).to_csv("outputs/tables/KS_combined_summary.csv", index=False)

print("\n" + "="*70)
print("BEST-FIT SUMMARY FOR DISSERTATION DISCUSSION")
print("="*70)
for e, r in all_results.items():
    best_dist = max(r, key=lambda x: r[x]["p"])
    v = r[best_dist]
    df_str = f" (df={v['df']})" if best_dist == "Student's-t" else ""
    print(f"  {e.upper():<7}: Best model is {best_dist}{df_str} with p-value = {v['p']:.6f}")

print("\n" + "="*70)
print("05_ks_tests.py Completed Successfully")
print("="*70)