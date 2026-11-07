File : 05_ks_tests.py


import os
import json
import numpy as np
import pandas as pd
from scipy.stats import norm, laplace, cauchy, t as student_t, kstest
from scipy.optimize import minimize_scalar

# Ensure output directories exist
os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/processed", exist_ok=True)

print("=" * 70)
print("Loading Symmetrized Residuals for K-S Testing")
print("=" * 70)

# Load symmetrized dataset
df = pd.read_csv("outputs/processed/symmetrized_residuals.csv")
print(df.head())

# Target columns including both wm+ and wm-
residual_columns = {
    "mean": "Sym_Residual_Mean",
    "wmean_plus": "Sym_Residual_Weighted_Plus",
    "wmean_minus": "Sym_Residual_Weighted_Minus",
    "median": "Sym_Residual_Median"
}

# Backward compatibility alias if present
if "Sym_Residual_Weighted" in df.columns and "wmean" not in residual_columns:
    residual_columns["wmean"] = "Sym_Residual_Weighted"

# CDF definitions
def gaussian_cdf(x):
    return norm.cdf(x)

def laplace_cdf(x):
    return laplace.cdf(x)

def cauchy_cdf(x):
    return cauchy.cdf(x)

def student_cdf(x, nu):
    return student_t.cdf(x, df=nu)

# KS test for a given distribution and scale factor
def ks_test_distribution(r, dist_name, S, df_t=None):
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

# Optimize scale to maximize the KS p-value
def optimize_scale(r, dist_name, df_t=None, S_min=0.05, S_max=5.0):
    res = minimize_scalar(
        lambda S: -ks_test_distribution(r, dist_name, S, df_t).pvalue,
        bounds=(S_min, S_max),
        method="bounded"
    )
    S_opt = res.x
    result = ks_test_distribution(r, dist_name, S_opt, df_t)
    return S_opt, result.statistic, result.pvalue

# Find best Student's-t degrees of freedom and scale
def optimize_student_t(r, df_min=2, df_max=50):
    best = None
    for nu in range(df_min, df_max + 1):
        S_opt, D_opt, p_opt = optimize_scale(r, "Student's-t", df_t=nu)
        if best is None or p_opt > best["p"]:
            best = {"S": S_opt, "D": D_opt, "p": p_opt, "df": nu}
    return best

# Run KS scale optimization for all four distributions
def analyze_residuals(r):
    out = {}
    for dist in ["Gaussian", "Laplace", "Cauchy"]:
        S, D, p = optimize_scale(r, dist)
        out[dist] = {"S": float(S), "D": float(D), "p": float(p)}

    st = optimize_student_t(r)
    out["Student's-t"] = {
        "S": float(st["S"]),
        "D": float(st["D"]),
        "p": float(st["p"]),
        "df": int(st["df"])
    }
    return out

# Main execution
all_results = {}
combined_rows = []

print("\n" + "=" * 70)
print("Running Kolmogorov-Smirnov Goodness-of-Fit Tests")
print("=" * 70)

for estimator, column in residual_columns.items():
    if column not in df.columns:
        print(f"\nSkipping {estimator.upper()} ({column} not found in CSV).")
        continue

    residuals = df[column].dropna().to_numpy()
    if len(residuals) == 0:
        print(f"\nSkipping {estimator.upper()} (No valid residuals found).")
        continue

    # Gaussian test at the fixed nominal scale S = 1
    res_s1 = ks_test_distribution(residuals, "Gaussian", 1.0)
    D0, p0 = res_s1.statistic, res_s1.pvalue
    print(f"\n{estimator.upper()} - Gaussian Baseline at S = 1.0:")
    print(f"  D = {D0:.6f} | p-value = {p0:.8g}")

    # Optimized-scale tests
    all_results[estimator] = analyze_residuals(residuals)
    print(f"\nOptimized Results for Estimator: {estimator.upper()}")
    print("-" * 55)

    for dist, v in all_results[estimator].items():
        df_val = v.get("df", None)
        res_s1_dist = ks_test_distribution(
            residuals, dist, 1.0, df_t=df_val
        )
        s1_D, s1_p = res_s1_dist.statistic, res_s1_dist.pvalue

        combined_rows.append({
            "Estimator": estimator,
            "Distribution": dist,
            "S1_D": s1_D,
            "S1_pvalue": s1_p,
            "S* (Scale)": v["S"],
            "D (K-S Stat)": v["D"],
            "p-value": v["p"],
            "df (nu)": v.get("df", np.nan)
        })

        df_str = f", df={v['df']}" if dist == "Student's-t" else ""
        print(
            f"  {dist:<13} -> S*={v['S']:.4f} | "
            f"D={v['D']:.4f} | p={v['p']:.6f}{df_str}"
        )

# Save JSON summary
with open("outputs/processed/ks_results.json", "w") as f:
    json.dump(all_results, f, indent=4)

# Save individual CSV tables
for estimator, results in all_results.items():
    col = residual_columns[estimator]
    if col not in df.columns:
        continue

    res_arr = df[col].dropna().to_numpy()
    rows = []

    for dist, v in results.items():
        df_val = v.get("df", None)
        res_s1 = ks_test_distribution(
            res_arr, dist, 1.0, df_t=df_val
        )
        rows.append({
            "Distribution": dist,
            "S1_D": res_s1.statistic,
            "S1_pvalue": res_s1.pvalue,
            "S* (Scale)": v["S"],
            "D (K-S Stat)": v["D"],
            "p-value": v["p"],
            "df (nu)": v.get("df", np.nan)
        })

    pd.DataFrame(rows).to_csv(
        f"outputs/tables/KS_{estimator}.csv", index=False
    )

# Save combined CSV summaries
df_combined = pd.DataFrame(combined_rows)
df_combined.to_csv("outputs/tables/KS_combined_summary.csv", index=False)
df_combined.to_csv("outputs/processed/ks_results.csv", index=False)

# Print best-fit summary
print("\n" + "=" * 70)
print("BEST-FIT SUMMARY FOR DISSERTATION DISCUSSION")
print("=" * 70)

for estimator, results in all_results.items():
    best_dist = max(results, key=lambda x: results[x]["p"])
    v = results[best_dist]
    df_str = f" (df={v['df']})" if best_dist == "Student's-t" else ""
    print(
        f"  {estimator.upper():<12}: Best model is {best_dist}{df_str}, "
        f"p={v['p']:.6f} (S*={v['S']:.4f})"
    )

print("\n" + "=" * 70)
print("05_ks_tests.py Completed Successfully")
print("=" * 70)
