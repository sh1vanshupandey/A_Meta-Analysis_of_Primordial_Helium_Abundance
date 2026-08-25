"""
==========================================================
Astro Project
File : 06_tracer_analysis.py
Purpose
-------
Perform tracer-wise meta-analysis. Calculate central 
estimates for each major physical observation category,
compute tracer-wise residuals, and execute unbinned 
Kolmogorov-Smirnov goodness-of-fit tests on the large-sample 
subgroups (H II regions and CMB).
==========================================================
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.special import comb
from scipy.stats import norm, laplace, cauchy, t as student_t, kstest
from scipy.optimize import minimize_scalar

# Ensure output directories exist
os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/processed", exist_ok=True)

print("="*70)
print("Loading Cleaned Dataset for Tracer-Wise Analysis")
print("="*70)
df = pd.read_csv("outputs/processed/helium_cleaned.csv")

# Use the standardized Tracer column directly from Step 1
tracer_counts = df["Tracer"].value_counts()
print("\nObservational Tracer Counts in Cleaned Dataset:")
print(tracer_counts)


def calculate_gott_median_uncertainty(data, alpha=0.3173):
    """
    Calculate the Median and its exact Gott et al. (2001) binomial 
    confidence interval at 68.27% confidence (1-sigma).
    """
    sorted_d = np.sort(np.asarray(data, float))
    n = len(sorted_d)
    median_val = np.median(sorted_d)
    
    if n == 0:
        return np.nan, np.nan, np.nan
    if n == 1:
        return median_val, 0.0, 0.0

    # Generate exact binomial probabilities for p=0.5
    p_binom = np.array([comb(n, i) * 0.5**n for i in range(n + 1)])
    cumsum = np.cumsum(p_binom)

    # Locate the lower and upper bounds of the interval
    left_idx = np.argmin(np.abs(cumsum - alpha / 2))
    right_idx = np.argmin(np.abs(cumsum - (1.0 - alpha / 2)))

    # Prevent out-of-bounds indexing for very small samples
    left_idx = max(0, min(left_idx, n - 1))
    right_idx = max(0, min(right_idx, n - 1))

    lower_bound = sorted_d[left_idx]
    upper_bound = sorted_d[right_idx]

    return median_val, median_val - lower_bound, upper_bound - median_val


# --- Calculate Central Estimators per Tracer ---
tracer_estimates = {}

for tracer in tracer_counts.index:
    subset = df[df["Tracer"] == tracer]
    Y_vals = subset["Y"].dropna().to_numpy()
    N_sub = len(Y_vals)
    
    if N_sub == 0:
        continue

    # 1. Arithmetic Mean (AM) & standard error
    am_val = Y_vals.mean()
    sigma_am = np.sqrt(np.sum((Y_vals - am_val)**2) / N_sub**2) if N_sub > 1 else 0.0

    # 2. Weighted Mean (WM) & standard error
    weighted_sub = subset[subset["sigma"] > 0]
    if len(weighted_sub) > 0:
        weights = 1 / weighted_sub["sigma"]**2
        wm_val = np.sum(weighted_sub["Y"] * weights) / weights.sum()
        sigma_wm = np.sqrt(1 / weights.sum())
    else:
        wm_val, sigma_wm = np.nan, np.nan

    # 3. Median Statistics (Gott binomial)
    med_val, err_minus, err_plus = calculate_gott_median_uncertainty(Y_vals)

    tracer_estimates[tracer] = {
        "N": N_sub,
        "N_weighted": len(weighted_sub),
        "mean": am_val,
        "sigma_mean": sigma_am,
        "wmean": wm_val,
        "sigma_wmean": sigma_wm,
        "median": med_val,
        "sigma_minus": err_minus,
        "sigma_plus": err_plus
    }

# Save tracer central estimates to a table
summary_rows = []
for tracer, r in tracer_estimates.items():
    summary_rows.append({
        "Tracer": tracer,
        "N": r["N"],
        "Arithmetic_Mean": r["mean"],
        "Arithmetic_Mean_Error": r["sigma_mean"],
        "Weighted_Mean": r["wmean"],
        "Weighted_Mean_Error": r["sigma_wmean"],
        "Median": r["median"],
        "Median_Error_Lower": r["sigma_minus"],
        "Median_Error_Upper": r["sigma_plus"]
    })
df_estimates = pd.DataFrame(summary_rows)
df_estimates.to_csv("outputs/tables/tracer_central_estimates.csv", index=False)


# --- Calculate Tracer-wise Normalized Residuals ---
print("\nCalculating Tracer-Wise Normalized Residuals...")
df["Tracer_Residual_Mean"] = np.nan
df["Tracer_Residual_Weighted"] = np.nan
df["Tracer_Residual_Median"] = np.nan

for tracer, r in tracer_estimates.items():
    idx = df.index[df["Tracer"] == tracer]
    
    # 1. Tracer Arithmetic Mean Residuals
    denom_am = np.sqrt(df.loc[idx, "sigma"]**2 + r["sigma_mean"]**2)
    valid_am = denom_am > 0
    df.loc[idx[valid_am], "Tracer_Residual_Mean"] = (df.loc[idx[valid_am], "Y"] - r["mean"]) / denom_am[valid_am]

    # 2. Tracer Weighted Mean Residuals
    if not np.isnan(r["wmean"]):
        denom_wm_sq = df.loc[idx, "sigma"]**2 - r["sigma_wmean"]**2
        valid_wm = denom_wm_sq > 0
        df.loc[idx[valid_wm], "Tracer_Residual_Weighted"] = (df.loc[idx[valid_wm], "Y"] - r["wmean"]) / np.sqrt(denom_wm_sq[valid_wm])

    # 3. Tracer Median Residuals
    err_med = np.where(df.loc[idx, "Y"] >= r["median"], r["sigma_plus"], r["sigma_minus"])
    denom_med = np.sqrt(df.loc[idx, "sigma"]**2 + err_med**2)
    valid_med = denom_med > 0
    df.loc[idx[valid_med], "Tracer_Residual_Median"] = (df.loc[idx[valid_med], "Y"] - r["median"]) / denom_med[valid_med]


# --- Kolmogorov-Smirnov Goodness-of-Fit Tests ---
# Perfect mirroring to isolate tail shape
def symmetrize_residuals(x):
    clean_x = x[np.isfinite(x)]
    return np.concatenate([-clean_x, clean_x])

def ks_test_dist(r, dist_name, S, df_t=None):
    scaled_r = r / S
    if dist_name == "Gaussian":
        return kstest(scaled_r, norm.cdf)
    elif dist_name == "Laplace":
        return kstest(scaled_r, laplace.cdf)
    elif dist_name == "Cauchy":
        return kstest(scaled_r, cauchy.cdf)
    elif dist_name == "Student's-t":
        return kstest(scaled_r, lambda val: student_t.cdf(val, df=df_t))

def optimize_scale(r, dist_name, df_t=None):
    objective = lambda S: -ks_test_dist(r, dist_name, S, df_t)[1]
    res = minimize_scalar(objective, bounds=(0.05, 5.0), method="bounded")
    S_opt = res.x
    D, p = ks_test_dist(r, dist_name, S_opt, df_t)
    return S_opt, D, p

def run_ks_suite(r):
    out = {}
    for dist in ["Gaussian", "Laplace", "Cauchy"]:
        S, D, p = optimize_scale(r, dist)
        out[dist] = {"S": float(S), "D": float(D), "p": float(p)}
        
    # Student's-t optimization (capped at df_max=50 for massive speedup)
    best_t = None
    for nu in range(2, 51):
        S, D, p = optimize_scale(r, "Student's-t", nu)
        if best_t is None or p > best_t[2]:
            best_t = (S, D, p, nu)
            
    out["Student's-t"] = {
        "S": float(best_t[0]),
        "D": float(best_t[1]),
        "p": float(best_t[2]),
        "df": int(best_t[3])
    }
    return out


# Execute K-S test only on major sample categories (N > 10)
tracer_ks_results = {}
for tracer, r in tracer_estimates.items():
    if r["N"] <= 10:
        continue  # Skip small sample sizes to avoid noise
        
    g = df[df["Tracer"] == tracer]
    tracer_ks_results[tracer] = {}
    
    cols = [
        ("mean", "Tracer_Residual_Mean"),
        ("wmean", "Tracer_Residual_Weighted"),
        ("median", "Tracer_Residual_Median")
    ]
    
    for est_name, col_name in cols:
        raw_res = g[col_name].dropna().to_numpy()
        if len(raw_res) >= 3:
            sym_res = symmetrize_residuals(raw_res)
            tracer_ks_results[tracer][est_name] = run_ks_suite(sym_res)

# Export and print results
with open("outputs/processed/tracer_ks_results.json", "w") as f:
    json.dump(tracer_ks_results, f, indent=4)

# Create summary rows for CSV export
ks_summary_rows = []
for tracer, es in tracer_ks_results.items():
    for est, ds in es.items():
        for dist, v in ds.items():
            ks_summary_rows.append({
                "Tracer": tracer,
                "Estimator": est,
                "Distribution": dist,
                "S* (Scale)": v["S"],
                "D (K-S Stat)": v["D"],
                "p-value": v["p"],
                "df (nu)": v.get("df", np.nan)
            })
pd.DataFrame(ks_summary_rows).to_csv("outputs/tables/tracer_ks_summary.csv", index=False)

# Save updated dataset with tracer residuals
df.to_csv("outputs/processed/tracer_residuals.csv", index=False)

print("\n" + "="*70)
print("TRACER-WISE ESTIMATORS SUMMARY")
print("="*70)
print(df_estimates.to_string(index=False))

print("\n" + "="*70)
print("TRACER-WISE K-S BEST-FIT RESULTS")
print("="*70)
for tracer, es in tracer_ks_results.items():
    print(f"\nTracer: {tracer}")
    print("-" * 50)
    for est, ds in es.items():
        best_dist = max(ds, key=lambda x: ds[x]["p"])
        v = ds[best_dist]
        df_str = f" (df={v['df']})" if best_dist == "Student's-t" else ""
        print(f"  {est.upper():<7} Estimator -> Best Fit: {best_dist}{df_str} | p-value = {v['p']:.6f}")

print("\n" + "="*70)
print("06_tracer_analysis.py Completed Successfully")
print("="*70)