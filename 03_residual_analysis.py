"""
=========================================================
Astro Project
File : 03_residual_analysis.py
Purpose:
Compute normalized and symmetrized residuals for all estimators.
=========================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# 1. Load Dataset & Central Estimates
print("="*70)
print("Loading Cleaned Dataset & Estimates")
print("="*70)
df = pd.read_csv("outputs/processed/helium_cleaned.csv")

with open("outputs/processed/central_estimates.json") as f:
    results = json.load(f)

mean = results["mean"]
sigma_mean = results["sigma_mean"]
weighted_mean = results["weighted_mean"]
sigma_weighted = results["sigma_weighted"]
median = results["median"]
sigma_plus = results.get("sigma_plus", 0.0)
sigma_minus = results.get("sigma_minus", 0.0)
sigma_med_rd = results.get("sigma_med_RD", (sigma_plus + sigma_minus) / 2.0)

print(f"Arithmetic Mean (AM) : {mean:.6f} ± {sigma_mean:.6f}")
print(f"Weighted Mean (WM)   : {weighted_mean:.6f} ± {sigma_weighted:.6f}")
print(f"Median (Med)         : {median:.6f} (+{sigma_plus:.6f}/-{sigma_minus:.6f})")

# 2. Vectorized Residual Calculations
print("\nCalculating Normalized Residuals (Vectorized)...")

# Arithmetic Mean Residuals
df["Residual_Mean"] = (df["Y"] - mean) / np.sqrt(df["sigma"]**2 + sigma_mean**2)

# Weighted Mean Residuals (Plus / Uncorrelated: wm+)
df["Residual_Weighted_Plus"] = (df["Y"] - weighted_mean) / np.sqrt(df["sigma"]**2 + sigma_weighted**2)

# Weighted Mean Residuals (Minus / Correlated: wm-)
# Handles non-positive denominators (sigma^2 <= sigma_wm^2) by setting to NaN (no artificial clipping)
var_diff = df["sigma"]**2 - sigma_weighted**2
res_wm_minus = np.full(len(df), np.nan, dtype=float)
valid_wm_minus_mask = var_diff > 0
res_wm_minus[valid_wm_minus_mask] = (df["Y"].values[valid_wm_minus_mask] - weighted_mean) / np.sqrt(var_diff.values[valid_wm_minus_mask])

df["Residual_Weighted_Minus"] = res_wm_minus
# Alias for backward compatibility with existing downstream scripts
df["Residual_Weighted"] = df["Residual_Weighted_Minus"]

# Median Residuals (handles asymmetric uncertainties if present, or uses standard median error)
if "sigma_plus" in results and "sigma_minus" in results:
    sigma_med = np.where(df["Y"] >= median, sigma_plus, sigma_minus)
else:
    sigma_med = sigma_med_rd
df["Residual_Median"] = (df["Y"] - median) / np.sqrt(df["sigma"]**2 + sigma_med**2)

# 3. Create Symmetrized Residuals (CRITICAL FOR STEPS 4 & 5)
# Duplicates valid non-NaN residuals with negative signs to isolate scale and tail thickness
print("Generating Symmetrized Residuals...")

def create_symmetrized_array(series):
    valid_data = series.dropna().values
    return np.concatenate([valid_data, -valid_data])

sym_mean = create_symmetrized_array(df["Residual_Mean"])
sym_wm_plus = create_symmetrized_array(df["Residual_Weighted_Plus"])
sym_wm_minus = create_symmetrized_array(df["Residual_Weighted_Minus"])
sym_median = create_symmetrized_array(df["Residual_Median"])

# Helper function to pad arrays with NaN to equal lengths for DataFrame export
max_len = max(len(sym_mean), len(sym_wm_plus), len(sym_wm_minus), len(sym_median))

def pad_array(arr, target_len):
    if len(arr) < target_len:
        padded = np.full(target_len, np.nan)
        padded[:len(arr)] = arr
        return padded
    return arr

# Save raw residuals to the main dataset
os.makedirs("outputs/processed", exist_ok=True)
df.to_csv("outputs/processed/residuals.csv", index=False)

# Save symmetrized residuals in a dedicated file for distribution fitting
df_sym = pd.DataFrame({
    "Sym_Residual_Mean": pad_array(sym_mean, max_len),
    "Sym_Residual_Weighted_Plus": pad_array(sym_wm_plus, max_len),
    "Sym_Residual_Weighted_Minus": pad_array(sym_wm_minus, max_len),
    "Sym_Residual_Weighted": pad_array(sym_wm_minus, max_len),  # Alias for backward compatibility
    "Sym_Residual_Median": pad_array(sym_median, max_len)
})
df_sym.to_csv("outputs/processed/symmetrized_residuals.csv", index=False)

# 4. Residual Statistics Output
def display_residual_summary(name, series):
    data = series.dropna().values
    print()
    print("="*70)
    print(f"Residual Statistics: {name}")
    print("="*70)
    print(f"Count (Valid) : {len(data)} / {len(series)}")
    if len(data) > 0:
        print(f"Mean          : {np.mean(data):.6f}")
        print(f"Std Dev       : {np.std(data):.6f}")
        print(f"Min / Max     : {np.min(data):.6f} / {np.max(data):.6f}")

display_residual_summary("Arithmetic Mean (Raw)", df["Residual_Mean"])
display_residual_summary("Weighted Mean [wm+] (Raw)", df["Residual_Weighted_Plus"])
display_residual_summary("Weighted Mean [wm-] (Raw)", df["Residual_Weighted_Minus"])
display_residual_summary("Median (Raw)", df["Residual_Median"])
display_residual_summary("Median (Symmetrized)", pd.Series(sym_median))

# 5. Plot Histograms
os.makedirs("outputs/plots", exist_ok=True)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df["Residual_Weighted_Plus"].dropna(), bins=20, edgecolor='black', alpha=0.5, color='green', label='wm+ (Uncorrelated)')
plt.hist(df["Residual_Weighted_Minus"].dropna(), bins=20, edgecolor='black', alpha=0.5, color='purple', label='wm- (Correlated)')
plt.xlabel("Raw Normalized Residual")
plt.ylabel("Frequency")
plt.title("Weighted Mean Residuals Comparison (wm+ vs wm-)")
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(df_sym["Sym_Residual_Median"].dropna(), bins=20, edgecolor='black', alpha=0.7, color='salmon')
plt.xlabel("Symmetrized Residual")
plt.ylabel("Frequency")
plt.title("Symmetrized Median Residuals")

plt.tight_layout()
plt.savefig("outputs/plots/residual_median_comparison.png", dpi=300)
plt.close()

print("\n" + "="*70)
print("03_residual_analysis.py Completed Successfully")
print("="*70)           