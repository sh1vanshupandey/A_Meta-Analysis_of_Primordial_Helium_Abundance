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
sigma_plus = results["sigma_plus"]
sigma_minus = results["sigma_minus"]

print(f"Arithmetic Mean (AM) : {mean:.6f} ± {sigma_mean:.6f}")
print(f"Weighted Mean (WM)   : {weighted_mean:.6f} ± {sigma_weighted:.6f}")
print(f"Median (Med)         : {median:.6f} (+{sigma_plus:.6f}/-{sigma_minus:.6f})")

# 2. Vectorized Residual Calculations
print("\nCalculating Normalized Residuals (Vectorized)...")

# Arithmetic Mean Residuals
df["Residual_Mean"] = (df["Y"] - mean) / np.sqrt(df["sigma"]**2 + sigma_mean**2)

# Weighted Mean Residuals (uses subtraction correction with clipping for safety)
df["Residual_Weighted"] = (df["Y"] - weighted_mean) / np.sqrt(
    np.clip(df["sigma"]**2 - sigma_weighted**2, 1e-12, None)
)

# Median Residuals (handles asymmetric uncertainties)
sigma_med = np.where(df["Y"] >= median, sigma_plus, sigma_minus)
df["Residual_Median"] = (df["Y"] - median) / np.sqrt(df["sigma"]**2 + sigma_med**2)

# 3. Create Symmetrized Residuals (CRITICAL FOR STEPS 4 & 5)
# This duplicates each residual with a negative sign to isolate scale and tail thickness
print("Generating Symmetrized Residuals...")
sym_mean = np.concatenate([df["Residual_Mean"].values, -df["Residual_Mean"].values])
sym_weighted = np.concatenate([df["Residual_Weighted"].values, -df["Residual_Weighted"].values])
sym_median = np.concatenate([df["Residual_Median"].values, -df["Residual_Median"].values])

# Save raw residuals to the main dataset
os.makedirs("outputs/processed", exist_ok=True)
df.to_csv("outputs/processed/residuals.csv", index=False)

# Save symmetrized residuals in a dedicated file for distribution fitting
df_sym = pd.DataFrame({
    "Sym_Residual_Mean": sym_mean,
    "Sym_Residual_Weighted": sym_weighted,
    "Sym_Residual_Median": sym_median
})
df_sym.to_csv("outputs/processed/symmetrized_residuals.csv", index=False)

# 4. Residual Statistics Output
def display_residual_summary(name, data):
    print()
    print("="*70)
    print(f"Residual Statistics: {name}")
    print("="*70)
    print(f"Count   : {len(data)}")
    print(f"Mean    : {np.mean(data):.6f}")
    print(f"Std Dev : {np.std(data):.6f}")
    print(f"Min/Max : {np.min(data):.6f} / {np.max(data):.6f}")

display_residual_summary("Arithmetic Mean (Raw)", df["Residual_Mean"])
display_residual_summary("Weighted Mean (Raw)", df["Residual_Weighted"])
display_residual_summary("Median (Raw)", df["Residual_Median"])
display_residual_summary("Median (Symmetrized)", df_sym["Sym_Residual_Median"])

# 5. Plot Histograms
os.makedirs("outputs/plots", exist_ok=True)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(df["Residual_Median"], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel("Raw Normalized Residual")
plt.ylabel("Frequency")
plt.title("Raw Median Residuals")

plt.subplot(1, 2, 2)
plt.hist(df_sym["Sym_Residual_Median"], bins=20, edgecolor='black', alpha=0.7, color='salmon')
plt.xlabel("Symmetrized Residual")
plt.ylabel("Frequency")
plt.title("Symmetrized Median Residuals")

plt.tight_layout()
plt.savefig("outputs/plots/residual_median_comparison.png", dpi=300)
plt.close()

print("\n" + "="*70)
print("03_residual_analysis.py Completed Successfully")
print("="*70)