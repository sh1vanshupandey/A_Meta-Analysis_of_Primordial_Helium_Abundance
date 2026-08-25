"""Calculate central statistical estimates and uncertainties."""
"""
==========================================================
Astro Project
File : 02_basic_statistics.py
Purpose
-------
Calculate the three cosmological central estimators
(Arithmetic Mean, Weighted Mean, and Median Statistics)
along with their associated statistical uncertainties
for the Primordial Helium dataset.
==========================================================
"""

import pandas as pd
import numpy as np
import os
import json
from scipy.special import comb

# 1. Load Cleaned Dataset
df = pd.read_csv("outputs/processed/helium_cleaned.csv")

print("="*70)
print("Loading Cleaned Primordial Helium Dataset")
print("="*70)
print(f"Total active measurements (N) = {len(df)}")
print(df.head())

# 2. Arithmetic Mean (AM) & Standard Error (Eq. 5 of Rackers et al. 2024)
mean = df["Y"].mean()
N = len(df)
# Matched exactly to the equation in Rackers et al. (2024) and Ramakrishnan & Desai (2023)
sigma_mean = np.sqrt(np.sum((df["Y"] - mean)**2) / N**2)

print("\n" + "="*70)
print("1. Arithmetic Mean Central Estimator")
print("="*70)
print(f"Arithmetic Mean (AM)       : {mean:.6f}")
print(f"Arithmetic Mean Error (SEM): {sigma_mean:.6f}")

# 3. Weighted Mean (WM) & Standard Error (Eq. 2 & 3 of Rackers et al. 2024)
# We ensure only measurements with valid errors are used (already ensured in cleaning)
weighted_df = df[df["sigma"] > 0].copy()
weighted_df["weight"] = 1 / weighted_df["sigma"]**2

weighted_mean = (weighted_df["Y"] * weighted_df["weight"]).sum() / weighted_df["weight"].sum()
sigma_weighted = np.sqrt(1 / weighted_df["weight"].sum())

print("\n" + "="*70)
print("2. Weighted Mean Central Estimator")
print("="*70)
print(f"Weighted Mean (WM)         : {weighted_mean:.6f}")
print(f"Weighted Mean Error (WME)  : {sigma_weighted:.6f}")

# 4. Standard Dispersion Metrics
std = df["Y"].std()
variance = df["Y"].var()
maximum = df["Y"].max()
minimum = df["Y"].min()
data_range = maximum - minimum

print("\n" + "="*70)
print("3. Sample Dispersion Metrics")
print("="*70)
print(f"Standard Deviation (s)     : {std:.6f}")
print(f"Variance (s^2)             : {variance:.6f}")
print(f"Range (Max - Min)          : {data_range:.6f} [{minimum:.6f} to {maximum:.6f}]")

# 5. Median Statistics & Rigorous Gott et al. (2001) Binomial Confidence Intervals
sorted_Y = np.sort(df["Y"].values)
median = np.median(sorted_Y)

# Define binomial distribution parameters for N trials with p = 0.5
alpha = 1.0 - 0.6827  # 1-sigma standard significance

# Generate the complete list of binomial probabilities (size N+1, from i=0 to N)
p_binom = np.array([comb(N, i) * 0.5**N for i in range(N + 1)])
cumsum = np.cumsum(p_binom)

# Find Left (L) index where cumulative sum is closest to alpha/2 (0.15865)
left_idx = np.argmin(np.abs(cumsum - alpha / 2))
# Find Right (R) index where cumulative sum is closest to 1 - alpha/2 (0.84135)
right_idx = np.argmin(np.abs(cumsum - (1.0 - alpha / 2)))

# Map to sorted observations
lower_bound = sorted_Y[left_idx]
upper_bound = sorted_Y[right_idx]

sigma_minus = median - lower_bound
sigma_plus = upper_bound - median
actual_area = cumsum[right_idx] - cumsum[left_idx]

# 6. Alternative Median Error (Ramakrishnan & Desai 2023, Eq. 1)
# Uses sample standard deviation and is valid under mild non-Gaussianity
sigma_med_RD = std * np.sqrt(np.pi / (2 * N))

print("\n" + "="*70)
print("4. Robust Median Statistics (Gott et al. 2001)")
print("="*70)
print(f"Median (Med)               : {median:.6f}")
print(f"Lower 1-sigma Bound (Y_L)  : {lower_bound:.6f} (Index {left_idx})")
print(f"Upper 1-sigma Bound (Y_R)  : {upper_bound:.6f} (Index {right_idx})")
print(f"Binomial Gott Error (-)    : -{sigma_minus:.6f}")
print(f"Binomial Gott Error (+)    : +{sigma_plus:.6f}")
print(f"Actual Binomial Coverage   : {actual_area * 100:.2f}% (Target: 68.27%)")
print(f"Ramakrishnan-Desai Error   : ±{sigma_med_RD:.6f}")

# 7. Compile Results and Export
os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/processed", exist_ok=True)

# Save structured table
summary = pd.DataFrame({
    "Statistic": [
        "Arithmetic Mean", "Arithmetic Mean Error", 
        "Weighted Mean", "Weighted Mean Error", 
        "Median", "Median Gott Error Lower", "Median Gott Error Upper",
        "Median Desai Error", "Standard Deviation", "Variance"
    ],
    "Value": [
        mean, sigma_mean, 
        weighted_mean, sigma_weighted, 
        median, sigma_minus, sigma_plus,
        sigma_med_RD, std, variance
    ]
})
summary.to_csv("outputs/tables/basic_statistics.csv", index=False)

# Save JSON file for downstream pipeline scripts (03 through 07)
results = {
    "mean": mean,
    "sigma_mean": sigma_mean,
    "weighted_mean": weighted_mean,
    "sigma_weighted": sigma_weighted,
    "median": median,
    "sigma_minus": sigma_minus,
    "sigma_plus": sigma_plus,
    "sigma_med_RD": sigma_med_RD,
    "std": std,
    "variance": variance
}

with open("outputs/processed/central_estimates.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n" + "="*70)
print("Pipeline Comparison and Differences")
print("="*70)
print(f"Weighted Mean - Arithmetic Mean : {weighted_mean - mean:+.6f}")
print(f"Median - Arithmetic Mean        : {median - mean:+.6f}")
print(f"Weighted Mean - Median          : {weighted_mean - median:+.6f}")

print("\nAll central statistical estimates saved successfully.")
print("="*70)
print("02_basic_statistics.py Completed Successfully")
print("="*70)