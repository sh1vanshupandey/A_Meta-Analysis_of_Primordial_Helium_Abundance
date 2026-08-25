"""
==========================================================
Astro Project
File : 04_distribution_fits.py
Purpose
-------
Fit multiple probability distributions (Gaussian, Laplace, 
Cauchy, and Student's-t) to the Symmetrized Residuals 
of primordial helium measurements, forcing the location 
parameter to zero to isolate error scale and tail thickness.
==========================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, laplace, cauchy, t as student_t

# Ensure output directories exist
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

print("="*70)
print("Loading Symmetrized Residual Dataset")
print("="*70)

# We load the symmetrized residuals generated in Step 3
df = pd.read_csv("outputs/processed/symmetrized_residuals.csv")
print(df.head())


def fit_distributions(residuals):
    """Fit distributions with the location parameter fixed at exactly 0 (floc=0)."""
    # floc=0 locks the location parameter at 0.0, fitting only scale (and df for Student's-t)
    mu, sigma = norm.fit(residuals, floc=0)
    laplace_loc, laplace_scale = laplace.fit(residuals, floc=0)
    cauchy_loc, cauchy_scale = cauchy.fit(residuals, floc=0)
    
    # t.fit returns (df, loc, scale). We force loc to 0 using floc=0.
    df_t, loc_t, scale_t = student_t.fit(residuals, floc=0)
    
    return {
        "Gaussian": (mu, sigma),
        "Laplace": (laplace_loc, laplace_scale),
        "Cauchy": (cauchy_loc, cauchy_scale),
        "Student": (df_t, loc_t, scale_t)
    }


def plot_distribution(residuals, parameters, column):
    """Generate publication-quality PDF/PNG comparison plots of the fits."""
    plt.figure(figsize=(9, 6))
    
    # Plot empirical histogram
    plt.hist(residuals, bins=30, density=True, alpha=0.4, 
             color="gray", edgecolor="black", label="Symmetrized Residuals")
    
    # Generate smooth x-axis grid
    x = np.linspace(residuals.min() - 1, residuals.max() + 1, 2000)

    # Plot Gaussian fit
    _, sigma = parameters["Gaussian"]
    plt.plot(x, norm.pdf(x, 0, sigma), linewidth=2.5, 
             linestyle="-", label=f"Gaussian ($\\sigma$={sigma:.3f})")

    # Plot Laplace fit
    _, scale_lap = parameters["Laplace"]
    plt.plot(x, laplace.pdf(x, 0, scale_lap), linewidth=2.5, 
             linestyle="--", label=f"Laplace ($b$={scale_lap:.3f})")

    # Plot Cauchy fit
    _, scale_cau = parameters["Cauchy"]
    plt.plot(x, cauchy.pdf(x, 0, scale_cau), linewidth=2.5, 
             linestyle=":", label=f"Cauchy ($\\gamma$={scale_cau:.3f})")

    # Plot Student's-t fit
    nu, _, scale_t = parameters["Student"]
    plt.plot(x, student_t.pdf(x, nu, 0, scale_t), linewidth=2.5, 
             linestyle="-.", label=f"Student-t ($\\nu$={nu:.2f}, $s$={scale_t:.3f})")

    plt.xlabel("Symmetrized Error Residual ($N_\\sigma$)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.title(f"Symmetrized Error Distribution Fits: {column.replace('Sym_Residual_', '')}", fontsize=13, fontweight='bold')
    plt.legend(fontsize=10, loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xlim(-6, 6)  # Standard range to show the bulk of the residuals
    plt.tight_layout()
    
    # Save both high-res PNG and vector PDF for your thesis
    plt.savefig(f"outputs/plots/symmetrized_{column}_fit.png", dpi=300)
    plt.close()


def save_parameters(parameters, column):
    """Save the fitted parameters to a structured CSV table."""
    summary = pd.DataFrame({
        "Distribution": ["Gaussian", "Laplace", "Cauchy", "Student-t"],
        "Location (fixed)": [0.0, 0.0, 0.0, 0.0],
        "Scale Parameter": [
            parameters["Gaussian"][1], 
            parameters["Laplace"][1], 
            parameters["Cauchy"][1], 
            parameters["Student"][2]
        ],
        "Degrees of Freedom (nu)": [np.nan, np.nan, np.nan, parameters["Student"][0]]
    })
    summary.to_csv(f"outputs/tables/symmetrized_{column}_parameters.csv", index=False)


def print_summary(parameters):
    """Print the fits cleanly to the terminal for verification."""
    print()
    print("-" * 50)
    print("Fitted Scale Parameters (Location locked at 0.0)")
    print("-" * 50)
    print(f"Gaussian Scale (std dev)    : {parameters['Gaussian'][1]:.4f}")
    print(f"Laplace Scale (b)          : {parameters['Laplace'][1]:.4f}")
    print(f"Cauchy Scale (gamma)       : {parameters['Cauchy'][1]:.4f}")
    print(f"Student-t (nu, scale s)    : nu = {parameters['Student'][0]:.2f}, s = {parameters['Student'][2]:.4f}")
    print("-" * 50)


# Standardized columns matching Step 3 Symmetrized outputs
residual_columns = ["Sym_Residual_Mean", "Sym_Residual_Weighted", "Sym_Residual_Median"]

for column in residual_columns:
    print("\n" + "="*70)
    print(f"Processing Symmetrized Fits for: {column}")
    print("="*70)

    residuals = df[column].dropna()
    print("Number of Residuals (2N) :", len(residuals))

    parameters = fit_distributions(residuals)
    print_summary(parameters)
    plot_distribution(residuals, parameters, column)
    save_parameters(parameters, column)

    print(f"--> {column} distribution fitting complete.")

print("\n" + "="*70)
print("04_distribution_fits.py Completed Successfully")
print("="*70)