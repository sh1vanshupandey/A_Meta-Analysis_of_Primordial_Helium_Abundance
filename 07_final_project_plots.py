"""
==========================================================
Astro Project
File : 07_final_project_plots.py
Purpose
-------
Generate publication-quality figures and visualizations 
representing the 50-year meta-analysis of primordial 
helium abundance measurements. Produces both PNG and 
vector PDF formats suitable for an MSc Physics dissertation.
==========================================================
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set professional plotting aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'text.usetex': False,  # Set to True if you have LaTeX installed locally
})

# Ensure output directories exist
os.makedirs('outputs/plots', exist_ok=True)

print('=' * 70)
print('Loading Cleaned and Processed Datasets for Plotting')
print('=' * 70)

df = pd.read_csv('outputs/processed/helium_cleaned.csv')
df_res = pd.read_csv('outputs/processed/residuals.csv')
df_sym = pd.read_csv('outputs/processed/symmetrized_residuals.csv')
df_tracers = pd.read_csv('outputs/tables/tracer_central_estimates.csv')

print(f'Loaded {len(df)} measurements, {len(df_tracers)} tracer groups.')

# ---------------------------------------------------------
# Plot 1: Primordial Helium (Yp) vs. Publication Year (Colored by Tracer)
# ---------------------------------------------------------
print('\nGenerating Plot 1: Yp vs. Publication Year...')
plt.figure(figsize=(10, 6))

for tracer in df['Tracer'].unique():
  subset = df[df['Tracer'] == tracer]
  plt.scatter(
      subset['YYYY'],
      subset['Y'],
      s=45,
      alpha=0.75,
      label=tracer,
      edgecolors='none',
  )

plt.axhline(
    0.247, color='black', linestyle='--', alpha=0.6, label='Standard BBN (0.247)'
)
plt.axhline(
    0.2463, color='blue', linestyle=':', alpha=0.6, label='Planck 2018 (0.2463)'
)

plt.xlabel('Publication Year', fontweight='bold')
plt.ylabel(r'Primordial Helium Mass Fraction ($Y_p$)', fontweight='bold')
plt.title(
    '50 Years of Primordial Helium Abundance Determinations',
    fontweight='bold',
    pad=15,
)
plt.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='gray')
plt.grid(True, linestyle=':', alpha=0.6)
plt.ylim(0.15, 0.35)  # Focus on the physically interesting scatter
plt.tight_layout()
plt.savefig('outputs/plots/Yp_vs_year.png', dpi=300)
plt.savefig('outputs/plots/Yp_vs_year.pdf')  # Save vector version
plt.close()

# ---------------------------------------------------------
# Plot 2: Yp with Total Uncertainties (Error Bars) vs. Year
# ---------------------------------------------------------
print('Generating Plot 2: Yp with Error Bars...')
plt.figure(figsize=(10, 6))

# Plot H II regions and CMB with their error bars separately to highlight the two main pillars
for tracer, color, marker in [
    ('H II regions / BCGs', 'salmon', 'o'),
    ('CMB', 'royalblue', 's'),
]:
  subset = df[df['Tracer'] == tracer]
  plt.errorbar(
      subset['YYYY'],
      subset['Y'],
      yerr=subset['sigma'],
      fmt=marker,
      color=color,
      ms=5,
      alpha=0.7,
      capsize=2,
      label=f'{tracer} (N={len(subset)})',
  )

# Plot other minor tracers in light gray to avoid cluttering the plot
minor_tracers = df[~df['Tracer'].isin(['H II regions / BCGs', 'CMB'])]
plt.errorbar(
    minor_tracers['YYYY'],
    minor_tracers['Y'],
    yerr=minor_tracers['sigma'],
    fmt='v',
    color='gray',
    ms=4,
    alpha=0.4,
    capsize=1,
    label='Other Tracers',
)

plt.axhline(0.247, color='black', linestyle='--', alpha=0.6, label='Standard BBN')
plt.xlabel('Publication Year', fontweight='bold')
plt.ylabel(
    r'Primordial Helium Mass Fraction ($Y_p \pm 1\sigma$)', fontweight='bold'
)
plt.title(
    'Chronological Evolution of $Y_p$ with Reported Uncertainties',
    fontweight='bold',
    pad=15,
)
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.ylim(0.15, 0.35)
plt.tight_layout()
plt.savefig('outputs/plots/Yp_vs_year_errorbars.png', dpi=300)
plt.savefig('outputs/plots/Yp_vs_year_errorbars.pdf')
plt.close()

# ---------------------------------------------------------
# Plot 3: Subplot of Yp and Uncertainty Histograms
# ---------------------------------------------------------
print('Generating Plot 3: Abundance & Uncertainty Histograms...')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Yp Histogram
ax1.hist(
    df['Y'].dropna(), bins=25, color='skyblue', edgecolor='black', alpha=0.7
)
ax1.axvline(
    0.244,
    color='red',
    linestyle='--',
    linewidth=1.5,
    label='Literature Median (0.244)',
)
ax1.set_xlabel(r'Helium Abundance Value ($Y_p$)', fontweight='bold')
ax1.set_ylabel('Frequency (Count)', fontweight='bold')
ax1.set_title('Distribution of Reported $Y_p$ Values', fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.6)

# Uncertainty Histogram
ax2.hist(
    df['sigma'].dropna(),
    bins=25,
    color='lightcoral',
    edgecolor='black',
    alpha=0.7,
)
ax2.set_xlabel(r'Total Uncertainty ($\sigma_{tot}$)', fontweight='bold')
ax2.set_ylabel('Frequency (Count)', fontweight='bold')
ax2.set_title(r'Distribution of Reported $1\sigma$ Uncertainties', fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('outputs/plots/histograms_combined.png', dpi=300)
plt.savefig('outputs/plots/histograms_combined.pdf')
plt.close()

# ---------------------------------------------------------
# Plot 4: Uncertainty vs. Year (Convergence Metric)
# ---------------------------------------------------------
print('Generating Plot 4: Uncertainty vs. Year...')
plt.figure(figsize=(10, 5))
plt.scatter(
    df['YYYY'],
    df['sigma'],
    color='darkorchid',
    s=40,
    alpha=0.7,
    edgecolors='black',
    linewidths=0.5,
)
plt.xlabel('Publication Year', fontweight='bold')
plt.ylabel(r'Total Reported Uncertainty ($\sigma_{tot}$)', fontweight='bold')
plt.title(
    'Evolution of Measurement Precision Over 50 Years', fontweight='bold', pad=15
)
plt.grid(True, linestyle=':', alpha=0.6)
plt.yscale(
    'log'
)  # Log scale highlights the multi-magnitude improvements in precision
plt.tight_layout()
plt.savefig('outputs/plots/uncertainty_vs_year.png', dpi=300)
plt.savefig('outputs/plots/uncertainty_vs_year.pdf')
plt.close()

# ---------------------------------------------------------
# Plot 5: Tracer Counts
# ---------------------------------------------------------
print('Generating Plot 5: Tracer Counts Bar Chart...')
plt.figure(figsize=(8, 5))
counts = df['Tracer'].value_counts()
colors = plt.cm.tab10(np.linspace(0, 1, len(counts)))

plt.bar(counts.index, counts, color=colors, edgecolor='black', alpha=0.8)
plt.xticks(rotation=35, ha='right')
plt.xlabel('Observational Method (Tracer Class)', fontweight='bold')
plt.ylabel('Number of Historical Determinations', fontweight='bold')
plt.title(
    'Composition of the Primordial Helium Literature', fontweight='bold', pad=15
)
plt.grid(True, axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('outputs/plots/tracer_counts.png', dpi=300)
plt.savefig('outputs/plots/tracer_counts.pdf')
plt.close()

# ---------------------------------------------------------
# Plot 6: Tracer-wise Central Estimates Comparison
# ---------------------------------------------------------
print('Generating Plot 6: Tracer Central Estimates Comparison...')
plt.figure(figsize=(10, 6))
x_pos = np.arange(len(df_tracers))

# 1. Plot Arithmetic Means
plt.errorbar(
    x_pos - 0.15,
    df_tracers['Arithmetic_Mean'],
    yerr=df_tracers['Arithmetic_Mean_Error'],
    fmt='o',
    color='tomato',
    capsize=4,
    elinewidth=1.5,
    label='Arithmetic Mean',
)

# 2. Plot Weighted Means
valid_wm = df_tracers['Weighted_Mean'].notna()
plt.errorbar(
    x_pos[valid_wm],
    df_tracers.loc[valid_wm, 'Weighted_Mean'],
    yerr=df_tracers.loc[valid_wm, 'Weighted_Mean_Error'],
    fmt='s',
    color='royalblue',
    capsize=4,
    elinewidth=1.5,
    label='Weighted Mean',
)

# 3. Plot Medians
plt.errorbar(
    x_pos + 0.15,
    df_tracers['Median'],
    yerr=[
        df_tracers['Median_Error_Lower'],
        df_tracers['Median_Error_Upper'],
    ],
    fmt='^',
    color='forestgreen',
    capsize=4,
    elinewidth=1.5,
    label='Median (Gott binomial)',
)

plt.xticks(x_pos, df_tracers['Tracer'], rotation=35, ha='right')
plt.ylabel(r'Primordial Helium Estimate ($Y_p$)', fontweight='bold')
plt.title(
    'Central Statistical Estimates Across Observational Methods',
    fontweight='bold',
    pad=15,
)
plt.legend(loc='upper right', frameon=True, facecolor='white')
plt.grid(True, linestyle=':', alpha=0.6)
plt.ylim(0.18, 0.28)
plt.tight_layout()
plt.savefig('outputs/plots/tracer_central_estimates.png', dpi=300)
plt.savefig('outputs/plots/tracer_central_estimates.pdf')
plt.close()

# ---------------------------------------------------------
# Plot 7: Global Raw Residuals Histograms (Includes wm+ and wm-)
# ---------------------------------------------------------
print('Generating Plot 7: Global Raw Residuals...')
fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

cols = [
    ('Residual_Mean', 'Arithmetic Mean Residuals', 'tomato'),
    (
        'Residual_Weighted_Plus',
        'Weighted Mean Residuals [wm+] (Uncorrelated)',
        'mediumseagreen',
    ),
    (
        'Residual_Weighted_Minus',
        'Weighted Mean Residuals [wm-] (Correlated)',
        'royalblue',
    ),
    ('Residual_Median', 'Median Residuals', 'forestgreen'),
]

for ax, (col, title, color) in zip(axes, cols):
  # Fallback to legacy column if wm- isn't explicitly named in old CSVs
  plot_col = (
      col
      if col in df_res.columns
      else ('Residual_Weighted' if col == 'Residual_Weighted_Minus' else col)
  )

  if plot_col in df_res.columns:
    ax.hist(
        df_res[plot_col].dropna(),
        bins=30,
        color=color,
        edgecolor='black',
        alpha=0.7,
        density=True,
    )

  ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
  ax.set_ylabel('Probability Density', fontweight='bold')
  ax.set_title(title, fontweight='bold')
  ax.grid(True, linestyle=':', alpha=0.6)
  ax.set_xlim(-5, 5)

plt.xlabel('Normalized Residual ($N_\\sigma$)', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/plots/global_residuals.png', dpi=300)
plt.savefig('outputs/plots/global_residuals.pdf')
plt.close()

# ---------------------------------------------------------
# Plot 8: Symmetrized Residuals Histograms (Includes wm+ and wm-)
# ---------------------------------------------------------
print('Generating Plot 8: Symmetrized Residuals...')
fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

sym_cols = [
    ('Sym_Residual_Mean', 'Symmetrized Arithmetic Mean Residuals', 'tomato'),
    (
        'Sym_Residual_Weighted_Plus',
        'Symmetrized Weighted Mean Residuals [wm+]',
        'mediumseagreen',
    ),
    (
        'Sym_Residual_Weighted_Minus',
        'Symmetrized Weighted Mean Residuals [wm-]',
        'royalblue',
    ),
    ('Sym_Residual_Median', 'Symmetrized Median Residuals', 'forestgreen'),
]

for ax, (col, title, color) in zip(axes, sym_cols):
  plot_col = (
      col
      if col in df_sym.columns
      else (
          'Sym_Residual_Weighted'
          if col == 'Sym_Residual_Weighted_Minus'
          else col
      )
  )

  if plot_col in df_sym.columns:
    ax.hist(
        df_sym[plot_col].dropna(),
        bins=30,
        color=color,
        edgecolor='black',
        alpha=0.7,
        density=True,
    )

  ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
  ax.set_ylabel('Probability Density', fontweight='bold')
  ax.set_title(title, fontweight='bold')
  ax.grid(True, linestyle=':', alpha=0.6)
  ax.set_xlim(-5, 5)

plt.xlabel('Symmetrized Error Residual ($N_\\sigma$)', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/plots/symmetrized_residuals.png', dpi=300)
plt.savefig('outputs/plots/symmetrized_residuals.pdf')
plt.close()

# ---------------------------------------------------------
# Plot 9: Tracer Medians Overview
# ---------------------------------------------------------
print('Generating Plot 9: Tracer Medians Overview...')
plt.figure(figsize=(8, 5))
plt.errorbar(
    x_pos,
    df_tracers['Median'],
    yerr=[
        df_tracers['Median_Error_Lower'],
        df_tracers['Median_Error_Upper'],
    ],
    fmt='o',
    color='forestgreen',
    ms=6,
    capsize=4,
    elinewidth=1.5,
)
plt.xticks(x_pos, df_tracers['Tracer'], rotation=35, ha='right')
plt.ylabel(r'Median Helium Abundance ($Y_{p, med}$)', fontweight='bold')
plt.title(
    'Median Primordial Helium Abundances by Tracer Class',
    fontweight='bold',
    pad=15,
)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('outputs/plots/tracer_medians.png', dpi=300)
plt.savefig('outputs/plots/tracer_medians.pdf')
plt.close()

# Save Metadata Summary
with open('outputs/processed/plot_summary.json', 'w') as f:
  json.dump(
      {
          'total_measurements': len(df),
          'measurements_with_errors': len(df[df['sigma'] > 0]),
          'tracer_groups': df['Tracer'].nunique(),
      },
      f,
      indent=4,
  )

print('\n' + '=' * 70)
print('07_final_project_plots.py Completed Successfully')
print('=' * 70)