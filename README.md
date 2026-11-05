# A Meta-Analysis of Primordial Helium Abundance (\(Y_p\))

A computational and statistical meta-analysis of published primordial helium-4 mass-fraction measurements. The project examines central estimators, reported measurement uncertainties, normalized residuals, distributional shape, and differences among observational tracer classes.

**Dataset analysed:** 131 retained measurements after the project's documented cleaning procedure.  
**Analysis language:** Python  
**Primary methodological reference:** Ramakrishnan & Desai (2023), [arXiv:2310.17860](https://arxiv.org/abs/2310.17860).

---

## Scientific motivation

The primordial helium mass fraction, \(Y_p\), is an important observable in Big Bang nucleosynthesis (BBN) and observational cosmology. Published determinations span multiple decades and use different observational methods. Their quoted uncertainties can therefore reflect different statistical assumptions and systematic effects.

This project compares three central estimators—arithmetic mean, inverse-variance weighted mean, and median—and investigates whether normalized residuals are consistent with nominal Gaussian errors. It also compares alternative residual distributions and repeats the analysis within observational tracer classes.

## Analysis workflow

Run the seven scripts in order from the repository root.

| Script | Purpose | Main outputs |
|---|---|---|
| `01_load_data.py` | Read the Excel dataset, calculate combined uncertainties, apply cleaning filters, standardize tracer labels, and sort the retained measurements. | `outputs/processed/helium_cleaned.csv` |
| `02_basic_statistics.py` | Calculate the global arithmetic mean, weighted mean, standard deviations, and Gott binomial median interval. | `outputs/tables/basic_statistics.csv`; `outputs/processed/central_estimates.json` |
| `03_residual_analysis.py` | Calculate normalized residuals for the mean, weighted mean \(wm^+\), weighted mean \(wm^-\), and median; construct symmetrized residual arrays. | `outputs/processed/residuals.csv`; `outputs/processed/symmetrized_residuals.csv` |
| `04_distribution_fits.py` | Fit Gaussian, Laplace, Cauchy, and Student's-\(t\) distributions to the symmetrized residuals with location fixed at zero. | `outputs/tables/`; `outputs/plots/` |
| `05_ks_tests.py` | Run one-sample KS comparisons, optimize the scale factor, and search Student's-\(t\) degrees of freedom. | `outputs/processed/ks_results.json`; `outputs/tables/KS_combined_summary.csv` and estimator tables |
| `06_tracer_analysis.py` | Calculate tracer-specific central estimates and residuals, then run distribution/KS comparisons for tracer groups with more than 10 measurements. | `outputs/tables/tracer_central_estimates.csv`; `outputs/tables/tracer_ks_summary.csv`; `outputs/processed/tracer_residuals.csv`; `outputs/processed/tracer_ks_results.json` |
| `07_final_project_plots.py` | Generate overview figures, tracer comparisons, and raw/symmetrized residual histograms. | PNG and PDF files in `outputs/plots/` |

### Residual definitions

For measurement \(i\), with abundance \(Y_i\), quoted uncertainty \(\sigma_i\), and estimator uncertainty \(\sigma_{\rm est}\), the project uses the following normalized residual definitions.

**Arithmetic mean:**

\[
r_{{\rm AM},i}=\frac{Y_i-\bar Y}{\sqrt{\sigma_i^2+\sigma_{\rm AM}^2}}.
\]

**Weighted mean, \(wm^+\) (plus / uncorrelated formulation):**

\[
r_{wm^+,i}=\frac{Y_i-Y_{wm}}{\sqrt{\sigma_i^2+\sigma_{wm}^2}}.
\]

**Weighted mean, \(wm^-\) (minus / correlated formulation):**

\[
r_{wm^-,i}=\frac{Y_i-Y_{wm}}{\sqrt{\sigma_i^2-\sigma_{wm}^2}}.
\]

For \(wm^-\), measurements with \(\sigma_i^2-\sigma_{wm}^2\leq 0\) are assigned `NaN`; the denominator is not clipped to an artificial positive value.

**Median:** the residual uses the side-specific Gott median uncertainty selected according to whether the measurement lies above or below the sample median.

The symmetrized sample is formed by appending the negative of each valid residual to the original residual array. This is used to examine the scale and tail shape of the residual distribution.

## Reported analysis snapshot

The following values summarize the current project run as reported in the project notes. They should be checked against the corresponding CSV/JSON outputs before being used in a final publication or dissertation table.

### Global central estimates

| Estimator | Reported result |
|---|---:|
| Arithmetic mean | \(Y_p=0.24428\pm0.00197\) |
| Inverse-variance weighted mean | \(Y_p=0.24624\pm0.00016\) |
| Median (Gott binomial interval) | \(Y_p=0.24460^{+0.00040}_{-0.00160}\) |

The small formal uncertainty of the weighted mean should be interpreted in the context of heterogeneous measurements and possible unaccounted systematics; it should not automatically be treated as the total uncertainty on the field-wide value.

### Residual distributions and KS comparisons

The current analysis reports low KS p-values for the nominal Gaussian comparison at fixed scale \(S=1\) across all four residual definitions. After optimizing the scale and comparing candidate distributions, the reported preferences include:

| Residual definition | Reported optimized comparison |
|---|---|
| Arithmetic mean | Laplace: \(S^*\approx1.53\), \(p\approx0.996\) |
| Weighted mean \(wm^+\) | Student's-\(t\): \(\nu=2\), \(S^*=1.3943\), \(p=0.8617\) |
| Weighted mean \(wm^-\) | Student's-\(t\): \(\nu=2\), \(S^*=1.3976\), \(p=0.8528\) |
| Median | Laplace: \(S^*\approx1.74\), \(p\approx0.961\) |

These are exploratory model-comparison results. The standard one-sample KS p-values are not conventionally calibrated here because the symmetrized observations contain dependent \((r_i,-r_i)\) pairs, and the scale (and Student's-\(t\) degrees of freedom) are selected using the same sample. Avoid interpreting the optimized p-values as ordinary confirmatory significance levels or as proof that one physical mechanism caused the observed scatter.

### Tracer-wise analysis

The cleaned dataset contains seven standardized tracer categories. The tracer KS analysis is restricted to groups with \(N>10\). The main groups reported for the project include:

- Extragalactic H II regions / blue compact galaxies: \(N=72\).
- Cosmic Microwave Background (CMB): \(N=33\).
- Globular clusters: \(N=13\).

The reported H II region / BCG analysis requires scale factors of approximately 2 or larger across the four residual definitions. This indicates that the scatter in this subgroup is broader than the nominal residual scale in the tested models. The result alone does not identify which astrophysical systematic is responsible.

## Repository layout

```text
.
├── 01_load_data.py
├── 02_basic_statistics.py
├── 03_residual_analysis.py
├── 04_distribution_fits.py
├── 05_ks_tests.py
├── 06_tracer_analysis.py
├── 07_final_project_plots.py
├── data/
│   └── helium_abundances_Y_p.xlsx
└── outputs/
    ├── processed/
    ├── tables/
    └── plots/
```

The exact generated output files depend on the version of the scripts being run. The scripts write intermediate datasets, JSON summaries, CSV tables, and figures beneath `outputs/`.

## Requirements

The scripts use the following Python packages:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `openpyxl` (to read the Excel input file)

A virtual environment is recommended. For example:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy matplotlib openpyxl
```

Before running the pipeline, ensure the source workbook is located at `data/helium_abundances_Y_p.xlsx` and create the output directories:

```bash
mkdir -p outputs/processed outputs/tables outputs/plots
```

## Running the pipeline

From the repository root, execute:

```bash
python3 01_load_data.py
python3 02_basic_statistics.py
python3 03_residual_analysis.py
python3 04_distribution_fits.py
python3 05_ks_tests.py
python3 06_tracer_analysis.py
python3 07_final_project_plots.py
```

Each step depends on files produced by earlier steps, so run them sequentially. The pipeline has not been packaged as an installable Python module.

## Important implementation notes

1. **Uncertainty convention:** `01_load_data.py` calculates the statistical uncertainty as the average of the lower and upper error columns, then combines it in quadrature with the systematic uncertainty. Confirm that the workbook's lower and upper error columns are stored as positive magnitudes before interpreting this calculation.
2. **Legacy weighted-residual alias:** the residual-analysis script also writes `Residual_Weighted` as an alias of `Residual_Weighted_Minus` for backward compatibility. The symmetrized file similarly contains `Sym_Residual_Weighted` as an alias of `Sym_Residual_Weighted_Minus`. The fitting and KS scripts should be checked to ensure this alias is not unintentionally counted as an additional independent estimator in summary tables.
3. **Tracer threshold:** the current tracer KS workflow skips groups with 10 or fewer measurements. Tracer central estimates are calculated for all available categories.
4. **Reproducibility:** retain the input workbook, script versions, and generated CSV/JSON outputs together when archiving a run. Record any exclusions or changes to tracer mappings.

## References

- Ramakrishnan, S. & Desai, S. (2023), *A meta-analysis of distance measurements to M87*, [arXiv:2310.17860](https://arxiv.org/abs/2310.17860).
- Gott, J. R., Vogeley, M. S., Podariu, S. & Ratra, B. (2001), *Median Statistics, H\(_0\), and the Age of the Universe*, **The Astrophysical Journal**, 549, 1.
- Rackers et al. (2024), reference cited in the project scripts for the arithmetic- and weighted-mean estimators. Add the complete bibliographic entry used in the dissertation before publication.

## Author and supervision

**Student:** Shivanshu Pandey, MSc Physics, IIT Hyderabad  
**Supervisor:** Dr. Shantanu Desai, IIT Hyderabad
