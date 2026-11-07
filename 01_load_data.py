
#File : 01_load_data.py


import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)
pd.set_option("display.precision", 6)

print("="*70)
print("Loading Primordial Helium Dataset")
print("="*70)

file_path = "data/helium_abundances_Y_p.xlsx"
df = pd.read_excel(file_path, sheet_name="helium abundance numbers_final")

print("\nDataset Loaded Successfully\n")
print("="*70)
print("Dataset Shape")
print("="*70)
print(df.shape)
print("\nRows :", df.shape[0])
print("Columns :", df.shape[1])

print("\n"+"="*70)
print("Column Names")
print("="*70)
print(df.columns.tolist())

print("\n"+"="*70)
print("First Five Rows")
print("="*70)
print(df.head())

print("\n"+"="*70)
print("Last Five Rows")
print("="*70)
print(df.tail())

print("\n"+"="*70)
print("Dataset Information")
print("="*70)
df.info()

print("\n"+"="*70)
print("Statistical Summary")
print("="*70)
print(df.describe())

print("\n"+"="*70)
print("Missing Values")
print("="*70)
print(df.isnull().sum())

print("\n"+"="*70)
print("Duplicate Rows")
print("="*70)
duplicates = df.duplicated().sum()
print("Duplicates :", duplicates)

df = df[["YYYY","Y","Error_l","Error_u","Syst","Tracer","Classification","Source"]]
print(df.head())

# Drop rows missing Y first
df = df.dropna(subset=["Y"])

# --- [UPDATE 1: Rigorous Uncertainty Calculation (Quadrature)] ---
df["Error_l"] = pd.to_numeric(df["Error_l"], errors='coerce').fillna(0.0)
df["Error_u"] = pd.to_numeric(df["Error_u"], errors='coerce').fillna(0.0)
df["sigma_stat"] = (df["Error_l"] + df["Error_u"]) / 2
df["sigma_syst"] = pd.to_numeric(df["Syst"], errors='coerce').fillna(0.0)
df["sigma"] = np.sqrt(df["sigma_stat"]**2 + df["sigma_syst"]**2)

print(df[["Y","sigma_stat","sigma_syst","sigma"]].head())

print("\n"+"="*70)
print("Negative Errors")
print("="*70)
print((df["sigma"] < 0).sum())

print("\n"+"="*70)
print("Zero Error Measurements")
print("="*70)
zero_sigma = df[df["sigma"] == 0]
print(zero_sigma)

# --- [UPDATE 2: Scientific Exclusion & Zero-Error Drop] ---
print("\n"+"="*70)
print("Applying Scientific Cleaning Filters")
print("="*70)
initial_len = len(df)

# Drop zero-error measurements to prevent division by zero (inf weights) in Weighted Mean
df = df[df["sigma"] > 0]
print(f"Dropped {initial_len - len(df)} measurements with zero reported error.")

# Filter out uninformative outlier (McCarthy 1984) and upper limit (Dunkley 2009)
mccarthy_filter = (df["Source"] == "1984AJ.....89..433M") & (df["Y"] == 0.25)
dunkley_filter = (df["Source"] == "2009ApJS..180..306D") & (df["Y"] == 0.45)
df = df[~(mccarthy_filter | dunkley_filter)]
print(f"Dropped McCarthy (1984) and Dunkley et al. (2009). Active rows remaining: {len(df)}")


# --- [UPDATE 3: Tracer Term Homogenization] ---
print("\n"+"="*70)
print("Standardizing Tracer Categories")
print("="*70)

tracer_map = {
    'Extragalactic HII regions': 'H II regions / BCGs',
    'BCGs': 'H II regions / BCGs',
    'Irregular galaxies': 'H II regions / BCGs',
    'GCs': 'Globular Clusters',
    'RRLs': 'Radio Recombination Lines',
    'Nebulae': 'Nebulae',
    'Stars': 'Stars',
    'CMB': 'CMB',
    'Misc': 'Misc',
    'Miscellaneous': 'Misc'
}
df['Tracer'] = df['Tracer'].map(tracer_map).fillna('Others')


print("\n"+"="*70)
print("Tracer Types (Standardized)")
print("="*70)
print(df["Tracer"].unique())

print("\n"+"="*70)
print("Measurements Per Tracer (Standardized)")
print("="*70)
print(df["Tracer"].value_counts())

print("\n"+"="*70)
print("Classification")
print("="*70)
print(df["Classification"].value_counts())

# Sort chronologically
df = df.sort_values("YYYY")
df.reset_index(drop=True, inplace=True)

print("\n"+"="*70)
print("Clean Dataset Preview")
print("="*70)
print(df.head())

# Save the homogenized, cleaned dataset to output folder
df.to_csv("outputs/processed/helium_cleaned.csv", index=False)
print("\nCleaned dataset saved successfully to outputs/processed/helium_cleaned.csv.")

print("\n"+"="*70)
print("01_load_data.py Completed Successfully")
print("="*70)
