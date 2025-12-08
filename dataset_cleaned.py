import pandas as pd
import numpy as np

# Read the merged panel data
df = pd.read_excel('master_panel_.xlsx')

# Convert numeric columns
numeric_columns = [
    'Government_Effectiveness',
    'Regulatory_Quality',
    'GDP',
    'Labour_Force_Total',
    'Tax_Revenue_GDP',
    'Agriculture_GDP',
    'Trade_Openness_GDP',
    'Human_Digitalization_InternetUsers',
    'Remittances_GDP',
    'Inflation',
    'Unemployment_Rate',
    'Rule_of_Law',
    'Corruption_Perception_Index'
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
print("\n" + "="*60 + "\n")
print("AFTER CORRECTION:")
print(df.dtypes)
# Compute inflation rate (%) from CPI if available
df = df.sort_values(['Country', 'Year'])  # important for pct_change
if 'Inflation' in df.columns:
    df['Inflation_Rate'] = df.groupby('Country')['Inflation'].pct_change() * 100
    # Drop original "Inflation" column
    df = df.drop(columns=['Inflation'])
else:
    df['Inflation_Rate'] = np.nan
    print("Inflation column not found; Inflation_Rate set to NaN.")
# Create GDP per Worker using Labour_Force_Total as denominator
df['GDP_per_Worker'] = df['GDP'] / df['Labour_Force_Total']
df.loc[np.isinf(df['GDP_per_Worker']), 'GDP_per_Worker'] = np.nan

# Drop GDP and Labour_Force_Total columns
columns_to_drop = ['GDP', 'Labour_Force_Total','ISO3']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# --- Year first, then countries ---
# Parameters
density_threshold = 0.9
complete_delta_max = 1
id_cols = ["Country", "Year"]
vars_cols = [c for c in df.columns if c not in id_cols]

# Compute density and complete-country counts by year
rows = []
for year, g in df.groupby("Year"):
    total_cells = len(g) * len(vars_cols)
    non_missing_cells = g[vars_cols].notna().sum().sum()
    density = non_missing_cells / total_cells if total_cells else 0.0

    # Countries with all vars present for that year (row-level completeness)
    complete_countries = (g[vars_cols].notna().all(axis=1)).sum()

    rows.append({
        "Year": year,
        "Total_Data_Density": density,
        "Complete_Countries": complete_countries
    })

summary = pd.DataFrame(rows).sort_values("Year")
summary["Complete_Countries_Diff"] = summary["Complete_Countries"].diff().fillna(summary["Complete_Countries"])

# Find turning point: first year where density ≥ threshold and complete-country growth stabilizes
candidate = summary[
    (summary["Total_Data_Density"] >= density_threshold) &
    (summary["Complete_Countries_Diff"] <= complete_delta_max)
].head(1)

print("\nYearly coverage summary:")
print(summary)

heuristic_turning_year = int(candidate.iloc[0]["Year"]) if not candidate.empty else None
print(f"\nTurning point (not applied): {heuristic_turning_year}")

# turning point to 2000 regardless of heuristic
turning_year = 2000
print(f"Turning point year used: {turning_year}")

# 1) Filter by year (keep years >= turning point)
df = df[df["Year"] >= turning_year].reset_index(drop=True)
print("\nYears kept:", sorted(df["Year"].unique()))

# 2) Coverage per country on the filtered years
vars_cols = [c for c in df.columns if c not in id_cols]
country_coverage = (
    df.groupby("Country")[vars_cols]
      .apply(lambda x: x.notna().sum().sum() / (x.shape[0] * x.shape[1]))
      .reset_index(name="Coverage")
)
print("\nCountry coverage summary (post year-filter):")
print(country_coverage.sort_values("Coverage"))
country_threshold = 0.8
countries_keep = country_coverage[country_coverage["Coverage"] >= country_threshold]["Country"]

print(f"\nCountries kept (>= {country_threshold*100:.0f}% coverage): {len(countries_keep)}")
print(countries_keep.tolist())

# Filter dataset by countries
df = df[df["Country"].isin(countries_keep)].reset_index(drop=True)

# Final dataset
df.to_excel("master_panel_final_2000.xlsx", index=False)
print("\n✓ Final dataset saved: master_panel_final_2000.xlsx")
# Understanding the type of data - analyze all numeric columns in final dataset
print("\n" + "="*80)
print("UNDERSTANDING YOUR DATA SCALES")
print("="*80)

# Get all numeric columns from the final dataframe (excluding ID columns)
id_cols_final = ["Country", "Year"]
final_numeric_cols = [col for col in df.columns if col not in id_cols_final and pd.api.types.is_numeric_dtype(df[col])]

for col in sorted(final_numeric_cols):
    min_val = df[col].min()
    max_val = df[col].max()
    mean_val = df[col].mean()
    
    print(f"\n{col}:")
    print(f"  Range: {min_val:.4f} to {max_val:.4f}")
    print(f"  Mean: {mean_val:.4f}")
    
    # Interpret the scale
    if 0 <= max_val <= 1:
        print(f"  → Likely: PROPORTION (0-1) - multiply by 100 to get percentage")
    elif 0 <= max_val <= 100 and min_val >= 0:
        print(f"  → Likely: PERCENTAGE (0-100) - already in percentage form")
    elif -3 <= min_val <= 3 and -3 <= max_val <= 3:
        print(f"  → Likely: INDEX/SCORE (standardized scale)")
    elif max_val > 1000:
        print(f"  → Likely: ABSOLUTE VALUE (large numbers, possibly GDP/population)")
    else:
        print(f"  → Likely: MIXED SCALE or OTHER - check source documentation")
#interpolate 







