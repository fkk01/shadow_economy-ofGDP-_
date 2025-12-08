import pandas as pd
import numpy as np

# --- Load data ---
df = pd.read_excel('master_panel_.xlsx')

# --- Convert numeric columns ---
numeric_columns = [
    'Government_Effectiveness', 'Regulatory_Quality', 'GDP', 'Labour_Force_Total',
    'Tax_Revenue_GDP', 'Agriculture_GDP', 'Trade_Openness_GDP', 
    'Human_Digitalization_InternetUsers', 'Remittances_GDP', 'Inflation',
    'Unemployment_Rate', 'Rule_of_Law', 'Corruption_Perception_Index'
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("\nAFTER CORRECTION:")
print(df.dtypes)

# --- Compute Inflation Rate ---
df = df.sort_values(['Country', 'Year'])
if 'Inflation' in df.columns:
    df['Inflation_Rate'] = df.groupby('Country')['Inflation'].pct_change() * 100
    df = df.drop(columns=['Inflation'])
else:
    df['Inflation_Rate'] = np.nan
    print("Inflation column not found; Inflation_Rate set to NaN.")

# --- GDP per Worker ---
df['GDP_per_Worker'] = df['GDP'] / df['Labour_Force_Total']
df.loc[np.isinf(df['GDP_per_Worker']), 'GDP_per_Worker'] = np.nan
df = df.drop(columns=[col for col in ['GDP', 'Labour_Force_Total','ISO3'] if col in df.columns])

# --- Filter years and countries ---
turning_year = 2000
df = df[df["Year"] >= turning_year].reset_index(drop=True)

id_cols = ["Country", "Year"]
vars_cols = [c for c in df.columns if c not in id_cols]

# Country coverage
country_coverage = (
    df.groupby("Country")[vars_cols]
      .apply(lambda x: x.notna().sum().sum() / (x.shape[0] * x.shape[1]))
      .reset_index(name="Coverage")
)
country_threshold = 0.8
countries_keep = country_coverage[country_coverage["Coverage"] >= country_threshold]["Country"]
df = df[df["Country"].isin(countries_keep)].reset_index(drop=True)

# --- Interpolation 2001 for Government_Effectiveness (excluding Montenegro) ---
df_interp = df[df['Country'] != 'Montenegro'].copy()
montenegro_data = df[df['Country'] == 'Montenegro'].copy()

for country in df_interp['Country'].unique():
    country_data = df_interp[df_interp['Country'] == country]
    years = country_data['Year'].values
    if 2000 in years and 2002 in years:
        val_2000 = country_data[country_data['Year']==2000]['Government_Effectiveness'].values[0]
        val_2002 = country_data[country_data['Year']==2002]['Government_Effectiveness'].values[0]
        interp_value = val_2000 + (val_2002 - val_2000) / 2
        row_idx = df_interp[(df_interp['Country']==country) & (df_interp['Year']==2001)].index
        if len(row_idx) > 0:
            df_interp.loc[row_idx[0], 'Government_Effectiveness'] = interp_value
        else:
            new_row = country_data[country_data['Year']==2000].iloc[0].copy()
            new_row['Year'] = 2001
            new_row['Government_Effectiveness'] = interp_value
            numeric_cols_except_gov = [col for col in df_interp.columns if col not in ['Country','Year','Government_Effectiveness'] and pd.api.types.is_numeric_dtype(df_interp[col])]
            for col in numeric_cols_except_gov:
                new_row[col] = np.nan
            df_interp = pd.concat([df_interp, pd.DataFrame([new_row])], ignore_index=True)

df = pd.concat([df_interp, montenegro_data], ignore_index=True)
df = df.sort_values(['Country','Year']).reset_index(drop=True)

# --- Estimate 2024 using average annual change ---
df_2024 = df[df['Country'] != 'Montenegro'].copy()
montenegro_data = df[df['Country'] == 'Montenegro'].copy()

for country in df_2024['Country'].unique():
    country_data = df_2024[df_2024['Country']==country]
    ge_series = country_data[country_data['Year'].between(2000,2023)]['Government_Effectiveness']
    if ge_series.isna().sum() <= 5:
        ge_2000, ge_2023 = ge_series.iloc[0], ge_series.iloc[-1]
        avg_change = (ge_2023 - ge_2000)/23
        ge_2024 = ge_2023 + avg_change
        row_idx = df_2024[(df_2024['Country']==country) & (df_2024['Year']==2024)].index
        if len(row_idx) > 0:
            df_2024.loc[row_idx[0],'Government_Effectiveness'] = ge_2024
        else:
            new_row = country_data[country_data['Year']==2023].iloc[0].copy()
            new_row['Year'] = 2024
            new_row['Government_Effectiveness'] = ge_2024
            df_2024 = pd.concat([df_2024,pd.DataFrame([new_row])], ignore_index=True)

df = pd.concat([df_2024, montenegro_data], ignore_index=True)
df = df.sort_values(['Country','Year']).reset_index(drop=True)

# --- Save final dataset ---
df.to_excel("master_panel_final_2000.xlsx", index=False)
print("\n✓ Final dataset saved: master_panel_final_2000.xlsx")

# --- Analyze numeric scales ---
final_numeric_cols = [col for col in df.columns if col not in id_cols and pd.api.types.is_numeric_dtype(df[col])]
for col in sorted(final_numeric_cols):
    min_val, max_val, mean_val = df[col].min(), df[col].max(), df[col].mean()
    print(f"\n{col}: Range={min_val:.4f} to {max_val:.4f}, Mean={mean_val:.4f}")
    if 0 <= max_val <= 1:
        print("  → PROPORTION (0-1)")
    elif 0 <= max_val <= 100 and min_val >=0:
        print("  → PERCENTAGE (0-100)")
    elif -3 <= min_val <= 3 and -3 <= max_val <= 3:
        print("  → INDEX/SCORE")
    elif max_val > 1000:
        print("  → ABSOLUTE VALUE")
    else:
        print("  → MIXED SCALE / check source")









