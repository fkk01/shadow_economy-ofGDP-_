import pandas as pd
import numpy as np

# Read the merged panel data
df = pd.read_excel('master_panel_merged.xlsx')

# Display specific columns
print("="*80)
print("PREVIEW OF KEY VARIABLES")
print("="*80)
specific_cols = ['Country', 'Year', 'Inflation', 'Tax_Revenue_GDP', 'Agriculture_GDP', 'Trade_Openness_GDP','Government_Effectiveness']
print(df[specific_cols].head(20))
print(f"\n{df[specific_cols].info()}")

print("\n" + "="*80)
print("PANEL DATA SUMMARY REPORT")
print("="*80)

# Basic information
print(f"\nTotal observations: {len(df)}")
print(f"Number of countries: {df['Country'].nunique()}")
print(f"Number of years: {df['Year'].nunique()}")
print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")

# List countries
print(f"\nCountries in dataset:")
countries = sorted(df['Country'].unique())
for i, country in enumerate(countries, 1):
    print(f"  {i:2d}. {country}")

# List years
print(f"\nYears covered:")
years = sorted(df['Year'].unique())
print(f"  {', '.join(map(str, years))}")

# Variable analysis
print("\n" + "="*80)
print("VARIABLE COVERAGE ANALYSIS")
print("="*80)

# Get all variables (excluding Country, ISO3, Year)
id_cols = ['Country', 'ISO3', 'Year']
variables = [col for col in df.columns if col not in id_cols]

print(f"\nTotal variables: {len(variables)}")
print(f"\nData availability by variable:")
print(f"{'Variable':<45} {'Non-Missing':<12} {'Missing':<12} {'Coverage %':<12}")
print("-"*80)

coverage_data = []
for var in variables:
    non_missing = df[var].notna().sum()
    missing = df[var].isna().sum()
    coverage_pct = (non_missing / len(df)) * 100
    coverage_data.append({
        'Variable': var,
        'Non_Missing': non_missing,
        'Missing': missing,
        'Coverage': coverage_pct
    })
    print(f"{var:<45} {non_missing:<12} {missing:<12} {coverage_pct:>10.1f}%")

# Summary statistics
print("\n" + "="*80)
print("COVERAGE SUMMARY")
print("="*80)

coverage_df = pd.DataFrame(coverage_data)
print(f"\nVariables with 100% coverage: {(coverage_df['Coverage'] == 100).sum()}")
print(f"Variables with 75-99% coverage: {((coverage_df['Coverage'] >= 75) & (coverage_df['Coverage'] < 100)).sum()}")
print(f"Variables with 50-74% coverage: {((coverage_df['Coverage'] >= 50) & (coverage_df['Coverage'] < 75)).sum()}")
print(f"Variables with 25-49% coverage: {((coverage_df['Coverage'] >= 25) & (coverage_df['Coverage'] < 50)).sum()}")
print(f"Variables with <25% coverage: {(coverage_df['Coverage'] < 25).sum()}")
print(f"Variables with 0% coverage (completely missing): {(coverage_df['Coverage'] == 0).sum()}")

# Identify completely missing variables
completely_missing = coverage_df[coverage_df['Coverage'] == 0]['Variable'].tolist()
if completely_missing:
    print(f"\nCompletely missing variables:")
    for var in completely_missing:
        print(f"  - {var}")

# Identify variables with low coverage
low_coverage = coverage_df[coverage_df['Coverage'] < 25]['Variable'].tolist()
if low_coverage and set(low_coverage) != set(completely_missing):
    print(f"\nVariables with very low coverage (<25%):")
    for var in low_coverage:
        if var not in completely_missing:
            pct = coverage_df[coverage_df['Variable'] == var]['Coverage'].iloc[0]
            print(f"  - {var}: {pct:.1f}%")

# Country-level coverage
print("\n" + "="*80)
print("COUNTRY-LEVEL COVERAGE")
print("="*80)

print(f"\nData completeness by country (% of variables with data):")
print(f"{'Country':<30} {'Complete Vars':<15} {'Coverage %':<12}")
print("-"*60)

country_coverage = []
for country in sorted(df['Country'].unique()):
    country_data = df[df['Country'] == country]
    complete_vars = 0
    for var in variables:
        if country_data[var].notna().any():
            complete_vars += 1
    coverage_pct = (complete_vars / len(variables)) * 100
    country_coverage.append({
        'Country': country,
        'Complete_Vars': complete_vars,
        'Coverage': coverage_pct
    })
    print(f"{country:<30} {complete_vars}/{len(variables):<14} {coverage_pct:>10.1f}%")

# Year-level coverage
print("\n" + "="*80)
print("YEAR-LEVEL COVERAGE")
print("="*80)

print(f"\nData completeness by year (average % coverage across all variables):")
print(f"{'Year':<10} {'Avg Coverage %':<15} {'Non-Missing Obs':<20}")
print("-"*50)

for year in sorted(df['Year'].unique()):
    year_data = df[df['Year'] == year]
    total_cells = len(year_data) * len(variables)
    non_missing_cells = year_data[variables].notna().sum().sum()
    coverage_pct = (non_missing_cells / total_cells) * 100
    print(f"{year:<10} {coverage_pct:>12.1f}%   {non_missing_cells}/{total_cells}")

# Missing data patterns
print("\n" + "="*80)
print("MISSING DATA PATTERNS")
print("="*80)

# Countries with most complete data
country_cov_df = pd.DataFrame(country_coverage).sort_values('Coverage', ascending=False)
print(f"\nTop 5 countries with most complete data:")
for i, row in country_cov_df.head(5).iterrows():
    print(f"  {row['Country']:<30} {row['Coverage']:.1f}%")

print(f"\nBottom 5 countries with least complete data:")
for i, row in country_cov_df.tail(5).iterrows():
    print(f"  {row['Country']:<30} {row['Coverage']:.1f}%")

# Observations with complete data
complete_obs = df[variables].notna().all(axis=1).sum()
print(f"\n" + "="*80)
print(f"Observations with ALL variables present: {complete_obs} ({(complete_obs/len(df))*100:.1f}%)")
print(f"Observations with at least ONE missing variable: {len(df) - complete_obs} ({((len(df)-complete_obs)/len(df))*100:.1f}%)")
print("="*80)