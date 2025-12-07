import pandas as pd
import numpy as np

# Read the merged panel data
df = pd.read_excel('master_panel_merged.xlsx')

# Convert numeric columns
numeric_columns = [
    'Government_Effectiveness',
    'Regulatory_Quality',
    'GDP',
    'Labour_Force_Participation_Rate',
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

# Create GDP per Worker using Labour_Force_Participation_Rate as denominator
df['GDP_per_Worker'] = df['GDP'] / df['Labour_Force_Participation_Rate']
df.loc[np.isinf(df['GDP_per_Worker']), 'GDP_per_Worker'] = np.nan

# Drop GDP and Labour_Force_Participation_Rate columns
columns_to_drop = ['GDP', 'Labour_Force_Participation_Rate','ISO3']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Final dataset
df.to_excel('master_panel_corrected.xlsx', index=False)
print("✓ Final dataset saved: master_panel_corrected.xlsx")
