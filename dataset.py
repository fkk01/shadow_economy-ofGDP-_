import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# --- 1. Define countries and ISO3 codes ---
countries = [
    ("Albania", "ALB"),
    ("Belarus", "BLR"),
    ("Bosnia and Herzegovina", "BIH"),
    ("Bulgaria", "BGR"),
    ("Croatia", "HRV"),
    ("Cyprus", "CYP"),
    ("Czech Republic", "CZE"),
    ("Estonia", "EST"),
    ("Greece", "GRC"),
    ("Hungary", "HUN"),
    ("Kosovo", "XKX"),
    ("Latvia", "LVA"),
    ("Lithuania", "LTU"),
    ("Moldova", "MDA"),
    ("Montenegro", "MNE"),
    ("North Macedonia", "MKD"),
    ("Poland", "POL"),
    ("Romania", "ROU"),
    ("Serbia", "SRB"),
    ("Slovakia", "SVK"),
    ("Slovenia", "SVN"),
]

countries_mapping = {name: iso3 for name, iso3 in countries}
country_list = [name for name, _ in countries]
iso3_list = [iso3 for _, iso3 in countries]
years = list(range(1995, 2025))

# --- Country name variations assumption ---
# Data files may contain different country name variations.
# This mapping normalizes all variations to our standard country names.
COUNTRY_NAME_VARIATIONS = {
    # Czech variations: Both "Czechia" and "Czech Republic" -> "Czech Republic"
    'czechia': 'Czech Republic',
    'czech republic': 'Czech Republic',
    # Slovakia variations: Both "Slovakia" and "Slovak Republic" -> "Slovakia"
    'slovakia': 'Slovakia',
    'slovak republic': 'Slovakia',
}

# --- 2. Create master panel ---
rows = []
for name, iso3 in countries:
    for year in years:
        rows.append({"Country": name, "ISO3": iso3, "Year": year})

master = pd.DataFrame(rows)

# --- 3. Add empty columns for all variables ---
cols = [
    "Government_Effectiveness",
    "Regulatory_Quality",
    "GDP",
    "Labour_Force_Total",
    "Tax_Revenue_GDP",
    "Agriculture_GDP",
    "Trade_Openness_GDP",
    "Human_Digitalization_InternetUsers",
    "Remittances_GDP",
    "Inflation",
    "Unemployment_Rate",
    "Rule_of_Law",
    "Corruption_Perception_Index",
]

for c in cols:
    master[c] = pd.NA

# Define file path prefix
FILE_PATH_PREFIX = 'data/cause variables/'

print("Starting data merging process...")
print(f"Master panel shape: {master.shape}\n")

# ============================================
# Helper functions
# ============================================
def normalize_country_names(df, country_col):
    """Normalize country names from data files to match our country list.
    Uses COUNTRY_NAME_VARIATIONS assumption defined at the top of the file.
    """
    # Strip whitespace first
    df[country_col] = df[country_col].str.strip()
    
    # Map country names using the variations assumption
    def map_country(name):
        if pd.isna(name):
            return name
        name_str = str(name).strip()
        name_lower = name_str.lower()
        
        # Remove common suffixes like ", Republic of"
        name_clean = name_str
        if ', republic of' in name_lower:
            name_clean = name_str.split(',')[0].strip()
            name_lower = name_clean.lower()
        
        # Check if this country name has a variation mapping
        if name_lower in COUNTRY_NAME_VARIATIONS:
            return COUNTRY_NAME_VARIATIONS[name_lower]
        else:
            # Return cleaned name (capitalized properly)
            return name_clean
    
    df[country_col] = df[country_col].apply(map_country)
    return df

def safe_merge(master_df, new_data, column_name, on_cols=['Country', 'Year']):
    """Safely merge new data into master dataframe"""
    if new_data.empty:
        print(f"  ⚠ Warning: No data to merge for {column_name}")
        return master_df
    
    # Remove any existing column with _new suffix
    if f"{column_name}_new" in master_df.columns:
        master_df = master_df.drop(columns=[f"{column_name}_new"])
    
    master_df = master_df.merge(new_data, on=on_cols, how='left', suffixes=('', '_new'))
    
    if f"{column_name}_new" in master_df.columns:
        master_df[column_name] = master_df[f"{column_name}_new"].combine_first(master_df[column_name])
        master_df = master_df.drop(columns=[f"{column_name}_new"])
    
    non_null = master_df[column_name].notna().sum()
    print(f"  ✓ Merged {column_name}: {non_null} non-null values")
    return master_df

# ============================================
# 1. AGRICULTURE (% of GDP)
# ============================================
print("1. Processing Agriculture data...")
try:
    agr = pd.read_excel(FILE_PATH_PREFIX + 'Agriculture(%of GDP).xlsx', sheet_name=0)
    agr.columns = agr.columns.str.strip()
    
    # Filter for Agriculture series only
    agr = agr[agr['Series Name'].str.contains('Agriculture', case=False, na=False)]
    
    # Get year columns (format: "1995 [YR1995]")
    year_cols = [col for col in agr.columns if '[YR' in str(col)]
    
    # Melt to long format
    agr_long = agr.melt(id_vars=['Country Name'], value_vars=year_cols, 
                        var_name='Year_Col', value_name='Agriculture_GDP')
    
    # Extract year from column name (e.g., "1995 [YR1995]" -> 1995)
    agr_long['Year'] = agr_long['Year_Col'].str.extract(r'(\d{4})')[0].astype(int)
    agr_long = agr_long.drop(columns=['Year_Col'])
    agr_long = agr_long.rename(columns={'Country Name': 'Country'})
    
    # Filter and clean
    agr_long['Agriculture_GDP'] = pd.to_numeric(agr_long['Agriculture_GDP'], errors='coerce')
    agr_long = agr_long[agr_long['Country'].isin(country_list) & agr_long['Year'].isin(years)]
    agr_long = agr_long.dropna(subset=['Agriculture_GDP'])
    agr_long = agr_long.drop_duplicates(subset=['Country', 'Year'], keep='last')
    
    master = safe_merge(master, agr_long[['Country', 'Year', 'Agriculture_GDP']], 'Agriculture_GDP')
    
except Exception as e:
    print(f"  ✗ Error: {e}")

# ============================================
# 2. CPI - Inflation (Consumer Price Index)
# ============================================
print("\n2. Processing CPI (Inflation/Consumer Price Index) data...")
try:
    cpi_inflation = pd.read_csv(FILE_PATH_PREFIX + 'CPI.csv')
    print(f"  Columns: {cpi_inflation.columns.tolist()[:10]}")
    
    # Strip whitespace from column names
    cpi_inflation.columns = cpi_inflation.columns.str.strip()
    # Find country column (could be 'COUNTRY', 'Country', etc.)
    country_col = None
    for col in cpi_inflation.columns:
        col_lower = str(col).lower()
        if col_lower == 'country':
            country_col = col
            break
    
    if not country_col:
        print("  ⚠ Could not find country column")
    else:
        print(f"  Using country column: {country_col}")
        
        # Identify year columns (simple 4-digit years like "1990", "1991", etc.)
        year_cols = []
        id_vars = [country_col]
        
        # Keep other metadata columns as ID vars if they exist (to preserve them during melt)
        metadata_cols = ['DATASET', 'SERIES_CODE', 'OBS_MEASURE', 'INDEX_TYPE', 
                        'COICOP_1999', 'TYPE_OF_TRANSFORMATION', 'FREQUENCY', 'SCALE']
        for col in metadata_cols:
            if col in cpi_inflation.columns:
                id_vars.append(col)
        
        # Identify year columns - simple 4-digit years
        for col in cpi_inflation.columns:
            if col == country_col or col in id_vars:
                continue
            # Check if column is a 4-digit year
            col_str = str(col).strip()
            if col_str.isdigit() and len(col_str) == 4:
                year_val = int(col_str)
                if year_val in years:
                    year_cols.append(col)
        
        if not year_cols:
            print("  ⚠ Could not find year columns. Attempting to melt all non-ID columns...")
            # Fallback: melt all columns except ID columns
            cpi_long = cpi_inflation.melt(id_vars=id_vars, var_name='Year', value_name='Inflation')
            cpi_long['Year'] = pd.to_numeric(cpi_long['Year'], errors='coerce')
        else:
            print(f"  Found {len(year_cols)} year columns")
            
            # Melt the dataframe
            cpi_long = cpi_inflation.melt(id_vars=id_vars, value_vars=year_cols, var_name='Year', value_name='Inflation')
            cpi_long['Year'] = pd.to_numeric(cpi_long['Year'], errors='coerce')
        
        # Rename country column and normalize country names
        cpi_long = cpi_long.rename(columns={country_col: 'Country'})
        cpi_long = normalize_country_names(cpi_long, 'Country')
        
        # Filter for our countries and years
        cpi_long = cpi_long[
            cpi_long['Country'].isin(country_list) & 
            cpi_long['Year'].isin(years)
        ].copy()
        
        # Remove duplicates and convert to numeric (handle missing values)
        cpi_long['Inflation'] = pd.to_numeric(cpi_long['Inflation'], errors='coerce')
        cpi_long = cpi_long.dropna(subset=['Country', 'Year', 'Inflation'])
        cpi_long = cpi_long.drop_duplicates(subset=['Country', 'Year'], keep='last')
        
        if not cpi_long.empty:
            master = safe_merge(master, cpi_long[['Country', 'Year', 'Inflation']], 'Inflation')
        else:
            print("  ⚠ No CPI/Inflation data found after filtering")
            
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================
# 3. GDP (constant US$)
# ============================================
print("\n3. Processing GDP data...")
try:
    # Try different file name variations
    try:
        gdp = pd.read_excel(FILE_PATH_PREFIX + 'GDP costant (US$).xlsx', sheet_name=0)
    except:
        gdp = pd.read_excel(FILE_PATH_PREFIX + 'GDP constant (US$).xlsx', sheet_name=0)
    
    print(f"  Columns: {gdp.columns.tolist()[:8]}")
    
    # Find country column (could be 'Country Name', 'Country', etc.)
    country_col = None
    for col in gdp.columns:
        col_lower = str(col).lower()
        if 'country' in col_lower and 'name' in col_lower:
            country_col = col
            break
        elif 'country' in col_lower and country_col is None:
            country_col = col
    
    if not country_col:
        print("  ⚠ Could not find country column")
    else:
        print(f"  Using country column: {country_col}")
        
        # Identify year columns (columns that start with year numbers like "1990 [YR1990]")
        year_cols = []
        id_vars = [country_col]
        
        # Check if there's a Series Name column (might need to filter)
        series_col = None
        for col in gdp.columns:
            if 'series' in str(col).lower() and 'name' in str(col).lower():
                series_col = col
                id_vars.append(series_col)
                break
        
        # Identify year columns - they contain year numbers like "1990 [YR1990]"
        for col in gdp.columns:
            if col == country_col or (series_col and col == series_col):
                continue
            # Try to extract year from column name
            col_str = str(col)
            # Look for 4-digit year at the start
            year_match = None
            if len(col_str) >= 4 and col_str[:4].isdigit():
                year_match = int(col_str[:4])
            elif '[' in col_str:
                # Try to extract from format like "1990 [YR1990]"
                parts = col_str.split('[')
                if parts[0].strip()[:4].isdigit():
                    year_match = int(parts[0].strip()[:4])
            
            if year_match and year_match in years:
                year_cols.append(col)
        
        if not year_cols:
            print("  ⚠ Could not find year columns. Attempting to melt all non-ID columns...")
            # Fallback: melt all columns except ID columns
            gdp_long = gdp.melt(id_vars=id_vars, var_name='Year_Col', value_name='GDP')
            # Try to extract year from Year_Col
            gdp_long['Year'] = gdp_long['Year_Col'].astype(str).str.extract(r'(\d{4})')[0]
            gdp_long['Year'] = pd.to_numeric(gdp_long['Year'], errors='coerce')
            gdp_long = gdp_long.drop(columns=['Year_Col'])
        else:
            print(f"  Found {len(year_cols)} year columns")
            
            # Melt the dataframe
            gdp_long = gdp.melt(id_vars=id_vars, value_vars=year_cols, var_name='Year_Col', value_name='GDP')
            
            # Extract year number from column names (e.g., "1990 [YR1990]" -> 1990)
            gdp_long['Year'] = gdp_long['Year_Col'].astype(str).str.extract(r'(\d{4})')[0]
            gdp_long['Year'] = pd.to_numeric(gdp_long['Year'], errors='coerce')
            gdp_long = gdp_long.drop(columns=['Year_Col'])
        
        # Rename country column and normalize country names
        gdp_long = gdp_long.rename(columns={country_col: 'Country'})
        gdp_long = normalize_country_names(gdp_long, 'Country')
        
        # Filter for our countries and years
        gdp_long = gdp_long[
            gdp_long['Country'].isin(country_list) & 
            gdp_long['Year'].isin(years)
        ].copy()
        
        # If there's a Series Name column, filter for GDP series only
        if series_col and series_col in gdp_long.columns:
            # Keep only GDP-related series
            gdp_series = gdp_long[gdp_long[series_col].str.contains('GDP', case=False, na=False)].copy()
            if not gdp_series.empty:
                gdp_long = gdp_series
            gdp_long = gdp_long.drop(columns=[series_col])
        
        # Remove duplicates and convert GDP to numeric (handle ".." as NaN)
        gdp_long['GDP'] = pd.to_numeric(gdp_long['GDP'], errors='coerce')
        gdp_long = gdp_long.dropna(subset=['Country', 'Year', 'GDP'])
        gdp_long = gdp_long.drop_duplicates(subset=['Country', 'Year'], keep='last')
        
        if not gdp_long.empty:
            master = safe_merge(master, gdp_long[['Country', 'Year', 'GDP']], 'GDP')
        else:
            print("  ⚠ No GDP data found after filtering")
            
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================
# 4. INTERNET USERS
# ============================================
print("\n4. Processing Internet Users data...")
try:
    # Try both file extensions
    try:
        internet = pd.read_excel(FILE_PATH_PREFIX + 'Individuals using the Internet (% of population).xlsx', 
                                sheet_name=0, skiprows=3)
    except:
        internet = pd.read_excel(FILE_PATH_PREFIX + 'Individuals using the Internet.xls', 
                                sheet_name=0, skiprows=3)
    
    country_col = None
    for col in internet.columns:
        if 'country' in str(col).lower():
            country_col = col
            break
    
    if country_col:
        internet_long = internet.melt(id_vars=[country_col], var_name='Year', 
                                     value_name='Human_Digitalization_InternetUsers')
        internet_long = internet_long.rename(columns={country_col: 'Country'})
        internet_long = normalize_country_names(internet_long, 'Country')
        internet_long['Year'] = pd.to_numeric(internet_long['Year'], errors='coerce')
        internet_long = internet_long[internet_long['Country'].isin(country_list) & 
                                     internet_long['Year'].isin(years)]
        master = safe_merge(master, internet_long, 'Human_Digitalization_InternetUsers')
except Exception as e:
    print(f"  ✗ Error: {e}")

# ============================================
# 5. LABOR FORCE
# ============================================
print("\n5. Processing Labor Force data...")
try:
    # Try different file name variations and sheet names
    labor = None
    for filename in ['Labor Force.xlsx', 'Labor_Force.xlsx', 'Labor Force.xls']:
        for sheet_name in ['Data', 0]:
            try:
                labor = pd.read_excel(FILE_PATH_PREFIX + filename, sheet_name=sheet_name)
                print(f"  Loaded from: {filename}, sheet: {sheet_name}")
                break
            except:
                continue
        if labor is not None:
            break
    
    if labor is None:
        print("  ⚠ Could not load Labor Force file")
    else:
        print(f"  Columns: {labor.columns.tolist()[:8]}")
        
        # Find country column (could be 'Country Name', 'Country', etc.)
        country_col = None
        for col in labor.columns:
            col_lower = str(col).lower()
            if 'country' in col_lower and 'name' in col_lower:
                country_col = col
                break
            elif 'country' in col_lower and country_col is None:
                country_col = col
        
        if not country_col:
            print("  ⚠ Could not find country column")
        else:
            print(f"  Using country column: {country_col}")
            
            # Identify year columns (columns that start with year numbers like "1990 [YR1990]")
            year_cols = []
            id_vars = [country_col]
            
            # Check if there's a Series Name column (might need to filter)
            series_col = None
            for col in labor.columns:
                if 'series' in str(col).lower() and 'name' in str(col).lower():
                    series_col = col
                    id_vars.append(series_col)
                    break
            
            # Identify year columns - they contain year numbers like "1990 [YR1990]"
            for col in labor.columns:
                if col == country_col or (series_col and col == series_col):
                    continue
                # Try to extract year from column name
                col_str = str(col)
                # Look for 4-digit year at the start
                year_match = None
                if len(col_str) >= 4 and col_str[:4].isdigit():
                    year_match = int(col_str[:4])
                elif '[' in col_str:
                    # Try to extract from format like "1990 [YR1990]"
                    parts = col_str.split('[')
                    if parts[0].strip()[:4].isdigit():
                        year_match = int(parts[0].strip()[:4])
                
                if year_match and year_match in years:
                    year_cols.append(col)
            
            if not year_cols:
                print("  ⚠ Could not find year columns. Attempting to melt all non-ID columns...")
                # Fallback: melt all columns except ID columns
                labor_long = labor.melt(id_vars=id_vars, var_name='Year_Col', value_name='Labour_Force_Participation_Rate')
                # Try to extract year from Year_Col
                labor_long['Year'] = labor_long['Year_Col'].astype(str).str.extract(r'(\d{4})')[0]
                labor_long['Year'] = pd.to_numeric(labor_long['Year'], errors='coerce')
                labor_long = labor_long.drop(columns=['Year_Col'])
            else:
                print(f"  Found {len(year_cols)} year columns")
                
                # Melt the dataframe
                labor_long = labor.melt(id_vars=id_vars, value_vars=year_cols, var_name='Year_Col', value_name='Labour_Force_Participation_Rate')
                
                # Extract year number from column names (e.g., "1990 [YR1990]" -> 1990)
                labor_long['Year'] = labor_long['Year_Col'].astype(str).str.extract(r'(\d{4})')[0]
                labor_long['Year'] = pd.to_numeric(labor_long['Year'], errors='coerce')
                labor_long = labor_long.drop(columns=['Year_Col'])
            
            # Rename country column and normalize country names
            labor_long = labor_long.rename(columns={country_col: 'Country'})
            labor_long = normalize_country_names(labor_long, 'Country')
            
            # Filter for our countries and years
            labor_long = labor_long[
                labor_long['Country'].isin(country_list) & 
                labor_long['Year'].isin(years)
            ].copy()
            
            # If there's a Series Name column, filter for Labor Force series only
            if series_col and series_col in labor_long.columns:
                # Keep only Labor Force-related series
                labor_series = labor_long[labor_long[series_col].str.contains('Labor|Labour', case=False, na=False)].copy()
                if not labor_series.empty:
                    labor_long = labor_series
                labor_long = labor_long.drop(columns=[series_col])
            
            # Remove duplicates and convert to numeric (handle ".." as NaN)
            labor_long['Labour_Force_Total'] = pd.to_numeric(labor_long['Labour_Force_Total'], errors='coerce')
            labor_long = labor_long.dropna(subset=['Country', 'Year', 'Labour_Force_Total'])
            labor_long = labor_long.drop_duplicates(subset=['Country', 'Year'], keep='last')
            
            if not labor_long.empty:
                master = safe_merge(master, labor_long[['Country', 'Year', 'Labour_Force_Total']], 'Labour_Force_Total')
            else:
                print("  ⚠ No Labor Force data found after filtering")
                
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================
# 6. REMITTANCES
# ============================================
print("\n6. Processing Remittances data...")
try:
    remit = pd.read_excel(FILE_PATH_PREFIX + 'remittances.xls', sheet_name='Data', skiprows=3)
    
    country_col = None
    for col in remit.columns:
        if 'country' in str(col).lower():
            country_col = col
            break
    
    if country_col:
        remit_long = remit.melt(id_vars=[country_col], var_name='Year', value_name='Remittances_GDP')
        remit_long = remit_long.rename(columns={country_col: 'Country'})
        remit_long = normalize_country_names(remit_long, 'Country')
        remit_long['Year'] = pd.to_numeric(remit_long['Year'], errors='coerce')
        remit_long = remit_long[remit_long['Country'].isin(country_list) & 
                               remit_long['Year'].isin(years)]
        master = safe_merge(master, remit_long, 'Remittances_GDP')
except Exception as e:
    print(f"  ✗ Error: {e}")
# ============================================
# 7. TAX REVENUE
# ============================================
print("\n7. Processing Tax Revenue data...")
try:
    tax = pd.read_excel(FILE_PATH_PREFIX + 'tax_revenue%GDP.xls', sheet_name='Data', skiprows=3)
    
    country_col = None
    for col in tax.columns:
        if 'country' in str(col).lower():
            country_col = col
            break
    
    if country_col:
        tax_long = tax.melt(id_vars=[country_col], var_name='Year', value_name='Tax_Revenue_GDP')
        tax_long = tax_long.rename(columns={country_col: 'Country'})
        tax_long = normalize_country_names(tax_long, 'Country')
        tax_long['Year'] = pd.to_numeric(tax_long['Year'], errors='coerce')
        tax_long = tax_long[tax_long['Country'].isin(country_list) & 
                           tax_long['Year'].isin(years)]
        master = safe_merge(master, tax_long, 'Tax_Revenue_GDP')
except Exception as e:
    print(f"  ✗ Error: {e}")

# ============================================
# 8. TRADE OPENNESS
# ============================================
print("\n8. Processing Trade Openness data...")
try:
    # Try both extensions
    try:
        trade = pd.read_excel(FILE_PATH_PREFIX + 'Trade Openness (%of GDP).xlsx', 
                             sheet_name=0, skiprows=3)
    except:
        trade = pd.read_excel(FILE_PATH_PREFIX + 'Trade Openness (%of GDP).xls', 
                             sheet_name=0, skiprows=3)
    
    country_col = None
    for col in trade.columns:
        if 'country' in str(col).lower():
            country_col = col
            break
    
    if country_col:
        trade_long = trade.melt(id_vars=[country_col], var_name='Year', value_name='Trade_Openness_GDP')
        trade_long = trade_long.rename(columns={country_col: 'Country'})
        trade_long = normalize_country_names(trade_long, 'Country')
        trade_long['Year'] = pd.to_numeric(trade_long['Year'], errors='coerce')
        trade_long = trade_long[trade_long['Country'].isin(country_list) & 
                               trade_long['Year'].isin(years)]
        master = safe_merge(master, trade_long, 'Trade_Openness_GDP')
except Exception as e:
    print(f"  ✗ Error: {e}")

# ============================================
# 9. UNEMPLOYMENT
# ============================================
print("\n9. Processing Unemployment data...")
try:
    unemp = pd.read_excel(FILE_PATH_PREFIX + 'Unemployment, total (% of total labor force).xls', 
                         sheet_name='Data', skiprows=3)
    
    country_col = None
    for col in unemp.columns:
        if 'country' in str(col).lower():
            country_col = col
            break
    
    if country_col:
        unemp_long = unemp.melt(id_vars=[country_col], var_name='Year', value_name='Unemployment_Rate')
        unemp_long = unemp_long.rename(columns={country_col: 'Country'})
        unemp_long = normalize_country_names(unemp_long, 'Country')
        unemp_long['Year'] = pd.to_numeric(unemp_long['Year'], errors='coerce')
        unemp_long = unemp_long[unemp_long['Country'].isin(country_list) & 
                               unemp_long['Year'].isin(years)]
        master = safe_merge(master, unemp_long, 'Unemployment_Rate')
except Exception as e:
    print(f"  ✗ Error: {e}")

# ============================================
# 10. GOVERNMENT EFFECTIVENESS & REGULATORY QUALITY (WGI)
# ============================================
print("\n10. Processing WGI (Government Effectiveness & Regulatory Quality & Rule of Law) data...")
try:
    wgi = pd.read_excel(FILE_PATH_PREFIX + 'wgidataset.xlsx', sheet_name=0)
    wgi.columns = wgi.columns.str.strip()
    
    # Check and map column names (WGI dataset uses 'countryname' and 'year')
    country_col = None
    year_col = None
    indicator_col = None
    estimate_col = None
    
    for col in wgi.columns:
        col_lower = str(col).lower()
        if 'country' in col_lower:
            country_col = col
        elif col_lower == 'year':
            year_col = col
        elif 'indicator' in col_lower:
            indicator_col = col
        elif 'estimate' in col_lower:
            estimate_col = col
    
    if not all([country_col, year_col, indicator_col, estimate_col]):
        print(f"  ⚠ Warning: Could not find all required columns.")
        print(f"    Found: country={country_col}, year={year_col}, indicator={indicator_col}, estimate={estimate_col}")
        print(f"    Available columns: {wgi.columns.tolist()}")
    else:
        # Rename columns for consistency
        wgi_clean = wgi.rename(columns={
            country_col: 'Country',
            year_col: 'Year',
            indicator_col: 'indicator',
            estimate_col: 'estimate'
        })
        
        # Normalize country names using the standard function
        wgi_clean = normalize_country_names(wgi_clean, 'Country')
        
        # Additional WGI-specific mappings if needed
        wgi_clean['Country'] = wgi_clean['Country'].replace({
            'Macedonia, FYR': 'North Macedonia',
        })
        
        # Show unique countries in WGI dataset for debugging
        wgi_countries = wgi_clean['Country'].unique()
        print(f"  Found {len(wgi_countries)} unique countries in WGI dataset")
        print(f"  Sample WGI countries: {sorted(wgi_countries)[:10]}")
        
        # Check which of our countries are in WGI
        matching_countries = set(wgi_clean['Country'].unique()) & set(country_list)
        print(f"  Matching countries: {len(matching_countries)}/{len(country_list)}")
        if len(matching_countries) < len(country_list):
            missing = set(country_list) - matching_countries
            print(f"  ⚠ Missing countries in WGI: {sorted(missing)[:5]}")
        
        # Filter for our countries and years
        wgi_filtered = wgi_clean[
            wgi_clean['Country'].isin(country_list) & 
            wgi_clean['Year'].isin(years)
        ].copy()
        
        print(f"  Filtered rows: {len(wgi_filtered)}")
        print(f"  Unique indicators: {wgi_filtered['indicator'].unique()}")
        
        # Extract Government Effectiveness (ge)
        wgi_ge = wgi_filtered[wgi_filtered['indicator'].str.lower().str.strip() == 'ge'].copy()
        print(f"  GE rows found: {len(wgi_ge)}")
        if not wgi_ge.empty:
            wgi_ge = wgi_ge[['Country', 'Year', 'estimate']].copy()
            wgi_ge = wgi_ge.rename(columns={'estimate': 'Government_Effectiveness'})
            # Remove any duplicates
            wgi_ge = wgi_ge.drop_duplicates(subset=['Country', 'Year'], keep='last')
            master = safe_merge(master, wgi_ge, 'Government_Effectiveness')
        else:
            print("  ⚠ No Government Effectiveness (ge) data found")
            # Debug: show what indicators we have
            unique_indicators = wgi_filtered['indicator'].unique()
            print(f"    Available indicators: {unique_indicators}")
        
        # Extract Regulatory Quality (rq)
        wgi_rq = wgi_filtered[wgi_filtered['indicator'].str.lower().str.strip() == 'rl'].copy()
        print(f"  RQ rows found: {len(wgi_rq)}")
        if not wgi_rq.empty:
            wgi_rq = wgi_rq[['Country', 'Year', 'estimate']].copy()
            wgi_rq = wgi_rq.rename(columns={'estimate': 'Regulatory_Quality'})
            # Remove any duplicates
            wgi_rq = wgi_rq.drop_duplicates(subset=['Country', 'Year'], keep='last')
            master = safe_merge(master, wgi_rq, 'Regulatory_Quality')
        else:
            print("  ⚠ No Regulatory Quality (rl) data found")
            # Debug: show what indicators we have
            unique_indicators = wgi_filtered['indicator'].unique()
            print(f"    Available indicators: {unique_indicators}")

         # Extract Rule of Law (rl)
        wgi_rl = wgi_filtered[wgi_filtered['indicator'].str.lower().str.strip() == 'rl'].copy()
        print(f"  RL rows found: {len(wgi_rl)}")
        if not wgi_rl.empty:
            wgi_rl = wgi_rl[['Country', 'Year', 'estimate']].copy()
            wgi_rl = wgi_rl.rename(columns={'estimate': 'Rule_of_Law'})
            # Remove any duplicates
            wgi_rl = wgi_rl.drop_duplicates(subset=['Country', 'Year'], keep='last')
            master = safe_merge(master, wgi_rl, 'Rule_of_Law')
        else:
            print("  ⚠ No Rule of Lae (rl) data found")
            # Debug: show what indicators we have
            unique_indicators = wgi_filtered['indicator'].unique()
            print(f"    Available indicators: {unique_indicators}")

            
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
#Corruption Perception Index 
print("\n11. Processing CPI (Corruption Perception Index) data...")
# Read 2012-2024 data (sheet index 3)
cpi_main = pd.read_excel(FILE_PATH_PREFIX + 'Corruption_Perception_index.xlsx', sheet_name=3)
cpi_main.columns = cpi_main.columns.str.strip()
country_col = 'Country / Territory' if 'Country / Territory' in cpi_main.columns else 'Country'
cpi_score_col = [col for col in cpi_main.columns if 'cpi' in str(col).lower() and 'score' in str(col).lower()][0]
cpi_2012_2024 = cpi_main[[country_col, 'Year', cpi_score_col]].copy()
cpi_2012_2024 = cpi_2012_2024.rename(columns={country_col: 'Country', cpi_score_col: 'Corruption_Perception_Index'})
cpi_2012_2024 = cpi_2012_2024[cpi_2012_2024['Country'].isin(country_list) & cpi_2012_2024['Year'].between(2012, 2024)]
# Read 2009-2010 data (sheet index 2)
cpi_0910 = pd.read_excel(FILE_PATH_PREFIX + 'Corruption_Perception_index.xlsx', sheet_name=2)
cpi_0910.columns = cpi_0910.columns.str.strip()
country_col_0910 = [col for col in cpi_0910.columns if 'country' in str(col).lower()][0]
cpi_2009 = cpi_0910[[country_col_0910, 'CPI_2009_Score']].copy()
cpi_2009['Year'] = 2009
cpi_2009 = cpi_2009.rename(columns={country_col_0910: 'Country', 'CPI_2009_Score': 'Corruption_Perception_Index'})
cpi_2009 = cpi_2009[cpi_2009['Country'].isin(country_list)]
cpi_2010 = cpi_0910[[country_col_0910, 'CPI_2010_Score']].copy()
cpi_2010['Year'] = 2010
cpi_2010 = cpi_2010.rename(columns={country_col_0910: 'Country', 'CPI_2010_Score': 'Corruption_Perception_Index'})
cpi_2010 = cpi_2010[cpi_2010['Country'].isin(country_list)]
# Combine all years
cpi_all = pd.concat([cpi_2009, cpi_2010, cpi_2012_2024], ignore_index=True)
cpi_all['Corruption_Perception_Index'] = pd.to_numeric(cpi_all['Corruption_Perception_Index'], errors='coerce')
cpi_all = cpi_all.dropna(subset=['Country', 'Year', 'Corruption_Perception_Index'])
cpi_all = cpi_all.drop_duplicates(subset=['Country', 'Year'], keep='last')
cpi_all = cpi_all[['Country', 'Year', 'Corruption_Perception_Index']]
master = safe_merge(master, cpi_all, 'Corruption_Perception_Index')
    
# Save files
master.to_excel("master_panel_merged_all_data.xlsx", index=False)

