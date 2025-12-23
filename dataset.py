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
    ("Latvia", "LVA"),
    ("Lithuania", "LTU"),
    ("Moldova", "MDA"),
    ("North Macedonia", "MKD"),
    ("Poland", "POL"),
    ("Romania", "ROU"),
    ("Serbia", "SRB"),
    ("Slovakia", "SVK"),
    ("Slovenia", "SVN"),
    ("Turkey", "TUR")
]

countries_mapping = {name: iso3 for name, iso3 in countries}
country_list = [name for name, _ in countries]
iso3_list = [iso3 for _, iso3 in countries]
years = list(range(1995, 2025))

# --- Country name variations ---
COUNTRY_NAME_VARIATIONS = {
    'Czechia': 'Czech Republic',
    'Czech republic': 'Czechia',
    'slovakia': 'Slovakia',
    'slovak republic': 'Slovakia',
    'macedonia, fyr': 'North Macedonia',
    'turkey': 'Turkey',
    'türkiye': 'Turkey',
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
    "Digitalization_InternetUsers",
    "Human_Capital_index",
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
    """Normalize country names from data files to match our country list."""
    df[country_col] = df[country_col].str.strip()
    
    def map_country(name):
        if pd.isna(name):
            return name
        name_str = str(name).strip()
        name_lower = name_str.lower()
        
        # Remove common suffixes
        name_clean = name_str
        if ', republic of' in name_lower:
            name_clean = name_str.split(',')[0].strip()
            name_lower = name_clean.lower()
        
        # Check variations mapping
        if name_lower in COUNTRY_NAME_VARIATIONS:
            return COUNTRY_NAME_VARIATIONS[name_lower]
        else:
            return name_clean
    
    df[country_col] = df[country_col].apply(map_country)
    return df

def safe_merge(master_df, new_data, column_name, on_cols=['Country', 'Year']):
    """Safely merge new data into master dataframe"""
    if new_data.empty:
        print(f"  ⚠ Warning: No data to merge for {column_name}")
        return master_df
    
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
    
    agr = agr[agr['Series Name'].str.contains('Agriculture', case=False, na=False)]
    year_cols = [col for col in agr.columns if '[YR' in str(col)]
    
    agr_long = agr.melt(id_vars=['Country Name'], value_vars=year_cols, 
                        var_name='Year_Col', value_name='Agriculture_GDP')
    
    agr_long['Year'] = agr_long['Year_Col'].str.extract(r'(\d{4})')[0].astype(int)
    agr_long = agr_long.drop(columns=['Year_Col'])
    agr_long = agr_long.rename(columns={'Country Name': 'Country'})
    agr_long = normalize_country_names(agr_long, 'Country')
    
    agr_long['Agriculture_GDP'] = pd.to_numeric(agr_long['Agriculture_GDP'], errors='coerce')
    agr_long = agr_long[agr_long['Country'].isin(country_list) & agr_long['Year'].isin(years)]
    agr_long = agr_long.dropna(subset=['Agriculture_GDP'])
    agr_long = agr_long.drop_duplicates(subset=['Country', 'Year'], keep='last')
    
    master = safe_merge(master, agr_long[['Country', 'Year', 'Agriculture_GDP']], 'Agriculture_GDP')
    
except Exception as e:
    print(f"  ✗ Error: {e}")

# ============================================
# 2. CPI - Inflation
# ============================================
print("\n2. Processing CPI (Inflation) data...")
try:
    cpi_inflation = pd.read_csv(FILE_PATH_PREFIX + 'CPI.csv')
    cpi_inflation.columns = cpi_inflation.columns.str.strip()
    
    country_col = None
    for col in cpi_inflation.columns:
        if str(col).lower() == 'country':
            country_col = col
            break
    
    if not country_col:
        print("  ⚠ Could not find country column")
    else:
        id_vars = [country_col]
        metadata_cols = ['DATASET', 'SERIES_CODE', 'OBS_MEASURE', 'INDEX_TYPE', 
                        'COICOP_1999', 'TYPE_OF_TRANSFORMATION', 'FREQUENCY', 'SCALE']
        for col in metadata_cols:
            if col in cpi_inflation.columns:
                id_vars.append(col)
        
        year_cols = []
        for col in cpi_inflation.columns:
            if col in id_vars:
                continue
            col_str = str(col).strip()
            if col_str.isdigit() and len(col_str) == 4:
                year_val = int(col_str)
                if year_val in years:
                    year_cols.append(col)
        
        if year_cols:
            cpi_long = cpi_inflation.melt(id_vars=id_vars, value_vars=year_cols, 
                                         var_name='Year', value_name='Inflation')
            cpi_long['Year'] = pd.to_numeric(cpi_long['Year'], errors='coerce')
        else:
            cpi_long = cpi_inflation.melt(id_vars=id_vars, var_name='Year', value_name='Inflation')
            cpi_long['Year'] = pd.to_numeric(cpi_long['Year'], errors='coerce')
        
        cpi_long = cpi_long.rename(columns={country_col: 'Country'})
        cpi_long = normalize_country_names(cpi_long, 'Country')
        
        cpi_long = cpi_long[
            cpi_long['Country'].isin(country_list) & 
            cpi_long['Year'].isin(years)
        ].copy()
        
        cpi_long['Inflation'] = pd.to_numeric(cpi_long['Inflation'], errors='coerce')
        cpi_long = cpi_long.dropna(subset=['Country', 'Year', 'Inflation'])
        cpi_long = cpi_long.drop_duplicates(subset=['Country', 'Year'], keep='last')
        
        if not cpi_long.empty:
            master = safe_merge(master, cpi_long[['Country', 'Year', 'Inflation']], 'Inflation')
        else:
            print("  ⚠ No CPI/Inflation data found after filtering")
            
except Exception as e:
    print(f"  ✗ Error: {e}")

# ============================================
# 3. GDP (constant US$)
# ============================================
print("\n3. Processing GDP data...")
try:
    try:
        gdp = pd.read_excel(FILE_PATH_PREFIX + 'GDP costant (US$).xlsx', sheet_name=0)
    except:
        gdp = pd.read_excel(FILE_PATH_PREFIX + 'GDP constant (US$).xlsx', sheet_name=0)
    
    gdp.columns = gdp.columns.str.strip()
    
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
        id_vars = [country_col]
        series_col = None
        for col in gdp.columns:
            if 'series' in str(col).lower() and 'name' in str(col).lower():
                series_col = col
                id_vars.append(series_col)
                break
        
        year_cols = []
        for col in gdp.columns:
            if col in id_vars:
                continue
            col_str = str(col)
            year_match = None
            if len(col_str) >= 4 and col_str[:4].isdigit():
                year_match = int(col_str[:4])
            elif '[' in col_str:
                parts = col_str.split('[')
                if parts[0].strip()[:4].isdigit():
                    year_match = int(parts[0].strip()[:4])
            
            if year_match and year_match in years:
                year_cols.append(col)
        
        if year_cols:
            gdp_long = gdp.melt(id_vars=id_vars, value_vars=year_cols, 
                               var_name='Year_Col', value_name='GDP')
            gdp_long['Year'] = gdp_long['Year_Col'].astype(str).str.extract(r'(\d{4})')[0]
            gdp_long['Year'] = pd.to_numeric(gdp_long['Year'], errors='coerce')
            gdp_long = gdp_long.drop(columns=['Year_Col'])
        else:
            gdp_long = gdp.melt(id_vars=id_vars, var_name='Year_Col', value_name='GDP')
            gdp_long['Year'] = gdp_long['Year_Col'].astype(str).str.extract(r'(\d{4})')[0]
            gdp_long['Year'] = pd.to_numeric(gdp_long['Year'], errors='coerce')
            gdp_long = gdp_long.drop(columns=['Year_Col'])
        
        gdp_long = gdp_long.rename(columns={country_col: 'Country'})
        gdp_long = normalize_country_names(gdp_long, 'Country')
        
        gdp_long = gdp_long[
            gdp_long['Country'].isin(country_list) & 
            gdp_long['Year'].isin(years)
        ].copy()
        
        if series_col and series_col in gdp_long.columns:
            gdp_series = gdp_long[gdp_long[series_col].str.contains('GDP', case=False, na=False)].copy()
            if not gdp_series.empty:
                gdp_long = gdp_series
            gdp_long = gdp_long.drop(columns=[series_col])
        
        gdp_long['GDP'] = pd.to_numeric(gdp_long['GDP'], errors='coerce')
        gdp_long = gdp_long.dropna(subset=['Country', 'Year', 'GDP'])
        gdp_long = gdp_long.drop_duplicates(subset=['Country', 'Year'], keep='last')
        
        if not gdp_long.empty:
            master = safe_merge(master, gdp_long[['Country', 'Year', 'GDP']], 'GDP')
        else:
            print("  ⚠ No GDP data found after filtering")
            
except Exception as e:
    print(f"  ✗ Error: {e}")

# ============================================
# 4. INTERNET USERS
# ============================================
print("\n4. Processing Internet Users data...")
try:
    try:
        internet = pd.read_excel(FILE_PATH_PREFIX + 'Individuals using the Internet (% of population).xlsx', 
                                sheet_name=0, skiprows=3)
    except:
        internet = pd.read_excel(FILE_PATH_PREFIX + 'Individuals using the Internet.xls', 
                                sheet_name=0, skiprows=3)
    
    internet.columns = internet.columns.str.strip()
    
    country_col = None
    for col in internet.columns:
        if 'country' in str(col).lower():
            country_col = col
            break
    
    if country_col:
        internet_long = internet.melt(id_vars=[country_col], var_name='Year', 
                                     value_name='Digitalization_InternetUsers')
        internet_long = internet_long.rename(columns={country_col: 'Country'})
        internet_long = normalize_country_names(internet_long, 'Country')
        internet_long['Year'] = pd.to_numeric(internet_long['Year'], errors='coerce')
        internet_long['Digitalization_InternetUsers'] = pd.to_numeric(
            internet_long['Digitalization_InternetUsers'], errors='coerce')
        internet_long = internet_long[internet_long['Country'].isin(country_list) & 
                                     internet_long['Year'].isin(years)]
        internet_long = internet_long.dropna(subset=['Digitalization_InternetUsers'])
        internet_long = internet_long.drop_duplicates(subset=['Country', 'Year'], keep='last')
        master = safe_merge(master, internet_long[['Country', 'Year', 'Digitalization_InternetUsers']], 
                          'Digitalization_InternetUsers')
except Exception as e:
    print(f"  ✗ Error: {e}")

# ============================================
# 5. HUMAN CAPITAL INDEX
# ============================================
print("\n5. Processing Human Capital Index data...")
try:
    hci = pd.read_csv(FILE_PATH_PREFIX + 'human_capital_indexcvs.csv')
    hci.columns = hci.columns.str.strip()
    
    print(f"  Original shape: {hci.shape}")
    print(f"  Unique variables: {hci['Variable name'].unique()}")
    
    hci_filtered = hci[hci['Country'].isin(country_list)].copy()
    
    print(f"  Countries found in HCI data: {hci_filtered['Country'].nunique()}")
    
    missing_countries = set(country_list) - set(hci_filtered['Country'].unique())
    if missing_countries:
        print(f"  ⚠️  Countries NOT found in HCI data: {sorted(missing_countries)}")
    
    year_columns = [str(year) for year in range(1995, 2024)]
    available_years = [col for col in year_columns if col in hci_filtered.columns]
    
    if available_years:
        print(f"  Years available: {available_years[0]} to {available_years[-1]}")
        
        id_vars = ['ISO code', 'Country', 'Variable code', 'Variable name']
        hci_long = pd.melt(
            hci_filtered,
            id_vars=id_vars,
            value_vars=available_years,
            var_name='Year',
            value_name='Value'
        )
        
        hci_long['Year'] = hci_long['Year'].astype(int)
        hci_long = hci_long[hci_long['Year'].isin(years)]
        
        hci_pivot = hci_long.pivot_table(
            index=['Country', 'Year'],
            columns='Variable name',
            values='Value',
            aggfunc='first'
        ).reset_index()
        
        print(f"  Final HCI data shape: {hci_pivot.shape}")
        
        hci_col_name = None
        for col in hci_pivot.columns:
            if 'human capital' in str(col).lower() and 'index' in str(col).lower():
                hci_col_name = col
                break
        
        if hci_col_name:
            print(f"  Using HCI column: {hci_col_name}")
            
            hci_merge = hci_pivot[['Country', 'Year', hci_col_name]].copy()
            hci_merge = hci_merge.rename(columns={hci_col_name: 'Human_Capital_index'})
            
            hci_merge['Human_Capital_index'] = pd.to_numeric(hci_merge['Human_Capital_index'], errors='coerce')
            hci_merge = hci_merge.dropna(subset=['Human_Capital_index'])
            hci_merge = hci_merge.drop_duplicates(subset=['Country', 'Year'], keep='last')
            
            master = safe_merge(master, hci_merge, 'Human_Capital_index')
        else:
            print("  ⚠️  Could not find Human Capital Index column")
            print(f"  Available columns: {[col for col in hci_pivot.columns if col not in ['Country', 'Year']]}")
    
except Exception as e:
    print(f"  ✗ Error: {e}")

# ============================================
# 6. LABOUR FORCE
# ============================================
print("\n6. Processing Labour Force data...")
try:
    labour = None
    for filename in ['Labour force, total.xlsx', 'Labor force, total.xlsx', 'Labour_force_total.xlsx']:
        try:
            labour = pd.read_excel(FILE_PATH_PREFIX + filename, sheet_name=0)
            print(f"  Loaded from: {filename}")
            break
        except:
            continue
    
    if labour is None:
        print("  ⚠ Could not load Labour Force file")
    else:
        labour.columns = labour.columns.str.strip()
        
        country_col = None
        for col in labour.columns:
            if 'country' in str(col).lower() and 'name' in str(col).lower():
                country_col = col
                break
        
        if country_col:
            if 'Series Name' in labour.columns:
                labour = labour[labour['Series Name'].str.contains('Labour force|Labor force', 
                                                                  case=False, na=False)]
            
            year_cols = [col for col in labour.columns if '[YR' in str(col)]
            
            labour_long = labour.melt(id_vars=[country_col], value_vars=year_cols, 
                                     var_name='Year_Col', value_name='Labour_Force_Total')
            
            labour_long['Year'] = labour_long['Year_Col'].str.extract(r'(\d{4})')[0].astype(int)
            labour_long = labour_long.drop(columns=['Year_Col'])
            labour_long = labour_long.rename(columns={country_col: 'Country'})
            labour_long = normalize_country_names(labour_long, 'Country')
            
            labour_long['Labour_Force_Total'] = pd.to_numeric(labour_long['Labour_Force_Total'], errors='coerce')
            labour_long = labour_long[labour_long['Country'].isin(country_list) & 
                                     labour_long['Year'].isin(years)]
            labour_long = labour_long.dropna(subset=['Labour_Force_Total'])
            labour_long = labour_long.drop_duplicates(subset=['Country', 'Year'], keep='last')
            
            master = safe_merge(master, labour_long[['Country', 'Year', 'Labour_Force_Total']], 
                              'Labour_Force_Total')
    
except Exception as e:
    print(f"  ✗ Error: {e}")

# ============================================
# 7. REMITTANCES
# ============================================
print("\n7. Processing Remittances data...")
try:
    remit = pd.read_excel(FILE_PATH_PREFIX + 'remittances.xls', sheet_name='Data', skiprows=3)
    remit.columns = remit.columns.str.strip()
    
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
        remit_long['Remittances_GDP'] = pd.to_numeric(remit_long['Remittances_GDP'], errors='coerce')
        remit_long = remit_long[remit_long['Country'].isin(country_list) & 
                               remit_long['Year'].isin(years)]
        remit_long = remit_long.dropna(subset=['Remittances_GDP'])
        remit_long = remit_long.drop_duplicates(subset=['Country', 'Year'], keep='last')
        master = safe_merge(master, remit_long[['Country', 'Year', 'Remittances_GDP']], 'Remittances_GDP')
except Exception as e:
    print(f"  ✗ Error: {e}")

# ============================================
# 8. TAX REVENUE
# ============================================
print("\n8. Processing Tax Revenue data...")
try:
    tax = pd.read_excel(FILE_PATH_PREFIX + 'tax_revenue%GDP.xls', sheet_name='Data', skiprows=3)
    tax.columns = tax.columns.str.strip()
    
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
        tax_long['Tax_Revenue_GDP'] = pd.to_numeric(tax_long['Tax_Revenue_GDP'], errors='coerce')
        tax_long = tax_long[tax_long['Country'].isin(country_list) & 
                           tax_long['Year'].isin(years)]
        tax_long = tax_long.dropna(subset=['Tax_Revenue_GDP'])
        tax_long = tax_long.drop_duplicates(subset=['Country', 'Year'], keep='last')
        master = safe_merge(master, tax_long[['Country', 'Year', 'Tax_Revenue_GDP']], 'Tax_Revenue_GDP')
except Exception as e:
    print(f"  ✗ Error: {e}")

# ============================================
# 9. TRADE OPENNESS
# ============================================
print("\n9. Processing Trade Openness data...")
try:
    try:
        trade = pd.read_excel(FILE_PATH_PREFIX + 'Trade Openness (%of GDP).xlsx', 
                             sheet_name=0, skiprows=3)
    except:
        trade = pd.read_excel(FILE_PATH_PREFIX + 'Trade Openness (%of GDP).xls', 
                             sheet_name=0, skiprows=3)
    
    trade.columns = trade.columns.str.strip()
    
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
        trade_long['Trade_Openness_GDP'] = pd.to_numeric(trade_long['Trade_Openness_GDP'], errors='coerce')
        trade_long = trade_long[trade_long['Country'].isin(country_list) & 
                               trade_long['Year'].isin(years)]
        trade_long = trade_long.dropna(subset=['Trade_Openness_GDP'])
        trade_long = trade_long.drop_duplicates(subset=['Country', 'Year'], keep='last')
        master = safe_merge(master, trade_long[['Country', 'Year', 'Trade_Openness_GDP']], 'Trade_Openness_GDP')
except Exception as e:
    print(f"  ✗ Error: {e}")

# ============================================
# 10. UNEMPLOYMENT
# ============================================
print("\n10. Processing Unemployment data...")
try:
    unemp = pd.read_excel(FILE_PATH_PREFIX + 'Unemployment, total (% of total labor force).xls', 
                         sheet_name='Data', skiprows=3)
    unemp.columns = unemp.columns.str.strip()
    
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
        unemp_long['Unemployment_Rate'] = pd.to_numeric(unemp_long['Unemployment_Rate'], errors='coerce')
        unemp_long = unemp_long[unemp_long['Country'].isin(country_list) & 
                               unemp_long['Year'].isin(years)]
        unemp_long = unemp_long.dropna(subset=['Unemployment_Rate'])
        unemp_long = unemp_long.drop_duplicates(subset=['Country', 'Year'], keep='last')
        master = safe_merge(master, unemp_long[['Country', 'Year', 'Unemployment_Rate']], 'Unemployment_Rate')
except Exception as e:
    print(f"  ✗ Error: {e}")

# ============================================
# 11. WGI (Governance Indicators)
# ============================================
print("\n11. Processing WGI (Governance Indicators) data...")
try:
    wgi = pd.read_excel(FILE_PATH_PREFIX + 'wgidataset.xlsx', sheet_name=0)
    wgi.columns = wgi.columns.str.strip()
    
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
    else:
        wgi_clean = wgi.rename(columns={
            country_col: 'Country',
            year_col: 'Year',
            indicator_col: 'indicator',
            estimate_col: 'estimate'
        })
        
        wgi_clean = normalize_country_names(wgi_clean, 'Country')
        
        wgi_filtered = wgi_clean[
            wgi_clean['Country'].isin(country_list) & 
            wgi_clean['Year'].isin(years)
        ].copy()
        
        print(f"  Filtered rows: {len(wgi_filtered)}")
        
        # Government Effectiveness (ge)
        wgi_ge = wgi_filtered[wgi_filtered['indicator'].str.lower().str.strip() == 'ge'].copy()
        if not wgi_ge.empty:
            wgi_ge = wgi_ge[['Country', 'Year', 'estimate']].copy()
            wgi_ge = wgi_ge.rename(columns={'estimate': 'Government_Effectiveness'})
            wgi_ge['Government_Effectiveness'] = pd.to_numeric(wgi_ge['Government_Effectiveness'], errors='coerce')
            wgi_ge = wgi_ge.dropna(subset=['Government_Effectiveness'])
            wgi_ge = wgi_ge.drop_duplicates(subset=['Country', 'Year'], keep='last')
            master = safe_merge(master, wgi_ge, 'Government_Effectiveness')
        
        # Regulatory Quality (rq)
        wgi_rq = wgi_filtered[wgi_filtered['indicator'].str.lower().str.strip() == 'rq'].copy()
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
#   
# Save files
master.to_excel("master_panel_merged_all_data.xlsx", index=False)
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
