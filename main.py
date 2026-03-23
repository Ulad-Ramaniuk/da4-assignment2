# =============================================================================
# DA4 Assignment 2: CO2 Emissions and GDP — Panel Data Analysis
# Author: Uladzislau Ramaniuk
# =============================================================================

# =============================================================================
# 0. IMPORTS
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
from statsmodels.tsa.stattools import adfuller, coint
import wbdata
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD & RESHAPE DATA
# =============================================================================
df = pd.read_csv('data/wdi_raw.csv')

# Keep only required series
series_needed = ['NY.GDP.PCAP.PP.KD', 'EN.GHG.CO2.PC.CE.AR5', 'EG.USE.PCAP.KG.OE']
df = df[df['Series Code'].isin(series_needed)]

# Reshape from wide to long format
year_cols = [col for col in df.columns if col.startswith('19') or col.startswith('20')]
df_long = df.melt(
    id_vars=['Country Name', 'Country Code', 'Series Code'],
    value_vars=year_cols,
    var_name='Year',
    value_name='Value'
)

# Clean year column and convert values to numeric
df_long['year'] = df_long['Year'].str[:4].astype(int)
df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')

# Pivot to wide format with one column per indicator
df_wide = df_long.pivot_table(
    index=['Country Name', 'Country Code', 'year'],
    columns='Series Code',
    values='Value'
).reset_index()

df_wide.columns.name = None
df_wide = df_wide.rename(columns={
    'NY.GDP.PCAP.PP.KD': 'gdp_pc',
    'EN.GHG.CO2.PC.CE.AR5': 'co2_pc',
    'EG.USE.PCAP.KG.OE': 'energy'
})

# Filter to 1992 onwards (as per assignment)
df_wide = df_wide[df_wide['year'] >= 1992]

# =============================================================================
# 2. DATA CLEANING & COVERAGE
# =============================================================================

# Drop countries with less than 50% coverage for GDP and CO2
coverage = df_wide.groupby('Country Name')[['gdp_pc', 'co2_pc']].apply(
    lambda x: x.notna().mean() * 100
).round(1)

good_coverage = coverage[(coverage['gdp_pc'] >= 50) & (coverage['co2_pc'] >= 50)].index
df_clean = df_wide[df_wide['Country Name'].isin(good_coverage)]

n_dropped = df_wide['Country Name'].nunique() - df_clean['Country Name'].nunique()
print(f"Coverage filter: kept {df_clean['Country Name'].nunique()} countries, dropped {n_dropped}")
dropped_countries = set(df_wide['Country Name'].unique()) - set(good_coverage)
print("Dropped by coverage filter:", sorted(dropped_countries))

# Drop rows missing both key variables
df_clean = df_clean.dropna(subset=['gdp_pc', 'co2_pc'])

# Remove World Bank aggregate groups
aggregate_keywords = [
    'income', 'Asia', 'Africa', 'Europe', 'America', 'Caribbean',
    'World', 'IDA', 'IBRD', 'OECD', 'dividend', 'states', 'area',
    'Arab World', 'Pacific', 'South Asia', 'North America',
    'Fragile', 'Heavily indebted', 'Least developed', 'Middle East'
]

def is_aggregate(name):
    return any(keyword in name for keyword in aggregate_keywords)

df_clean = df_clean[~df_clean['Country Name'].apply(is_aggregate)]

# Remove non-sovereign territories
territories = [
    'Puerto Rico (US)', 'Virgin Islands (U.S.)', 'Cayman Islands',
    'Turks and Caicos Islands', 'Aruba', 'Bermuda', 'Greenland',
    'Faroe Islands', 'Hong Kong SAR, China', 'Macao SAR, China'
]
df_clean = df_clean[~df_clean['Country Name'].isin(territories)]

print("Dropped by territory filter:", territories)

# Remove zero values (log undefined)
df_clean = df_clean[df_clean['co2_pc'] > 0]
df_clean = df_clean[df_clean['gdp_pc'] > 0]

print(f"Final sample: {df_clean['Country Name'].nunique()} countries, {df_clean.shape[0]} observations")

# =============================================================================
# 3. ADD SERVICES SHARE VIA API & INCOME GROUP CLASSIFICATIONS
# =============================================================================

# Download services share of GDP
indicators = {'NV.SRV.TOTL.ZS': 'services_share'}
df_services = wbdata.get_dataframe(indicators).reset_index()
df_services = df_services.rename(columns={'date': 'year'})
df_services['year'] = df_services['year'].astype(int)

df_clean = df_clean.merge(
    df_services,
    left_on=['Country Name', 'year'],
    right_on=['country', 'year'],
    how='left'
).drop(columns=['country'])

# Download income group classifications
income_data = wbdata.get_countries()
income_df = pd.DataFrame([
    {'Country Code': c['id'], 'income_group': c['incomeLevel']['value']}
    for c in income_data
])
income_df = income_df[~income_df['income_group'].isin(['Aggregates', 'Not classified'])]
df_clean = df_clean.merge(income_df, on='Country Code', how='left')

# =============================================================================
# 4. LOG TRANSFORMATIONS
# =============================================================================
df_clean['log_gdp']      = np.log(df_clean['gdp_pc'])
df_clean['log_co2']      = np.log(df_clean['co2_pc'])
df_clean['log_energy']   = np.log(df_clean['energy'])
df_clean['log_services'] = np.log(df_clean['services_share'])
df_clean['log_gdp_sq']   = df_clean['log_gdp'] ** 2

print("\n--- Summary Statistics ---")
print(df_clean[['log_gdp', 'log_co2', 'log_energy', 'log_services']].describe().round(3))

# =============================================================================
# 5. CROSS-SECTION OLS (MODELS 1 & 2)
# =============================================================================

# Model 1: Cross-section OLS for 2005 (linear)
df_2005 = df_clean[df_clean['year'] == 2005]
model1 = smf.ols('log_co2 ~ log_gdp', data=df_2005).fit(cov_type='HC3')

# Model 1b: Cross-section OLS for 2005 (nonlinear/EKC)
model1_nl = smf.ols('log_co2 ~ log_gdp + log_gdp_sq', data=df_2005).fit(cov_type='HC3')

# Model 2: Cross-section OLS for last year (linear)
last_year = df_clean['year'].max()
df_last = df_clean[df_clean['year'] == last_year]
model2 = smf.ols('log_co2 ~ log_gdp', data=df_last).fit(cov_type='HC3')

# Model 2b: Cross-section OLS for last year (nonlinear/EKC)
model2_nl = smf.ols('log_co2 ~ log_gdp + log_gdp_sq', data=df_last).fit(cov_type='HC3')

print(f"\n--- Model 1: Cross-section OLS 2005 (N={len(df_2005)}) ---")
print(f"log_gdp coef: {model1.params['log_gdp']:.4f}, SE: {model1.bse['log_gdp']:.4f}, p-value: {model1.pvalues['log_gdp']:.4f}, R²: {model1.rsquared:.3f}")

print(f"\n--- Model 1b: Cross-section OLS 2005 Nonlinear (N={len(df_2005)}) ---")
print(f"log_gdp coef: {model1_nl.params['log_gdp']:.4f}, SE: {model1_nl.bse['log_gdp']:.4f}, p-value: {model1_nl.pvalues['log_gdp']:.4f}, R²: {model1_nl.rsquared:.3f}")

print(f"\n--- Model 2: Cross-section OLS {last_year} (N={len(df_last)}) ---")
print(f"log_gdp coef: {model2.params['log_gdp']:.4f}, SE: {model2.bse['log_gdp']:.4f}, p-value: {model2.pvalues['log_gdp']:.4f}, R²: {model2.rsquared:.3f}")

print(f"\n--- Model 2b: Cross-section OLS {last_year} Nonlinear (N={len(df_last)}) ---")
print(f"log_gdp coef: {model2_nl.params['log_gdp']:.4f}, SE: {model2_nl.bse['log_gdp']:.4f}, p-value: {model2_nl.pvalues['log_gdp']:.4f}, R²: {model2_nl.rsquared:.3f}")

# =============================================================================
# 6. FIRST DIFFERENCE MODELS (MODELS 3, 4, 5)
# =============================================================================
df_fd = df_clean.sort_values(['Country Code', 'year']).copy()
df_fd['d_log_co2']      = df_fd.groupby('Country Code')['log_co2'].diff()
df_fd['d_log_gdp']      = df_fd.groupby('Country Code')['log_gdp'].diff()
df_fd['d_log_energy'] = df_fd.groupby('Country Code')['log_energy'].diff()
df_fd['d_log_gdp_lag2'] = df_fd.groupby('Country Code')['d_log_gdp'].shift(2)
df_fd['d_log_gdp_lag6'] = df_fd.groupby('Country Code')['d_log_gdp'].shift(6)

# Model 3: First difference, no lags
model3 = smf.ols('d_log_co2 ~ d_log_gdp + year', data=df_fd).fit(cov_type='HC3')

# Model 4: First difference, 2-year lags
model4 = smf.ols('d_log_co2 ~ d_log_gdp + d_log_gdp_lag2 + year', data=df_fd).fit(cov_type='HC3')

# Model 5: First difference, 6-year lags
model5 = smf.ols('d_log_co2 ~ d_log_gdp + d_log_gdp_lag6 + year', data=df_fd).fit(cov_type='HC3')

print(f"\n--- Model 3: First Difference, No Lags (N={model3.nobs:.0f}) ---")
print(f"d_log_gdp coef: {model3.params['d_log_gdp']:.4f}, SE: {model3.bse['d_log_gdp']:.4f}, p-value: {model3.pvalues['d_log_gdp']:.4f}")
print(f"year trend: {model3.params['year']:.6f}, SE: {model3.bse['year']:.4f}, p-value: {model3.pvalues['year']:.4f}")

print(f"\n--- Model 4: First Difference, 2-Year Lags (N={model4.nobs:.0f}) ---")
print(f"d_log_gdp coef: {model4.params['d_log_gdp']:.4f}, SE: {model4.bse['d_log_gdp']:.4f}, p-value: {model4.pvalues['d_log_gdp']:.4f}")
print(f"d_log_gdp_lag2 coef: {model4.params['d_log_gdp_lag2']:.4f}, SE: {model4.bse['d_log_gdp_lag2']:.4f}, p-value: {model4.pvalues['d_log_gdp_lag2']:.4f}")
print(f"year trend: {model4.params['year']:.6f}, SE: {model4.bse['year']:.4f}, p-value: {model4.pvalues['year']:.4f}")

print(f"\n--- Model 5: First Difference, 6-Year Lags (N={model5.nobs:.0f}) ---")
print(f"d_log_gdp coef: {model5.params['d_log_gdp']:.4f}, SE: {model5.bse['d_log_gdp']:.4f}, p-value: {model5.pvalues['d_log_gdp']:.4f}")
print(f"d_log_gdp_lag6 coef: {model5.params['d_log_gdp_lag6']:.4f}, SE: {model5.bse['d_log_gdp_lag6']:.4f}, p-value: {model5.pvalues['d_log_gdp_lag6']:.4f}")
print(f"year trend: {model5.params['year']:.6f}, SE: {model5.bse['year']:.4f}, p-value: {model5.pvalues['year']:.4f}")

# =============================================================================
# 7. FIXED EFFECTS MODEL (MODEL 6)
# =============================================================================
df_fe = df_clean.copy().set_index(['Country Code', 'year'])
df_fe['log_gdp_sq'] = df_fe['log_gdp'] ** 2

# Model 6: Fixed Effects, linear
model6 = PanelOLS.from_formula(
    'log_co2 ~ log_gdp + EntityEffects + TimeEffects',
    data=df_fe
).fit(cov_type='clustered', cluster_entity=True)

# Model 6b: Fixed Effects, nonlinear (EKC test)
model6_nl = PanelOLS.from_formula(
    'log_co2 ~ log_gdp + log_gdp_sq + EntityEffects + TimeEffects',
    data=df_fe
).fit(cov_type='clustered', cluster_entity=True)

print(f"\n--- Model 6: Fixed Effects Linear (N={model6.nobs}) ---")
print(f"log_gdp coef: {model6.params['log_gdp']:.4f}, SE: {model6.std_errors['log_gdp']:.4f}, p-value: {model6.pvalues['log_gdp']:.4f}")
print(f"Within R²: {model6.rsquared_within:.3f}")

print(f"\n--- Model 6b: Fixed Effects Nonlinear/EKC (N={model6_nl.nobs}) ---")
print(f"log_gdp coef: {model6_nl.params['log_gdp']:.4f}, SE: {model6_nl.std_errors['log_gdp']:.4f}, p-value: {model6_nl.pvalues['log_gdp']:.4f}")
print(f"log_gdp_sq coef: {model6_nl.params['log_gdp_sq']:.4f}, SE: {model6_nl.std_errors['log_gdp_sq']:.4f}, p-value: {model6_nl.pvalues['log_gdp_sq']:.4f}")
print(f"Within R²: {model6_nl.rsquared_within:.3f}")

# EKC Turning point
b1 = model6_nl.params['log_gdp']
b2 = model6_nl.params['log_gdp_sq']
tp_log = -b1 / (2 * b2)
tp_gdp = np.exp(tp_log)
print(f"EKC Turning point: log GDP = {tp_log:.3f} → GDP per capita = ${tp_gdp:,.0f} PPP")

# =============================================================================
# 8. CONFOUNDER MODELS (ENERGY USE)
# =============================================================================

# Model 1 with energy confounder
model1_c = smf.ols('log_co2 ~ log_gdp + log_energy', data=df_2005).fit(cov_type='HC3')

# Model 4 with energy confounder
model4_c = smf.ols('d_log_co2 ~ d_log_gdp + d_log_gdp_lag2 + year + d_log_energy',
                    data=df_fd).fit(cov_type='HC3')

# Model 6 with energy confounder
model6_c = PanelOLS.from_formula(
    'log_co2 ~ log_gdp + log_energy + EntityEffects + TimeEffects',
    data=df_fe
).fit(cov_type='clustered', cluster_entity=True)

print(f"\n--- Model 1 + Energy Confounder (N={model1_c.nobs:.0f}) ---")
print(f"log_gdp:    {model1_c.params['log_gdp']:.4f}  SE={model1_c.bse['log_gdp']:.4f}  p={model1_c.pvalues['log_gdp']:.4f}  (was {model1.params['log_gdp']:.4f} without energy)")
print(f"log_energy: {model1_c.params['log_energy']:.4f}  SE={model1_c.bse['log_energy']:.4f}  p={model1_c.pvalues['log_energy']:.4f}")

print(f"\n--- Model 4 + Energy Confounder (N={model4_c.nobs:.0f}) ---")
print(f"d_log_gdp:    {model4_c.params['d_log_gdp']:.4f}  SE={model4_c.bse['d_log_gdp']:.4f}  p={model4_c.pvalues['d_log_gdp']:.4f}  (was {model4.params['d_log_gdp']:.4f} without energy)")
print(f"d_log_energy: {model4_c.params['d_log_energy']:.4f}  SE={model4_c.bse['d_log_energy']:.4f}  p={model4_c.pvalues['d_log_energy']:.4f}")
print(f"d_log_gdp_lag2: {model4_c.params['d_log_gdp_lag2']:.4f}  SE={model4_c.bse['d_log_gdp_lag2']:.4f}  p={model4_c.pvalues['d_log_gdp_lag2']:.4f}")

print(f"\n--- Model 6 + Energy Confounder (N={model6_c.nobs}) ---")
print(f"log_gdp:    {model6_c.params['log_gdp']:.4f}  SE={model6_c.std_errors['log_gdp']:.4f}  p={model6_c.pvalues['log_gdp']:.4f}  (was {model6.params['log_gdp']:.4f} without energy)")
print(f"log_energy: {model6_c.params['log_energy']:.4f}  SE={model6_c.std_errors['log_energy']:.4f}  p={model6_c.pvalues['log_energy']:.4f}")

# =============================================================================
# 9. APPENDIX: ADDITIONAL CONFOUNDERS & EXTENSIONS
# =============================================================================

# Model 6 with services share confounder
model6_srv = PanelOLS.from_formula(
    'log_co2 ~ log_gdp + log_services + EntityEffects + TimeEffects',
    data=df_fe
).fit(cov_type='clustered', cluster_entity=True)

# Model 6 with both confounders
model6_both = PanelOLS.from_formula(
    'log_co2 ~ log_gdp + log_energy + log_services + EntityEffects + TimeEffects',
    data=df_fe
).fit(cov_type='clustered', cluster_entity=True)

print(f"\n--- Appendix: Model 6 + Services Confounder ---")
print(f"log_gdp coef: {model6_srv.params['log_gdp']:.4f}")
print(f"log_services coef: {model6_srv.params['log_services']:.4f}, p-value: {model6_srv.pvalues['log_services']:.4f}")

print(f"\n--- Appendix: Model 6 + Both Confounders ---")
print(f"log_gdp coef: {model6_both.params['log_gdp']:.4f}")
print(f"log_energy coef: {model6_both.params['log_energy']:.4f}")
print(f"log_services coef: {model6_both.params['log_services']:.4f}")

# =============================================================================
# 10. INCOME GROUP HETEROGENEITY
# =============================================================================
print("\n--- Income Group Heterogeneity (Fixed Effects) ---")

income_results = {}
for group in ['Low income', 'Lower middle income', 'Upper middle income', 'High income']:
    df_group = df_fe[df_fe['income_group'] == group]
    m = PanelOLS.from_formula(
        'log_co2 ~ log_gdp + EntityEffects + TimeEffects',
        data=df_group
    ).fit(cov_type='robust')
    coef = m.params['log_gdp']
    se   = m.std_errors['log_gdp']
    ci_l = m.conf_int().loc['log_gdp', 'lower']
    ci_h = m.conf_int().loc['log_gdp', 'upper']
    income_results[group] = {'coef': coef, 'se': se, 'ci_l': ci_l, 'ci_h': ci_h, 'n': m.nobs}
    print(f"{group}: coef={coef:.3f}, SE={se:.3f}, CI=[{ci_l:.3f},{ci_h:.3f}], N={m.nobs}")

# =============================================================================
# 11. UNIT ROOT & COINTEGRATION TESTS
# =============================================================================
print("\n--- Unit Root Tests (ADF) ---")

def panel_adf_test(series_name):
    results = []
    for country in df_clean['Country Code'].unique():
        data = df_clean[df_clean['Country Code'] == country][series_name].dropna()
        if len(data) >= 10:
            pval = adfuller(data, maxlag=1, autolag=None)[1]
            results.append(pval)
    pvals = np.array(results)
    pct_stationary = (pvals < 0.05).mean() * 100
    print(f"{series_name}: {pct_stationary:.1f}% stationary (p<0.05), median p-value: {np.median(pvals):.3f}")

panel_adf_test('log_gdp')
panel_adf_test('log_co2')

print("\n--- Cointegration Test ---")
results_coint = []
for country in df_clean['Country Code'].unique():
    data = df_clean[df_clean['Country Code'] == country][['log_gdp', 'log_co2']].dropna()
    if len(data) >= 10:
        _, pval, _ = coint(data['log_gdp'], data['log_co2'])
        results_coint.append(pval)

pvals = np.array(results_coint)
print(f"Cointegrated: {(pvals < 0.05).mean() * 100:.1f}% of countries, median p-value: {np.median(pvals):.3f}")

# =============================================================================
# 12. FIGURES
# =============================================================================

# Figure 1: Scatter plot GDP vs CO2 (2005 and 2024)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors = {
    'High income': '#2196F3',
    'Upper middle income': '#4CAF50',
    'Lower middle income': '#FF9800',
    'Low income': '#F44336'
}

label_countries = ['Belarus', 'United States', 'China', 'India',
                   'Germany', 'Qatar', 'Chad', 'Congo, Dem. Rep.',
                   'Austria', 'Hungary']

for ax, year, coef, title in zip(axes, [2005, last_year],
                                  [model1.params['log_gdp'], model2.params['log_gdp']],
                                  ['2005', str(last_year)]):
    df_year = df_clean[df_clean['year'] == year].dropna(subset=['income_group'])
    for group, color in colors.items():
        mask = df_year['income_group'] == group
        ax.scatter(df_year[mask]['log_gdp'], df_year[mask]['log_co2'],
                   alpha=0.6, color=color, label=group, s=40)
    m, b = np.polyfit(df_year['log_gdp'], df_year['log_co2'], 1)
    x_line = np.linspace(df_year['log_gdp'].min(), df_year['log_gdp'].max(), 100)
    ax.plot(x_line, m * x_line + b, color='black', linewidth=1.5,
            linestyle='--', label=f'OLS coef = {coef:.2f}')
    for _, row in df_year[df_year['Country Name'].isin(label_countries)].iterrows():
        ax.annotate(row['Country Name'], (row['log_gdp'], row['log_co2']),
                    fontsize=7, alpha=0.8, xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Log GDP per capita (PPP)')
    ax.set_ylabel('Log CO2 per capita')
    ax.set_title(f'GDP vs CO2 per capita ({title})')
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('output/figures/scatter_gdp_co2.png', dpi=150)
plt.close()
print("\nFigure 1 saved: output/figures/scatter_gdp_co2.png")

# Figure 2: Income group coefficient plot
groups  = ['Low\nincome', 'Lower middle\nincome', 'Upper middle\nincome', 'High\nincome']
keys    = ['Low income', 'Lower middle income', 'Upper middle income', 'High income']
coefs   = [income_results[k]['coef'] for k in keys]
ci_low  = [income_results[k]['ci_l'] for k in keys]
ci_high = [income_results[k]['ci_h'] for k in keys]
errors  = [[c - l for c, l in zip(coefs, ci_low)],
           [h - c for c, h in zip(coefs, ci_high)]]

plt.figure(figsize=(8, 5))
plt.errorbar(groups, coefs, yerr=errors, fmt='o', capsize=5,
             color='steelblue', markersize=8)
plt.axhline(y=model6.params['log_gdp'], color='red', linestyle='--',
            alpha=0.7, label=f'Full sample ({model6.params["log_gdp"]:.2f})')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.ylabel('GDP-CO2 Elasticity')
plt.title('GDP-CO2 Elasticity by Income Group\n(Fixed Effects, Robust SE)')
plt.legend()
plt.tight_layout()
plt.savefig('output/figures/income_group_coefs.png', dpi=150)
plt.close()
print("Figure 2 saved: output/figures/income_group_coefs.png")

print("\n=== Analysis complete. All results printed above. ===")


# =============================================================================
# 13. REGRESSION TABLES
# =============================================================================

def stars(p):
    if p < 0.01: return '***'
    elif p < 0.05: return '**'
    elif p < 0.1: return '*'
    return ''

def fmt_coef(coef, se, pval):
    return f"{coef:.4f}{stars(pval)}", f"({se:.4f})"

def make_model_table(title, models, col_names, variables, var_labels, extra_rows, notes):
    """Build one HTML table in PanelOLS summary style"""
    
    ncols = len(models)
    
    # Header
    t = f'<h3 style="margin-bottom:2px">{title}</h3>\n'
    t += '<table>\n'
    
    # Column headers
    t += '<thead>\n<tr>\n'
    t += '<td class="varname"></td>\n'
    for name in col_names:
        t += f'<td class="header">{name}</td>\n'
    t += '</tr>\n</thead>\n<tbody>\n'
    
    # Separator
    t += '<tr class="sep"><td colspan="' + str(ncols+1) + '"></td></tr>\n'
    
    # Coefficient rows
    for var, label in zip(variables, var_labels):
        # Coefficient row
        t += '<tr>\n'
        t += f'<td class="varname">{label}</td>\n'
        for m in models:
            params = m.params if hasattr(m, 'std_errors') else m.params
            ses = m.std_errors if hasattr(m, 'std_errors') else m.bse
            pvals = m.pvalues
            if var in params.index:
                coef, se = fmt_coef(params[var], ses[var], pvals[var])
                t += f'<td class="val">{coef}</td>\n'
            else:
                t += '<td class="val"></td>\n'
        t += '</tr>\n'
        # SE row
        t += '<tr>\n'
        t += '<td class="varname"></td>\n'
        for m in models:
            params = m.params if hasattr(m, 'std_errors') else m.params
            ses = m.std_errors if hasattr(m, 'std_errors') else m.bse
            pvals = m.pvalues
            if var in params.index:
                _, se = fmt_coef(params[var], ses[var], pvals[var])
                t += f'<td class="se">{se}</td>\n'
            else:
                t += '<td class="se"></td>\n'
        t += '</tr>\n'
    
    # Separator before summary stats
    t += '<tr class="sep"><td colspan="' + str(ncols+1) + '"></td></tr>\n'
    
    # Extra rows (N, R², FE, etc.)
    for label, values in extra_rows:
        t += '<tr>\n'
        t += f'<td class="varname">{label}</td>\n'
        for v in values:
            t += f'<td class="val">{v}</td>\n'
        t += '</tr>\n'
    
    # Closing separator
    t += '<tr class="sep-bottom"><td colspan="' + str(ncols+1) + '"></td></tr>\n'
    
    t += '</tbody>\n</table>\n'
    t += f'<p class="note">{notes}</p>\n<br>\n'
    
    return t

# ── Build HTML ────────────────────────────────────────────────────────────────
html = """
<html>
<head>
<meta charset="UTF-8">
<style>
  body {
    font-family: 'Courier New', Courier, monospace;
    margin: 60px;
    color: #000;
    font-size: 13px;
    background: #fff;
  }
  h2 { font-size: 15px; font-weight: bold; margin-bottom: 20px; border-bottom: 2px solid black; padding-bottom: 4px; }
  h3 { font-size: 13px; font-weight: bold; margin-bottom: 2px; text-align: center; }
  .note { font-size: 11px; margin-top: 4px; color: #333; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 4px; font-size: 13px; }
  td { padding: 1px 16px; text-align: right; border: none; }
  td.varname { text-align: left; min-width: 240px; padding-left: 0; }
  td.header { text-align: right; font-weight: bold; border-bottom: 1px solid black; padding-bottom: 3px; }
  td.val { text-align: right; }
  td.se { text-align: right; color: #333; }
  tr.sep td { border-top: 1px solid black; padding: 0; height: 4px; }
  tr.sep-bottom td { border-top: 2px solid black; padding: 0; height: 4px; }
  thead tr td { border-top: 2px solid black; }
</style>
</head>
<body>
<h2>Regression Results: CO&#x2082; Emissions and GDP</h2>
"""

# ── TABLE 1: Cross-section OLS ────────────────────────────────────────────────
html += make_model_table(
    title='Table 1: Cross-Section OLS Models',
    models=[model1, model1_nl, model2, model2_nl],
    col_names=[f'OLS 2005<br>(1)', 'OLS 2005 EKC<br>(2)', f'OLS {last_year}<br>(3)', f'OLS {last_year} EKC<br>(4)'],
    variables=['log_gdp', 'log_gdp_sq'],
    var_labels=['Log GDP per capita', 'Log GDP per capita²'],
    extra_rows=[
        ('Observations', [f'{m.nobs:.0f}' for m in [model1, model1_nl, model2, model2_nl]]),
        ('R²',           [f'{m.rsquared:.3f}' for m in [model1, model1_nl, model2, model2_nl]]),
        ('SE type',      ['HC3 Robust'] * 4),
    ],
    notes='*** p&lt;0.01, ** p&lt;0.05, * p&lt;0.1. HC3 heteroskedasticity-robust SE in parentheses. EKC = quadratic specification.'
)

# ── TABLE 2: First Difference Models ─────────────────────────────────────────
html += make_model_table(
    title='Table 2: First Difference Models',
    models=[model3, model4, model5],
    col_names=['FD No Lags<br>(1)', 'FD 2yr Lags<br>(2)', 'FD 6yr Lags<br>(3)'],
    variables=['d_log_gdp', 'd_log_gdp_lag2', 'd_log_gdp_lag6', 'year'],
    var_labels=['&Delta; Log GDP per capita', '&Delta; Log GDP (lag 2)', '&Delta; Log GDP (lag 6)', 'Time trend'],
    extra_rows=[
        ('Observations', [f'{m.nobs:.0f}' for m in [model3, model4, model5]]),
        ('R²',           [f'{m.rsquared:.3f}' for m in [model3, model4, model5]]),
        ('SE type',      ['HC3 Robust'] * 3),
    ],
    notes='*** p&lt;0.01, ** p&lt;0.05, * p&lt;0.1. HC3 robust SE in parentheses. Dependent variable: &Delta; Log CO&#x2082; per capita.'
)

# ── TABLE 3: Fixed Effects Models ─────────────────────────────────────────────
html += make_model_table(
    title='Table 3: Fixed Effects Models',
    models=[model6, model6_c, model6_srv, model6_both, model6_nl],
    col_names=['Baseline<br>(1)', '+ Energy<br>(2)', '+ Services<br>(3)', '+ Both<br>(4)', 'EKC<br>(5)'],
    variables=['log_gdp', 'log_gdp_sq', 'log_energy', 'log_services'],
    var_labels=['Log GDP per capita', 'Log GDP per capita²', 'Log Energy use', 'Log Services share'],
    extra_rows=[
        ('Country FE',    ['Yes'] * 5),
        ('Time FE',       ['Yes'] * 5),
        ('Observations',  [str(m.nobs) for m in [model6, model6_c, model6_srv, model6_both, model6_nl]]),
        ('Within R²',     [f'{m.rsquared_within:.3f}' for m in [model6, model6_c, model6_srv, model6_both, model6_nl]]),
        ('SE type',       ['Clustered by country'] * 5),
        (f'EKC turning point: ${tp_gdp:,.0f} PPP', [''] * 5),
    ],
    notes='*** p&lt;0.01, ** p&lt;0.05, * p&lt;0.1. SE clustered by country in parentheses. EKC turning point calculated as exp(-&beta;&#x2081;/2&beta;&#x2082;).'
)

# ── TABLE 4: Confounder Models ────────────────────────────────────────────────
html += make_model_table(
    title='Table 4: Confounder Models (Models 1, 4, 6 + Energy)',
    models=[model1, model1_c, model4, model4_c, model6, model6_c],
    col_names=['OLS 2005<br>(1)', 'OLS 2005+E<br>(2)', 'FD 2yr<br>(3)', 'FD 2yr+E<br>(4)', 'FE<br>(5)', 'FE+E<br>(6)'],
    variables=['log_gdp', 'd_log_gdp', 'log_energy', 'd_log_gdp_lag2', 'year'],
    var_labels=['Log GDP per capita', '&Delta; Log GDP per capita', 'Log Energy use', '&Delta; Log GDP (lag 2)', 'Time trend'],
    extra_rows=[
        ('Observations', [f'{model1.nobs:.0f}', f'{model1_c.nobs:.0f}',
                          f'{model4.nobs:.0f}', f'{model4_c.nobs:.0f}',
                          str(model6.nobs), str(model6_c.nobs)]),
        ('R² / Within R²', [f'{model1.rsquared:.3f}', f'{model1_c.rsquared:.3f}',
                             f'{model4.rsquared:.3f}', f'{model4_c.rsquared:.3f}',
                             f'{model6.rsquared_within:.3f}', f'{model6_c.rsquared_within:.3f}']),
    ],
    notes='*** p&lt;0.01, ** p&lt;0.05, * p&lt;0.1. +E = with energy use confounder added.'
)

html += '</body></html>'

with open('output/tables/regression_tables.html', 'w', encoding='utf-8') as f:
    f.write(html)
print("Tables saved: output/tables/regression_tables.html")