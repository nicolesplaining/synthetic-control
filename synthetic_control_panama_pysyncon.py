import pandas as pd
import numpy as np
import pysyncon
import matplotlib.pyplot as plt

# 1. Read data and ensure dtype consistency
df = pd.read_csv('all_countries_trade_with_china_2000_2019.csv')
df['reporterCode'] = df['reporterCode'].astype(int)  # ensure integer

# 2. Parameters
TREATMENT_UNIT = 591
TREATMENT_YEAR = 2017
PRE_TREATMENT_YEARS = list(range(2000, TREATMENT_YEAR))   # [2000..2016]
POST_TREATMENT_YEARS = list(range(TREATMENT_YEAR, 2020))  # [2017..2019]
ALL_YEARS = PRE_TREATMENT_YEARS + POST_TREATMENT_YEARS

outcome_var = 'Total_Bilateral_Trade'
predictors = ['CIFValue', 'FOBValue', 'qty', 'netWgt', 'grossWgt']

# 3. Dataprep
prep = pysyncon.Dataprep(
    df,
    predictors=predictors,
    predictors_op="mean",
    dependent=outcome_var,
    unit_variable='reporterCode',
    time_variable='Year',
    treatment_identifier=TREATMENT_UNIT,
    controls_identifier=[c for c in df['reporterCode'].unique() if c != TREATMENT_UNIT],
    time_predictors_prior=PRE_TREATMENT_YEARS,
    time_optimize_ssr=PRE_TREATMENT_YEARS
)

# 4. (Optional) inspect prep.foo to confirm shape
if hasattr(prep, 'foo'):
    foo = prep.foo.copy().sort_values(['reporterCode','Year'])
    print("prep.foo shape:", foo.shape)
    print("Unique reporterCode:", foo['reporterCode'].nunique())
    print("Year range for treatment unit:", foo[foo['reporterCode'] == TREATMENT_UNIT]['Year'].unique())

# 5. Fit Synth
synth = pysyncon.Synth()
synth.fit(dataprep=prep)

# 6. Extract donor weights
try:
    weights = synth.weights()
except:
    # if weights() isn’t available, try synth.weights_ or synth.W
    weights = getattr(synth, 'weights_', None) or getattr(synth, 'W', None)
print("Donor weights (nonzero):")
for c, w in weights.items():
    if w > 1e-3:
        print(f"  Country {c}: {w:.4f}")

# 7. Get full synthetic trajectory via public API (if supported)
#    Otherwise, fallback to private methods
if hasattr(synth, 'predict'):
    Y_synth_full = synth.predict()       # should be length 20
    gaps_full = synth.gaps_             # should be length 20
else:
    # Private fallback (but check shapes first!)
    Z0, Z1 = prep.make_outcome_mats()
    Y_synth_full = synth._synthetic(Z0)
    gaps_full = synth._gaps(Z0, Z1)

# 8. “Actual” series: sort by Year to ensure correct order
subset = prep.foo[prep.foo['reporterCode'] == TREATMENT_UNIT].sort_values('Year')
actual = subset[outcome_var].values      # length should be 20
years = subset['Year'].values

# 9. Compute pre‐treatment RMSE
n_pre = len(PRE_TREATMENT_YEARS)  # 17
rmse_pre = np.sqrt(np.mean((actual[:n_pre] - Y_synth_full[:n_pre])**2))
print(f"Pre‐treatment RMSE: {rmse_pre:.2f}")

# 10. Print post‐treatment gaps
print("\nPost‐treatment gaps:")
for offset, year in enumerate(POST_TREATMENT_YEARS):
    idx = n_pre + offset
    print(f"  {year}: {gaps_full.iloc[idx]:.2f}")

# 11. Plot actual vs synthetic
plt.figure(figsize=(10,6))
plt.plot(years, actual, label='Panama (actual)', linewidth=2)
plt.plot(ALL_YEARS, Y_synth_full, label='Synthetic Control', linewidth=2)
plt.axvline(x=TREATMENT_YEAR, color='red', linestyle='--', label='Treatment')
plt.xlabel('Year')
plt.ylabel('Total Bilateral Trade')
plt.title("Panama’s Trade vs. Synthetic Control (2000–2019)")
plt.legend()
plt.grid(True)
plt.savefig('synthetic_control_vs_actual.png')
plt.close()

# 12. Plot post‐treatment treatment effect only
plt.figure(figsize=(10,6))
post_gaps = pd.Series(gaps_full, index=ALL_YEARS).loc[POST_TREATMENT_YEARS]
plt.plot(POST_TREATMENT_YEARS, post_gaps, marker='o')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Treatment Effect')
plt.title("Treatment Effect on Panama’s Trade (2017–2019)")
plt.grid(True)
plt.savefig('post_treatment_effect.png')
plt.close()
