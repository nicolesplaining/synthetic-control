import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import itertools

df = pd.read_csv('all_countries_trade_with_china_2000_2019.csv')

TREATMENT_YEAR = 2017
POST_TREATMENT_YEARS = list(range(TREATMENT_YEAR, 2020))

panama_data = df[df['reporterCode'] == 591].copy() # panama reporterCode 591
donor_data = df[df['reporterCode'] != 591].copy()

all_covariates = ['Total_Bilateral_Trade', 'CIFValue', 'FOBValue', 'qty', 'netWgt', 'grossWgt']

covariate_combinations = []
for r in range(1, len(all_covariates) + 1):
    covariate_combinations.extend(list(itertools.combinations(all_covariates, r)))

standardize_options = [True, False]

def get_all_donors(donor_data):
    return donor_data['reporterCode'].unique()

def get_correlated_donors(donor_data, panama_data, PRE_TREATMENT_YEARS, threshold=0):
    donor_correlations = {}
    panama_years = set(panama_data[panama_data['Year'].isin(PRE_TREATMENT_YEARS)]['Year'])
    for code in donor_data['reporterCode'].unique():
        donor = donor_data[donor_data['reporterCode'] == code]
        donor_years = set(donor[donor['Year'].isin(PRE_TREATMENT_YEARS)]['Year'])
        common_years = sorted(list(panama_years & donor_years))
        if len(common_years) < 10:
            continue  # skip donors with too little overlap
        panama_vals = panama_data[panama_data['Year'].isin(common_years)]['Total_Bilateral_Trade']
        donor_vals = donor[donor['Year'].isin(common_years)]['Total_Bilateral_Trade']
        if len(panama_vals) != len(donor_vals):
            continue  # skip if still mismatched
        if len(panama_vals) < 2:
            continue  # skip if not enough data for correlation
        correlation = np.corrcoef(donor_vals, panama_vals)[0,1]
        if np.isnan(correlation):
            continue
        if correlation > threshold:
            donor_correlations[code] = correlation
    return np.array(list(donor_correlations.keys()))

def get_similar_volume_donors(donor_data, panama_data, PRE_TREATMENT_YEARS, std_threshold=1):
    panama_mean = panama_data[panama_data['Year'].isin(PRE_TREATMENT_YEARS)]['Total_Bilateral_Trade'].mean()
    panama_std = panama_data[panama_data['Year'].isin(PRE_TREATMENT_YEARS)]['Total_Bilateral_Trade'].std()
    
    similar_donors = []
    for code in donor_data['reporterCode'].unique():
        donor = donor_data[donor_data['reporterCode'] == code]
        donor_mean = donor[donor['Year'].isin(PRE_TREATMENT_YEARS)]['Total_Bilateral_Trade'].mean()
        if abs(donor_mean - panama_mean) <= std_threshold * panama_std:
            similar_donors.append(code)
    return np.array(similar_donors)

def get_complete_data_donors(donor_data, PRE_TREATMENT_YEARS):
    complete_donors = []
    for code in donor_data['reporterCode'].unique():
        donor = donor_data[donor_data['reporterCode'] == code]
        if len(donor[donor['Year'].isin(PRE_TREATMENT_YEARS)]) == len(PRE_TREATMENT_YEARS):
            complete_donors.append(code)
    return np.array(complete_donors)

def get_donor_pool_options(PRE_TREATMENT_YEARS):
    return {
        'all': lambda d: get_all_donors(d),
        'correlated': lambda d: get_correlated_donors(d, panama_data, PRE_TREATMENT_YEARS),
        'similar_volume': lambda d: get_similar_volume_donors(d, panama_data, PRE_TREATMENT_YEARS),
        'complete_data': lambda d: get_complete_data_donors(d, PRE_TREATMENT_YEARS)
    }

best_overall_rmse = float('inf')
best_overall_result = None
best_overall_start_year = None

for start_year in range(2000, 2007):
    PRE_TREATMENT_YEARS = list(range(start_year, TREATMENT_YEAR))
    ALL_YEARS = PRE_TREATMENT_YEARS + POST_TREATMENT_YEARS
    donor_pool_options = get_donor_pool_options(PRE_TREATMENT_YEARS)

    best_rmse = float('inf')
    best_result = None

    for covariates, standardize, pool_name in itertools.product(covariate_combinations, standardize_options, donor_pool_options.keys()):
        # donor pool
        donor_countries = donor_pool_options[pool_name](donor_data)
        if len(donor_countries) < 2:
            continue
        num_donors = len(donor_countries)
        num_years = len(PRE_TREATMENT_YEARS)
        num_covs = len(covariates)

        # pre-treatment matrix for all countries (for standardization)
        all_pre = df[df['Year'].isin(PRE_TREATMENT_YEARS)].copy()
        all_pre_matrix = all_pre.pivot(index='Year', columns='reporterCode')[list(covariates)].fillna(0).values.reshape(-1, num_covs)
        mean = all_pre_matrix.mean(axis=0)
        std = all_pre_matrix.std(axis=0)
        std[std == 0] = 1

        # panama pre-treatment matrix (covariates × years)
        panama_pre = panama_data.set_index('Year').reindex(PRE_TREATMENT_YEARS)[list(covariates)].fillna(0).values.T

        # donor pre-treatment matrix (covariates × years × donors)
        donor_pre_matrix = np.zeros((num_covs, num_years, num_donors))
        for i, code in enumerate(donor_countries):
            donor_country = donor_data[donor_data['reporterCode'] == code].set_index('Year').reindex(PRE_TREATMENT_YEARS)[list(covariates)].fillna(0).values.T
            donor_pre_matrix[:, :, i] = donor_country

        # standardize
        if standardize:
            panama_pre = (panama_pre - mean[:, None]) / std[:, None]
            for i in range(num_donors):
                donor_pre_matrix[:, :, i] = (donor_pre_matrix[:, :, i] - mean[:, None]) / std[:, None]

        # obj function with regularization and penalty
        def synthetic_control_objective(weights, donor_matrix, panama_matrix, reg_param=1e-6, penalty_param=1e3):
            synthetic = np.tensordot(donor_matrix, weights, axes=([2], [0]))
            reg_term = reg_param * np.sum(weights ** 2)
            penalty_term = penalty_param * (np.sum(weights) - 1) ** 2
            return np.sum((synthetic - panama_matrix) ** 2) + reg_term + penalty_term

        initial_weights = np.ones(num_donors) / num_donors
        bounds = [(0, 1) for _ in range(num_donors)]

        # L-BFGS-B optimization
        result = minimize(
            synthetic_control_objective,
            initial_weights,
            args=(donor_pre_matrix, panama_pre),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-8}
        )
        if not result.success:
            continue
        optimal_weights = result.x / np.sum(result.x)

        # synthetic control for all years (unstandardized)
        panama_actual = panama_data.set_index('Year').reindex(ALL_YEARS)['Total_Bilateral_Trade'].fillna(0).values
        synthetic_control = np.zeros(len(ALL_YEARS))
        for i, code in enumerate(donor_countries):
            donor_country = donor_data[donor_data['reporterCode'] == code].set_index('Year').reindex(ALL_YEARS)['Total_Bilateral_Trade'].fillna(0).values
            synthetic_control += optimal_weights[i] * donor_country

        # treatment effect
        treatment_effect = panama_actual[len(PRE_TREATMENT_YEARS):] - synthetic_control[len(PRE_TREATMENT_YEARS):]

        # pre-treatment RMSE
        pre_treatment_rmse = np.sqrt(np.mean((panama_actual[:len(PRE_TREATMENT_YEARS)] - 
                                            synthetic_control[:len(PRE_TREATMENT_YEARS)]) ** 2))

        if pre_treatment_rmse < best_rmse:
            best_rmse = pre_treatment_rmse
            best_result = {
                'covariates': covariates,
                'standardize': standardize,
                'donor_pool': pool_name,
                'weights': optimal_weights,
                'donor_countries': donor_countries,
                'synthetic_control': synthetic_control,
                'panama_actual': panama_actual,
                'treatment_effect': treatment_effect,
                'pre_treatment_rmse': pre_treatment_rmse,
                'start_year': start_year
            }
    if best_result and best_rmse < best_overall_rmse:
        best_overall_rmse = best_rmse
        best_overall_result = best_result
        best_overall_start_year = start_year
        plt.figure(figsize=(12, 6))
        plt.plot(list(range(start_year, 2020)), best_result['panama_actual'], label='Panama', linewidth=2)
        plt.plot(list(range(start_year, 2020)), best_result['synthetic_control'], label='Synthetic Control', linewidth=2)
        plt.axvline(x=TREATMENT_YEAR, color='r', linestyle='--', label='Treatment Year')
        plt.xlabel('Year')
        plt.ylabel('Total Bilateral Trade with China')
        plt.title(f'Best Synthetic Control: {best_result["covariates"]} | Standardize={best_result["standardize"]} | Pool={best_result["donor_pool"]} | Start={start_year}')
        plt.legend()
        plt.grid(True)
        plt.savefig('best_synthetic_control_panama.png')
        plt.close()
        with open('best_synthetic_control_results.txt', 'w') as f:
            f.write(f'Best Covariates: {best_result["covariates"]}\n')
            f.write(f'Standardized: {best_result["standardize"]}\n')
            f.write(f'Donor Pool: {best_result["donor_pool"]}\n')
            f.write(f'Pre-treatment Start Year: {start_year}\n')
            f.write(f'Pre-treatment RMSE: {best_result["pre_treatment_rmse"]:,.2f}\n')
            f.write('\nAll Donor Weights:\n')
            for country, weight in zip(best_result['donor_countries'], best_result['weights']):
                f.write(f'Country {country}: {weight:.6f}\n')
            f.write('\nTreatment Effect (2018-2019):\n')
            for i, year in enumerate(POST_TREATMENT_YEARS):
                f.write(f'{year}: {best_result["treatment_effect"][i]:,.2f}\n')

# After the main loop, save the best 'all' donor pool result for the best pre-treatment start year
if best_overall_result is not None:
    # Find the best 'all' donor pool result for the best start year
    TREATMENT_YEAR = 2017
    POST_TREATMENT_YEARS = list(range(TREATMENT_YEAR, 2020))
    start_year = best_overall_result['start_year']
    PRE_TREATMENT_YEARS = list(range(start_year, TREATMENT_YEAR))
    ALL_YEARS = PRE_TREATMENT_YEARS + POST_TREATMENT_YEARS
    donor_pool_options = get_donor_pool_options(PRE_TREATMENT_YEARS)
    best_all_rmse = float('inf')
    best_all_result = None
    for covariates, standardize in itertools.product(covariate_combinations, standardize_options):
        donor_countries = donor_pool_options['all'](donor_data)
        if len(donor_countries) < 2:
            continue
        num_donors = len(donor_countries)
        num_years = len(PRE_TREATMENT_YEARS)
        num_covs = len(covariates)
        all_pre = df[df['Year'].isin(PRE_TREATMENT_YEARS)].copy()
        all_pre_matrix = all_pre.pivot(index='Year', columns='reporterCode')[list(covariates)].fillna(0).values.reshape(-1, num_covs)
        mean = all_pre_matrix.mean(axis=0)
        std = all_pre_matrix.std(axis=0)
        std[std == 0] = 1
        panama_pre = panama_data.set_index('Year').reindex(PRE_TREATMENT_YEARS)[list(covariates)].fillna(0).values.T
        donor_pre_matrix = np.zeros((num_covs, num_years, num_donors))
        for i, code in enumerate(donor_countries):
            donor_country = donor_data[donor_data['reporterCode'] == code].set_index('Year').reindex(PRE_TREATMENT_YEARS)[list(covariates)].fillna(0).values.T
            donor_pre_matrix[:, :, i] = donor_country
        if standardize:
            panama_pre = (panama_pre - mean[:, None]) / std[:, None]
            for i in range(num_donors):
                donor_pre_matrix[:, :, i] = (donor_pre_matrix[:, :, i] - mean[:, None]) / std[:, None]
        def synthetic_control_objective(weights, donor_matrix, panama_matrix, reg_param=1e-6, penalty_param=1e3):
            synthetic = np.tensordot(donor_matrix, weights, axes=([2], [0]))
            reg_term = reg_param * np.sum(weights ** 2)
            penalty_term = penalty_param * (np.sum(weights) - 1) ** 2
            return np.sum((synthetic - panama_matrix) ** 2) + reg_term + penalty_term
        initial_weights = np.ones(num_donors) / num_donors
        bounds = [(0, 1) for _ in range(num_donors)]
        result = minimize(
            synthetic_control_objective,
            initial_weights,
            args=(donor_pre_matrix, panama_pre),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-8}
        )
        if not result.success:
            continue
        optimal_weights = result.x / np.sum(result.x)
        panama_actual = panama_data.set_index('Year').reindex(ALL_YEARS)['Total_Bilateral_Trade'].fillna(0).values
        synthetic_control = np.zeros(len(ALL_YEARS))
        for i, code in enumerate(donor_countries):
            donor_country = donor_data[donor_data['reporterCode'] == code].set_index('Year').reindex(ALL_YEARS)['Total_Bilateral_Trade'].fillna(0).values
            synthetic_control += optimal_weights[i] * donor_country
        treatment_effect = panama_actual[len(PRE_TREATMENT_YEARS):] - synthetic_control[len(PRE_TREATMENT_YEARS):]
        pre_treatment_rmse = np.sqrt(np.mean((panama_actual[:len(PRE_TREATMENT_YEARS)] - 
                                            synthetic_control[:len(PRE_TREATMENT_YEARS)]) ** 2))
        if pre_treatment_rmse < best_all_rmse:
            best_all_rmse = pre_treatment_rmse
            best_all_result = {
                'covariates': covariates,
                'standardize': standardize,
                'donor_pool': 'all',
                'weights': optimal_weights,
                'donor_countries': donor_countries,
                'synthetic_control': synthetic_control,
                'panama_actual': panama_actual,
                'treatment_effect': treatment_effect,
                'pre_treatment_rmse': pre_treatment_rmse,
                'start_year': start_year
            }
    if best_all_result is not None:
        plt.figure(figsize=(12, 6))
        plt.plot(list(range(start_year, 2020)), best_all_result['panama_actual'], label='Panama', linewidth=2)
        plt.plot(list(range(start_year, 2020)), best_all_result['synthetic_control'], label='Synthetic Control (All Donors)', linewidth=2)
        plt.axvline(x=TREATMENT_YEAR, color='r', linestyle='--', label='Treatment Year')
        plt.xlabel('Year')
        plt.ylabel('Total Bilateral Trade with China')
        plt.title(f'Synthetic Control (All Donors): {best_all_result["covariates"]} | Standardize={best_all_result["standardize"]} | Start={start_year}')
        plt.legend()
        plt.grid(True)
        plt.savefig('synthetic_control_all_donors.png')
        plt.close()
        with open('synthetic_control_all_donors_results.txt', 'w') as f:
            f.write(f'Covariates: {best_all_result["covariates"]}\n')
            f.write(f'Standardized: {best_all_result["standardize"]}\n')
            f.write(f'Donor Pool: all\n')
            f.write(f'Pre-treatment Start Year: {start_year}\n')
            f.write(f'Pre-treatment RMSE: {best_all_result["pre_treatment_rmse"]:,.2f}\n')
            f.write('\nAll Donor Weights:\n')
            for country, weight in zip(best_all_result['donor_countries'], best_all_result['weights']):
                f.write(f'Country {country}: {weight:.6f}\n')
            f.write('\nTreatment Effect (2018-2019):\n')
            for i, year in enumerate(POST_TREATMENT_YEARS):
                f.write(f'{year}: {best_all_result["treatment_effect"][i]:,.2f}\n')

print('Best configuration across all pre-treatment start years:')
print(f'Pre-treatment Start Year: {best_overall_result["start_year"]}')
print(f'Covariates: {best_overall_result["covariates"]}')
print(f'Standardized: {best_overall_result["standardize"]}')
print(f'Donor Pool: {best_overall_result["donor_pool"]}')
print(f'Pre-treatment RMSE: {best_overall_result["pre_treatment_rmse"]:,.2f}')
print('\nAll Donor Weights:')
for country, weight in zip(best_overall_result['donor_countries'], best_overall_result['weights']):
    print(f'Country {country}: {weight:.6f}')
print('\nTreatment Effect (2018-2019):')
for i, year in enumerate(POST_TREATMENT_YEARS):
    print(f'{year}: {best_overall_result["treatment_effect"][i]:,.2f}') 