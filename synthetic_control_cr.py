import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import itertools

# Read the combined data file
df = pd.read_csv('all_countries_trade_with_china_2000_2019.csv')

# Print data summary
print("\nData summary:")
print("Total number of records:", len(df))
print("Number of unique countries:", df['reporterCode'].nunique())
print("Years covered:", sorted(df['Year'].unique()))

TREATMENT_YEAR = 2007
POST_TREATMENT_YEARS = list(range(TREATMENT_YEAR, 2020))

# Filter for CR data and donor data
cr_data = df[df['reporterCode'] == 188].copy()  # CR reporterCode 188
donor_data = df[df['reporterCode'] != 188].copy()
# Exclude ES (222), DR (214), and Panama (591) from donor pool
donor_data = donor_data[~donor_data['reporterCode'].isin([222, 214, 591])]

# Print data summary
print("\nCR data shape:", cr_data.shape)
print("Donor data shape:", donor_data.shape)
print("Number of donor countries:", donor_data['reporterCode'].nunique())
print("\nCR data columns:", cr_data.columns.tolist())

all_covariates = ['Total_Bilateral_Trade', 'CIFValue', 'FOBValue', 'qty', 'netWgt', 'grossWgt']

covariate_combinations = []
for r in range(1, len(all_covariates) + 1):
    covariate_combinations.extend(list(itertools.combinations(all_covariates, r)))

standardize_options = [True, False]
log_transform_options = [True, False]

def get_all_donors(donor_data):
    return donor_data['reporterCode'].unique()

def get_correlated_donors(donor_data, cr_data, PRE_TREATMENT_YEARS, threshold=0):
    donor_correlations = {}
    cr_years = set(cr_data[cr_data['Year'].isin(PRE_TREATMENT_YEARS)]['Year'])
    for code in donor_data['reporterCode'].unique():
        donor = donor_data[donor_data['reporterCode'] == code]
        donor_years = set(donor[donor['Year'].isin(PRE_TREATMENT_YEARS)]['Year'])
        common_years = sorted(list(cr_years & donor_years))
        if len(common_years) < 10:
            continue  # skip donors with too little overlap
        cr_vals = cr_data[cr_data['Year'].isin(common_years)]['CIFValue']
        donor_vals = donor[donor['Year'].isin(common_years)]['CIFValue']
        if len(cr_vals) != len(donor_vals):
            continue  # skip if still mismatched
        if len(cr_vals) < 2:
            continue  # skip if not enough data for correlation
        correlation = np.corrcoef(donor_vals, cr_vals)[0,1]
        if np.isnan(correlation):
            continue
        if correlation > threshold:
            donor_correlations[code] = correlation
    return np.array(list(donor_correlations.keys()))

def get_similar_volume_donors(donor_data, cr_data, PRE_TREATMENT_YEARS, std_threshold=1):
    cr_mean = cr_data[cr_data['Year'].isin(PRE_TREATMENT_YEARS)]['CIFValue'].mean()
    cr_std = cr_data[cr_data['Year'].isin(PRE_TREATMENT_YEARS)]['CIFValue'].std()
    
    similar_donors = []
    for code in donor_data['reporterCode'].unique():
        donor = donor_data[donor_data['reporterCode'] == code]
        donor_mean = donor[donor['Year'].isin(PRE_TREATMENT_YEARS)]['CIFValue'].mean()
        if abs(donor_mean - cr_mean) <= std_threshold * cr_std:
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
        'correlated': lambda d: get_correlated_donors(d, cr_data, PRE_TREATMENT_YEARS),
        'similar_volume': lambda d: get_similar_volume_donors(d, cr_data, PRE_TREATMENT_YEARS),
        'complete_data': lambda d: get_complete_data_donors(d, PRE_TREATMENT_YEARS)
    }

def safe_log_transform(x):
    return np.log1p(x)  # log1p(x) = log(1 + x) to handle zeros

def calculate_covariate_weights(cr_data, donor_data, covariates, PRE_TREATMENT_YEARS):
    """Calculate weights for covariates based on their importance in predicting the target variable."""
    weights = []
    target = cr_data[cr_data['Year'].isin(PRE_TREATMENT_YEARS)]['CIFValue'].values
    
    for cov in covariates:
        # Calculate correlation between covariate and target for CR
        cr_cov = cr_data[cr_data['Year'].isin(PRE_TREATMENT_YEARS)][cov].values
        if np.all(cr_cov == 0) or np.all(target == 0):
            cr_corr = 0
        else:
            corr_matrix = np.corrcoef(cr_cov, target)
            if np.isnan(corr_matrix[0,1]):
                cr_corr = 0
            else:
                cr_corr = np.abs(corr_matrix[0,1])
        
        # Calculate average correlation across donor countries
        donor_corrs = []
        for code in donor_data['reporterCode'].unique():
            donor = donor_data[donor_data['reporterCode'] == code]
            if len(donor[donor['Year'].isin(PRE_TREATMENT_YEARS)]) == len(PRE_TREATMENT_YEARS):
                donor_cov = donor[donor['Year'].isin(PRE_TREATMENT_YEARS)][cov].values
                donor_target = donor[donor['Year'].isin(PRE_TREATMENT_YEARS)]['CIFValue'].values
                
                if np.all(donor_cov == 0) or np.all(donor_target == 0):
                    continue
                    
                corr_matrix = np.corrcoef(donor_cov, donor_target)
                if not np.isnan(corr_matrix[0,1]):
                    donor_corrs.append(np.abs(corr_matrix[0,1]))
        
        # Combine CR and donor correlations
        if donor_corrs:
            avg_donor_corr = np.mean(donor_corrs)
            weight = (cr_corr + avg_donor_corr) / 2
        else:
            weight = cr_corr
        
        weights.append(weight)
    
    # Normalize weights to sum to 1, with a minimum weight of 0.1
    weights = np.array(weights)
    weights = np.maximum(weights, 0.1)  # Ensure minimum weight
    weights = weights / np.sum(weights)
    return weights

best_overall_rmse = float('inf')
best_overall_result = None
best_overall_start_year = None

# Only use 2000 as the start year
for start_year in [2000]:  # Changed from [2000, 2001, 2002, 2003, 2004, 2005]
    PRE_TREATMENT_YEARS = list(range(start_year, TREATMENT_YEAR))
    ALL_YEARS = PRE_TREATMENT_YEARS + POST_TREATMENT_YEARS
    donor_pool_options = get_donor_pool_options(PRE_TREATMENT_YEARS)

    best_rmse = float('inf')
    best_result = None

    for covariates, standardize, log_transform, pool_name in itertools.product(
        covariate_combinations, standardize_options, log_transform_options, donor_pool_options.keys()
    ):
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
        
        if log_transform:
            all_pre_matrix = safe_log_transform(all_pre_matrix)
        
        mean = all_pre_matrix.mean(axis=0)
        std = all_pre_matrix.std(axis=0)
        std[std == 0] = 1

        cr_pre = cr_data.set_index('Year').reindex(PRE_TREATMENT_YEARS)[list(covariates)].fillna(0).values.T
        if log_transform:
            cr_pre = safe_log_transform(cr_pre)

        donor_pre_matrix = np.zeros((num_covs, num_years, num_donors))
        for i, code in enumerate(donor_countries):
            donor_country = donor_data[donor_data['reporterCode'] == code].set_index('Year').reindex(PRE_TREATMENT_YEARS)[list(covariates)].fillna(0).values.T
            if log_transform:
                donor_country = safe_log_transform(donor_country)
            donor_pre_matrix[:, :, i] = donor_country

        if standardize:
            cr_pre = (cr_pre - mean[:, None]) / std[:, None]
            for i in range(num_donors):
                donor_pre_matrix[:, :, i] = (donor_pre_matrix[:, :, i] - mean[:, None]) / std[:, None]

        def synthetic_control_objective(weights, donor_matrix, cr_matrix, reg_param=1e-6, penalty_param=1e3):
            synthetic = np.tensordot(donor_matrix, weights, axes=([2], [0]))
            reg_term = reg_param * np.sum(weights ** 2)
            penalty_term = penalty_param * (np.sum(weights) - 1) ** 2 # TODO: check reg and penalty
            return np.sum((synthetic - cr_matrix) ** 2) + reg_term + penalty_term

        initial_weights = np.ones(num_donors) / num_donors
        bounds = [(0, 1) for _ in range(num_donors)]

        result = minimize(
            synthetic_control_objective,
            initial_weights,
            args=(donor_pre_matrix, cr_pre),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-8}
        )
        if not result.success:
            continue
        optimal_weights = result.x / np.sum(result.x)

        # synthetic control for all years
        cr_actual = cr_data.set_index('Year').reindex(ALL_YEARS)['CIFValue'].fillna(0).values
        if log_transform:
            cr_actual = safe_log_transform(cr_actual)
        
        synthetic_control = np.zeros(len(ALL_YEARS))
        for i, code in enumerate(donor_countries):
            donor_country = donor_data[donor_data['reporterCode'] == code].set_index('Year').reindex(ALL_YEARS)['CIFValue'].fillna(0).values
            if log_transform:
                donor_country = safe_log_transform(donor_country)
            synthetic_control += optimal_weights[i] * donor_country

        # Convert back from log scale if needed
        if log_transform:
            cr_actual = np.expm1(cr_actual)
            synthetic_control = np.expm1(synthetic_control)

        # treatment effect
        treatment_effect = cr_actual[len(PRE_TREATMENT_YEARS):] - synthetic_control[len(PRE_TREATMENT_YEARS):]

        # pre-treatment RMSE
        pre_treatment_rmse = np.sqrt(np.mean((cr_actual[:len(PRE_TREATMENT_YEARS)] - 
                                            synthetic_control[:len(PRE_TREATMENT_YEARS)]) ** 2))

        if pre_treatment_rmse < best_rmse:
            # Calculate covariate weights
            covariate_weights = calculate_covariate_weights(cr_data, donor_data, covariates, PRE_TREATMENT_YEARS)
            
            best_rmse = pre_treatment_rmse
            best_result = {
                'covariates': covariates,
                'standardize': standardize,
                'log_transform': log_transform,
                'donor_pool': pool_name,
                'weights': optimal_weights,
                'donor_countries': donor_countries,
                'synthetic_control': synthetic_control,
                'cr_actual': cr_actual,
                'treatment_effect': treatment_effect,
                'pre_treatment_rmse': pre_treatment_rmse,
                'start_year': start_year,
                'covariate_weights': covariate_weights
            }
    if best_result and best_rmse < best_overall_rmse:
        best_overall_rmse = best_rmse
        best_overall_result = best_result
        best_overall_start_year = start_year
        
        # Plot synthetic control
        plt.figure(figsize=(12, 6))
        plt.plot(list(range(start_year, 2020)), best_result['cr_actual'], label='Costa Rica', linewidth=2)
        plt.plot(list(range(start_year, 2020)), best_result['synthetic_control'], label='Synthetic Control', linewidth=2)
        plt.axvline(x=TREATMENT_YEAR, color='r', linestyle='--', label='Treatment Year')
        plt.xlabel('Year')
        plt.ylabel('Imports from China (CIF Value)')
        plt.title(f'Best Synthetic Control: {best_result["covariates"]} | Standardize={best_result["standardize"]} | Log={best_result["log_transform"]} | Pool={best_result["donor_pool"]} | Start={start_year}')
        plt.legend()
        plt.grid(True)
        plt.savefig('best_synthetic_control_cr.png')
        plt.close()
        
        # Plot covariate weights
        plt.figure(figsize=(10, 6))
        plt.bar(best_result['covariates'], best_result['covariate_weights'])
        plt.title('Covariate Weights in Synthetic Control')
        plt.xlabel('Covariates')
        plt.ylabel('Weight')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('covariate_weights_cr.png')
        plt.close()
        
        with open('best_synthetic_control_cr_results.txt', 'w') as f:
            f.write(f'Best Covariates: {best_result["covariates"]}\n')
            f.write(f'Standardized: {best_result["standardize"]}\n')
            f.write(f'Log Transformed: {best_result["log_transform"]}\n')
            f.write(f'Donor Pool: {best_result["donor_pool"]}\n')
            f.write(f'Pre-treatment Start Year: {start_year}\n')
            f.write(f'Pre-treatment RMSE: {best_result["pre_treatment_rmse"]:,.2f}\n')
            f.write('\nCovariate Weights:\n')
            for cov, weight in zip(best_result['covariates'], best_result['covariate_weights']):
                f.write(f'{cov}: {weight:.6f}\n')
            f.write('\nAll Donor Weights:\n')
            for country, weight in zip(best_result['donor_countries'], best_result['weights']):
                f.write(f'Country {country}: {weight:.6f}\n')
            f.write('\nTreatment Effect (2008-2019):\n')
            for i, year in enumerate(POST_TREATMENT_YEARS):
                f.write(f'{year}: {best_result["treatment_effect"][i]:,.2f}\n')

print('Best configuration across all pre-treatment start years:')
print(f'Pre-treatment Start Year: {best_overall_result["start_year"]}')
print(f'Covariates: {best_overall_result["covariates"]}')
print(f'Standardized: {best_overall_result["standardize"]}')
print(f'Log Transformed: {best_overall_result["log_transform"]}')
print(f'Donor Pool: {best_overall_result["donor_pool"]}')
print(f'Pre-treatment RMSE: {best_overall_result["pre_treatment_rmse"]:,.2f}')
print('\nCovariate Weights:')
for cov, weight in zip(best_overall_result['covariates'], best_overall_result['covariate_weights']):
    print(f'{cov}: {weight:.6f}')
print('\nAll Donor Weights:')
for country, weight in zip(best_overall_result['donor_countries'], best_overall_result['weights']):
    print(f'Country {country}: {weight:.6f}')
print('\nTreatment Effect (2008-2019):')
for i, year in enumerate(POST_TREATMENT_YEARS):
    print(f'{year}: {best_overall_result["treatment_effect"][i]:,.2f}') 