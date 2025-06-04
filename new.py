import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import itertools

# Load data
df = pd.read_csv('all_countries_trade_with_china_2000_2019.csv')

# Parameters
TREATMENT_YEAR = 2017
TREATMENT_COUNTRY_CODE = 591  # Panama

# Separate treated unit from donor pool
panama_data = df[df['reporterCode'] == TREATMENT_COUNTRY_CODE].copy()
donor_data = df[df['reporterCode'] != TREATMENT_COUNTRY_CODE].copy()

def get_donor_pools(donor_data, panama_data, pre_treatment_years):
    """Create different donor pools based on various criteria"""
    
    def get_all_donors():
        return donor_data['reporterCode'].unique()
    
    def get_correlated_donors(threshold=0.3, min_years=10):
        """Select donors with positive correlation in pre-treatment period"""
        donor_correlations = {}
        panama_pre = panama_data[panama_data['Year'].isin(pre_treatment_years)]
        panama_pre = panama_pre.set_index('Year')['Total_Bilateral_Trade']
        
        for code in donor_data['reporterCode'].unique():
            donor_pre = donor_data[
                (donor_data['reporterCode'] == code) & 
                (donor_data['Year'].isin(pre_treatment_years))
            ]
            
            if len(donor_pre) < min_years:
                continue
                
            donor_pre = donor_pre.set_index('Year')['Total_Bilateral_Trade']
            
            # Get common years
            common_years = panama_pre.index.intersection(donor_pre.index)
            if len(common_years) < min_years:
                continue
                
            panama_vals = panama_pre.loc[common_years]
            donor_vals = donor_pre.loc[common_years]
            
            if len(panama_vals) >= 2 and panama_vals.std() > 0 and donor_vals.std() > 0:
                correlation = np.corrcoef(panama_vals, donor_vals)[0, 1]
                if not np.isnan(correlation) and correlation > threshold:
                    donor_correlations[code] = correlation
        
        return np.array(list(donor_correlations.keys()))
    
    def get_complete_data_donors():
        """Select donors with complete data in pre-treatment period"""
        complete_donors = []
        for code in donor_data['reporterCode'].unique():
            donor_years = donor_data[donor_data['reporterCode'] == code]['Year'].unique()
            if all(year in donor_years for year in pre_treatment_years):
                complete_donors.append(code)
        return np.array(complete_donors)
    
    def get_similar_scale_donors(std_multiplier=2):
        """Select donors with similar trade volume scale"""
        panama_mean = panama_data[
            panama_data['Year'].isin(pre_treatment_years)
        ]['Total_Bilateral_Trade'].mean()
        panama_std = panama_data[
            panama_data['Year'].isin(pre_treatment_years)
        ]['Total_Bilateral_Trade'].std()
        
        similar_donors = []
        for code in donor_data['reporterCode'].unique():
            donor_mean = donor_data[
                (donor_data['reporterCode'] == code) & 
                (donor_data['Year'].isin(pre_treatment_years))
            ]['Total_Bilateral_Trade'].mean()
            
            if abs(donor_mean - panama_mean) <= std_multiplier * panama_std:
                similar_donors.append(code)
        
        return np.array(similar_donors)
    
    return {
        'all': get_all_donors(),
        'correlated': get_correlated_donors(),
        'complete_data': get_complete_data_donors(),
        'similar_scale': get_similar_scale_donors()
    }

def prepare_data_for_synthetic_control(panama_data, donor_data, donor_codes, years, 
                                      standardize=False, log_transform=False):
    """Prepare outcome data in correct format for synthetic control with preprocessing options"""
    
    # Panama outcomes (treated unit)
    panama_outcomes = panama_data[panama_data['Year'].isin(years)].set_index('Year')
    panama_outcomes = panama_outcomes.reindex(years)['Total_Bilateral_Trade'].fillna(0).values
    
    # Donor outcomes matrix (donors x time periods)
    donor_outcomes = []
    valid_donors = []
    
    for code in donor_codes:
        donor_country = donor_data[donor_data['reporterCode'] == code]
        donor_country = donor_country[donor_country['Year'].isin(years)].set_index('Year')
        donor_country = donor_country.reindex(years)['Total_Bilateral_Trade'].fillna(0).values
        
        # Only include donors with non-zero variance and positive values (for log transform)
        if log_transform and np.any(donor_country <= 0):
            continue
        if np.std(donor_country) > 0:
            donor_outcomes.append(donor_country)
            valid_donors.append(code)
    
    if len(donor_outcomes) == 0:
        return None, None, None
    
    donor_outcomes = np.array(donor_outcomes)
    valid_donors = np.array(valid_donors)
    
    # Apply log transformation if requested
    if log_transform:
        if np.any(panama_outcomes <= 0):
            print("    Warning: Panama has zero/negative values, cannot log transform")
            return None, None, None
        panama_outcomes = np.log(panama_outcomes)
        donor_outcomes = np.log(donor_outcomes)
    
    # Apply standardization if requested
    if standardize:
        # Calculate mean and std from all data (Panama + donors)
        all_data = np.concatenate([panama_outcomes.reshape(1, -1), donor_outcomes])
        mean_val = np.mean(all_data)
        std_val = np.std(all_data)
        
        if std_val == 0:
            print("    Warning: Zero variance in data, cannot standardize")
            return None, None, None
        
        panama_outcomes = (panama_outcomes - mean_val) / std_val
        donor_outcomes = (donor_outcomes - mean_val) / std_val
    
    return panama_outcomes, donor_outcomes, valid_donors

def synthetic_control_weights(panama_pre, donor_pre, debug=False):
    """
    Standard synthetic control optimization with debugging
    """
    n_donors = len(donor_pre)
    
    if debug:
        print(f"      Debug: {n_donors} donors, {len(panama_pre)} time periods")
        print(f"      Panama pre-treatment: {panama_pre}")
        print(f"      Donor shapes: {donor_pre.shape}")
        print(f"      Donor ranges: min={donor_pre.min():.2f}, max={donor_pre.max():.2f}")
    
    def objective(weights):
        synthetic = np.dot(weights, donor_pre)
        mse = np.sum((panama_pre - synthetic) ** 2)
        if debug and len(weights) == n_donors:  # Only print for valid weight vectors
            print(f"      Trying weights sum={weights.sum():.6f}, MSE={mse:.6f}")
        return mse
    
    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    # Bounds: weights between 0 and 1
    bounds = [(0, 1) for _ in range(n_donors)]
    
    # Try multiple methods and initial conditions
    methods = ['SLSQP', 'trust-constr']
    initial_conditions = [
        np.ones(n_donors) / n_donors,  # Equal weights
        np.random.dirichlet(np.ones(n_donors)),  # Random weights
        np.zeros(n_donors)  # Start with zeros
    ]
    initial_conditions[2][0] = 1.0  # Give all weight to first donor initially
    
    best_result = None
    best_objective = float('inf')
    
    for method in methods:
        for i, initial_weights in enumerate(initial_conditions):
            if debug:
                print(f"      Trying method {method}, initial condition {i}")
                print(f"      Initial weights: {initial_weights}")
            
            try:
                result = minimize(
                    objective,
                    initial_weights,
                    method=method,
                    bounds=bounds,
                    constraints=constraints,
                    options={'ftol': 1e-12, 'maxiter': 5000, 'disp': debug}
                )
                
                if debug:
                    print(f"      Result: success={result.success}, fun={result.fun:.8f}")
                    print(f"      Final weights: {result.x}")
                    print(f"      Weights sum: {result.x.sum():.8f}")
                
                if result.success and result.fun < best_objective:
                    # Normalize weights to ensure they sum to 1
                    normalized_weights = result.x / np.sum(result.x)
                    best_result = normalized_weights
                    best_objective = result.fun
                    
                    if debug:
                        print(f"      New best! MSE={result.fun:.8f}")
                    
            except Exception as e:
                if debug:
                    print(f"      Error: {e}")
                continue
    
    if best_result is not None:
        if debug:
            # Verify the final result
            final_synthetic = np.dot(best_result, donor_pre)
            final_mse = np.sum((panama_pre - final_synthetic) ** 2)
            print(f"      Final verification: MSE={final_mse:.8f}")
            print(f"      Panama actual: {panama_pre}")
            print(f"      Synthetic: {final_synthetic}")
            print(f"      Difference: {panama_pre - final_synthetic}")
        
        return best_result, best_objective
    else:
        # Fallback: use equal weights if optimization fails
        print("    Warning: Optimization failed, using equal weights")
        equal_weights = np.ones(n_donors) / n_donors
        synthetic = np.dot(equal_weights, donor_pre)
        fallback_objective = np.sum((panama_pre - synthetic) ** 2)
        
        if debug:
            print(f"      Fallback MSE: {fallback_objective:.8f}")
        
        return equal_weights, fallback_objective

def evaluate_synthetic_control(panama_data, donor_data, donor_codes, pre_years, post_years,
                             standardize=False, log_transform=False):
    """Evaluate synthetic control for given configuration with preprocessing options"""
    
    all_years = pre_years + post_years
    
    # Prepare data with preprocessing options
    panama_outcomes, donor_outcomes, valid_donors = prepare_data_for_synthetic_control(
        panama_data, donor_data, donor_codes, all_years, standardize, log_transform
    )
    
    if panama_outcomes is None or len(valid_donors) < 2:
        if panama_outcomes is None:
            print(f"    Error: Data preparation failed")
        else:
            print(f"    Error: Only {len(valid_donors)} valid donors found")
        return None
    
    # Split into pre and post treatment
    n_pre = len(pre_years)
    panama_pre = panama_outcomes[:n_pre]
    donor_pre = donor_outcomes[:, :n_pre]
    
    # Check for valid data
    if np.any(np.isnan(panama_pre)) or np.any(np.isnan(donor_pre)):
        print("    Error: NaN values found in data")
        return None
    
    if np.std(panama_pre) == 0:
        if len(panama_pre) == 1:
            print("    Error: Single pre-treatment year = zero variance (need ≥2 years)")
        else:
            print("    Error: Panama pre-treatment outcomes have zero variance")
        return None
    
    # Get optimal weights with debugging for the first successful case
    weights, pre_treatment_loss = synthetic_control_weights(panama_pre, donor_pre, debug=True)
    
    if weights is None:
        print("    Error: Optimization failed")
        return None
    
    # Calculate synthetic control for all periods
    synthetic_outcomes = np.dot(weights, donor_outcomes)
    
    # Calculate treatment effects
    panama_post = panama_outcomes[n_pre:]
    synthetic_post = synthetic_outcomes[n_pre:]
    treatment_effects = panama_post - synthetic_post
    
    # Calculate pre-treatment RMSE
    pre_treatment_rmse = np.sqrt(np.mean((panama_pre - synthetic_outcomes[:n_pre]) ** 2))
    
    # For log-transformed data, convert treatment effects back to levels for interpretation
    if log_transform:
        # Treatment effect in logs ≈ percentage change
        treatment_effects_pct = (np.exp(treatment_effects) - 1) * 100
    else:
        treatment_effects_pct = None
    
    return {
        'weights': weights,
        'donor_codes': valid_donors,
        'panama_outcomes': panama_outcomes,
        'synthetic_outcomes': synthetic_outcomes,
        'treatment_effects': treatment_effects,
        'treatment_effects_pct': treatment_effects_pct,
        'pre_treatment_rmse': pre_treatment_rmse,
        'pre_treatment_loss': pre_treatment_loss,
        'years': all_years,
        'n_pre': n_pre,
        'standardize': standardize,
        'log_transform': log_transform
    }

def run_synthetic_control_analysis():
    """Main analysis function with preprocessing options"""
    
    best_results = {}
    
    # First, let's examine the data to understand what we're working with
    print(f"Total countries in dataset: {df['reporterCode'].nunique()}")
    print(f"Years available: {sorted(df['Year'].unique())}")
    print(f"Panama data points: {len(panama_data)}")
    print(f"Donor data points: {len(donor_data)}")
    
    # Preprocessing options to try
    preprocessing_options = [
        {'standardize': False, 'log_transform': False, 'name': 'raw'},
        {'standardize': True, 'log_transform': False, 'name': 'standardized'},
        {'standardize': False, 'log_transform': True, 'name': 'log'},
        {'standardize': True, 'log_transform': True, 'name': 'log_standardized'}
    ]
    
    # Try different pre-treatment periods - need at least 2 years for variance calculation
    for start_year in range(2015, 2017):  # 2015-2016 (2 years) and 2016 (1 year for special test)
        pre_treatment_years = list(range(start_year, TREATMENT_YEAR))
        post_treatment_years = list(range(TREATMENT_YEAR, 2020))
        
        # Need at least 1 year, but note that 1 year will have zero variance
        if len(pre_treatment_years) < 1:
            continue
        
        # Special message for single year case
        if len(pre_treatment_years) == 1:
            print(f"\nTesting pre-treatment period: {start_year}-{TREATMENT_YEAR-1} (Note: Single year = zero variance)")
        else:
            print(f"\nTesting pre-treatment period: {start_year}-{TREATMENT_YEAR-1}")
        
        print(f"\nTesting pre-treatment period: {start_year}-{TREATMENT_YEAR-1}")
        
        # Get donor pools
        donor_pools = get_donor_pools(donor_data, panama_data, pre_treatment_years)
        
        for pool_name, donor_codes in donor_pools.items():
            if len(donor_codes) < 2:
                print(f"  Skipping {pool_name} pool: only {len(donor_codes)} donors")
                continue
            
            # Try different preprocessing options
            for preprocess in preprocessing_options:
                config_name = f"{pool_name}_{preprocess['name']}"
                print(f"  Testing {config_name} with {len(donor_codes)} donors")
                
                result = evaluate_synthetic_control(
                    panama_data, donor_data, donor_codes, 
                    pre_treatment_years, post_treatment_years,
                    standardize=preprocess['standardize'],
                    log_transform=preprocess['log_transform']
                )
                
                if result is None:
                    continue
                
                # Store result with configuration info
                config_key = f"{start_year}_{config_name}"
                result.update({
                    'start_year': start_year,
                    'pool_name': pool_name,
                    'preprocessing': preprocess['name'],
                    'config_key': config_key
                })
                
                best_results[config_key] = result
                
                print(f"    ✓ Success! Pre-treatment RMSE: {result['pre_treatment_rmse']:.4f}")
    
    return best_results

def plot_results(result, title_suffix=""):
    """Plot synthetic control results with proper scaling"""
    
    years = result['years']
    panama = result['panama_outcomes']
    synthetic = result['synthetic_outcomes']
    n_pre = result['n_pre']
    
    # For log-transformed data, convert back to levels for plotting
    if result.get('log_transform', False):
        panama_plot = np.exp(panama)
        synthetic_plot = np.exp(synthetic)
        ylabel = 'Total Bilateral Trade with China (Levels)'
    else:
        panama_plot = panama
        synthetic_plot = synthetic
        ylabel = 'Total Bilateral Trade with China'
        if result.get('standardize', False):
            ylabel += ' (Standardized)'
    
    plt.figure(figsize=(12, 8))
    
    # Main plot
    plt.subplot(2, 1, 1)
    plt.plot(years, panama_plot, 'b-', linewidth=2, label='Panama (Actual)')
    plt.plot(years, synthetic_plot, 'r--', linewidth=2, label='Synthetic Panama')
    plt.axvline(x=TREATMENT_YEAR, color='gray', linestyle='-', alpha=0.7, label='Treatment')
    plt.xlabel('Year')
    plt.ylabel(ylabel)
    plt.title(f'Synthetic Control Analysis{title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Treatment effects plot
    plt.subplot(2, 1, 2)
    post_years = years[n_pre:]
    treatment_effects = result['treatment_effects']
    
    # For log data, show percentage effects
    if result.get('log_transform', False):
        treatment_effects_pct = (np.exp(treatment_effects) - 1) * 100
        plt.bar(post_years, treatment_effects_pct, alpha=0.7, color='green')
        plt.ylabel('Treatment Effect (%)')
        plt.title('Estimated Treatment Effects (Percentage Change)')
    else:
        plt.bar(post_years, treatment_effects, alpha=0.7, color='green')
        plt.ylabel('Treatment Effect')
        plt.title('Estimated Treatment Effects')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('Year')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt

def print_results_summary(result):
    """Print detailed results"""
    
    print(f"\nSynthetic Control Results:")
    print(f"Pre-treatment period: {result['start_year']}-{TREATMENT_YEAR-1}")
    print(f"Donor pool: {result['pool_name']}")
    print(f"Preprocessing: {result.get('preprocessing', 'raw')}")
    print(f"Standardized: {result.get('standardize', False)}")
    print(f"Log transformed: {result.get('log_transform', False)}")
    print(f"Number of donors: {len(result['donor_codes'])}")
    print(f"Pre-treatment RMSE: {result['pre_treatment_rmse']:.4f}")
    
    print(f"\nDonor weights (showing weights > 0.01):")
    for i, (code, weight) in enumerate(zip(result['donor_codes'], result['weights'])):
        if weight > 0.01:
            print(f"  Country {code}: {weight:.4f}")
    
    print(f"\nTreatment effects:")
    post_years = result['years'][result['n_pre']:]
    for year, effect in zip(post_years, result['treatment_effects']):
        if result.get('log_transform', False):
            pct_effect = (np.exp(effect) - 1) * 100
            print(f"  {year}: {effect:.4f} (log units) ≈ {pct_effect:.2f}% change")
        else:
            print(f"  {year}: {effect:,.2f}")
    
    # Average treatment effect
    avg_effect = np.mean(result['treatment_effects'])
    if result.get('log_transform', False):
        avg_pct_effect = (np.exp(avg_effect) - 1) * 100
        print(f"\nAverage treatment effect: {avg_effect:.4f} (log units) ≈ {avg_pct_effect:.2f}% change")
    else:
        print(f"\nAverage treatment effect: {avg_effect:,.2f}")

# Run the analysis
if __name__ == "__main__":
    print("Running Synthetic Control Analysis...")
    print("=" * 50)
    
    results = run_synthetic_control_analysis()
    
    if not results:
        print("No valid results found!")
    else:
        # Find best result based on pre-treatment fit
        best_result = min(results.values(), key=lambda x: x['pre_treatment_rmse'])
        
        print(f"\nBest configuration found:")
        print("=" * 50)
        print_results_summary(best_result)
        
        # Plot best result
        plt = plot_results(best_result, f" - Best Fit ({best_result['config_key']})")
        plt.savefig('best_synthetic_control_corrected.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Also show results for 'all' donor pool if available
        all_results = [r for r in results.values() if r['pool_name'] == 'all']
        if all_results:
            best_all = min(all_results, key=lambda x: x['pre_treatment_rmse'])
            print(f"\nBest 'all donors' configuration:")
            print("=" * 50)
            print_results_summary(best_all)
            
            plt = plot_results(best_all, f" - All Donors ({best_all['config_key']})")
            plt.savefig('synthetic_control_all_donors_corrected.png', dpi=300, bbox_inches='tight')
            plt.show()