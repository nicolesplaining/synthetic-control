import pandas as pd
import glob
import os

def process_trade_data():
    data_files = glob.glob('data/C_A_*_classic') + glob.glob('data/new/C_A_*_classic')
    dfs = []
    for file in data_files:
        try:
            df = pd.read_csv(file, sep='\t', low_memory=False)
            df = df[df['partnerCode'] == 156] # China partnerCode 156
            year = int(file.split('_')[-2])
            df['Year'] = year 
            if 2000 <= year <= 2019:
                numeric_cols = ['CIFValue', 'FOBValue', 'qty', 'netWgt', 'grossWgt']
                for col in numeric_cols:
                    if col in df.columns:  
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    else:
                        df[col] = 0 
                df['Total_Bilateral_Trade'] = df['CIFValue'] + df['FOBValue']
                grouped = df.groupby(['Year', 'reporterCode']).agg({
                    'Total_Bilateral_Trade': 'sum',
                    'CIFValue': 'sum',
                    'FOBValue': 'sum',
                    'qty': 'sum',
                    'netWgt': 'sum',
                    'grossWgt': 'sum'
                }).reset_index()
                
                dfs.append(grouped)
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    if not dfs:
        print("No data found for years 2000-2019 with China as partner.")
        return

    final_df = pd.concat(dfs, ignore_index=True)

    final_df = final_df.groupby(['Year', 'reporterCode'], as_index=False).agg({
        'Total_Bilateral_Trade': 'sum',
        'CIFValue': 'sum',
        'FOBValue': 'sum',
        'qty': 'sum',
        'netWgt': 'sum',
        'grossWgt': 'sum'
    })
    country_year_counts = final_df.groupby('reporterCode')['Year'].nunique()
    insufficient_countries = country_year_counts[country_year_counts < 10].index.tolist()
    if insufficient_countries:
        print("\nCountries with insufficient data (fewer than 10 years, excluded):")
        for code in insufficient_countries:
            print(f"Country {code}")
        final_df = final_df[~final_df['reporterCode'].isin(insufficient_countries)]
    final_df = final_df.sort_values(['Year', 'reporterCode'])
    final_df.to_csv('all_countries_trade_with_china_2000_2019.csv', index=False)
    print("\nData saved to 'all_countries_trade_with_china_2000_2019.csv'")
    print("\nSummary of variables:")
    print(final_df.describe())
    print(f"\nNumber of unique countries: {final_df['reporterCode'].nunique()}")
    print(f"Number of years: {final_df['Year'].nunique()}")
    print(f"Years covered: {final_df['Year'].min()} to {final_df['Year'].max()}")

if __name__ == "__main__":
    process_trade_data() 