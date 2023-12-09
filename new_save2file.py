import pandas as pd
import re

# Path of the latest uploaded file
latest_file_path = './results.txt'

# Reading the contents of the latest uploaded file
with open(latest_file_path, 'r') as file:
    latest_content = file.read()

# Regular expressions used for parsing
algo_bconfig_pattern_updated = re.compile(r'Running with --algo=(\d) --bconfig=(\d+)')
performance_pattern_refined = re.compile(r'Average elapsed time: \(([\d.]+)\) s, performance: \(\s*([\d.]+)\) GIPS')

# Parsing logic
latest_refined_results = []

for section in latest_content.split("---------------------------------------"):
    algo_bconfig_match = algo_bconfig_pattern_updated.search(section)
    performance_match = performance_pattern_refined.search(section)
    
    if algo_bconfig_match and performance_match:
        algo = int(algo_bconfig_match.group(1))
        bconfig = int(algo_bconfig_match.group(2))
        performance = float(performance_match.group(2))
        
        latest_refined_results.append({
            "algo": algo,
            "bconfig": bconfig,
            "performance": performance
        })

# Converting the parsed results into a DataFrame
df_latest_refined_results = pd.DataFrame(latest_refined_results)

# Pivoting the DataFrame
pivot_df = df_latest_refined_results.pivot(index='algo', columns='bconfig', values='performance')

# Renaming the columns to a more descriptive format
pivot_df.columns = ['bconfig = ' + str(col) for col in pivot_df.columns]

# Saving the pivoted data to a CSV file
latest_csv_file_path = './pivoted_performance_results_latest.csv'
pivot_df.to_csv(latest_csv_file_path, index=True)

# Output path of the CSV file
latest_csv_file_path
