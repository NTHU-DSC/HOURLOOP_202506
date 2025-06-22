import pandas as pd

# Load and Merge dataframes
df1 = pd.read_csv("Data\\Raw\\Data_v8.csv")
df2 = pd.read_csv("Data\\Raw\\Data_v8_after.csv")
merged_df = pd.concat([df1, df2], ignore_index=True)

# Process each ship_method
for ship_method, group in merged_df.groupby('ship_method'):
    sorted_group = group.sort_values(by='date')
    
    # Split into latest 10% for testing and remaining 90%
    n_test_samples = round(len(sorted_group) * 0.10)
    test_sample = sorted_group.tail(n_test_samples)
    test_sample.to_csv(f"Data\\test\\{ship_method}_test.csv", index=False)
    remaining = sorted_group.iloc[:-n_test_samples]
    remaining.to_csv(f"Data\\train\\{ship_method}_train.csv", index=False)