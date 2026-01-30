import pandas as pd

# Load data again
file_path = 'data/e-shop clothing 2008.csv'
df = pd.read_csv(file_path, delimiter=';')

# 1. Print ALL column names so we can choose our target
print("--- COLUMN NAMES ---")
print(df.columns.tolist())

# 2. Check for missing data (Empty cells break algorithms)
print("\n--- MISSING DATA CHECK ---")
print(df.isnull().sum())

# 3. Analyze the 'Session' data
# We need to know how many unique user journeys we have.
total_sessions = df['session ID'].nunique()
print(f"\nTotal Unique Sessions: {total_sessions}")

# 4. Check the 'Page' categories
# Let's see what the values look like in the 'page 1 (main category)' column
# This is likely what we want to predict.
if 'page 1 (main category)' in df.columns:
    print("\n--- Unique Categories (Page 1) ---")
    print(df['page 1 (main category)'].unique())
    print(f"Count of categories: {len(df['page 1 (main category)'].unique())}")