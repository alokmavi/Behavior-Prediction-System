import pandas as pd

# Define file path
file_path = 'data/e-shop clothing 2008.csv'

print("Loading dataset... this might take a second.")

# Load the data
# Note: delimiter=';' is crucial because this CSV uses semicolons!
df = pd.read_csv(file_path, delimiter=';')

# 1. Verification: Print the shape (Rows, Columns)
print(f"SUCCESS! Data Loaded.")
print(f"Total User Actions: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}")

# 2. Preview the first 5 rows
print("\n--- First 5 Rows of Data ---")
print(df.head())

# 3. Check for 'Session ID' (Crucial for our project)
if 'session ID' in df.columns:
    print("\n[✓] 'session ID' column found. We are ready for Day 2.")
else:
    print("\n[X] ERROR: 'session ID' not found. Check column names:")
    print(df.columns)