import pandas as pd

# 1. Load the data
df = pd.read_csv('data/e-shop clothing 2008.csv', delimiter=';')

# 2. Sort the data
# We must ensure the clicks are in the correct order of time!
# We sort by 'session ID' and then 'order' (the sequence of clicks).
df = df.sort_values(by=['session ID', 'order'])

# 3. Map Numbers to Names (Making it readable)
# According to the dataset documentation:
category_map = {
    1: 'Trousers',
    2: 'Skirts',
    3: 'Blouses',
    4: 'Sale'
}
df['page_name'] = df['page 1 (main category)'].map(category_map)

print("Mapping applied. Example data:")
print(df[['session ID', 'order', 'page_name']].head())

# 4. Group by Session to create "Journeys"
# This combines all pages visited by one user into a single list.
print("\nCreating user sequences... (This takes a moment)")
sequences = df.groupby('session ID')['page_name'].apply(list)

# 5. Filter out short sessions
# If a user only clicked 1 page, we can't "predict" the next step.
# We keep only sessions with length > 1.
filtered_sequences = sequences[sequences.apply(len) > 1]

print(f"\nOriginal Sessions: {len(sequences)}")
print(f"Useful Sessions (Length > 1): {len(filtered_sequences)}")

# 6. Save for Day 3
# We save this processed list to a file so we don't have to clean again.
filtered_sequences.to_csv('data/processed_sequences.csv')
print("\n[SUCCESS] 'processed_sequences.csv' saved to data folder!")