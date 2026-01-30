import pandas as pd
import numpy as np
import ast  # To convert string representation of list back to list
import json # To save our 'Brain' (Model)

# 1. Load the processed data
print("Loading processed data...")
df = pd.read_csv('data/processed_sequences.csv')

# 2. Convert text strings back to lists
# (The CSV saved them as text "['A', 'B']", we need real lists ['A', 'B'])
print("Converting text to lists... (This takes a moment)")
# We assume the column name from your output is 'page_name' based on previous step
# If the CSV header is 'page_name', this works.
df['sequence'] = df['page_name'].apply(ast.literal_eval)

# 3. Build the Markov Chain (The "Brain")
# Dictionary structure: { 'Current_Page': { 'Next_Page_A': count, 'Next_Page_B': count } }
markov_chain = {}

print("Training the model on 18,000+ sessions...")

for sequence in df['sequence']:
    # Get unique pages in this sequence to remove duplicates if needed,
    # but for Markov, we want every transition step.
    # Loop through the sequence: A -> B -> C
    for i in range(len(sequence) - 1):
        current_page = sequence[i]
        next_page = sequence[i+1]
        
        # Create dictionary entry if it doesn't exist
        if current_page not in markov_chain:
            markov_chain[current_page] = {}
        
        if next_page not in markov_chain[current_page]:
            markov_chain[current_page][next_page] = 0
            
        # Add a "vote" for this path
        markov_chain[current_page][next_page] += 1

# 4. Convert Counts to Probabilities
# Instead of "50 clicks", we want "0.5 probability"
model_json = {}

for page, transitions in markov_chain.items():
    total_visits = sum(transitions.values())
    model_json[page] = {}
    
    for next_p, count in transitions.items():
        # Calculate probability (Count / Total)
        probability = count / total_visits
        model_json[page][next_p] = round(probability, 2)

# 5. Save the Model
# We save this dictionary as a JSON file. The website will read this file.
with open('data/markov_model.json', 'w') as f:
    json.dump(model_json, f)

print("\n[SUCCESS] Model Trained & Saved to 'data/markov_model.json'")
print("--- Example Knowledge ---")
# Let's see what the AI learned about 'Trousers'
if 'Trousers' in model_json:
    print(f"If a user is on 'Trousers', they will go to:")
    print(model_json['Trousers'])