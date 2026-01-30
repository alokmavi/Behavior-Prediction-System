import json

# 1. Load the "Brain"
print("Loading the Model...")
with open('data/markov_model.json', 'r') as f:
    model = json.load(f)

print("[SUCCESS] Model Loaded.\n")

# 2. Define the Prediction Function
def get_recommendations(current_page):
    """
    Input: The page the user is currently on (e.g., 'Trousers')
    Output: A list of the top 3 most likely next pages.
    """
    
    # Check if the page exists in our model
    if current_page not in model:
        return [] # Return empty if we don't know this page
    
    # Get all possible next steps and their probabilities
    next_pages = model[current_page]
    
    # Sort them by probability (Highest first)
    # We use a lambda function to sort by the value (probability)
    sorted_pages = sorted(next_pages.items(), key=lambda item: item[1], reverse=True)
    
    # Return the top 3
    return sorted_pages[:3]

# 3. Test the System (The "Simulation")
test_pages = ['Trousers', 'Skirts', 'Sale']

print("--- SYSTEM TEST ---")

for page in test_pages:
    print(f"\nUser is currently looking at: [ {page} ]")
    
    predictions = get_recommendations(page)
    
    if predictions:
        print("System predicts they will go to:")
        for i, (next_page, prob) in enumerate(predictions, 1):
            # Convert 0.8 to 80% for display
            print(f"  {i}. {next_page} ({int(prob * 100)}% chance)")
    else:
        print("  No data available for this page.")

print("\n-------------------")