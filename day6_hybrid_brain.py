import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import json

# 1. Load Data & Create "Content"
# We pretend our pages have descriptions (Standard in real projects)
data = {
    'page_name': ['Trousers', 'Skirts', 'Blouses', 'Sale', 'Jackets', 'Accessories'],
    'description': [
        'legs cotton denim casual clothing',
        'legs summer casual flowy clothing',
        'upper formal office wear clothing',
        'discount cheap offer clothing',
        'upper winter warm heavy clothing',
        'jewelry belts hats extra add-on'
    ]
}
df_content = pd.DataFrame(data)

# 2. Build the Content Engine (TF-IDF)
print("Training Content-Based Model...")
tfidf = TfidfVectorizer(stop_words='english')
# Convert descriptions to a matrix of numbers
tfidf_matrix = tfidf.fit_transform(df_content['description'])

# Calculate Similarity (How close is 'Trousers' to 'Skirts'?)
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Map page names to indices
indices = pd.Series(df_content.index, index=df_content['page_name']).drop_duplicates()

def get_content_recommendations(page):
    if page not in indices:
        return []
    idx = indices[page]
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort by similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get top 3 (excluding itself)
    sim_scores = sim_scores[1:4]
    
    # Return page names
    page_indices = [i[0] for i in sim_scores]
    return df_content['page_name'].iloc[page_indices].tolist()

# 3. Test it
print("--- Content Engine Test ---")
print(f"Content similar to 'Trousers': {get_content_recommendations('Trousers')}")

# 4. Save this 'Hybrid Knowledge'
hybrid_map = {}
for page in df_content['page_name']:
    hybrid_map[page] = get_content_recommendations(page)

with open('data/hybrid_model.json', 'w') as f:
    json.dump(hybrid_map, f)

print("[SUCCESS] Hybrid Model Saved.")