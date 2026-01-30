import pandas as pd
import numpy as np
import faiss
import pickle
from sklearn.decomposition import TruncatedSVD

# --- CONFIGURATION ---
NUM_PRODUCTS = 10000   # 10k items
NUM_USERS = 5000       # 5k users
NUM_INTERACTIONS = 100000 # 100k clicks
LATENT_DIM = 50        # Vector size

print(f"🚀 INITIALIZING PRODUCTION PIPELINE")

# 1. GENERATE SYNTHETIC DATA
print("--- [1/4] Generating Synthetic Data Lake ---")
adjectives = ['Sleek', 'Durable', 'Wireless', 'Ergonomic', 'Vintage', 'Smart', 'Luxury']
nouns = ['Watch', 'Phone', 'Laptop', 'Shoes', 'Headphones', 'Camera', 'Bag']
products = [f"{np.random.choice(adjectives)} {np.random.choice(nouns)} {i}" for i in range(NUM_PRODUCTS)]
df_products = pd.DataFrame({'product_id': range(NUM_PRODUCTS), 'title': products})

# Generate interactions
user_ids = np.random.randint(0, NUM_USERS, NUM_INTERACTIONS)
product_ids = np.random.choice(range(NUM_PRODUCTS), NUM_INTERACTIONS)
df_interactions = pd.DataFrame({'user_id': user_ids, 'product_id': product_ids, 'rating': np.random.randint(1, 6, NUM_INTERACTIONS)})

print(f"✔ Generated {len(df_interactions)} rows of logs.")

# 2. MATRIX FACTORIZATION (SVD)
print("--- [2/4] Training SVD Model ---")

# CRITICAL FIX: Ensure ALL product IDs exist in the pivot table, even if unclicked
pivot_table = df_interactions.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)

# Reindex columns to ensure 0 to 9999 are ALL present
# This fixes the "key < ntotal" crash
pivot_table = pivot_table.reindex(columns=range(NUM_PRODUCTS), fill_value=0)

svd = TruncatedSVD(n_components=LATENT_DIM, random_state=42)
user_features = svd.fit_transform(pivot_table)
product_features = svd.components_.T 

print(f"✔ SVD Trained. Matrix Shape: {product_features.shape}")

# 3. BUILD FAISS INDEX
print("--- [3/4] Building FAISS Vector Index ---")
product_vectors = product_features.astype('float32')
index = faiss.IndexFlatL2(LATENT_DIM) 
index.add(product_vectors)
print(f"✔ FAISS Index Built. Total Vectors: {index.ntotal}")

# 4. SAVE
print("--- [4/4] Serializing Assets ---")
with open('data/faiss_store.pkl', 'wb') as f:
    pickle.dump({
        'index': index, 
        'products': df_products, 
        'pivot': pivot_table
    }, f)

print("\n✨ PIPELINE COMPLETE.")