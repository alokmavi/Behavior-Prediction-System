import pandas as pd
import numpy as np
import faiss
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import time

# --- CONFIGURATION ---
NUM_PRODUCTS = 10000   # FAANG Scale: 10k items
NUM_USERS = 5000       # 5k users
NUM_INTERACTIONS = 100000 # 100k clicks
LATENT_DIM = 50        # The "Features" vector size

print(f"🚀 INITIALIZING PRODUCTION PIPELINE")
print(f"Target: {NUM_PRODUCTS} Products | {NUM_INTERACTIONS} Interactions")

# 1. GENERATE SYNTHETIC "BIG DATA"
# Real projects don't type data manually. We generate it.
print("--- [1/4] Generating Synthetic Data Lake ---")

# Fake Product Names (Adjective + Noun)
adjectives = ['Sleek', 'Durable', 'Wireless', 'Ergonomic', 'Vintage', 'Smart', 'Luxury']
nouns = ['Watch', 'Phone', 'Laptop', 'Shoes', 'Headphones', 'Camera', 'Bag']
products = [f"{np.random.choice(adjectives)} {np.random.choice(nouns)} {i}" for i in range(NUM_PRODUCTS)]
df_products = pd.DataFrame({'product_id': range(NUM_PRODUCTS), 'title': products})

# Fake Interactions (User -> Product)
# We create a "bias" so some items are popular (Real world distribution)
user_ids = np.random.randint(0, NUM_USERS, NUM_INTERACTIONS)
product_ids = np.random.choice(range(NUM_PRODUCTS), NUM_INTERACTIONS, p=np.random.dirichlet(np.ones(NUM_PRODUCTS)/10))
df_interactions = pd.DataFrame({'user_id': user_ids, 'product_id': product_ids, 'rating': np.random.randint(1, 6, NUM_INTERACTIONS)})

print(f"✔ Generated {len(df_interactions)} rows of interaction logs.")

# 2. MATRIX FACTORIZATION (SVD) - The "Netflix Prize" Algorithm
# We convert the User-Product interaction matrix into "Embeddings"
print("--- [2/4] Training SVD Model (Matrix Factorization) ---")

# Create Pivot Table (Users x Products) -> Sparse Matrix
# Using a small chunk for demonstration speed, but logic holds for sparse matrices
pivot_table = df_interactions.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)

# SVD compresses this huge matrix into small "feature vectors"
svd = TruncatedSVD(n_components=LATENT_DIM, random_state=42)
user_features = svd.fit_transform(pivot_table)
product_features = svd.components_.T # Transpose to get product vectors

print(f"✔ Model Trained. User Vector Shape: {user_features.shape}")

# 3. BUILD VECTOR SEARCH INDEX (FAISS)
# This is the Secret Sauce. Instead of looping, we index vectors.
print("--- [3/4] Building FAISS Vector Index ---")

# FAISS requires float32 format
product_vectors = product_features.astype('float32')

# Create the Index (L2 Distance / Euclidean)
index = faiss.IndexFlatL2(LATENT_DIM) 
index.add(product_vectors)

print(f"✔ FAISS Index Built. Total Vectors Indexed: {index.ntotal}")

# 4. SAVE ARTIFACTS
# In production, these go to S3 or a Model Registry
print("--- [4/4] Serializing Assets ---")
with open('data/faiss_store.pkl', 'wb') as f:
    pickle.dump({
        'index': index, 
        'products': df_products, 
        'pivot': pivot_table # Saving pivot mainly to map user_id back to row index
    }, f)

print("\n✨ PIPELINE COMPLETE. System ready for High-Throughput Inference.")