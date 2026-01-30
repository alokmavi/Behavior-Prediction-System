from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

# Initialize API
app = FastAPI(title="Recommendation Engine Microservice", version="2.0")

# Load the Brain (Global State)
print("Loading FAISS Index...")
with open('data/faiss_store.pkl', 'rb') as f:
    data = pickle.load(f)
    index = data['index']
    df_products = data['products']
    pivot_table = data['pivot'] # To look up user history

class RecommendationRequest(BaseModel):
    user_id: int
    top_k: int = 5

@app.get("/")
def health_check():
    return {"status": "online", "engine": "FAISS", "items_indexed": index.ntotal}

@app.post("/recommend")
def get_recommendations(request: RecommendationRequest):
    # 1. Look up User
    # In a real DB, we'd query PostgreSQL. Here we look at the pivot table.
    if request.user_id not in pivot_table.index:
        raise HTTPException(status_code=404, detail="User not found in training set")
    
    # Get the user's "history vector" (their raw ratings)
    user_vector = pivot_table.loc[request.user_id].values.reshape(1, -1)
    
    # 2. In REAL production, we would multiply this by the SVD matrix to get the 
    # latent vector. For simplicity in this demo, we simulated the 'user_features'
    # in the training step. We will skip the math re-calculation and simulate
    # a "User Query Vector" by averaging the vectors of items they liked.
    
    # Simulation: We search for items similar to the user's *last liked item*
    # (This is Item-to-Item collaborative filtering via Vector Search)
    
    # Get user's last rated product
    user_history = pivot_table.loc[request.user_id]
    liked_products = user_history[user_history > 3].index.tolist()
    
    if not liked_products:
        return {"message": "User has no likes, returning popular items"}

    # Take the last item they liked
    last_item_id = liked_products[-1]
    
    # Get that item's vector from FAISS (reconstruct it)
    # Note: FAISS usually stores vectors. We use the product_id as index.
    query_vector = index.reconstruct(last_item_id).reshape(1, -1)
    
    # 3. FAISS SEARCH (The Magic)
    # Search the index for the 'top_k' nearest neighbors
    distances, indices = index.search(query_vector, request.top_k + 1)
    
    # 4. Format Result
    recommendations = []
    for i in range(1, len(indices[0])): # Skip 0 because it's the item itself
        idx = indices[0][i]
        dist = float(distances[0][i])
        product_name = df_products.iloc[idx]['title']
        recommendations.append({
            "product_id": int(idx),
            "title": product_name,
            "similarity_score": round(1 - dist, 4) # Rough approximation
        })
        
    return {
        "user_id": request.user_id,
        "based_on_item": df_products.iloc[last_item_id]['title'],
        "recommendations": recommendations
    }