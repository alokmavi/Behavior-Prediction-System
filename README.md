# Distributed Recommendation Engine with Vector Search

## 1. Project Overview
This repository hosts a production-grade **Hybrid Recommendation System** designed to handle cold-start scenarios and high-throughput inference. Unlike traditional collaborative filtering implementations that rely on in-memory matrix operations (O(N) complexity), this system utilizes **Facebook AI Similarity Search (FAISS)** to achieve **O(log N)** retrieval latency for approximate nearest neighbor searches.

The architecture mimics a real-world enterprise deployment by decoupling the inference engine (FastAPI Microservice) from the user interface (Streamlit), enabling independent scaling of frontend and backend components.

## 2. System Architecture
The system operates on a Microservices architecture:

1.  **ETL & Training Pipeline:** Generates synthetic "Big Data" (100k+ logs), applies **Singular Value Decomposition (SVD)** for dimensionality reduction, and indexes latent vectors.
2.  **Inference Service (Backend):** A **FastAPI** instance that serves recommendations via REST endpoints, utilizing the FAISS index for sub-millisecond similarity lookups.
3.  **Client Application (Frontend):** A **Streamlit** dashboard that consumes the API, visualizing user-item interactions and vector similarity scores.

## 3. Key Technical Features
* **Vector Space Retrieval:** Implements **FAISS (IndexFlatL2)** for high-speed similarity search across 10,000+ product vectors.
* **Latent Factor Model:** Uses **Matrix Factorization (SVD)** to compress sparse user-interaction matrices into dense latent features ($k=50$).
* **Microservices Design:** Decoupled REST API backend preventing frontend-blocking operations during heavy computation.
* **Synthetic Data Engineering:** Includes a robust generation pipeline to simulate power-law distributions in user behavior and handle data sparsity.

## 4. Technology Stack
* **Language:** Python 3.9+
* **API Framework:** FastAPI, Uvicorn
* **Machine Learning:** Scikit-Learn (SVD/TruncatedSVD), FAISS (CPU)
* **Data Processing:** Pandas, NumPy
* **Frontend:** Streamlit

## 5. Installation & Setup
**Note:** This system simulates an enterprise environment. You must generate the data artifacts locally before the API can start.

### Step 1: Clone and Install Dependencies
git clone [https://github.com/alokmavi/Behavior-Prediction-System.git](https://github.com/alokmavi/Behavior-Prediction-System.git)
cd Behavior-Prediction-System
pip install -r requirements.txt
### Step 2: Run the ETL Pipeline (Critical)

This script generates 100,000 synthetic interaction logs, trains the SVD model, and builds the FAISS vector index. The output is a serialized artifact (Data/faiss_store.pkl).

python day7_production_engine.py
Expected Output: "✨ PIPELINE COMPLETE. System ready for High-Throughput Inference."

### Step 3: Launch the Inference Backend

Start the FastAPI microservice. Keep this terminal open.

uvicorn day7_api:app --reload
The API will become available at http://127.0.0.1:8000.

### Step 4: Launch the User Dashboard

Open a new terminal window and start the frontend application.

streamlit run day5_app.py
## 6. API Documentation
Once the backend is running, full Swagger/OpenAPI documentation is available at: http://127.0.0.1:8000/docs

Key Endpoints:

GET /: Health check and index status.

POST /recommend: Accepts a user_id and returns top-k nearest neighbors based on the user's latent vector history.

## 7. Project Structure
Bash
├── Data/                   # Stores generated artifacts (ignored by Git)
├── day7_production_engine.py # ETL Pipeline: Data Gen -> SVD -> FAISS
├── day7_api.py             # FastAPI Microservice
├── day5_app.py             # Frontend Dashboard
├── requirements.txt        # Dependency list
└── README.md               # Documentation
Author: Alok Kumar Mavi B.Tech Computer Science | Specialization in AI & Data Systems