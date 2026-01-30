# 🧠 Hybrid User Behavior Prediction & Recommendation System

## 📌 Project Overview
A production-ready **Web Usage Mining Engine** that predicts user navigation patterns on e-commerce platforms. Unlike traditional systems that rely solely on clickstream data, this project implements a **Hybrid Ensemble Architecture**:
1.  **Sequential Pattern Mining:** Uses **First-Order Markov Chains** to predict the next likely action based on crowd behavior.
2.  **Content-Based Filtering:** Uses **TF-IDF Vectorization** and **Cosine Similarity** to recommend semantically related products when behavioral data is sparse (solving the *Cold Start Problem*).

## 🚀 Key Features
* **Dual-Engine Architecture:** Seamlessly switches between Probability-based and Content-based recommendations.
* **Real-Time Visualization:** Dynamic **Directed Graph (DiGraph)** rendering of user paths using NetworkX.
* **Interactive Dashboard:** Streamlit-based UI with live session simulation, confidence metrics, and probability distributions.
* **Data Processing:** Pipelines for sessionizing raw server logs (24k+ transactions) and text vectorization.

## 🛠️ Tech Stack
* **Core Logic:** Python 3.9+, Pandas, NumPy
* **Machine Learning:** Scikit-Learn (TF-IDF, Cosine Similarity)
* **Graph Theory:** NetworkX
* **UI/UX:** Streamlit

## ⚙️ How to Run
1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Train the Models:**
    ```bash
    python day3_markov_model.py  # Trains the Behavior Engine
    python day6_hybrid_brain.py  # Trains the Content Engine
    ```
3.  **Launch Dashboard:**
    ```bash
    streamlit run day5_app.py
    ```

---
*Developed by Alok Kumar Mavi | B.Tech Computer Science*