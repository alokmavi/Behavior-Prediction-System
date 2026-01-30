import streamlit as st
import requests
import pandas as pd

# API CONFIG
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Enterprise RecSys", layout="wide")
st.title("⚡ Enterprise-Grade Recommendation System")
st.markdown("Architecture: **Streamlit (Frontend)** → **FastAPI (Microservice)** → **FAISS (Vector Engine)**")

# Check API Health
try:
    health = requests.get(f"{API_URL}/").json()
    st.success(f"Backend Connected: {health['status']} | Indexed Items: {health['items_indexed']}")
except:
    st.error("❌ Backend Offline. Please run 'uvicorn day7_api:app --reload'")
    st.stop()

# UI
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("User Simulation")
    # We pretend to login as a user ID (from our synthetic dataset)
    user_id = st.number_input("Enter User ID (0-4999)", min_value=0, max_value=4999, value=42)
    
    if st.button("Get Personal Recommendations"):
        with st.spinner("Querying Vector Database..."):
            try:
                payload = {"user_id": user_id, "top_k": 5}
                response = requests.post(f"{API_URL}/recommend", json=payload)
                data = response.json()
                
                if "recommendations" in data:
                    st.session_state['results'] = data
                else:
                    st.warning(data['detail'])
            except Exception as e:
                st.error(f"API Error: {e}")

with col2:
    if 'results' in st.session_state:
        res = st.session_state['results']
        st.info(f"Because user {res['user_id']} liked **'{res['based_on_item']}'**...")
        
        # Display as cards
        for item in res['recommendations']:
            st.markdown(f"""
            <div style="padding:15px; border-radius:5px; background-color:#262730; margin-bottom:10px; border-left: 5px solid #4CAF50;">
                <h4 style="margin:0">{item['title']}</h4>
                <small>Vector Similarity Distance: {item['similarity_score']}</small>
            </div>
            """, unsafe_allow_html=True)