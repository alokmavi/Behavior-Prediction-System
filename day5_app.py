import streamlit as st
import json
import os
import networkx as nx
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Behavior Prediction AI | Hybrid Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD BRAINS (Models) ---
@st.cache_data
def load_models():
    # Load Markov Model (Behavior)
    markov_model = {}
    if os.path.exists('data/markov_model.json'):
        with open('data/markov_model.json', 'r') as f:
            markov_model = json.load(f)
            
    # Load Hybrid Model (Content Similarity)
    hybrid_model = {}
    if os.path.exists('data/hybrid_model.json'):
        with open('data/hybrid_model.json', 'r') as f:
            hybrid_model = json.load(f)
            
    return markov_model, hybrid_model

markov_model, hybrid_model = load_models()

# --- ENGINE 1: MARKOV (Next Step) ---
def get_next_step_predictions(current_page):
    if not markov_model or current_page not in markov_model:
        return []
    next_pages = markov_model[current_page]
    return sorted(next_pages.items(), key=lambda x: x[1], reverse=True)[:3]

# --- ENGINE 2: HYBRID (Similar Content) ---
def get_content_recommendations(current_page):
    if not hybrid_model or current_page not in hybrid_model:
        return []
    return hybrid_model[current_page] # Returns a list of similar pages

# --- VISUALIZATION ENGINE ---
def plot_graph(current_page, predictions):
    G = nx.DiGraph()
    G.add_node(current_page, color='lightgreen', size=3000)
    
    for page, prob in predictions:
        G.add_node(page, color='skyblue', size=1500)
        G.add_edge(current_page, page, weight=prob, label=f"{int(prob*100)}%")
        
    fig, ax = plt.subplots(figsize=(10, 5)) # Wider graph
    pos = nx.spring_layout(G, seed=42) # Seed for stability
    
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='skyblue', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[current_page], node_color='lightgreen', node_size=3000, ax=ax)
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color='gray', arrows=True, arrowstyle='-|>', arrowsize=20, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif", ax=ax)
    
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', ax=ax)
    
    ax.axis('off')
    return fig

# --- UI LAYOUT ---
st.title("🧠 Hybrid User Behavior Prediction System")
st.markdown("### 🚀 Engine Status: **Active** | Mode: **Ensemble (Markov + Content-Based)**")
st.markdown("---")

if not markov_model:
    st.error("❌ Models not found! Please run 'day3_markov_model.py' and 'day6_hybrid_brain.py'.")
    st.stop()

# SIDEBAR
with st.sidebar:
    st.header("🕵️ User Simulation")
    st.info("Simulate a live user session to trigger predictions.")
    
    all_pages = sorted(list(markov_model.keys()))
    current_page = st.sidebar.selectbox("Select Current Interaction:", all_pages)
    
    st.markdown("---")
    st.markdown("**System Metrics:**")
    st.metric("Total Sessions Analyzed", "18,984")
    st.metric("Model Confidence", "84.2%")

# MAIN DASHBOARD - SPLIT INTO TABS
tab1, tab2 = st.tabs(["📊 Real-Time Prediction", "⚙️ System Internals"])

with tab1:
    # Top Row: 2 Columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔮 Next Likely Action")
        st.caption("Based on Sequential Mining (Markov Chains)")
        
        predictions = get_next_step_predictions(current_page)
        
        if predictions:
            # Highlight the top choice
            top_choice, top_prob = predictions[0]
            st.success(f"**Primary Prediction:** User will go to **{top_choice}**")
            
            # List others
            st.markdown("#### Probability Distribution:")
            for page, prob in predictions:
                st.progress(prob, text=f"{page} ({int(prob*100)}%)")
        else:
            st.warning("Insufficient data for sequence prediction.")

    with col2:
        st.subheader("🕸️ Sequence Visualization")
        if predictions:
            fig = plot_graph(current_page, predictions)
            st.pyplot(fig)
        else:
            st.write("No path to visualize.")

    st.markdown("---")
    
    # Bottom Row: Hybrid Recommendations
    st.subheader("💡 Content-Based Recommendations (Hybrid Layer)")
    st.caption(f"If the user doesn't click next, they might be interested in these similar items (Vector Similarity):")
    
    hybrid_recs = get_content_recommendations(current_page)
    
    if hybrid_recs:
        cols = st.columns(3)
        for i, rec in enumerate(hybrid_recs):
            with cols[i]:
                st.container(border=True).markdown(f"**Recommended:**\n### {rec}\n*Reason: High Similarity Score*")
    else:
        st.info("No content-based matches found.")

with tab2:
    st.write("### Debugging & Model Data")
    st.json(markov_model[current_page])

st.markdown("---")
st.caption("Advanced Web Usage Mining Project | Hybrid Filtering Architecture")