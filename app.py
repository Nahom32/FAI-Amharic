import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.set_page_config(page_title="Bias Mitigation Demo", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home","📊 Model Comparisons", "📊 Model Comparison", "🧪 Live Demo"])

# ----------------------------
# Page 1: Home
# ----------------------------
if page == "🏠 Home":
    st.title("Bias Mitigation in Amharic Hate Speech Detection")
    st.markdown("""
                
    This interactive demo showcases the comparison of models before and after applying adversarial bias mitigation techniques.
                
        1. Identify and Measure Bias Across Demographic Groups
            Analyze model predictions to detect disparities in performance metrics (e.g., true positive rate, false positive rate) among different ethnic or social groups.

        2. Implement Debiasing Techniques
            Integrate methods such as adversarial training  strategies to reduce unfair treatment and prediction disparities.

        3. Evaluate Fairness and Performance Trade-offs
            Quantitatively assess the impact of debiasing methods on both accuracy and fairness metrics to understand trade-offs and improvements.

        4. Provide Group-wise Interpretability and Reporting
            Generate detailed reports of model behavior across groups to ensure transparency and support informed decision-making.            
    """)

# ----------------------------
# Page 2: Model Comparison
# ----------------------------
elif page == "📊 Model Comparisons":
    st.title("Model Comparison")

    # ----------------------------
    # Replication Comparison Section
    # ----------------------------
    st.subheader("📸 Visual Replication Comparison")

    # Add columns for side-by-side image display
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Paper Figures**")
        for i in range(1, 5):
            st.image(f"images/paper_fig{i}.jpg", caption=f"Paper Figure {i}", use_container_width=True)
    
    with col2:
        st.markdown("**Replicated Results**")
        for i in range(1, 5):
            st.image(f"images/project_fig{i}.jpg", caption=f"Replicated Figure {i}", use_container_width=True)
    
elif page == "📊 Model Comparison":
    st.title("Model Comparison")

    # Dropdown selections
    task_model = st.selectbox("Select Task Model", ["LSTM", "BiLSTM"])
    fairness_version = st.selectbox("Select Version", ["Before Mitigation", "After Mitigation"])

    # Simulated data for demo
    sample_data = {
        ("LSTM", "Before Mitigation"): {"Accuracy": 0.61, "Accuracy Disparity": 0.13, "FNR Disparity": 0.04},
        ("LSTM", "After Mitigation"): {"Accuracy": 0.58, "Accuracy Disparity": 0.05, "FNR Disparity": 0.08},
        ("BiLSTM", "Before Mitigation"): {"Accuracy": 0.62, "Accuracy Disparity": 0.04, "FNR Disparity": 0.41},
        ("BiLSTM", "After Mitigation"): {"Accuracy": 0.57, "Accuracy Disparity": 0.07, "FNR Disparity": 0.09},
    }

    metrics = sample_data[(task_model, fairness_version)]

    # Display selected model's metrics
    st.subheader(f"Metrics for {task_model} - {fairness_version}")
    st.metric("Accuracy", f"{metrics['Accuracy']:.2f}")
    st.metric("Accuracy Disparity", f"{metrics['Accuracy Disparity']:.2f}")
    st.metric("FNR Disparity", f"{metrics['FNR Disparity']:.2f}")

    # Plot bar chart
    st.subheader("Disparity Metrics Comparison")
    fig, ax = plt.subplots()
    ax.bar(["Accuracy Disparity", "FNR Disparity"], 
           [metrics['Accuracy Disparity'], metrics['FNR Disparity']], 
           color=["skyblue", "salmon"])
    ax.set_ylim(0, 1)
    st.pyplot(fig)

# ----------------------------
# Page 3: Live Demo
# ----------------------------
elif page == "🧪 Live Demo":
    st.title("Try the Model (Live Prediction)")

    st.markdown("Use a trained model to test your own Amharic text.")

    model_choice = st.selectbox("Choose Model for Prediction", ["Adversarial LSTM", "Adversarial BiLSTM"])

    user_input = st.text_area("Enter Amharic Text:", height=100)

    if st.button("Predict"):
        st.write(f"Using {model_choice} model...")
        st.warning("Prediction demo requires model loading. Add prediction code here.")
        # TODO: Add actual tokenizer + model prediction here.
        st.success("Prediction: Normal Speech ✅")
