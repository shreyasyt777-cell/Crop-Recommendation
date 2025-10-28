# Crop_app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os

st.set_page_config(page_title="🌾 Smart Crop Recommendation System", layout="wide")

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.image("https://thumbs.dreamstime.com/b/cute-cartoon-fruits-faces-blue-background-plum-orange-lemon-watermelon-pear-good-picture-391466388.jpg?w=992", use_container_width=True)
    st.title("🌱 Crop Recommendation")
    st.markdown("**Find the best crop to grow based on your soil and weather conditions.** 🌿")
    st.markdown("---")
    st.markdown("💡 *Grow smarter, not harder — let data guide your farming decisions!* 🌾")

# ---------------------------
# Load Model Files
# ---------------------------
required_files = ["crop_model.pkl", "scaler.pkl", "label_encoder.pkl"]
missing = [f for f in required_files if not os.path.exists(f)]

if missing:
    st.error(f"❌ Model files not found! Please run `model_training.py` first to generate them.")
    st.stop()

rf = pickle.load(open("crop_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# ---------------------------
# Crop Recommendation Section
# ---------------------------
st.header("🌱 Crop Recommendation Tool")
st.markdown("Provide the soil and climate parameters below:")

col1, col2, col3 = st.columns(3)
with col1:
    N = st.number_input("Nitrogen (N)", 0, 150, 50)
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
with col2:
    P = st.number_input("Phosphorus (P)", 0, 150, 50)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0)
with col3:
    K = st.number_input("Potassium (K)", 0, 200, 50)
    ph = st.number_input("pH Value", 0.0, 14.0, 6.5)
rainfall = st.slider("🌧️ Rainfall (mm)", 0.0, 300.0, 100.0)

if st.button("🔍 Recommend Crops"):
    sample = pd.DataFrame([{
        "N": N, "P": P, "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])
    x_s = scaler.transform(sample)
    probs = rf.predict_proba(x_s)[0]
    top3_idx = np.argsort(probs)[::-1][:3]
    top3_crops = [(le.inverse_transform([i])[0], probs[i]) for i in top3_idx]

    st.success(f"✅ Best crop to plant: **{top3_crops[0][0].upper()}** 🌾")
    st.write("### 🌿 Top 3 Recommended Crops:")
    for crop, p in top3_crops:
        st.write(f" - {crop.title()} ({p*100:.2f}%)")

    # Plot probabilities
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(x=[c[0] for c in top3_crops], y=[c[1]*100 for c in top3_crops], palette="viridis", ax=ax)
    ax.set_ylabel("Probability (%)")
    ax.set_title("Top 3 Crop Probabilities")
    st.pyplot(fig)

st.markdown("---")
st.markdown("<center>🌾 Empowering farmers with AI-driven insights — grow smarter every season! 🌱</center>", unsafe_allow_html=True)
