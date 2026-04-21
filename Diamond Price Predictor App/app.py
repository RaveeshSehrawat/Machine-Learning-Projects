import streamlit as st
import pandas as pd
import pickle

# Load model (make sure LightGBM is installed if your model uses it)
with open("diamond_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Diamond Price Predictor")

st.title("💎 Diamond Price Predictor")
st.write("Enter the diamond features to predict price")

# Sliders
carat = st.slider("Carat", 0.2, 5.0, 1.0)
depth = st.slider("Depth", 40.0, 80.0, 60.0)
table = st.slider("Table", 40.0, 90.0, 60.0)

x = st.slider("Length (x)", 0.0, 11.0, 5.0)
y = st.slider("Width (y)", 0.0, 60.0, 5.0)
z = st.slider("Height (z)", 0.0, 40.0, 3.0)

# Derived feature
area = x * y * z

# Categorical inputs
cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])

# Encoding mappings (must match training)
cut_map = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
color_map = {"D": 0, "E": 1, "F": 2, "G": 3, "H": 4, "I": 5, "J": 6}
clarity_map = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}

# Prediction
if st.button("Predict Price"):
    input_df = pd.DataFrame({
        "carat": [carat],
        "cut": [cut_map[cut]],
        "color": [color_map[color]],
        "clarity": [clarity_map[clarity]],
        "depth": [depth],
        "table": [table],
        "area": [area]
    })

    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Price: ${prediction:.2f}")