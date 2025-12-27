import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(
    page_title="Restaurant Rating Predictor",
    layout="centered"
)

# -------- Load files safely --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
le_city = pickle.load(open(os.path.join(BASE_DIR, "encoder_city.pkl"), "rb"))
le_cuisine = pickle.load(open(os.path.join(BASE_DIR, "encoder_cuisine.pkl"), "rb"))

# -------- UI --------
st.title("🍽️ Restaurant Rating Prediction App")
st.write("Predict restaurant ratings using a Machine Learning model")

# Numeric inputs
votes = st.number_input(
    "Number of Votes",
    min_value=0,
    step=1
)

price_range = st.selectbox(
    "Price Range (1 = Low, 4 = High)",
    [1, 2, 3, 4]
)

# 🔥 DROPDOWNS instead of text input
city = st.selectbox(
    "Select City",
    sorted(le_city.classes_)
)

cuisine = st.selectbox(
    "Select Cuisine",
    sorted(le_cuisine.classes_)
)

# -------- Prediction --------
if st.button("Predict Rating"):
    city_encoded = le_city.transform([city])[0]
    cuisine_encoded = le_cuisine.transform([cuisine])[0]

    prediction = model.predict(
        [[votes, price_range, city_encoded, cuisine_encoded]]
    )

    rating = round(float(prediction[0]), 2)

    # Clamp rating between 0 and 5 (realistic)
    rating = max(0, min(5, rating))

    st.success(f"⭐ Predicted Restaurant Rating: **{rating} / 5**")
