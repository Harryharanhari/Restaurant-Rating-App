import streamlit as st
import pandas as pd
import pickle

# Load saved model & encoders
model = pickle.load(open("model.pkl", "rb"))
le_city = pickle.load(open("encoder_city.pkl", "rb"))
le_cuisine = pickle.load(open("encoder_cuisine.pkl", "rb"))

st.set_page_config(page_title="Restaurant Rating Predictor", layout="centered")

st.title("🍽️ Restaurant Rating Prediction App")
st.write("Predict restaurant ratings using Machine Learning")

# User Inputs
votes = st.number_input("Number of Votes", min_value=0, step=1)
price_range = st.selectbox("Price Range (1 = Low, 4 = High)", [1, 2, 3, 4])

city = st.text_input("City Name")
cuisine = st.text_input("Cuisine Type")

if st.button("Predict Rating"):
    try:
        city_encoded = le_city.transform([city])[0]
        cuisine_encoded = le_cuisine.transform([cuisine])[0]

        prediction = model.predict([[votes, price_range, city_encoded, cuisine_encoded]])

        st.success(f"⭐ Predicted Restaurant Rating: {round(prediction[0], 2)}")

    except:
        st.error("City or Cuisine not found in training data.")
