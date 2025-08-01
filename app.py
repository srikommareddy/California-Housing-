#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load the trained ANN model (without compiling)
model = load_model("california_ann_model.h5", compile=False)

# Load the fitted scaler (used during training)
scaler = joblib.load("scaler.pkl")

# Set page title
st.title("California Housing Price Prediction App")

st.markdown("### Enter the input features below:")

# Define input fields
MedInc = st.number_input("Median Income (10k USD)", min_value=0.0, max_value=20.0, value=3.0)
HouseAge = st.number_input("House Age (years)", min_value=1.0, max_value=52.0, value=20.0)
AveRooms = st.number_input("Average Number of Rooms", min_value=0.5, max_value=15.0, value=5.0)
AveBedrms = st.number_input("Average Number of Bedrooms", min_value=0.5, max_value=5.0, value=1.0)
Population = st.number_input("Population", min_value=1.0, max_value=5000.0, value=1000.0)
AveOccup = st.number_input("Average Occupancy", min_value=0.5, max_value=10.0, value=3.0)
Latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=36.0)
Longitude = st.number_input("Longitude", min_value=-124.0, max_value=-114.0, value=-120.0)

# Predict button
if st.button("Predict"):
    # Prepare the input data
    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                            Population, AveOccup, Latitude, Longitude]])

    # Apply the same scaler used in training
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0][0]

    st.success(f"Predicted Median House Value: **${prediction * 100000:.2f}**")

