#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load saved model and scaler
model = load_model("california_ann_model.h5")
scaler = joblib.load("scaler.pkl")

st.title("California Housing Price Prediction")

# Input fields
feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
user_input = []

for name in feature_names:
    val = st.number_input(f"Enter {name}", value=0.0)
    user_input.append(val)

# Predict
if st.button("Predict Price"):
    X = np.array(user_input).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)
    st.success(f"Predicted Median House Value: ${prediction[0][0] * 100000:.2f}")

