# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import joblib
import os

# Train a simple regression model and save it
def train_model():
    X, y = make_regression(n_samples=200, n_features=3, noise=10, random_state=42)
    df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
    df['target'] = y

    X_train, X_test, y_train, y_test = train_test_split(df[['feature1', 'feature2', 'feature3']], df['target'], test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, 'model.pkl')

# Load the model if it exists, otherwise train it
if os.path.exists("model.pkl"):
    model = joblib.load("model.pkl")
else:
    train_model()
    model = joblib.load("model.pkl")

# Streamlit App
st.title("Simple ML Prediction App")

st.write("Enter the values for the features below to get a predicted result.")

# User input
f1 = st.number_input("Enter Feature 1", value=0.0)
f2 = st.number_input("Enter Feature 2", value=0.0)
f3 = st.number_input("Enter Feature 3", value=0.0)

input_data = pd.DataFrame([[f1, f2, f3]], columns=['feature1', 'feature2', 'feature3'])

# Predict button
if st.button("Predict"):
    result = model.predict(input_data)[0]
    st.subheader(f"Predicted Value: {result:.2f}")

    # Simple visualization
    st.write("Feature Impact (approximate):")
    impacts = input_data.values[0] * model.coef_

    fig, ax = plt.subplots()
    ax.bar(['feature1', 'feature2', 'feature3'], impacts, color='skyblue')
    ax.set_ylabel("Impact on Prediction")
    st.pyplot(fig)
