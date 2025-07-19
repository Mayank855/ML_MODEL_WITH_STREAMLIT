# 🔮 Streamlit ML Model Deployment App

This project demonstrates how to deploy a trained **Machine Learning model** using **Streamlit**. The app allows users to input data, generate predictions, and understand model output through a simple visualization.

---

## ✅ What This App Does

- Trains a simple **Linear Regression** model on synthetic data
- Saves the trained model automatically (`model.pkl`)
- Takes user input for 3 numerical features
- Predicts an output value based on the inputs
- Shows how each input feature impacts the prediction using a bar chart

---

## 🛠️ Steps We Followed to Build This App

### 1. **Trained a Machine Learning Model**

- Used `sklearn.datasets.make_regression()` to generate synthetic data
- Trained a **Linear Regression** model with 3 features
- Saved the trained model using `joblib`

### 2. **Built a Streamlit Web Application**

- Created a web interface using Streamlit
- Used `number_input()` widgets to accept user data
- Predicted the target using the trained model
- Displayed the result on screen

### 3. **Added Visualization**

- Multiplied each input with its corresponding model coefficient
- Displayed a **bar chart** to show feature impact on the prediction

### 4. **Made Everything Self-Contained**

- The app automatically trains and saves the model if not found
- No need for external datasets
- All logic is written in one file: `app.py`

---

## 📁 Project Structure

