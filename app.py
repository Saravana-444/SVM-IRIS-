import streamlit as st
import pickle
import numpy as np

# Load model
with open("SVM.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🌸 Iris Flower Prediction (SVM)")

# Inputs
sepal_length = st.number_input("Sepal Length (cm)", value=5.1)
sepal_width  = st.number_input("Sepal Width (cm)", value=3.5)
petal_length = st.number_input("Petal Length (cm)", value=1.4)
petal_width  = st.number_input("Petal Width (cm)", value=0.2)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)

    st.success(f"🌼 Predicted Iris Species: **{prediction[0]}**")
