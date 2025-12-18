import streamlit as st
import pickle
import numpy as np

# Load the trained SVM model
with open("SVM.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🌸 Iris Flower Prediction (SVM Model)")

st.write("Enter flower measurements to predict the Iris species")

# Iris input features
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.1)
sepal_width  = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, value=1.4)
petal_width  = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)

    species_map = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    st.success(f"🌼 Predicted Iris Species: **{species_map[prediction[0]]}**")
