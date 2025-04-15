import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset and model
training_dataset = pd.read_csv("Training.csv")
doc_dataset = pd.read_csv("doctors_dataset.csv", names=["Name", "Description"])

X = training_dataset.iloc[:, :-1]
y = training_dataset["prognosis"]

labelencoder = LabelEncoder()
y_encoded = labelencoder.fit_transform(y)

model = DecisionTreeClassifier()
model.fit(X, y_encoded)

# Prepare doctor suggestions
diseases = labelencoder.classes_
doctors = pd.DataFrame({
    "disease": diseases,
    "name": doc_dataset["Name"],
    "link": doc_dataset["Description"]
})

# UI
st.title("🩺 Healthcare Diagnosis Chatbot")
st.write("Select the symptoms you are experiencing:")

symptoms = list(X.columns)
selected_symptoms = st.multiselect("Choose your symptoms", symptoms)

if st.button("Diagnose"):
    # Create input vector
    input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]
    
    # Predict
    prediction = model.predict([input_vector])
    disease = labelencoder.inverse_transform(prediction)[0]
    
    # Find doctor info
    doc_info = doctors[doctors["disease"] == disease].iloc[0]
    
    st.success(f"🩺 You may have **{disease}**")
    st.write(f"**Suggested Doctor:** {doc_info['name']}")
    st.markdown(f"[🔗 Visit Doctor Site]({doc_info['link']})")
