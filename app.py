import streamlit as st
import pandas as pd
import joblib

# st.cache_resource pour charger le modèle et le scaler une seule fois
@st.cache_resource
def load_resources():
    model = joblib.load('diabetes_risk_prediction_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_resources()

st.title("Prédiction du Risque de Diabète")
st.markdown("Entrez les informations du patient :")

# Widgets de saisie
glucose = st.number_input('Glucose (mg/dL)', min_value=0, max_value=300, value=120)
bmi = st.number_input('IMC (kg/m²)', min_value=0.0, max_value=70.0, value=25.0)
age = st.number_input('Âge', min_value=0, max_value=120, value=30)
dpf = st.number_input('Fonction de Préd. Génétique au Diabète', min_value=0.0, max_value=2.5, value=0.5)

if st.button('Prédire le Risque'):
    input_data = pd.DataFrame([[glucose, bmi, age, dpf]],
                              columns=['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction'])

    # Mise à l'échelle des données d'entrée avec le scaler entraîné
    scaled_input_data = scaler.transform(input_data)

    prediction = model.predict(scaled_input_data)

    if prediction[0] == 'High Risk':
        st.error("Le patient est classé comme **risque élevé** de développer le diabète.")
    else:
        st.success("Le patient est classé comme **faible risque** de développer le diabète.")