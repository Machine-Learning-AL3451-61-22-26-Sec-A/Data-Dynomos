import streamlit as st
import pandas as pd

# Sample Hypothetical Data
data = pd.DataFrame({
    'Fever': [1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
    'Cough': [1, 1, 0, 1, 0, 0, 1, 0, 1, 1],
    'Fatigue': [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    'Travel_History': [1, 1, 1, 0, 0, 0, 1, 0, 0, 1],
    'COVID_Positive': [1, 1, 0, 1, 0, 0, 1, 0, 0, 1]
})

# Display the data
st.title("Bayesian Network for COVID-19 Diagnosis")
st.write("Sample Data:")
st.write(data)

# Define the Bayesian Network structure and parameters
# We assume conditional independence of symptoms given COVID_Positive
# We manually calculate probabilities based on the data
def predict_covid_positive(fever, cough, fatigue, travel_history):
    covid_positive_prob = data[data['COVID_Positive'] == 1]['COVID_Positive'].sum() / len(data)
    fever_given_covid_positive_prob = (data[(data['COVID_Positive'] == 1) & (data['Fever'] == fever)]['Fever'].sum() + 1) / \
                                       (data[data['COVID_Positive'] == 1]['COVID_Positive'].sum() + 2)
    cough_given_covid_positive_prob = (data[(data['COVID_Positive'] == 1) & (data['Cough'] == cough)]['Cough'].sum() + 1) / \
                                       (data[data['COVID_Positive'] == 1]['COVID_Positive'].sum() + 2)
    fatigue_given_covid_positive_prob = (data[(data['COVID_Positive'] == 1) & (data['Fatigue'] == fatigue)]['Fatigue'].sum() + 1) / \
                                         (data[data['COVID_Positive'] == 1]['COVID_Positive'].sum() + 2)
    travel_history_given_covid_positive_prob = (data[(data['COVID_Positive'] == 1) & (data['Travel_History'] == travel_history)]['Travel_History'].sum() + 1) / \
                                                (data[data['COVID_Positive'] == 1]['COVID_Positive'].sum() + 2)
    return covid_positive_prob * fever_given_covid_positive_prob * cough_given_covid_positive_prob * fatigue_given_covid_positive_prob * travel_history_given_covid_positive_prob

# Input for symptoms
st.write("Enter symptoms to diagnose:")
fever = st.selectbox("Fever", [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
cough = st.selectbox("Cough", [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
fatigue = st.selectbox("Fatigue", [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
travel_history = st.selectbox("Travel History", [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Predict the probability of COVID-19
prob_covid_positive = predict_covid_positive(fever, cough, fatigue, travel_history)

# Display the result
st.write("Probability of being COVID-19 positive:")
st.write(prob_covid_positive)
