import streamlit as st
import pandas as pd
import numpy as np
import joblib

df_test = pd.read_csv("Resources/admissions.csv")

categorical = []
numerical = ['date_admission']

customers = df_test[categorical + numerical].iloc[:10].to_dict(orient='records')

def predict(df, dv, model):
    cat = df[categorical + numerical].to_dict(orient='rows')
    
    X = dv.transform(cat)

    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

# predict the probability of the 10 customers for all three antibiotics ciprofloxacin, gentamicin, and amoxicillin_clavulanic_acid
dv_ciprofloxacin = joblib.load("dv_ciprofloxacin.pkl")
model_ciprofloxacin = joblib.load("model_ciprofloxacin.pkl")
y_pred_ciprofloxacin = predict(df_test, dv_ciprofloxacin, model_ciprofloxacin)

dv_gentamicin = joblib.load("dv_gentamicin.pkl")
model_gentamicin = joblib.load("model_gentamicin.pkl")
y_pred_gentamicin = predict(df_test, dv_gentamicin, model_gentamicin)

dv_amoxicillin_clavulanic_acid = joblib.load("dv_amoxicillin_clavulanic_acid.pkl")
model_amoxicillin_clavulanic_acid = joblib.load("model_amoxicillin_clavulanic_acid.pkl")
y_pred_amoxicillin_clavulanic_acid = predict(df_test, dv_amoxicillin_clavulanic_acid, model_amoxicillin_clavulanic_acid)

# if the probability is greater than 0.5, then create a new column called ciprofloxacin_previous_R and set the value to True else False
df_test['ciprofloxacin_previous_R'] = y_pred_ciprofloxacin > 0.5
df_test['gentamicin_previous_R'] = y_pred_gentamicin > 0.5
df_test['amoxicillin_clavulanic_acid_previous_R'] = y_pred_amoxicillin_clavulanic_acid > 0.5


def previous_resistance(patient_id, antibiotic):
    if patient_id in df_test.patient_id.values:
        if antibiotic == "ciprofloxacin":
            return df_test[df_test.patient_id == patient_id].ciprofloxacin_previous_R.values[0]
        elif antibiotic == "gentamicin":
            return df_test[df_test.patient_id == patient_id].gentamicin_previous_R.values[0]
        elif antibiotic == "amoxicillin_clavulanic_acid":
            return df_test[df_test.patient_id == patient_id].amoxicillin_clavulanic_acid_previous_R.values[0]
    else:
        return False
    
def predict(df, dv, model):
    cat = df[categorical + numerical].to_dict(orient='rows')
    
    X = dv.transform(cat)

    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

def doctor_alert(patient_id, antibiotic):
    if previous_resistance(patient_id, antibiotic):
        return "Patient has resistance to " + antibiotic
    else:
        if antibiotic == "ciprofloxacin":
            y_pred = predict(df_test[df_test.patient_id == patient_id], dv_ciprofloxacin, model_ciprofloxacin)
            return "Patient has a " + str(y_pred[0]) + " chance of resistance to " + antibiotic
        elif antibiotic == "gentamicin":
            y_pred = predict(df_test[df_test.patient_id == patient_id], dv_gentamicin, model_gentamicin)
            return "Patient has a " + str(y_pred[0]) + " chance of resistance to " + antibiotic
        elif antibiotic == "amoxicillin_clavulanic_acid":
            y_pred = predict(df_test[df_test.patient_id == patient_id], dv_amoxicillin_clavulanic_acid, model_amoxicillin_clavulanic_acid)
            return "Patient has a " + str(y_pred[0]) + " chance of resistance to " + antibiotic
        
st.title('Antibiotic Resistance Prediction')

patient_id = st.number_input('Enter patient ID', min_value=1, max_value=10000, value=4440)
antibiotic = st.selectbox('Select antibiotic', ["ciprofloxacin", "gentamicin", "amoxicillin_clavulanic_acid"])

if st.button('Predict'):
    result = doctor_alert(patient_id, antibiotic)
    st.write(result)
