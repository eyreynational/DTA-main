# importing python modules.
import streamlit as st
import joblib
import numpy as np
import time

# loading pickle files gotten from model
lightgbm_pickle = open("deployment/lightgbm.pickle", "rb")
lgbm_model = joblib.load(lightgbm_pickle)

# column name for each column in the diabetes dataset.
column_names = ['cholesterol', 'glucose', 'hdl_chol', 'chol_hdl_ratio', 'age',
                'gender', 'weight', 'height', 'bmi', 'systolic_bp', 'diastolic_bp', 'waist', 'hip',
                'waist_hip_ratio', 'diabetes']


# function to receive user information.
def inputs():
    # creating form for data inputs.
    with st.form(key="diabetes_data"):
        name = st.text_input("Patient's Name: ")
        gender_obj = st.selectbox(label="Patient's Gender: ", options=["Male", "Female"])
        if gender_obj == "Male":
            gender = 1
        else:
            gender = 0

        age = st.slider(label="Patient's Age: ", min_value=0, max_value=100)
        chol = st.slider(label="Patient's Cholesterol Level(mg/dL): ", min_value=40, max_value=400)
        glucose = st.slider(label="Patient's Sugar Level(mg/dL): ", min_value=40, max_value=250)
        height_cm = st.number_input(label="Patient's Height(cm): ")
        height = height_cm * 0.393701
        weight_kg = st.number_input("Patient's Weight in(kg): ")
        weight = weight_kg * 2.205
        hdl_chol = st.slider(label="Patient's HDL Cholesterol(mg/dL): ", min_value=0, max_value=100)
        waist = st.number_input("Patient's Waist Size(inches): ", step=1)
        hip = st.number_input("Patient's Hip Size(inches): ", step=1)
        systolic_bp = st.number_input(label="Patient's Systolic Blood Pressure(mmHg): ", step=1)
        diastolic_bp = st.number_input(label="Patient's Diastolic Blood Pressure(mmHg): ", step=1)
        submit = st.form_submit_button("Submit Test")
        if submit:
            bmi = weight_kg / ((height_cm / 100)**2)
            chol_hdl_ratio = chol / hdl_chol
            waist_hip_ratio = waist / hip
            patient_data = [chol, glucose, hdl_chol, chol_hdl_ratio, age, gender, weight, height, bmi,
                            systolic_bp, diastolic_bp, waist, hip, waist_hip_ratio]
        else:
            patient_data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    return patient_data


# function to create a dataframe and carry out prediction.
def predict(var_name):
    pred = [var_name]
    np_pred = np.array(pred)
    score = lgbm_model.predict(np_pred)
    return score


# function to run streamlit app
def run():
    st.title("Diabetes Test App")
    st.write("Diabetes is known as a very deadly disease if not diagnosed early. To make it easier for health "
             "practitioners to diagnose this disease early, previous data have been accumulated to predict an accurate "
             "result for new patients. "
             "The Doctor is to retrieve necessary information from the patients to carry out this test."
             " A diabetic patient should be notified early and should commence treatment immediately.")
    info = inputs()
    dia_score = predict(info)
    with st.spinner(text="Diagnosing....."):
        time.sleep(5)
    if dia_score == 1:
        st.error("Positive. Diabetes diagnosed.")
    else:
        st.success("Negative. Diabetes not Diagnosed.")
       

if __name__ == "__main__":
    run()
