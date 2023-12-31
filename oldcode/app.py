#from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
import streamlit as st

st.set_page_config(page_title="Team NishJay",page_icon="logo.png")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Credit Score Calc SyStem")

@st.cache_resource
def getModel():
    return load_model('credit_scoring_model.h5')

with st.form("Credit Score Calculator"):
    age = st.number_input("Age:", min_value=0, step=1)
    revolving_utilization = st.number_input("Revolving Utilization of Unsecured Lines (0 to 1):", min_value=0.0, max_value=1.0)
    num_30_59_days_past_due = st.number_input("Number of Times 30-59 Days Past Due:", min_value=0, step=1)
    debt_ratio = st.number_input("Debt Ratio:", min_value=0.0)
    monthly_income = st.number_input("Monthly Income:", min_value=0.0)
    num_open_credit_lines = st.number_input("Number of Open Credit Lines and Loans:", min_value=0, step=1)
    num_90_days_late = st.number_input("Number of Times 90 Days Late:", min_value=0, step=1)
    num_real_estate_loans = st.number_input("Number of Real Estate Loans or Lines:", min_value=0, step=1)
    num_60_89_days_past_due = st.number_input("Number of Times 60-89 Days Past Due:", min_value=0, step=1)
    num_dependents = st.number_input("Number of Dependents:", min_value=0, step=1)

    if st.form_submit_button("Predict Credit Risk"):
        with open('model.pkl','rb') as file:
            loaded_model = pickle.load(file)
            user_input = np.array([
                revolving_utilization,
                age,
                num_30_59_days_past_due,
                debt_ratio,
                monthly_income,
                num_open_credit_lines,
                num_90_days_late,
                num_real_estate_loans,
                num_60_89_days_past_due,
                num_dependents
                ]).reshape(1, -1) 

            #scaler = StandardScaler()        
            #user_input_scaled = scaler.transform(user_input)

            prediction_prob = loaded_model.predict_proba(user_input)
           # prediction_class = (prediction_prob > 0.5).astype(int)[0]
            #st.write(prediction_class)
            print(prediction_prob)
            st.write(prediction_prob[0][0]*1000)
            st.write("Credit Risk Prediction Result: [Your Result Here]")
