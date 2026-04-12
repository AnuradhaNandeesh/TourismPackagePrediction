import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="AnuradhaNandeesh/tourism-package-prediction", filename="best_tourism_package_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a Tourism Package prediction based on its operational parameters.
Please enter the  customer data below to get a prediction.
""")

# User input
Age = st.number_input("Age ", min_value=1, max_value=150, step=1)
TypeofContact =  st.selectbox("Type of Contact ", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier ", [1,2,3])
DurationOfPitch = st.number_input("Duration Of Pitch", min_value=1, max_value=50, step=1)
Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business"])
Gender =  st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Nuumber of Persons visting", min_value=1, max_value=10, step=1)
NumberOfFollowups = st.number_input("Number of Follow ups", min_value=1, max_value=10, step=1)
ProductPitched = st.selectbox("Product Pitched", ["Deluxe", "Basic", "Standard"])
PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=3, max_value=5, step=1)
MaritalStatus = st.selectbox("Marital Status", ["Single", "Divorced", "Married","Unmarried"])
NumberOfTrips = st.number_input("Number of Trips", min_value=0, max_value=10, step=1),
Passport = st.selectbox("Passport", [1,0])
PitchSatisfactionScore = st.number_input("PitchSatisfactionScore", min_value=1, max_value=5, step=1)
OwnCar = st.selectbox("OwnCar", [1,0])
NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=10, step=1)
Designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("MonthlyIncome", min_value=1000, max_value=1000000)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome": MonthlyIncome

if st.button("Tourism Package Prediction"):
    prediction = model.predict(input_data)[0]
    result = "Product Can be taken" if prediction == 1 else "Product could Not be Taken"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
