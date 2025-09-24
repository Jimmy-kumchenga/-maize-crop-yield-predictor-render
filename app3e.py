import streamlit as st
import pandas as pd
import joblib

# Load the trained Linear Regression model and label encoders
model = joblib.load("maize_yield_linear_model.joblib")  # Use the linear model
label_encoders = joblib.load("label_encoders.joblib")

feature_names = [
    "region", "soil_type", "temperature", "rainfall", "seed_variety",
    "fertilizer_amount", "pesticide_use", "farmer_experience", "land_size",
    "planting_date", "previous_crop", "irrigation"
]

st.title("Malawi Maize Yield Predictor (Linear Model)")
st.header("Enter your farm details:")

region = st.selectbox("Region", label_encoders["region"].classes_)
soil_type = st.selectbox("Soil Type", label_encoders["soil_type"].classes_)

temperature_level = st.selectbox("Temperature Level (Low=Cool, Medium=Warm, High=Hot)", ["Low", "Medium", "High"])
rainfall_level = st.selectbox("Rainfall Level (Low=Dry, Medium=Average, High=Wet)", ["Low", "Medium", "High"])

seed_variety = st.selectbox("Seed Variety", label_encoders["seed_variety"].classes_)
fertilizer_amount = st.number_input("Fertilizer Amount (kg)", min_value=0.0, step=1.0)
pesticide_use = st.selectbox("Do you use pesticides?", ["No", "Yes"])
farmer_experience = st.number_input("Years of maize farming experience", min_value=1, step=1)

land_size_acres = st.number_input("Land Size (acres)", min_value=0.1, step=0.1)
st.caption("Enter your total field size in acres (e.g., 0.5, 1.0, 3.25)")
land_size_hectares = land_size_acres * 0.4047

planting_date = st.selectbox("Planting Date", label_encoders["planting_date"].classes_)
previous_crop = st.selectbox("Previous Crop", label_encoders["previous_crop"].classes_)
irrigation = st.selectbox("Do you use irrigation?", ["No", "Yes"])

temperature_map = {"Low": 0, "Medium": 1, "High": 2}
rainfall_map = {"Low": 0, "Medium": 1, "High": 2}

input_dict = {
    "region": region,
    "soil_type": soil_type,
    "temperature": temperature_map[temperature_level],
    "rainfall": rainfall_map[rainfall_level],
    "seed_variety": seed_variety,
    "fertilizer_amount": fertilizer_amount,
    "pesticide_use": 1 if pesticide_use == "Yes" else 0,
    "farmer_experience": farmer_experience,
    "land_size": land_size_hectares,
    "planting_date": planting_date,
    "previous_crop": previous_crop,
    "irrigation": 1 if irrigation == "Yes" else 0,
}

# Encode categorical inputs
for col in label_encoders:
    input_dict[col] = label_encoders[col].transform([input_dict[col]])[0]

if st.button("Predict Maize Yield"):
    input_df = pd.DataFrame([input_dict], columns=feature_names)
    predicted_yield_per_hectare = model.predict(input_df)[0]
    total_yield_kg = predicted_yield_per_hectare * land_size_hectares * 1000
    st.success(f"Your predicted maize yield is: **{total_yield_kg:.0f} kg** for your field.")

st.caption("Yield is shown in kilograms. Land size input is in acres. Model: Linear Regression | Data: Malawi Maize Yield")
