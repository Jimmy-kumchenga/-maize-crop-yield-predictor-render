import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Malawi Maize Yield Predictor", page_icon="🌽")

# Load the improved model
MODEL_PATH = "improved_rf_yield_model.pkl"
model = joblib.load(MODEL_PATH)

# App title
st.title("🌽 Malawi Maize Yield Predictor")
st.write("### Enter farm details below to estimate **maize yield in kg/ha.**")

# Input features
year = st.slider("Year", 2011, 2025, 2023)
maize_type = st.selectbox("Maize Type", ["Local", "Hybrid", "OPV"])
region = st.selectbox("Region", ["Northern", "Central", "Southern"])
soil_quality = st.selectbox("Soil Quality", ["Poor", "Average", "Excellent"])
fertilizer_type = st.selectbox("Fertilizer Type", ["None", "Organic", "Inorganic", "Mixed"])
irrigated = st.radio("Is the field irrigated?", ["Yes", "No"])
crop_rotation = st.radio("Practice crop rotation?", ["Yes", "No"])
farmer_experience = st.slider("Farmer Experience (years)", 0, 40, 5)
area_ha = st.slider("Size of farm (hectares)", 0.1, 10.0, 1.0)

# Rainfall and temperature levels
rainfall_level = st.selectbox("Rainfall Level", ["Low", "Moderate", "High"])
temperature_level = st.selectbox("Temperature Level", ["Low", "Moderate", "High"])

# Map levels to values
rainfall_value = {"Low": 700, "Moderate": 1100, "High": 1600}[rainfall_level]
temperature_value = {"Low": 18.0, "Moderate": 25.0, "High": 32.0}[temperature_level]

# Fertilizer amount
fertilizer_kg = st.slider("Fertilizer Used (kg/ha)", 0, 200, 50)

# Prepare input data
input_data = pd.DataFrame([{
    "Year": year,
    "Maize_Type": maize_type,
    "Region": region,
    "Soil_Quality": soil_quality,
    "Fertilizer_Type": fertilizer_type,
    "Irrigated": 1 if irrigated == "Yes" else 0,
    "Crop_Rotation": 1 if crop_rotation == "Yes" else 0,
    "Farmer_Experience": farmer_experience,
    "Area_ha": area_ha,
    "Rainfall_mm": rainfall_value,
    "Avg_Temp_C": temperature_value,
    "Fertilizer_kg_ha": fertilizer_kg
}])

# Predict
if st.button("Predict Yield"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"🌾 Estimated Yield: **{prediction:.2f} kg/ha**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.markdown("Made with ❤️ in Malawi | Powered by Machine Learning")
