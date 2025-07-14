import streamlit as st
import pandas as pd
import joblib
import os

# Set page title
st.set_page_config(page_title="Malawi Maize Yield Predictor", page_icon="🌽")

# Title
st.title("🌽 Malawi Maize Yield Predictor")
st.write("Provide basic farm details to estimate **yield (kg/ha)**.")

# Load the model
MODEL_PATH = "improved_rf_yield_model.pkl"  # Change this if you're using a different model file name
model = joblib.load(MODEL_PATH)

# Collect user inputs
with st.form("yield_form"):
    st.subheader("Farm Inputs")

    year = st.selectbox("Year", list(range(2011, 2026))[::-1])

    maize_type = st.selectbox("Type of Maize", ["Hybrid", "Local", "OPV"])
    region = st.selectbox("Region", ["Northern", "Central", "Southern"])
    soil_quality = st.selectbox("Soil Quality", ["Poor", "Average", "Excellent"])
    fertilizer_type = st.selectbox("Fertilizer Type", ["Organic", "Inorganic", "Mixed"])

    irrigated = st.radio("Is the farm irrigated?", ["Yes", "No"])
    rotation = st.radio("Do you practice crop rotation?", ["Yes", "No"])

    farmer_experience = st.slider("Farmer Experience (years)", 0, 40, 5)
    area = st.slider("Farm Size (ha)", 0.1, 10.0, 1.0)
    rainfall = st.slider("Estimated Rainfall (mm)", 300, 2000, 1000)
    temperature = st.slider("Average Temperature (°C)", 18.0, 35.0, 25.0)
    fertilizer_kg_ha = st.slider("Fertilizer Used (kg/ha)", 0, 500, 150)

    submitted = st.form_submit_button("📈 Predict Yield")

# When form is submitted
if submitted:
    # Convert binary fields
    irrigated_bin = 1 if irrigated == "Yes" else 0
    rotation_bin = 1 if rotation == "Yes" else 0

    # Create input DataFrame with correct feature order
    input_data = pd.DataFrame([{
        "Year": year,
        "Maize_Type": maize_type,
        "Region": region,
        "Soil_Quality": soil_quality,
        "Fertilizer_Type": fertilizer_type,
        "Irrigated": irrigated_bin,
        "Crop_Rotation": rotation_bin,
        "Farmer_Experience": farmer_experience,
        "Area_ha": area,
        "Rainfall_mm": rainfall,
        "Avg_Temp_C": temperature,
        "Fertilizer_kg_ha": fertilizer_kg_ha
    }])

    # Ensure the feature columns are in the same order as model expects
    expected_columns = [
        "Year", "Maize_Type", "Region", "Soil_Quality",
        "Fertilizer_Type", "Irrigated", "Crop_Rotation",
        "Farmer_Experience", "Area_ha", "Rainfall_mm",
        "Avg_Temp_C", "Fertilizer_kg_ha"
    ]

    input_data = input_data[expected_columns]

    # Make prediction
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"✅ Estimated Yield: **{prediction:.2f} kg/ha**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
