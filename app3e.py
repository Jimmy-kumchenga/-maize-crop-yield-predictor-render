import streamlit as st
import pandas as pd
import joblib
import requests

# Your OpenWeatherMap API key
api_key = "b2ce5ea1b6698bca803b5699bac6e194"

# Malawi district coordinates (add all 28 for full coverage)
districts_coords = {
    "Lilongwe": {"lat": -13.9833, "lon": 33.7833},
    "Blantyre": {"lat": -15.7861, "lon": 35.0058},
    "Mzuzu": {"lat": -11.4656, "lon": 34.0207},
    "Zomba": {"lat": -15.3833, "lon": 35.3333},
    "Mangochi": {"lat": -14.4781, "lon": 35.2645},
    "Kasungu": {"lat": -13.0333, "lon": 33.4833},
    "Balaka": {"lat": -14.9790, "lon": 35.1709},
    "Blantyre": {"lat": -15.7861, "lon": 35.0058},
    "Chikwawa": {"lat": -16.0333, "lon": 34.8000},
    "Chiradzulu": {"lat": -15.7167, "lon": 35.1833},
    "Chitipa": {"lat": -9.7031, "lon": 33.2707},
    "Dedza": {"lat": -14.3667, "lon": 34.3333},
    "Dowa": {"lat": -13.6500, "lon": 33.9333},
    "Karonga": {"lat": -9.9333, "lon": 33.9333},
    "Kasungu": {"lat": -13.0333, "lon": 33.4833},
    "Likoma": {"lat": -12.0667, "lon": 34.7333},
    "Lilongwe": {"lat": -13.9833, "lon": 33.7833},
    "Machinga": {"lat": -14.9667, "lon": 35.5167},
    "Mangochi": {"lat": -14.4781, "lon": 35.2645},
    "Mchinji": {"lat": -13.8000, "lon": 32.8833},
    "Mulanje": {"lat": -16.0333, "lon": 35.5000},
    "Mwanza": {"lat": -15.6000, "lon": 34.5167},
    "Mzimba": {"lat": -11.9000, "lon": 33.6000},
    "Mzuzu": {"lat": -11.4656, "lon": 34.0207},
    "Nkhata Bay": {"lat": -11.6000, "lon": 34.3000},
    "Nkhotakota": {"lat": -12.9167, "lon": 34.3000},
    "Nsanje": {"lat": -16.9333, "lon": 35.2667},
    "Ntcheu": {"lat": -14.8167, "lon": 34.6333},
    "Ntchisi": {"lat": -13.3333, "lon": 34.0167},
    "Phalombe": {"lat": -15.6667, "lon": 35.6500},
    "Rumphi": {"lat": -11.0167, "lon": 33.8667},
    "Salima": {"lat": -13.7833, "lon": 34.4333},
    "Thyolo": {"lat": -16.0667, "lon": 35.1333},
    "Zomba": {"lat": -15.3833, "lon": 35.3333}
}

def get_weather(lat, lon, api_key):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    temperature = data['main']['temp']
    rainfall = data.get('rain', {}).get('1h', 0)
    return temperature, rainfall

# Load trained model and label encoders
model = joblib.load("maize_yield_linear_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")

feature_names = [
    "region", "soil_type", "temperature", "rainfall", "seed_variety",
    "fertilizer_amount", "pesticide_use", "farmer_experience", "land_size",
    "planting_date", "previous_crop", "irrigation"
]

st.title("Mlimi Adziwe🌽 )")
st.header("Enter your farm details:")

district = st.selectbox("Select District", list(districts_coords.keys()))
lat = districts_coords[district]["lat"]
lon = districts_coords[district]["lon"]

if st.button("Fetch Real-Time Weather Data"):
    try:
        temperature, rainfall = get_weather(lat, lon, api_key)
        st.success(f"District: {district}\nTemperature: {temperature}°C\nRainfall (last hour): {rainfall} mm")
    except Exception as e:
        st.error("Could not fetch weather data. Please check your internet connection and API key.")
else:
    temperature, rainfall = None, None

region = st.selectbox("Region", label_encoders["region"].classes_)
soil_type = st.selectbox("Soil Type", label_encoders["soil_type"].classes_)
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

# If weather data was fetched, use it; otherwise ask user to fetch
if temperature is None or rainfall is None:
    st.warning("Please click the 'Fetch Real-Time Weather Data' button to get temperature and rainfall for your district.")
else:
    st.write(f"Using weather data: Temperature={temperature}°C, Rainfall={rainfall} mm (last hour)")

    # Prepare input dictionary for prediction
    input_dict = {
        "region": region,
        "soil_type": soil_type,
        "temperature": temperature,
        "rainfall": rainfall,
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

st.caption("Yield is shown in kilograms. Land size input is in acres. Model: Linear Regression | Data: Malawi Maize Yield. Real-time weather data powered by OpenWeatherMap.")

