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

st.title("Cholinga cha Kukolola Chimanga ku Malawi (Linear Model)")
st.header("Lembani zinthu za munda wanu:")

region = st.selectbox("Chigawo", label_encoders["region"].classes_)
soil_type = st.selectbox("Mtundu wa nthaka", label_encoders["soil_type"].classes_)

temperature_level = st.selectbox("Kutentha (Low=Ozizira, Medium=Ofunda, High=Otentha)", ["Low", "Medium", "High"])
rainfall_level = st.selectbox("Kuchuluka kwa mvula (Low=Yochepa, Medium=Yokwanira, High=Yambiri)", ["Low", "Medium", "High"])

seed_variety = st.selectbox("Mtundu wa mbeu", label_encoders["seed_variety"].classes_)
fertilizer_amount = st.number_input("Kuchuluka kwa feteleza (kg)", min_value=0.0, step=1.0)
pesticide_use = st.selectbox("Mumagwiritsa ntchito mankhwala ophera tizilombo?", ["Ayi", "Inde"])
farmer_experience = st.number_input("Zaka zomwe mwalima chimanga", min_value=1, step=1)

land_size_acres = st.number_input("Kukula kwa munda (maeka)", min_value=0.1, step=0.1)
st.caption("Lembani kukula kwa munda wanu mu maeka (monga 0.5, 1.0, 3.25)")
land_size_hectares = land_size_acres * 0.4047

planting_date = st.selectbox("Nthawi yolima", label_encoders["planting_date"].classes_)
previous_crop = st.selectbox("Chomera cholima kale", label_encoders["previous_crop"].classes_)
irrigation = st.selectbox("Mumagwiritsa ntchito madzi a ulimi?", ["Ayi", "Inde"])

temperature_map = {"Low": 0, "Medium": 1, "High": 2}
rainfall_map = {"Low": 0, "Medium": 1, "High": 2}

input_dict = {
    "region": region,
    "soil_type": soil_type,
    "temperature": temperature_map[temperature_level],
    "rainfall": rainfall_map[rainfall_level],
    "seed_variety": seed_variety,
    "fertilizer_amount": fertilizer_amount,
    "pesticide_use": 1 if pesticide_use == "Inde" else 0,
    "farmer_experience": farmer_experience,
    "land_size": land_size_hectares,
    "planting_date": planting_date,
    "previous_crop": previous_crop,
    "irrigation": 1 if irrigation == "Inde" else 0,
}

# Encode categorical inputs
for col in label_encoders:
    input_dict[col] = label_encoders[col].transform([input_dict[col]])[0]

if st.button("Werengani Kukolola Chimanga"):
    input_df = pd.DataFrame([input_dict], columns=feature_names)
    predicted_yield_per_hectare = model.predict(input_df)[0]
    total_yield_kg = predicted_yield_per_hectare * land_size_hectares * 1000
    st.success(f"Kukolola kwanu kwa chimanga: **{total_yield_kg:.0f} kg** pa munda wanu")

st.caption("Zotsatira zikuonetsedwa mu makilogalamu. Kukula kwa munda kulowetsa mu maeka. Model: Linear Regression | Data: Malawi Maize Yield")
