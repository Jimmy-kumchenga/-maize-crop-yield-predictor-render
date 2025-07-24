import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_malawi_maize.csv")
    weather = pd.read_csv("malawi_district_weather..xlsx")
    return df, weather

df, weather_data = load_data()

st.title("🌽  Mlimi Adziwe")
st.markdown("🔍 Enter farm details below to estimate **maize yield in bags per hectare (1 bag = 50kg)**.")

# Input form
year = st.slider("Year", 2011, 2025, 2024)
district = st.selectbox("Select Your District", weather_data["District"].unique())
maize_type = st.selectbox("Maize Type", df["Maize_Type"].unique())
area = st.number_input("Farm Size (in hectares)", min_value=0.1, max_value=100.0, value=1.0)
fertilizer = st.slider("Fertilizer Used (kg/ha)", 0, 300, 50)

# Fetch weather values from district
district_weather = weather_data[weather_data["District"] == district].iloc[0]
avg_temp = district_weather["Avg_Temp_C"]
rainfall = district_weather["Rainfall_mm"]

# Preprocessing
X = df.drop("Yield_kg_ha", axis=1)
y = df["Yield_kg_ha"]
X = pd.get_dummies(X)

# Prepare user input
input_data = pd.DataFrame([{
    "Year": year,
    "Area_Cultivated_ha": area,
    "Rainfall_mm": rainfall,
    "Avg_Temp_C": avg_temp,
    "Fertilizer_kg_ha": fertilizer,
    **{f"Maize_Type_{maize_type}": 1}
}], columns=X.columns).fillna(0)

# Model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Prediction
if st.button("Predict Yield"):
    predicted_kg_per_ha = model.predict(input_data)[0]
    predicted_bags = predicted_kg_per_ha / 50
    total_bags = predicted_bags * area
    st.success(f"🌾 Estimated Yield: **{predicted_bags:.2f} bags/ha**")
    st.info(f"📦 Total for your farm ({area} ha): **{total_bags:.2f} bags**")
