import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os

# Title
st.title("🌽 Malawi Maize Yield Predictor")
st.write("Provide farm details to estimate **yield (kg/ha)**.")

# Load dataset
DATA_PATH = "synthetic_malawi_maize.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# Preprocess
df.dropna(inplace=True)

X = df.drop("Yield_kg_ha", axis=1)
y = df["Yield_kg_ha"]

categorical_features = ["Maize_Type", "Region", "Soil_Quality", "Fertilizer_Type"]
binary_features = ["Irrigated", "Crop_Rotation"]
numeric_features = ["Year", "Farmer_Experience", "Area_ha", "Rainfall_mm", "Avg_Temp_C", "Fertilizer_kg_ha"]

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("bin", "passthrough", binary_features),
    ("num", StandardScaler(), numeric_features)
])

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# User Input
st.subheader("🌾 Enter your farm details")

year = st.slider("Year", 2011, 2025, 2024)
maize_type = st.selectbox("Maize Type", df["Maize_Type"].unique())
region = st.selectbox("Region", df["Region"].unique())
soil_quality = st.selectbox("Soil Quality", df["Soil_Quality"].unique())
fertilizer_type = st.selectbox("Fertilizer Type", df["Fertilizer_Type"].unique())
irrigated = st.selectbox("Is the land irrigated?", [0, 1])
crop_rotation = st.selectbox("Crop rotation practiced?", [0, 1])
experience = st.number_input("Farmer experience (years)", 0, 50, 5)
area = st.number_input("Farm size (ha)", 0.1, 100.0, 1.0)
rainfall = st.slider("Estimated rainfall (mm)", 500, 2000, 1000)
temperature = st.slider("Estimated average temp (°C)", 15.0, 35.0, 24.5)
fertilizer_kg = st.slider("Fertilizer used (kg/ha)", 0, 300, 50)

# Predict
if st.button("Predict Yield"):
    input_df = pd.DataFrame([{
        "Year": year,
        "Maize_Type": maize_type,
        "Region": region,
        "Soil_Quality": soil_quality,
        "Fertilizer_Type": fertilizer_type,
        "Irrigated": irrigated,
        "Crop_Rotation": crop_rotation,
        "Farmer_Experience": experience,
        "Area_ha": area,
        "Rainfall_mm": rainfall,
        "Avg_Temp_C": temperature,
        "Fertilizer_kg_ha": fertilizer_kg
    }])

    prediction = model_pipeline.predict(input_df)[0]
    st.success(f"🌾 Predicted Maize Yield: **{prediction:.2f} kg/ha**")
