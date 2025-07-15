import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load your dataset
@st.cache_data
def load_data():
    return pd.read_csv("synthetic_malawi_maize.csv")

df = load_data()

# App title
st.title("🌽 Mlimi adziwe yield predictor")
st.markdown("🔍 Enter farm details below to estimate **maize yield in bags per hectare** (1 bag = 50kg).")

# Define input fields
year = st.slider("Year", 2011, 2025, 2024)
maize_type = st.selectbox("Maize Type", df["Maize_Type"].unique())
area = st.number_input("Farm Size (ha)", min_value=0.1, max_value=100.0, value=1.0)
rainfall = st.slider("Estimated Rainfall (mm)", 500, 2000, 1000)
temperature = st.slider("Average Temperature (°C)", 15.0, 35.0, 25.0)
fertilizer = st.slider("Fertilizer Used (kg/ha)", 0, 300, 50)

# Prepare data
X = df.drop("Yield_kg_ha", axis=1)
y = df["Yield_kg_ha"]

# One-hot encode maize type
X = pd.get_dummies(X)
input_data = pd.DataFrame([{
    "Year": year,
    "Area_Cultivated_ha": area,
    "Rainfall_mm": rainfall,
    "Avg_Temp_C": temperature,
    "Fertilizer_kg_ha": fertilizer,
    **{f"Maize_Type_{maize_type}": 1}
}], columns=X.columns).fillna(0)

# Train model in the app
model = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Predict yield
if st.button("Predict Yield"):
    prediction_kg = model.predict(input_data)[0]
    prediction_bags = prediction_kg / 50
    st.success(f"🌾 Predicted Maize Yield: **{prediction_bags:.2f} bags/ha** (50kg bags)")
