import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
df = pd.read_csv("synthetic_maize_yield_malawi.csv")

# Encode categorical features
categorical_cols = ["region", "soil_type", "seed_variety", "planting_date", "previous_crop"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df.drop("yield", axis=1)
y = df["yield"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Linear Regression R² Score: {r2:.3f}")
print(f"Linear Regression RMSE: {rmse:.3f}")

# Save model and encoders
joblib.dump(model, "maize_yield_linear_model.joblib")
joblib.dump(label_encoders, "label_encoders.joblib")
print("Linear model saved as maize_yield_linear_model.joblib")
print("Label encoders saved as label_encoders.joblib")
