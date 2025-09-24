import pandas as pd
import numpy as np

np.random.seed(42)

regions = ["Central", "Northern", "Southern"]
soil_types = ["sandy", "clay", "loam"]
seed_varieties = ["local", "hybrid", "OPV"]
planting_dates = ["early", "normal", "late"]
previous_crops = ["maize", "groundnuts", "soybean", "none"]

n_samples = 500

data = {
    "region": np.random.choice(regions, n_samples),
    "soil_type": np.random.choice(soil_types, n_samples),
    "temperature": np.random.uniform(18, 32, n_samples).round(1),
    "rainfall": np.random.uniform(400, 1200, n_samples).round(1),
    "seed_variety": np.random.choice(seed_varieties, n_samples),
    "fertilizer_amount": np.random.uniform(0, 250, n_samples).round(1),
    "pesticide_use": np.random.choice([0, 1], n_samples),
    "farmer_experience": np.random.randint(1, 31, n_samples),
    "land_size": np.random.uniform(0.5, 5.0, n_samples).round(2),
    "planting_date": np.random.choice(planting_dates, n_samples),
    "previous_crop": np.random.choice(previous_crops, n_samples),
    "irrigation": np.random.choice([0, 1], n_samples),
}

df = pd.DataFrame(data)

# Generate yield with some plausible relationships
yield_base = 1.5  # average base yield in tons/ha
yield_mod = (
    0.7 * (df["seed_variety"] == "hybrid") +
    0.3 * (df["seed_variety"] == "OPV") +
    0.5 * (df["soil_type"] == "loam") +
    0.2 * (df["irrigation"] == 1) +
    0.01 * df["fertilizer_amount"] +
    0.05 * df["pesticide_use"] +
    0.02 * df["farmer_experience"] +
    -0.3 * (df["soil_type"] == "sandy") +
    -0.2 * (df["planting_date"] == "late") +
    np.random.normal(0, 0.3, n_samples)
)

df["yield"] = (yield_base + yield_mod).round(2)
df["yield"] = df["yield"].clip(lower=0.5, upper=5.5)  # realistic bounds

# Save to CSV
df.to_csv("synthetic_maize_yield_malawi.csv", index=False)
print("Synthetic dataset generated: synthetic_maize_yield_malawi.csv")
