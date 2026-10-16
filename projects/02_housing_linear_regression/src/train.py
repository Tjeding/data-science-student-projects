from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# 1. Load Dataset
# -------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORTS_DIR = PROJECT_DIR / "reports"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(PROJECT_DIR / "data" / "raw" / "BostonHousing.csv")

print("Initial Shape:", df.shape)

# -------------------------------
# 2. Clean Data
# -------------------------------
df.drop_duplicates(inplace=True)
print("Shape After Removing Duplicates:", df.shape)

print("\nMissing Values:\n", df.isnull().sum())

# -------------------------------
# 3. Feature Selection
#    Choose common meaningful predictors:
#    RM (avg rooms), LSTAT (% lower-income),
#    PTRATIO (pupil-teacher ratio)
# -------------------------------
selected_features = ["rm", "lstat", "ptratio"]
X = df[selected_features]
y = df["medv"]

# -------------------------------
# 4. Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 5. Standardization (fit on training data only)
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 6. Build the Linear Regression Model
# -------------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# -------------------------------
# 7. Evaluation
# -------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared Score:", r2)

# -------------------------------
# 8. Display Coefficients
# -------------------------------
coeff_df = pd.DataFrame(
    model.coef_, selected_features, columns=['Coefficient']
)
print("\nModel Coefficients (per one-standard-deviation feature increase):\n", coeff_df)

# -------------------------------
# 9. Predict Sample House
# -------------------------------
sample = pd.DataFrame([[6.5, 12.0, 18]], columns=selected_features)
sample_scaled = scaler.transform(sample)

predicted_price = model.predict(sample_scaled)
print("\nPredicted House Price:", predicted_price[0])

# -------------------------------
# Save Outputs
# -------------------------------
coeff_df.to_csv(REPORTS_DIR / "regression_coefficients.csv")
df.to_csv(PROCESSED_DIR / "cleaned_boston_housing.csv", index=False)

print("\nFiles Saved:")
print(f" - {REPORTS_DIR / 'regression_coefficients.csv'}")
print(f" - {PROCESSED_DIR / 'cleaned_boston_housing.csv'}")
