import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("BostonHousing.csv")

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
# 4. Normalization
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 5. Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# 6. Build the Linear Regression Model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -------------------------------
# 7. Evaluation
# -------------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# -------------------------------
# 8. Display Coefficients
# -------------------------------
coeff_df = pd.DataFrame(
    model.coef_, selected_features, columns=['Coefficient']
)
print("\nModel Coefficients:\n", coeff_df)

# -------------------------------
# 9. Predict Sample House
# -------------------------------
sample = np.array([[6.5, 12.0, 18]])  # RM, LSTAT, PTRATIO
sample_scaled = scaler.transform(sample)

predicted_price = model.predict(sample_scaled)
print("\nPredicted House Price:", predicted_price[0])

# -------------------------------
# Save Outputs
# -------------------------------
coeff_df.to_csv("simple_regression_coefficients.csv")
df.to_csv("cleaned_boston_housing.csv", index=False)

print("\nFiles Saved:")
print(" - simple_regression_coefficients.csv")
print(" - cleaned_boston_housing.csv")
