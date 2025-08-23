import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read the dataset
data = pd.read_csv("Clean_Dataset.csv")
# Remove the 'flight' column as it's not needed for the prediction model
data = data.drop(["flight"], axis=1)

# Define categorical columns
categorical_columns = ["airline", "source_city", "departure_time", "stops", "arrival_time", "destination_city", "class"]
# Convert categorical columns to 'category' data type for better performance with LightGBM
for col in categorical_columns:
    data[col] = data[col].astype("category")

# Separate independent variables (x) and the dependent variable (y)
x = data.drop(["price"], axis=1)
y = data.price

# Sanitize column names by replacing special characters with underscores (required for LightGBM)
x.columns = ["".join(c if c.isalnum() else "_" for c in col) for col in x.columns]

from sklearn.model_selection import train_test_split
# Split the data into training and testing sets (10% of the data is for testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

import lightgbm as lgb
# Initialize and train the LightGBM Regressor model
# n_estimators=100: sets the number of boosting rounds (trees)
# random_state=42: ensures reproducibility
model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
# Fit the model using the training data and specify categorical features
model.fit(x_train, y_train, categorical_feature=categorical_columns)
# Make predictions on the test set
predict = model.predict(x_test)

from sklearn.metrics import r2_score
# Calculate the R-squared (R2) score to evaluate model performance
r2 = r2_score(y_test, predict)

print(f"R-Kare (R2) Skoru: {r2:.2f}")
print(f"Başarı Yüzdesi: %{r2 * 100:.2f}")

# Read and prepare the new test data
new_data = pd.read_csv("test_csvFile.csv")
new_data = new_data.drop(["flight"], axis=1)
new_data = new_data.drop(["price"], axis=1)

# Convert categorical columns in the new data to 'category' type
for col in categorical_columns:
    new_data[col] = new_data[col].astype("category")
# Make predictions on the new data
new_predict = model.predict(new_data)
print(new_predict)

from sklearn.metrics import mean_absolute_error
# Calculate the Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, predict)
print("MAE:", mae)