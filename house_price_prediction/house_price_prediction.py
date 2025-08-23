import pandas as pd  # For data manipulation and reading CSVs
import numpy as np  # For numerical operations
from lightgbm import LGBMRegressor  # LightGBM regression model
from sklearn.model_selection import train_test_split  # To split data into training and testing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Metrics to evaluate model
from sklearn.compose import ColumnTransformer  # To apply different preprocessing to different columns
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Scaling numeric and encoding categorical features
from sklearn.pipeline import Pipeline  # To chain preprocessing and model steps together

# Load training data
data_train = pd.read_csv("train.csv")  # Reads 'train.csv' into a DataFrame
x = data_train.drop(["Id","SalePrice"], axis=1)  # Feature set: drop ID and target columns
y = data_train.SalePrice  # Target variable: SalePrice

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)  
# 85% training, 15% testing, fixed random state for reproducibility

# Identify numeric and categorical columns
numeric_features = x.select_dtypes(include=["int64", "float64"]).columns  # Columns with numeric data
categorical_features = x.select_dtypes(include=["object"]).columns  # Columns with categorical (string) data

# Preprocessing steps for the pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),  # Scale numeric features
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)  # Encode categorical features
    ]
)

# Create a machine learning pipeline
pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),  # Step 1: preprocessing
    ("model", LGBMRegressor(random_state=42))  # Step 2: model
])

# Train the model
pipeline.fit(x_train, y_train)  # Fit both preprocessing and model on training data
preds = pipeline.predict(x_test)  # Predict on the test set
print(preds)  # Show predicted SalePrice values

# Evaluate model performance
mae = mean_absolute_error(y_test, preds)  # Mean Absolute Error
mse = mean_squared_error(y_test, preds)  # Mean Squared Error
r2 = r2_score(y_test, preds)  # R-squared score

# Print evaluation results
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

# Predict on new, unseen data
data_test = pd.read_csv("test.csv")  # Load test data
data_test_without_id = data_test.drop(["Id"], axis=1)  # Remove ID column
new_pred = pipeline.predict(data_test_without_id)  # Make predictions
data_test_id = data_test.Id  # Keep IDs for submission

# Prepare submission DataFrame
df = pd.DataFrame({
    "Id": data_test_id,
    "SalePrice": new_pred
})

# Show first 25 rows of the submission
print(df.head(25))
