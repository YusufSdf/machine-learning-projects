# Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

# Reading the dataset from a CSV file
data = pd.read_csv("water_potability.csv")

# Separating the target variable 'y' from the features 'x'
y = data.Potability
x_ = data.drop(["Potability"],axis=1)

# Creating an imputer to fill missing values using the median strategy
imputer = SimpleImputer(strategy="median")

# Applying the imputer to the features to fill missing values
x = imputer.fit_transform(x_)

# Splitting the data into training (90%) and testing (10%) sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# Initializing the LightGBM classifier
lgb_model = lgb.LGBMClassifier()

# This is a commented-out parameter grid for hyperparameter tuning
# param_grid = {
#     "n_estimators":[500, 1000, 1500],
#     "learning_rate":[0.01, 0.05, 0.1],
#     "num_leaves": [20, 31, 50]
# }

# Using a predefined set of "best" hyperparameters
param_grid_best = {
    "n_estimators":500,
    "learning_rate":0.01,
    "num_leaves": 50
}

# This is a commented-out line for setting up GridSearchCV
# gridSearch = GridSearchCV(estimator=lgb_model,param_grid=param_grid,cv=5,scoring="accuracy",n_jobs=-1)

# Creating a new LightGBM model with the specified best hyperparameters
lgb_model = lgb.LGBMClassifier(**param_grid_best)

# Training the model with the training data
lgb_model.fit(x_train,y_train)

# Making predictions on the test data
y_pred = lgb_model.predict(x_test)

# Commented-out lines for fitting and printing the results from GridSearchCV
# gridSearch.fit(x_train,y_train)
# print(gridSearch.best_params_)
# print(gridSearch.best_score_)

# Commented-out line for printing the accuracy score of the test set
# print(accuracy_score(y_test,y_pred))

# Reading a new, external dataset
data_new = pd.read_csv("water_ext_.csv")

# Separating the features 'x' from the new dataset
x_new = data_new.drop(["Potability"],axis=1)

# Making predictions on the new dataset
y_pred = lgb_model.predict(x_new)

# Printing the prediction results

print(y_pred)
