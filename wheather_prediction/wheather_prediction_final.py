import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
# Veri setini oku
data = pd.read_csv("weather_prediction_dataset.csv")

# Extract all column names
# Tüm kolon isimlerini al
columns = data.columns

# Automatically get all city names from column names
# Kolon isimlerinden tüm şehirleri otomatik al
cities = set([col.split("_")[0] for col in columns if "_" in col])

# Show all cities and ask the user to select one
# Tüm şehirleri yazdır ve kullanıcıdan seçim yapmasını iste
print(cities)
city = input("Select city: ").upper()

# Check if the input city exists, exit if not
# Girilen şehir geçerli mi kontrol et, değilse programı durdur
if city not in cities:
    print("City not found! Please choose a valid city.")
    exit()

# Select only columns related to the chosen city
# Sadece seçilen şehirle ilgili kolonları seç
data_dus = data.loc[:, data.columns.str.contains(city)]

# Define features (X) and target (y)
# Özellikler (X) ve hedef değişken (y) belirle
x = data_dus.drop([f"{city}_temp_min"], axis=1)
y = data_dus[f"{city}_temp_min"]

# Split the data into training and testing sets
# Veriyi eğitim ve test olarak ayır
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# Initialize and train the LightGBM regressor
# LightGBM regresyon modelini oluştur ve eğit
lgb_model = LGBMRegressor(n_estimators=1000, random_state=42)
lgb_model.fit(x_train, y_train)

# Predict on the test set
# Test verisi üzerinde tahmin yap
y_pred = lgb_model.predict(x_test)

# Evaluate the model using MAE and RMSE
# Modeli MAE ve RMSE ile değerlendir
mae = mean_absolute_error(y_pred=y_pred, y_true=y_test)
rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_test))

# Print the results
# Sonuçları yazdır
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
