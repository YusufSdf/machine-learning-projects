import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV

# Veri setini yükleme
# TR: Wine kalite verisini CSV dosyasından yükle
# EN: Load the wine quality dataset from CSV file
data = pd.read_csv("wine_quality_merged.csv")

# 'type' kolonunu sayısal değerlere dönüştürme
# TR: 'type' kolonu, kırmızı=0, beyaz=1 olarak maplendi
# EN: Map 'type' column to numerical values, red=0, white=1
data.type = data.type.map({"red":0,"white":1})

# Özellikler ve hedef değişkeni ayırma
# TR: X = özellikler, y = quality (hedef)
# EN: X = features, y = target variable (quality)
X = data.drop(["quality"],axis=1)
y = data.quality

# Veriyi eğitim ve test olarak ayırma
# TR: %15 test boyutu ile train-test split
# EN: Train-test split with 15% test size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# GridSearchCV ile parametre arama (yorum satırında bırakıldı)
# TR: Parametre grid ve çoklu metrik ile GridSearchCV örneği (çalıştırılmadı)
# EN: Example of GridSearchCV with param grid and multiple metrics (commented out)
# paramGrid = {
#     "n_estimators": [500, 1000, 1500],
#     "learning_rate":[0.01, 0.05, 0.1],
#     "num_leaves": [20, 31, 50]
# }
# scoring = {
#     "R2":"r2",
#     "MAE":"neg_mean_absolute_error",
#     "RMSE":"neg_root_mean_squared_error"
# }
# lgb_model = LGBMRegressor()
# grid_search = GridSearchCV(param_grid=paramGrid, estimator=lgb_model, n_jobs=-1, cv=5, scoring=scoring, refit="R2")
# grid_search.fit(X_train,y_train)
# print(grid_search.best_params_)

# LightGBM modelini oluşturma ve eğitme
# TR: Modeli belirli parametrelerle tanımlayıp eğit
# EN: Define and train the model with specific parameters
lgb_model = LGBMRegressor(n_estimators=1000,learning_rate=0.05,num_leaves=50,random_state=42)
lgb_model.fit(X_train,y_train)

# Test seti tahminleri
# TR: Test verisi için tahmin yap
# EN: Predict on the test set
y_pred = lgb_model.predict(X_test)

# Model performansını ölçme
# TR: MAE, MSE ve R² skorlarını hesapla
# EN: Calculate MAE, MSE, and R² scores
mae = mean_absolute_error(y_pred=y_pred , y_true=y_test)
mse = mean_squared_error(y_pred=y_pred , y_true=y_test)
r2 = r2_score(y_pred=y_pred , y_true=y_test)

# Skorları yazdırma
# TR: Sonuçları ekrana bas
# EN: Print the results
print(f"mae: {mae}\nmse: {mse}\nr2: {r2}")
