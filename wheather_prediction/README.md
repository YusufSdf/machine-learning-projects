# Weather Prediction Model for European Cities

This Python script allows you to predict the minimum temperature (temp_min) for any selected European city in the dataset using LightGBM.

Features

Automatically extracts city names from column headers.

Dynamically selects city data based on user input.

Splits data into training (90%) and testing (10%) sets.

Trains a LightGBM regressor and evaluates it with MAE and RMSE metrics.

Usage

Make sure the dataset weather_prediction_dataset.csv is in the same folder.

Run the script:

python weather_prediction_model.py


You will see a list of cities. Type the city name you want to predict (case-insensitive).

The model will train and output MAE and RMSE.

# Avrupa Şehirleri İçin Hava Tahmin Modeli

Bu Python scripti, veri setindeki herhangi bir Avrupa şehri için minimum sıcaklığı (temp_min) tahmin etmeyi sağlar. Model olarak LightGBM kullanılmıştır.

Özellikler

Kolon isimlerinden şehirleri otomatik olarak çıkarır.

Kullanıcı seçimine göre şehir verilerini dinamik olarak seçer.

Veriyi %90 eğitim, %10 test olarak ayırır.

LightGBM regresyon modeli eğitilir ve MAE ile RMSE metriğiyle değerlendirilir.

Kullanım

weather_prediction_dataset.csv dosyasının script ile aynı klasörde olduğundan emin olun.

Scripti çalıştırın:

python weather_prediction_model.py


Ekranda şehirler listelenecek. Tahmin yapmak istediğiniz şehir adını yazın (büyük/küçük harf fark etmez).

Model eğitilecek ve MAE ile RMSE değerleri ekrana yazdırılacak.
