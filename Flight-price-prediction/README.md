Türkçe Açıklama
Bu Python projesi, LightGBM makine öğrenmesi modelini kullanarak uçak bileti fiyatlarını tahmin etmeyi amaçlamaktadır. Projede kullanılan "Clean_Dataset.csv" ve "test_csvFile.csv" veri setleri, farklı havayolu firmalarına, kalkış ve varış şehirlerine, uçuş saatlerine ve durak sayılarına ait bilgileri içermektedir.

Temel Adımlar:

Veri Yükleme ve Hazırlık: Gerekli kütüphaneler (NumPy, Matplotlib, Pandas) ile veri setleri okunur. Model eğitimini etkilemeyen "flight" sütunu veri setlerinden çıkarılır ve kategorik değişkenler uygun formata dönüştürülür.

Model Eğitimi: Veri seti, eğitim ve test olmak üzere ikiye ayrılır. LightGBM regresyon modeli, eğitim verisi üzerinde kategorik özellikleri de kullanarak eğitilir.

Model Değerlendirme ve Tahmin: Eğitilen modelin performansı R-Kare (R2) Skoru ve Ortalama Mutlak Hata (MAE) gibi metriklerle değerlendirilir. Ardından, yeni bir test veri seti ("test_csvFile.csv") üzerinde fiyat tahminleri yapılır.

Bu kod, veri bilimi ve makine öğrenmesi alanında çalışanlar için uçak bileti fiyatlandırması gibi regresyon problemlerine LightGBM ile nasıl yaklaşılacağını gösteren basit ama etkili bir örnek sunar.

English Description
This Python project aims to predict flight ticket prices using the LightGBM machine learning model. The datasets used in the project, "Clean_Dataset.csv" and "test_csvFile.csv", contain information about different airlines, source and destination cities, departure/arrival times, and the number of stops.

Key Steps:

Data Loading and Preparation: The necessary libraries (NumPy, Matplotlib, Pandas) are used to read the datasets. The "flight" column, which doesn't affect model training, is removed, and the categorical variables are converted to the appropriate format.

Model Training: The dataset is split into training and testing sets. The LightGBM regressor model is then trained on the training data, utilizing the categorical features.

Model Evaluation and Prediction: The trained model's performance is evaluated using metrics such as R-squared (R2) Score and Mean Absolute Error (MAE). Subsequently, price predictions are made on a new test dataset ("test_csvFile.csv").

This code provides a simple yet effective example for data science and machine learning enthusiasts on how to approach regression problems, such as flight price prediction, using LightGBM.
