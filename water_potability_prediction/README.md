# 🇹🇷 Türkçe Açıklama

Bu proje, LightGBM algoritması kullanılarak suyun içilebilirliğini (potability) tahmin etmeye odaklanmaktadır.

Veri Seti: water_potability.csv dosyası, çeşitli kimyasal ve fiziksel parametreler içermektedir (pH, sertlik, çözünmüş katı madde vb.).

Eksik Veriler: SimpleImputer ile median stratejisi kullanılarak doldurulmuştur.

Model: LightGBM sınıflandırıcısı (LGBMClassifier) ile eğitim yapılmıştır.

Parametreler:

n_estimators=500

learning_rate=0.01

num_leaves=50

Veri Bölünmesi: %90 eğitim, %10 test.

Değerlendirme: Modelin doğruluğu (accuracy_score) hesaplanabilir.

Yeni Veri Testi: water_ext_.csv dosyası üzerinden tahmin yapılmıştır.

Bu kod, hem model eğitimi hem de yeni veriler üzerinde tahmin için kullanılabilir.

# 🇬🇧 English Description

This project focuses on predicting water potability using the LightGBM algorithm.

Dataset: water_potability.csv contains several chemical and physical parameters (pH, hardness, dissolved solids, etc.).

Missing Values: Filled using SimpleImputer with the median strategy.

Model: Trained with LightGBM classifier (LGBMClassifier).

Hyperparameters:

n_estimators=500

learning_rate=0.01

num_leaves=50

Data Split: 90% training, 10% testing.

Evaluation: Accuracy (accuracy_score) can be calculated.

New Data Testing: Predictions are made on water_ext_.csv.

This code can be used both for training the model and for making predictions on new datasets.
