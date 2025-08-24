Bu proje, bir veri kümesindeki çeşitli faktörleri kullanarak bir kişinin yaşadığı stres türünü tahmin eden bir makine öğrenimi modeli oluşturur. Proje, veri ön işleme, model eğitimi ve model performans değerlendirme adımlarını içerir.

Proje Tanımı
Bu Python betiği, sağlanan "Stress_Dataset.csv" veri kümesini okur ve onu bir makine öğrenimi modeli için hazırlar. LightGBM (LGBMClassifier) modeli, veriler üzerindeki kategorik ve sayısal özellikleri kullanarak üç ana stres türünden birini tahmin etmek üzere eğitilir: Negatif Stres (Distress), Pozitif Stres (Eustress) veya Stres Yok (No Stress).

Kullanılan Kütüphaneler
Bu projenin çalışması için aşağıdaki Python kütüphaneleri gereklidir:

pandas: Veri işleme ve analizi için.

numpy: Sayısal işlemler için.

scikit-learn: Model seçimi, veri ön işleme ve performans metrikleri için.

lightgbm: Modelin kendisi için.

re: Metin temizleme için.

Bu kütüphaneleri terminalinizden aşağıdaki komutla kurabilirsiniz:
pip install pandas numpy scikit-learn lightgbm

Dosya Yapısı
stress_model.py: Bu README'nin açıklandığı ana Python betiği.

Stress_Dataset.csv: Betiğin çalıştığı klasörde bulunması gereken ham veri dosyası.

Betiğin Açıklaması
Veri Okuma: Betik, Stress_Dataset.csv dosyasını bir DataFrame'e okuyarak başlar.

Veri Ön İşleme:

Kategorik Değişken Dönüşümü: Hedef değişken (Which type of stress...) get_dummies kullanılarak sayısal bir formata dönüştürülür.

Sütun Adı Temizliği: Sütun adları, LightGBM modelinin gereksinimlerini karşılamak için özel karakterlerden arındırılır.

Hedef Değişken Tanımlama: Makine öğrenimi modelinin tahmin edeceği ana hedef (Stress_Type_Num) oluşturulur ve geri kalan sütunlar özellikler (x) olarak ayrılır.

Model Eğitimi:

Veri, eğitim ve test kümelerine ayrılır (train_test_split).

Bir LGBMClassifier modeli, eğitim verileriyle (x_train, y_train) eğitilir.

Model Değerlendirme:

Eğitilen model, test verilerinin bir alt kümesi üzerinde tahminler yapar.

Modelin doğruluğu, accuracy_score, classification_report ve confusion_matrix gibi metrikler kullanılarak değerlendirilir.

# Stress Level Prediction Model
This project creates a machine learning model that predicts the type of stress a person is experiencing based on various factors in a dataset. The project includes steps for data preprocessing, model training, and model performance evaluation.

Project Description
This Python script reads the provided "Stress_Dataset.csv" dataset and prepares it for a machine learning model. A LightGBM (LGBMClassifier) model is trained to predict one of three main stress types: Negative Stress (Distress), Positive Stress (Eustress), or No Stress, using the categorical and numerical features in the data.

Required Libraries
The following Python libraries are required for this project to run:

pandas: For data manipulation and analysis.

numpy: For numerical operations.

scikit-learn: For model selection, data preprocessing, and performance metrics.

lightgbm: For the model itself.

re: For text cleaning.

You can install these libraries using the following command in your terminal:
pip install pandas numpy scikit-learn lightgbm

File Structure
stress_model.py: The main Python script described in this README.

Stress_Dataset.csv: The raw data file, which should be located in the same directory as the script.

Script Breakdown
Data Loading: The script begins by reading the Stress_Dataset.csv file into a pandas DataFrame.

Data Preprocessing:

Categorical Variable Conversion: The target variable (Which type of stress...) is converted into a numerical format using get_dummies.

Column Name Cleaning: Column names are cleaned of special characters to be compatible with the LightGBM model's requirements.

Target Variable Identification: The main target (Stress_Type_Num) for the machine learning model is created, and the remaining columns are separated as features (x).

Model Training:

The data is split into training and testing sets (train_test_split).

An LGBMClassifier model is trained with the training data (x_train, y_train).

Model Evaluation:

The trained model makes predictions on a subset of the test data.

The model's accuracy is evaluated using metrics like accuracy_score, classification_report, and confusion_matrix.
