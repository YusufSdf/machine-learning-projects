
# Proje Açıklaması
Bu proje, bir evin özelliklerine dayalı olarak satış fiyatını tahmin etmek için makine öğrenimi kullanır. Emlak verilerini analiz etmek ve gelecekteki satış fiyatlarını yüksek doğrulukla tahmin etmek amacıyla, güçlü bir gradient boosting algoritması olan LightGBM modeli tercih edilmiştir.

Veri işleme ve modelleme adımları, temiz ve verimli bir iş akışı sağlamak için scikit-learn Pipeline içinde düzenlenmiştir. Bu yaklaşım, nümerik ve kategorik verilerin ayrı ayrı işlenmesini kolaylaştırır:

Nümerik Veri: Fiyat tahmininin doğruluğunu artırmak için StandardScaler ile ölçeklendirilir.

Kategorik Veri: Makine öğrenimi modelinin anlayabileceği sayısal formata dönüştürülmesi için OneHotEncoder kullanılır.

Sonuç olarak, model eğitilir, performansı değerlendirilir ve yeni, görülmemiş evler için fiyat tahminleri yapılır.

Kullanılan Teknolojiler
Python: Projenin temel programlama dili.

Pandas & NumPy: Veri manipülasyonu ve analizi için.

Scikit-learn: Makine öğrenimi araçları, veri ön işleme, modelleme ve pipeline oluşturma için.

LightGBM: Etkili ve hızlı bir gradient boosting framework'ü.

# Project Description
This project uses machine learning to predict house sale prices based on various property features. It leverages LightGBM, a highly efficient gradient boosting algorithm, to analyze real estate data and make accurate predictions for future sale prices.

The data preprocessing and modeling steps are structured within a scikit-learn Pipeline to create a clean and efficient workflow. This approach simplifies the handling of different data types:

Numeric Data: Scaled using StandardScaler to improve model performance.

Categorical Data: Converted into a numerical format that the model can understand using OneHotEncoder.

Ultimately, the model is trained, its performance is evaluated, and predictions are made on new, unseen house data.

Technologies Used
Python: The core programming language for the project.

Pandas & NumPy: For data manipulation and analysis.

Scikit-learn: Provides machine learning tools, including data preprocessing, modeling, and pipeline creation.

LightGBM: An efficient and fast gradient boosting framework.
