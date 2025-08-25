# Wine Quality Prediction

## Açıklama / Description
Bu proje, kırmızı ve beyaz şarapların kimyasal özelliklerine göre kalitesini tahmin etmeyi amaçlamaktadır. 
Veri seti, "wine_quality_merged.csv" dosyasından yüklenmektedir.

This project aims to predict the quality of red and white wines based on their chemical properties.
The dataset is loaded from "wine_quality_merged.csv".

## Kullanılan Teknolojiler / Technologies
- Python 3.x
- pandas
- scikit-learn
- lightgbm

## Özellikler / Features
- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol
- type (red=0, white=1)

## Hedef / Target
- quality (şarabın puanı)

## Kurulum / Installation
1. Python ve gerekli kütüphaneleri yükleyin:
   ```bash
   pip install pandas scikit-learn lightgbm
