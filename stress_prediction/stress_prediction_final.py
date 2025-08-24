import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
import re

# Veri setini oku / Load the dataset
data = pd.read_csv("Stress_Dataset.csv")

# Kategorik değişkenleri one-hot encoding ile sayısal hale getir / Convert categorical variable to numeric via one-hot encoding
data_dum =  pd.get_dummies(data=data,columns=["Which type of stress do you primarily experience?"]).astype(int)

# Her satır için en baskın stres türünü bul / Find dominant stress type for each row
data_dum["stress_type"] = data_dum[[
    "Which type of stress do you primarily experience?_Distress (Negative Stress) - Stress that causes anxiety and impairs well-being.", 
    "Which type of stress do you primarily experience?_Eustress (Positive Stress) - Stress that motivates and enhances performance.", 
    "Which type of stress do you primarily experience?_No Stress - Currently experiencing minimal to no stress."
]].idxmax(1)

# Stres türünü sayısal hale getir / Encode stress type to numeric
le = LabelEncoder()
data_dum["Stress_Type_Num"] = le.fit_transform(data_dum["stress_type"])

# Orijinal kategorik sütunları kaldır / Drop original categorical columns
data_dum.drop([
    "stress_type",
    "Which type of stress do you primarily experience?_Distress (Negative Stress) - Stress that causes anxiety and impairs well-being.",
    "Which type of stress do you primarily experience?_No Stress - Currently experiencing minimal to no stress.",
    "Which type of stress do you primarily experience?_Eustress (Positive Stress) - Stress that motivates and enhances performance."
],axis=1,inplace=True)

# Sütun isimlerinden özel karakterleri kaldır / Clean column names
data_dum.columns = data_dum.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

# Özellikleri ve hedef değişkeni ayır / Split features and target
x = data_dum.drop(["Stress_Type_Num"],axis=1)
y = data_dum.Stress_Type_Num

# Veriyi eğitim ve test olarak ayır / Split data into train and test sets
x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# Modeli tanımla ve eğit / Define and train model
lg_model = LGBMClassifier(n_estimators=500,random_state=42)
lg_model.fit(X=x_train , y=y_train)

# Test setinden örnek al ve tahmin et / Sample from test set and predict
X_sample, y_sample = x_test.sample(frac=0.5, random_state=1), y_test.sample(frac=0.5, random_state=1)
y_pred = lg_model.predict(X= X_sample)

# Başarı oranını hesapla / Calculate accuracy
score = accuracy_score(y_true=y_sample, y_pred=y_pred)
print("Accuracy:", score)

# Sınıflandırma raporu ve confusion matrix / Print classification report and confusion matrix
print("\nClassification Report:\n", classification_report(y_sample, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_sample, y_pred))
