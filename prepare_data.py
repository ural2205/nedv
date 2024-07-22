import pandas as pd
import joblib

# Загрузка данных из файла
file_path = "D:/Центральный университет/data.csv"
data = pd.read_csv(file_path)

# Преобразование категориальных данных в числовые
data = pd.get_dummies(data, columns=['Apartment type', 'Metro station', 'Region', 'Renovation'])

# Сохранение списка признаков
feature_names = data.drop(columns=['Price']).columns

# Сохранение списка признаков в файл
joblib.dump(feature_names, 'feature_names.pkl')