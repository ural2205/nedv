import joblib
import streamlit as st
import pandas as pd
import numpy as np

# Загрузка модели и списка признаков
model = joblib.load('xgb_model.pkl')
feature_names = joblib.load('feature_names.pkl')

st.title('Прогнозирование цен на недвижимость')

# Ввод данных пользователем
area = st.number_input('Площадь (кв.м)', min_value=10, max_value=500, value=50)
living_area = st.number_input('Жилая площадь (кв.м)', min_value=10, max_value=500, value=30)
kitchen_area = st.number_input('Площадь кухни (кв.м)', min_value=5, max_value=100, value=10)
floor = st.number_input('Этаж', min_value=1, max_value=50, value=2)
num_floors = st.number_input('Количество этажей в доме', min_value=1, max_value=50, value=10)
minutes_to_metro = st.number_input('Минут до метро', min_value=0, max_value=60, value=10)
num_rooms = st.number_input('Количество комнат', min_value=1, max_value=10, value=2)

# Создание DataFrame для новых данных
input_data = pd.DataFrame([[area, living_area, kitchen_area, floor, num_floors, minutes_to_metro, num_rooms]], 
                          columns=['Area', 'Living area', 'Kitchen area', 'Floor', 'Number of floors', 'Minutes to metro', 'Number of rooms'])

# Преобразование категориальных данных в числовые (для совместимости с обученной моделью)
input_data = pd.get_dummies(input_data)

# Создание полного DataFrame с нулями для всех признаков
input_data_full = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

# Обновление input_data_full значениями из input_data
input_data_full.update(input_data)

# Прогнозирование
prediction = model.predict(input_data_full)

st.subheader('Результаты прогнозирования')
st.write(f'Прогнозируемая цена: {prediction[0]:,.2f} руб.')

# Добавление дополнительных метрик и информации
st.write(f'Введенные данные:')
st.write(f'Площадь: {area} кв.м')
st.write(f'Жилая площадь: {living_area} кв.м')
st.write(f'Площадь кухни: {kitchen_area} кв.м')
st.write(f'Этаж: {floor}')
st.write(f'Количество этажей в доме: {num_floors}')
st.write(f'Минут до метро: {minutes_to_metro}')
st.write(f'Количество комнат: {num_rooms}')
