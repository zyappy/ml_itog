import streamlit as st
import seaborn as sns
from matplotlib import pyplot as plt
import sqlite3
import pandas as pd

from global_data import GlobalData
from prepare_data import prepare_data


st.title('Загрузка данных')

# Создаем чекбоксы с радиокнопками для выбора типа модели
model_type = st.radio("Выберите тип исходных данных:", ["Данные из гидродинамического симулятора"])#, "Данные из Техрежима"])

# Используйте выбранный тип модели в вашем коде
st.write(f"Выбран тип модели: {model_type}")

# загрузка данных для тестирования моделей
uploaded_file_test = st.file_uploader("Загрузите файл данных (xlsx)", type=["xlsx"], key='test')

# -------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------------------
try:
    if GlobalData.data_load is None: GlobalData.data_load = uploaded_file_test
except: 
    GlobalData.data_load = uploaded_file_test

#Вывод данных датасета и статистики по нему
if GlobalData.data_load is not None:

    st.title('Предварительная обработка данных')

    #Отредактировать prepare_data 
    # Загружаем исходный датасет
    if model_type == "Данные из Техрежима":
        data_result = prepare_data(GlobalData.data_load, type='Field')
        st.header('Исходный загруженный датасет')

        df = data_result['init_dataset']
        st.dataframe(df)

        st.markdown('Тип загруженного датасета: {}'.format(type(df)))    
        st.markdown('\n'*2)
        st.title('Статистика по загруженному датасету')
        st.dataframe(df.describe())
        st.markdown('Строк в датасете: {}'.format(df.shape[0]))
        st.markdown('Столбцов в датасете: {}'.format(df.shape[1]))

        st.markdown('\n'*2)
        st.title("Тепловые карты корреляции")
        plt.figure(figsize=(12, 10))
        plt.subplot(1, 1, 1)
        sns.heatmap(df.corr(), annot=True, cmap='summer')
        plt.title('Корреляции параметров датасета')
        st.pyplot(plt)

        st.title('Статистика по обработанному датасету')
        st.dataframe(data_result['final_dataset'].describe())
        st.markdown('Строк в датасете: {}'.format(data_result['final_dataset'].shape[0]))
        st.markdown('Столбцов в датасете: {}'.format(data_result['final_dataset'].shape[1]))
    
        #st.title('Список скважин, которые были в начальном датасете')
        #st.dataframe(data_result['original_wells_list'])

        #st.title('Список скважин, которые остались после обработки датасета')
        #st.dataframe(data_result['filtered_wells_list'])
        GlobalData.data_type = data_result['filtered_wells_list']
        #Указать название нужного итогового датасета после обработки
        #Помещение обработанного датасета в свойство класса для передачи между страницами
        GlobalData.data_train = data_result['final_dataset']

    elif model_type == "Данные из гидродинамического симулятора":
        GlobalData.data_type = ['HDM']
        
        data_result = prepare_data(GlobalData.data_load, type='HDM')
        st.header('Исходный загруженный датасет')

        df = data_result['init_dataset']
        st.dataframe(df)

        st.markdown('Тип загруженного датасета: {}'.format(type(df)))    
        st.markdown('\n'*2)
        st.title('Статистика по загруженному датасету')
        st.dataframe(df.describe())
        st.markdown('Строк в датасете: {}'.format(df.shape[0]))
        st.markdown('Столбцов в датасете: {}'.format(df.shape[1]))

        st.markdown('\n'*2)
        st.title("Тепловые карты корреляции")
        plt.figure(figsize=(12, 10))
        plt.subplot(1, 1, 1)
        sns.heatmap(df.corr(), annot=True, cmap='summer')
        plt.title('Корреляции параметров датасета')
        st.pyplot(plt)

        GlobalData.data_train = data_result['init_dataset']

    GlobalData.data_train.to_excel('init_data.xlsx')
    