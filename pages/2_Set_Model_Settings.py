import streamlit as st
from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt

from global_data import GlobalData
from prepare_data import prepare_data
from LSTM_model import well_LSTM
import plot_all


st.title('Выбор параметров модели')

#Проверка на получение датасета с предыдущей страницы
try: 
    if GlobalData.data_train is not None:
            st.title('Датасет до обучения')
            st.dataframe(GlobalData.data_train)
    else: a=1
except: a=2

s = ["Односторонняя модель LSTM", "Двусторонняя модель LSTM"]

#Отрисовка страницы, если датасет передан из предыдущей страницы
# try: 
if GlobalData.data_train is not None:
        #---------------------------------------------
        #---------------------------------------------  

        #Выбор модели ИИ
        model_choice = st.radio("Выберите модель для тестирования", ["Односторонняя модель LSTM", "Двусторонняя модель LSTM"])

        if model_choice == 'Односторонняя модель LSTM': 
            n = 0
            GlobalData.model_type = 1
        else: 
            n = 1
            GlobalData.model_type = 2

        #Окна ввода параметров модели и их вывода в итнерфейс
        epoch = st.text_input("Введите количество эпох для обучения:")
        #check_int = True
        try:
            user_input_as_int = int(epoch)
            GlobalData.model_epoch = int(user_input_as_int)
            st.write("Вы ввели число:", str(int(user_input_as_int)))
            #if float(user_input_as_int).is_integer(): check_int = False
        except ValueError:
            st.write("Пожалуйста, введите корректное целочисленное число.")

        batch_size = st.text_input("Введите размер батча (batch_size):")
        try:
            user_input_as_int = int(batch_size)
            GlobalData.model_batch_size = user_input_as_int
            st.write("Вы ввели число:", str(int(user_input_as_int)))
            #if float(user_input_as_int).is_integer(): check_int = False
        except ValueError:
            st.write("Пожалуйста, введите корректное целочисленное число.")


        #Функция проверки введенных параметров модели и их вывода в интерфейс
        def run_model(model_type, epoch, batch_size):
            # Здесь вы можете определить действие, которое нужно выполнить
            try:
                input_value = int(epoch*batch_size)
                if input_value <= 0:
                    result = "Пожалуйста, введите целочисленные значения больше нуля"
                if input_value > 0:
                    #result = "Модель запущена на обучение с настройками: количество эпох = {}, размер батча = {}".format(epoch, batch_size)
                    result = 'Обучение модели завершено'
                    #st.write("Обучение модели завершено")
                    
######################################################################################
                    #Функция модели - запуск обучения. 
                    #Аргументы (GlobalData.model_type, 
                    #           GlobalData.model_epoch, 
                    #           GlobalData.model_batch_size)

                    def train_models(wells_name:list, 
                        dataset:pd.DataFrame(),
                        type:str, 
                        train_size:int, lookback:int, lookforward:int,
                        path_model='Empty', path_scaler='Empty',
                        model_type=2,
                        EPOCHS=1, BATCH_SIZE=1):
            
                        for well in wells_name:
                            well_model = well_LSTM(well, dataset, type=type)
                            well_model._prepare_data(train_size=train_size, lookback=lookback, lookforward=lookforward)
                            well_model.create_model(type=model_type)
                            data_for_metric = well_model.fit(EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE, flag=2)
                            well_model._predict()
                            data_for_plot = well_model.data_for_plot()
                            # well_model.plot()
                            # well_model.save_model(path_model=path_model,
                            #                     path_scaler=path_scaler)
                            print(f'Модель скважины {well} обучена и сохранена')
                        
                        return data_for_plot, data_for_metric

                    data_for_plot, data_for_metric  = train_models(GlobalData.data_type, GlobalData.data_train, 'Field', 0.85, 45, 1, model_type=model_type, EPOCHS=epoch, BATCH_SIZE=batch_size)
                    data_for_plot.to_excel('data_for_plot.xlsx')
                    try:
                        GlobalData.data_train = GlobalData.data_train.drop('NumericDate', axis=1)
                    except:
                        n=2
                

                    #Вывод метрик качества
                    temp_text = 'Временные данные'
                    st.header('Метрики качества модели')
                    # Построение графика
                    fig, ax = plt.subplots()
                    ax.plot(data_for_metric.history['loss'], label='LSTM Training Loss')
                    ax.plot(data_for_metric.history['val_loss'], label='LSTM Validation Loss')
                    ax.legend()
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('Training and Validation Loss')

                    # Отображение графика в Streamlit
                    st.pyplot(fig)
######################################################################################

                return result
            except ValueError:
                result = "Пожалуйста, введите корректные данные, ошибка \n{}".format(ValueError) 
                return result


        # Кнопка для запуска обучения модели
        if st.button("Запустить обучение модели"):

            # Действие, которое выполняется при нажатии кнопки
            try:
                st.write("Модель запущена на обучение с настройками: тип модели = {}".format(s[n]))
                st.write("Модель запущена на обучение с настройками: количество эпох = {}".format(epoch))
                st.write("Модель запущена на обучение с настройками: размер батча = {}".format(batch_size))
                st.write('Идет обучение...')
                result = run_model(GlobalData.model_type, int(GlobalData.model_epoch), int(GlobalData.model_batch_size))
                #Временный датафрейм
                data_result = pd.read_excel('data_for_plot.xlsx')
                st.dataframe(data_result)
                #Вывод графиков
                plot_all.plot_all(data_result)
                st.write(result)
                st.write("Модель готова к прогнозу")
                GlobalData.model_status = True
            

            except ValueError:
                result = "Пожалуйста, введите корректные данные, ошибка {}".format(ValueError) 
                st.write("Результат действия:", result)

            #st.write("Обучение модели завершено")
else: st.title('Модель не обучена!')
# except: st.title('Введите исходные данные на предыдущих страницах, а также проверьте чтобы число эпох и размер батча были целочисленными')