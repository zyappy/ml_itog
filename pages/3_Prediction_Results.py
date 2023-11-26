import streamlit as st
from global_data import GlobalData
from prepare_data import prepare_data
import pandas as pd
import numpy as np

from LSTM_model import well_LSTM


uploaded_file_y = None
s = ["Односторонняя модель LSTM", "Двусторонняя модель LSTM"]
s1 = ["Постоянные данные"] #["Датасет",
st.title('Прогнозирование данных')

def add_data(df, new_qoil, new_qwater):
    # Определим новую дату как следующий месяц от последней даты в датафрейме
    last_date = df['Дата'].max()
    new_date = last_date + pd.DateOffset(months=1)

    # Создадим новую строку с новыми значениями и новой датой
    new_row = {'Дата': new_date, 'Суммарная Qн, т/сут': new_qoil, 'Суммарная Qж, м3/сут': new_qwater}

    # Добавим новую строку в датафрейм
    df = df.append(new_row, ignore_index=True)

    return df



try: 
    if GlobalData.data_train is not None \
        and GlobalData.model_type is not None \
        and GlobalData.model_epoch is not None \
        and GlobalData.model_batch_size is not None \
        and GlobalData.model_status == True:

            st.title('Параметры модели')
            st.markdown('Тренировочный датасет')
            GlobalData.data_full = pd.read_excel('init_data.xlsx')
            GlobalData.data_full = GlobalData.data_full[['Дата','Суммарная Qн, т/сут', 'Суммарная Qж, м3/сут']]
            st.dataframe(GlobalData.data_train)
            
            if GlobalData.model_type == False:
                st.markdown('Тип модели: {}'.format(s[0]))
            else:
                 st.markdown('Тип модели: {}'.format(s[1]))
            st.markdown('Количество эпох: {}'.format(GlobalData.model_epoch))
            st.markdown('Размер батча: {}'.format(GlobalData.model_batch_size))      
    
            model_choice = st.radio("Выберите тип прогноза", ["Постоянные данные"]) #["Датасет",
            if model_choice == "Постоянные данные": GlobalData.pred_type = True
            #if model_choice == 'Датасет': GlobalData.pred_type = False
            #else: GlobalData.pred_type = True

            if GlobalData.pred_type == False: 
                #Ввод датасета для прогноза (теста)
                st.write("Введите выборку для тестирования модели")
                uploaded_file_y = st.file_uploader("Загрузите файл данных (xlsx)", type=["xlsx"], key='test')

            else:
            #Окна вввода постоянных параметров на прогноз 
                Qliq = st.text_input("Введите дебит жидкости:")
                try:
                    user_input_as_int = float(Qliq)
                    GlobalData.pred_Qliq = user_input_as_int
                    st.write("Вы ввели число:", user_input_as_int)
                except ValueError:
                    st.write("Пожалуйста, введите корректное число.")

                Qoil = st.text_input("Введите дебит нефти:")
                try:
                    user_input_as_int = float(Qoil)
                    GlobalData.pred_Qoil = user_input_as_int
                    st.write("Вы ввели число:", user_input_as_int)
                except ValueError:
                    st.write("Пожалуйста, введите корректное число.")

                #n_mounth = st.text_input("Введите прогнозный период в месяцах:")
                #try:
                #    user_input_as_int = float(n_mounth)
                #    GlobalData.pred_mounth = user_input_as_int
                #    st.write("Вы ввели число:", user_input_as_int)
                #except ValueError:
                #    st.write("Пожалуйста, введите корректное число.")
                #st.markdown('Готовность к расчетам: {}'.format(str((float(fr)*float(Qinj)*float(n_mounth))>0)))


            try: 
                #if uploaded_file_y is not None: 
                #    st.title('Предварительный просмотр данных')
                #    data_result = prepare_data(uploaded_file_y)
                #    st.dataframe(data_result['init_dataset'])
                
                #uploaded_file_y is not None or
                if GlobalData.pred_Qoil is not None and GlobalData.pred_Qliq is not None:
                    if  st.button("Добавить данные в таблицу"):
                    # df = pd.read_excel('D:\LSTM\sb_model_25_11_2023\data\HDM_test.xlsx')
                        GlobalData.data_full = add_data(GlobalData.data_full, GlobalData.pred_Qoil, GlobalData.pred_Qliq)
                        st.write(str(GlobalData.pred_Qoil))
                        st.write(str(GlobalData.pred_Qliq))

                        st.dataframe(GlobalData.data_full.tail(20))
                        GlobalData.data_full.to_excel('init_data.xlsx')
                    


                    #Действие, которое выполняется при нажатии кнопки
                if st.button("Запустить расчет прогноза"):
                        
                        st.write("Модель запущена на прогноз")
                        st.write("Расчет прогноза...")

######################################################################################
                        #Определяем по каким данным посчитать прогноз (датасет или постоянные значения)
                        #if GlobalData.pred_type == False:
                            #Функция модели - запуск прогноза. 
                            #Аргументы функции (GlobalData.pred_type, 
                            #           GlobalData.model_type, 
                            #           data_result, 
                            #           GlobalData.model_epoch, 
                            #           GlobalData.model_batch_size)
                        #else:
                            #Функция модели - запуск прогноза. 
                            #Аргументы функции (GlobalData.pred_type, 
                            #           GlobalData.model_type, 
                            #           GlobalData.pred_liq,
                            #           GlobalData.pred_Qoil,
                            #           data_result, 
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
                                    well_model._prepare_data(train_size=1, lookback=lookback, lookforward=lookforward)
                                    well_model.create_model(type=model_type)
                                    well_model.fit(EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE, flag=1)
                                    print(1)
                                    data_for_plot = well_model.predict_new()
                                    print(2)
                                    # well_model.plot()
                                    # well_model.save_model(path_model=path_model,
                                    #                     path_scaler=path_scaler)
                                    print(f'Модель скважины {well} обучена и сохранена')
                                return data_for_plot
                        data_for_plot  = train_models(['HDM'], GlobalData.data_full, 'Field', 1, 45, 1, model_type=1, EPOCHS=GlobalData.model_epoch, BATCH_SIZE=GlobalData.model_batch_size)
                        st.dataframe(GlobalData.data_full.drop('NumericDate', axis=1).tail(20))
                        st.header('Q нефти на следующий месяц: {:0.1f}, т/сут'.format(data_for_plot[0, 0]))
                        st.header('Q жидкости на следующий месяц: {:0.1f}, м3/сут'.format( data_for_plot[0, 1]))

                        # #Временный датафрейм
                        # import pandas as pd
                        # data_result = pd.read_excel(r'C:\Users\serge\Documents\analys_ml\itog\model_22_11_2023\data\InData.xlsx')
                        # st.dataframe(data_result)

                        # #Вывод графиков
                        # import plot_all
                        # plot_all.plot_all(data_result)
######################################################################################

            except Exception as e:
                print(e)

    else: st.title('Модель не обучена!')
except: st.title('Введите исходные данные на предыдущих страницах')