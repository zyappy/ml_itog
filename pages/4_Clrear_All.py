from global_data import GlobalData
import streamlit as st

if st.button("Очистить данные"):
    GlobalData.data_train = None  #Датафрейм обучения
    GlobalData.data_test = None   #Датафрейм провGlobalData.
    GlobalData.model = None       #Сама модель
    GlobalData.model_type = None  #Тип модели
    GlobalData.model_epoch = None       #количество эпох
    GlobalData.model_batch_size = None  #Размер батча
    GlobalData.model_status = None       #Сама моGlobalData.
    GlobalData.err_stat = None    #Метрики качеGlobalData.
    GlobalData.data_pred = None   #Датафрейм прогноза
    GlobalData.pred_type = None    #Тип прогноза
    GlobalData.pred_fr = None    #Частота ЭЦН для прогноза
    GlobalData.pred_Qinj = None    #Значение закачки воды
    GlobalData.pred_mounth = None    #Количество месяцев на прогноз
    st.markdown('Данные очищены')