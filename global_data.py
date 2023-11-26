class GlobalData:
    def __init__(self):
        
        self.data_train = None  #Датафрейм обучения
        self.data_test = None   #Датафрейм проверки
        self.data_full = None   #Датафрейм для прогноза
        self.data_load = None   #Датафрейм загрузки
        
        self.model = None       #Сама модель
        self.data_type = None  #Тип данных (ГДМ/ТР)
        self.model_type = None  #Тип модели
        self.model_epoch = None       #количество эпох
        self.model_batch_size = None  #Размер батча
        self.model_status = None       #Сама модель
        
        self.err_stat = None    #Метрики качества
        
        self.data_pred = None   #Датафрейм прогноза
        self.pred_type = None    #Тип прогноза
        self.pred_fr = None    #Частота ЭЦН для прогноза
        self.pred_Qinj = None    #Значение закачки воды
        self.pred_mounth = None    #Количество месяцев на прогноз

        self.pred_Qoil = None   #Новый дебит нефти
        self.pred_Qliq = None   #Новый дебит жидкости
        self.pred_Date = None   #Новая дата факта


#obj = GlobalData()
#obj.data_train = ""