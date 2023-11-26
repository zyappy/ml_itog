import pandas as pd
import matplotlib.pyplot as plt

import LSTM_model
from LSTM_model import well_LSTM

df = pd.read_excel('E:\study\АД\Profect_LSTM_well\model2.1\sb_model_25_11_2023\data\HDM_test.xlsx')

wells_name = ['61','128','135','146','160','170','176','177','222','235','501','503','706','710','712','714']

# well_test = well_LSTM('61', df)
# well_test._prepare_data(train_size=5, lookback=8, lookforward=1)
# well_test.create_model()
# well_test.fit(EPOCHS=20)
# well_test.predict()
# well_test.plot()
# well_test.save_model(path_model='E:\study\АД\Profect_LSTM_well\model_22_11_2023\models_data\models',
#                      path_scaler='E:\study\АД\Profect_LSTM_well\model_22_11_2023\models_data\scalers')

# test = well_test.predict_new(steps_ahead=2)
# plt.plot(test[:,0])
# plt.plot(df['Скважина_61_Qн, т/сут'].tail(20))
# plt.xlabel('Дата')
# plt.ylabel('Qн')
# plt.legend()
# plt.show()

def train_models(wells_name:list, 
                 dataset:pd.DataFrame(),
                 type:str, 
                 train_size:int, lookback:int, lookforward:int,
                 path_model:str, path_scaler:str,
                 model_type=2,
                 EPOCHS=40, BATCH_SIZE=1):
    
    for well in wells_name:
        well_model = well_LSTM(well, dataset, type=type)
        well_model._prepare_data(train_size=train_size, lookback=lookback, lookforward=lookforward)
        well_model.create_model(type=model_type)
        hist = well_model.fit(EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE, flag=2)
        well_model._predict()
        df_test = well_model.data_for_plot()
        well_model.plot()
        well_model.save_model(path_model=path_model,
                              path_scaler=path_scaler)
        print(f'Модель скважины {well} обучена и сохранена')
    
    return df_test, hist


path_model='E:\study\АД\Profect_LSTM_well\model_22_11_2023\models_data\models'
path_scaler='E:\study\АД\Profect_LSTM_well\model_22_11_2023\models_data\scalers'
df_test, hist  = train_models(['HDM'], df, 'Field', 0.85, 45, 1, path_model, path_scaler, model_type=1)
# df_test.to_excel('pfpfpf.xlsx')
#  160 - ну норм

df_test2, hist2  = train_models(['HDM'], df, 'Field', 0.85, 45, 1, path_model, path_scaler, model_type=2)


plt.plot(hist.history['loss'], label='LSTM_mono Training Loss')
plt.plot(hist.history['val_loss'], label='LSTM_mono Validation Loss')
plt.plot(hist2.history['loss'], label='LSTM_bi Training Loss')
plt.plot(hist2.history['val_loss'], label='LSTM_bi Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()
