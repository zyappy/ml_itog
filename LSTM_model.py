import pandas as pd
import numpy as np
# Библиотеки для нейронных сетей
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout
import joblib
# Библиотеки для построения графиков
import matplotlib.pyplot as plt
import plotly.express as px


class well_LSTM:
    def __init__(self, well_name: str, dataset: pd.DataFrame(), type='Field'):
        dataset['NumericDate'] = pd.to_numeric(dataset['Дата'])
        self.well_name = well_name
        self.dataset = dataset
        if type == 'Field':
            self.features = ['Суммарная Qн, т/сут', 'Суммарная Qж, м3/сут']
            self.target = ['Суммарная Qн, т/сут', 'Суммарная Qж, м3/сут']
            self.well_name = 'Field'
        elif type == 'Wells':
            # self.features = ['NumericDate', f'Скважина_{well_name}_Буферное давление, атм', f'Скважина_{well_name}_Частота работы ЭЦН, Гц', 'Закачка_Qж, м3/сут', f'Скважина_{well_name}_Qн, т/сут', f'Скважина_{well_name}_Qж, м3/сут']
            # self.features = [f'Скважина_{well_name}_Буферное давление, атм', f'Скважина_{well_name}_Частота работы ЭЦН, Гц', f'Скважина_{well_name}_Qн, т/сут', f'Скважина_{well_name}_Qж, м3/сут']
            self.features = [f'Скважина_{well_name}_Qн, т/сут', f'Скважина_{well_name}_Qж, м3/сут']
            self.target = [f'Скважина_{well_name}_Qн, т/сут', f'Скважина_{well_name}_Qж, м3/сут']
            # self.target = ['Q']

    def _prepare_data(self, train_size=0.8, lookback=5, lookforward=1):

        # Масштабирование данных
        self.scaler_features = MinMaxScaler()
        self.scaler_target = MinMaxScaler()
        
        # Создание временных последовательностей для обучения LSTM
        look_back = lookback  # число временных шагов, которые модель будет использовать для предсказания следующего значения
        X, y = [], []

        scaled_features = self.scaler_features.fit_transform(self.dataset[self.features])  
        scaler_target = self.scaler_target.fit_transform(self.dataset[self.target])
        self.scaled_features = scaled_features

        for i in range(len(scaled_features) - look_back):
            X.append(scaled_features[i:(i + look_back), :])
            y.append(scaled_features[i + look_back, -2:])  

        X, y = np.array(X), np.array(y)
        self.y = y

        # Разделение данных на обучающий и тестовый наборы
        train_size = int((len(scaled_features) - look_back) * train_size)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        self.y_train = y_train
        self.y_test = y_test

        self.X_train = X_train
        self.X_test = X_test  

        self.input_shape=(X.shape[1], X.shape[2])
        self.train_size = train_size
        self.look_back = lookback
        self.lookforward = lookforward
        return self

    def create_model(self, type=2):
        self.type = type
        lstm_model = Sequential()
        # 1 - Однонаправленный
        # 2 - Двунаправленный
        if type == 1:
           lstm_model.add(LSTM(units=256, return_sequences=True, input_shape=self.input_shape))
           lstm_model.add(LSTM(128, return_sequences=False))
           lstm_model.add(Dense(50))
           lstm_model.add(Dense(2))
        if type == 2:
           #Двунаправленный LSTM для (чтобы информация из более ранней части последовательности была доступна для последней, и наоборот)
           lstm_model.add(Bidirectional(LSTM(units=50, return_sequences=True, input_shape=self.input_shape)))
           lstm_model.add(Dropout(0.2)) #решение проблемы переобучения
           lstm_model.add(LSTM(units= 30, return_sequences=True))
           lstm_model.add(Dropout(0.2)) # 0.2 - вероятность, чем меньше, тем лучше 0.2, 0.3, 0.4. 0.8 много
           lstm_model.add(LSTM(units= 30, return_sequences=True))
           lstm_model.add(Dropout(0.2))
           lstm_model.add(LSTM(units= 30, return_sequences=False))
           lstm_model.add(Dropout(0.2))
           lstm_model.add(Dense(units = 2 * self.lookforward, activation='relu'))

        lstm_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        self.lstm_model = lstm_model
        
        return self

    def fit(self, EPOCHS=150, BATCH_SIZE=2, flag=1):
        if flag==1:
            return self.lstm_model.fit(self.X_train, self.y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
        else:
            return self.lstm_model.fit(self.X_train, self.y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(self.X_test, self.y_test))

    def _predict(self):
        # Прогнозирование на тестовых данных
        y_pred = self.lstm_model.predict(self.X_test)
        y_recalc = self.lstm_model.predict(self.X_train)
        # # Инверсия шкалы для получения оригинальных значений
        # self.y_pred_inv = self.scaler_target.inverse_transform(np.concatenate((self.X_test[:, -2:, :], y_pred), axis=1)).squeeze()
        # self.y_test_inv = self.scaler_target.inverse_transform(np.concatenate((self.X_test[:, -2:, :], self.y_test), axis=1)).squeeze()

        # self.y_recalc_inv = self.scaler_target.inverse_transform(np.concatenate((self.X_train[:, -2:, :], y_recalc), axis=1))[:, :, :]
        # self.y_train_inv = self.scaler_target.inverse_transform(np.concatenate((self.X_train[:, -2:, :], self.y_train), axis=1))[:, :, :]

        # Обратное масштабирование предсказанных и фактических значений
        self.y_pred_inv = self.scaler_target.inverse_transform(y_pred)
        self.y_recalc_inv = self.scaler_target.inverse_transform(y_recalc)
        self.y_test_inv = self.scaler_target.inverse_transform(self.y_test)
        self.y_train_inv = self.scaler_target.inverse_transform(self.y_train)

        return self
    
    def data_for_plot(self):

        df_plot = pd.DataFrame()

        size = len(self.dataset)

        y_pred_inv_oil_list = [pd.NA] * size
        y_recalc_inv_oil_list = [pd.NA] * size
        y_hist_inv_oil_list = [pd.NA] * size

        y_pred_inv_liq_list = [pd.NA] * size
        y_recalc_inv_liq_list = [pd.NA] * size
        y_hist_inv_liq_list = [pd.NA] * size


        # Заменяем последние элементы list1 значениями из list2
        y_hist_inv_oil_list[-len(self.y_test_inv[:, 0]):] = self.y_test_inv[:, 0]
        y_hist_inv_oil_list[self.look_back:len(self.y_train_inv[:, 0]) + self.look_back] = self.y_train_inv[:, 0]
        y_pred_inv_oil_list[-len(self.y_pred_inv[:, 0]):] = self.y_pred_inv[:, 0]
        y_recalc_inv_oil_list[self.look_back:len(self.y_recalc_inv[:, 0]) + self.look_back] = self.y_recalc_inv[:, 0]

        y_hist_inv_liq_list[-len(self.y_test_inv[:, 1]):] = self.y_test_inv[:, 1]
        y_hist_inv_liq_list[self.look_back:len(self.y_train_inv[:, 1]) + self.look_back] = self.y_train_inv[:, 1]
        y_pred_inv_liq_list[-len(self.y_pred_inv[:, 1]):] = self.y_pred_inv[:, 1]
        y_recalc_inv_liq_list[self.look_back:len(self.y_recalc_inv[:, 1]) + self.look_back] = self.y_recalc_inv[:, 1]
  
        df_plot['Date'] = self.dataset['Дата']
        df_plot['Qн_hist'] = y_hist_inv_oil_list
        df_plot['Qн_train'] = y_recalc_inv_oil_list
        df_plot['Qн_test'] = y_pred_inv_oil_list
        df_plot['Qж_hist'] = y_hist_inv_liq_list
        df_plot['Qж_train'] = y_recalc_inv_liq_list
        df_plot['Qж_test'] = y_pred_inv_liq_list

        return df_plot        

    def predict_new(self, steps_ahead=1):
        print(1)
        initial_data = self.scaled_features[-self.look_back:, :]
        print(2)
        predicted_values = self.scaled_features[-1:, :]
        print(3)
        for i in range(steps_ahead):
            # Преобразование данных в формат, подходящий для входа в модель
            input_data = initial_data.reshape((1, self.look_back, initial_data.shape[1]))
            print(4)    
            # Предсказание следующего значения
            next_value = self.lstm_model.predict(input_data)
            print(5)
            # Добавление предсказанного значения в список
            # np.vstack((arr[:-1], new_row))
            predicted_values = np.concatenate((predicted_values, np.array([[next_value[0, 0], next_value[0, 1]]])))
            print(6)
            # Обновление initial_data для последующих предсказаний
            initial_data = np.concatenate((initial_data[1:], np.array([[next_value[0, 0], next_value[0, 1]]])))
            print(7)
        # Инверсия шкалы для получения оригинальных значений
        print(8)
        predicted_values_inv = self.scaler_target.inverse_transform(predicted_values[1:])
        print(9)
        return predicted_values_inv


        

    def plot(self):
        # train_time = np.arange(0, self.split_index, 1)
        # test_time = np.arange(self.split_index, len(self.dataset), 1)  

        # # Визуализация результатов для P1Q
        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 2, 1)

        # plt.plot(test_time, self.y_test_inv[:, 0],  c='g')
        # plt.plot(test_time, self.y_pred_inv[:, 0], label='Прогноз P1', c='r')
        # plt.plot(train_time, self.y_recalc_inv[:, 0], label='Обучение P1', c='b')
        # plt.plot(train_time, self.y_train_inv[:, 0], label='Историческая добыча P1',  c='g')
        # plt.title('Прогнозирование дебита скважины Р1')
        # plt.xlabel('Время, неделя', fontsize=18)
        # plt.ylabel('Дебит, м3/сут', fontsize=18)
        # plt.legend()


        # # Визуализация результатов для P2Q
        # plt.subplot(1, 2, 2)
        # plt.plot(test_time, self.y_test_inv[:, 1],  c='g')
        # plt.plot(test_time, self.y_pred_inv[:, 1], label='Прогноз P2', c='r')
        # plt.plot(train_time, self.y_recalc_inv[:, 1], label='Обучение P2', c='b')
        # plt.plot(train_time, self.y_train_inv[:, 1], label='Историческая добыча P2',  c='g')
        # plt.title('Прогнозирование дебита скважины Р2')
        # plt.xlabel('Время, неделя', fontsize=18)
        # plt.ylabel('Дебит, м3/сут', fontsize=18)
        # plt.legend()
        # Визуализация результатов
        plt.plot(self.dataset['Дата'][self.train_size + self.look_back:], self.y_test_inv[:, 0], label='Фактическое значение Qн')
        plt.plot(self.dataset['Дата'][self.train_size + self.look_back:], self.y_pred_inv[:, 0], label='Прогноз Qн')
        plt.plot(self.dataset['Дата'][self.look_back:self.train_size + self.look_back], self.y_train_inv[:, 0], label='Фактическое значение Qн')
        plt.plot(self.dataset['Дата'][self.look_back:self.train_size + self.look_back], self.y_recalc_inv[:, 0], label='Обучение')
        # plt.plot(self.dataset['Дата'][self.train_size + self.look_back:], self.y_test_inv[:, 1], label='Фактическое значение Qж')
        # plt.plot(self.dataset['Дата'][self.train_size + self.look_back:], self.y_pred_inv[:, 1], label='Прогноз Qж')
        # plt.xlabel('Дата')
        plt.ylabel('Значение')
        plt.legend()
        plt.show()


        plt.show()

    def _oops(self):
        return self.lstm_model

    def save_model(self, path_model:str, path_scaler:str):
        # Сохранение модели
        self.lstm_model.save(f'{path_model}\Well_{self.well_name}_model.h5')

        # Сохранение параметров масштабирования
        joblib.dump(self.scaled_features, f'{path_scaler}\Well_{self.well_name}_scaler.pkl')

        return self.lstm_model, self.scaled_features
    
# df = pd.read_excel('D:\LSTM\sb_model_25_11_2023\init_data.xlsx')
# well_model = well_LSTM(['HDM'], df, type='Field')
# well_model._prepare_data(train_size=1, lookback=45, lookforward=1)
# well_model.create_model(type=1)
# well_model.fit(EPOCHS=1, BATCH_SIZE=1)
# data_for_plot = well_model.predict_new()
# print(f'Q нефти: {data_for_plot[0, 0]}')
# print(f'Q жидкости: {data_for_plot[0, 1]}')
          