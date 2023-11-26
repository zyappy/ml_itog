import pandas as pd
import numpy as np

def prepare_data(file, type='Field'):
    '''
    Переменные, которые хранятся в результате вывода функции:

    init_dataset - Исходный датасет

    df_inj_full - Датасет с данными по ежемесячной закачке воды

    transformed_dataset - Сводная таблица

    transformed_dataset_2 - Отфильтрованная сводная таблица, где заменены пропуски

    final_dataset - Итоговый датасет с данными по закачке и фильтрацией

    original_wells_list - Список скважин, которые были в начальном датасете
    
    filtered_wells_list - Список скважин, которые остались после фильтрации

    '''
    result_data = {}

    if type == 'Field':
        init_dataset = pd.read_excel(file)
        # Список со всеми датафреймами и прочими списками и результатами 
        result_data['init_dataset'] = init_dataset

        # Создадим отдельный датафрейм с сумарной закачкой с суммарной закачкой
        df_inj = init_dataset[init_dataset['Тип'] == 'Нагнетательная'].groupby('Дата')['Qж, м3/сут'].sum().reset_index()

        # Создаем новый датафрейм с полным набором дат
        min_date = df_inj['Дата'].min()
        max_date = df_inj['Дата'].max()
        all_dates = pd.date_range(min_date, max_date, freq='MS')  # MS означает начало месяца (Month Start)
        full_date_df = pd.DataFrame({'Дата': all_dates})

        # Добавляем в Датафрейм с закачкой воды пропущенные даты
        df_inj_full = pd.merge(full_date_df, df_inj, on='Дата', how='left')
        # Заполним Nan значения по значения в предыдущих датах
        df_inj_full['Qж, м3/сут'] = df_inj_full['Qж, м3/сут'].fillna(method='ffill')
        result_data['df_inj_full'] = df_inj_full

        # Скопируем исходный датасет 
        # Удалим дипликаты по датам
        # Удалим из датасета нагнетательные скважины
        # Оставим скважины, которые работают на ЭЦН, либо имеет значение Nan
        df2 = init_dataset.copy()
        df2 = df2.drop_duplicates(subset=['Номер скважины', 'Дата'], keep='first')
        df2 = df2[df2['Тип'] != 'Нагнетательная']
        df2 = df2[(df2['Способ эксплуатации'] == 'ЭЦН')|(df2['Способ эксплуатации'] == np.nan)]

        # Создадим массив с именами скважин, чтобы использовать его в дальнейшем
        well_list = df2['Номер скважины'].unique()

        # Преобразование датасета в сводную таблицу
        # Изменение имени столбцов
        transformed_dataset = df2.pivot(index='Дата', 
                                columns='Номер скважины', 
                                values=['Qн, т/сут', 
                                        'Qж, м3/сут', 
                                        'Буферное давление, атм', 
                                        'Частота работы ЭЦН, Гц', 
                                        'Способ эксплуатации']).reset_index()
        transformed_dataset.columns = [f'Скважина_{col[1]}_{col[0]}' for col in transformed_dataset.columns]
        result_data['transformed_dataset'] = transformed_dataset

        # Скопируем сформированную сводную таблицу
        transformed_dataset_2 = transformed_dataset.copy()

        # Заменим пропуски в датасете
        for well_name in well_list:
            not_work = transformed_dataset_2[f'Скважина_{well_name}_Способ эксплуатации'].isna()
            transformed_dataset_2.loc[not_work, f'Скважина_{well_name}_Qн, т/сут'] = 0
            transformed_dataset_2.loc[not_work, f'Скважина_{well_name}_Qж, м3/сут'] = 0
            transformed_dataset_2.loc[not_work, f'Скважина_{well_name}_Частота работы ЭЦН, Гц'] = 0
            transformed_dataset_2.loc[not_work, f'Скважина_{well_name}_Буферное давление, атм'] = 0
            
            zero_debit_condition = (transformed_dataset_2[f'Скважина_{well_name}_Qн, т/сут'] == 0) & (transformed_dataset_2[f'Скважина_{well_name}_Частота работы ЭЦН, Гц'].isna())
            transformed_dataset_2.loc[zero_debit_condition, f'Скважина_{well_name}_Частота работы ЭЦН, Гц'] = 0
            transformed_dataset_2.loc[zero_debit_condition, f'Скважина_{well_name}_Буферное давление, атм'] = 0
        
            nonzero_debit_condition = (transformed_dataset_2[f'Скважина_{well_name}_Qн, т/сут'] != 0) & (transformed_dataset_2[f'Скважина_{well_name}_Частота работы ЭЦН, Гц'].isna())
            transformed_dataset_2.loc[nonzero_debit_condition, f'Скважина_{well_name}_Частота работы ЭЦН, Гц'] = int(transformed_dataset_2[f'Скважина_{well_name}_Частота работы ЭЦН, Гц'].max())
            transformed_dataset_2.loc[nonzero_debit_condition, f'Скважина_{well_name}_Буферное давление, атм'] = int(transformed_dataset_2[f'Скважина_{well_name}_Буферное давление, атм'].max())    
            
            nan_debit_condition = (transformed_dataset_2[f'Скважина_{well_name}_Qн, т/сут'].isna()) & ((transformed_dataset_2[f'Скважина_{well_name}_Частота работы ЭЦН, Гц'].isna())|(transformed_dataset_2[f'Скважина_{well_name}_Буферное давление, атм'].isna()))
            transformed_dataset_2.loc[nan_debit_condition, f'Скважина_{well_name}_Qн, т/сут'] = 0
            transformed_dataset_2.loc[nan_debit_condition, f'Скважина_{well_name}_Qж, м3/сут'] = 0
        
        # Удалим столбцы со способом эксплуатации, так как они больше не нужны
        filtered_columns = transformed_dataset_2.filter(like='_Способ эксплуатации').columns
        transformed_dataset_2 = transformed_dataset_2.drop(columns=filtered_columns)

        # Переименнуем столбец с датами
        transformed_dataset_2 = transformed_dataset_2.rename(columns={transformed_dataset_2.columns[0]: 'Дата'})

        # Добавим в датасет пропущенные даты
        transformed_dataset_2 = pd.merge(full_date_df, transformed_dataset_2, on='Дата', how='left')

        # Заменим пропуски предыдущим по дате значением
        transformed_dataset_2 = transformed_dataset_2.fillna(method='ffill')

        # Фильтруем датасет от скважин:
        # 1. Которые не работают последние x месяцев
        # 2. Удаляем скважины, общее время работы которых меньше y * размерность датасета
        y = 0.6
        stop_period = 3
        # Поиск и вывод имени столбца, удовлетворяющего условиям
        drop_wells = []
        for column in transformed_dataset_2.columns:
            if column.endswith("_Qн, т/сут") and ((transformed_dataset_2[column].tail(stop_period) == 0).all() or ((transformed_dataset_2[column] == 0).sum() > int(len(transformed_dataset_2) * (1 - y)))):
                drop_wells.append(column)

        drop_wells = [well.replace('_Qн, т/сут', '') for well in drop_wells]

        columns_to_drop = [col for col in transformed_dataset_2.columns if any(well in col for well in drop_wells)]
        transformed_dataset_2 = transformed_dataset_2.drop(columns=columns_to_drop)

        # Добавим полученный датасет в словарь
        result_data['transformed_dataset_2'] = transformed_dataset_2

        # Сформируем финальный датасет
        final_dataset = pd.merge(transformed_dataset_2, df_inj_full, on='Дата')
        final_dataset = final_dataset.rename(columns={final_dataset.columns[-1]: 'Закачка_Qж, м3/сут'})

        # Сформируем итоговый список скважин
        wells_name = []
        for col in final_dataset:
            if col.endswith("_Qн, т/сут"):
                wells_name.append(col)

        wells_name = [well.replace('_Qн, т/сут', '') for well in wells_name]

        # Выберем только те столбцы, где в названии есть 'Qн'
        selected_qoil = final_dataset.filter(like='_Qн, т/сут')
        # Суммируем значения по строкам
        sum_by_row_qoil = selected_qoil.sum(axis=1)
        # Добавляем результат в новый столбец 'Сумма_Qн' в DataFrame
        final_dataset['Суммарная Qн, т/сут'] = sum_by_row_qoil

        # Выберем только те столбцы, где в названии есть 'Qж'
        selected_qliq = final_dataset.filter(like='_Qж, м3/сут')
        del selected_qliq['Закачка_Qж, м3/сут']
        # Суммируем значения по строкам
        sum_by_row_qliq = selected_qliq.sum(axis=1)
        # Добавляем результат в новый столбец 'Сумма_Qн' в DataFrame
        final_dataset['Суммарная Qж, м3/сут'] = sum_by_row_qliq

        result_data['final_dataset'] = final_dataset

        # Добавим исходный список скважин и после фильтрации
        result_data['original_wells_list'] = well_list
        result_data['filtered_wells_list'] = wells_name
    
    elif type =='HDM':
        init_dataset = pd.read_excel(file)
        # Список со всеми датафреймами и прочими списками и результатами 
        result_data['init_dataset'] = init_dataset

    return result_data




