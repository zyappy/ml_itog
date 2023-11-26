import plotly.graph_objects as go
import pandas as pd
import streamlit as st

def plot_all(df):
  
        # Добавим условие проверки наличия данных
    if df.empty:
        st.warning("Данные отсутствуют. Загрузите данные для построения графика.")
        return

    def plot_interactive_graph(df):
        fig = go.Figure()

        # df.replace(-999, pd.NA, inplace=True)

        # Добавляем слои данных
        for well_name in ['Qн', 'Qж']:
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df[f'{well_name}_hist'],
                mode='lines+markers',
                name=f'{well_name}_hist',
                visible='legendonly'
            ))

            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df[f'{well_name}_train'],
                mode='lines+markers',
                name=f'{well_name}_train',
                visible='legendonly'
            ))

            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df[f'{well_name}_test'],
                mode='lines+markers',
                name=f'{well_name}_test',
                visible='legendonly'
            ))

        # Настраиваем выпадающий список для выбора скважины
        well_dropdown = [{'label': well_name, 
                            'method': 'update', 
                            'args': [{'visible': [well_name in trace.name for trace in fig.data]},
                            {'title': f'График для {well_name}'}]}
                            for well_name in ['Qн', 'Qж']]

        fig.update_layout(
            updatemenus=[
                dict(
                    type='dropdown',
                    direction='down',
                    active=0,
                    x=1.05,
                    y=0.8,
                    buttons=well_dropdown
                ),
            ]
        )

        # Настраиваем остальные параметры макета
        fig.update_layout(
            title='График для',
            xaxis_title='Дата',
            yaxis_title='Q'
        )

        #fig.show()
        st.plotly_chart(fig)

    # Тестирование функции
    plot_interactive_graph(df)