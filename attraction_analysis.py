import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys

# Список всех аттракционов
ATTRACTIONS = [
    'Астродром',
    'Аэротакси',
    'Вальс часов',
    'Лунный экспресс',
    'Торнадо',
    'Авиатор',
    'Воздушный трамвай'
]

def analyze_attraction(attraction_name):
    """Анализ и построение графиков для указанного аттракциона"""

    # Загрузка данных
    df = pd.read_csv('Все_с_канаткой_полные_данные.csv', encoding='utf-8')
    df['Дата'] = pd.to_datetime(df['Дата'])

    # Проверяем наличие колонки
    if attraction_name not in df.columns:
        print(f"Ошибка: аттракцион '{attraction_name}' не найден в данных")
        print(f"Доступные аттракционы: {ATTRACTIONS}")
        return

    # Фильтруем только строки где есть данные по аттракциону
    df_filtered = df[df[attraction_name].notna() & (df[attraction_name] > 0)].copy()

    if len(df_filtered) == 0:
        print(f"Нет данных для аттракциона '{attraction_name}'")
        return

    # Определяем тип дня
    df_filtered['is_weekend_holiday'] = df_filtered['Тип дня'].isin(['Выходной', 'Праздник'])

    # Определяем наличие дождя
    df_filtered['is_rain'] = df_filtered['Тип погоды'].str.contains('дождь|Дождь', case=False, na=False)

    # Разделяем на рабочие и выходные дни
    df_workdays = df_filtered[~df_filtered['is_weekend_holiday']].reset_index(drop=True)
    df_weekends = df_filtered[df_filtered['is_weekend_holiday']].reset_index(drop=True)

    # Создаём фигуру с двумя графиками
    fig, (ax1_work, ax1_weekend) = plt.subplots(2, 1, figsize=(16, 12))

    def plot_graph(ax1, data, title, color, attr_name):
        """Функция для построения графика"""
        if len(data) == 0:
            ax1.text(0.5, 0.5, 'Нет данных', ha='center', va='center', fontsize=14)
            ax1.set_title(title, fontsize=14, fontweight='bold')
            return

        ax2 = ax1.twinx()

        # Маска для дождливых дней
        rain_mask = data['is_rain']

        # Линия билетов
        ax1.plot(data.index, data[attr_name],
                 color=color, linewidth=2, alpha=0.8, zorder=1)

        # Точки для дождливых дней
        if rain_mask.sum() > 0:
            ax1.scatter(data.index[rain_mask], data.loc[rain_mask, attr_name],
                        color='cyan', s=60, marker='o', zorder=3, edgecolors='black', linewidth=1)

        # Линия температуры
        ax2.plot(data.index, data['Температура средняя (°C)'],
                 color='purple', linewidth=2, alpha=0.7, linestyle='--', zorder=2)

        # Настройка осей
        ax1.set_xlabel('Дата', fontsize=12)
        ax1.set_ylabel('Количество проданных билетов', fontsize=12, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2.set_ylabel('Средняя температура (°C)', fontsize=12, color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')

        ax1.set_title(title, fontsize=14, fontweight='bold')

        # Легенда
        legend_elements = [
            Line2D([0], [0], color=color, linewidth=2, label='Билеты'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markeredgecolor='black', markersize=8, label='Дождь'),
            Line2D([0], [0], color='purple', linewidth=2, linestyle='--', label='Температура'),
        ]
        ax1.legend(handles=legend_elements, loc='upper left')

        ax1.grid(True, alpha=0.3)

        # Вертикальные линии разделения годов
        years = data['Дата'].dt.year
        for i in range(1, len(data)):
            if years.iloc[i] != years.iloc[i-1]:
                ax1.axvline(x=i, color='black', linewidth=2, linestyle='-', alpha=0.7)
                ax1.text(i, ax1.get_ylim()[1], f'{years.iloc[i]}', ha='left', va='bottom', fontsize=10, fontweight='bold')

        # Форматирование оси X
        tick_step = max(1, len(data) // 15)
        tick_positions = range(0, len(data), tick_step)
        tick_labels = [data.loc[i, 'Дата'].strftime('%d.%m.%y') for i in tick_positions]
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels(tick_labels, rotation=45, ha='right')

    # Строим график для рабочих дней
    plot_graph(ax1_work, df_workdays, f'{attraction_name} - Рабочие дни', 'blue', attraction_name)

    # Строим график для выходных
    plot_graph(ax1_weekend, df_weekends, f'{attraction_name} - Выходные и праздники', 'red', attraction_name)

    plt.tight_layout()

    # Сохраняем с именем аттракциона
    safe_name = attraction_name.replace(' ', '_')
    plt.savefig(f'{safe_name}_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()  # Закрываем фигуру вместо show() для batch-режима

    # Считаем статистику
    holidays_count = len(df_filtered[df_filtered['Тип дня'] == 'Праздник'])
    weekends_only = len(df_filtered[df_filtered['Тип дня'] == 'Выходной'])

    print(f"\n=== {attraction_name} ===")
    print(f"Всего дней с данными: {len(df_filtered)}")
    print(f"Рабочих дней: {len(df_workdays)} (из них с дождём: {df_workdays['is_rain'].sum()})")
    print(f"Выходных: {weekends_only}")
    print(f"Праздников: {holidays_count}")
    print(f"Выходных + праздников: {len(df_weekends)} (из них с дождём: {df_weekends['is_rain'].sum()})")
    print(f"Файл сохранён: {safe_name}_analysis.png")


def analyze_all():
    """Анализ всех аттракционов"""
    for attraction in ATTRACTIONS:
        print(f"\nОбработка: {attraction}...")
        analyze_attraction(attraction)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--all':
            analyze_all()
        else:
            # Аргумент - название аттракциона
            attraction = ' '.join(sys.argv[1:])
            analyze_attraction(attraction)
    else:
        print("Использование:")
        print("  python attraction_analysis.py <название аттракциона>")
        print("  python attraction_analysis.py --all")
        print(f"\nДоступные аттракционы: {', '.join(ATTRACTIONS)}")
