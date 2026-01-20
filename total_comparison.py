import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Список аттракционов для анализа
ATTRACTIONS = [
    'Астродром',
    'Аэротакси',
    'Вальс часов',
    'Лунный экспресс',
    'Торнадо',
    'Авиатор',
]

# Аттракционы для суммы (все кроме трамвая)
SUM_ATTRACTIONS = ATTRACTIONS.copy()

WINDOW_SIZE = 14
MIN_TOTAL_THRESHOLD = 500  # Минимальный порог для суммы

# Вертикальные линии-метки (день года)
VERTICAL_LINES = {
    'Аэротакси': [(195, '14 июл')],
    'Вальс часов': [(228, '16 авг')],
    'Астродром': [(212, '31 июл')],
}

# Загрузка данных
df = pd.read_csv('Все_с_канаткой_полные_данные.csv', encoding='utf-8')

# Удаляем колонки аномалий если есть
cols_to_drop = [col for col in df.columns if '_аномалия_%' in col]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)

df['Дата'] = pd.to_datetime(df['Дата'])
df['Год'] = df['Дата'].dt.year
df['ДеньГода'] = df['Дата'].dt.dayofyear

# Определяем тип дня
df['is_weekend_holiday'] = df['Тип дня'].isin(['Выходной', 'Праздник'])

# Вычисляем сумму по всем аттракционам (кроме трамвая)
df['Сумма_без_трамвая'] = df[SUM_ATTRACTIONS].sum(axis=1, skipna=True)

# Фильтруем: только дни где сумма >= порога
df_filtered = df[df['Сумма_без_трамвая'] >= MIN_TOTAL_THRESHOLD].copy()

total_days = len(df)
filtered_days = len(df_filtered)

print(f"Всего дней: {total_days}")
print(f"Дней с суммой (без трамвая) >= {MIN_TOTAL_THRESHOLD}: {filtered_days} (отсечено {total_days - filtered_days})")
print(f"  2024: {len(df_filtered[df_filtered['Год'] == 2024])}")
print(f"  2025: {len(df_filtered[df_filtered['Год'] == 2025])}")
print("=" * 60)


def normalize_to_total(df, attraction):
    """Нормализует аттракцион относительно суммы всех (без трамвая) (%)"""
    result = pd.Series(index=df.index, dtype=float)
    for idx in df.index:
        total_val = df.loc[idx, 'Сумма_без_трамвая']
        attr_val = df.loc[idx, attraction]
        if pd.notna(total_val) and total_val > 0 and pd.notna(attr_val):
            result.loc[idx] = (attr_val / total_val) * 100
        else:
            result.loc[idx] = np.nan
    return result


def calculate_rolling_median_by_year(df, column, year, is_weekend, window=WINDOW_SIZE):
    """Скользящая медиана для конкретного года и типа дня"""
    mask = (df['Год'] == year) & (df['is_weekend_holiday'] == is_weekend)
    data = df[mask].copy().sort_values('ДеньГода')

    if len(data) < 3:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    median = data[column].rolling(window=window, min_periods=3, center=True).median()

    return data['ДеньГода'], median


def plot_all_attractions_comparison(df):
    """Строит все аттракционы в одном файле (друг над другом)"""

    # Нормализуем все аттракционы
    for attraction in ATTRACTIONS:
        norm_col = f'{attraction}_norm'
        df[norm_col] = normalize_to_total(df, attraction)

    # Создаём фигуру: 6 аттракционов x 2 типа дня = 12 графиков
    fig, axes = plt.subplots(len(ATTRACTIONS), 2, figsize=(18, 4 * len(ATTRACTIONS)))

    for i, attraction in enumerate(ATTRACTIONS):
        norm_col = f'{attraction}_norm'

        for j, (is_weekend, day_type) in enumerate([(False, 'Рабочие'), (True, 'Выходные')]):
            ax = axes[i, j]

            # 2024 год - оранжевая линия
            days_2024, median_2024 = calculate_rolling_median_by_year(df, norm_col, 2024, is_weekend)
            if len(days_2024) > 0:
                ax.plot(days_2024.values, median_2024.values,
                       color='orange', linewidth=2.5, label='2024', zorder=3)

            # 2025 год - синяя линия
            days_2025, median_2025 = calculate_rolling_median_by_year(df, norm_col, 2025, is_weekend)
            if len(days_2025) > 0:
                ax.plot(days_2025.values, median_2025.values,
                       color='blue', linewidth=2.5, label='2025', zorder=4)

            # Автоматический диапазон Y по данным
            all_values = pd.concat([median_2024, median_2025]).dropna()
            if len(all_values) > 0:
                data_min = all_values.min()
                data_max = all_values.max()
                margin = (data_max - data_min) * 0.1 if data_max > data_min else 1
                y_min_auto = max(0, data_min - margin)
                y_max_auto = data_max + margin
            else:
                y_min_auto, y_max_auto = 0, 100

            # Настройки
            ax.set_ylim(y_min_auto, y_max_auto)
            ax.set_xlim(90, 300)
            ax.set_ylabel('% от суммы', fontsize=9)
            ax.set_title(f'{attraction} — {day_type}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)

            # Месяцы на оси X
            month_ticks = [
                (91, 'Апр'), (121, 'Май'), (152, 'Июн'),
                (182, 'Июл'), (213, 'Авг'), (244, 'Сен'), (274, 'Окт')
            ]
            ax.set_xticks([t[0] for t in month_ticks])
            ax.set_xticklabels([t[1] for t in month_ticks], fontsize=8)

            # Вертикальные линии-метки для конкретных аттракционов
            if attraction in VERTICAL_LINES:
                for day_of_year, label in VERTICAL_LINES[attraction]:
                    ax.axvline(x=day_of_year, color='red', linewidth=2, linestyle='--', alpha=0.8)
                    ax.text(day_of_year + 2, y_max_auto * 0.9, label, fontsize=8, color='red', fontweight='bold')

    plt.tight_layout()
    filename = 'all_attractions_vs_total_comparison.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nФайл сохранён: {filename}")

    # Статистика
    print(f"\nСтатистика по годам (% от суммы всех без трамвая):")
    print("-" * 50)
    for attraction in ATTRACTIONS:
        norm_col = f'{attraction}_norm'
        print(f"\n{attraction}:")
        for year in [2024, 2025]:
            year_data = df[(df['Год'] == year) & df[norm_col].notna()]
            work_med = year_data[~year_data['is_weekend_holiday']][norm_col].median()
            week_med = year_data[year_data['is_weekend_holiday']][norm_col].median()
            work_str = f"{work_med:.1f}%" if pd.notna(work_med) else "—"
            week_str = f"{week_med:.1f}%" if pd.notna(week_med) else "—"
            print(f"  {year}: раб {work_str} | вых {week_str}")

    return df


# Обрабатываем все аттракционы в одном файле
print(f"\nСравнение 2024 vs 2025 относительно суммы всех (без трамвая)")
print(f"Медиана {WINDOW_SIZE} дней")
print("=" * 60)

df_filtered = plot_all_attractions_comparison(df_filtered)

print("\n" + "=" * 60)
print("Готово!")
