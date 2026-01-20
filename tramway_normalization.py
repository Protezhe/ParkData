import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Список аттракционов (без Воздушного трамвая - он базовый)
ATTRACTIONS = [
    'Астродром',
    'Аэротакси',
    'Вальс часов',
    'Лунный экспресс',
    'Торнадо',
    'Авиатор',
]

BASE_ATTRACTION = 'Воздушный трамвай'
WINDOW_SIZE = 14  # Окно для скользящей медианы (14 дней - стабильнее)
MIN_TRAMWAY_THRESHOLD = 1000  # Минимальный порог для трамвая (отсекаем плохие дни)

# Лимиты осей Y для разных типов аттракционов
Y_LIMITS = {
    'Астродром': (0, 40),
    'Аэротакси': (0, 40),
    'Вальс часов': (0, 40),
    'Лунный экспресс': (0, 100),
    'Торнадо': (0, 40),
    'Авиатор': (0, 40),
}

# Загрузка данных
df = pd.read_csv('Все_с_канаткой_полные_данные.csv', encoding='utf-8')

# Удаляем колонки аномалий если есть
cols_to_drop = [col for col in df.columns if '_аномалия_%' in col]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)

df['Дата'] = pd.to_datetime(df['Дата'])

# Определяем тип дня
df['is_weekend_holiday'] = df['Тип дня'].isin(['Выходной', 'Праздник'])

# Фильтруем: только дни где трамвай >= порога (отсекаем "плохой якорь")
df_with_tramway = df[df[BASE_ATTRACTION].notna() & (df[BASE_ATTRACTION] >= MIN_TRAMWAY_THRESHOLD)].copy()

total_days = len(df[df[BASE_ATTRACTION].notna()])
filtered_days = len(df_with_tramway)

print(f"Всего дней с данными по {BASE_ATTRACTION}: {total_days}")
print(f"Дней с трамваем >= {MIN_TRAMWAY_THRESHOLD}: {filtered_days} (отсечено {total_days - filtered_days})")
print("=" * 60)


def normalize_to_tramway(df, attraction):
    """Нормализует аттракцион относительно Воздушного трамвая (%)"""
    result = pd.Series(index=df.index, dtype=float)

    for idx in df.index:
        tramway_val = df.loc[idx, BASE_ATTRACTION]
        attr_val = df.loc[idx, attraction]

        if pd.notna(tramway_val) and tramway_val > 0 and pd.notna(attr_val):
            result.loc[idx] = (attr_val / tramway_val) * 100
        else:
            result.loc[idx] = np.nan

    return result


def calculate_rolling_median(df, column, window=WINDOW_SIZE):
    """Скользящая медиана отдельно для рабочих и выходных дней"""
    result = pd.Series(index=df.index, dtype=float)

    # Рабочие дни
    workdays = df[~df['is_weekend_holiday']].copy()
    if len(workdays) > 0:
        workdays_median = workdays[column].rolling(window=window, min_periods=3, center=True).median()
        for idx in workdays.index:
            result.loc[idx] = workdays_median.loc[idx]

    # Выходные
    weekends = df[df['is_weekend_holiday']].copy()
    if len(weekends) > 0:
        weekends_median = weekends[column].rolling(window=window, min_periods=3, center=True).median()
        for idx in weekends.index:
            result.loc[idx] = weekends_median.loc[idx]

    return result


def plot_attraction_vs_tramway(df, attraction):
    """Строит ЧИСТЫЙ график: точки + медиана. Без дождя, без лишнего."""

    # Нормализуем
    norm_col = f'{attraction}_norm'
    df[norm_col] = normalize_to_tramway(df, attraction)

    # Скользящая медиана
    median_col = f'{attraction}_median'
    df[median_col] = calculate_rolling_median(df, norm_col)

    # Фильтруем данные с значениями
    df_valid = df[df[norm_col].notna()].reset_index(drop=True)

    if len(df_valid) == 0:
        print(f"Нет данных для {attraction}")
        return df

    # Разделяем на рабочие и выходные
    df_workdays = df_valid[~df_valid['is_weekend_holiday']].reset_index(drop=True)
    df_weekends = df_valid[df_valid['is_weekend_holiday']].reset_index(drop=True)

    # Лимиты оси Y
    y_min, y_max = Y_LIMITS.get(attraction, (0, 50))

    # Создаём фигуру с двумя графиками
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    def plot_single(ax, data, title, color):
        if len(data) == 0:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', fontsize=14)
            ax.set_title(title)
            return

        # Точки (факт)
        ax.scatter(data.index, data[norm_col], color=color, s=25, alpha=0.5, zorder=2)

        # Скользящая медиана (толстая линия)
        ax.plot(data.index, data[median_col], color='darkred', linewidth=3, zorder=4)

        # Настройки осей
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('', fontsize=10)
        ax.set_ylabel(f'% от трамвая', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')

        ax.grid(True, alpha=0.3)

        # Аннотации для выбросов (> y_max)
        outliers = data[data[norm_col] > y_max]
        for _, row in outliers.iterrows():
            ax.annotate(f'{row[norm_col]:.0f}%',
                       xy=(row.name, y_max * 0.95),
                       fontsize=8, color='red', alpha=0.7)

        # Вертикальные линии разделения годов
        years = data['Дата'].dt.year
        for i in range(1, len(data)):
            if years.iloc[i] != years.iloc[i-1]:
                ax.axvline(x=i, color='black', linewidth=1.5, linestyle='-', alpha=0.5)
                ax.text(i + 1, y_max * 0.95, f'{years.iloc[i]}', fontsize=9, fontweight='bold')

        # Форматирование оси X
        tick_step = max(1, len(data) // 12)
        tick_positions = range(0, len(data), tick_step)
        tick_labels = [data.loc[i, 'Дата'].strftime('%d.%m') for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)

    # Графики
    plot_single(ax1, df_workdays, f'{attraction} — Рабочие дни', 'blue')
    plot_single(ax2, df_weekends, f'{attraction} — Выходные и праздники', 'red')

    # Общая легенда
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, alpha=0.5, label='Факт'),
        Line2D([0], [0], color='darkred', linewidth=3, label=f'Медиана {WINDOW_SIZE} дн.'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99), fontsize=10)

    plt.tight_layout()

    safe_name = attraction.replace(' ', '_')
    filename = f'{safe_name}_vs_tramway.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Статистика
    work_median = df_workdays[norm_col].median() if len(df_workdays) > 0 else np.nan
    weekend_median = df_weekends[norm_col].median() if len(df_weekends) > 0 else np.nan

    print(f"\n{attraction}:")
    print(f"  Рабочие: {work_median:.1f}%  |  Выходные: {weekend_median:.1f}%")
    print(f"  → {filename}")

    return df


# Обрабатываем все аттракционы
print(f"\nНормализация относительно {BASE_ATTRACTION}")
print(f"Медиана {WINDOW_SIZE} дней, порог трамвая >= {MIN_TRAMWAY_THRESHOLD}")
print("=" * 60)

for attraction in ATTRACTIONS:
    if attraction in df_with_tramway.columns:
        df_with_tramway = plot_attraction_vs_tramway(df_with_tramway, attraction)

print("\n" + "=" * 60)
print("Готово!")
