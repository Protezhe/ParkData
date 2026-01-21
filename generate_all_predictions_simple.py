"""
Скрипт для генерации графиков предсказаний для всех аттракционов.
Использует простую модель без прокси (только погода и тип дня).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

WINDOW_SIZE = 14

# Список аттракционов для анализа
ATTRACTIONS = [
    'Астродром',
    'Аэротакси',
    'Вальс часов',
    'Лунный экспресс',
    'Торнадо',
    'Авиатор',
    'Воздушный трамвай'
]


def load_and_prepare_data():
    """Загрузка и подготовка данных"""
    df = pd.read_csv('Все_с_канаткой_полные_данные.csv', encoding='utf-8')
    df['Дата'] = pd.to_datetime(df['Дата'])
    df['Год'] = df['Дата'].dt.year
    df['ДеньГода'] = df['Дата'].dt.dayofyear
    df['Месяц'] = df['Дата'].dt.month
    df['ДеньНедели'] = df['Дата'].dt.dayofweek

    # Тип дня
    df['is_weekend'] = df['Тип дня'].isin(['Выходной', 'Праздник']).astype(int)
    df['is_holiday'] = (df['Тип дня'] == 'Праздник').astype(int)

    # Погода
    df['is_rain'] = df['Тип погоды'].isin(['Дождь', 'Небольшой дождь']).astype(int)
    df['is_heavy_rain'] = (df['Тип погоды'] == 'Дождь').astype(int)

    # Температура
    df['Темп_диапазон'] = df['Температура макс (°C)'] - df['Температура мин (°C)']
    df['Темп_комфорт'] = 1 - np.abs(df['Температура средняя (°C)'] - 20) / 20
    df['Темп_комфорт'] = df['Темп_комфорт'].clip(0, 1)

    # Дни недели
    df['День_1'] = (df['ДеньНедели'] == 1).astype(int)
    df['День_5'] = (df['ДеньНедели'] == 5).astype(int)

    return df


def calculate_rolling_median(data, column, is_weekend, window=WINDOW_SIZE):
    """Скользящая медиана для типа дня"""
    mask = data['is_weekend'] == is_weekend
    subset = data[mask].copy().sort_values('ДеньГода')

    if len(subset) < 3:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    median = subset[column].rolling(window=window, min_periods=3, center=True).median()
    return subset['ДеньГода'], median


def train_and_predict(df, attraction_name):
    """Обучение модели и предсказание для конкретного аттракциона"""

    # Фильтруем данные
    df_filtered = df[df[attraction_name].notna() & (df[attraction_name] > 0)].copy()

    if len(df_filtered) == 0:
        return None, None, None, None, None

    # Оптимальный набор признаков (без прокси)
    feature_cols = [
        'is_weekend',
        'is_rain',
        'Темп_комфорт',
        'Темп_диапазон',
        'is_heavy_rain',
        'День_5',
        'День_1',
        'is_holiday',
    ]

    # Разделение на обучающую и тестовую выборки
    train_df = df_filtered[df_filtered['Год'] == 2024].copy()
    test_df = df_filtered[df_filtered['Год'] == 2025].copy()

    if len(train_df) < 10 or len(test_df) < 10:
        return None, None, None, None, None

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[attraction_name]
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df[attraction_name]

    # Обучение модели
    model = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=4,
        min_samples_split=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Предсказание
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)

    # Метрики
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    test_df['Предсказание'] = y_pred

    return train_df, test_df, mae, r2, model


def plot_prediction(attraction_name, train_df, test_df, mae, r2):
    """Построение графика для одного аттракциона"""

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    day_types = [(0, 'Рабочие дни'), (1, 'Выходные и праздники')]

    for j, (is_weekend, day_type_name) in enumerate(day_types):
        # Верхний ряд: сравнение предсказаний и реальности для 2025
        ax_top = axes[0, j]

        # Реальные данные 2025
        days_real, median_real = calculate_rolling_median(test_df, attraction_name, is_weekend)
        if len(days_real) > 0:
            ax_top.plot(days_real.values, median_real.values,
                       color='blue', linewidth=2, marker='o', markersize=4,
                       markerfacecolor='cyan', markeredgecolor='blue',
                       label='Реальные данные 2025', zorder=4)

        # Предсказанные данные 2025
        days_pred, median_pred = calculate_rolling_median(test_df, 'Предсказание', is_weekend)
        if len(days_pred) > 0:
            ax_top.plot(days_pred.values, median_pred.values,
                       color='red', linewidth=2, marker='s', markersize=4,
                       markerfacecolor='pink', markeredgecolor='red',
                       linestyle='--', label='Предсказание модели', zorder=3)

        # Температура на второй оси Y
        temp_data = test_df[test_df['is_weekend'] == is_weekend].sort_values('ДеньГода')
        if len(temp_data) > 0:
            ax_temp = ax_top.twinx()
            temp_median = temp_data['Температура средняя (°C)'].rolling(
                window=WINDOW_SIZE, min_periods=3, center=True).median()
            ax_temp.plot(temp_data['ДеньГода'].values, temp_median.values,
                        color='purple', linewidth=1.5, linestyle=':',
                        alpha=0.7, label='Температура')
            ax_temp.set_ylabel('Средняя температура (C)', color='purple', fontsize=10)
            ax_temp.tick_params(axis='y', labelcolor='purple')

        # Закрашиваем дождливые дни
        rain_days = test_df[(test_df['is_weekend'] == is_weekend) &
                            (test_df['is_rain'] == 1)]['ДеньГода'].values
        for day in rain_days:
            ax_top.axvspan(day - 0.5, day + 0.5, color='lightblue', alpha=0.4, zorder=1)

        ax_top.set_ylabel('Количество проданных билетов', fontsize=10)
        ax_top.set_title(f'{attraction_name} - {day_type_name}', fontsize=12, fontweight='bold')
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(loc='upper left', fontsize=9)
        ax_top.set_xlim(90, 290)

        # Месяцы на оси X
        month_ticks = [
            (91, 'Апр'), (121, 'Май'), (152, 'Июн'),
            (182, 'Июл'), (213, 'Авг'), (244, 'Сен'), (274, 'Окт')
        ]
        ax_top.set_xticks([t[0] for t in month_ticks])
        ax_top.set_xticklabels([t[1] for t in month_ticks], fontsize=9)

        # Нижний ряд: данные 2024
        ax_bottom = axes[1, j]

        # Реальные данные 2024
        days_2024, median_2024 = calculate_rolling_median(train_df, attraction_name, is_weekend)
        if len(days_2024) > 0:
            ax_bottom.plot(days_2024.values, median_2024.values,
                          color='blue', linewidth=2, marker='o', markersize=4,
                          markerfacecolor='cyan', markeredgecolor='blue',
                          label='Реальные данные 2024 (обучение)', zorder=4)

        # Температура 2024
        temp_2024 = train_df[train_df['is_weekend'] == is_weekend].sort_values('ДеньГода')
        if len(temp_2024) > 0:
            ax_temp2 = ax_bottom.twinx()
            temp_median_2024 = temp_2024['Температура средняя (°C)'].rolling(
                window=WINDOW_SIZE, min_periods=3, center=True).median()
            ax_temp2.plot(temp_2024['ДеньГода'].values, temp_median_2024.values,
                         color='purple', linewidth=1.5, linestyle=':',
                         alpha=0.7, label='Температура')
            ax_temp2.set_ylabel('Средняя температура (C)', color='purple', fontsize=10)
            ax_temp2.tick_params(axis='y', labelcolor='purple')

        # Закрашиваем дождливые дни 2024
        rain_2024 = train_df[(train_df['is_weekend'] == is_weekend) &
                             (train_df['is_rain'] == 1)]['ДеньГода'].values
        for day in rain_2024:
            ax_bottom.axvspan(day - 0.5, day + 0.5, color='lightblue', alpha=0.4, zorder=1)

        ax_bottom.set_xlabel('Дата', fontsize=10)
        ax_bottom.set_ylabel('Количество проданных билетов', fontsize=10)
        ax_bottom.set_title(f'{attraction_name} 2024 (обучающие данные) - {day_type_name}',
                           fontsize=12, fontweight='bold')
        ax_bottom.grid(True, alpha=0.3)
        ax_bottom.legend(loc='upper left', fontsize=9)
        ax_bottom.set_xlim(90, 290)
        ax_bottom.set_xticks([t[0] for t in month_ticks])
        ax_bottom.set_xticklabels([t[1] for t in month_ticks], fontsize=9)

    # Общая легенда
    fig.text(0.5, 0.02,
             'Голубая заливка = дождливые дни | Пунктирная фиолетовая = температура',
             ha='center', fontsize=10, style='italic')

    plt.suptitle(f'Предсказание посещаемости {attraction_name}: модель vs реальность\n'
                 f'MAE = {mae:.1f} билетов, R2 = {r2:.3f}',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = f'{attraction_name.lower().replace(" ", "_")}_prediction_comparison.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    return filename


def main():
    print("=" * 60)
    print("Генерация графиков предсказаний для всех аттракционов")
    print("(простая модель без прокси)")
    print("=" * 60)

    # Загрузка данных
    print("\nЗагрузка данных...")
    df = load_and_prepare_data()

    results = []

    for attraction in ATTRACTIONS:
        print(f"\n{'-' * 60}")
        print(f"Обработка: {attraction}")

        train_df, test_df, mae, r2, model = train_and_predict(df, attraction)

        if train_df is None:
            print(f"  Недостаточно данных для {attraction}")
            continue

        print(f"  Обучающая выборка (2024): {len(train_df)} дней")
        print(f"  Тестовая выборка (2025): {len(test_df)} дней")
        print(f"  MAE: {mae:.1f} билетов")
        print(f"  R2: {r2:.3f}")

        # Построение графика
        filename = plot_prediction(attraction, train_df, test_df, mae, r2)
        print(f"  График сохранен: {filename}")

        results.append({
            'Аттракцион': attraction,
            'MAE': mae,
            'R2': r2,
            'Файл': filename
        })

    # Итоговая таблица
    print("\n" + "=" * 60)
    print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 60)
    print(f"{'Аттракцион':<20} {'MAE':<12} {'R2':<10}")
    print("-" * 60)

    for r in sorted(results, key=lambda x: x['MAE']):
        print(f"{r['Аттракцион']:<20} {r['MAE']:<12.1f} {r['R2']:<10.3f}")

    print("\n" + "=" * 60)
    print(f"Всего обработано аттракционов: {len(results)}")
    print("Готово!")


if __name__ == '__main__':
    main()
