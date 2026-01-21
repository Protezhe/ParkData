"""
Прогноз для Вальса часов с ростом Аэротакси относительно прогноза модели.

Логика:
1. Строим прогноз Аэротакси с помощью модели (как в aerotaxi_model_with_tramvay.py)
2. Сравниваем реальные данные Аэротакси 2025 с прогнозом
3. Коэффициент роста = реальные данные / прогноз
4. Применяем этот коэффициент к реальным данным Вальса часов 2024
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

PROXY_ATTRACTION = 'Воздушный трамвай'
WINDOW_SIZE = 14


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


def train_prediction_model(df, attraction_name):
    """
    Обучает модель для предсказания посещаемости аттракциона.
    Возвращает предсказания для 2025 года.
    """
    # Фильтруем данные
    df_filtered = df[df[attraction_name].notna() & (df[attraction_name] > 0) &
                     df[PROXY_ATTRACTION].notna() & (df[PROXY_ATTRACTION] > 0)].copy()

    feature_cols = [
        PROXY_ATTRACTION,
        'is_weekend',
        'is_rain',
        'is_heavy_rain',
        'Темп_комфорт',
        'Темп_диапазон',
        'is_holiday',
        'День_1',
        'День_5',
    ]

    # Разделение на обучающую и тестовую выборки
    train_df = df_filtered[df_filtered['Год'] == 2024].copy()
    test_df = df_filtered[df_filtered['Год'] == 2025].copy()

    if len(train_df) < 10 or len(test_df) < 10:
        return None, None

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[attraction_name]
    X_test = test_df[feature_cols].fillna(0)

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

    test_df['Прогноз_модели'] = y_pred

    return train_df, test_df


def calculate_growth_coefficients(df_predicted, attraction_name):
    """
    Рассчитывает коэффициенты роста: реальные данные / прогноз модели
    Возвращает словарь {(месяц, is_weekend): коэффициент}
    """
    growth_coefficients = {}

    for month in range(4, 11):
        for is_weekend in [0, 1]:
            month_data = df_predicted[
                (df_predicted['Месяц'] == month) &
                (df_predicted['is_weekend'] == is_weekend)
            ]

            if len(month_data) > 0:
                real_median = month_data[attraction_name].median()
                pred_median = month_data['Прогноз_модели'].median()

                if pred_median > 0:
                    growth_coefficients[(month, is_weekend)] = real_median / pred_median
                else:
                    growth_coefficients[(month, is_weekend)] = 1.0
            else:
                growth_coefficients[(month, is_weekend)] = 1.0

    return growth_coefficients


def apply_growth_to_valts(df, base_attraction, growth_coefficients):
    """
    Применяет коэффициенты роста к реальным данным Вальса часов 2024
    """
    # Реальные данные Вальса часов за 2024
    df_2024 = df[(df['Год'] == 2024) &
                 (df[base_attraction].notna()) &
                 (df[base_attraction] > 0)].copy()

    # Применяем коэффициенты к каждой строке
    df_2024['С_ростом_Аэротакси'] = df_2024.apply(
        lambda row: row[base_attraction] * growth_coefficients.get((row['Месяц'], row['is_weekend']), 1.0),
        axis=1
    )

    return df_2024


def calculate_rolling_median(data, column, is_weekend, window=WINDOW_SIZE):
    """Скользящая медиана для типа дня"""
    mask = data['is_weekend'] == is_weekend
    subset = data[mask].copy().sort_values('ДеньГода')

    if len(subset) < 3:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    median = subset[column].rolling(window=window, min_periods=3, center=True).median()
    return subset['ДеньГода'], median


def plot_comparison(df, base_attraction, reference_attraction, growth_coefficients):
    """Построение сравнительного графика"""

    # Получаем данные с ростом
    df_valts_with_growth = apply_growth_to_valts(df, base_attraction, growth_coefficients)

    # Реальные данные Вальса часов за 2024
    df_valts_2024 = df[(df['Год'] == 2024) &
                       (df[base_attraction].notna()) &
                       (df[base_attraction] > 0)].copy()

    # Данные Аэротакси для справки
    df_aerotaxi_2024 = df[(df['Год'] == 2024) &
                          (df[reference_attraction].notna()) &
                          (df[reference_attraction] > 0)].copy()

    df_aerotaxi_2025 = df[(df['Год'] == 2025) &
                          (df[reference_attraction].notna()) &
                          (df[reference_attraction] > 0)].copy()

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    day_types = [(0, 'Рабочие дни'), (1, 'Выходные и праздники')]
    month_names = {4: 'Апр', 5: 'Май', 6: 'Июн', 7: 'Июл',
                   8: 'Авг', 9: 'Сен', 10: 'Окт'}

    for j, (is_weekend, day_type_name) in enumerate(day_types):
        # Верхний ряд: Вальс часов с ростом vs без роста
        ax_top = axes[0, j]

        # Вальс часов 2024 (реальные данные)
        days_real, median_real = calculate_rolling_median(
            df_valts_2024, base_attraction, is_weekend)
        if len(days_real) > 0:
            ax_top.plot(days_real.values, median_real.values,
                       color='blue', linewidth=2, marker='o', markersize=4,
                       markerfacecolor='cyan', markeredgecolor='blue',
                       label=f'{base_attraction} 2024 (реальные данные)', zorder=4)

        # Вальс часов с ростом Аэротакси
        days_growth, median_growth = calculate_rolling_median(
            df_valts_with_growth, 'С_ростом_Аэротакси', is_weekend)
        if len(days_growth) > 0:
            ax_top.plot(days_growth.values, median_growth.values,
                       color='green', linewidth=2.5, marker='D', markersize=5,
                       markerfacecolor='lightgreen', markeredgecolor='darkgreen',
                       label=f'{base_attraction} с ростом {reference_attraction}', zorder=5)

        # Температура
        temp_data = df_valts_2024[df_valts_2024['is_weekend'] == is_weekend].sort_values('ДеньГода')
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
        rain_days = df_valts_2024[(df_valts_2024['is_weekend'] == is_weekend) &
                                  (df_valts_2024['is_rain'] == 1)]['ДеньГода'].values
        for day in rain_days:
            ax_top.axvspan(day - 0.5, day + 0.5, color='lightblue', alpha=0.4, zorder=1)

        # Показываем помесячные коэффициенты
        growth_text = "Коэффициенты: "
        for month in range(4, 11):
            rate = growth_coefficients.get((month, is_weekend), 1.0)
            growth_text += f"{month_names[month]}={rate:.2f} "

        ax_top.set_ylabel('Количество проданных билетов', fontsize=10)
        ax_top.set_title(
            f'{base_attraction} - {day_type_name}\n{growth_text}',
            fontsize=11, fontweight='bold')
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(loc='upper left', fontsize=9)
        ax_top.set_xlim(90, 290)

        month_ticks = [
            (91, 'Апр'), (121, 'Май'), (152, 'Июн'),
            (182, 'Июл'), (213, 'Авг'), (244, 'Сен'), (274, 'Окт')
        ]
        ax_top.set_xticks([t[0] for t in month_ticks])
        ax_top.set_xticklabels([t[1] for t in month_ticks], fontsize=9)

        # Нижний ряд: Аэротакси для справки
        ax_bottom = axes[1, j]

        # Аэротакси 2024
        days_aero_2024, median_aero_2024 = calculate_rolling_median(
            df_aerotaxi_2024, reference_attraction, is_weekend)
        if len(days_aero_2024) > 0:
            ax_bottom.plot(days_aero_2024.values, median_aero_2024.values,
                          color='orange', linewidth=2, marker='s', markersize=4,
                          markerfacecolor='yellow', markeredgecolor='orange',
                          label=f'{reference_attraction} 2024', zorder=3, alpha=0.7)

        # Аэротакси 2025
        days_aero_2025, median_aero_2025 = calculate_rolling_median(
            df_aerotaxi_2025, reference_attraction, is_weekend)
        if len(days_aero_2025) > 0:
            ax_bottom.plot(days_aero_2025.values, median_aero_2025.values,
                          color='red', linewidth=2, marker='o', markersize=4,
                          markerfacecolor='pink', markeredgecolor='red',
                          linestyle='--', label=f'{reference_attraction} 2025 (реальные)',
                          zorder=4)

        # Температура 2024
        temp_2024 = df_aerotaxi_2024[df_aerotaxi_2024['is_weekend'] == is_weekend].sort_values('ДеньГода')
        if len(temp_2024) > 0:
            ax_temp2 = ax_bottom.twinx()
            temp_median_2024 = temp_2024['Температура средняя (°C)'].rolling(
                window=WINDOW_SIZE, min_periods=3, center=True).median()
            ax_temp2.plot(temp_2024['ДеньГода'].values, temp_median_2024.values,
                         color='purple', linewidth=1.5, linestyle=':',
                         alpha=0.7, label='Температура')
            ax_temp2.set_ylabel('Средняя температура (C)', color='purple', fontsize=10)
            ax_temp2.tick_params(axis='y', labelcolor='purple')

        # Закрашиваем дождливые дни
        rain_2024 = df_aerotaxi_2024[(df_aerotaxi_2024['is_weekend'] == is_weekend) &
                                     (df_aerotaxi_2024['is_rain'] == 1)]['ДеньГода'].values
        for day in rain_2024:
            ax_bottom.axvspan(day - 0.5, day + 0.5, color='lightblue', alpha=0.4, zorder=1)

        ax_bottom.set_xlabel('Дата', fontsize=10)
        ax_bottom.set_ylabel('Количество проданных билетов', fontsize=10)
        ax_bottom.set_title(f'{reference_attraction} (для справки) - {day_type_name}',
                           fontsize=12, fontweight='bold')
        ax_bottom.grid(True, alpha=0.3)
        ax_bottom.legend(loc='upper left', fontsize=9)
        ax_bottom.set_xlim(90, 290)
        ax_bottom.set_xticks([t[0] for t in month_ticks])
        ax_bottom.set_xticklabels([t[1] for t in month_ticks], fontsize=9)

    fig.text(0.5, 0.02,
             'Голубая заливка = дождливые дни | Пунктирная фиолетовая = температура',
             ha='center', fontsize=10, style='italic')

    plt.suptitle(
        f'Прогноз: "{base_attraction}" с ростом "{reference_attraction}" относительно прогноза модели\n'
        f'Коэффициент роста = Реальные данные Аэротакси 2025 / Прогноз модели',
        fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = f'{base_attraction.lower().replace(" ", "_")}_with_{reference_attraction.lower().replace(" ", "_")}_model_growth.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    return filename


def print_growth_table(growth_coefficients, reference_attraction):
    """Печать таблицы коэффициентов"""
    month_names = {4: 'Апрель', 5: 'Май', 6: 'Июнь', 7: 'Июль',
                   8: 'Август', 9: 'Сентябрь', 10: 'Октябрь'}

    print(f"\nКоэффициенты роста {reference_attraction} (Реальные 2025 / Прогноз модели):")
    print("=" * 60)
    print(f"{'Месяц':<12} | {'Рабочие дни':>12} | {'Выходные':>12}")
    print("-" * 60)

    for month in range(4, 11):
        rate_work = growth_coefficients.get((month, 0), 1.0)
        rate_weekend = growth_coefficients.get((month, 1), 1.0)
        print(f"{month_names[month]:<12} | {rate_work:>11.2f}x | {rate_weekend:>11.2f}x")


def main():
    print("=" * 70)
    print("Прогноз Вальса часов с ростом Аэротакси относительно прогноза модели")
    print("=" * 70)

    # Загрузка данных
    print("\nЗагрузка данных...")
    df = load_and_prepare_data()

    base_attraction = 'Вальс часов'
    reference_attraction = 'Аэротакси'

    # Обучаем модель для Аэротакси и получаем прогнозы
    print(f"\nОбучение модели для {reference_attraction}...")
    train_aero, test_aero = train_prediction_model(df, reference_attraction)

    if train_aero is None or test_aero is None:
        print("Недостаточно данных для обучения модели")
        return

    print(f"  Обучено на {len(train_aero)} днях 2024 года")
    print(f"  Прогноз для {len(test_aero)} дней 2025 года")

    # Рассчитываем коэффициенты роста
    print(f"\nРасчет коэффициентов роста (Реальные/Прогноз)...")
    growth_coefficients = calculate_growth_coefficients(test_aero, reference_attraction)

    # Выводим таблицу
    print_growth_table(growth_coefficients, reference_attraction)

    # Создаем график
    print(f"\nСоздание графика для {base_attraction}...")
    filename = plot_comparison(df, base_attraction, reference_attraction, growth_coefficients)

    print(f"\nГрафик сохранен: {filename}")
    print("\n" + "=" * 70)
    print("Готово!")


if __name__ == '__main__':
    main()
