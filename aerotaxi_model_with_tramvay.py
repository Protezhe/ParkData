"""
Предсказательная модель для посещаемости аттракционов.
Улучшенная версия с использованием Воздушного трамвая как прокси общей посещаемости.

Ключевое улучшение: Воздушный трамвай учитывает годовой тренд и внешние факторы,
что решает проблему завышения прогнозов в 2025 году.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# НАСТРОЙКИ - ИЗМЕНИТЕ ЗДЕСЬ НАЗВАНИЕ АТТРАКЦИОНА
# =============================================================================
ATTRACTION_NAME = 'Лунный экспресс'  # Название колонки аттракциона в CSV файле
PROXY_ATTRACTION = 'Воздушный трамвай'  # Прокси общей посещаемости парка
# =============================================================================

WINDOW_SIZE = 14

# Загрузка данных
df = pd.read_csv('Все_с_канаткой_полные_данные.csv', encoding='utf-8')
df['Дата'] = pd.to_datetime(df['Дата'])
df['Год'] = df['Дата'].dt.year
df['ДеньГода'] = df['Дата'].dt.dayofyear
df['Месяц'] = df['Дата'].dt.month
df['ДеньНедели'] = df['Дата'].dt.dayofweek

# === ПРИЗНАКИ ===

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
df['День_1'] = (df['ДеньНедели'] == 1).astype(int)  # Вторник
df['День_5'] = (df['ДеньНедели'] == 5).astype(int)  # Суббота

# Фильтруем только строки где есть ОБА значения (аттракцион и прокси)
df = df[df[ATTRACTION_NAME].notna() & (df[ATTRACTION_NAME] > 0) &
        df[PROXY_ATTRACTION].notna() & (df[PROXY_ATTRACTION] > 0)].copy()

print(f"Всего записей с данными {ATTRACTION_NAME} и {PROXY_ATTRACTION}: {len(df)}")
print(f"  2024: {len(df[df['Год'] == 2024])}")
print(f"  2025: {len(df[df['Год'] == 2025])}")
print("=" * 60)

# Набор признаков с прокси-аттракционом
feature_cols = [
    PROXY_ATTRACTION,  # Прокси общей посещаемости парка
    'is_weekend',
    'is_rain',
    'is_heavy_rain',
    'Темп_комфорт',
    'Темп_диапазон',
    'is_holiday',
    'День_1',
    'День_5',
]

# Разделение на обучающую (2024) и тестовую (2025) выборки
train_df = df[df['Год'] == 2024].copy()
test_df = df[df['Год'] == 2025].copy()

print(f"\nОбучающая выборка (2024): {len(train_df)} дней")
print(f"Тестовая выборка (2025): {len(test_df)} дней")

# Подготовка данных
X_train = train_df[feature_cols].fillna(0)
y_train = train_df[ATTRACTION_NAME]

X_test = test_df[feature_cols].fillna(0)
y_test = test_df[ATTRACTION_NAME]

# Обучение модели
print(f"\nОбучение модели Gradient Boosting с {PROXY_ATTRACTION}...")
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

print(f"\nМетрики на тестовой выборке (2025):")
print(f"  MAE (средняя абсолютная ошибка): {mae:.1f} билетов")
print(f"  R2 (коэффициент детерминации): {r2:.3f}")

# Добавляем предсказания в тестовый датафрейм
test_df['Предсказание'] = y_pred

# Важность признаков
print("\nВажность признаков:")
feature_importance = pd.DataFrame({
    'Признак': feature_cols,
    'Важность': model.feature_importances_
}).sort_values('Важность', ascending=False)
for _, row in feature_importance.iterrows():
    bar = '#' * int(row['Важность'] * 50)
    print(f"  {row['Признак']:<22} {row['Важность']:.3f} {bar}")


def calculate_rolling_median(data, column, is_weekend, window=WINDOW_SIZE):
    """Скользящая медиана для типа дня"""
    mask = data['is_weekend'] == is_weekend
    subset = data[mask].copy().sort_values('ДеньГода')

    if len(subset) < 3:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    median = subset[column].rolling(window=window, min_periods=3, center=True).median()
    return subset['ДеньГода'], median


def plot_prediction_comparison():
    """Строит графики сравнения предсказаний с реальностью"""

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    day_types = [(0, 'Рабочие дни'), (1, 'Выходные и праздники')]

    for j, (is_weekend, day_type_name) in enumerate(day_types):
        # Верхний ряд: сравнение предсказаний и реальности для 2025
        ax_top = axes[0, j]

        # Реальные данные 2025
        days_real, median_real = calculate_rolling_median(test_df, ATTRACTION_NAME, is_weekend)
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
        ax_top.set_title(f'{ATTRACTION_NAME} 2025 - {day_type_name}', fontsize=12, fontweight='bold')
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

        # Нижний ряд: данные 2024 (для справки)
        ax_bottom = axes[1, j]

        # Реальные данные 2024
        days_2024, median_2024 = calculate_rolling_median(train_df, ATTRACTION_NAME, is_weekend)
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
        ax_bottom.set_title(f'{ATTRACTION_NAME} 2024 (обучающие данные) - {day_type_name}',
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

    plt.suptitle(f'Предсказание посещаемости {ATTRACTION_NAME} (модель с {PROXY_ATTRACTION})\n'
                 f'MAE = {mae:.1f} билетов, R2 = {r2:.3f}',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = f'{ATTRACTION_NAME.lower().replace(" ", "_")}_prediction_with_tramvay.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nГрафик сохранен: {filename}")
    return filename


# Статистика по месяцам
print("\n" + "=" * 60)
print("Сравнение по месяцам (медиана):")
print("-" * 60)
print(f"{'Месяц':<12} {'Реально':<12} {'Предсказано':<12} {'Разница':<12}")
print("-" * 60)

for month in sorted(test_df['Месяц'].unique()):
    month_data = test_df[test_df['Месяц'] == month]
    real_median = month_data[ATTRACTION_NAME].median()
    pred_median = month_data['Предсказание'].median()
    diff = pred_median - real_median
    month_names = {4: 'Апрель', 5: 'Май', 6: 'Июнь', 7: 'Июль',
                   8: 'Август', 9: 'Сентябрь', 10: 'Октябрь'}
    print(f"{month_names.get(month, month):<12} {real_median:<12.0f} {pred_median:<12.0f} {diff:+.0f}")

# Построение графиков
plot_prediction_comparison()

print("\n" + "=" * 60)
print("Готово!")
