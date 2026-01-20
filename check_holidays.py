import pandas as pd
from datetime import date

# Российские праздники (фиксированные даты)
RUSSIAN_HOLIDAYS = {
    # Новогодние каникулы
    (1, 1): "Новый год",
    (1, 2): "Новогодние каникулы",
    (1, 3): "Новогодние каникулы",
    (1, 4): "Новогодние каникулы",
    (1, 5): "Новогодние каникулы",
    (1, 6): "Новогодние каникулы",
    (1, 7): "Рождество Христово",
    (1, 8): "Новогодние каникулы",
    # Другие праздники
    (2, 23): "День защитника Отечества",
    (3, 8): "Международный женский день",
    (5, 1): "Праздник Весны и Труда",
    (5, 9): "День Победы",
    (6, 12): "День России",
    (11, 4): "День народного единства",
}

def is_holiday(dt):
    """Проверяет, является ли дата праздником"""
    key = (dt.month, dt.day)
    return key in RUSSIAN_HOLIDAYS

def get_holiday_name(dt):
    """Возвращает название праздника или None"""
    key = (dt.month, dt.day)
    return RUSSIAN_HOLIDAYS.get(key, None)

# Загрузка данных
df = pd.read_csv('Все_с_канаткой_полные_данные.csv', encoding='utf-8')
df['Дата'] = pd.to_datetime(df['Дата'])

# Обновляем колонку "Тип дня"
def update_day_type(row):
    dt = row['Дата']
    current_type = row['Тип дня']

    if is_holiday(dt):
        return "Праздник"
    return current_type

# Применяем обновление
df['Тип дня'] = df.apply(update_day_type, axis=1)

# Показываем изменения
holidays_found = df[df['Тип дня'] == 'Праздник']
print(f"Найдено праздничных дней: {len(holidays_found)}")
print("\nПраздничные дни в данных:")
for _, row in holidays_found.iterrows():
    holiday_name = get_holiday_name(row['Дата'])
    print(f"  {row['Дата'].strftime('%Y-%m-%d')} ({row['День недели']}) - {holiday_name}")

# Сохраняем обновлённый файл
df.to_csv('Все_с_канаткой_полные_данные.csv', index=False, encoding='utf-8')
print(f"\nФайл обновлён!")

# Статистика
print(f"\nСтатистика по типам дней:")
print(df['Тип дня'].value_counts())
