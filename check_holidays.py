import pandas as pd
import argparse
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

def _read_csv_any_name() -> tuple[str, pd.DataFrame]:
    # В проекте встречаются два визуально похожих имени файла из-за юникод-нормализации "й"
    candidates = [
        'Все_с_канаткой_полные_данные.csv',
        'Все_с_канаткой_полные_данные.csv',
    ]
    last_err = None
    for path in candidates:
        try:
            df = pd.read_csv(path, encoding='utf-8')
            return path, df
        except FileNotFoundError as e:
            last_err = e
    raise FileNotFoundError(f"Не найден CSV. Пробовал: {', '.join(candidates)}") from last_err


def update_holidays(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])

    def update_day_type(row):
        dt = row['Дата']
        current_type = row['Тип дня']
        if is_holiday(dt):
            return "Праздник"
        return current_type

    df['Тип дня'] = df.apply(update_day_type, axis=1)
    return df


def update_weather(df: pd.DataFrame, lat: float, lon: float, alt: int | None = None) -> pd.DataFrame:
    # Ленивая зависимость: чтобы режим праздников работал даже без meteostat
    import meteostat as ms
    import ssl
    import urllib.request

    # Исправление SSL для macOS
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    df = df.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])

    if 'Осадки (мм)' not in df.columns:
        df['Осадки (мм)'] = pd.NA
    if 'Скорость ветра (км/ч)' not in df.columns:
        df['Скорость ветра (км/ч)'] = pd.NA

    prcp_missing = df['Осадки (мм)'].isna() | (df['Осадки (мм)'].astype(str).str.strip() == '') | (df['Осадки (мм)'].astype(str).str.contains('Нет данных', na=False))
    wspd_missing = df['Скорость ветра (км/ч)'].isna() | (df['Скорость ветра (км/ч)'].astype(str).str.strip() == '') | (df['Скорость ветра (км/ч)'].astype(str).str.contains('Нет данных', na=False))
    need_mask = prcp_missing | wspd_missing

    if int(need_mask.sum()) == 0:
        print("Погодные колонки уже заполнены — обновлять нечего.")
        return df

    start = df.loc[need_mask, 'Дата'].min().date()
    end = df.loc[need_mask, 'Дата'].max().date()

    print(f"Загрузка погодных данных из Meteostat для периода {start} - {end}...")

    point = ms.Point(lat, lon, alt) if alt is not None else ms.Point(lat, lon)
    stations = ms.stations.nearby(point, limit=4)
    
    # Получаем данные напрямую из станций (может содержать prcp)
    ts_raw = ms.daily(stations, start, end)
    ts_df = ts_raw.fetch()
    
    # Интерполируем для точки
    wx_df = ms.interpolate(ts_raw, point).fetch()
    
    if wx_df is None or len(wx_df) == 0:
        print("Предупреждение: не удалось получить данные из Meteostat.")
        return df
    
    wx_df = wx_df.reset_index()
    
    # Проверяем название колонки с датой
    if 'time' in wx_df.columns:
        wx_df = wx_df.rename(columns={'time': 'Дата'})
    elif wx_df.index.name:
        wx_df = wx_df.reset_index()
        if 'time' in wx_df.columns:
            wx_df = wx_df.rename(columns={'time': 'Дата'})
    
    wx_df['Дата'] = pd.to_datetime(wx_df['Дата'])
    
    # Если prcp нет в интерполированных данных, пробуем взять из сырых данных станций
    if 'prcp' not in wx_df.columns and ts_df is not None and len(ts_df) > 0:
        ts_df = ts_df.reset_index()
        if 'time' in ts_df.columns:
            ts_df = ts_df.rename(columns={'time': 'Дата'})
        elif ts_df.index.name == 'time':
            ts_df = ts_df.reset_index()
            if 'time' in ts_df.columns:
                ts_df = ts_df.rename(columns={'time': 'Дата'})
        ts_df['Дата'] = pd.to_datetime(ts_df['Дата'])
        if 'prcp' in ts_df.columns:
            # Берем среднее значение осадков по всем станциям для каждой даты
            prcp_avg = ts_df.groupby('Дата')['prcp'].mean().reset_index()
            prcp_avg.columns = ['Дата', 'prcp']
            wx_df = wx_df.merge(prcp_avg, on='Дата', how='left')
            print(f"Данные об осадках получены из сырых данных станций для {len(prcp_avg)} дат")
    
    # Проверяем наличие нужных колонок
    has_prcp = 'prcp' in wx_df.columns
    has_wspd = 'wspd' in wx_df.columns
    
    if not has_prcp and not has_wspd:
        print(f"Не найдены нужные колонки. Доступные: {list(wx_df.columns)}")
        return df
    
    # Формируем финальный датафрейм только с нужными колонками
    cols_to_select = ['Дата']
    if has_prcp:
        cols_to_select.append('prcp')
    if has_wspd:
        cols_to_select.append('wspd')
    
    wx_df = wx_df[cols_to_select].copy()
    
    # Если prcp все еще нет, создаем пустую колонку
    if not has_prcp:
        wx_df['prcp'] = pd.NA
        print("Предупреждение: данные об осадках недоступны для этого периода.")

    # Убеждаемся, что даты в правильном формате для merge
    wx_df['Дата'] = pd.to_datetime(wx_df['Дата']).dt.normalize()
    df['Дата'] = pd.to_datetime(df['Дата']).dt.normalize()
    
    merged = df.merge(wx_df, on='Дата', how='left')
    
    # Исправляем типы данных перед присваиванием
    if 'prcp' in merged.columns:
        merged['prcp'] = pd.to_numeric(merged['prcp'], errors='coerce')
    if 'wspd' in merged.columns:
        merged['wspd'] = pd.to_numeric(merged['wspd'], errors='coerce')
    
    # Отладка: проверяем совпадения
    if len(wx_df) > 0:
        matched_dates = merged[merged['prcp'].notna() | merged['wspd'].notna()]['Дата'].unique()
        print(f"Найдено совпадений по датам: {len(matched_dates)} из {len(wx_df)}")
    
    # Заполняем только там, где данные есть и они не пустые
    if 'prcp' in merged.columns:
        prcp_values = merged.loc[prcp_missing, 'prcp']
        prcp_valid = prcp_values.notna()
        if prcp_valid.any():
            merged.loc[prcp_missing & prcp_valid, 'Осадки (мм)'] = merged.loc[prcp_missing & prcp_valid, 'prcp'].astype(float)
    
    if 'wspd' in merged.columns:
        wspd_values = merged.loc[wspd_missing, 'wspd']
        wspd_valid = wspd_values.notna()
        if wspd_valid.any():
            merged.loc[wspd_missing & wspd_valid, 'Скорость ветра (км/ч)'] = merged.loc[wspd_missing & wspd_valid, 'wspd'].astype(float)
    
    merged = merged.drop(columns=[col for col in ['prcp', 'wspd'] if col in merged.columns])
    
    filled_prcp = (merged.loc[prcp_missing, 'Осадки (мм)'].notna() & 
                   (merged.loc[prcp_missing, 'Осадки (мм)'].astype(str) != '') &
                   (~merged.loc[prcp_missing, 'Осадки (мм)'].astype(str).str.contains('Нет данных', na=False))).sum()
    filled_wspd = (merged.loc[wspd_missing, 'Скорость ветра (км/ч)'].notna() & 
                   (merged.loc[wspd_missing, 'Скорость ветра (км/ч)'].astype(str) != '') &
                   (~merged.loc[wspd_missing, 'Скорость ветра (км/ч)'].astype(str).str.contains('Нет данных', na=False))).sum()
    
    print(f"Заполнено: {filled_prcp} значений осадков, {filled_wspd} значений скорости ветра")
    if len(wx_df) > 0:
        print(f"Получено {len(wx_df)} записей из Meteostat за период {wx_df['Дата'].min().date()} - {wx_df['Дата'].max().date()}")
    
    return merged


def main():
    parser = argparse.ArgumentParser(description="Обновление праздников и/или погодных колонок в CSV")
    parser.add_argument('--holidays', action='store_true', help='Обновить колонку "Тип дня" по фиксированным праздникам РФ')
    parser.add_argument('--weather', action='store_true', help='Дозаполнить "Осадки (мм)" и "Скорость ветра (км/ч)" из Meteostat')
    parser.add_argument('--lat', type=float, default=None, help='Широта точки (для Meteostat)')
    parser.add_argument('--lon', type=float, default=None, help='Долгота точки (для Meteostat)')
    parser.add_argument('--alt', type=int, default=None, help='Высота, м (опционально, для Meteostat)')
    args = parser.parse_args()

    if not args.holidays and not args.weather:
        # Исторически этот скрипт обновлял праздники — сохраняем поведение по умолчанию
        args.holidays = True

    csv_path, df = _read_csv_any_name()

    if args.holidays:
        df = update_holidays(df)
        holidays_found = df[df['Тип дня'] == 'Праздник']
        print(f"Найдено праздничных дней: {len(holidays_found)}")
        print("\nПраздничные дни в данных:")
        for _, row in holidays_found.iterrows():
            holiday_name = get_holiday_name(row['Дата'])
            day_name = row.get('День недели', '')
            print(f"  {row['Дата'].strftime('%Y-%m-%d')} ({day_name}) - {holiday_name}")

    if args.weather:
        if args.lat is None or args.lon is None:
            raise SystemExit("Для --weather нужно указать --lat и --lon (а --alt опционально).")
        df = update_weather(df, lat=args.lat, lon=args.lon, alt=args.alt)

    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\nФайл обновлён: {csv_path}")

    print(f"\nСтатистика по типам дней:")
    print(df['Тип дня'].value_counts())


if __name__ == '__main__':
    main()
