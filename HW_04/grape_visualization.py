## Streamlit 시각화
# - [배] 20년간 추이 분석 2004-2024, [전주]에서만
# - [포도] 20년간 추이 분석 2004-2024, [김제, 화성, 완주, 거창, 상주, 천안, 춘천, 옥천]

import pandas as pd
import numpy as np
import streamlit as st
import requests
from io import StringIO
from streamlit_folium import folium_static
import folium
from folium.plugins import MarkerCluster

def dataframe_load(loc, year1, year2, month, day):
    url = f'https://api.taegon.kr/station/{loc}/?sy={year1}&ey={year2}&format=csv'
    response = requests.get(url)
    csv_data = response.content.decode('utf-8')
    df = pd.read_csv(StringIO(csv_data), skipinitialspace=True)

    df.columns = df.columns.str.strip()  # 컬럼 이름 공백 제거
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df.set_index('date', inplace=True)
    filtered_df = df[df.index.map(lambda d: (d.month > month) or (d.month == month and d.day >= day))]

    return filtered_df


# 함수 형태로 변환
def calculate_blossom_date_mDVR(df):
    month = 2
    day = 14

    # 전날의 최고기온과 다음날의 최저기온 계산
    df['tmax_yesterday'] = df['tmax'].shift(1)
    df['tmin_tomorrow'] = df['tmin'].shift(-1)

    # 시간대별 온도 계산하는 함수
    def calculate_hourly_temperature(row, hour):
        m = row['tmin']  # 현재의 최저기온
        h = row['tmax']  # 현재의 최고기온
        hy = row['tmax_yesterday']  # 전날의 최고기온
        mt = row['tmin_tomorrow']  # 다음날의 최저기온

        if 0 <= hour <= 3:
            return (hy - m) * (np.sin((4 - hour) * np.pi / 30) ** 2) + m
        elif 4 <= hour <= 13:
            return (h - m) * (np.sin((hour - 4) * np.pi / 18) ** 2) + m
        elif 14 <= hour <= 23:
            return (h - mt) * (np.sin((28 - hour) * np.pi / 30) ** 2) + mt
        else:
            return np.nan

    # 각 시간별로 온도를 계산하여 새로운 컬럼 추가
    for hour in range(24):
        df[f'temp_{hour}h'] = df.apply(lambda row: calculate_hourly_temperature(row, hour), axis=1)

    # DVR2 계산
    for i in range(24):
        df[f'temp_{i}h'] = df[f'temp_{i}h'].apply(
            lambda x: np.exp(35.27 - 12094 * (x + 273) ** -1) if x < 20 else np.exp(5.82 - 3474 * (x + 273) ** -1))

    # 시간대별 온도의 합계를 계산하고 DVR2 누적값을 계산
    df['cumsum_temp'] = df.loc[:, 'temp_0h':'temp_23h'].sum(axis=1)
    df['cumsum_dvr2'] = df['cumsum_temp'].cumsum()

    # DVR2 누적값이 0.9593 이상이 되는 첫 번째 날짜 추출
    result = df[df['cumsum_dvr2'] >= 0.9593].iloc[0]
    blossom = f'{int(result["year"])}년 {int(result["month"])}월 {int(result["day"])}일'

    return blossom


def calculate_blossom_date_DVR(df, A=107.94, B=0.9):

    # DVRi 계산
    df['DVRi'] = df['tavg'].apply(lambda x: (1 / (A * (B ** x))) * 100 if x >= 5 else 0)

    # DVS 계산 (누적 DVRi)
    df['DVS'] = df['DVRi'].cumsum()

    # DVS가 100이 넘는 첫 번째 날짜 추출
    result = df[df['DVS'] > 100].iloc[0][['year', 'month', 'day']]
    blossom = f'{int(result["year"])}년 {int(result["month"])}월 {int(result["day"])}일'

    return blossom

# 결과 저장하기
def add_blossom_data(df, location, year, blossomdate):
    new_row = pd.DataFrame({'location': [location], 'year': [year], 'blossomdate': [blossomdate]})
    return pd.concat([df, new_row], ignore_index=True)

def location(latlon):
    grapeloc = ['김제시', '완주군', '거창군', '상주시', '옥천군']

    drop_indices = []
    for index, row in latlon.iterrows():
        if row['city'] not in grapeloc:
            drop_indices.append(index)

    latlon.drop(drop_indices, axis=0, inplace=True)

    return latlon

def main():
    dataframe_load()

    latlon = pd.read_csv('korea_administrative_division_latitude_longitude.csv')
    latlon = location(latlon)


    m = folium.Map(location=[36.350412, 127.384548], zoom_start=10)
    marker_cluster = MarkerCluster().add_to(m)

    for idx, row in latlon.iterrows():
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=row["city"],
        ).add_to(marker_cluster)

    folium_static(m)



if __name__ == "__main__":
    main()