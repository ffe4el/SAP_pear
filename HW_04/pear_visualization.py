## Streamlit 시각화
# - [배] 20년간 추이 분석 2004-2024, [전주]에서만
# - [포도] 20년간 추이 분석 2004-2024, [김제, 화성, 완주, 거창, 상주, 천안, 춘천, 옥천]

import pandas as pd
import numpy as np
import streamlit as st
import requests
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import platform
from streamlit_folium import folium_static
import folium
from folium.plugins import MarkerCluster
from folium import Popup
import plotly.graph_objects as go



# 시스템에 따라 다른 폰트 적용
if platform.system() == 'Darwin':  # macOS
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':  # Windows
    rc('font', family='Malgun Gothic')
else:  # Linux, 기타
    rc('font', family='NanumGothic')

# 한글 폰트 적용 후 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False


data = ['90-속초', '93-북춘천', '95-철원', '98-동두천', '99-파주', '100-대관령', '101-춘천', '102-백령도', '104-북강릉', '105-강릉', '106-동해',
        '108-서울', '112-인천', '114-원주', '115-울릉도', '116-관악산', '119-수원', '121-영월', '127-충주', '129-서산', '130-울진', '131-옥천',
        '133-대전', '135-추풍령', '136-안동', '137-상주', '138-포항', '140-군산', '143-대구', '146-완주', '152-울주', '155-창원', '156-나주',
        '159-부산', '162-통영', '164-무안', '165-목포', '168-여수', '169-흑산도', '170-완도', '172-고창', '174-순천',
        '177-홍성', '184-제주', '185-고산', '187-성산', '188-성산', '189-서귀포', '192-사천', '201-강화', '202-양평',
        '203-이천','211-인제', '212-홍천', '214-삼척', '216-태백', '217-정선', '221-제천', '226-보은', '232-천안', '235-보령', '236-부여',
        '238-금산', '239-세종', '243-부안', '244-임실', '245-정읍', '247-남원', '248-장수', '251-고창', '252-영광', '253-김해', '254-순창',
        '255-북창원','256-주암', '257-양산', '258-보성', '259-강진', '260-장흥', '261-해남','262-고흥', '263-의령', '264-함양', '265-성산포',
        '266-광양', '268-진도','271-봉화', '272-영주', '273-문경', '276-청송', '277-영덕', '278-의성','279-구미', '281-영천',
        '283-경주', '284-거창', '285-합천', '288-밀양', '289-산청', '294-거제', '295-남해']

# 도시 이름을 받아 해당 도시의 코드를 반환하는 함수
def find_area_code(city_name, data):
    for item in data:
        area_code, area_name = item.split('-')
        if city_name in area_name:
            return area_code
    return None

# 도시 코드를 받아 해당 도시의 이름을 반환하는 함수
def find_area_name(loc_code, data):
    for item in data:
        area_code, area_name = item.split('-')
        if loc_code in area_code:
            return area_name
    return None

# 선택된 도시 이름들을 코드로 변환하는 함수
def location(city_names):
    area_codes = []
    for city in city_names:
        area_code = find_area_code(city, data)
        if area_code:
            area_codes.append(area_code)
        else:
            st.write(f'{city}에 해당하는 데이터가 없습니다.')
    return area_codes


def dataframe_load(loc, year1, year2, month, day):
    url = f'https://api.taegon.kr/station/{loc}/?sy={year1}&ey={year2}&format=csv'
    response = requests.get(url)
    csv_data = response.content.decode('utf-8')
    df = pd.read_csv(StringIO(csv_data), skipinitialspace=True)

    df = df[['year', 'month', 'day', 'tavg', 'tmax', 'tmin']]
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df.set_index('date', inplace=True)
    filtered_df = df[df.index.map(lambda d: (d.month > month) or (d.month == month and d.day >= day))]

    return filtered_df


def calculate_blossom_date_CD(df, Tc=5.4, Hr=272):
    """
    Tc와 Hr 값을 이용하여 누적 cd 값이 Hr 이상이 되는 첫 번째 시점의 날짜를 계산하는 함수.

    Parameters:
    df (DataFrame): tmin, tmax, tavg, year, month, day 등의 열이 포함된 데이터프레임
    Tc (float): 기준 기온 값 (default=5.4)
    Hr (float): 누적된 cd 값의 임계값 (default=272)

    Returns:
    str: 누적 cd 값이 Hr 이상이 되는 첫 번째 날짜 (예: '2023년 4월 15일')
    """

    # 계산 함수 정의
    def calculate_cd(row, Tc):
        tmin = row['tmin']
        tmax = row['tmax']
        tavg = row['tavg']

        # 조건 1: Tc가 tmin과 tmax 사이에 있을 때
        if 0 <= Tc <= tmin <= tmax:
            return tavg - Tc

        # 조건 2: tmin <= Tc <= tmax (Tc가 tmin과 tmax 사이에 있을 때)
        elif 0 <= tmin <= Tc <= tmax:
            return (tmax - Tc) / 2

        # 조건 3: tmin과 tmax가 모두 Tc 이하일 때
        elif 0 <= tmin <= tmax <= Tc:
            return 0

        # 조건 4: tmin이 0 이하이고 tmax가 Tc 이하일 때
        elif tmin <= 0 <= tmax <= Tc:
            return 0

        # 조건 5: tmin이 0 이하이고 Tc가 tmax보다 작을 때
        elif tmin <= 0 <= Tc <= tmax:
            return (tmax - Tc) / 2

        # 그 외의 경우
        else:
            return 0

    # 각 행에 대해 cd 값 계산
    df['cd'] = df.apply(lambda row: calculate_cd(row, Tc), axis=1)

    # 누적 합계 계산
    df['cumsum_cd'] = df['cd'].cumsum()

    # Hr 이상이 되는 첫 번째 시점의 결과 추출
    result = df[df['cumsum_cd'] >= Hr].iloc[0]
    # blossom을 datetime 형태로 변환 (시분초 없이 날짜만)
    blossom = pd.to_datetime(f'{int(result["year"])}-{int(result["month"])}-{int(result["day"])}', format='%Y-%m-%d')

    return blossom


def calculate_blossom_date_mDVR(df):
    """
        시간대별 온도를 계산하고 발육속도(DVR2)의 누적 합계가 0.9593 이상이 되는 날짜를 계산하는 함수.

        Parameters:
        df (DataFrame): tmin, tmax, year, month, day 등의 열이 포함된 데이터프레임

        Returns:
        str: DVR2의 누적 합계가 0.9593 이상이 되는 날짜 (예: '2023년 4월 15일')
        """

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
    # blossom을 datetime 형태로 변환 (시분초 없이 날짜만)
    blossom = pd.to_datetime(f'{int(result["year"])}-{int(result["month"])}-{int(result["day"])}', format='%Y-%m-%d')

    return blossom


def calculate_blossom_date_DVR(df, A=107.94, B=0.9):

    # DVRi 계산
    df['DVRi'] = df['tavg'].apply(lambda x: (1 / (A * (B ** x))) * 100 if x >= 5 else 0)

    # DVS 계산 (누적 DVRi)
    df['DVS'] = df['DVRi'].cumsum()

    # DVS가 100이 넘는 첫 번째 날짜 추출
    result = df[df['DVS'] > 100].iloc[0][['year', 'month', 'day']]
    # blossom을 datetime 형태로 변환 (시분초 없이 날짜만)
    blossom = pd.to_datetime(f'{int(result["year"])}-{int(result["month"])}-{int(result["day"])}', format='%Y-%m-%d')


    return blossom



# 결과 저장
def add_blossom_date(loclist, year1, year2):
    """
        loc_codes에 있는 각 위치에 대해 DVR, mDVR, CD 결과를 계산하고
        결과를 DataFrame으로 저장하는 함수.

        Parameters:
        loc_codes (list): 지역 코드들
        year1 (int): 시작 연도
        year2 (int): 종료 연도

        Returns:
        DataFrame: 각 위치와 해당 연도의 DVR, mDVR, CD에 대한 blossomdate 결과가 포함된 DataFrame
        """
    result_df = pd.DataFrame(columns=['location', 'year', 'dvr_blossomdate', 'mdvr_blossomdate', 'cd_blossomdate'])

    for loc in loclist:
        # 전체 데이터프레임을 사용하고 특정 연도에 맞는 데이터를 필터링
        dvr_data = dataframe_load(loc, year1, year2, 1, 30)
        mdvr_cd_data = dataframe_load(loc, year1, year2, 2, 15)

        # 지역코드를 지역명으로 변환해서 변수에 저장
        loc_name = find_area_name(loc, data)

        for year in range(year1, year2+1):
            # 각 연도에 맞게 데이터를 필터링
            dvr_filtered = dvr_data[dvr_data.index.year == year]
            mdvr_filtered = mdvr_cd_data[mdvr_cd_data.index.year == year]

            # DVR, mDVR, CD 결과 계산
            dvr_result = calculate_blossom_date_DVR(dvr_filtered)
            mdvr_result = calculate_blossom_date_mDVR(mdvr_filtered)
            cd_result = calculate_blossom_date_CD(mdvr_filtered)

            new_row = {'location': loc_name, 'year': year, #여기서 location에는 숫자 코드가 저장되어있다....!
                       'dvr_blossomdate': dvr_result,
                       'mdvr_blossomdate': mdvr_result,
                       'cd_blossomdate': cd_result}

            result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)

    return result_df


# 포도 결과 저장
def calculate_blossom_date_CD_grape(df, Tc=5.6, Hr=275):
    """
    Tc와 Hr 값을 이용하여 누적 cd 값이 Hr 이상이 되는 첫 번째 시점의 날짜를 계산하는 함수.

    Parameters:
    df (DataFrame): tmin, tmax, tavg, year, month, day 등의 열이 포함된 데이터프레임
    [포도의 경우]
    Tc (float): 기준 기온 값 (default=5.6)
    Hr (float): 누적된 cd 값의 임계값 (default=275)

    Returns:
    str: 누적 cd 값이 Hr 이상이 되는 첫 번째 날짜 (예: '2023-4-15')
    """

    # 계산 함수 정의
    def calculate_cd(row, Tc):
        tmin = row['tmin']
        tmax = row['tmax']
        tavg = row['tavg']

        # 조건 1: Tc가 tmin과 tmax 사이에 있을 때
        if 0 <= Tc <= tmin <= tmax:
            return tavg - Tc

        # 조건 2: tmin <= Tc <= tmax (Tc가 tmin과 tmax 사이에 있을 때)
        elif 0 <= tmin <= Tc <= tmax:
            return (tmax - Tc) / 2

        # 조건 3: tmin과 tmax가 모두 Tc 이하일 때
        elif 0 <= tmin <= tmax <= Tc:
            return 0

        # 조건 4: tmin이 0 이하이고 tmax가 Tc 이하일 때
        elif tmin <= 0 <= tmax <= Tc:
            return 0

        # 조건 5: tmin이 0 이하이고 Tc가 tmax보다 작을 때
        elif tmin <= 0 <= Tc <= tmax:
            return (tmax - Tc) / 2

        # 그 외의 경우
        else:
            return 0

    # 각 행에 대해 cd 값 계산
    df['cd'] = df.apply(lambda row: calculate_cd(row, Tc), axis=1)

    # 누적 합계 계산
    df['cumsum_cd'] = df['cd'].cumsum()

    # Hr 이상이 되는 첫 번째 시점의 결과 추출
    result = df[df['cumsum_cd'] >= Hr].iloc[0]
    # blossom을 datetime 형태로 변환 (시분초 없이 날짜만)
    blossom = pd.to_datetime(f'{int(result["year"])}-{int(result["month"])}-{int(result["day"])}', format='%Y-%m-%d')

    return blossom

def calculate_blossom_date_DVR_grape(df):
    """
    DVR 계산식: DVR = 0.0019 * tavg + 0.0187
    DVS가 1.0을 넘는 첫 번째 날짜를 추출
    """

    # DVR 계산 (DVR = 0.0019 * tavg + 0.0187) 기준온도 10도!
    df['DVRi'] = df['tavg'].apply(lambda x: 0.0019 * x + 0.0187 if x >= 10 else 0)

    # DVS 계산 (누적 DVRi)
    df['DVS'] = df['DVRi'].cumsum()

    # DVS가 1.0을 넘는 첫 번째 날짜 추출
    result = df[df['DVS'] > 1.0].iloc[0][['year', 'month', 'day']]

    # blossom을 datetime 형태로 변환 (시분초 없이 날짜만)
    blossom = pd.to_datetime(f'{int(result["year"])}-{int(result["month"])}-{int(result["day"])}', format='%Y-%m-%d')

    return blossom

def add_blossom_date_grape(loclist, year1, year2):
    """
        loc_codes에 있는 각 위치에 대해 DVR, mDVR, CD 결과를 계산하고
        결과를 DataFrame으로 저장하는 함수.

        Parameters:
        loc_codes (list): 지역 코드들
        year1 (int): 시작 연도
        year2 (int): 종료 연도

        Returns:
        DataFrame: 각 위치와 해당 연도의 CD에 대한 blossomdate 결과가 포함된 DataFrame
        """
    result_df = pd.DataFrame(columns=['location', 'year', 'dvr_blossomdate', 'cd_blossomdate'])

    for loc in loclist:
        # 전체 데이터프레임을 사용하고 특정 연도에 맞는 데이터를 필터링
        cd_data = dataframe_load(loc, year1, year2, 1, 30)

        # 지역코드를 지역명으로 변환해서 변수에 저장
        loc_name = find_area_name(loc, data)

        for year in range(year1, year2+1):
            # 각 연도에 맞게 데이터를 필터링
            dvr_cd_filtered = cd_data[cd_data.index.year == year]

            # DVR, mDVR, CD 결과 계산
            dvr_result = calculate_blossom_date_DVR_grape(dvr_cd_filtered)
            cd_result = calculate_blossom_date_CD_grape(dvr_cd_filtered)

            new_row = {'location': loc_name, 'year': year, #여기서 location에는 숫자 코드가 저장되어있다....!
                       'dvr_blossomdate': dvr_result, 'cd_blossomdate': cd_result}

            result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)

    return result_df

# 결과 병합
def merge_locations(result_df, latlon):
    """
    result_df의 location과 latlon의 city를 기준으로 병합하되,
    location 컬럼에 포함된 단어가 city에 포함되어 있으면 같은 것으로 처리하여 병합.

    Parameters:
    result_df (DataFrame): location, year, blossomdate 등이 포함된 데이터프레임
    latlon (DataFrame): city, 위도, 경도 정보가 포함된 데이터프레임

    Returns:
    DataFrame: 병합된 데이터프레임
    """
    # 병합할 결과를 담을 빈 DataFrame 생성
    merged_df = pd.DataFrame()

    # 각 location의 값이 latlon의 city에 포함되어 있는지 확인하고 병합
    for loc in result_df['location']:
        # 해당 location이 포함된 city를 찾아 병합 (contains 함수 사용)
        matching_rows = latlon[latlon['city'].str.contains(loc)][['longitude', 'latitude']]
        if not matching_rows.empty:
            # location에 해당하는 행만 추출하여 병합
            temp_df = result_df[result_df['location'] == loc]

            # matching_rows의 위도, 경도 값을 해당 location의 모든 행에 적용
            longitude = matching_rows.iloc[0]['longitude']
            latitude = matching_rows.iloc[0]['latitude']

            temp_df['longitude'] = longitude
            temp_df['latitude'] = latitude

            # 병합된 결과를 merged_df에 추가
            merged_df = pd.concat([merged_df, temp_df], ignore_index=True)

    # 중복된 location과 year를 기준으로 중복 제거
    merged_df = merged_df.drop_duplicates(subset=['location', 'year'])

    return merged_df


# 개화일을 수치로 변환하는 함수 (기준: 3월 30일을 1로 설정)
def convert_date_to_num(date):
    ref_date = pd.Timestamp(year=date.year, month=3, day=30)  # 기준 3월 30일
    return (date - ref_date).days + 1

# 개화일을 수치로 변환하는 함수 (기준: 4월 1일을 1로 설정)
def convert_date_to_num_pear(date):
    ref_date = pd.Timestamp(year=date.year, month=4, day=1)  # 기준 4월 1일
    return (date - ref_date).days + 1


# Streamlit 상호작용 입력 [배: 기본세팅]
loclist_all = ['이천', '상주', '천안', '인천', '영천', '완주', '나주', '사천', '울주']
loc_codes_all = location(loclist_all)
latlon = pd.read_csv('korea_administrative_division_latitude_longitude.csv')


# 사이드바를 통해 페이지 선택
st.sidebar.title("Navigation")
page = st.sidebar.radio("메뉴를 선택하세요:", ["[배]개화일시 지도 시각화", "[배]개화일시 모델별 그래프", "[포도]개화일시 지도 시각화", "[포도]개화일시 모델 그래프", "연도별 누적온도 그래프"])


# ============== 개화일시 시각화 페이지 ==============
if page == "[배]개화일시 지도 시각화":
    st.title("🌸 배 20년간 만개기 예측 시기")

    # Streamlit 상호작용 입력
    loclist = st.multiselect(
        "분석할 지역을 선택하세요",
        options=['이천', '상주', '천안', '인천', '영천', '완주', '나주', '사천', '울주'],  # 선택할 수 있는 지역들
        default=['이천', '천안']  # 기본 선택 값
    )
    loc_codes = location(loclist)
    year3 = st.number_input("보고 싶은 연도를 입력하세요", min_value=2004, max_value=2024, value=2021)

    # 기준 선택 (dvr_blossom, mdvr_blossom, cd_blossom 중 하나)
    blossom_type = st.selectbox("기준 선택", ["dvr_blossomdate", "mdvr_blossomdate", "cd_blossomdate"])

    # latlon 파일 병합
    result_df = add_blossom_date(loc_codes, year3, year3)
    merged_df = merge_locations(result_df, latlon)

    # DVR, mDVR, CD blossomdate 컬럼의 시분초 제거
    merged_df['dvr_blossomdate'] = pd.to_datetime(merged_df['dvr_blossomdate']).dt.strftime('%Y-%m-%d')
    merged_df['mdvr_blossomdate'] = pd.to_datetime(merged_df['mdvr_blossomdate']).dt.strftime('%Y-%m-%d')
    merged_df['cd_blossomdate'] = pd.to_datetime(merged_df['cd_blossomdate']).dt.strftime('%Y-%m-%d')


    # 지도 시각화
    st.subheader(f"{year3}년도 지역별 {blossom_type} 기준 지도")
    m = folium.Map(location=[36.350412, 127.384548], zoom_start=8)
    marker_cluster = MarkerCluster().add_to(m)

    # year3와 일치하는 데이터만 필터링
    merged_filtered_df = merged_df[merged_df['year'] == year3]

    # DVR 시기별로 빠른 순으로 색상 지정 (컬러맵을 이용해 색상 생성)
    merged_df = merged_df.sort_values(by='dvr_blossomdate')  # DVR 시기 빠른 순으로 정렬
    color_scale = ['#00FF00', '#7FFF00', '#FFFF00', '#FF7F00', '#FF0000']  # 녹색 -> 빨간색 순으로

    if not merged_filtered_df.empty:
        for idx, row in merged_filtered_df.iterrows():
            # 선택한 기준에 따라 날짜 표시
            selected_blossom = pd.to_datetime(row.get(blossom_type, "No data")).strftime('%Y-%m-%d') if row.get(
                blossom_type) != "No data" else "No data"

            # 날짜 형식으로 변환 (시분초 제거)
            dvr_blossom = pd.to_datetime(row.get("dvr_blossomdate", "No data")).strftime('%Y-%m-%d') if row.get(
                "dvr_blossomdate") != "No data" else "No data"
            mdvr_blossom = pd.to_datetime(row.get("mdvr_blossomdate", "No data")).strftime('%Y-%m-%d') if row.get(
                "mdvr_blossomdate") != "No data" else "No data"
            cd_blossom = pd.to_datetime(row.get("cd_blossomdate", "No data")).strftime('%Y-%m-%d') if row.get(
                "cd_blossomdate") != "No data" else "No data"

            # HTML 팝업에 넣기
            popup_html = f"""
                            <div style="font-family: Arial, sans-serif; font-size: 14px; line-height: 1.5;">
                                <h4 style="margin: 0; padding-bottom: 5px; color: #333;">Blossom Dates</h4>
                                <ul style="list-style-type: none; padding-left: 0;">
                                    <li><strong>Selected Blossom Date ({blossom_type}):</strong></li>
                                    <li><strong>DVR Blossom Date:</strong> {dvr_blossom}</li>
                                    <li><strong>mDVR Blossom Date:</strong> {mdvr_blossom}</li>
                                    <li><strong>CD Blossom Date:</strong> {cd_blossom}</li>
                                </ul>
                            </div>
                            """

            popup = Popup(popup_html, max_width=300)
            color = color_scale[idx % len(color_scale)]  # 색상 선택 (빠른 순으로)
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=25,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                tooltip=selected_blossom,  # 마커 위에 마우스를 올렸을 때 표시될 내용
                popup=popup,  # 마커를 클릭했을 때 나타나는 내용
            ).add_to(marker_cluster)

        folium_static(m)
    else:
        st.write("지도에 표시할 데이터가 없습니다.")

# 데이터 분석 페이지
elif page == "[배]개화일시 모델별 그래프":
    st.title("💐개화일시 모델별 그래프")

    year1 = st.number_input("시작 연도", min_value=2004, max_value=2024, value=2004)
    year2 = st.number_input("종료 연도", min_value=2004, max_value=2024, value=2024)

    # 지역 선택 (selectbox 사용)
    selected_location1 = st.selectbox("지역을 선택하세요", loclist_all)
    selected_location = [selected_location1]

    # 데이터를 로드 (2004년 ~ 2024년 데이터 로드)
    loc_code = location(selected_location)
    result_df = add_blossom_date(loc_code, year1, year2)
    merged_df = merge_locations(result_df, latlon)

    # 날짜 데이터를 변환
    merged_df['dvr_blossomdate'] = pd.to_datetime(merged_df['dvr_blossomdate'])
    merged_df['mdvr_blossomdate'] = pd.to_datetime(merged_df['mdvr_blossomdate'])
    merged_df['cd_blossomdate'] = pd.to_datetime(merged_df['cd_blossomdate'])

    # 날짜를 수치로 변환
    merged_df['dvr_blossomdate_num'] = merged_df['dvr_blossomdate'].apply(convert_date_to_num)
    merged_df['mdvr_blossomdate_num'] = merged_df['mdvr_blossomdate'].apply(convert_date_to_num)
    merged_df['cd_blossomdate_num'] = merged_df['cd_blossomdate'].apply(convert_date_to_num)

    # Streamlit에 데이터프레임 표시
    st.dataframe(merged_df)

    # 그래프 그리기
    fig, ax = plt.subplots()

    # DVR, mDVR, CD 모델의 개화일을 그리기
    ax.plot(merged_df['year'], merged_df['dvr_blossomdate_num'], 'o-', label='DVR', color='red')
    ax.plot(merged_df['year'], merged_df['mdvr_blossomdate_num'], 'o-', label='mDVR', color='green')
    ax.plot(merged_df['year'], merged_df['cd_blossomdate_num'], 'o-', label='CD', color='blue')

    # y축을 날짜로 표시
    y_ticks = range(1, 39)  # 4월 1일부터 5월 8일까지의 범위
    y_labels = pd.date_range(start="2020-04-01", end="2020-05-08").strftime('%m-%d')
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # x축을 정수로 표시
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # x축을 정수로 설정

    # 축과 레이블 설정
    ax.set_xlabel("Year")
    ax.set_ylabel("Full Bloom Dates")
    ax.set_title("[배] 개화일시 모델별 그래프")
    ax.legend()

    # 레이아웃 설정
    plt.tight_layout()

    # Streamlit에 그래프 표시
    st.pyplot(fig)


# 연도별 누적온도 그래프
elif page == "연도별 누적온도 그래프":
    st.title("📈연도별 누적온도 그래프")

    year1 = st.number_input("시작 연도", min_value=2004, max_value=2024, value=2004)
    year2 = st.number_input("종료 연도", min_value=2004, max_value=2024, value=2024)

    # 지역 선택 (selectbox 사용)
    selected_location1 = st.selectbox("지역을 선택하세요", loclist_all)
    selected_location = [selected_location1]

    dvr_data = dataframe_load(find_area_code(selected_location1, data), year1, year2, 1, 1)

    # 연도별 누적 평균 기온을 계산하기 위해 'date' 인덱스를 활용
    dvr_data = dvr_data.sort_index()

    # 월과 일 정보를 따로 추출하여 각 날짜별로 그룹화 가능하게 합니다.
    dvr_data['month'] = dvr_data.index.month
    dvr_data['day_of_year'] = dvr_data.index.dayofyear

    # 필요한 연도 목록만 필터링
    selected_years = [2004, 2009, 2014, 2019, 2024]

    # 그래프 설정
    fig, ax = plt.subplots()

    # 각 선택된 연도별로 누적 평균 기온을 1월~12월까지 플롯
    for year in selected_years:
        if year in dvr_data.index.year.unique():  # 해당 연도의 데이터가 존재하는 경우
            yearly_data = dvr_data[dvr_data.index.year == year]
            yearly_data['cumsum_tavg'] = yearly_data['tavg'].cumsum()  # 연도별 누적 기온
            ax.plot(yearly_data['day_of_year'], yearly_data['cumsum_tavg'], label=str(year))  # 연도별 누적 기온 그래프

    # x축을 1월부터 12월로 설정
    ax.set_xticks([31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])  # 월별로 일수에 따른 x좌표 설정
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    # 그래프 설정
    ax.set_xlabel('Month')
    ax.set_ylabel('Cumulative Average Temperature (℃)')
    ax.set_title('Yearly Cumulative Average Temperature (Selected Years)')
    ax.legend()

    # Streamlit에 그래프 출력
    st.pyplot(fig)


elif page == "[포도]개화일시 지도 시각화":
    st.title("🌸 포도 20년간 만개기 예측 시기")

    # Streamlit 상호작용 입력
    loclist = st.multiselect(
        "분석할 지역을 선택하세요",
        options=['거창', '상주', '옥천', '완주'],  # 선택할 수 있는 지역들
        default=['완주']  # 기본 선택 값
    )
    loc_codes = location(loclist)
    year3 = st.number_input("보고 싶은 연도를 입력하세요", min_value=2004, max_value=2024, value=2021)

    blossom_type = st.selectbox("기준 선택", ["dvr_blossomdate", "cd_blossomdate"])

    # latlon 파일 병합
    result_df = add_blossom_date_grape(loc_codes, year3, year3)
    merged_df = merge_locations(result_df, latlon)

    # DVR, mDVR, CD blossomdate 컬럼의 시분초 제거
    merged_df['dvr_blossomdate'] = pd.to_datetime(merged_df['dvr_blossomdate']).dt.strftime('%Y-%m-%d')
    merged_df['cd_blossomdate'] = pd.to_datetime(merged_df['cd_blossomdate']).dt.strftime('%Y-%m-%d')

    # 지도 시각화
    st.subheader(f"{year3}년도 지역별 {blossom_type} 기준 지도")
    m = folium.Map(location=[36.350412, 127.384548], zoom_start=8)
    marker_cluster = MarkerCluster().add_to(m)

    # year3와 일치하는 데이터만 필터링
    merged_filtered_df = merged_df[merged_df['year'] == year3]

    # DVR 시기별로 빠른 순으로 색상 지정 (컬러맵을 이용해 색상 생성)
    merged_df = merged_df.sort_values(by='dvr_blossomdate')  # DVR 시기 빠른 순으로 정렬
    color_scale = ['#00FF00', '#7FFF00', '#FF7F00', '#FF0000']  # 녹색 -> 빨간색 순으로

    if not merged_filtered_df.empty:
        for idx, row in merged_filtered_df.iterrows():
            # 선택한 기준에 따라 날짜 표시
            selected_blossom = pd.to_datetime(row.get(blossom_type, "No data")).strftime('%Y-%m-%d') if row.get(
                blossom_type) != "No data" else "No data"

            # 날짜 형식으로 변환 (시분초 제거)
            dvr_blossom = pd.to_datetime(row.get("dvr_blossomdate", "No data")).strftime('%Y-%m-%d') if row.get(
                "dvr_blossomdate") != "No data" else "No data"
            cd_blossom = pd.to_datetime(row.get("cd_blossomdate", "No data")).strftime('%Y-%m-%d') if row.get(
                "cd_blossomdate") != "No data" else "No data"

            # HTML 팝업에 넣기
            popup_html = f"""
                                <div style="font-family: Arial, sans-serif; font-size: 14px; line-height: 1.5;">
                                    <h4 style="margin: 0; padding-bottom: 5px; color: #333;">Blossom Dates</h4>
                                    <ul style="list-style-type: none; padding-left: 0;">
                                        <li><strong>Selected Blossom Date ({blossom_type}):</strong></li>
                                        <li><strong>DVR Blossom Date:</strong> {dvr_blossom}</li>
                                        <li><strong>CD Blossom Date:</strong> {cd_blossom}</li>
                                    </ul>
                                </div>
                                """

            popup = Popup(popup_html, max_width=300)
            color = color_scale[idx % len(color_scale)]  # 색상 선택 (빠른 순으로)
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=25,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                tooltip=selected_blossom,  # 마커 위에 마우스를 올렸을 때 표시될 내용
                popup=popup,  # 마커를 클릭했을 때 나타나는 내용
            ).add_to(marker_cluster)

        folium_static(m)
    else:
        st.write("지도에 표시할 데이터가 없습니다.")




elif page == "[포도]개화일시 모델 그래프":
    # Streamlit 타이틀
    st.title("💐[포도] 개화일시 모델 그래프")

    # 연도 입력
    year1 = st.number_input("시작 연도", min_value=2004, max_value=2024, value=2004)
    year2 = st.number_input("종료 연도", min_value=2004, max_value=2024, value=2024)

    # 지역 선택 (selectbox 사용)
    selected_location1 = st.selectbox("지역을 선택하세요", ['거창', '상주', '옥천', '완주'])
    selected_location = [selected_location1]

    # 데이터를 로드 (2004년 ~ 2024년 데이터 로드)
    loc_code = location(selected_location)
    result_df = add_blossom_date_grape(loc_code, year1, year2)
    merged_df = merge_locations(result_df, latlon)

    # 날짜 데이터를 변환
    merged_df['dvr_blossomdate'] = pd.to_datetime(merged_df['dvr_blossomdate'])
    merged_df['cd_blossomdate'] = pd.to_datetime(merged_df['cd_blossomdate'])

    # 날짜를 수치로 변환
    merged_df['dvr_blossomdate_num'] = merged_df['dvr_blossomdate'].apply(convert_date_to_num)
    merged_df['cd_blossomdate_num'] = merged_df['cd_blossomdate'].apply(convert_date_to_num)

    # Streamlit에 데이터프레임 표시
    st.dataframe(merged_df)

    # 그래프 그리기
    fig, ax = plt.subplots(figsize=(10, 6))

    # DVR, mDVR, CD 모델의 개화일을 그리기
    ax.plot(merged_df['year'], merged_df['dvr_blossomdate_num'], 'o-', label='DVR', color='red')
    ax.plot(merged_df['year'], merged_df['cd_blossomdate_num'], 'o-', label='CD', color='blue')

    # y축을 날짜로 표시
    y_ticks = range(1, 43)
    y_labels = pd.date_range(start="2020-03-30", end="2020-05-10").strftime('%m-%d')
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # x축을 정수로 표시
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # x축을 정수로 설정

    # 축과 레이블 설정
    ax.set_xlabel("Year")
    ax.set_ylabel("Full Bloom Dates")
    ax.set_title("개화일시 모델별 그래프")
    ax.legend()
    # 레이아웃 설정
    plt.tight_layout()

    # Streamlit에 그래프 표시
    st.pyplot(fig)
