import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# 설정
sns.set_style("whitegrid")
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.family"] = "AppleGothic"  

st.title("EDA 시각화")

# 수정된 base_path (pages → streamlit_app → jeju-traffic-predictor/output)
base_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'output')
)

X_train = joblib.load(os.path.join(base_path, 'X_train.pkl'))
y_train = joblib.load(os.path.join(base_path, 'y_train.pkl'))

train = X_train.copy()
train['target'] = y_train

# 상관관계 히트맵
st.subheader("1. 수치형 변수 간 상관관계 히트맵")

numeric_cols = train.select_dtypes(include=["number"])
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.markdown("""
- `maximum_speed_limit`(제한속도)와 `target`(평균속도)이 가장 높은 양의 상관관계를 보임
- `weight_restricted`, `road_type`도 양의 상관관계를 보이며 도로 조건이 속도에 영향을 줌
- 반대로 `road_rating`, `lane_count`는 약한 음의 상관관계를 보임
""")

# 도로 등급별 속도
st.subheader("2. 도로 등급(road_rating)별 평균 속도 분포")
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=train, x="road_rating", y="target", ax=ax)
st.pyplot(fig)

st.markdown("""
- 도로 등급이 낮을수록 평균 속도는 다소 높아지는 경향을 보임
- 이는 높은 등급의 도로가 오히려 도심 내부에 위치해 신호나 교차로가 많고, 차량 정체가 자주 발생하기 때문일 수 있음
- 반면 등급이 낮은 도로는 외곽 지역에 있어 차량 흐름이 원활한 경우가 많아 평균 속도가 높게 나타날 수 있음
""")

# 차로 수별 속도
st.subheader("3. 차로 수(lane_count)별 평균 속도 분포")
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=train, x="lane_count", y="target", ax=ax)
st.pyplot(fig)

st.markdown("""
- 차로 수가 많다고 해서 반드시 속도가 빠르진 않으며, 도심 지역일수록 오히려 느릴 수 있음
- 이는 많은 차로 수가 있는 도로가 대체로 교통량이 많은 중심부에 위치해 있으며, 신호등, 교차로, 보행자 등의 요소로 인해 평균 속도가 낮게 나올 수 있기 때문
- 반대로 차로 수가 적은 도로는 외곽이나 한적한 구간일 가능성이 높아 제한 요소가 적음
""")

# 시간대별 시계열 평균 속도
st.subheader("4. 시간대별 평균 속도 변화")
if "base_hour" in train.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=train, x="base_hour", y="target", ci=None, ax=ax)
    ax.set_title("시간대(base_hour)별 평균 속도")
    ax.set_xlabel("시간대 (0~23시)")
    ax.set_ylabel("평균 속도")
    st.pyplot(fig)

    st.markdown("""
        - 17~18시 사이에 속도가 가장 낮으며 출퇴근 시간의 정체를 반영함
        - 일반적으로 오전 7-9시, 오후 5-7시는 도로에 차량이 집중되며, 이러한 피크 시간대에 속도 저하가 두드러짐
        - 반면, 새벽 시간대(0~5시)는 차량이 거의 없어 상대적으로 평균 속도가 높은 패턴을 보임
        """)
    
# 계절별 평균 속도
st.subheader("5. 계절(season)에 따른 평균 속도")
season_map = {0: 'spring', 1: 'summer', 2: 'fall', 3: 'winter'}
train['season_label'] = train['season'].map(season_map)

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=train, x="season_label", y="target", ax=ax)
st.pyplot(fig)

st.markdown("""
- 여름 시즌(7~8월)이 가장 낮은 평균 속도를 보임, 이는 관광객 유입과 관련될 수 있음
- 특히 제주 지역의 경우 여름철 관광 성수기에 교통량이 급증하며, 도로 정체가 발생하는 구간이 늘어남
- 반면, 봄과 가을은 기후가 쾌적하면서도 교통량이 상대적으로 분산되어 속도가 높은 경향을 보임
""")

# 시간대 유형별 평균 속도
st.subheader("6. 시간대 유형(time_type)에 따른 평균 속도")

# time_type 파생
train['time_type'] = train['base_hour'].apply(lambda x:
    'commute' if x in [7,8,17,18] else (
    'late_night' if x in [0,1,2,3,4,5] else 'normal'))

time_map = {
    'commute': '출퇴근시간',
    'late_night': '심야시간',
    'normal': '일반시간'
}
train['time_type_label'] = train['time_type'].map(time_map)

order = ['출퇴근시간', '일반시간', '심야시간']
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=train, x="time_type_label", y="target", order=order, ax=ax)
st.pyplot(fig)

st.markdown("""
- 출퇴근 시간대 속도가 가장 느림 → 혼잡시간대임을 알 수 있음
- 심야시간에는 교통량이 거의 없어 속도가 빠르며, 일반시간은 두 그룹의 중간 정도 속도로 나타남
- 이 결과는 실제 혼잡 시간대를 반영하여 교통 정책 수립이나 도로 최적 운영 시간 분석 등에 활용될 수 있음
""")