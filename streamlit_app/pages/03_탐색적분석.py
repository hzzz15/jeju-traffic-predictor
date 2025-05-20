import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import requests
from io import BytesIO
import platform

# 시각화 설정
sns.set_style("whitegrid")
plt.rcParams['axes.unicode_minus'] = False

# 플랫폼별 폰트 설정
if platform.system() == "Darwin":  # macOS
    plt.rcParams["font.family"] = "AppleGothic"
else:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.sans-serif"] = ["NanumGothic", "Arial", "sans-serif"]

# 컬럼명 영어 매핑
column_map = {
    "시간대": "time_type",
}

# 제목
st.title("탐색적 분석 (EDA) 리포트")

# Hugging Face에서 joblib 파일 불러오기
@st.cache_data
def load_from_huggingface(filename):
    import mimetypes

    url = f"https://huggingface.co/hzz15/jeju-traffic-files/resolve/main/{filename}?download=true"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "text" in content_type or "html" in content_type:
            st.error(f"{filename} 응답이 텍스트입니다. 파일이 존재하지 않거나 링크가 잘못되었습니다.")
            st.stop()

        # 바이너리 확인: 파일 크기가 너무 작아도 의심
        if len(response.content) < 1000:
            st.error(f"{filename} 파일 크기가 비정상적으로 작습니다. 업로드 상태를 확인하세요.")
            st.stop()

        return joblib.load(BytesIO(response.content))

    except Exception as e:
        st.error(f"{filename} 로드 중 오류 발생: {e}")
        st.stop()

# 데이터 불러오기
X_train = load_from_huggingface('X_train_sample.pkl')
y_train = load_from_huggingface('y_train.pkl')

train = X_train.copy()
train['target'] = y_train

# 1. 수치형 변수 간 상관관계 히트맵
st.subheader("1. 수치형 변수 간 상관관계")
numeric_cols = train.select_dtypes(include=["number"])
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)
st.markdown("""
- `maximum_speed_limit`(제한속도)와 `target`(평균속도)이 가장 높은 양의 상관관계를 보임
- `weight_restricted`, `road_type`도 양의 상관관계를 보이며 도로 조건이 속도에 영향을 줌
- 반대로 `road_rating`, `lane_count`는 약한 음의 상관관계를 보임
""")

# 2. 도로 등급별 평균 속도
st.subheader("2. 도로 등급별 평균 속도")
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=train, x="road_rating", y="target", ax=ax)
st.pyplot(fig)
st.markdown("""
- 도로 등급이 낮을수록 평균 속도는 다소 높아지는 경향을 보임
- 이는 높은 등급의 도로가 도심에 많아 신호, 교차로 영향 때문일 수 있음
""")

# 3. 차로 수별 평균 속도
st.subheader("3. 차로 수별 평균 속도")
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=train, x="lane_count", y="target", ax=ax)
st.pyplot(fig)
st.markdown("""
- 차로 수가 많다고 반드시 속도가 빠르진 않음
- 도심일수록 차로가 많지만 혼잡도가 높을 수 있음
""")

# 4. 시간대별 평균 속도 변화
st.subheader("4. 시간대별 평균 속도 변화")
if "base_hour" in train.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=train, x="base_hour", y="target", ci=None, ax=ax)
    ax.set_title("Average Speed by Hour")
    ax.set_xlabel("Hour (0~23)")
    ax.set_ylabel("Average Speed")
    st.pyplot(fig)
    st.markdown("""
    - 17~18시 속도가 가장 낮으며 퇴근 시간 정체 반영
    - 새벽(0~5시) 시간대는 평균 속도 높음
    """)


# 5. 계절별 평균 속도
st.subheader("5. 계절별 평균 속도")
season_map = {0: 'Spring', 1: 'Summer', 2: 'Fall', 3: 'Winter'}
train['season_label'] = train['season'].map(season_map)
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=train, x="season_label", y="target", ax=ax)
st.pyplot(fig)
st.markdown("""
- 여름(7~8월) 평균 속도 가장 낮음 → 관광객 유입 영향
- 봄/가을은 교통 분산으로 상대적으로 빠름
""")

# 6. 시간대 유형별 평균 속도
st.subheader("6. 시간대 유형별 평균 속도")
train['time_type'] = train['base_hour'].apply(lambda x:
    'Commute' if x in [7,8,17,18] else (
    'Late Night' if x in [0,1,2,3,4,5] else 'Normal'))
order = ['Commute', 'Normal', 'Late Night']
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=train, x="time_type", y="target", order=order, ax=ax)
st.pyplot(fig)
st.markdown("""
- 출퇴근 시간 속도가 가장 느림
- 심야시간은 가장 빠름 → 교통량 적음
""")