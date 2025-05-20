import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import requests
from io import BytesIO
import platform

sns.set_style("whitegrid")
plt.rcParams['axes.unicode_minus'] = False

# 환경별 폰트 설정
if platform.system() == "Darwin": 
    plt.rcParams["font.family"] = "AppleGothic"
else:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.sans-serif"] = ["NanumGothic", "Arial", "sans-serif"]

# 제목
st.title("탐색적 분석 (EDA) 리포트")

# Hugging Face에서 파일 로드
@st.cache_data
def load_from_huggingface(filename):
    url = f"https://huggingface.co/hzz15/jeju-traffic-files/resolve/main/{filename}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return joblib.load(BytesIO(response.content))
    except Exception as e:
        st.error(f"HuggingFace 파일 로드 실패: {e}")
        return None

X_train = load_from_huggingface('X_train.pkl')
y_train = load_from_huggingface('y_train.pkl')

if X_train is None or y_train is None:
    st.stop()  

train = X_train.copy()
train['target'] = y_train

# 1. 상관관계 히트맵
st.subheader("1. 수치형 변수 간 상관관계")
numeric_cols = train.select_dtypes(include=["number"])
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

# 2. 도로 등급별 평균 속도
st.subheader("2. 도로 등급별 평균 속도")
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=train, x="road_rating", y="target", ax=ax)
st.pyplot(fig)

# 3. 차로 수별 평균 속도 
st.subheader("3. 차로 수별 평균 속도")
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=train, x="lane_count", y="target", ax=ax)
st.pyplot(fig)

# 4. 시간대별 평균 속도 변화
st.subheader("4. 시간대별 평균 속도 변화")
if "base_hour" in train.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=train, x="base_hour", y="target", ci=None, ax=ax)
    ax.set_title("Average Speed by Hour")
    ax.set_xlabel("Hour (0~23)")
    ax.set_ylabel("Average Speed")
    st.pyplot(fig)

# 5. 계절별 평균 속도 
st.subheader("5. 계절별 평균 속도")
season_map = {0: 'Spring', 1: 'Summer', 2: 'Fall', 3: 'Winter'}
train['season_label'] = train['season'].map(season_map)

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=train, x="season_label", y="target", ax=ax)
st.pyplot(fig)

# 6. 시간대 유형별 평균 속도 
st.subheader("6. 시간대 유형별 평균 속도")

train['time_type'] = train['base_hour'].apply(lambda x:
    'Commute' if x in [7,8,17,18] else (
    'Late Night' if x in [0,1,2,3,4,5] else 'Normal'))

order = ['Commute', 'Normal', 'Late Night']
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=train, x="time_type", y="target", order=order, ax=ax)
st.pyplot(fig)
