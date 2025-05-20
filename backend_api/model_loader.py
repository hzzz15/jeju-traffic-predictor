import joblib
import os
import pandas as pd
import requests
import io

# Hugging Face 모델 파일 URL 
HF_MODEL_URL = "https://huggingface.co/hzz15/jeju-traffic-files/resolve/main/rf_model_light.pkl"

# 로컬 인코더 저장 경로
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "model_artifacts"))

def safe_transform(le, vals, default=0):
    out = []
    for v in vals:
        try:
            out.append(int(le.transform([v])[0]))
        except:
            out.append(default)
    return out

def load_artifacts():
    # 1️. 모델은 Hugging Face에서 로드
    response = requests.get(HF_MODEL_URL)
    model = joblib.load(io.BytesIO(response.content))

    # 2️. 인코더와 feature_columns는 로컬에서 로드
    return {
        "model": model,
        "features": joblib.load(os.path.join(BASE, 'feature_columns.pkl')),
        "le_road": joblib.load(os.path.join(BASE, 'le_road.pkl')),
        "le_time": joblib.load(os.path.join(BASE, 'le_시간대.pkl')),
        "le_season": joblib.load(os.path.join(BASE, 'le_season.pkl')),
        "le_august": joblib.load(os.path.join(BASE, 'le_adjacent_august.pkl')),
    }

def preprocess_input(data: dict, encoders: dict):
    df = pd.DataFrame([data])

    # 디버깅 로그 추가
    print("[DEBUG] 원본 입력:", df)

    # 시간대 숫자화
    df['time_code'] = df['time_type'].apply(lambda x: 1 if x == 'worktime' else 0)

    # 계절 숫자화
    season_map = {'봄': 0, '여름': 1, '가을': 2, '겨울': 3}
    df['season_code'] = df['season'].map(season_map).fillna(0).astype(int)

    # 7~9월 여부 숫자화
    august_map = {'Y': 1, 'N': 0}
    df['august_code'] = df['adjacent_august'].map(august_map).fillna(0).astype(int)

    # 인코딩
    df['road_name'] = safe_transform(encoders['le_road'], df['road_name'])
    df['시간대'] = safe_transform(encoders['le_time'], df['time_code'])
    df['season'] = safe_transform(encoders['le_season'], df['season_code'])
    df['adjacent_august'] = safe_transform(encoders['le_august'], df['august_code'])

    print("[DEBUG] 인코딩된 최종 DF:", df[['road_name', '시간대', 'season', 'adjacent_august']])

    return df[['road_name', '시간대', 'season', 'adjacent_august']]

