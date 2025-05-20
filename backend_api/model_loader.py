import joblib
import pandas as pd
import os
import requests
import io

# Hugging Face 파일 URL
HF_MODEL_URL = "https://huggingface.co/hzz15/jeju-traffic-files/resolve/main/rf_model.pkl"
HF_FEATURES_URL = "https://huggingface.co/hzz15/jeju-traffic-files/resolve/main/feature_columns.pkl"

# 로컬 인코더 경로
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
    # 모델 로드 (Hugging Face)
    model_resp = requests.get(HF_MODEL_URL)
    model = joblib.load(io.BytesIO(model_resp.content))

    # feature 리스트 로드 (Hugging Face)
    feat_resp = requests.get(HF_FEATURES_URL)
    features = joblib.load(io.BytesIO(feat_resp.content))

    return {
        "model": model,
        "features": features,
        "le_road": joblib.load(os.path.join(BASE, 'le_road.pkl')),
        "le_time": joblib.load(os.path.join(BASE, 'le_시간대.pkl')),
        "le_season": joblib.load(os.path.join(BASE, 'le_season.pkl')),
        "le_august": joblib.load(os.path.join(BASE, 'le_adjacent_august.pkl')),
    }

def preprocess_input(data: dict, encoders: dict):
    df = pd.DataFrame([data])

    # 파생 변수 및 인코딩
    df['time_code'] = df['time_type'].apply(lambda x: 1 if x == 'worktime' else 0)
    season_map = {'봄': 0, '여름': 1, '가을': 2, '겨울': 3}
    df['season_code'] = df['season'].map(season_map).fillna(0).astype(int)
    august_map = {'Y': 1, 'N': 0}
    df['august_code'] = df['adjacent_august'].map(august_map).fillna(0).astype(int)

    # 인코딩된 컬럼
    df['road_name'] = safe_transform(encoders['le_road'], df['road_name'])
    df['시간대'] = safe_transform(encoders['le_time'], df['time_code'])
    df['season'] = safe_transform(encoders['le_season'], df['season_code'])
    df['adjacent_august'] = safe_transform(encoders['le_august'], df['august_code'])

    return df[['road_name', '시간대', 'season', 'adjacent_august']]
