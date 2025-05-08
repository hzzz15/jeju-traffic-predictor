import os
import pandas as pd
import joblib

def load_model():
    base     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    model    = joblib.load(os.path.join(base, 'rf_model.pkl'))
    features = joblib.load(os.path.join(base, 'feature_columns.pkl'))
    return model, features

def safe_transform(le, vals, default=0):
    out = []
    for v in vals:
        try:
            out.append(int(le.transform([v])[0]))
        except:
            out.append(default)
    return out

def preprocess_input(user_input: dict) -> pd.DataFrame:
    df = pd.DataFrame([user_input])

    # 시간대 구분 --> 업무시간(1)/비업무시간(0)
    df['시간대'] = df['time_type'].apply(lambda x: 1 if x == 'worktime' else 0)

    # 계절 --> 0:봄,1:여름,2:가을,3:겨울
    season_map = {'봄': 0, '여름': 1, '가을': 2, '겨울': 3}
    df['season_code'] = df['season'].map(season_map).fillna(0).astype(int)

    # 7~9월 여부 --> 1/0 매핑
    august_map = {'Y': 1, 'N': 0}
    df['august_code'] = df['adjacent_august'].map(august_map).fillna(0).astype(int)


    # 개별 인코더 로드 
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    le_road   = joblib.load(os.path.join(base, 'le_road.pkl'))
    le_time   = joblib.load(os.path.join(base, 'le_시간대.pkl'))
    le_season = joblib.load(os.path.join(base, 'le_season.pkl'))
    le_august = joblib.load(os.path.join(base, 'le_adjacent_august.pkl'))

    # transform 수행
    df['road_name']       = safe_transform(le_road, df['road_name'])
    df['시간대']           = safe_transform(le_time, df['시간대'])
    df['season']          = safe_transform(le_season, df['season_code'])
    df['adjacent_august'] = safe_transform(le_august, df['august_code'])

    # 최종 피처만 반환
    return df[['road_name', '시간대', 'season', 'adjacent_august']].copy()

def predict_speed(input_df: pd.DataFrame) -> float:
    model, features = load_model()
    # 누락된 feature는 0으로 채움
    for c in features:
        if c not in input_df.columns:
            input_df[c] = 0
    # 컬럼 순서 맞추기
    input_df = input_df[features]
    return float(model.predict(input_df)[0])