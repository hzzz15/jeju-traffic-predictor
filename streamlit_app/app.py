import os
import joblib
import streamlit as st
from utils import preprocess_input, predict_speed

st.set_page_config(page_title="제주 도로 속도 예측기", layout="centered")

# 설명
st.title("제주 도로 평균 속도 예측기")
st.markdown("""
- 도로명, 요일, 시간대 구분(업무시간/비업무시간), 계절, 7~9월 여부를 입력하면 **예상 평균 속도**를 예측합니다.
- 예측 결과는 **관광객 유입 분석**, **혼잡도 예측**, **도로 운영 최적화** 등에 활용될 수 있습니다.
""")

# 레이블 인코더에서 도로명 리스트 동적 로딩
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
encoders = joblib.load(os.path.join(base_path, 'label_encoders.pkl'))
road_list = encoders['le_road'].classes_.tolist()

# 사용자 입력 옵션
weekday_list = ['월', '화', '수', '목', '금', '토', '일']
season_list  = ['봄', '여름', '가을', '겨울']
august_list  = ['Y', 'N']
time_options = ['업무시간 (08:00~20:00)', '비업무시간 (나머지)']

# 입력 폼
st.subheader("예측 조건 입력")
road     = st.selectbox("도로명", road_list)
weekday  = st.selectbox("요일", weekday_list)
time_sel = st.selectbox("시간대 구분", time_options)
season   = st.selectbox("계절", season_list)
august   = st.selectbox("7~9월 여부", august_list)

# 예측 실행
if st.button("평균 속도 예측"):
    user_input = {
        "road_name":       road,
        "weekday":         weekday,
        "time_type":       "worktime" if time_sel.startswith("업무시간") else "resttime",
        "season":          season,
        "adjacent_august": august
    }

    # 전처리 및 DataFrame 생성
    df_enc = preprocess_input(user_input)

    # 터미널 로그: 인코딩 결과 확인
    print("[DEBUG] Encoded input:", df_enc.iloc[0].to_dict())

    # 예측
    try:
        pred = predict_speed(df_enc)
        st.success(f"예측된 평균 속도는 **{pred:.2f} km/h** 입니다.")
    except Exception as e:
        st.error(f"예측 중 오류 발생: {e}")