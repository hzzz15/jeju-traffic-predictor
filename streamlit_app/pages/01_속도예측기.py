import streamlit as st
from utils import predict_speed_api

# 페이지 설정
st.set_page_config(page_title="제주 도로 속도 예측기", layout="centered")

# 타이틀
st.title("🚗 제주 도로 평균 속도 예측기")
st.markdown("""
- 도로명, 요일, 시간대, 계절, 7~9월 여부를 입력하면 평균 속도를 예측합니다.
- 예측 결과는 교통 운영 정책 수립이나 관광 시기 분석에 활용할 수 있습니다.
""")

# 사용자 입력
road_list = ['1100도로', '1132번지방도', '남조로', '서귀포시내', '번영로']  # le_road.classes_ 참고해서 고정해도 OK
weekday_list = ['월', '화', '수', '목', '금', '토', '일']
season_list  = ['봄', '여름', '가을', '겨울']
august_list  = ['Y', 'N']
time_options = ['08~20시', '20~08시']

st.subheader("입력 조건 선택")
road     = st.selectbox("도로명", road_list)
weekday  = st.selectbox("요일", weekday_list)
time_sel = st.selectbox("시간대", time_options)
season   = st.selectbox("계절", season_list)
august   = st.selectbox("7~9월 여부", august_list)

# 예측 요청
if st.button("평균 속도 예측"):
    user_input = {
        "road_name":       road,
        "weekday":         weekday,
        "time_type":       "worktime" if time_sel == '08~20시' else "resttime",
        "season":          season,
        "adjacent_august": august
    }

    try:
        pred = predict_speed_api(user_input)
        st.success(f"예측된 평균 속도는 **{pred:.2f} km/h** 입니다.")
    except Exception as e:
        st.error(f"예측 요청 중 오류 발생: {e}")
