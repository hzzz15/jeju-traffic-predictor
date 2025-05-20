import streamlit as st
from utils import predict_speed_api

# 페이지 설정
st.set_page_config(page_title="제주 도로 속도 예측기", layout="centered")

# 타이틀
st.title("제주 도로 평균 속도 예측기")
st.markdown("""
- 도로명, 요일, 시간대, 계절, 7~9월 여부를 입력하면 평균 속도를 예측합니다.
- 예측 결과는 교통 운영 정책 수립이나 관광 시기 분석에 활용할 수 있습니다.
""")

# 사용자 입력
road_list = [
    '경찰로', '고평교', '관광단지1로', '관광단지2로', 
    '관광단지로', '관덕로', '권학로', '기타', '남조로', 
    '동문로', '동부관광도로', '동홍로', '번영로', '산서로', 
    '삼무로', '삼봉로', '삼성로', '새서귀로', '서사로', 
    '수영장길', '시민광장로', '시청로', '신광로', '신대로', 
    '신산로', '아봉로', '애원로', '애조로', '어시천교', 
    '연동로', '연북2교', '연북로', '연삼로', '외도천교', 
    '일반국도11호선', '일반국도12호선', '일반국도16호선', 
    '일반국도95호선', '일반국도99호선', '일주동로', '임항로', 
    '제2거로교', '중문로', '중산간서로', '중앙로', '중정로', 
    '지방도1112호선', '지방도1115호선', '지방도1116호선', 
    '지방도1117호선', '지방도1118호선', '지방도1119호선', 
    '지방도1120호선', '지방도1132호선', '지방도1136호선', 
    '지방도97호선', '첨단로', '태평로', '한천로', '호근로', '호서중앙로'
]
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
