import os
import joblib
import shap
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform

sns.set_style("whitegrid")
plt.rcParams['axes.unicode_minus'] = False

# 폰트 설정
if platform.system() == "Darwin": 
    plt.rcParams["font.family"] = "AppleGothic"
else:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.sans-serif"] = ["NanumGothic", "Arial", "sans-serif"]

# 데이터 로딩
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'output'))

shap_sample_X = joblib.load(os.path.join(base_path, "shap_sample_X.pkl"))
shap_values = joblib.load(os.path.join(base_path, "shap_values.pkl"))

# SHAP Explanation 객체 생성
shap_exp = shap.Explanation(values=shap_values,
                            data=shap_sample_X,
                            feature_names=shap_sample_X.columns)

# Streamlit 내용
st.title("SHAP 기반 모델 해석 리포트")
st.markdown("""
이 보고서는 모델 예측 결과에 영향을 미친 주요 변수들을  
SHAP 분석을 통해 시각화하고 **도로별 변수 패턴과 영향력**을 확인합니다.
""")

# SHAP Summary Plot - 전체 영향력
st.markdown("### 전체 변수 영향력 (SHAP Summary Plot)")

fig1 = plt.figure()
shap.plots.beeswarm(shap_exp, show=False)
st.pyplot(fig1)

# 상위 5개 도로 기준 SHAP 영향도 분석
st.markdown("### 상위 5개 도로별 변수 영향력 분석")

top_5_roads = shap_sample_X['road_name'].value_counts().head(5).index.tolist()

for road in top_5_roads:
    st.markdown(f"#### 도로명: `{road}`")
    
    road_mask = shap_sample_X['road_name'] == road
    road_shap_values = shap_values[road_mask.to_numpy()]

    mean_importance = abs(road_shap_values).mean(axis=0)
    top_features = pd.DataFrame({
        'Feature': shap_sample_X.columns,
        'Mean |SHAP value|': mean_importance
    }).sort_values(by='Mean |SHAP value|', ascending=False).head(5)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=top_features, x='Mean |SHAP value|', y='Feature', ax=ax)
    ax.set_title(f"{road} 도로에서 가장 영향력 있는 변수들")
    st.pyplot(fig)

    top_feature = top_features.iloc[0]['Feature']
    st.markdown(f"""
    - `{road}` 도로에서는 **`{top_feature}`** 변수가 예측에 가장 큰 영향을 주었습니다.
    - 이는 해당 변수의 변화가 속도 예측 결과에 **직접적인 기여**를 한다는 의미입니다.
    - 도로 설계나 운영 전략 수립 시 이 변수에 주목할 필요가 있습니다.
    """)

st.markdown("### 전체 요약")

st.markdown("""
- 대부분 도로에서 `시간대`, `계절`, `adjacent_august` 등의 변수가 상위 영향 요인으로 나타났습니다.
- 이는 **시간적 요인**이 도로 평균 속도에 큰 영향을 미친다는 모델의 학습 결과를 반영합니다.
- 도로별 상위 변수를 분석해 **도로 맞춤형 혼잡도 대응 전략** 수립이 가능할 것으로 보입니다.
""")
