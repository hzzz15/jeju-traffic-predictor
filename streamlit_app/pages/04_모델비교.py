import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.family"] = "AppleGothic" 

st.set_page_config(page_title="모델 성능 비교", layout="wide")
st.title("모델 성능 비교 리포트")

# 모델별 MAE 점수
mae_scores = {
    "XGBoost": 3.0684,
    "LightGBM": 3.6500,
    "RandomForest": 2.9420,
    "ExtraTrees": 2.9634,
}
baseline_mae = 13.2307

df_mae = pd.DataFrame(mae_scores.items(), columns=["모델", "MAE"]).sort_values(by="MAE")
best_model = df_mae.iloc[0]

# 표
st.subheader("모델별 MAE 비교")
st.dataframe(df_mae, use_container_width=True)

# 바 차트
st.subheader("성능 시각화")
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(df_mae["모델"], df_mae["MAE"], color="skyblue")
ax.axhline(baseline_mae, color="red", linestyle="--", label="Baseline")
ax.set_ylabel("Mean Absolute Error")
ax.set_title("모델별 MAE 비교")
ax.legend()
st.pyplot(fig)

# 최종 모델 및 개선율
improvement = (baseline_mae - best_model["MAE"]) / baseline_mae * 100
st.success(f"최종 모델: **{best_model['모델']}** / MAE: {best_model['MAE']:.4f}")
st.markdown(f"**Baseline 대비 성능 개선율**: `{improvement:.2f}%`")
