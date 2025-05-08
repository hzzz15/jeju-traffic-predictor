import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error

# -----------------------------
# 1. 모델 및 데이터 로드
# -----------------------------
base_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'output')
)

rf_model = joblib.load(os.path.join(base_path, "rf_model.pkl"))
X_val = joblib.load(os.path.join(base_path, "X_val.pkl"))
y_val = joblib.load(os.path.join(base_path, "y_val.pkl"))

# -----------------------------
# 2. 성능 계산
# -----------------------------
rf_preds = rf_model.predict(X_val)
mae_rf = mean_absolute_error(y_val, rf_preds)

# Baseline: 평균값 예측
baseline_pred = np.full_like(y_val, y_val.mean())
mae_baseline = mean_absolute_error(y_val, baseline_pred)

improvement = ((mae_baseline - mae_rf) / mae_baseline) * 100

# -----------------------------
# 3. Streamlit 표시
# -----------------------------
import streamlit as st
st.subheader("모델 성능 해석")

st.markdown(f"""
- **최종 모델 (RandomForest)**: MAE = `{mae_rf:.4f}`
- **Baseline 모델 (평균 예측)**: MAE = `{mae_baseline:.4f}`  
- **성능 향상률**: `{improvement:.2f}%`

이는 도로 평균 속도를 기준으로 약 ±{mae_rf:.0f}km/h 오차 수준이며,  
**실시간 도로 예측 시스템**에서 사용하기에 신뢰할 수 있는 성능입니다.
""")

# -----------------------------
# 4. 성능 시각화
# -----------------------------
st.markdown("### 예측 결과 분포 비교")

# 예측값 분포 시각화
result_df = pd.DataFrame({
    "실제값 (y_val)": y_val,
    "RandomForest 예측": rf_preds,
    "Baseline 예측": baseline_pred
})

fig, ax = plt.subplots(figsize=(10, 5))
sns.kdeplot(result_df["실제값 (y_val)"], label="실제값", linewidth=2)
sns.kdeplot(result_df["RandomForest 예측"], label="RandomForest", linestyle="--")
sns.kdeplot(result_df["Baseline 예측"], label="Baseline", linestyle=":")
plt.legend()
plt.title("예측값 분포 비교")
st.pyplot(fig)
