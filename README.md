# Jeju Traffic Predictor
> 제주도 도로 교통 데이터를 기반으로 평균 주행 속도를 예측하고, 다양한 머신러닝 모델 성능 비교 및 해석 가능한 분석을 수행한 프로젝트

## 📁 프로젝트 구조

```
jeju-traffic-predictor/
│
├── data/                  # 원본 데이터 (train.csv, test.csv)
├── output/                # 전처리 및 모델 예측 결과 파일
│   ├── X_train.pkl        # 학습 데이터
│   ├── y_train.pkl
│   ├── X_val.pkl          # 검증 데이터
│   ├── y_val.pkl
│   ├── test_X.pkl         # 테스트 데이터
│   ├── rf_model.pkl       # 학습된 RandomForest 모델
│   ├── final_prediction_rf.csv        # RF 예측 결과
│   ├── final_prediction_baseline.csv # 평균 예측 baseline 결과
│   └── feature_columns.pkl            # 사용된 최종 feature 리스트
│
├── notebooks/
│   ├── 01_data_overview.ipynb         # 데이터 로드 및 기초 통계 확인
│   ├── 02_feature_engineering.ipynb   # 결승치 처리 및 파생 변수 생성
│   ├── 03_eda_analysis.ipynb          # 변수별 시각화 및 패턴 탐색
│   ├── 04_model_training.ipynb        # 다양한 모델 학습 및 성능 비교
│   └── 05_model_interpretation.ipynb  # Permutation / SHAP 기반 모델 해석
```

---

## 진행 내용 요약

### 01. 데이터 개요 및 정제
- train.csv/test.csv 불러오기 및 자료형 최적화 (`int32`, `float32`)
- 결측치 확인 → 'road_name' 컬럼 다단계 기준으로 보간
- target 분포 및 기초 통계 확인

### 02. 피처 엔지니어링
- 날짜 기반 파생 변수: 연도, 월, 요일, 계절, 시간대, 출퇴근 여부 등 생성
- 불필요 컬럼 제거: `vehicle_restricted`, `height_restricted` 등
- 범주형 라벨 인코딩 (start/end_node_name, road_name 등)

### 03. 시각화 및 EDA
- 상관관계 히트맵
- 주요 변수별 평균속도 boxplot 시각화
- season, 시간대(time_type) 기반 target 변화 확인

### 04. 모델 학습 및 평가
- 사용 모델: XGBoost, LightGBM, RandomForest, ExtraTrees
- 평가지표: MAE (Mean Absolute Error)
- 결과:
  - **RandomForest MAE**: 2.9420 (최고 성능)
  - **Baseline 평균 예측 모델**: 약 13
  - 약 **77% 이상 성능 향상**

### 05. 모델 해석
- Permutation Importance 기반 변수 중요도 분석
- SHAP 기반 주요 피처 해석 및 영향 시각화

---

## 예측 결과
- `final_prediction_rf.csv`: 최종 RandomForest 모델 예측 결과
- `final_prediction_baseline.csv`: 평균값 기반 Baseline 예측 결과
- `rf_model.pkl`: Streamlit 등 배포용 모델 저장

---

## 사용 기술
- Python, Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn, XGBoost, LightGBM, SHAP

---

