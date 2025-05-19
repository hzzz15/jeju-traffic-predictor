# Jeju Traffic Predictor
> 제주도 도로 교통 데이터를 기반으로 평균 주행 속도를 예측하고, 다양한 머신러닝 모델 성능 비교 및 해석 가능한 분석을 수행한 프로젝트

---

## 프로젝트 구조

```
jeju-traffic-predictor/
│
├── data/                # 원본 데이터 (train.csv, test.csv)
├── output/              # 예측 결과 및 저장된 모델
├── notebooks/
│   ├── 01_data_overview.ipynb        # 데이터 로드 및 기초 통계 확인
│   ├── 02_feature_engineering.ipynb  # 결측치 처리 및 파생 변수 생성
│   ├── 03_eda_analysis.ipynb         # 변수별 시각화 및 패턴 탐색
│   ├── 04_model_training.ipynb       # 다양한 모델 학습 및 성능 비교
│   └── 05_model_interpretation.ipynb  # Permutation / SHAP 기반 해석
├── streamlit_app/       # Streamlit 웹 앱 소스
└── backend_api/         # FastAPI 서버 코드
```
---

## 진행 내용 요약

### 01. 데이터 개요 및 정제

* train.csv/test.csv 불러오기 및 자료형 최적화 (`int32`, `float32`)
* 결측치 확인 → 'road\_name' 컬럼 다단계 기준으로 보간
* target 분포 및 기초 통계 확인

### 02. 피처 엔지니어링

* 날짜 기반 파생 변수: 연도, 월, 요일, 계절, 시간대, 출퇴근 여부 등 생성
* 불필요 컬럼 제거: `vehicle_restricted`, `height_restricted` 등
* 범주형 라벨 인코딩 (start/end\_node\_name, road\_name 등)

### 03. 시각화 및 EDA

* 상관관계 히트맵
* 주요 변수별 평균속도 boxplot 시각화
* season, 시간대(time\_type) 기반 target 변화 확인

### 04. 모델 학습 및 평가

* 사용 모델: XGBoost, LightGBM, RandomForest, ExtraTrees
* 평가지표: MAE (Mean Absolute Error)
* 결과:

  * **RandomForest MAE**: 2.9420 (최고 성능)
  * **Baseline 평균 예측 모델**: 약 13
  * 약 **77% 이상 성능 향상**

### 05. 모델 해석

* Permutation Importance 기반 변수 중요도 분석
* SHAP 기반 주요 피처 해석 및 영향 시각화

---

## 기술 스택
* 언어: Python 3.x
* EDA & 전처리: pandas, numpy, matplotlib, seaborn
* 모델링: scikit-learn, XGBoost, LightGBM
* 모델 해석: SHAP, permutation\_importance
* 웹 서비스: Streamlit (프론트), FastAPI (백엔드)

---

## 배포 주소 및 모델 저장소
* **웹 예측 서비스**
  [https://jeju-traffic-predictor-74szjgtndyx8y5tvw4kekx.streamlit.app/](https://jeju-traffic-predictor-74szjgtndyx8y5tvw4kekx.streamlit.app/)
  → 실시간 도로/시간대 조건 입력 후 평균속도 예측 가능

* **모델 및 전처리 파일 저장소 (Hugging Face)**
  [https://huggingface.co/hzz15/jeju-traffic-files/tree/main](https://huggingface.co/hzz15/jeju-traffic-files/tree/main)
  → 학습된 RandomForest 모델, 라벨 인코더, 피처 목록 등 포함
