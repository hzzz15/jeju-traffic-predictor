## 프로젝트 개요

제주도는 인구 증가와 관광객 유입으로 인해 **교통 체증이 점점 심각한 사회 문제**로 떠오르고 있습니다.  
본 프로젝트는 **제주도 도로의 평균 차량 속도를 예측하는 회귀 모델**을 비교 분석함으로써,  **교통 정책 수립 및 실시간 예측 모델 개발의 기초 자료 확보**를 목표로 합니다.

- **데이터 규모**: 학습 데이터 약 470만 건, 테스트 데이터 약 29만 건
- **목표**: 다양한 머신러닝 회귀 모델을 활용해 `target(평균 속도)` 예측 성능 비교
- **활용 모델**: XGBoost, LightGBM, RandomForest, ExtraTrees
- **성과 지표**: MAE (Mean Absolute Error)

---

## 📂 프로젝트 구조
```bash
📁 jeju-traffic-prediction/ 
┗ 📄 제주도 교통량 회귀 예측.ipynb
```
- `제주도 교통량 회귀 예측.ipynb`  
  : 데이터 로딩, 전처리, EDA, 파생 변수 생성, 모델 학습 및 비교까지 **전체 분석 과정을 포함**한 단일 노트북 파일입니다.

---

## 🛠️ 기술 스택

| 구분 | 사용 기술 |
|------|------------|
| **언어** | Python 3.9 |
| **데이터 처리** | pandas, numpy |
| **시각화 도구** | matplotlib, seaborn |
| **모델링 및 평가** | scikit-learn, XGBoost, LightGBM |
| **교차검증** | StratifiedKFold (3-Fold) |
| **개발 환경** | Jupyter Notebook (ipynb) |

---

## 📖 라이선스 및 참고자료 
- 데이터 출처: [데이콘 제주도 도로 교통량 예측 경진대회](https://dacon.io/competitions/official/235985/overview/description)

