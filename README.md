# SCM 다품목 재고 최적화 프로젝트

## 프로젝트 개요
- **목표**: 다품목 SCM 환경에서 최적 발주량, 최소/최대 재고 정책, 발주 주기 최적화
- **전략**: 예정 발주(85%) + 긴급 발주(15%) 병행
- **확장성**: 현재 20개 품목 → 향후 5,000개 품목까지 확대 가능
- **발주 단위 단순화**: 100 단위 / 1000 단위 (운영 효율성 확보)

## 데이터
- **기간**: 총 24개월 데이터
- **학습/검증 분할**: 18개월 학습용 + 6개월 검증용
- **구성**: 출고 데이터, 입고 데이터, 전처리 데이터셋

## 모델링 접근
- **수요 예측**: Exponential Smoothing, ARIMA, Prophet, LSTM(선택)
- **최적화 정책**: Min-Max 재고 정책, EOQ, 안전재고 계산
- **강화학습**: DQN / PPO 기반 비용 최적화 (Advanced)

## KPI (성공 지표)
- 수요 예측 정확도: MAPE < 15%
- 비용 절감율: ≥ 15%
- 출고 충족률: ≥ 99%
- 긴급 발주율: 15% ± 2%

## 기술 스택
- **Python**: 3.11
- **데이터 처리**: Pandas, NumPy
- **시계열 예측**: StatsModels, Prophet, TensorFlow/Keras
- **최적화**: SciPy, Pyomo
- **강화학습**: OpenAI Gym, Stable-Baselines3
- **시각화/대시보드**: Matplotlib, Plotly, Dash
- **배포 환경**: Flask/FastAPI, Docker

## 프로젝트 일정 (예시)
> ⚠️ 아직 확정되지 않은 부분은 주석 처리했습니다.

<!--
- Phase 1 (Week 1-2): 데이터 전처리 및 품목 분류
- Phase 2 (Week 3-4): 수요 예측 모델 개발 및 검증
- Phase 3 (Week 5): Min-Max 정책 수립, 발주 추천 시스템
- Phase 4 (Week 6): 강화학습 기반 비용 최적화 및 보고서 작성
-->

## 산출물
- Forecasted_Demand.csv (예측 결과)
- Optimal_Inventory_Policy.csv (최적 재고 정책)
- Cost_Analysis_Report.xlsx (비용 분석)
- Dashboard.html (실시간 모니터링 대시보드)
- Implementation_Guide.pdf (시스템 도입 가이드)

## 사용 방법
```bash
# 환경 세팅
python3.11 -m venv venv
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt

# 데이터 전처리 실행
python 01_data_preprocessing.py

# 수요 예측 실행
python 04_demand_forecast_arima.py
