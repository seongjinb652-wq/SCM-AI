##
# SCM 다품목 재고 최적화 프로젝트
## 1. 프로젝트 개요 및 목표

### 1.1 프로젝트 배경
- **목표**: 다품목 SCM 환경에서 최적 발주량, 최소/최대 재고 정책, 발주 주기 최적화
- **전략**: 예정 발주(85%) + 긴급 발주(15%) 병행
- **확장성**: 현재 20개 품목 → 향후 5,000개 품목까지 확대 가능
- **현황**: 20개 품목의 수동 재고 관리 → 예측 기반 자동화 시스템 필요
- **확장성**: 향후 5,000개 품목 관리 대비 확장 가능한 프레임워크 구축
- **핵심 전략**: 예정 발주(85%) + 긴급 발주(15%) 병행으로 현실성과 안정성 확보
- **발주 단위 단순화**: 100 단위 / 1000 단위 (운영 효율성 확보)

### 1.2 주요 목표
1. **수요 예측 정확성 향상**: MAPE < 15% 달성
2. **재고 비용 최소화**: 보관비용 + 발주비용 + 긴급발주비용 총 15% 이상 절감
3. **출고 대응률**: 예정 발주로 85% 이상 커버, 긴급 발주로 15% 안정적 처리
4. **운영 효율성**: 발주량 단위화(100/1000단위), 자동화된 발주 의사결정 시스템 구축
5. **확장성**: 5,000개 품목까지 적용 가능한 스케일러블 정책 개발


#============================================= 요약 시작 =======================
## 프로젝트 개요
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



# 다품목 SCM 재고 최적화 프로젝트 계획서

**프로젝트명**: SCM 다품목 재고 정책 수립 및 최적화 시스템 구축  
**기간**: 2026년 1분기~2분기  
**대상 업체**: A사(연매출 300억, 창고 1개), B사(연매출 9,000억, 창고 5개)  
**관리 품목**: 현황 20개 → 향후 5,000개 (3가지 유형으로 단순화)

#============================================= 요약 끝 =======================
---

## 2. 데이터 구조 및 전처리 (완료 단계)

### 2.1 원본 데이터 및 전처리

| 구분 | 파일명 | 설명 | 상태 |
|------|--------|------|------|
| **원본 데이터** | `A_company_outbound_24months.csv` | A사 24개월 출고 데이터 (SKU, 일자, 수량, 가격) | ✓ 완료 |
| | `A_company_inbound_24months.csv` | A사 24개월 입고 데이터 (발주일, 입고일, 수량, 비용) | ✓ 완료 |
| | `B_company_outbound_24months.csv` | B사 24개월 출고 데이터 (SKU, 일자, 수량, 창고별) | ✓ 완료 |
| | `B_company_inbound_24months.csv` | B사 24개월 입고 데이터 (발주일, 입고일, 창고별) | ✓ 완료 |
| **전처리 완료** | `A_company_preprocessed.csv` | 결측치 제거, 이상치 처리, 날짜 정규화 | ✓ 완료 |
| | `B_company_preprocessed.csv` | 결측치 제거, 이상치 처리, 날짜 정규화 | ✓ 완료 |
| **분석용 데이터셋** | `A_company_train_18m.csv` | A사 학습용 (0~18개월) | ✓ 준비 |
| | `A_company_test_6m.csv` | A사 검증용 (18~24개월) | ✓ 준비 |
| | `B_company_train_18m.csv` | B사 학습용 (0~18개월) | ✓ 준비 |
| | `B_company_test_6m.csv` | B사 검증용 (18~24개월) | ✓ 준비 |

### 2.2 전처리 프로세스
```python
# 스크립트명: 01_data_preprocessing.py
- 결측치 처리 (선형 보간법)
- 이상치 탐지 (IQR 기준)
- 날짜/카테고리 정규화
- 학습/검증 데이터 분할
- 통계 분석 리포트 생성
```

---

## 3. 품목 분류 및 특성 분석

### 3.1 품목 유형 분류 (ABC 분석 + 변동성)

| 유형 | 특성 | 재고 전략 | 발주 주기 | 안전재고 |
|------|------|----------|----------|---------|
| **Type A (안정형)** | 안정적 수요, 낮은 변동성 | 정량 발주(EOQ) | 월 1회 | 평균 수요 5~7일분 |
| **Type B (변동형)** | 중간 수요, 계절성/프로모션 영향 | 수요-기반 동적 발주 | 2주~월 1회 | 평균 수요 10~14일분 |
| **Type C (불안정형)** | 불규칙 수요, 높은 변동성 | 예측-기반 + 긴급 발주 병행 | 주 1~2회 | 평균 수요 14~21일분 |

### 3.2 분류 분석 스크립트
```python
# 스크립트명: 02_product_classification.py
- SKU별 수요 통계 (평균, 표준편차, CV)
- ABC 분석 (매출액 기준)
- 변동성 지수(CV: Coefficient of Variation) 계산
- 품목 유형 자동 분류
- 분류 결과 레포트: 03_product_classification_report.csv
```

---

## 4. 수요 예측 모델링

### 4.1 예측 모델 구성

| 단계 | 모델 | 특성 | 대상 품목 유형 | 파일명 |
|------|------|------|----------------|--------|
| **1단계: 기본예측** | 지수평활(Exponential Smoothing) | 빠른 학습, 계산 효율 | Type A | `04_demand_forecast_es.py` |
| | ARIMA | 시계열 자기상관 반영 | Type A, B | `04_demand_forecast_arima.py` |
| | Prophet (Facebook) | 계절성/트렌드/프로모션 처리 | Type B, C | `04_demand_forecast_prophet.py` |
| **2단계: 앙상블** | 가중 평균 (Weighted Ensemble) | 3개 모델 최적 가중치 조합 | Type A, B, C | `04_demand_forecast_ensemble.py` |
| **3단계: 신경망** | LSTM (선택사항) | 복잡 패턴 학습 | Type C (불안정형) | `04_demand_forecast_lstm.py` |

### 4.2 모델 평가 지표
- **MAPE (Mean Absolute Percentage Error)**: 평균 오차율 (목표 < 15%)
- **RMSE (Root Mean Square Error)**: 편차 정도
- **MAE (Mean Absolute Error)**: 평균 절대값 오차
- **Coverage Rate**: 안전재고로 실제 수요 커버율

### 4.3 예측 결과 산출물
```
05_demand_forecast_results/
├── A_company_forecast_predictions.csv (월별 예측값, 신뢰구간)
├── B_company_forecast_predictions.csv
├── forecast_accuracy_report.csv (MAPE, RMSE 등)
├── forecast_visualization.png (시각화)
└── forecast_by_product_type.csv (유형별 성능 비교)
```

---

## 5. 최적 발주 정책 수립

### 5.1 Min-Max 재고 정책 (기본 틀)

#### 함수: `calculate_min_max_inventory()`
```python
def calculate_min_max_inventory(
    avg_demand,              # 평균 일일 수요
    std_dev_demand,          # 수요 표준편차
    lead_time,               # 리드타임 (일)
    service_level,           # 서비스 수준 (%)
    holding_cost,            # 보관 비용 (원/단위/일)
    order_cost,              # 발주 비용 (원/회)
    unit_cost                # 상품 단가 (원)
):
    """
    Min Stock = (평균수요 × 리드타임) + 안전재고
    Max Stock = Min Stock + 경제발주량(EOQ)
    Safety Stock = Z값 × 표준편차 × √(리드타임)
    """
    # 안전 계수(Z값): 서비스 수준에 따른 정규분포 Z값
    z_value = get_z_value(service_level)
    
    # 안전재고 = Z × σ × √L
    safety_stock = z_value * std_dev_demand * math.sqrt(lead_time)
    
    # 최소 재고 = 리드타임 수요 + 안전재고
    min_stock = (avg_demand * lead_time) + safety_stock
    
    # 경제발주량(EOQ) = √(2DS/H)
    eoq = math.sqrt((2 * avg_demand * 365 * order_cost) / (holding_cost * 365))
    
    # 최대 재고 = 최소 재고 + EOQ
    max_stock = min_stock + eoq
    
    return {
        'min_stock': round_to_lot_size(min_stock),
        'max_stock': round_to_lot_size(max_stock),
        'safety_stock': round_to_lot_size(safety_stock),
        'eoq': round_to_lot_size(eoq),
        'reorder_point': round_to_lot_size(min_stock)
    }
```

### 5.2 발주량 단위 단순화
```python
def round_to_lot_size(quantity, lot_sizes=[100, 1000]):
    """
    발주량을 100 또는 1000 단위로 반올림
    Type A: 100단위
    Type B: 100단위
    Type C: 50단위 (유연성 필요시)
    """
    if quantity <= 100:
        return max(100, (int(quantity / 100) + 1) * 100)
    else:
        return ((int(quantity / 1000) + 1) * 1000)
```

### 5.3 산출물 파일

| 파일명 | 내용 | 형식 |
|--------|------|------|
| `06_min_max_inventory_policy_A.csv` | A사 품목별 Min/Max 재고 정책 (20개 품목) | CSV |
| `06_min_max_inventory_policy_B.csv` | B사 품목별 Min/Max 재고 정책 (20개 품목) | CSV |
| `06_order_policy_detail.csv` | 품목별 발주 주기, 발주 수량, 안전재고 상세 | CSV |
| `06_reorder_point_calculation.csv` | 재주문 지점(ROP) 및 리드타임 고려 | CSV |

---

## 6. 긴급 발주 및 비용 최적화

### 6.1 긴급 발주 전략 (15% 커버)
```python
# 스크립트명: 07_emergency_order_optimization.py

def emergency_order_strategy(
    predicted_demand,        # 예측 수요
    current_inventory,       # 현재 재고
    min_stock,               # 최소 재고
    emergency_lead_time=3,   # 긴급 발주 리드타임 (일)
    emergency_cost_factor=1.5 # 긴급 발주 비용 배수
):
    """
    긴급 발주 조건:
    1. 현재 재고 < 최소 재고 AND 예측 수요 > 0
    2. 예상 부족량이 발생할 경우
    """
    shortage = max(0, predicted_demand - current_inventory)
    
    if shortage > 0:
        emergency_qty = shortage + safety_buffer
        emergency_cost = emergency_qty * unit_cost * emergency_cost_factor
        return {
            'order': True,
            'qty': emergency_qty,
            'cost': emergency_cost,
            'rate': shortage / predicted_demand
        }
    return {'order': False}
```

### 6.2 비용 구조 분석

| 비용 항목 | 계산식 | 비고 |
|----------|--------|------|
| **발주 비용** | 발주 횟수 × 발주당 비용 | 연간 총합 |
| **보관 비용** | 평균 재고 × 보관비율 (보통 20%/년) | 자본 비용 + 물리적 보관비 |
| **긴급 발주 비용** | 긴급 발주 수량 × 단가 × 1.5배수 | 배송료, 수수료 등 추가 |
| **부족 비용** | 미충족 수요 × 손실가격계수 | 기회비용 |

### 6.3 최적화 결과
```
08_cost_optimization_results/
├── A_company_cost_analysis.csv (발주비, 보관비, 긴급비 분석)
├── B_company_cost_analysis.csv
├── emergency_order_frequency.csv (월별 긴급 발주 발생률)
├── total_cost_comparison.csv (개선 전후 비교)
└── cost_optimization_summary.txt
```

---

## 7. 강화학습 기반 정책 최적화 (Advanced)

### 7.1 강화학습 프레임워크
```python
# 스크립트명: 09_reinforcement_learning_optimization.py

class InventoryOptimizationEnv:
    """
    상태(State): 현재 재고, 예측 수요, 리드타임 상태
    행동(Action): 발주량 결정 (또는 발주 안함)
    보상(Reward): -1 × (발주비 + 보관비 + 긴급발주비)
    """
    
    def step(self, action):
        # 재고 업데이트
        # 비용 계산
        # 다음 상태 반환
        pass

# 알고리즘: DQN (Deep Q-Network) 또는 PPO (Proximal Policy Optimization)
# 목표: 총 비용 최소화하면서 충족률 85% 이상 유지
```

### 7.2 강화학습 산출물
```
09_rl_optimization_results/
├── rl_policy_converged.pkl (학습된 정책 모델)
├── rl_training_log.csv (에포크별 보상/손실)
├── rl_optimal_policy_results.csv (최적 발주 정책)
└── rl_vs_baseline_comparison.txt (개선율 분석)
```

---

## 8. 시스템 통합 및 의사결정 지원

### 8.1 통합 대시보드 및 리포팅
```python
# 스크립트명: 10_integrated_dashboard.py

def generate_inventory_dashboard(
    company_id,
    date_range,
    product_type
):
    """
    실시간 재고 현황
    - 현재 재고 vs Min/Max
    - 예측 수요 vs 실제 수요
    - 발주 추천 여부
    - 비용 절감률
    """
    return dashboard_data
```

### 8.2 발주 추천 시스템
```
11_order_recommendation_system/
├── A_company_order_recommendation_daily.csv (일일 발주 추천)
├── B_company_order_recommendation_daily.csv
├── recommendation_confidence.csv (신뢰도 지수)
└── manual_review_alert.csv (검토 필요 건)
```

---

## 9. 확장성 및 템플릿화

### 9.1 5,000개 품목 확장 전략
```python
# 스크립트명: 12_scalable_pipeline.py

def batch_process_all_products(
    company_id,
    product_list,
    config_dict
):
    """
    병렬 처리 (Multiprocessing)
    - 품목당 약 50~100ms 처리 시간
    - 5,000개 품목 → 약 5~10분 처리
    - 일일 자동 업데이트 가능
    """
    results = parallel_execute(
        calculate_inventory_policy,
        product_list,
        num_workers=8
    )
    return results
```

### 9.2 템플릿 파일
```
12_scalable_templates/
├── product_config_template.json (품목별 설정 템플릿)
├── company_config_template.json (회사별 설정)
├── parameter_optimization_guide.md (매개변수 최적화 가이드)
└── maintenance_checklist.md (정기 점검 항목)
```

---

## 10. 최종 산출물 요약

### 10.1 핵심 산출 파일 (최종 결과물)

| 순번 | 파일명 | 용도 | 형식 | 담당자 |
|-----|--------|------|------|--------|
| 1 | `FINAL_A_company_optimal_policy.csv` | A사 최종 재고 정책 | CSV | 분석팀 |
| 2 | `FINAL_B_company_optimal_policy.csv` | B사 최종 재고 정책 | CSV | 분석팀 |
| 3 | `FINAL_Forecasted_Demand_24m.csv` | 예측 수요 (24개월) | CSV | 예측팀 |
| 4 | `FINAL_Cost_Analysis_Report.xlsx` | 비용 절감 분석 | Excel | 재무팀 |
| 5 | `FINAL_System_Implementation_Guide.pdf` | 시스템 도입 가이드 | PDF | 운영팀 |
| 6 | `FINAL_Dashboard_Template.html` | 대시보드 (실시간 모니터링) | HTML | IT팀 |
| 7 | `FINAL_Scaling_Playbook_5000items.md` | 5,000개 품목 확장 가이드 | Markdown | 전략팀 |

### 10.2 분석 결과 요약
```
FINAL_Executive_Summary.txt
- 핵심 성과 지표 (KPI)
  ├ 수요 예측 정확도: MAPE ___% (목표 15%)
  ├ 비용 절감율: ___% (보관비용, 발주비용, 긴급발주비용)
  ├ 출고 충족률: ___% (목표 99% 이상)
  └ ROI: ___개월

- 품목 유형별 정책 효과
  ├ Type A: 안전재고 ___ → ___ (-__%)
  ├ Type B: 발주 주기 ___ → ___ 최적화
  └ Type C: 긴급 발주율 ___ → ___ (-__%)

- 시스템 도입 일정
  ├ Phase 1 (현재): 20개 품목 파일럿 (Q1 완료)
  ├ Phase 2: 100개 품목 확대 (Q2)
  └ Phase 3: 5,000개 품목 풀 스케일 (Q3~Q4)
```

---

## 11. 프로젝트 실행 일정

| 단계 | 시기 | 산출물 | 담당 |
|------|------|--------|------|
| **Phase 1: 분석 준비** | 1월 | 데이터 전처리 완료, 품목 분류 | 분석팀 |
| **Phase 2: 예측 모델** | 2월 | 수요 예측 모델 및 검증 | ML팀 |
| **Phase 3: 정책 수립** | 3월 | Min-Max 정책, 발주 추천 | 최적화팀 |
| **Phase 4: 강화학습** | 3월 | RL 기반 비용 최적화 | AI팀 |
| **Phase 5: 시스템 통합** | 4월 | 대시보드, 자동화 파이프라인 | 개발팀 |
| **Phase 6: 검증 & 개선** | 5월 | 파일럿 운영, 성과 검증 | 운영팀 |
| **Phase 7: 확장 계획** | 6월 | 5,000개 품목 확장 전략 수립 | 전략팀 |

---

## 12. 기술 스택 및 환경

### 12.1 개발 환경
- **언어**: Python 3.9+
- **데이터 처리**: Pandas, NumPy
- **시계열 예측**: StatsModels, Prophet, TensorFlow/Keras
- **최적화**: SciPy, Pyomo
- **강화학습**: OpenAI Gym, Stable-Baselines3
- **시각화**: Matplotlib, Plotly, Dash
- **배포**: Flask/FastAPI, Docker

### 12.2 폴더 구조
```
project_root/
├── 01_data/
│   ├── raw/
│   │   ├── A_company_outbound_24months.csv
│   │   ├── A_company_inbound_24months.csv
│   │   ├── B_company_outbound_24months.csv
│   │   └── B_company_inbound_24months.csv
│   ├── processed/
│   │   ├── A_company_preprocessed.csv
│   │   ├── B_company_preprocessed.csv
│   │   ├── A_company_train_18m.csv
│   │   ├── A_company_test_6m.csv
│   │   ├── B_company_train_18m.csv
│   │   └── B_company_test_6m.csv
│   └── metadata/
│       └── data_dictionary.csv
├── 02_scripts/
│   ├── 01_data_preprocessing.py
│   ├── 02_product_classification.py
│   ├── 03_product_classification_report.csv
│   ├── 04_demand_forecast_es.py
│   ├── 04_demand_forecast_arima.py
│   ├── 04_demand_forecast_prophet.py
│   ├── 04_demand_forecast_ensemble.py
│   ├── 04_demand_forecast_lstm.py
│   ├── 05_forecast_evaluation.py
│   ├── 06_min_max_inventory_calculation.py
│   ├── 07_emergency_order_optimization.py
│   ├── 08_cost_analysis.py
│   ├── 09_reinforcement_learning_optimization.py
│   ├── 10_integrated_dashboard.py
│   ├── 11_order_recommendation_system.py
│   ├── 12_scalable_pipeline.py
│   └── utils/
│       ├── data_utils.py
│       ├── forecast_utils.py
│       ├── optimization_utils.py
│       └── config.py
├── 03_results/
│   ├── 05_demand_forecast_results/
│   ├── 06_inventory_policy_results/
│   ├── 07_emergency_order_results/
│   ├── 08_cost_optimization_results/
│   ├── 09_rl_optimization_results/
│   ├── 10_dashboard_results/
│   └── FINAL_Executive_Summary.txt
├── 04_models/
│   ├── forecast_models/
│   ├── rl_models/
│   └── saved_pipelines/
├── 05_docs/
│   ├── Project_Plan.md (본 문서)
│   ├── Technical_Documentation.pdf
│   ├── Implementation_Guide.pdf
│   └── User_Manual.pdf
├── 06_dashboard/
│   └── app.py (Dash/Flask 웹 앱)
└── README.md
```

---

## 13. 성공 지표 (KPI)

| KPI | 목표값 | 측정 방법 | 검증 주기 |
|-----|--------|----------|----------|
| **수요 예측 정확도** | MAPE < 15% | 검증 데이터 기준 | 월 1회 |
| **비용 절감율** | 15% 이상 | (기존 비용 - 신규 비용) / 기존 비용 | 분기 1회 |
| **출고 충족률** | 99% 이상 | (충족된 수요 / 전체 수요) × 100 | 주 1회 |
| **발주 정확도** | 발주량 오차 ±10% | 실제 발주량 vs 추천량 | 월 1회 |
| **시스템 안정성** | 99.5% 이상 | 시스템 가용시간 / 전체시간 | 일 1회 |
| **긴급 발주율** | 15% ± 2% | 긴급 발주 / 전체 발주 | 월 1회 |

---

## 14. 위험 요소 및 대응 방안

| 위험 요소 | 영향도 | 발생 확률 | 대응 방안 |
|----------|--------|----------|----------|
| 수요 급격한 변화 | 높음 | 중간 | 강화학습 모델로 적응성 강화, 월간 모델 재학습 |
| 데이터 품질 문제 | 높음 | 중간 | 정기적 데이터 감시 체계, 이상치 알림 설정 |
| 시스템 과부하 | 중간 | 낮음 | 병렬 처리 설계, 클라우드 확장성 고려 |
| 이해관계자 저항 | 중간 | 중간 | 충분한 교육 및 변경 관리, 점진적 전환 |
| 비용 산정 오류 | 낮음 | 낮음 | 정기 비용 감사, 회계팀 협조 |

---

## 15. 부록

### A. 주요 함수 참고
- Min-Max 재고 계산 함수 (Section 5.1)
- 발주량 단위 반올림 함수 (Section 5.2)
- 긴급 발주 결정 함수 (Section 6.1)
- 강화학습 환경 클래스 (Section 7.1)

### B. 데이터 사전 (Data Dictionary)
`01_data/metadata/data_dictionary.csv` 참고

### C. 외부 참고 문헌
- Wagner-Whitin 모델 (최적 발주 정책)
- Newsvendor 문제 (단일 기간 재고 최적화)
- Markov Decision Process (MDP 기반 RL)

---

**작성자**: SCM 분석팀  
**작성일**: 2026년 1월 16일  
**최종 승인**: 프로젝트 관리자
