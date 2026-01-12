# main.py
from data_preprocessor import load_data, aggregate_by_period
from visualization import plot_weighted_ma

# main.py (미래 - 아직 작성 안함)

"""
최종 통합 실행 스크립트

1. 강화학습 모델 로드/학습
2. run_simulation.py 호출
3. 결과 시각화
4. 최적 파라미터 추천
5. 대시보드 표시
"""

# TODO: 강화학습 통합
# TODO: 시각화 추가
# TODO: 추천 시스템 구현

if __name__ == "__main__":
    print("통합 실행 예정...")







# 데이터 로드 & 집계
df = load_data()
df_weekly = aggregate_by_period(df, period='W')

# 고객 시연용 시각화
df_weekly = plot_weighted_ma(df_weekly, weights=[4,3,2,1])

# main.py
"""SCM-AI 메인 실행 스크립트"""

from utils import setup_environment, validate_customer_data
from statistical_analyzer import analyze_all_periods
from data_preprocessor import load_data, preprocess_data

def main():
    """메인 실행 함수"""
    # 0. 환경 설정
    setup_environment()
    
    # 1. 데이터 검증
    print("\n" + "="*60)
    print("1단계: 데이터 검증")
    print("="*60)
    results = validate_customer_data()
    
    # 2. 통계 분석 (스파이크 제거)
    print("\n" + "="*60)
    print("2단계: 통계 분석 (트리밍)")
    print("="*60)
    df = load_data()
    df = preprocess_data(df)
    trimmed_stats = analyze_all_periods(df, top_percent=0.15)
    
    print("\n=== Weekly Trimmed Stats ===")
    print(trimmed_stats['weekly'].head(20))
    
    # 3. Min-Max 파라미터 계산 (TODO)
    # calculate_minmax_params(trimmed_stats['weekly'])
    
    # 4. RL 학습 (TODO)
    # train_rl_model(results['weekly'])
    
    # 5. 성능 평가 (TODO)
    # evaluate_performance()
    
    print("\n✅ 모든 작업 완료")


if __name__ == "__main__":
    main()

#################################
# 후반부 정리후
###############################
# main.py
from inventory_simulation import sim_stage1, sim_stage2, sim_stage3

# Stage 1 실행
orders_s1 = sim_stage1(df_daily, params)

# Stage 2 실행
orders_s2 = sim_stage2(df_daily, params, importance, cooldown_days=3)

# Stage 3 실행
cfg_s3 = {
    "130100000001": {
        "lead_time": 10,
        "notice_days": 10,
        "z": 1.65,
        "span": 30,
        "moq_factor": 0.7
    },
    # ... 다른 품목들
}
orders_s3, backorders_s3 = sim_stage3(df_daily, cfg_s3)

