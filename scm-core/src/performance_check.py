# performance_check.py

def check_performance_basic(df_orders, params):
    """
    기본 성과 확인 (빠른 실행, 핵심 지표만)
    - 발주 수량 비율
    - 평균 재고
    - 전체 재고 부족 일수
    """
    # ... (원본 코드)
    pass

def check_performance_detailed(df_orders, params, importance):
    """
    상세 성과 확인 (품목 등급별 재고 안정성 분석)
    - 발주 수량 비율
    - 평균 재고
    - 등급별 재고 부족 일수 및 품목당 평균
    
    Args:
        importance: dict, Item별 등급 ('high', 'normal', 'low')
    """
    # ... (개선된 코드)
    pass

# 선택적으로 사용
def check_performance(df_orders, params, importance=None):
    """
    통합 인터페이스: importance 제공 여부에 따라 자동 선택
    """
    if importance is None:
        return check_performance_basic(df_orders, params)
    else:
        return check_performance_detailed(df_orders, params, importance)
###################################################################
#  사용 예시
# 빠른 확인
# result_basic = check_performance_basic(df_orders, params)

# 상세 분석
# result_detailed = check_performance_detailed(df_orders, params, importance)

# 또는 통합 인터페이스
# result = check_performance(df_orders, params)  # basic
# result = check_performance(df_orders, params, importance)  # detailed