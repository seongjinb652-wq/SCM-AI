# statistical_analyzer.py
"""
통계 분석 모듈 - 스파이크 제거 및 안정적 통계 산출

Note:
    - 상위 10~20% 급증 구간 제외 (트리밍)
    - 안정적인 평균, 중앙값, MAD 계산
    - Min-Max 재고 정책 파라미터 산출 기반
"""

import pandas as pd
import numpy as np


def trimmed_stats(series, top_percent=0.15):
    """
    상위 급증 구간 제외 후 통계 계산 (스파이크 제거)
    
    Args:
        series: 수요 데이터 (Outflow 등)
        top_percent: 제외할 상위 비율 (기본: 0.15 = 15%)
    
    Returns:
        dict: 통계 정보
            - mean: 평균
            - median: 중앙값
            - mad: 중앙값 절대편차 (Median Absolute Deviation)
            - cutoff: 컷오프 값
            - n_original: 원본 데이터 수
            - n_trimmed: 트리밍 후 데이터 수
    
    Example:
        >>> stats = trimmed_stats(df['Outflow_sum'], top_percent=0.15)
        >>> print(f"안정적 평균: {stats['mean']:.2f}")
    """
    series = series.dropna()
    
    # 상위 X% 컷오프
    cutoff = np.percentile(series, 100 * (1 - top_percent))
    trimmed = series[series <= cutoff]
    
    # 통계 계산
    mean_val = trimmed.mean()
    median_val = trimmed.median()
    mad_val = (np.abs(trimmed - median_val)).median()
    
    return {
        "mean": mean_val,
        "median": median_val,
        "mad": mad_val,
        "cutoff": cutoff,
        "n_original": len(series),
        "n_trimmed": len(trimmed)
    }


def make_trimmed_summary(df_grouped, label, top_percent=0.15):
    """
    품목별 트리밍 통계 요약
    
    Args:
        df_grouped: 집계된 데이터 (Item, Outflow_sum 포함)
        label: 집계 단위 ("Daily", "Weekly", "Monthly", "Quarterly")
        top_percent: 제외할 상위 비율
    
    Returns:
        DataFrame: 품목별 트리밍 통계
    """
    results = []
    
    for item, g in df_grouped.groupby("Item"):
        stats = trimmed_stats(g["Outflow_sum"], top_percent=top_percent)
        stats["Item"] = item
        stats["freq"] = label
        results.append(stats)
    
    return pd.DataFrame(results)


def analyze_all_periods(df, top_percent=0.15):
    """
    모든 기간별 트리밍 통계 분석 (일/주/월/분기)
    
    Args:
        df: 원본 데이터
        top_percent: 트리밍 비율
    
    Returns:
        dict: 기간별 통계 결과
    """
    from data_preprocessor import make_inventory_summary
    
    # 기간별 집계
    df_daily, _ = make_inventory_summary(df, period=None)
    df_weekly, _ = make_inventory_summary(df, period='W')
    df_monthly, _ = make_inventory_summary(df, period='ME')
    df_quarterly, _ = make_inventory_summary(df, period='QE')
    
    # 트리밍 통계
    summary_daily = make_trimmed_summary(df_daily, "Daily", top_percent)
    summary_weekly = make_trimmed_summary(df_weekly, "Weekly", top_percent)
    summary_monthly = make_trimmed_summary(df_monthly, "Monthly", top_percent)
    summary_quarterly = make_trimmed_summary(df_quarterly, "Quarterly", top_percent)
    
    return {
        'daily': summary_daily,
        'weekly': summary_weekly,
        'monthly': summary_monthly,
        'quarterly': summary_quarterly
    }


# 사용 예시 (주석)
# from statistical_analyzer import analyze_all_periods
# 
# results = analyze_all_periods(df, top_percent=0.15)
# print(results['weekly'].head(20))