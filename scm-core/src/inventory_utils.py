import math

# def round_up_100(x):   return int(math.ceil(x / 100.0) * 100)    # 100 단위 라운드 업  (따로 선언)
# def round_up_1000(x):  return int(math.ceil(x / 1000.0) * 1000)  # 1000 단위 라운드 업 (따로 선언)

def calc_min_max(df_grouped, lead_time=12, top_percent=0.15):
    results = []
    for item, g in df_grouped.groupby("Item"):
        series = g["Outflow_sum"].dropna()
        cutoff = np.percentile(series, 100 * (1 - top_percent))   # 스파이크 컷 기준
        trimmed = series[series <= cutoff]                       # 스파이크 제거 데이터
        base = trimmed.median()                                  # 중앙값
        mad = (np.abs(trimmed - base)).median()                  # MAD
        mean_recent = trimmed.tail(14).mean()                    # 최근 14일 평균

        min_val = base * lead_time + mad * 1.5                   # Min 계산
        max_val = min_val + base * 7                             # Max 계산
        min_order_qty_07 = mean_recent * 0.7                     # 최저 발주량 (0.7 배)
        min_order_qty_05 = mean_recent * 0.5                     # 최저 발주량 (0.5 배)

        results.append({
            "Item": item,
            "Base(median)": round(base, 1),                      # 중앙값
            "MAD": round(mad, 1),                                # MAD
            "Min": round_up_100(min_val),                        # Min (100 단위 올림)
            "Max": round_up_100(max_val),                        # Max (100 단위 올림)
            "MinOrderQty_07": round_up_100(min_order_qty_07),    # 최저 발주량 0.7 (100 단위 올림)
            "MinOrderQty_05": round_up_100(min_order_qty_05),    # 최저 발주량 0.5 (100 단위 올림)
            "n_original": len(series),                           # 원래 데이터 개수
            "n_trimmed": len(trimmed)                            # 스파이크 제거 후 데이터 개수
        })
    return pd.DataFrame(results)

# inventory_utils.py

import math
import numpy as np
import pandas as pd

# ========== 라운딩 함수 ==========
def round_up_100(x):
    """100 단위로 올림"""
    return int(math.ceil(x / 100.0) * 100)

def round_up_1000(x):
    """1000 단위로 올림"""
    return int(math.ceil(x / 1000.0) * 1000)

# ========== 변동성 분석 ==========
def is_high_variability(series, threshold_cv=1.0):
    """변동성이 큰 품목인지 판별 (CV >= 1.0)"""
    mean_val, std_val = series.mean(), series.std()
    cv = std_val / mean_val if mean_val > 0 else 0
    return cv >= threshold_cv

# ========== Min-Max 계산 (버전 1 - 단일 리드타임) ==========
def calc_min_max_v1(df_grouped, lead_time=12, top_percent=0.15):
    """기본 Min-Max 계산 (단일 리드타임 버전)"""
    results = []
    for item, g in df_grouped.groupby("Item"):
        series = g["Outflow_sum"].dropna()
        cutoff = np.percentile(series, 100 * (1 - top_percent))
        trimmed = series[series <= cutoff]
        base = trimmed.median()
        mad = (np.abs(trimmed - base)).median()
        mean_recent = trimmed.tail(14).mean()

        min_val = base * lead_time + mad * 1.5
        max_val = min_val + base * 7
        min_order_qty_07 = mean_recent * 0.7
        min_order_qty_05 = mean_recent * 0.5

        results.append({
            "Item": item,
            "Base(median)": round(base, 1),
            "MAD": round(mad, 1),
            "Min": round_up_100(min_val),
            "Max": round_up_100(max_val),
            "MinOrderQty_07": round_up_100(min_order_qty_07),
            "MinOrderQty_05": round_up_100(min_order_qty_05),
            "n_original": len(series),
            "n_trimmed": len(trimmed)
        })
    return pd.DataFrame(results)

# ========== Min-Max 계산 (버전 2 - 품목별 리드타임 + 변동성 반영) ==========
def calc_min_max(df_grouped, lead_times, top_percent=0.15):
    """개선된 Min-Max 계산 (품목별 리드타임, 변동성 반영)"""
    results = []
    for item, g in df_grouped.groupby("Item"):
        series = g["Outflow_sum"].dropna()
        cutoff = np.percentile(series, 100 * (1 - top_percent))
        trimmed = series[series <= cutoff]
        base, mad = trimmed.median(), (np.abs(trimmed - trimmed.median())).median()
        mean_recent = trimmed.tail(14).mean()
        high_var = is_high_variability(trimmed)
        lead_time = lead_times.get(item, 12)
        
        min_val = base * lead_time + mad * 1.5
        max_val = min_val + base * (10 if high_var else 5)
        min_order_qty = mean_recent * 0.7

        results.append({
            "Item": item,
            "Base(median)": round(base, 1),
            "MAD": round(mad, 1),
            "LeadTime": lead_time,
            "HighVariability": high_var,
            "Min": round_up_100(min_val),
            "Max": round_up_100(max_val),
            "MinOrderQty": round_up_100(min_order_qty),
            "n_original": len(series),
            "n_trimmed": len(trimmed)
        })
    return pd.DataFrame(results)



# main.py 사용 예시
# from inventory_utils import calc_min_max, calc_min_max_v1
# from config import LEAD_TIMES, TOP_PERCENT

# 기본 버전
# df_minmax_simple = calc_min_max_v1(df_daily, lead_time=12)

# 개선 버전
# df_minmax_advanced = calc_min_max(df_daily, lead_times=LEAD_TIMES)

import numpy as np

def make_params(df_daily, lead_time=14):
    params = {}
    for item, g in df_daily.groupby("Item"):
        series = g["Outflow_sum"].dropna()
        base = series.median()                                # 중앙값
        mad = (np.abs(series - base)).median()                 # MAD
        mean_recent = series.tail(14).mean()                   # 최근 14일 평균

        min_val = base * lead_time + mad * 1.5                 # Min 계산
        max_val = min_val + base * 7                           # Max 계산
        min_order_qty_07 = mean_recent * 0.7                   # 최저 발주량 0.7
        min_order_qty_05 = mean_recent * 0.5                   # 최저 발주량 0.5

        params[item] = {
            "Base": base,
            "MAD": mad,
            "Min": min_val,
            "Max": max_val,
            "MinOrderQty_07": min_order_qty_07,
            "MinOrderQty_05": min_order_qty_05
        }
    return params

# 실행 예시
params = make_params(df_daily, lead_time=14)
print("=== params 예시 ===")
for k,v in list(params.items())[:3]:   # 앞 3개만 확인
    print(k, v)