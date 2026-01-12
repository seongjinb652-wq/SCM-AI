# import pandas as pd                       # 중복
# import numpy as np                        # 중복


def calc_min_max(df_grouped, lead_time=12, top_percent=0.15):        # Min–Max와 최저 발주량 계산 함수
    results = []
    for item, g in df_grouped.groupby("Item"):
        # 스파이크 제거
        series = g["Outflow_sum"].dropna()
        cutoff = np.percentile(series, 100 * (1 - top_percent))
        trimmed = series[series <= cutoff]

        # 기본 통계 계산
        base = trimmed.median()                # Base: 중앙값(median). 스파이크 제거 후 안정적인 대표값
        mad = (np.abs(trimmed - base)).median() # MAD: 변동성 지표. 중앙값 기준 편차의 중앙값 → 극단값 영향 적음
        mean_recent = trimmed.tail(14).mean()  # 최근 14일 trimmed 평균. 최신 흐름 반영

        # Min–Max 계산
        min_val = base * lead_time + mad * 1.5   # Min: 리드타임 동안의 중앙값 수요 + 변동 여유(MAD×1.5)
        max_val = min_val + base * 7             # Max: Min + 추가 7일치 중앙값 수요 → 안정적 보충 목표

        # 최저 발주량 설정
        min_order_qty_07 = mean_recent * 0.7     # 최근 14일 평균 ×0.7 → 보수적이지만 다소 큰 값, 계획 발주 비율 ↑
        min_order_qty_05 = mean_recent * 0.5     # 최근 14일 평균 ×0.5 → 더 작은 값, 과잉재고 위험 ↓, 긴급 발주 ↑

        results.append({
            "Item": item,
            "Base(median)": round(base, 1),       # 스파이크 제거 후 중앙값
            "MAD": round(mad, 1),                 # 변동성 지표
            "Min": round_up_100(min_val),         # Min: 100 단위로 라운드 업
            "Max": round_up_100(max_val),         # Max: 100 단위로 라운드 업
            "MinOrderQty_07": round_up_100(min_order_qty_07), # 최저 발주량 (0.7 배, 100 단위 올림)
            "MinOrderQty_05": round_up_100(min_order_qty_05), # 최저 발주량 (0.5 배, 100 단위 올림)
            "n_original": len(series),            # 원래 데이터 개수
            "n_trimmed": len(trimmed)             # 스파이크 제거 후 데이터 개수
        })

    return pd.DataFrame(results)


df_minmax_daily = calc_min_max(df_daily, lead_time=12, top_percent=0.15)   # 예시: 일간 데이터 기준으로 Min–Max 계산

# 결과 출력 (앞 20개 품목)
pd.set_option("display.width", 2000)
pd.set_option("display.max_columns", None)
print("=== Daily Min–Max & MinOrderQty ===")
print(df_minmax_daily.head(20).to_string(index=False))