# visualization.py
"""데이터 시각화 모듈 (고객 시연/보고용)"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_aggregated_trend(df, period_col='Day', item_col='Item', 
                          value_col='Outflow', ma_col='Outflow_ma3',
                          title_suffix="Aggregated Trend"):
    """
    집계된 데이터 트렌드 시각화
    """
    items = df[item_col].unique()
    
    for item in items:
        data = df[df[item_col] == item].copy()
        data[period_col] = data[period_col].astype(str)
        
        plt.figure(figsize=(12, 5))
        plt.plot(data[period_col], data[value_col], 
                marker='o', label=f"{value_col}")
        plt.plot(data[period_col], data[ma_col], 
                marker='x', linestyle='--', label="Moving Average")
        
        plt.title(f"Item {item} - {title_suffix}")
        plt.xlabel("Period")
        plt.ylabel(value_col)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def plot_weighted_ma(df, period_col='Day', item_col='Item',
                     value_col='Outflow', weights=[4,3,2,1]):
    """
    가중 이동평균 시각화 (주차별 분석용)
    
    Args:
        weights: 가중치 배열 (최근 → 과거 순서)
    """
    # 가중치 정규화
    w = np.array(weights) / sum(weights)
    
    def weighted_moving_average(x):
        return np.convolve(x, w, mode="same")
    
    # 가중 이동평균 계산
    df = df.copy()
    df[f"{value_col}_wma"] = (
        df.groupby(item_col)[value_col]
        .transform(lambda x: weighted_moving_average(x.values))
    )
    
    # 시각화
    items = df[item_col].unique()
    
    for item in items:
        data = df[df[item_col] == item].copy()
        data[period_col] = data[period_col].astype(str)
        
        plt.figure(figsize=(12, 5))
        plt.plot(data[period_col], data[value_col], 
                marker='o', label=f"{value_col}")
        plt.plot(data[period_col], data[f"{value_col}_wma"], 
                marker='x', linestyle='--', 
                label=f"Weighted MA {weights}")
        
        plt.title(f"Item {item} - Weighted Moving Average")
        plt.xlabel("Period")
        plt.ylabel(value_col)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    return df


def compare_before_after(df_before, df_after, metric='cost'):
    """
    RL 처리 전후 비교 시각화 (고객 시연용)
    
    Args:
        df_before: 전처리 전 데이터
        df_after: RL 최적화 후 데이터
        metric: 비교 지표 ('cost', 'service_level' 등)
    """
    # TODO: RL 완성 후 구현
    pass


# visualization.py에 추가

def compute_inventory(group, initial_stock=0):
    """
    재고 수준 계산
    
    Args:
        group: Item별 그룹 데이터
        initial_stock: 초기 재고
    
    Returns:
        Inventory 컬럼이 추가된 그룹
    """
    inv = []
    stock = initial_stock
    for inflow, outflow in zip(group["Inflow"], group["Outflow"]):
        stock += inflow - outflow
        inv.append(stock)
    group["Inventory"] = inv
    return group


def plot_inventory_analysis(df_weekly, item_list=None, ylim_rules=None):
    """
    재고 수준 분석 시각화 (처리 전/후 비교용)
    
    Args:
        df_weekly: 주차별 집계 데이터 (Inflow, Outflow, Inventory 포함)
        item_list: 특정 품목만 표시 (None이면 전체)
        ylim_rules: 품목별 y축 범위 (None이면 config 사용)
    
    Note:
        - 현재: 처리 전 데이터 확인용
        - 향후: RL 최적화 전/후 비교
    """
    import matplotlib.ticker as mticker
    from config import YLIM_RULES
    
    if ylim_rules is None:
        ylim_rules = YLIM_RULES
    
    items = item_list or df_weekly["Item"].unique()
    
    for item in items:
        data = df_weekly[df_weekly["Item"] == item].copy()
        data["Day"] = data["Day"].astype(str)
        
        plt.figure(figsize=(12, 5))
        
        plt.plot(data["Day"], data["Outflow"], 
                marker='o', label="Weekly Outflow")
        plt.plot(data["Day"], data["Outflow_wma"], 
                marker='x', linestyle='--', label="Weighted MA")
        plt.plot(data["Day"], data["Inflow"], 
                marker='s', linestyle='-', label="Weekly Inflow")
        plt.plot(data["Day"], data["Inventory"], 
                marker='^', linestyle='-', label="Inventory Level")
        
        plt.title(f"Item {item} - Inventory Analysis")
        plt.xlabel("Week")
        plt.ylabel("Units")
        plt.legend()
        plt.xticks(rotation=45)
        
        # 품목별 y축 범위
        item_str = str(item)
        ylim = ylim_rules.get(item_str, ylim_rules.get("default", 50000))
        plt.ylim(0, ylim)
        
        # 콤마 표시
        plt.gca().yaxis.set_major_formatter(
            mticker.StrMethodFormatter('{x:,.0f}')
        )
        
        plt.tight_layout()
        plt.show()