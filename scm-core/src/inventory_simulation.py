# inventory_simulation.py

import pandas as pd
import numpy as np
import math

def round_up_100(x):
    """100 단위로 올림"""
    return int(math.ceil(x / 100.0) * 100)

def ema(series, span=30):
    """지수이동평균 계산"""
    return series.ewm(span=span, adjust=False).mean()


#######################################################################
#  Stage 1: 기본 Min-Max (고정 리드타임 14일)
#######################################################################
def sim_stage1(df_daily, params):
    """
    단계 1: 기본 시뮬레이션
    
    특징:
    - 고정 리드타임 14일
    - 단순 Min-Max 로직
    - MinOrderQty_07 사용
    
    Args:
        df_daily: DataFrame (Item, Day, Outflow_sum)
        params: dict {Item: {Min, Max, MinOrderQty_07, MinOrderQty_05}}
    
    Returns:
        DataFrame: 발주 내역 (Item, Day, OrderType, Qty, Inventory)
    """
    lead_time = 14
    items = df_daily["Item"].unique()
    orders, inflow_pipeline, inventory = [], {}, {}
    
    for it in items:
        inventory[it] = params[it]["Max"] * 0.5
    
    for day in sorted(df_daily["Day"].unique()):
        g = df_daily[df_daily["Day"] == day]
        
        if day in inflow_pipeline:
            for it, qty in inflow_pipeline[day]:
                inventory[it] += qty
            del inflow_pipeline[day]
        
        for _, r in g.iterrows():
            it, out = r["Item"], r["Outflow_sum"]
            inventory[it] -= out
        
        target_day = day + pd.Timedelta(days=lead_time)
        for it in items:
            horizon = df_daily[
                (df_daily["Item"] == it) & 
                (df_daily["Day"] > day) & 
                (df_daily["Day"] <= target_day)
            ]
            demand_14 = horizon["Outflow_sum"].sum()
            proj_inv = inventory[it] - demand_14
            
            if proj_inv < params[it]["Min"]:
                qty = max(params[it]["MinOrderQty_07"], 
                         params[it]["Max"] - proj_inv)
                qty = round_up_100(qty)
                orders.append({
                    "Item": it, "Day": day, "OrderType": "Plan", 
                    "Qty": qty, "Inventory": inventory[it]
                })
                inflow_pipeline.setdefault(target_day, []).append((it, qty))
    
    return pd.DataFrame(orders)


#######################################################################
#  Stage 2: 스파이크 처리 + 중요도별 차등 + 쿨다운
#######################################################################
def sim_stage2(df_daily, params, importance, cooldown_days=3):
    """
    단계 2: 개선된 시뮬레이션
    
    특징:
    - 스파이크 수요 별도 처리 (P85 기준)
    - 중요도별 최저 발주량 차등 (high: 0.7, normal: 0.5)
    - 스파이크 후 쿨다운 (계획 발주 70% 축소)
    
    Args:
        df_daily: DataFrame (Item, Day, Outflow_sum)
        params: dict {Item: {Min, Max, MinOrderQty_07, MinOrderQty_05, Base}}
        importance: dict {Item: "high" | "normal" | "low"}
        cooldown_days: 스파이크 후 쿨다운 기간 (기본 3일)
    
    Returns:
        DataFrame: 발주 내역 (Item, Day, OrderType, Qty, Inventory)
    """
    lead_time = 14
    items = df_daily["Item"].unique()
    orders, inflow_pipeline, inventory, last_spike_day = [], {}, {}, {it: None for it in items}
    
    # 스파이크 컷오프 계산 (품목별 P85)
    cutoffs = df_daily.groupby("Item")["Outflow_sum"].apply(
        lambda s: np.percentile(s.dropna(), 85)
    ).to_dict()
    
    for it in items:
        inventory[it] = params[it]["Max"] * 0.5
    
    for day in sorted(df_daily["Day"].unique()):
        g = df_daily[df_daily["Day"] == day]
        
        if day in inflow_pipeline:
            for it, qty in inflow_pipeline[day]:
                inventory[it] += qty
            del inflow_pipeline[day]
        
        # 수요 + 스파이크 탐지
        for _, r in g.iterrows():
            it, out = r["Item"], r["Outflow_sum"]
            inventory[it] -= out
            
            if out > cutoffs[it]:  # 스파이크 발생
                cap = params[it]["Base"] * 2
                emergency_qty = round_up_100(min(out, cap))
                orders.append({
                    "Item": it, "Day": day, "OrderType": "Emergency", 
                    "Qty": emergency_qty, "Inventory": inventory[it]
                })
                inflow_pipeline.setdefault(
                    day + pd.Timedelta(days=lead_time), []
                ).append((it, emergency_qty))
                last_spike_day[it] = day
        
        # 계획 발주 판단
        target_day = day + pd.Timedelta(days=lead_time)
        for it in items:
            horizon = df_daily[
                (df_daily["Item"] == it) & 
                (df_daily["Day"] > day) & 
                (df_daily["Day"] <= target_day)
            ]
            proj_inv = inventory[it] - horizon["Outflow_sum"].sum()
            
            if proj_inv < params[it]["Min"]:
                # 중요도별 최저 발주량 차등
                moq = (params[it]["MinOrderQty_07"] 
                       if importance.get(it, "normal") == "high" 
                       else params[it]["MinOrderQty_05"])
                qty = max(moq, params[it]["Max"] - proj_inv)
                
                # 쿨다운 적용
                if last_spike_day[it] and (day - last_spike_day[it]).days <= cooldown_days:
                    qty = qty * 0.7
                
                qty = round_up_100(qty)
                orders.append({
                    "Item": it, "Day": day, "OrderType": "Plan", 
                    "Qty": qty, "Inventory": inventory[it]
                })
                inflow_pipeline.setdefault(target_day, []).append((it, qty))
    
    return pd.DataFrame(orders)


#######################################################################
#  Stage 3: 가변 리드타임 + 인지시점 다양화 + 서비스 수준(z) + EMA
#######################################################################
def sim_stage3(df_daily, cfg):
    """
    단계 3: 고급 시뮬레이션
    
    특징:
    - 품목별 가변 리드타임 (8-14일)
    - 인지시점(notice_days) 다양화
    - 서비스 수준(z-score) 기반 안전재고
    - EMA 기반 수요 예측
    - 백오더 추적
    
    Args:
        df_daily: DataFrame (Item, Day, Outflow_sum)
        cfg: dict {Item: {
            "lead_time": int,      # 리드타임 (일)
            "notice_days": int,    # 인지시점 (일)
            "z": float,            # 서비스 수준 (1.5 = 93%)
            "span": int,           # EMA 기간
            "moq_factor": float    # 최저발주량 배수 (0.5/0.7)
        }}
    
    Returns:
        tuple: (orders DataFrame, backorders DataFrame)
    """
    items = df_daily["Item"].unique()
    orders, inflow_pipeline, inventory, backorders = [], {}, {}, []
    
    # 품목별 Base(EMA)와 MAD 준비
    bases, mads = {}, {}
    for it, s in df_daily.groupby("Item")["Outflow_sum"]:
        s = s.fillna(0)
        bases[it] = ema(s, span=cfg.get(it, {}).get("span", 30)).iloc[-1]
        med = s.median()
        mads[it] = (np.abs(s - med)).median()
    
    # 초기 재고
    for it in items:
        lt = cfg.get(it, {}).get("lead_time", 12)
        inventory[it] = bases[it] * lt * 0.5
    
    for day in sorted(df_daily["Day"].unique()):
        if day in inflow_pipeline:
            for it, qty in inflow_pipeline[day]:
                inventory[it] += qty
            del inflow_pipeline[day]
        
        # 수요 반영 + 백오더
        for _, r in df_daily[df_daily["Day"] == day].iterrows():
            it, out = r["Item"], r["Outflow_sum"]
            if inventory[it] >= out:
                inventory[it] -= out
            else:
                backorders.append({
                    "Item": it, "Day": day, 
                    "Qty": out - inventory[it]
                })
                inventory[it] = 0
        
        # 발주 판단 (인지시점)
        for it in items:
            lt = cfg.get(it, {}).get("lead_time", 12)
            nd = cfg.get(it, {}).get("notice_days", lt)
            z = cfg.get(it, {}).get("z", 1.5)
            target_day = day + pd.Timedelta(days=nd)
            
            # Min 계산: Base×LT + z×MAD×√LT
            base = bases[it]
            mad = mads[it]
            min_threshold = base * lt + z * mad * (lt ** 0.5)
            max_target = min_threshold + base * (7 if lt >= 12 else 5)
            
            # 예상재고 계산
            horizon = df_daily[
                (df_daily["Item"] == it) & 
                (df_daily["Day"] > day) & 
                (df_daily["Day"] <= target_day)
            ]
            proj_inv = inventory[it] - horizon["Outflow_sum"].sum()
            
            if proj_inv < min_threshold:
                moq = cfg.get(it, {}).get("moq_factor", 0.5)
                mean_recent = df_daily[df_daily["Item"] == it].tail(14)["Outflow_sum"].mean()
                qty = max(mean_recent * moq, max_target - proj_inv)
                qty = round_up_100(qty)
                orders.append({
                    "Item": it, "Day": day, "OrderType": "Plan", 
                    "Qty": qty, "Inventory": inventory[it]
                })
                inflow_pipeline.setdefault(
                    day + pd.Timedelta(days=lt), []
                ).append((it, qty))
    
    return pd.DataFrame(orders), pd.DataFrame(backorders)


#######################################################################
#  Stage 4: 
#######################################################################
