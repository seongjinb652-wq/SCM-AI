# run_simulation.py

import pandas as pd
from inventory_utils import make_params
from inventory_simulation import sim_stage1, sim_stage2, sim_stage3
from performance_check import check_performance
from config import ITEM_IMPORTANCE

# ========== 데이터 로드 ==========
df_daily = pd.read_csv("data/sample_data.csv")
df_daily["Day"] = pd.to_datetime(df_daily["Day"])

# ========== 파라미터 생성 ==========
params = make_params(df_daily, lead_time=14)

# ========== Stage 1: 기본 Min-Max ==========
print("\n" + "="*60)
print("  Stage 1: 고정 리드타임 14일 + 단순 Min-Max")
print("="*60)

df_orders_stage1 = sim_stage1(df_daily, params)
result_stage1 = check_performance(df_orders_stage1, params, ITEM_IMPORTANCE)

for k, v in result_stage1.items():
    print(f"{k}: {v}")

# ========== Stage 2: 스파이크 처리 + 쿨다운 ==========
print("\n" + "="*60)
print("  Stage 2: 스파이크 처리 + 중요도별 차등 + 쿨다운")
print("="*60)

df_orders_stage2 = sim_stage2(df_daily, params, ITEM_IMPORTANCE, cooldown_days=3)
result_stage2 = check_performance(df_orders_stage2, params, ITEM_IMPORTANCE)

for k, v in result_stage2.items():
    print(f"{k}: {v}")

# ========== Stage 3: (미래) ==========
# print("\n" + "="*60)
# print("  Stage 3: 가변 리드타임 + EMA + 백오더")
# print("="*60)
# df_orders_stage3, backorders_stage3 = sim_stage3(df_daily, cfg)
# result_stage3 = check_performance(df_orders_stage3, params, ITEM_IMPORTANCE)

# ========== 결과 비교 ==========
print("\n" + "="*60)
print("  Stage 비교")
print("="*60)

comparison = pd.DataFrame({
    'Stage1': result_stage1,
    'Stage2': result_stage2
})
print(comparison.T)