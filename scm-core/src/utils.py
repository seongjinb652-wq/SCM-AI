# utils.py


import warnings
import pandas as pd
import math


def setup_environment():
    """
    실행 환경 초기화
    - Warning 숨기기
    - Pandas 출력 옵션 설정
    """
    # Warning 숨기기
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Pandas 출력 옵션
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 2000)
    pd.set_option("display.colheader_justify", "center")
    
    print("✅ 환경 설정 완료")


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)
pd.set_option("display.colheader_justify", "center")

"""유틸리티 함수 모음"""
#detect_encoding() - 인코딩 감지
#convert_to_parquet() - CSV → Parquet 변환
#data_health_check() - 데이터 상태 체크

import chardet
import pandas as pd
from pathlib import Path

def detect_encoding(file_path, sample_size=10000):
    """파일 인코딩 감지 (필요시 사용)"""
    with open(file_path, "rb") as f:
        rawdata = f.read(sample_size)
    result = chardet.detect(rawdata)
    print(f"Detected: {result}")
    return result['encoding']

def convert_to_parquet(csv_path, output_path):
    """CSV를 Parquet으로 변환"""
    df = pd.read_csv(csv_path)
    df.to_parquet(output_path)
    print(f"Saved: {output_path}")

# utils.py

def data_health_check(df):
    """데이터 상태 체크 (다운로드 후 검증용)"""
    print("=" * 50)
    print("📊 데이터 상태 체크")
    print("=" * 50)
    
    print(f"\n✓ Shape (행, 열): {df.shape}")
    print(f"✓ Row count: {len(df)}")
    
    print("\n✓ 결측치 건수:")
    print(df.isna().sum())
    
    print("\n✓ 컬럼별 고유값 개수:")
    print(df.nunique())
    
    print("\n✓ 기본 통계 요약:")
    print(df.describe())
    
    print("\n✓ 데이터 샘플 (앞부분 5행):")
    print(df.head())
    
    print("=" * 50)
```

### 옵션 2: 별도 파일
```
# utils.py 맨 아래에 추가

import matplotlib.pyplot as plt

def plot_aggregated_data(df, period_col='Day', item_col='Item', 
                         value_col='Outflow', ma_col='Outflow_ma3'):
    """
    집계된 데이터 시각화 (분기별/월별/주별)
    
    Args:
        df: aggregate_by_period()로 집계된 데이터
        period_col: 기간 컬럼명
        item_col: 아이템 컬럼명
        value_col: 값 컬럼명
        ma_col: 이동평균 컬럼명
    """
    items = df[item_col].unique()
    
    for item in items:
        data = df[df[item_col] == item].copy()
        
        # Period를 문자열로 변환
        data[period_col] = data[period_col].astype(str)
        
        plt.figure(figsize=(10, 5))
        plt.plot(data[period_col], data[value_col], 
                marker='o', label=f"{value_col}")
        plt.plot(data[period_col], data[ma_col], 
                marker='x', linestyle='--', label="3-Period MA")
        
        plt.title(f"Item {item} - Aggregated {value_col} & Moving Average")
        plt.xlabel("Period")
        plt.ylabel(value_col)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


# 사용 예시 (주석)
# from data_preprocessor import aggregate_by_period
# from utils import plot_aggregated_data
#
# df_quarterly = aggregate_by_period(df, period='Q')
# plot_aggregated_data(df_quarterly)


# main.py

from data_preprocessor import load_data, preprocess_data, make_inventory_summary
from utils import data_health_check

def validate_customer_data():
    """
    고객 데이터 검증 (RL 학습 전 확인)
    """
    print("=" * 60)
    print("고객 데이터 검증 시작")
    print("=" * 60)
    
    # 데이터 로드
    df = load_data()
    df = preprocess_data(df, full_clean=False)
    
    # 기본 상태 체크
    data_health_check(df)
    
    # 기간별 요약
    print("\n=== Daily Summary ===")
    df_daily, summary_daily = make_inventory_summary(df, period=None)
    print(summary_daily.head(3))
    
    print("\n=== Weekly Summary ===")
    df_weekly, summary_weekly = make_inventory_summary(df, period='W')
    print(summary_weekly.head(3))
    
    print("\n=== Monthly Summary ===")
    df_monthly, summary_monthly = make_inventory_summary(df, period='ME')
    print(summary_monthly.head(3))
    
    print("\n=== Quarterly Summary ===")
    df_quarterly, summary_quarterly = make_inventory_summary(df, period='QE')
    print(summary_quarterly.head(3))
    
    print("=" * 60)
    print("검증 완료")
    print("=" * 60)
    
    return {
        'daily': (df_daily, summary_daily),
        'weekly': (df_weekly, summary_weekly),
        'monthly': (df_monthly, summary_monthly),
        'quarterly': (df_quarterly, summary_quarterly)
    }


if __name__ == "__main__":
    # 고객 데이터 검증
    results = validate_customer_data()




def round_up_100(x):                      # 100 단위 라운드 업 함수
    return int(math.ceil(x / 100.0) * 100)

def round_up_1000(x):                     # 1000 단위 라운드 업 함수
    return int(math.ceil(x / 1000.0) * 1000)

