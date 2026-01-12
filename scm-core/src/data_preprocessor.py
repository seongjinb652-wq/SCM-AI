# data_preprocessor.py
import pandas as pd
from config import RAW_DATA_FILE, DATA_ENCODING
# data_preprocessor.py

from config import USE_SYNTHETIC_DATA, RAW_DATA_FILE, DATA_ENCODING

def load_data(file_path=None, encoding=None, use_synthetic=None):
    """
    데이터 로드 (실제 또는 가상)
    
    Args:
        use_synthetic: True면 가상 데이터 생성
                       None이면 config 설정 따름
    """
    if use_synthetic is None:
        use_synthetic = USE_SYNTHETIC_DATA
    
    if use_synthetic:
        print("⚠️ 가상 데이터 모드")
        from data_generator import generate_synthetic_scm_data
        from config import SYNTHETIC_DATA_CONFIG
        return generate_synthetic_scm_data(**SYNTHETIC_DATA_CONFIG)
    else:
        print("✅ 실제 데이터 로드")
        path = file_path or RAW_DATA_FILE
        enc = encoding or DATA_ENCODING
        return pd.read_csv(path, encoding=enc)

###################
def load_data(file_path=None, encoding=None):
    """
    데이터 로드
    
    Args:
        file_path: CSV 파일 경로 (None이면 config 사용)
        encoding: 인코딩 (None이면 config 사용)
    """
    path = file_path or RAW_DATA_FILE
    enc = encoding or DATA_ENCODING
    return pd.read_csv(path, encoding=enc)

def clean_outflow_column(df):
    """
    Outflow 컬럼 전처리
    - "0.00.0" → "0.0" 교체
    - 쉼표 제거
    - 숫자 변환
    """
    df = df.copy()
    
    df["Outflow"] = (
        df["Outflow"].astype(str)
        .str.replace("0.00.0", "0.0")
        .str.replace(",", "")
    )
    
    df["Outflow"] = pd.to_numeric(df["Outflow"], errors="coerce")
    
    nan_count = df["Outflow"].isna().sum()
    if nan_count > 0:
        print(f"⚠️ Outflow 컬럼에 NaN {nan_count}개 발견")
        print(df[df["Outflow"].isna()])
    
    return df

def preprocess_data(df):
    """전체 전처리 파이프라인"""
    df = clean_outflow_column(df)
    return df

def aggregate_by_period(df, period='D', item_col='Item', date_col='Day', value_col='Outflow'):
    """
    기간별 집계 (하이퍼파라미터 튜닝용)
    
    Args:
        df: 원본 데이터
        period: 집계 단위
            'D' - 일별 (기본)
            'W' - 주별
            'M' - 월별
            'Q' - 분기별
        item_col: 아이템 컬럼명
        date_col: 날짜 컬럼명
        value_col: 집계 대상 컬럼명
    
    Returns:
        집계된 DataFrame
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 기간별 집계
    df_agg = df.groupby([
        df[item_col], 
        df[date_col].dt.to_period(period)
    ])[value_col].sum().reset_index()
    
    # 이동평균 추가 (3기간)
    df_agg[f"{value_col}_ma3"] = (
        df_agg.groupby(item_col)[value_col]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    
    return df_agg

# 사용 예시 (주석으로)
# df_daily = aggregate_by_period(df, period='D')      # 일별
# df_weekly = aggregate_by_period(df, period='W')     # 주별  
# df_monthly = aggregate_by_period(df, period='M')    # 월별
# df_quarterly = aggregate_by_period(df, period='Q')  # 분기별


# data_preprocessor.py에 추가

def clean_numeric_columns(df, columns=['Inflow', 'Outflow']):
    """
    숫자 컬럼 정제 (확장 데이터 대비)
    
    - 쉼표 제거
    - 잘못된 패턴(".." 등) 교정
    - 숫자 변환 및 NaN 처리
    
    Args:
        df: 원본 데이터
        columns: 정제할 컬럼 리스트
    
    Returns:
        정제된 DataFrame
    
    Note:
        현재 20품목 → 향후 5000품목 확장 시 사용
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            print(f"⚠️ 컬럼 '{col}' 없음 - 건너뜀")
            continue
        
        # 문자열 정제
        df[col] = (
            df[col].astype(str)
            .str.strip()
            .str.replace(",", "")
            .str.replace("..", ".", regex=False)  # 잘못된 패턴 교정
        )
        
        # 숫자 변환
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # NaN 처리 (0으로 대체)
    df[columns] = df[columns].fillna(0.0)
    
    # 확인
    nan_counts = df[columns].isna().sum()
    if nan_counts.any():
        print(f"⚠️ NaN 발견: {nan_counts[nan_counts > 0]}")
    
    return df


# data_preprocessor.py 끝에 추가

def aggregate_inventory_by_period(df, period='D', initial_inventory=0,
                                  item_col='Item', date_col='Day',
                                  inflow_col='Inflow', outflow_col='Outflow'):
    """
    기간별 재고 집계 (일/주/월/분기)
    
    Args:
        df: 원본 데이터
        period: 집계 단위
            'D' - 일별
            'W' - 주별 (일요일 기준)
            'M' - 월별
            'Q' - 분기별
        initial_inventory: 초기 재고 (기초재고)
        item_col: 품목 컬럼명
        date_col: 날짜 컬럼명
        inflow_col: 입고 컬럼명
        outflow_col: 출고 컬럼명
    
    Returns:
        집계된 DataFrame (Inflow_sum, Outflow_sum, Inventory_calc 포함)
    
    Note:
        하이퍼파라미터 튜닝 시 사용 (분기→월→주→일 순서로)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([item_col, date_col])
    
    # 기간별 집계
    if period == 'D':
        # 일별: groupby만 사용
        df_agg = (
            df.groupby([item_col, date_col])
            .agg(
                Inflow_sum=(inflow_col, "sum"),
                Outflow_sum=(outflow_col, "sum")
            )
            .reset_index()
        )
    else:
        # 주/월/분기: resample 사용
        df_agg = (
            df.set_index(date_col)
            .groupby(item_col)
            .resample(period)
            .agg(
                Inflow_sum=(inflow_col, "sum"),
                Outflow_sum=(outflow_col, "sum")
            )
            .reset_index()
        )
    
    # 재고 계산 (누적)
    df_agg["Inventory_calc"] = (
        df_agg.groupby(item_col, group_keys=False)
        .apply(lambda g: initial_inventory
                       + g["Inflow_sum"].cumsum()
                       - g["Outflow_sum"].cumsum())
        .reset_index(drop=True)
    )
    
    return df_agg


# 사용 예시 (주석)
# df_daily = aggregate_inventory_by_period(df, period='D')
# df_weekly = aggregate_inventory_by_period(df, period='W')
# df_monthly = aggregate_inventory_by_period(df, period='M')
# df_quarterly = aggregate_inventory_by_period(df, period='Q')


# data_preprocessor.py 끝에 추가

def make_inventory_summary(df, period=None, initial_inventory=0):
    """
    기간별 재고 집계 + 요약 통계
    
    Args:
        df: 원본 데이터
        period: 집계 단위
            None - 일간
            'W' - 주간
            'ME' - 월간 (Month End)
            'QE' - 분기 (Quarter End)
        initial_inventory: 초기 재고
    
    Returns:
        grouped: 집계된 데이터 (Inflow_sum, Outflow_sum, Inventory_calc)
        summary: 요약 통계 (입/출고 일수, 평균, 표준편차 등)
    
    Note:
        고객 데이터 검증 및 RL 학습 전 데이터 상태 확인용
    """
    df = df.copy()
    df["Day"] = pd.to_datetime(df["Day"])
    df = df.sort_values(["Item", "Day"])
    
    # 기간별 집계
    if period:  # 주간/월간/분기
        grouped = (
            df.groupby("Item")
            .resample(period, on="Day")
            .agg(
                Inflow_sum=("Inflow", "sum"),
                Outflow_sum=("Outflow", "sum")
            )
            .reset_index()
        )
    else:  # 일간
        grouped = (
            df.groupby(["Item", "Day"])
            .agg(
                Inflow_sum=("Inflow", "sum"),
                Outflow_sum=("Outflow", "sum")
            )
            .reset_index()
        )
    
    # 재고 계산
    grouped["Inventory_calc"] = (
        grouped.groupby("Item")
        .apply(lambda g: initial_inventory
                       + g["Inflow_sum"].cumsum()
                       - g["Outflow_sum"].cumsum())
        .reset_index(drop=True)
    )
    
    # 요약 통계
    summary = (
        grouped.groupby("Item")
        .agg(
            inflow_days=("Inflow_sum", lambda x: (x > 0).sum()),
            outflow_days=("Outflow_sum", lambda x: (x > 0).sum()),
            inflow_mean=("Inflow_sum", "mean"),
            inflow_std=("Inflow_sum", "std"),
            outflow_mean=("Outflow_sum", "mean"),
            outflow_std=("Outflow_sum", "std"),
            inv_calc_mean=("Inventory_calc", "mean"),
            inv_calc_std=("Inventory_calc", "std")
        )
        .reset_index()
    )
    
    # 반올림
    float_cols = ["inflow_mean", "inflow_std", "outflow_mean", 
                  "outflow_std", "inv_calc_mean", "inv_calc_std"]
    summary[float_cols] = summary[float_cols].round(0).astype(int)
    
    return grouped, summary