# config.py
"""
SCM-AI 프로젝트 설정 파일
"""

# ===== Python 버전 호환성 =====
# Python 3.12+ UTC 지원 변경 대응
from datetime import datetime, UTC

# 현재 시각 확인 (필요 시)
# print("현재 UTC 시각:", datetime.now(UTC))


def setup_environment():
    """
    실행 환경 초기화
    - Warning 숨기기
    - Pandas 출력 옵션 설정
    """
    import warnings
    import pandas as pd
    
    # Warning 숨기기
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Pandas 출력 옵션
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 2000)
    pd.set_option("display.colheader_justify", "center")
    
    print("✅ 환경 설정 완료")

# ===== 경로 설정 =====
DATA_PATH = "data/"
MODEL_PATH = "models/"

# 데이터 파일
RAW_DATA_BACKUP = "data/raw/aaaa.csv"
RAW_DATA_FILE = "data/raw/scm_data_2024_2025.csv"
PROCESSED_DATA_FILE = "data/processed/scm_data_processed.parquet"
DATA_ENCODING = "utf-8-sig"


# 데이터 파일
RAW_DATA_BACKUP = "data/raw/aaaa.csv"  # 원본 백업
RAW_DATA_FILE = "data/raw/scm_data_2024_2025.csv"  # 작업용
PROCESSED_DATA_FILE = "data/processed/scm_data_processed.parquet"
DATA_ENCODING = "utf-8-sig"

# 재고 관리 파라미터
MIN_ORDER_QTY = 10
MAX_ORDER_QTY = 1000


# config.py 끝에 추가

# ===== 시각화 설정 =====
YLIM_RULES = {
    "130200000001": 300_000,
    "130100000001": 300_000,
    "130200000002": 100_000,
    "default": 50_000
}

# 가중 이동평균 가중치 (2주 기준)
WMA_WEIGHTS = [0.5, 1, 0.5]

# ===== 학습/검증 분할 =====
TRAIN_MONTHS = 18  # 또는 21
VALIDATION_MONTHS = 6  # 또는 3

# 고객사 A 데모 시
# config.py
USE_SYNTHETIC_DATA = True
SYNTHETIC_DATA_CONFIG = {
    "num_items": 50,
    "demand_pattern": "seasonal",
    "seed": 123  # 다른 패턴
}

# 고객사 B 실제 적용 시
USE_SYNTHETIC_DATA = False
RAW_DATA_FILE = "data/raw/company_b_data.csv"

