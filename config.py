"""土洋對作分析 — 共用參數與路徑設定"""
from pathlib import Path

# ---- 路徑 ----
ROOT = Path(__file__).resolve().parent
RAW_SRC = Path("/mnt/c/Users/user/Desktop/data")   # TEJ 原始 xlsx (Desktop)
RAW = ROOT / "data" / "raw"
PARQUET = ROOT / "data" / "parquet"
PANEL = ROOT / "data" / "panel_monthly.parquet"
SIGNALS = ROOT / "data" / "signals_monthly.parquet"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"

# ---- 檔名對照 (原檔 -> pipeline 名) ----
FILE_MAP = {
    "買賣超.xlsx":   "institutional",
    "融資融券.xlsx": "margin",
    "未調整股價.xlsx": "price_unadj",
    "調整股價.xlsx":  "price_adj",
    "產業分類.xlsx":  "industry",
}

# ---- 分析參數 ----
FWD_HORIZONS = [1, 3, 6, 12]      # forward 報酬月數
# 規模門檻：買賣超股數佔流通在外股數的絕對比率下限 (掃描用起點)
MIN_FLOW_RATIO = 0.001            # 0.1% of shares outstanding
MCAP_TIERS = [0.0, 0.5, 0.8, 1.0]  # 市值分層 (quantile 邊界): 小/中/大
MCAP_TIER_LABELS = ["小型", "中型", "大型"]

# ---- seaborn 樣式 ----
SNS_CONTEXT = "talk"
SNS_STYLE = "whitegrid"
SNS_PALETTE = "deep"
FIG_DPI = 120
