"""
Step 0 — 搬檔 + 重新命名 + 轉 parquet + 欄位標準化

從 Desktop 的 TEJ xlsx 複製到 data/raw (改名), 讀入後把中文欄位改成 snake_case,
存成 data/parquet/*.parquet 供後續快速讀取。原檔保留在 Desktop 當備份。
"""
import sys, shutil
import pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import config as C

# 每份檔案的欄位標準化對照 (中文 TEJ 欄名 -> snake_case)
RENAME = {
    "institutional": {
        "代號": "coid", "名稱": "name", "年月日": "date",
        "外資買賣超(千股)": "foreign_net_kshr", "外資買賣超市值(百萬)": "foreign_net_mn",
        "投信買賣超(千股)": "trust_net_kshr", "投信買賣超市值(百萬)": "trust_net_mn",
        "自營買賣超(千股)": "dealer_net_kshr", "自營買賣超市值(百萬)": "dealer_net_mn",
        "自營避險買賣超(千股)": "dealer_hedge_kshr", "自營避險買賣超(百萬)": "dealer_hedge_mn",
        "外資總投資股率%": "foreign_hold_pct", "投信持股率%": "trust_hold_pct",
    },
    "margin": {
        "代號": "coid", "名稱": "name", "年月日": "date",
        "融資餘額(張)": "margin_long_lot", "融資餘額(千元)": "margin_long_kd",
        "融券餘額(張)": "margin_short_lot", "融券餘額(千元)": "margin_short_kd",
        "券資比": "short_long_ratio",
    },
    "price_unadj": {
        "代號": "coid", "名稱": "name", "年月": "ym",
        "開盤價(元)": "open", "收盤價(元)": "close", "最高價(元)": "high", "最低價(元)": "low",
        "成交量(百萬股)_月": "vol_mnshr", "流通在外股數(千股)": "shares_out_kshr",
        "市值(百萬元)": "mktcap_mn", "週轉率％_月": "turnover_pct",
    },
    "price_adj": {
        "代號": "coid", "名稱": "name", "年月": "ym",
        "開盤價(元)_月": "open_adj", "收盤價(元)_月": "close_adj",
        "最高價(元)_月": "high_adj", "最低價(元)_月": "low_adj",
        "成交量(百萬股)_月": "vol_mnshr", "流通在外股數(千股)": "shares_out_kshr",
        "市值(百萬元)": "mktcap_mn", "週轉率％_月": "turnover_pct",
    },
    "industry": {
        "代號": "coid", "名稱": "name",
        "TSE產業名": "tse_industry", "TEJ產業名": "tej_industry", "上市別": "board",
    },
}


def ym_from_date(s: pd.Series) -> pd.Series:
    """年月日字串 (2025/12/31) 或 年月 (2025/12) -> Period[M] 字串 'YYYY-MM'"""
    dt = pd.to_datetime(s.astype(str).str.replace("/", "-"), errors="coerce")
    return dt.dt.to_period("M").astype(str)


def main():
    C.RAW.mkdir(parents=True, exist_ok=True)
    C.PARQUET.mkdir(parents=True, exist_ok=True)
    summary = []
    for src_name, key in C.FILE_MAP.items():
        src = C.RAW_SRC / src_name
        raw_dst = C.RAW / f"{key}.xlsx"
        print(f"[copy] {src_name} -> data/raw/{key}.xlsx")
        shutil.copy2(src, raw_dst)

        df = pd.read_excel(raw_dst, dtype={"代號": str})
        df = df.rename(columns=RENAME[key])

        # 統一日期欄 -> ym (Period 字串)
        if "date" in df.columns:
            df["ym"] = ym_from_date(df["date"])
        elif "ym" in df.columns:
            df["ym"] = ym_from_date(df["ym"])

        out = C.PARQUET / f"{key}.parquet"
        df.to_parquet(out, index=False)
        n_dates = df["ym"].nunique() if "ym" in df.columns else 0
        summary.append((key, df.shape[0], df.shape[1], df["coid"].nunique(), n_dates))
        print(f"        rows={df.shape[0]:,}  cols={df.shape[1]}  coids={df['coid'].nunique()}  months={n_dates}")

    print("\n=== Step 0 summary ===")
    print(f"{'file':<14}{'rows':>10}{'cols':>6}{'coids':>8}{'months':>8}")
    for k, r, c, n, m in summary:
        print(f"{k:<14}{r:>10,}{c:>6}{n:>8}{m:>8}")


if __name__ == "__main__":
    main()
