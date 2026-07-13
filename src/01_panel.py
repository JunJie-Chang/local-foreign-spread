"""
Step 1 — 建立月頻主表 (panel_monthly.parquet)

以 (coid, ym) 為主鍵合併 price_adj / price_unadj / institutional / margin / industry。
計算 forward 還原報酬 (1/3/6/12M, 次月起算)、流動性、市值分層、持股率月變化、買賣超正規化比率。
forward 報酬與月變化以 Period 對齊 (t+h 月缺資料 -> NaN), 避免因停牌造成的錯位。
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import config as C


def load(key):
    return pd.read_parquet(C.PARQUET / f"{key}.parquet")


def main():
    padj = load("price_adj")
    pund = load("price_unadj")
    inst = load("institutional")
    marg = load("margin")
    ind = load("industry")[["coid", "tse_industry", "tej_industry", "board"]]

    # --- base: 還原股價 (報酬用) + 未還原收盤/量 (流動性用) ---
    base = padj[["coid", "name", "ym", "close_adj", "shares_out_kshr",
                 "mktcap_mn", "turnover_pct"]].copy()
    base = base.merge(
        pund[["coid", "ym", "close", "vol_mnshr"]], on=["coid", "ym"], how="left")

    # --- merge 法人 / 融資融券 / 產業 ---
    inst_cols = ["coid", "ym", "foreign_net_kshr", "trust_net_kshr",
                 "dealer_net_kshr", "dealer_hedge_kshr",
                 "foreign_hold_pct", "trust_hold_pct"]
    base = base.merge(inst[inst_cols], on=["coid", "ym"], how="left")
    marg_cols = ["coid", "ym", "margin_long_lot", "margin_short_lot", "short_long_ratio"]
    base = base.merge(marg[marg_cols], on=["coid", "ym"], how="left")
    base = base.merge(ind, on="coid", how="left")

    # 主鍵唯一性檢查
    dup = base.duplicated(["coid", "ym"]).sum()
    assert dup == 0, f"duplicate (coid, ym) keys: {dup}"

    base["ymp"] = pd.PeriodIndex(base["ym"], freq="M")
    base = base.sort_values(["coid", "ymp"]).reset_index(drop=True)

    # --- forward 還原報酬 (Period 對齊) ---
    s_close = base.set_index(["coid", "ymp"])["close_adj"]
    for h in C.FWD_HORIZONS:
        tgt_idx = pd.MultiIndex.from_arrays([base["coid"].values, base["ymp"].values + h])
        fut = s_close.reindex(tgt_idx).values
        base[f"fwd_ret_{h}m"] = fut / base["close_adj"].values - 1.0

    # --- 持股率月變化 (與前一個「日曆月」比, 缺月則 NaN) ---
    for col, out in [("foreign_hold_pct", "d_foreign_hold"),
                     ("trust_hold_pct", "d_trust_hold")]:
        s = base.set_index(["coid", "ymp"])[col]
        prev_idx = pd.MultiIndex.from_arrays([base["coid"].values, base["ymp"].values - 1])
        prev = s.reindex(prev_idx).values
        base[out] = base[col].values - prev

    # --- 買賣超正規化 (佔流通在外股數比率; 千股 / 千股) ---
    so = base["shares_out_kshr"].replace(0, np.nan)
    base["foreign_net_ratio"] = base["foreign_net_kshr"] / so
    base["trust_net_ratio"] = base["trust_net_kshr"] / so

    # --- 流動性: 月成交值 (百萬元) = 月成交量(百萬股) * 未還原收盤(元) ---
    base["amount_mn"] = base["vol_mnshr"] * base["close"]

    # --- 市值分層 (每月 quantile) ---
    def tier(g):
        return pd.qcut(g, q=C.MCAP_TIERS, labels=C.MCAP_TIER_LABELS, duplicates="drop")
    base["mcap_tier"] = (base.groupby("ymp")["mktcap_mn"]
                         .transform(lambda g: tier(g) if g.notna().sum() > 10 else np.nan))
    base["year"] = base["ymp"].dt.year

    base.to_parquet(C.PANEL, index=False)

    # --- 摘要 ---
    print(f"panel rows={len(base):,}  cols={base.shape[1]}  "
          f"coids={base.coid.nunique()}  months={base.ymp.nunique()}")
    print("date range:", str(base.ymp.min()), "->", str(base.ymp.max()))
    print("\nforward 報酬非空比例:")
    for h in C.FWD_HORIZONS:
        c = f"fwd_ret_{h}m"
        print(f"  {c}: {base[c].notna().mean():.1%}  median={base[c].median():.3%}")
    print("\nsaved ->", C.PANEL)


if __name__ == "__main__":
    main()
