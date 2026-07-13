"""
Step 3 — 描述性統計 + seaborn 圖
1) 對作時序頻率 (每月佔比, 洋買土賣 vs 洋賣土買)
2) 產業分布 heatmap (產業 × 年 對作率)
3) 市值分層 / 上市別 對作率
4) 流量法 vs 持股率法 一致性
輸出: outputs/figures/desc_*.png, outputs/tables/desc_*.csv
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import config as C
import plotstyle as ps


def save_tab(df, name):
    C.TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.TAB / name, encoding="utf-8-sig")
    print("  tab ->", C.TAB / name)


def short_ind(s):
    """'M1100 水泥工業' -> '水泥工業'"""
    return s.astype(str).str.replace(r"^[A-Z]\d+\s*", "", regex=True)


def main():
    ps.setup()
    p = pd.read_parquet(C.SIGNALS)
    p["ymp"] = pd.PeriodIndex(p["ym"], freq="M")

    # ===== 1) 時序頻率 =====
    g = p.groupby("ymp")
    ts = pd.DataFrame({
        "洋買土賣": g.apply(lambda d: (d.category == "洋買土賣").mean(), include_groups=False),
        "洋賣土買": g.apply(lambda d: (d.category == "洋賣土買").mean(), include_groups=False),
    })
    ts["對作合計"] = ts["洋買土賣"] + ts["洋賣土買"]
    ts.index = ts.index.to_timestamp()
    save_tab(ts, "desc_timeseries.csv")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(ts.index, ts["洋買土賣"] * 100, label="洋買土賣", lw=1.8)
    ax.plot(ts.index, ts["洋賣土買"] * 100, label="洋賣土買", lw=1.8)
    ax.plot(ts.index, ts["對作合計"] * 100, label="對作合計", lw=2.4, color="k", alpha=.6)
    ax.set(title="每月土洋對作個股佔比 (2010–2025)", xlabel="", ylabel="佔全市場 %")
    ax.legend()
    ps.save(fig, "desc_timeseries.png")

    # ===== 2) 產業 × 年 對作率 heatmap =====
    d = p.dropna(subset=["tse_industry"]).copy()
    d["ind"] = short_ind(d["tse_industry"])
    piv = (d.groupby(["ind", "year"])["opp_flow"].mean() * 100).unstack("year")
    # 只留樣本較多的產業 (依平均對作率排序)
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    save_tab(piv, "desc_industry_year.csv")

    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(piv, cmap="rocket_r", ax=ax, cbar_kws={"label": "對作率 %"},
                linewidths=.3, linecolor="white")
    ax.set(title="各產業歷年土洋對作率 (%)", xlabel="", ylabel="")
    ps.save(fig, "desc_industry_heatmap.png")

    # ===== 3) 市值分層 / 上市別 =====
    by_tier = (p.dropna(subset=["mcap_tier"]).groupby("mcap_tier", observed=True)["opp_flow"]
               .mean() * 100).reindex(C.MCAP_TIER_LABELS)
    by_board = p.dropna(subset=["board"]).groupby("board")["opp_flow"].mean() * 100
    save_tab(by_tier.to_frame("對作率%"), "desc_by_mcap_tier.csv")
    save_tab(by_board.to_frame("對作率%"), "desc_by_board.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(x=by_tier.index, y=by_tier.values, ax=axes[0], hue=by_tier.index, legend=False)
    axes[0].set(title="市值分層對作率", xlabel="", ylabel="對作率 %")
    sns.barplot(x=by_board.index, y=by_board.values, ax=axes[1], hue=by_board.index, legend=False)
    axes[1].set(title="上市別對作率", xlabel="", ylabel="")
    ps.save(fig, "desc_mcap_board.png")

    # ===== 4) 流量法 vs 持股率法一致性 =====
    m = p["d_foreign_hold"].notna() & p["d_trust_hold"].notna()
    ct = pd.crosstab(p.loc[m, "opp_flow"], p.loc[m, "opp_hold"],
                     rownames=["流量法對作"], colnames=["持股率法對作"])
    save_tab(ct, "desc_consistency.csv")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(ct, annot=True, fmt=",d", cmap="Blues", ax=ax, cbar=False)
    ax.set(title=f"流量法 vs 持股率法對作 (一致率 {(p.loc[m,'opp_flow']==p.loc[m,'opp_hold']).mean():.0%})")
    ps.save(fig, "desc_consistency.png")

    print("\nStep 3 done.")


if __name__ == "__main__":
    main()
