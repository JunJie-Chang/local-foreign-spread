"""
Step 7 — 「跟投信」策略月頻回測 + alpha / beta 分離

策略 (大型股, 月頻換股, 等權, 忽略交易成本):
  - 多方 L: 大型 & 洋賣土買 (外資賣、投信買) -> 跟投信做多
  - 空方 S: 大型 & 洋買土賣 (外資買、投信賣) -> 跟投信做空
  - 多空 LS = L - S
基準 (資料內自建, 代替加權指數):
  - bench_all: 全市場市值加權月報酬
  - bench_big: 大型股市值加權月報酬 (隔離「大型股 beta」, 剩下才是選股 alpha)
以 CAPM 迴歸 R_p = alpha + beta * R_bench 取月 alpha, 年化並看顯著性。
輸出: outputs/figures/bt_*.png, outputs/tables/bt_*.csv
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import config as C
import plotstyle as ps


def capm(rp, rb):
    """R_p = a + b R_b; 回傳 月alpha, beta, alpha t值, R2"""
    m = rp.notna() & rb.notna()
    x, y = rb[m].values, rp[m].values
    res = stats.linregress(x, y)
    n = m.sum()
    # alpha 的 t 值 (intercept)
    yhat = res.intercept + res.slope * x
    resid = y - yhat
    s2 = (resid ** 2).sum() / (n - 2)
    se_a = np.sqrt(s2 * (1 / n + x.mean() ** 2 / ((x - x.mean()) ** 2).sum()))
    t_a = res.intercept / se_a
    return res.intercept, res.slope, t_a, res.rvalue ** 2


def perf(r, rb=None):
    r = r.dropna()
    ann_ret = r.mean() * 12
    ann_vol = r.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol else np.nan
    out = dict(月數=len(r), 年化報酬=ann_ret, 年化波動=ann_vol, Sharpe=sharpe,
               勝率=(r > 0).mean())
    if rb is not None:
        a, b, ta, r2 = capm(r, rb)
        out.update(月alpha=a, 年化alpha=a * 12, beta=b, alpha_t=ta, R2=r2)
    return out


def main():
    ps.setup()
    p = pd.read_parquet(C.SIGNALS)
    p["ymp"] = pd.PeriodIndex(p["ym"], freq="M")
    p = p.dropna(subset=["fwd_ret_1m", "mktcap_mn"])

    # ---- 基準: 市值加權月報酬 (t 月持有 -> fwd_ret_1m) ----
    def capwt(g):
        w = g["mktcap_mn"]
        return np.average(g["fwd_ret_1m"], weights=w)
    bench_all = p.groupby("ymp").apply(capwt, include_groups=False).rename("bench_all")
    big = p[p.mcap_tier == "大型"]
    bench_big = big.groupby("ymp").apply(capwt, include_groups=False).rename("bench_big")

    # ---- 策略腿: 大型 & 洋賣土買(多) / 洋買土賣(空), 等權 ----
    longs = (big[big.category == "洋賣土買"].groupby("ymp")["fwd_ret_1m"]
             .mean().rename("L"))
    shorts = (big[big.category == "洋買土賣"].groupby("ymp")["fwd_ret_1m"]
              .mean().rename("S"))
    n_long = big[big.category == "洋賣土買"].groupby("ymp").size().rename("n_long")

    bt = pd.concat([bench_all, bench_big, longs, shorts, n_long], axis=1).sort_index()
    bt["LS"] = bt["L"] - bt["S"]
    bt["L_excess"] = bt["L"] - bt["bench_big"]   # 多方相對大型大盤超額
    bt = bt.dropna(subset=["bench_all"])
    print(f"回測月數={len(bt)}, 多方每月平均持股={bt['n_long'].mean():.1f} 檔")

    # ---- 績效表 ----
    rows = {
        "多方 L（相對大型指數）": perf(bt["L"], bt["bench_big"]),
        "多空 L−S（相對大型指數）": perf(bt["LS"], bt["bench_big"]),
        "多方 L（相對全市場）": perf(bt["L"], bt["bench_all"]),
        "大型股市值加權指數": perf(bt["bench_big"]),
        "全市場市值加權指數": perf(bt["bench_all"]),
    }
    tbl = pd.DataFrame(rows).T
    show = tbl.copy()
    for c in ["年化報酬", "年化波動", "年化alpha", "月alpha"]:
        if c in show:
            show[c] = (show[c] * 100).round(2)
    for c in ["勝率", "R2"]:
        if c in show:
            show[c] = (show[c] * 100).round(1)
    for c in ["Sharpe", "beta", "alpha_t"]:
        if c in show:
            show[c] = show[c].round(2)
    save_cols = ["月數", "年化報酬", "年化波動", "Sharpe", "勝率",
                 "年化alpha", "beta", "alpha_t", "R2"]
    show = show.reindex(columns=save_cols)
    C.TAB.mkdir(parents=True, exist_ok=True)
    show.to_csv(C.TAB / "bt_performance.csv", encoding="utf-8-sig")
    print("  tab -> bt_performance.csv")
    print("\n=== 績效 (年化%, alpha 為扣 beta 後) ===")
    print(show.to_string())

    # 逐年多方超額 (相對大型大盤)
    bt_ts = bt.copy()
    bt_ts["year"] = bt_ts.index.year
    yr = (bt_ts.groupby("year")["L_excess"].sum() * 100).round(2)
    yr.to_csv(C.TAB / "bt_excess_by_year.csv", encoding="utf-8-sig")
    print("  tab -> bt_excess_by_year.csv")

    # ---- 圖 1: 累積淨值 ----
    idx = bt.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for col, lab, c in [("L", "多方 L（洋賣土買）", "#2c7fb8"),
                        ("bench_big", "大型股市值加權指數", "#7f7f7f"),
                        ("LS", "多空 L−S", "#d95f0e")]:
        nav = (1 + bt[col].fillna(0)).cumprod()
        ax.plot(idx, nav, label=lab, lw=2, color=c)
    ax.set(title="策略累積淨值（2010–2025，等權月頻換股，未計交易成本）",
           xlabel="", ylabel="累積淨值（起始＝1）")
    ax.legend()
    ps.save(fig, "bt_nav.png")

    # ---- 圖 2: 逐年多方超額報酬 ----
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = ["#2c7fb8" if v > 0 else "#d95f0e" for v in yr.values]
    ax.bar(yr.index.astype(str), yr.values, color=colors)
    ax.axhline(0, color="k", lw=.8)
    ax.set(title="多方 L 相對大型股指數之逐年超額報酬 (%)",
           xlabel="", ylabel="超額報酬 %")
    plt.setp(ax.get_xticklabels(), rotation=45)
    ps.save(fig, "bt_excess_year.png")

    ic_analysis(p)
    print("\nStep 7 done.")


def ic_analysis(p):
    """連續『土洋分歧』訊號的橫斷面 IC / ICIR 分析。
    訊號 = 月內 z(投信買超比率) - z(外資買超比率); 高=投信積極/外資保守。"""
    def zc(s):
        sd = s.std()
        return (s - s.mean()) / sd if sd and sd == sd else s * np.nan
    p = p.copy()
    p["sig"] = (p.groupby("ymp")["trust_net_ratio"].transform(zc)
                - p.groupby("ymp")["foreign_net_ratio"].transform(zc))

    def ic_series(df, ret, method="spearman", minn=15):
        d = df.dropna(subset=["sig", ret])
        return (d.groupby("ymp")
                .apply(lambda g: g["sig"].corr(g[ret], method=method)
                       if len(g) >= minn else np.nan, include_groups=False)
                .dropna())

    universes = {"大型": p[p.mcap_tier == "大型"], "全市場": p}
    rows, ts = [], {}
    for uname, udf in universes.items():
        for h in C.FWD_HORIZONS:
            ic = ic_series(udf, f"fwd_ret_{h}m")
            n, m, sd = len(ic), ic.mean(), ic.std()
            ir = m / sd if sd else np.nan
            rows.append(dict(母體=uname, horizon=f"{h}m", n月=n,
                             IC平均=m, IC標準差=sd, ICIR年化=ir * np.sqrt(12),
                             IC_t=m / sd * np.sqrt(n) if sd else np.nan,
                             IC為正比例=(ic > 0).mean()))
            if uname == "大型" and h == 1:
                ts["大型_rankIC_1m"] = ic
    ic_tbl = pd.DataFrame(rows)
    show = ic_tbl.copy()
    for c in ["IC平均", "IC標準差"]:
        show[c] = show[c].round(4)
    for c in ["ICIR年化", "IC_t"]:
        show[c] = show[c].round(2)
    show["IC為正比例"] = (show["IC為正比例"] * 100).round(1)
    show.to_csv(C.TAB / "bt_ic_summary.csv", encoding="utf-8-sig", index=False)
    print("  tab -> bt_ic_summary.csv")
    print("\n=== IC / ICIR (Spearman rank IC) ===")
    print(show.to_string(index=False))

    icm = ts["大型_rankIC_1m"]
    icm.index = icm.index.to_timestamp()
    icm.rename("rankIC").to_csv(C.TAB / "bt_ic_timeseries.csv", encoding="utf-8-sig")

    # ---- 圖 3: IC 時序 (大型, 1m) + 12M 滾動平均 + 累積 IC ----
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.bar(icm.index, icm.values, width=20, color="#9ecae1", label="月 rank IC")
    ax.plot(icm.index, icm.rolling(12).mean(), color="#08519c", lw=2.2,
            label="12M 滾動 IC")
    ax.axhline(icm.mean(), color="#d95f0e", ls="--", lw=1.4,
               label=f"平均 IC={icm.mean():.3f}")
    ax.axhline(0, color="k", lw=.7)
    ax2 = ax.twinx()
    ax2.plot(icm.index, icm.cumsum(), color="#238b45", lw=1.6, alpha=.7,
             label="累積 IC (右軸)")
    ax2.set_ylabel("累積 IC", color="#238b45")
    ax.set(title="土洋分歧訊號 橫斷面 rank IC (大型, 持有1M)", xlabel="", ylabel="月 IC")
    ax.legend(loc="upper left")
    ps.save(fig, "bt_ic_timeseries.png")

    # ---- 圖 4: IC by horizon (大型 vs 全市場) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    import seaborn as sns
    sns.barplot(data=ic_tbl, x="horizon", y="IC平均", hue="母體", ax=ax)
    ax.axhline(0, color="k", lw=.8)
    ax.set(title="平均 rank IC by 持有期間", xlabel="持有期間", ylabel="平均 IC")
    ps.save(fig, "bt_ic_horizon.png")


if __name__ == "__main__":
    main()
