"""
Step 8 — 可執行性檢驗：交易成本 + 訊號衰退歸因

A. 成本/換股頻率敏感度：對 L / LS 策略計算實際週轉率, 扣不同來回成本 (0.3/0.45/0.6%),
   比較月頻 vs 季頻換股後的淨年化報酬、淨 alpha。
B. 分期衰退：把 2010-2019 / 2020-2023 / 2024-2025 三段的 大型股 rank IC 與 L-S 淨值分開看。
C. 2024-2025 反轉歸因：哪些產業/個股讓訊號翻車 (外資變對)。
輸出: outputs/figures/cd_*.png, outputs/tables/cd_*.csv
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import config as C
import plotstyle as ps

COST_RT = [0.0, 0.003, 0.0045, 0.006]   # 來回成本 (含手續費+證交稅)


def short_ind(s):
    return s.astype(str).str.replace(r"^[A-Z]\d+\s*", "", regex=True)


def leg_panel(big, cat, retcol):
    """回傳 (每月報酬 Series, 每月單邊週轉率 Series) for 等權籃子。"""
    sub = big[big.category == cat].dropna(subset=[retcol])
    ret = sub.groupby("ymp")[retcol].mean()
    # 會員矩陣 -> 等權重 -> L1 週轉
    memb = (sub.assign(one=1).pivot_table(index="ymp", columns="coid",
            values="one", aggfunc="max", fill_value=0))
    w = memb.div(memb.sum(axis=1), axis=0)
    turn = 0.5 * w.diff().abs().sum(axis=1)   # 單邊週轉率
    turn.iloc[0] = 1.0                         # 首月建倉
    return ret, turn.reindex(ret.index)


def ann(mean_p, per):
    return mean_p * per


def capm_alpha(rp, rb, per):
    m = rp.notna() & rb.notna()
    x, y = rb[m].values, rp[m].values
    r = stats.linregress(x, y)
    return r.intercept * per, r.slope


def main():
    ps.setup()
    p = pd.read_parquet(C.SIGNALS)
    p["ymp"] = pd.PeriodIndex(p["ym"], freq="M")
    p = p.dropna(subset=["mktcap_mn"])
    big = p[p.mcap_tier == "大型"].copy()

    def capwt(g, ret):
        return np.average(g[ret].dropna(),
                          weights=g.loc[g[ret].notna(), "mktcap_mn"])
    bench_m = big.dropna(subset=["fwd_ret_1m"]).groupby("ymp").apply(
        lambda g: capwt(g, "fwd_ret_1m"), include_groups=False)
    bench_q = big.dropna(subset=["fwd_ret_3m"]).groupby("ymp").apply(
        lambda g: capwt(g, "fwd_ret_3m"), include_groups=False)

    # ---------- A. 成本 / 頻率敏感度 ----------
    # 月頻: 持有 1M; 季頻: 每 3 個月用當月訊號建倉, 持有 3M (fwd_ret_3m)
    Lm, LmT = leg_panel(big, "洋賣土買", "fwd_ret_1m")
    Sm, SmT = leg_panel(big, "洋買土賣", "fwd_ret_1m")
    LSm = (Lm - Sm).dropna()

    Lq_all, LqT_all = leg_panel(big, "洋賣土買", "fwd_ret_3m")
    Sq_all, SqT_all = leg_panel(big, "洋買土賣", "fwd_ret_3m")
    q_months = [m for i, m in enumerate(sorted(Lq_all.index)) if i % 3 == 0]
    Lq, Sq = Lq_all.reindex(q_months), Sq_all.reindex(q_months)
    LqT, SqT = LqT_all.reindex(q_months), SqT_all.reindex(q_months)
    LSq = (Lq - Sq).dropna()

    rows = []
    for c in COST_RT:
        # 月頻 LS: 兩腿週轉都要成本
        net_LSm = (Lm - c * LmT) - (Sm + 0 * SmT)  # placeholder, recompute below
        net_LSm = LSm - c * (LmT.reindex(LSm.index).fillna(0)
                             + SmT.reindex(LSm.index).fillna(0))
        # 月頻 純多 L
        net_Lm = (Lm - c * LmT).dropna()
        # 季頻 LS / L
        net_LSq = LSq - c * (LqT.reindex(LSq.index).fillna(0)
                             + SqT.reindex(LSq.index).fillna(0))
        net_Lq = (Lq - c * LqT).dropna()
        a_LSm, _ = capm_alpha(net_LSm, bench_m.reindex(net_LSm.index), 12)
        a_LSq, _ = capm_alpha(net_LSq, bench_q.reindex(net_LSq.index), 4)
        rows.append(dict(來回成本=f"{c*100:.2f}%",
                         月頻LS年化=ann(net_LSm.mean(), 12),
                         月頻LS_alpha=a_LSm,
                         月頻純多L年化=ann(net_Lm.mean(), 12),
                         季頻LS年化=ann(net_LSq.mean(), 4),
                         季頻LS_alpha=a_LSq,
                         季頻純多L年化=ann(net_Lq.mean(), 4)))
    cost_tbl = pd.DataFrame(rows)
    for col in cost_tbl.columns[1:]:
        cost_tbl[col] = (cost_tbl[col] * 100).round(2)
    cost_tbl.to_csv(C.TAB / "cd_cost_sensitivity.csv", encoding="utf-8-sig", index=False)
    print(f"月頻 LS 平均單邊週轉(多腿)={LmT.mean():.0%}, 季頻={LqT.mean():.0%}")
    print("\n=== A. 成本/頻率敏感度 (年化 %, alpha=扣beta) ===")
    print(cost_tbl.to_string(index=False))

    # 圖: 淨值 gross vs 0.45% 成本 (月頻 vs 季頻 LS)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for series, T1, T2, per, lab, c, ls in [
        (LSm, LmT, SmT, 12, "月頻 LS 毛", "#2c7fb8", "-"),
        (LSm, LmT, SmT, 12, "月頻 LS 淨(0.45%)", "#2c7fb8", "--"),
        (LSq, LqT, SqT, 4, "季頻 LS 毛", "#d95f0e", "-"),
        (LSq, LqT, SqT, 4, "季頻 LS 淨(0.45%)", "#d95f0e", "--")]:
        cc = 0.0045 if "淨" in lab else 0.0
        net = series - cc * (T1.reindex(series.index).fillna(0)
                             + T2.reindex(series.index).fillna(0))
        nav = (1 + net.fillna(0)).cumprod()
        ax.plot(net.index.to_timestamp(), nav, label=lab, color=c, ls=ls, lw=2)
    ax.set(title="多空 LS 淨值：毛 vs 扣 0.45% 來回成本", xlabel="", ylabel="累積淨值")
    ax.legend()
    ps.save(fig, "cd_cost_nav.png")

    # ---------- B. 分期 IC ----------
    def zc(s):
        sd = s.std()
        return (s - s.mean()) / sd if sd and sd == sd else s * np.nan
    p["sig"] = (p.groupby("ymp")["trust_net_ratio"].transform(zc)
                - p.groupby("ymp")["foreign_net_ratio"].transform(zc))
    bigp = p[p.mcap_tier == "大型"]
    ic = (bigp.dropna(subset=["sig", "fwd_ret_1m"]).groupby("ymp")
          .apply(lambda g: g["sig"].corr(g["fwd_ret_1m"], method="spearman")
                 if len(g) >= 15 else np.nan, include_groups=False).dropna())
    ic.index = ic.index.to_timestamp()
    segs = {"2010–2019": ("2010", "2019"), "2020–2023": ("2020", "2023"),
            "2024–2025": ("2024", "2025")}
    rows = []
    for name, (a, b) in segs.items():
        s = ic[a:b]
        rows.append(dict(期間=name, n月=len(s), 平均IC=round(s.mean(), 4),
                         年化ICIR=round(s.mean() / s.std() * np.sqrt(12), 2),
                         IC為正比例=round((s > 0).mean() * 100, 1)))
    seg_tbl = pd.DataFrame(rows)
    seg_tbl.to_csv(C.TAB / "cd_ic_subperiod.csv", encoding="utf-8-sig", index=False)
    print("\n=== B. 分期 大型股 rank IC (持有1M) ===")
    print(seg_tbl.to_string(index=False))

    # ---------- C. 2024-2025 反轉歸因 ----------
    rec = big[(big.opp_flow) & (big.year >= 2024)].copy()
    rec["follF"] = rec["foreign_side"] * rec["fwd_ret_12m"]
    rec = rec.dropna(subset=["follF"])
    rec["ind"] = short_ind(rec["tse_industry"])
    print(f"\n=== C. 2024-2025 大型股對作 n={len(rec)}, 平均跟外資12M={rec['follF'].mean():.2%} "
          f"(>0 表示外資已變對) ===")
    gi = rec.groupby("ind")["follF"]
    ind = pd.DataFrame({"n": gi.size(), "平均跟外資12M": gi.mean()})
    ind = ind[ind["n"] >= 15].sort_values("平均跟外資12M", ascending=False)
    ind.to_csv(C.TAB / "cd_2024_industry.csv", encoding="utf-8-sig")
    print("外資變最對(訊號翻車最兇)的產業:")
    print(ind.assign(平均跟外資12M=(ind["平均跟外資12M"]*100).round(1)).head(8).to_string())

    # 方向拆解: 是多腿(洋賣土買)壞了還是空腿(洋買土賣)?
    dirn = (rec.groupby(rec["foreign_side"].map({1: "洋買土賣(空腿)", -1: "洋賣土買(多腿)"}))
            ["fwd_ret_12m"].agg(n="size", 平均報酬=lambda s: s.mean()*100,
                                勝率=lambda s: (s > 0).mean()*100).round(1))
    dirn.to_csv(C.TAB / "cd_2024_direction.csv", encoding="utf-8-sig")
    print("\n方向拆解 (2024-2025):")
    print(dirn.to_string())

    fig, ax = plt.subplots(figsize=(11, 8))
    it = ind.reset_index()
    colors = ["#d95f0e" if v > 0 else "#2c7fb8" for v in it["平均跟外資12M"]]
    ax.barh(it["ind"], it["平均跟外資12M"] * 100, color=colors)
    ax.invert_yaxis(); ax.axvline(0, color="k", lw=.8)
    ax.set(title="2024–2025 大型股各產業『跟外資』12M 報酬\n(橘>0=外資變對, 訊號翻車)",
           xlabel="跟外資報酬 %", ylabel="")
    ps.save(fig, "cd_2024_industry.png")

    print("\nStep 8 done.")


if __name__ == "__main__":
    main()
