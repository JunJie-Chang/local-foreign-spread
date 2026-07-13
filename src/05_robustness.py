"""
Step 5 — 穩健性深掘 + 絕對勝率 + event-time 累積報酬曲線

1) 大型股「跟外資」報酬 by 年度、by 產業 (檢驗核心結論是否穩健)
2) 外資買方 vs 投信買方 的絕對勝率與平均報酬 (分市值層)
3) event-time 累積還原報酬曲線 (t+1..t+12), 分市值層
輸出: outputs/figures/rob_*.png, outputs/tables/rob_*.csv
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import config as C
import plotstyle as ps

HZ = C.FWD_HORIZONS
PATH_K = list(range(1, 13))  # event-time 月數


def save_tab(df, name, index=False):
    C.TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.TAB / name, encoding="utf-8-sig", index=index)
    print("  tab ->", C.TAB / name)


def short_ind(s):
    return s.astype(str).str.replace(r"^[A-Z]\d+\s*", "", regex=True)


def main():
    ps.setup()
    p = pd.read_parquet(C.SIGNALS)
    p["ymp"] = pd.PeriodIndex(p["ym"], freq="M")
    opp = p[p["opp_flow"]].copy()
    for h in HZ:
        opp[f"foll_F_{h}m"] = opp["foreign_side"] * opp[f"fwd_ret_{h}m"]

    # ================= 1) 大型股穩健性 =================
    big = opp[opp.mcap_tier == "大型"].copy()

    # by 年度 × horizon
    yr = pd.concat([big.groupby("year")[f"foll_F_{h}m"].mean() * 100 for h in HZ],
                   axis=1, keys=[f"{h}m" for h in HZ])
    save_tab(yr.reset_index(), "rob_largecap_year.csv")
    # 逐年 12m 為正(外資對)的年數 vs 為負(投信對)
    neg_years = (yr["12m"] < 0).sum()
    print(f"大型股: 16 年中有 {neg_years} 年『跟外資 12M 報酬為負』(投信占上風)")

    fig, ax = plt.subplots(figsize=(8, 11))
    sns.heatmap(yr, annot=True, fmt=".1f", center=0, cmap="RdBu_r",
                cbar_kws={"label": "跟外資報酬 %"}, ax=ax)
    ax.set(title="大型股逐年『跟外資』報酬 (%)\n(藍=投信對 / 紅=外資對)", xlabel="持有期間", ylabel="")
    ps.save(fig, "rob_largecap_year.png")

    # by 產業 (12m, 樣本 >= 100)
    big["ind"] = short_ind(big["tse_industry"])
    gi = big.groupby("ind")["foll_F_12m"]
    ind_tbl = pd.DataFrame({"n": gi.size(), "跟外資12m報酬%": gi.mean() * 100})
    ind_tbl = ind_tbl[ind_tbl["n"] >= 100].sort_values("跟外資12m報酬%")
    save_tab(ind_tbl.reset_index(), "rob_largecap_industry.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 11))
    colors = ["#2c7fb8" if v < 0 else "#d95f0e" for v in ind_tbl["跟外資12m報酬%"]]
    ax.barh(ind_tbl.index, ind_tbl["跟外資12m報酬%"], color=colors)
    ax.axvline(0, color="k", lw=.8)
    ax.set(title="大型股各產業『跟外資』12M 報酬 (%)\n(藍=投信對 / 橘=外資對)", xlabel="跟外資報酬 %", ylabel="")
    ps.save(fig, "rob_largecap_industry.png")

    # ================= 2) 絕對勝率: 外資買方 vs 投信買方 =================
    # 洋買土賣 = 外資是買方; 洋賣土買 = 投信是買方。買方勝率 = P(fwd_ret>0)
    rows = []
    for h in HZ:
        col = f"fwd_ret_{h}m"
        for label, cat in [("外資買方", "洋買土賣"), ("投信買方", "洋賣土買")]:
            for tier in C.MCAP_TIER_LABELS + ["全部"]:
                sub = p[p.category == cat]
                if tier != "全部":
                    sub = sub[sub.mcap_tier == tier]
                x = sub[col].dropna()
                if len(x) < 20:
                    continue
                rows.append(dict(買方=label, 市值層=tier, horizon=f"{h}m", n=len(x),
                                 勝率=(x > 0).mean(), 平均報酬=x.mean()))
    win = pd.DataFrame(rows)
    save_tab(win, "rob_winrate.csv")

    sub = win[win["市值層"] == "全部"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(data=sub, x="horizon", y="勝率", hue="買方", ax=axes[0])
    axes[0].axhline(0.5, color="k", lw=.8, ls="--")
    axes[0].set(title="絕對勝率 P(報酬>0)", xlabel="持有期間", ylabel="勝率")
    sns.barplot(data=sub.assign(平均報酬=sub["平均報酬"] * 100),
                x="horizon", y="平均報酬", hue="買方", ax=axes[1])
    axes[1].axhline(0, color="k", lw=.8)
    axes[1].set(title="買方後續平均報酬", xlabel="持有期間", ylabel="平均報酬 %")
    ps.save(fig, "rob_winrate.png")

    # ================= 3) event-time 累積報酬曲線 =================
    s_close = p.set_index(["coid", "ymp"])["close_adj"]
    cum = {}
    for k in PATH_K:
        tgt = pd.MultiIndex.from_arrays([opp["coid"].values, opp["ymp"].values + k])
        r = s_close.reindex(tgt).values / opp["close_adj"].values - 1.0
        cum[k] = opp["foreign_side"].values * r  # 跟外資
    cumdf = pd.DataFrame(cum, index=opp.index)
    cumdf["mcap_tier"] = opp["mcap_tier"].values

    curve = cumdf.groupby("mcap_tier", observed=True)[PATH_K].mean() * 100
    curve = curve.reindex(C.MCAP_TIER_LABELS)
    save_tab(curve.T.reset_index().rename(columns={"index": "month"}), "rob_cumpath.csv")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for tier in C.MCAP_TIER_LABELS:
        ax.plot(PATH_K, curve.loc[tier].values, marker="o", label=tier, lw=2)
    ax.axhline(0, color="k", lw=.8)
    ax.set(title="對作後『跟外資』累積還原報酬 (event-time)",
           xlabel="對作後月數", ylabel="累積報酬 %")
    ax.legend(title="市值層")
    ax.set_xticks(PATH_K)
    ps.save(fig, "rob_cumpath.png")

    print("\nStep 5 done.")


if __name__ == "__main__":
    main()
