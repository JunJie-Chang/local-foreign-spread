"""
Step 4 — 後續報酬事件研究 (誰對誰錯) + seaborn 圖

對作月之後 1/3/6/12M 還原報酬:
  - 依方向分組 (洋買土賣 / 洋賣土買), 對照 同買/同賣/無動作。
  - 「跟外資」報酬 = foreign_side * fwd_ret (外資買->做多, 外資賣->做空);
     >0 表示站在外資這邊會賺 = 外資對; <0 = 投信對。
  - 統計: n / 平均 / 中位數 / 勝率 / t 檢定; 分市值層與年度穩健性。
輸出: outputs/figures/event_*.png, outputs/tables/event_*.csv
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


def save_tab(df, name):
    C.TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.TAB / name, encoding="utf-8-sig", index=False)
    print("  tab ->", C.TAB / name)


def stat_row(x, label):
    x = pd.Series(x).dropna()
    n = len(x)
    if n < 5:
        return dict(group=label, n=n, mean=np.nan, median=np.nan,
                    hit=np.nan, t=np.nan, p=np.nan)
    t, p = stats.ttest_1samp(x, 0.0)
    return dict(group=label, n=n, mean=x.mean(), median=x.median(),
                hit=(x > 0).mean(), t=t, p=p)


def main():
    ps.setup()
    p = pd.read_parquet(C.SIGNALS)
    p["ymp"] = pd.PeriodIndex(p["ym"], freq="M")

    # 「跟外資」報酬 (僅對作列有意義)
    opp = p[p["opp_flow"]].copy()

    # ===== 1) 各組 forward 報酬統計表 (含對照組) =====
    rows = []
    for h in HZ:
        col = f"fwd_ret_{h}m"
        for cat in ["洋買土賣", "洋賣土買", "同買", "同賣", "無動作"]:
            r = stat_row(p.loc[p.category == cat, col], cat)
            r["horizon"] = f"{h}m"
            rows.append(r)
        # 全市場 baseline
        r = stat_row(p[col], "全市場")
        r["horizon"] = f"{h}m"
        rows.append(r)
    tbl = pd.DataFrame(rows)[["horizon", "group", "n", "mean", "median", "hit", "t", "p"]]
    save_tab(tbl, "event_group_stats.csv")
    print("\n各組後續報酬 (平均, %):")
    print(tbl.assign(mean=(tbl["mean"]*100).round(2), hit=(tbl["hit"]*100).round(1))
          .pivot(index="group", columns="horizon", values="mean").to_string())

    # ===== 2) 誰對誰錯: 跟外資報酬 (整體 + 分市值層) =====
    rows = []
    for h in HZ:
        opp[f"foll_F_{h}m"] = opp["foreign_side"] * opp[f"fwd_ret_{h}m"]
        r = stat_row(opp[f"foll_F_{h}m"], "全部對作"); r["horizon"] = f"{h}m"; rows.append(r)
        for tier in C.MCAP_TIER_LABELS:
            r = stat_row(opp.loc[opp.mcap_tier == tier, f"foll_F_{h}m"], tier)
            r["horizon"] = f"{h}m"; rows.append(r)
    follow = pd.DataFrame(rows)[["horizon", "group", "n", "mean", "median", "hit", "t", "p"]]
    save_tab(follow, "event_follow_foreign.csv")
    print("\n跟外資報酬 (>0=外資對, 平均 %):")
    print(follow.assign(mean=(follow["mean"]*100).round(2))
          .pivot(index="group", columns="horizon", values="mean").to_string())

    # ===== 圖 A: 各組 forward 報酬 bar (分 horizon) =====
    plotcats = ["洋買土賣", "洋賣土買", "同買", "同賣", "全市場"]
    pm = tbl[tbl.group.isin(plotcats)].copy()
    pm["平均報酬%"] = pm["mean"] * 100
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=pm, x="horizon", y="平均報酬%", hue="group",
                hue_order=plotcats, ax=ax)
    ax.axhline(0, color="k", lw=.8)
    ax.set(title="各籌碼型態後續平均還原報酬", xlabel="持有期間", ylabel="平均報酬 %")
    ax.legend(title="", ncol=3)
    ps.save(fig, "event_group_returns.png")

    # ===== 圖 B: 誰對誰錯 (跟外資報酬, 分市值層) =====
    fm = follow[follow.group != "全部對作"].copy()
    fm["跟外資報酬%"] = fm["mean"] * 100
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=fm, x="horizon", y="跟外資報酬%", hue="group",
                hue_order=C.MCAP_TIER_LABELS, ax=ax)
    ax.axhline(0, color="k", lw=.8)
    ax.set(title="站在外資一方的後續報酬 (>0 外資對 / <0 投信對)",
           xlabel="持有期間", ylabel="跟外資報酬 %")
    ax.legend(title="市值層")
    ps.save(fig, "event_who_is_right.png")

    # ===== 圖 C: 報酬分布 violin (3M) =====
    d = p[p.category.isin(["洋買土賣", "洋賣土買", "同買", "同賣", "無動作"])].copy()
    d["fwd_ret_3m%"] = d["fwd_ret_3m"] * 100
    d = d[d["fwd_ret_3m%"].between(-50, 60)]  # 截尾利於視覺
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(data=d, x="category", y="fwd_ret_3m%",
                   order=["洋買土賣", "洋賣土買", "同買", "同賣", "無動作"],
                   hue="category", legend=False, cut=0, ax=ax)
    ax.axhline(0, color="k", lw=.8)
    ax.set(title="3 個月後還原報酬分布 (截尾 ±50%)", xlabel="", ylabel="報酬 %")
    ps.save(fig, "event_dist_3m.png")

    # ===== 圖 D: 年度 × horizon 跟外資報酬 heatmap (時期穩健性) =====
    yr_rows = []
    for h in HZ:
        col = f"foll_F_{h}m"
        gy = opp.groupby("year")[col].mean() * 100
        yr_rows.append(gy.rename(f"{h}m"))
    yr = pd.concat(yr_rows, axis=1)
    save_tab(yr.reset_index(), "event_follow_by_year.csv")
    fig, ax = plt.subplots(figsize=(9, 11))
    sns.heatmap(yr, annot=True, fmt=".1f", center=0, cmap="RdBu_r",
                cbar_kws={"label": "跟外資報酬 %"}, ax=ax)
    ax.set(title="逐年『跟外資』報酬 (%)", xlabel="持有期間", ylabel="")
    ps.save(fig, "event_year_heatmap.png")

    print("\nStep 4 done.")


if __name__ == "__main__":
    main()
