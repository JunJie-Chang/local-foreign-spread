"""
Step 6 — 大型股「投信勝」為何集中在 2020–2023？歸因深掘

檢查四件事:
  A. 是不是被少數權值股帶動 (集中度: top-N 股票貢獻佔比)
  B. 是不是特定產業 (產業別平均與貢獻)
  C. 是不是被極端值拉動 (mean vs median vs 截尾平均)
  D. 是哪個方向主導 (洋買土賣 vs 洋賣土買)
輸出: outputs/figures/dd_*.png, outputs/tables/dd_*.csv
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

YEARS = [2020, 2021, 2022, 2023]


def save_tab(df, name, index=False):
    C.TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.TAB / name, encoding="utf-8-sig", index=index)
    print("  tab ->", C.TAB / name)


def short_ind(s):
    return s.astype(str).str.replace(r"^[A-Z]\d+\s*", "", regex=True)


def main():
    ps.setup()
    p = pd.read_parquet(C.SIGNALS)
    d = p[(p.opp_flow) & (p.mcap_tier == "大型") & (p.year.isin(YEARS))].copy()
    d["follF"] = d["foreign_side"] * d["fwd_ret_12m"]
    d = d.dropna(subset=["follF"])
    d["ind"] = short_ind(d["tse_industry"])
    N = len(d)
    tot = d["follF"].sum()
    print(f"大型股 2020-2023 對作事件 n={N}, 平均跟外資12M={d['follF'].mean():.2%}, 總和={tot:.1f}")

    # ---------- C. 極端值檢查 ----------
    x = d["follF"]
    trimmed = stats.trim_mean(x, 0.05)
    print(f"\n[C 極端值] mean={x.mean():.2%}  median={x.median():.2%}  "
          f"5%截尾mean={trimmed:.2%}  勝率(跟外資>0)={(x>0).mean():.1%}")

    # ---------- A. 個股集中度 ----------
    g = d.groupby(["coid", "name"])["follF"]
    by_stock = pd.DataFrame({"n": g.size(), "mean": g.mean(), "sum": g.sum()})
    by_stock = by_stock.sort_values("sum")  # 最負 = 拖累最多 (投信最贏)
    neg_sum = by_stock.loc[by_stock["sum"] < 0, "sum"].sum()
    top10 = by_stock.head(10)["sum"].sum()
    print(f"\n[A 集中度] 負貢獻總和={neg_sum:.1f}; 最負前10檔佔負貢獻 {top10/neg_sum:.0%}, "
          f"佔全體總和 {top10/tot:.0%}")
    show = by_stock.head(12).reset_index()
    show["平均跟外資12M%"] = (show["mean"] * 100).round(1)
    save_tab(show[["coid", "name", "n", "平均跟外資12M%", "sum"]], "dd_top_stocks.csv")
    print("最拖累前12檔 (投信最贏):")
    print(show[["coid", "name", "n", "平均跟外資12M%"]].to_string(index=False))

    # ---------- D. 方向拆解 ----------
    dir_tab = (d.groupby(d["foreign_side"].map({1: "洋買土賣", -1: "洋賣土買"}))["fwd_ret_12m"]
               .agg(n="size", 平均報酬="mean", 勝率=lambda s: (s > 0).mean()))
    dir_tab["平均報酬"] = (dir_tab["平均報酬"] * 100).round(2)
    dir_tab["勝率"] = (dir_tab["勝率"] * 100).round(1)
    save_tab(dir_tab.reset_index().rename(columns={"foreign_side": "型態"}), "dd_direction.csv")
    print("\n[D 方向] 2020-2023 大型股各方向後續 12M 報酬:")
    print(dir_tab.to_string())

    # ---------- B. 產業 ----------
    gi = d.groupby("ind")["follF"]
    ind_tab = pd.DataFrame({"n": gi.size(), "平均跟外資12M": gi.mean(), "貢獻sum": gi.sum()})
    ind_tab = ind_tab[ind_tab["n"] >= 30].sort_values("貢獻sum")
    save_tab(ind_tab.reset_index(), "dd_industry.csv", index=True)
    print("\n[B 產業] 貢獻最負(投信最贏)前幾產業:")
    print(ind_tab.assign(平均跟外資12M=(ind_tab["平均跟外資12M"]*100).round(1)).head(8).to_string())

    # ========== 圖 1: 個股貢獻 (最拖累前15) ==========
    b15 = by_stock.head(15).reset_index()
    b15["label"] = b15["coid"] + " " + b15["name"]
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(b15["label"], b15["sum"], color="#2c7fb8")
    ax.invert_yaxis()
    ax.set(title="大型股 2020–2023：跟外資報酬『貢獻最負』前15檔\n(值越負=投信在該股越贏)",
           xlabel="Σ 跟外資12M報酬 (事件加總)", ylabel="")
    ps.save(fig, "dd_top_stocks.png")

    # ========== 圖 2: 產業貢獻 ==========
    it = ind_tab.reset_index()
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#2c7fb8" if v < 0 else "#d95f0e" for v in it["貢獻sum"]]
    ax.barh(it["ind"], it["貢獻sum"], color=colors)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=.8)
    ax.set(title="大型股 2020–2023：跟外資報酬 產業貢獻\n(藍=投信贏 / 橘=外資贏)",
           xlabel="Σ 跟外資12M報酬", ylabel="")
    ps.save(fig, "dd_industry.png")

    # ========== 圖 3: 集中度 (累積貢獻曲線) ==========
    cum = by_stock["sum"].cumsum().reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(cum) + 1), cum.values, lw=2, color="#2c7fb8")
    ax.axhline(tot, color="k", ls="--", lw=1, label=f"全體總和 {tot:.0f}")
    ax.set(title="個股累積貢獻 (由最負排到最正)",
           xlabel="股票數 (由最拖累外資者起算)", ylabel="累積 Σ 跟外資12M報酬")
    ax.legend()
    ps.save(fig, "dd_concentration.png")

    print("\nStep 6 done.")


if __name__ == "__main__":
    main()
