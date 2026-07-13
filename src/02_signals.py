"""
Step 2 — 對作訊號 (signals_monthly.parquet)

主定義 (流量法): sign(外資月買賣超) != sign(投信月買賣超), 兩邊皆非零。
  - 洋買土賣 (opp_FbTs): 外資買、投信賣
  - 洋賣土買 (opp_FsTb): 外資賣、投信買
規模門檻: 兩邊 |買賣超/流通股數| >= MIN_FLOW_RATIO 才算「顯著對作」。
穩健性 (持股率法): sign(Δ外資持股率) != sign(Δ投信持股率)。
另標記對照組: 同買/同賣/單邊/無動作, 供事件研究比較。
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import config as C


def categorize(f, t):
    """f, t = 外資/投信 買賣超方向 (+1/0/-1) -> 類別字串"""
    if f > 0 and t < 0:
        return "洋買土賣"
    if f < 0 and t > 0:
        return "洋賣土買"
    if f > 0 and t > 0:
        return "同買"
    if f < 0 and t < 0:
        return "同賣"
    if f == 0 and t == 0:
        return "無動作"
    return "單邊"


def main():
    p = pd.read_parquet(C.PANEL)

    f_dir = np.sign(p["foreign_net_kshr"].fillna(0)).astype(int)
    t_dir = np.sign(p["trust_net_kshr"].fillna(0)).astype(int)
    p["foreign_dir"] = f_dir
    p["trust_dir"] = t_dir

    # 類別
    p["category"] = [categorize(f, t) for f, t in zip(f_dir, t_dir)]

    # 對作 flag (流量法, 未設門檻)
    p["opp_flow"] = p["category"].isin(["洋買土賣", "洋賣土買"])

    # 顯著對作 (加規模門檻: 兩邊都夠大)
    big = (p["foreign_net_ratio"].abs() >= C.MIN_FLOW_RATIO) & \
          (p["trust_net_ratio"].abs() >= C.MIN_FLOW_RATIO)
    p["opp_flow_sig"] = p["opp_flow"] & big

    # 外資方向 (對作時 +1=外資買 / -1=外資賣); 用於事件研究「跟誰」
    p["foreign_side"] = np.where(p["opp_flow"], f_dir, 0)

    # 穩健性: 持股率法對作
    dfh = np.sign(p["d_foreign_hold"].fillna(0)).astype(int)
    dth = np.sign(p["d_trust_hold"].fillna(0)).astype(int)
    p["opp_hold"] = (dfh != 0) & (dth != 0) & (dfh != dth)

    p.to_parquet(C.SIGNALS, index=False)

    # ---- 摘要 ----
    n = len(p)
    print(f"signals rows={n:,}\n")
    print("category 分布:")
    print((p["category"].value_counts(normalize=True) * 100).round(2).to_string())
    print(f"\n對作(流量, 未設門檻) 佔比: {p['opp_flow'].mean():.2%}")
    print(f"對作(流量 + 規模門檻>={C.MIN_FLOW_RATIO:.1%}): {p['opp_flow_sig'].mean():.2%}")
    print(f"對作(持股率法) 佔比: {p['opp_hold'].mean():.2%}")

    # 流量法 vs 持股率法一致性 (在有持股率變化資料的列)
    m = p["d_foreign_hold"].notna() & p["d_trust_hold"].notna()
    ct = pd.crosstab(p.loc[m, "opp_flow"], p.loc[m, "opp_hold"],
                     rownames=["流量對作"], colnames=["持股率對作"])
    print("\n流量法 vs 持股率法 一致性 (交叉表):")
    print(ct.to_string())
    agree = (p.loc[m, "opp_flow"] == p.loc[m, "opp_hold"]).mean()
    print(f"一致率: {agree:.1%}")
    print("\nsaved ->", C.SIGNALS)


if __name__ == "__main__":
    main()
