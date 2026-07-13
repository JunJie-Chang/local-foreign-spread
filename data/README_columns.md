# 資料欄位對照 (parquet)

所有檔 key：`coid` 證券代號、`name` 名稱、`ym` 年月 (Period `YYYY-MM` 字串)。
`date` 為原始月底日期字串 (僅 institutional/margin 有)。

## institutional.parquet (買賣超, 2010-01~2025-12, 1941 檔)
| 欄位 | 意義 | 單位 |
|---|---|---|
| foreign_net_kshr | 外資買賣超 | 千股 (流量) |
| foreign_net_mn | 外資買賣超市值 | 百萬元 |
| trust_net_kshr | 投信買賣超 | 千股 (流量) |
| trust_net_mn | 投信買賣超市值 | 百萬元 |
| dealer_net_kshr / _mn | 自營商(自行)買賣超 | 千股 / 百萬 |
| dealer_hedge_kshr / _mn | 自營商避險買賣超 | 千股 / 百萬 (~30% NA) |
| foreign_hold_pct | 外資總持股率 | % (存量/水位) |
| trust_hold_pct | 投信持股率 | % (存量/水位) |

## margin.parquet (融資融券, 1825 檔)
| margin_long_lot / _kd | 融資餘額 | 張 / 千元 |
| margin_short_lot / _kd | 融券餘額 | 張 / 千元 |
| short_long_ratio | 券資比 | (~8% NA) |

## price_unadj.parquet (未調整股價)
| open/high/low/close | 月開高低收 | 元 |
| vol_mnshr | 月成交量 | 百萬股 |
| shares_out_kshr | 流通在外股數 | 千股 |
| mktcap_mn | 市值 | 百萬元 |
| turnover_pct | 週轉率 | % |

## price_adj.parquet (調整/還原股價 — 算報酬用)
| open_adj/high_adj/low_adj/close_adj | 還原月開高低收 | 元 |
| vol_mnshr, shares_out_kshr, mktcap_mn, turnover_pct | 同上 | |

## industry.parquet (產業分類, static)
| tse_industry | TSE 產業名 | |
| tej_industry | TEJ 產業名 | |
| board | 上市別 | TSE / OTC |
