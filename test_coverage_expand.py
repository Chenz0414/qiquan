# -*- coding: utf-8 -*-
"""
覆盖率优化验证：
思路2 — A类甜点区放宽偏离度门槛
思路3 — B类甜点区换S6出场后放宽偏离度
"""

import pandas as pd
import numpy as np

def ev(s):
    s = s.dropna()
    if len(s) < 5:
        return None
    m, sd = s.mean(), s.std()
    if sd == 0:
        return None
    return m / sd * np.sqrt(len(s))

def wr(s):
    s = s.dropna()
    if len(s) == 0:
        return 0
    return (s > 0).sum() / len(s) * 100

def stats_line(label, pnl, indent=4):
    s = pnl.dropna()
    if len(s) < 5:
        return f"{' '*indent}{label}: N={len(s)} (样本不足)"
    e = ev(s)
    ev_str = f"EV={e:.2f}" if e else "N/A"
    return (f"{' '*indent}{label}: N={len(s)}, "
            f"avg={s.mean():.3f}%, cum={s.sum():.1f}%, "
            f"WR={wr(s):.0f}%, {ev_str}")


def main():
    df = pd.read_csv('output/signal_funnel.csv')
    print(f"加载 {len(df)} 笔信号\n")

    sweet = df[(df['er20'] >= 0.5) & (df['er20'] < 0.7)]

    # ================================================================
    # 思路2：A类甜点区，放宽偏离度
    # ================================================================
    print("=" * 70)
    print("思路2：甩尾型（A类）在ER甜点区，不同偏离度区间")
    print("=" * 70)

    a_sweet = sweet[sweet['type'] == 'A']
    print(f"\nA类甜点区总量: {len(a_sweet)} 笔\n")

    # 细分偏离度区间
    bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
            (0.5, 0.7), (0.7, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 99)]
    print("--- 逐区间（S6出场）---")
    for lo, hi in bins:
        seg = a_sweet[(a_sweet['dev_atr'] >= lo) & (a_sweet['dev_atr'] < hi)]
        lab = f"偏离 {lo}~{hi}ATR" if hi < 99 else f"偏离 >={lo}ATR"
        print(stats_line(lab, seg['s6_pnl']))

    # 累计：从不同门槛开始往上做
    print("\n--- 累计（S6出场，从某门槛开始全做）---")
    thresholds = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    for th in thresholds:
        seg = a_sweet[a_sweet['dev_atr'] >= th]
        lab = f"偏离 >= {th}ATR"
        print(stats_line(lab, seg['s6_pnl']))

    # 累计：从上往下，看能降到哪
    print("\n--- 累计（S6出场，上限固定，下限下探）---")
    for th in [0.3, 0.5, 0.7, 1.0]:
        seg = a_sweet[(a_sweet['dev_atr'] >= 0) & (a_sweet['dev_atr'] < th)]
        lab = f"偏离 < {th}ATR（低偏离区）"
        print(stats_line(lab, seg['s6_pnl']))

    # S5.1对比
    print("\n--- 低偏离区 S5.1 vs S6 ---")
    lo_dev = a_sweet[a_sweet['dev_atr'] < 0.3]
    print(stats_line("偏离<0.3 S6  ", lo_dev['s6_pnl']))
    print(stats_line("偏离<0.3 S5.1", lo_dev['s51_pnl']))

    lo_dev2 = a_sweet[a_sweet['dev_atr'] < 0.5]
    print(stats_line("偏离<0.5 S6  ", lo_dev2['s6_pnl']))
    print(stats_line("偏离<0.5 S5.1", lo_dev2['s51_pnl']))

    # 品种维度：看是不是少数品种撑起来的
    print("\n--- A类 ER甜点 偏离<0.3ATR 逐品种（S6）---")
    lo_dev_a = a_sweet[a_sweet['dev_atr'] < 0.3]
    sym_stats = []
    for sym in sorted(lo_dev_a['symbol'].unique()):
        seg = lo_dev_a[lo_dev_a['symbol'] == sym]
        s6 = seg['s6_pnl'].dropna()
        if len(s6) >= 3:
            sym_stats.append({
                'symbol': sym, 'N': len(s6),
                'avg': s6.mean(), 'cum': s6.sum(),
                'wr': wr(s6),
            })
    sym_df = pd.DataFrame(sym_stats).sort_values('cum', ascending=False)
    pos = (sym_df['avg'] > 0).sum()
    neg = (sym_df['avg'] <= 0).sum()
    print(f"    正向品种: {pos}/{pos+neg} ({100*pos/(pos+neg):.0f}%)")
    for _, r in sym_df.iterrows():
        flag = "+" if r['avg'] > 0 else "-"
        print(f"    {flag} {r['symbol']:6s}: N={r['N']:2.0f}, avg={r['avg']:.3f}%, cum={r['cum']:.1f}%, WR={r['wr']:.0f}%")


    # ================================================================
    # 思路3：B类甜点区，换S6后放宽偏离度
    # ================================================================
    print(f"\n{'=' * 70}")
    print("思路3：回踩型（B类）在ER甜点区，S6 vs S5.1 在不同偏离度")
    print("=" * 70)

    b_sweet = sweet[sweet['type'] == 'B']
    print(f"\nB类甜点区总量: {len(b_sweet)} 笔\n")

    # 逐区间对比
    b_bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5),
              (0.5, 1.0), (1.0, 2.0), (2.0, 99)]
    print("--- 逐区间 S6 vs S5.1 ---")
    for lo, hi in b_bins:
        seg = b_sweet[(b_sweet['dev_atr'] >= lo) & (b_sweet['dev_atr'] < hi)]
        lab = f"偏离 {lo}~{hi}ATR" if hi < 99 else f"偏离 >={lo}ATR"
        print(stats_line(f"{lab} S6  ", seg['s6_pnl']))
        print(stats_line(f"{lab} S5.1", seg['s51_pnl']))
        print()

    # 累计
    print("--- 累计（S6出场）---")
    for th in [0, 0.1, 0.2, 0.3, 0.5]:
        seg = b_sweet[b_sweet['dev_atr'] >= th]
        lab = f"偏离 >= {th}ATR"
        print(stats_line(lab, seg['s6_pnl']))

    # B类场景3原本：0.1~0.3用S5.1
    print("\n--- 对比：场景3原方案 vs 放宽方案 ---")
    s3_orig = b_sweet[(b_sweet['dev_atr'] >= 0.1) & (b_sweet['dev_atr'] < 0.3)]
    print(stats_line("原场景3 (0.1~0.3 S5.1)", s3_orig['s51_pnl']))
    print(stats_line("原场景3 (0.1~0.3 S6)  ", s3_orig['s6_pnl']))

    # 扩展方案：0.1~0.5用S6
    s3_expand = b_sweet[(b_sweet['dev_atr'] >= 0.1) & (b_sweet['dev_atr'] < 0.5)]
    print(stats_line("扩展到0.1~0.5 S6      ", s3_expand['s6_pnl']))
    print(stats_line("扩展到0.1~0.5 S5.1    ", s3_expand['s51_pnl']))

    # 全部B类用S6
    print(stats_line("B类全做 S6            ", b_sweet['s6_pnl']))

    # 品种维度
    print(f"\n--- B类 ER甜点 偏离>=0.3 逐品种（S6）---")
    b_hi = b_sweet[b_sweet['dev_atr'] >= 0.3]
    sym_stats_b = []
    for sym in sorted(b_hi['symbol'].unique()):
        seg = b_hi[b_hi['symbol'] == sym]
        s6 = seg['s6_pnl'].dropna()
        if len(s6) >= 2:
            sym_stats_b.append({
                'symbol': sym, 'N': len(s6),
                'avg': s6.mean(), 'cum': s6.sum(),
                'wr': wr(s6),
            })
    if sym_stats_b:
        sym_df_b = pd.DataFrame(sym_stats_b).sort_values('cum', ascending=False)
        pos_b = (sym_df_b['avg'] > 0).sum()
        neg_b = (sym_df_b['avg'] <= 0).sum()
        print(f"    正向品种: {pos_b}/{pos_b+neg_b} ({100*pos_b/(pos_b+neg_b):.0f}%)")
        for _, r in sym_df_b.iterrows():
            flag = "+" if r['avg'] > 0 else "-"
            print(f"    {flag} {r['symbol']:6s}: N={r['N']:2.0f}, avg={r['avg']:.3f}%, cum={r['cum']:.1f}%, WR={r['wr']:.0f}%")


if __name__ == '__main__':
    main()
