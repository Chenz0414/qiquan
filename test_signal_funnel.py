# -*- coding: utf-8 -*-
"""
信号漏斗分析：看每一层过滤砍掉了多少信号，被砍掉的信号EV如何
目标：找到覆盖率优化空间
"""

import pandas as pd
import numpy as np
from data_loader import load_all, BARS_PER_DAY, tick_size, sym_name
from signal_core import (SignalDetector, ExitTracker, classify_scenario,
                         DEFAULT_STOP_TICKS)


def collect_all_signals():
    """收集全部信号（不做任何场景过滤），记录每个信号的属性和S6/S5.1出场结果"""
    all_data = load_all(period_min=10, days=170, last_days=None,
                        emas=(5, 10, 20, 120), er_periods=(20, 40), atr_period=14)
    print(f"加载品种数: {len(all_data)}")

    records = []
    for sym_key, df in sorted(all_data.items()):
        n = len(df)
        signal_start = max(130, n - 120 * BARS_PER_DAY)
        ts = tick_size(sym_key)
        detector = SignalDetector(signal_types='ABC')

        for i in range(max(2, 130), n):
            row = df.iloc[i]
            if pd.isna(row['ema10']) or pd.isna(row['ema20']) or pd.isna(row['ema120']):
                continue
            if pd.isna(row['atr']) or row['atr'] <= 0:
                continue

            result = detector.process_bar(
                close=row['close'], high=row['high'], low=row['low'],
                ema10=row['ema10'], ema20=row['ema20'], ema120=row['ema120'],
            )
            if result is None or i < signal_start:
                continue

            er20 = row.get('er_20', 0)
            if pd.isna(er20):
                er20 = 0
            er40 = row.get('er_40', 0)
            if pd.isna(er40):
                er40 = 0
            atr = row['atr']
            deviation_atr = abs(result.entry_price - row['ema10']) / atr if atr > 0 else 0

            scenario = classify_scenario(result.signal_type, er20, deviation_atr)

            # 跑S6和S5.1出场
            tracker = ExitTracker(
                direction=result.direction,
                entry_price=result.entry_price,
                pullback_extreme=result.pullback_extreme,
                tick_size=ts,
                stop_ticks=DEFAULT_STOP_TICKS,
                ema5_strategies=True,
            )

            exit_results = {}
            for j in range(i + 1, n):
                bar = df.iloc[j]
                prev_bar = df.iloc[j - 1]
                if pd.isna(bar['ema10']):
                    continue
                ema5_val = bar.get('ema5', None)
                if pd.isna(ema5_val):
                    ema5_val = None

                if not tracker.all_done():
                    exit_events, _ = tracker.process_bar(
                        close=bar['close'], high=bar['high'], low=bar['low'],
                        ema10=bar['ema10'], prev_close=prev_bar['close'],
                        prev_high=prev_bar['high'], prev_low=prev_bar['low'],
                        ema5=ema5_val,
                    )
                    for ev in exit_events:
                        if ev.strategy not in exit_results:
                            exit_results[ev.strategy] = (ev.pnl_pct, ev.bars_held)
                if tracker.all_done():
                    break

            forced = tracker.force_close(df.iloc[-1]['close'])
            for ev in forced:
                if ev.strategy not in exit_results:
                    exit_results[ev.strategy] = (ev.pnl_pct, ev.bars_held)

            s6_pnl = exit_results.get('S6', (None, None))[0]
            s51_pnl = exit_results.get('S5.1', (None, None))[0]
            s6_bars = exit_results.get('S6', (None, None))[1]
            s51_bars = exit_results.get('S5.1', (None, None))[1]

            records.append({
                'symbol': sym_name(sym_key),
                'type': result.signal_type,
                'er20': round(er20, 3),
                'er40': round(er40, 3),
                'dev_atr': round(deviation_atr, 3),
                'scenario': scenario,
                'direction': result.direction,
                'pullback_bars': result.pullback_bars,
                's6_pnl': s6_pnl,
                's51_pnl': s51_pnl,
                's6_bars': s6_bars,
                's51_bars': s51_bars,
            })

    return pd.DataFrame(records)


def ev(series):
    """计算EV = mean / std * sqrt(n)"""
    s = series.dropna()
    if len(s) < 5:
        return None
    mean = s.mean()
    std = s.std()
    if std == 0:
        return None
    return mean / std * np.sqrt(len(s))


def analyze_funnel(df):
    """分析信号漏斗"""
    total = len(df)
    print(f"\n{'='*70}")
    print(f"信号漏斗分析（32品种 × 120天）")
    print(f"{'='*70}")

    # --- 第0层：全部信号 ---
    print(f"\n全部信号: {total} 笔")
    for t in ['A', 'B', 'C']:
        sub = df[df['type'] == t]
        print(f"  {t}类: {len(sub)} 笔 ({100*len(sub)/total:.1f}%)")

    # --- 第1层：ER过滤 ---
    print(f"\n{'─'*70}")
    print("第1层: ER(20) 过滤")
    er_low = df[df['er20'] < 0.5]
    er_ok = df[(df['er20'] >= 0.5) & (df['er20'] < 0.7)]
    er_high = df[df['er20'] >= 0.7]
    print(f"  ER < 0.5  : {len(er_low):5d} 笔 ({100*len(er_low)/total:.1f}%) — 被砍（门槛不够）")
    print(f"  ER 0.5~0.7: {len(er_ok):5d} 笔 ({100*len(er_ok)/total:.1f}%) — 甜点区")
    print(f"  ER >= 0.7 : {len(er_high):5d} 笔 ({100*len(er_high)/total:.1f}%) — 被砍（反转区）")

    # ER<0.5被砍的信号，用S6出场的EV
    for t in ['A', 'B', 'C']:
        sub = er_low[er_low['type'] == t]
        s6 = sub['s6_pnl'].dropna()
        if len(s6) >= 10:
            print(f"    {t}类 ER<0.5 用S6: N={len(s6)}, avg={s6.mean():.3f}%, cum={s6.sum():.1f}%, EV={ev(s6):.2f}" if ev(s6) else f"    {t}类 ER<0.5: N={len(s6)}, avg={s6.mean():.3f}%")

    # --- 第2层：按信号类型+偏离度 ---
    print(f"\n{'─'*70}")
    print("第2层: 偏离度过滤（在ER甜点区 0.5~0.7 内）")

    for t, label in [('A', '甩尾型'), ('B', '回踩型'), ('C', '蓄力型')]:
        sub = er_ok[er_ok['type'] == t]
        print(f"\n  {t}类（{label}）在甜点区: {len(sub)} 笔")

        if t == 'A':
            hit = sub[sub['dev_atr'] >= 1.0]
            miss = sub[sub['dev_atr'] < 1.0]
            print(f"    偏离 >= 1.0ATR（场景1命中）: {len(hit)} 笔")
            print(f"    偏离 <  1.0ATR（被砍）     : {len(miss)} 笔")
            # 细分被砍区间
            for lo, hi, lab in [(0, 0.3, '<0.3'), (0.3, 0.5, '0.3~0.5'), (0.5, 1.0, '0.5~1.0')]:
                seg = miss[(miss['dev_atr'] >= lo) & (miss['dev_atr'] < hi)]
                s6 = seg['s6_pnl'].dropna()
                if len(s6) >= 5:
                    e = ev(s6)
                    ev_str = f"EV={e:.2f}" if e else "EV=N/A"
                    print(f"      偏离{lab}: N={len(s6)}, avg={s6.mean():.3f}%, cum={s6.sum():.1f}%, {ev_str}")

        elif t == 'B':
            hit = sub[(sub['dev_atr'] >= 0.1) & (sub['dev_atr'] < 0.3)]
            miss_lo = sub[sub['dev_atr'] < 0.1]
            miss_hi = sub[sub['dev_atr'] >= 0.3]
            print(f"    偏离 0.1~0.3ATR（场景3命中）: {len(hit)} 笔")
            print(f"    偏离 < 0.1ATR（被砍）       : {len(miss_lo)} 笔")
            print(f"    偏离 >= 0.3ATR（被砍）      : {len(miss_hi)} 笔")
            for seg, lab in [(miss_lo, '<0.1'), (miss_hi, '>=0.3')]:
                s51 = seg['s51_pnl'].dropna()
                s6 = seg['s6_pnl'].dropna()
                if len(s6) >= 5:
                    e = ev(s6)
                    ev_str = f"EV={e:.2f}" if e else "EV=N/A"
                    print(f"      偏离{lab} S6: N={len(s6)}, avg={s6.mean():.3f}%, cum={s6.sum():.1f}%, {ev_str}")
                if len(s51) >= 5:
                    e = ev(s51)
                    ev_str = f"EV={e:.2f}" if e else "EV=N/A"
                    print(f"      偏离{lab} S5.1: N={len(s51)}, avg={s51.mean():.3f}%, cum={s51.sum():.1f}%, {ev_str}")

        elif t == 'C':
            # C类不要求ER，单独看
            sub_c_all = df[df['type'] == 'C']  # C类全部（不限ER）
            hit = sub_c_all[sub_c_all['dev_atr'] >= 2.0]
            miss = sub_c_all[sub_c_all['dev_atr'] < 2.0]
            print(f"    [C类看全部，不限ER]")
            print(f"    偏离 >= 2.0ATR（场景2命中）: {len(hit)} 笔")
            print(f"    偏离 <  2.0ATR（被砍）     : {len(miss)} 笔")
            for lo, hi, lab in [(0, 0.5, '<0.5'), (0.5, 1.0, '0.5~1.0'), (1.0, 2.0, '1.0~2.0')]:
                seg = miss[(miss['dev_atr'] >= lo) & (miss['dev_atr'] < hi)]
                s6 = seg['s6_pnl'].dropna()
                if len(s6) >= 5:
                    e = ev(s6)
                    ev_str = f"EV={e:.2f}" if e else "EV=N/A"
                    print(f"      偏离{lab}: N={len(s6)}, avg={s6.mean():.3f}%, cum={s6.sum():.1f}%, {ev_str}")

    # --- 第3层：不在甜点区的信号 ---
    print(f"\n{'─'*70}")
    print("ER<0.5 的信号按偏离度分布（看有没有不靠ER也能做的）")
    for t in ['A', 'B', 'C']:
        sub = er_low[er_low['type'] == t]
        if len(sub) < 10:
            continue
        print(f"\n  {t}类 ER<0.5: {len(sub)} 笔")
        if t == 'A':
            for lo, hi, lab in [(0, 0.5, '<0.5'), (0.5, 1.0, '0.5~1.0'), (1.0, 2.0, '1.0~2.0'), (2.0, 99, '>=2.0')]:
                seg = sub[(sub['dev_atr'] >= lo) & (sub['dev_atr'] < hi)]
                s6 = seg['s6_pnl'].dropna()
                if len(s6) >= 5:
                    e = ev(s6)
                    ev_str = f"EV={e:.2f}" if e else "EV=N/A"
                    print(f"    偏离{lab}: N={len(s6)}, avg={s6.mean():.3f}%, cum={s6.sum():.1f}%, {ev_str}")
        elif t == 'C':
            for lo, hi, lab in [(0, 1.0, '<1.0'), (1.0, 2.0, '1.0~2.0'), (2.0, 3.0, '2.0~3.0'), (3.0, 99, '>=3.0')]:
                seg = sub[(sub['dev_atr'] >= lo) & (sub['dev_atr'] < hi)]
                s6 = seg['s6_pnl'].dropna()
                if len(s6) >= 5:
                    e = ev(s6)
                    ev_str = f"EV={e:.2f}" if e else "EV=N/A"
                    print(f"    偏离{lab}: N={len(s6)}, avg={s6.mean():.3f}%, cum={s6.sum():.1f}%, {ev_str}")
        elif t == 'B':
            for lo, hi, lab in [(0, 0.1, '<0.1'), (0.1, 0.3, '0.1~0.3'), (0.3, 0.5, '0.3~0.5'), (0.5, 99, '>=0.5')]:
                seg = sub[(sub['dev_atr'] >= lo) & (sub['dev_atr'] < hi)]
                s6 = seg['s6_pnl'].dropna()
                s51 = seg['s51_pnl'].dropna()
                if len(s6) >= 5:
                    e = ev(s6)
                    ev_str = f"EV={e:.2f}" if e else "EV=N/A"
                    print(f"    偏离{lab} S6: N={len(s6)}, avg={s6.mean():.3f}%, cum={s6.sum():.1f}%, {ev_str}")
                if len(s51) >= 5:
                    e = ev(s51)
                    ev_str = f"EV={e:.2f}" if e else "EV=N/A"
                    print(f"    偏离{lab} S5.1: N={len(s51)}, avg={s51.mean():.3f}%, cum={s51.sum():.1f}%, {ev_str}")

    # --- 汇总 ---
    print(f"\n{'='*70}")
    print("汇总")
    covered = df[df['scenario'].notna()]
    uncovered = df[df['scenario'].isna()]
    print(f"  场景命中: {len(covered)} 笔 ({100*len(covered)/total:.1f}%)")
    print(f"  未覆盖  : {len(uncovered)} 笔 ({100*len(uncovered)/total:.1f}%)")

    # 未覆盖信号按类型×ER区间×偏离区间 top组合
    print(f"\n未覆盖信号中，样本>=20且S6 avg>0 的组合:")
    uncovered_groups = []
    for t in ['A', 'B', 'C']:
        for er_lo, er_hi, er_lab in [(0, 0.3, 'ER<0.3'), (0.3, 0.5, 'ER0.3~0.5'), (0.5, 0.7, 'ER0.5~0.7')]:
            for dv_lo, dv_hi, dv_lab in [(0, 0.1, 'dev<0.1'), (0.1, 0.3, 'dev0.1~0.3'),
                                          (0.3, 0.5, 'dev0.3~0.5'), (0.5, 1.0, 'dev0.5~1.0'),
                                          (1.0, 2.0, 'dev1.0~2.0'), (2.0, 99, 'dev>=2.0')]:
                seg = uncovered[(uncovered['type'] == t) &
                                (uncovered['er20'] >= er_lo) & (uncovered['er20'] < er_hi) &
                                (uncovered['dev_atr'] >= dv_lo) & (uncovered['dev_atr'] < dv_hi)]
                s6 = seg['s6_pnl'].dropna()
                s51 = seg['s51_pnl'].dropna()
                if len(s6) >= 20 and s6.mean() > 0:
                    uncovered_groups.append({
                        'combo': f"{t}类 {er_lab} {dv_lab}",
                        'N': len(s6),
                        'avg': s6.mean(),
                        'cum': s6.sum(),
                        'EV': ev(s6),
                        'exit': 'S6',
                    })
                if len(s51) >= 20 and s51.mean() > 0:
                    uncovered_groups.append({
                        'combo': f"{t}类 {er_lab} {dv_lab}",
                        'N': len(s51),
                        'avg': s51.mean(),
                        'cum': s51.sum(),
                        'EV': ev(s51),
                        'exit': 'S5.1',
                    })

    if uncovered_groups:
        ug = pd.DataFrame(uncovered_groups).sort_values('cum', ascending=False)
        for _, r in ug.iterrows():
            ev_str = f"EV={r['EV']:.2f}" if r['EV'] else "EV=N/A"
            print(f"  {r['combo']} ({r['exit']}): N={r['N']}, avg={r['avg']:.3f}%, cum={r['cum']:.1f}%, {ev_str}")
    else:
        print("  （无）")


if __name__ == '__main__':
    print("收集全部信号（含出场模拟）...")
    df = collect_all_signals()
    print(f"共收集 {len(df)} 笔信号")
    df.to_csv('output/signal_funnel.csv', index=False, encoding='utf-8-sig')
    analyze_funnel(df)
