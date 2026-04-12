# -*- coding: utf-8 -*-
"""
A类 + ER甜点 + 偏离<0.3ATR 稳健性验证
怀疑点：是否被某段时间/某些品种/少数大肉单撑起来的

检验维度：
1. 时间窗口（30/60/90/120天分别看）
2. 逐月拆分（看是哪个月赚的）
3. 逐品种（已知56%正向，细看分布）
4. 去极值（砍掉top5%大肉后还剩多少）
5. 细分0.1~0.2 vs 0.2~0.3（哪段在撑）
6. 对比场景1作为基线
"""

import pandas as pd
import numpy as np
from data_loader import load_all, BARS_PER_DAY, tick_size, sym_name
from signal_core import (SignalDetector, ExitTracker, classify_scenario,
                         DEFAULT_STOP_TICKS)


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
    return (s > 0).sum() / len(s) * 100 if len(s) > 0 else 0

def stats(label, pnl, indent=4):
    s = pnl.dropna()
    if len(s) < 3:
        return f"{' '*indent}{label}: N={len(s)} (不足)"
    e = ev(s)
    ev_str = f"EV={e:.2f}" if e else "N/A"
    w = wr(s)
    return (f"{' '*indent}{label}: N={len(s)}, avg={s.mean():.3f}%, "
            f"cum={s.sum():.1f}%, WR={w:.0f}%, {ev_str}")


def collect_with_time():
    """收集信号，带时间戳，方便按时间拆分"""
    all_data = load_all(period_min=10, days=170, last_days=None,
                        emas=(5, 10, 20, 120), er_periods=(20,), atr_period=14)
    print(f"加载品种数: {len(all_data)}")

    records = []
    for sym_key, df in sorted(all_data.items()):
        n = len(df)
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
            if result is None:
                continue
            if result.signal_type != 'A':
                continue

            er20 = row.get('er_20', 0)
            if pd.isna(er20):
                er20 = 0
            if er20 < 0.5 or er20 >= 0.7:
                continue

            atr = row['atr']
            dev = abs(result.entry_price - row['ema10']) / atr if atr > 0 else 0

            # 跑S6出场
            tracker = ExitTracker(
                direction=result.direction,
                entry_price=result.entry_price,
                pullback_extreme=result.pullback_extreme,
                tick_size=ts, stop_ticks=DEFAULT_STOP_TICKS,
                ema5_strategies=True,
            )
            s6_pnl, s6_bars = None, None
            for j in range(i + 1, n):
                bar = df.iloc[j]
                prev_bar = df.iloc[j - 1]
                if pd.isna(bar['ema10']):
                    continue
                ema5_val = bar.get('ema5', None)
                if pd.isna(ema5_val):
                    ema5_val = None
                if not tracker.all_done():
                    evts, _ = tracker.process_bar(
                        close=bar['close'], high=bar['high'], low=bar['low'],
                        ema10=bar['ema10'], prev_close=prev_bar['close'],
                        prev_high=prev_bar['high'], prev_low=prev_bar['low'],
                        ema5=ema5_val,
                    )
                    for e in evts:
                        if e.strategy == 'S6' and s6_pnl is None:
                            s6_pnl, s6_bars = e.pnl_pct, e.bars_held
                if tracker.all_done():
                    break

            forced = tracker.force_close(df.iloc[-1]['close'])
            for e in forced:
                if e.strategy == 'S6' and s6_pnl is None:
                    s6_pnl, s6_bars = e.pnl_pct, e.bars_held

            # 距数据末尾天数（用于时间窗口筛选）
            days_from_end = (n - 1 - i) / BARS_PER_DAY

            records.append({
                'symbol': sym_name(sym_key),
                'sym_key': sym_key,
                'datetime': row['datetime'] if 'datetime' in row.index else None,
                'bar_idx': i,
                'days_from_end': days_from_end,
                'er20': er20,
                'dev_atr': dev,
                'direction': result.direction,
                's6_pnl': s6_pnl,
                's6_bars': s6_bars,
            })

    return pd.DataFrame(records)


def main():
    print("收集 A类+ER甜点 全部信号...")
    df = collect_with_time()
    print(f"共 {len(df)} 笔 A类甜点区信号\n")

    # 目标组 vs 对照组
    target = df[df['dev_atr'] < 0.3]
    s1_orig = df[df['dev_atr'] >= 1.0]
    mid_zone = df[(df['dev_atr'] >= 0.3) & (df['dev_atr'] < 1.0)]

    # ================================================================
    # 1. 基线对比
    # ================================================================
    print("=" * 70)
    print("1. 基线对比")
    print("=" * 70)
    print(stats("目标: 偏离<0.3 S6 ", target['s6_pnl']))
    print(stats("场景1: 偏离>=1.0 S6", s1_orig['s6_pnl']))
    print(stats("中间: 0.3~1.0 S6  ", mid_zone['s6_pnl']))

    # ================================================================
    # 2. 细分0~0.1 / 0.1~0.2 / 0.2~0.3
    # ================================================================
    print(f"\n{'=' * 70}")
    print("2. 偏离度细分（看哪段在撑）")
    print("=" * 70)
    for lo, hi in [(0, 0.1), (0.1, 0.15), (0.15, 0.2), (0.2, 0.25), (0.25, 0.3)]:
        seg = target[(target['dev_atr'] >= lo) & (target['dev_atr'] < hi)]
        print(stats(f"偏离 {lo}~{hi}", seg['s6_pnl']))

    # ================================================================
    # 3. 时间窗口稳健性
    # ================================================================
    print(f"\n{'=' * 70}")
    print("3. 时间窗口稳健性（最近N天）")
    print("=" * 70)
    for window in [30, 60, 90, 120]:
        seg = target[target['days_from_end'] <= window]
        s1_seg = s1_orig[s1_orig['days_from_end'] <= window]
        print(f"\n  最近 {window} 天:")
        print(stats(f"目标<0.3", seg['s6_pnl'], 6))
        print(stats(f"场景1>=1.0", s1_seg['s6_pnl'], 6))

    # ================================================================
    # 4. 逐月拆分
    # ================================================================
    print(f"\n{'=' * 70}")
    print("4. 逐月拆分")
    print("=" * 70)
    if 'datetime' in target.columns and target['datetime'].notna().any():
        target_dt = target.copy()
        target_dt['month'] = pd.to_datetime(target_dt['datetime']).dt.to_period('M')
        for month in sorted(target_dt['month'].unique()):
            seg = target_dt[target_dt['month'] == month]
            print(stats(f"{month}", seg['s6_pnl']))
    else:
        # 用days_from_end估算月份
        print("  (无datetime字段，用距末尾天数分段)")
        for lo, hi in [(0, 30), (30, 60), (60, 90), (90, 120)]:
            seg = target[(target['days_from_end'] > lo) & (target['days_from_end'] <= hi)]
            label = f"距今{lo}~{hi}天"
            print(stats(label, seg['s6_pnl']))

    # ================================================================
    # 5. 去极值测试
    # ================================================================
    print(f"\n{'=' * 70}")
    print("5. 去极值（砍掉最赚的N%后还剩多少）")
    print("=" * 70)
    pnl = target['s6_pnl'].dropna().sort_values()
    n = len(pnl)
    for pct in [0, 1, 3, 5, 10]:
        if pct == 0:
            trimmed = pnl
        else:
            cut = int(n * pct / 100)
            trimmed = pnl.iloc[:-cut] if cut > 0 else pnl
        e = ev(trimmed)
        ev_str = f"EV={e:.2f}" if e else "N/A"
        print(f"    砍掉top{pct:2d}%: N={len(trimmed)}, avg={trimmed.mean():.3f}%, "
              f"cum={trimmed.sum():.1f}%, {ev_str}")

    # 也看看是不是底部大亏在拉低
    print()
    for pct in [1, 3, 5]:
        cut = int(n * pct / 100)
        trimmed = pnl.iloc[cut:]  # 砍掉最亏的
        e = ev(trimmed)
        ev_str = f"EV={e:.2f}" if e else "N/A"
        print(f"    砍掉bot{pct:2d}%: N={len(trimmed)}, avg={trimmed.mean():.3f}%, "
              f"cum={trimmed.sum():.1f}%, {ev_str}")

    # ================================================================
    # 6. PnL分布
    # ================================================================
    print(f"\n{'=' * 70}")
    print("6. PnL分布对比")
    print("=" * 70)
    for label, group in [("目标<0.3", target), ("场景1>=1.0", s1_orig)]:
        s = group['s6_pnl'].dropna()
        print(f"\n  {label} (N={len(s)}):")
        print(f"    P10={s.quantile(0.1):.3f}%, P25={s.quantile(0.25):.3f}%, "
              f"P50={s.quantile(0.5):.3f}%, P75={s.quantile(0.75):.3f}%, "
              f"P90={s.quantile(0.9):.3f}%")
        print(f"    大赚(>1%): {(s>1).sum()}笔 ({100*(s>1).sum()/len(s):.1f}%)")
        print(f"    小赚(0~1%): {((s>0)&(s<=1)).sum()}笔 ({100*((s>0)&(s<=1)).sum()/len(s):.1f}%)")
        print(f"    小亏(-1~0%): {((s>=-1)&(s<=0)).sum()}笔 ({100*((s>=-1)&(s<=0)).sum()/len(s):.1f}%)")
        print(f"    大亏(<-1%): {(s<-1).sum()}笔 ({100*(s<-1).sum()/len(s):.1f}%)")

    # ================================================================
    # 7. 多空拆分
    # ================================================================
    print(f"\n{'=' * 70}")
    print("7. 多空拆分")
    print("=" * 70)
    for label, group in [("目标<0.3", target), ("场景1>=1.0", s1_orig)]:
        for d, dname in [('long', '做多'), ('short', '做空')]:
            seg = group[group['direction'] == d]
            print(stats(f"{label} {dname}", seg['s6_pnl']))

    # ================================================================
    # 8. 逐品种正/负详情
    # ================================================================
    print(f"\n{'=' * 70}")
    print("8. 品种分散度详细分析")
    print("=" * 70)
    sym_list = []
    for sym in sorted(target['symbol'].unique()):
        seg = target[target['symbol'] == sym]
        s = seg['s6_pnl'].dropna()
        if len(s) >= 1:
            sym_list.append({
                'sym': sym, 'N': len(s), 'avg': s.mean(),
                'cum': s.sum(), 'wr': wr(s),
            })
    sym_df = pd.DataFrame(sym_list).sort_values('cum', ascending=False)
    pos = (sym_df['avg'] > 0).sum()
    tot = len(sym_df)

    # top5和bot5贡献
    top5_cum = sym_df.head(5)['cum'].sum()
    bot5_cum = sym_df.tail(5)['cum'].sum()
    rest_cum = sym_df.iloc[5:-5]['cum'].sum() if len(sym_df) > 10 else 0
    total_cum = sym_df['cum'].sum()

    print(f"  正向品种: {pos}/{tot} ({100*pos/tot:.0f}%)")
    print(f"  Top5品种贡献: {top5_cum:.1f}% (占总{100*top5_cum/total_cum:.0f}%)")
    print(f"  Bot5品种贡献: {bot5_cum:.1f}%")
    print(f"  中间品种贡献: {rest_cum:.1f}%")
    print(f"\n  去掉Top5后: cum={total_cum-top5_cum:.1f}%, avg={(target['s6_pnl'].dropna().sum()-top5_cum)/(len(target['s6_pnl'].dropna())-sym_df.head(5)['N'].sum()):.3f}%")


if __name__ == '__main__':
    main()
