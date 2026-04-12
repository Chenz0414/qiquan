# -*- coding: utf-8 -*-
"""
三个场景的统一稳健性验证
- 去极值（砍top 1/3/5/10%）
- 时间窗口（30/60/90/120天）
- 逐月拆分
- 品种集中度（Top5占比）
- PnL分布
- 多空拆分
"""

import pandas as pd
import numpy as np
from data_loader import load_all, BARS_PER_DAY, tick_size, sym_name
from signal_core import (SignalDetector, ExitTracker, classify_scenario,
                         DEFAULT_STOP_TICKS, SCENARIO_EXIT)


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


def collect_all():
    all_data = load_all(period_min=10, days=170, last_days=None,
                        emas=(5, 10, 20, 120), er_periods=(20,), atr_period=14)
    print(f"品种数: {len(all_data)}")

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
            atr = row['atr']
            dev = abs(result.entry_price - row['ema10']) / atr if atr > 0 else 0
            scenario = classify_scenario(result.signal_type, er20, dev)
            if scenario is None:
                continue

            # 对应出场
            tracker = ExitTracker(
                direction=result.direction,
                entry_price=result.entry_price,
                pullback_extreme=result.pullback_extreme,
                tick_size=ts, stop_ticks=DEFAULT_STOP_TICKS,
                ema5_strategies=True,
            )

            exit_pnl = {}
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
                        if e.strategy not in exit_pnl:
                            exit_pnl[e.strategy] = (e.pnl_pct, e.bars_held)
                if tracker.all_done():
                    break

            forced = tracker.force_close(df.iloc[-1]['close'])
            for e in forced:
                if e.strategy not in exit_pnl:
                    exit_pnl[e.strategy] = (e.pnl_pct, e.bars_held)

            # 取对应场景的出场
            exit_key = SCENARIO_EXIT[scenario]
            pnl_val = exit_pnl.get(exit_key, (None, None))[0]
            bars_val = exit_pnl.get(exit_key, (None, None))[1]

            days_from_end = (n - 1 - i) / BARS_PER_DAY

            records.append({
                'symbol': sym_name(sym_key),
                'scenario': scenario,
                'type': result.signal_type,
                'direction': result.direction,
                'er20': er20,
                'dev_atr': dev,
                'datetime': row.get('datetime', None),
                'days_from_end': days_from_end,
                'pnl': pnl_val,
                'bars': bars_val,
            })

    return pd.DataFrame(records)


def check_scenario(df, scenario_num, name):
    sub = df[df['scenario'] == scenario_num]
    pnl = sub['pnl'].dropna()
    n = len(pnl)

    print(f"\n{'#' * 70}")
    print(f"  场景{scenario_num}: {name}")
    print(f"  N={n}, avg={pnl.mean():.3f}%, cum={pnl.sum():.1f}%, WR={wr(pnl):.0f}%, EV={ev(pnl):.2f}" if ev(pnl) else f"  N={n}")
    print(f"{'#' * 70}")

    # --- 去极值 ---
    print(f"\n  [去极值] 砍掉最赚的N%后:")
    sorted_pnl = pnl.sort_values()
    for pct in [0, 1, 3, 5, 10, 20]:
        cut = int(n * pct / 100)
        trimmed = sorted_pnl.iloc[:-cut] if cut > 0 else sorted_pnl
        e = ev(trimmed)
        ev_str = f"EV={e:.2f}" if e else "N/A"
        mark = ""
        if pct > 0 and trimmed.mean() < 0:
            mark = " << 变负!"
        print(f"    top{pct:2d}%: N={len(trimmed)}, avg={trimmed.mean():.3f}%, cum={trimmed.sum():.1f}%, {ev_str}{mark}")

    # --- 时间窗口 ---
    print(f"\n  [时间窗口]")
    for w in [30, 60, 90, 120]:
        seg = sub[sub['days_from_end'] <= w]
        s = seg['pnl'].dropna()
        if len(s) >= 3:
            e = ev(s)
            ev_str = f"EV={e:.2f}" if e else "N/A"
            print(f"    最近{w:3d}天: N={len(s)}, avg={s.mean():.3f}%, cum={s.sum():.1f}%, WR={wr(s):.0f}%, {ev_str}")
        else:
            print(f"    最近{w:3d}天: N={len(s)} (不足)")

    # --- 逐月 ---
    print(f"\n  [逐月拆分]")
    if 'datetime' in sub.columns and sub['datetime'].notna().any():
        sub_dt = sub.copy()
        sub_dt['month'] = pd.to_datetime(sub_dt['datetime']).dt.to_period('M')
        monthly = []
        for month in sorted(sub_dt['month'].unique()):
            seg = sub_dt[sub_dt['month'] == month]
            s = seg['pnl'].dropna()
            if len(s) > 0:
                monthly.append({'month': str(month), 'N': len(s), 'cum': s.sum(), 'avg': s.mean()})
                print(f"    {month}: N={len(s):3d}, avg={s.mean():.3f}%, cum={s.sum():.1f}%")
        if monthly:
            m_df = pd.DataFrame(monthly)
            pos_months = (m_df['cum'] > 0).sum()
            tot_months = len(m_df[m_df['N'] >= 3])  # 只算有足够样本的月
            m_df_enough = m_df[m_df['N'] >= 3]
            pos_m = (m_df_enough['cum'] > 0).sum()
            print(f"    赚钱月份: {pos_m}/{len(m_df_enough)} (样本>=3的月)")
            # 单月集中度
            if len(m_df) > 0:
                top_month = m_df.loc[m_df['cum'].idxmax()]
                print(f"    最赚月: {top_month['month']} +{top_month['cum']:.1f}% (占总{100*top_month['cum']/pnl.sum():.0f}%)")

    # --- 品种集中度 ---
    print(f"\n  [品种集中度]")
    sym_list = []
    for sym in sorted(sub['symbol'].unique()):
        s = sub[sub['symbol'] == sym]['pnl'].dropna()
        if len(s) >= 1:
            sym_list.append({'sym': sym, 'N': len(s), 'avg': s.mean(), 'cum': s.sum(), 'wr': wr(s)})
    sym_df = pd.DataFrame(sym_list).sort_values('cum', ascending=False)
    pos = (sym_df['avg'] > 0).sum()
    tot = len(sym_df)
    total_cum = pnl.sum()

    print(f"    正向品种: {pos}/{tot} ({100*pos/tot:.0f}%)")
    if len(sym_df) >= 5:
        top5 = sym_df.head(5)
        bot5 = sym_df.tail(5)
        top5_cum = top5['cum'].sum()
        print(f"    Top5品种: {top5_cum:.1f}% (占总{100*top5_cum/total_cum:.0f}%)")
        print(f"    Top5: {', '.join(top5['sym'].values)}")
        rest_n = len(pnl) - top5['N'].sum()
        rest_cum = total_cum - top5_cum
        if rest_n > 0:
            print(f"    去Top5后: N={rest_n}, cum={rest_cum:.1f}%, avg={rest_cum/rest_n:.3f}%")
            mark = " << 变负!" if rest_cum < 0 else ""
            print(f"    {mark}")
    else:
        for _, r in sym_df.iterrows():
            flag = "+" if r['avg'] > 0 else "-"
            print(f"    {flag} {r['sym']}: N={r['N']:.0f}, avg={r['avg']:.3f}%, cum={r['cum']:.1f}%")

    # --- PnL分布 ---
    print(f"\n  [PnL分布]")
    print(f"    P10={pnl.quantile(0.1):.3f}%, P25={pnl.quantile(0.25):.3f}%, "
          f"P50={pnl.quantile(0.5):.3f}%, P75={pnl.quantile(0.75):.3f}%, "
          f"P90={pnl.quantile(0.9):.3f}%")
    print(f"    大赚(>1%): {(pnl>1).sum()}笔 ({100*(pnl>1).sum()/n:.1f}%)")
    print(f"    小赚(0~1%): {((pnl>0)&(pnl<=1)).sum()}笔 ({100*((pnl>0)&(pnl<=1)).sum()/n:.1f}%)")
    print(f"    小亏(-1~0%): {((pnl>=-1)&(pnl<=0)).sum()}笔 ({100*((pnl>=-1)&(pnl<=0)).sum()/n:.1f}%)")
    print(f"    大亏(<-1%): {(pnl<-1).sum()}笔 ({100*(pnl<-1).sum()/n:.1f}%)")

    # --- 多空 ---
    print(f"\n  [多空拆分]")
    for d, dname in [('long', '做多'), ('short', '做空')]:
        seg = sub[sub['direction'] == d]
        s = seg['pnl'].dropna()
        if len(s) >= 3:
            e = ev(s)
            ev_str = f"EV={e:.2f}" if e else "N/A"
            print(f"    {dname}: N={len(s)}, avg={s.mean():.3f}%, cum={s.sum():.1f}%, WR={wr(s):.0f}%, {ev_str}")
        else:
            print(f"    {dname}: N={len(s)} (不足)")


def main():
    print("收集全部场景信号...")
    df = collect_all()
    print(f"共 {len(df)} 笔场景信号\n")

    check_scenario(df, 1, "甩尾型 A+ER>=0.5+偏离>=1.0ATR -> S6")
    check_scenario(df, 2, "蓄力型 C+偏离>=2.0ATR -> S6")
    check_scenario(df, 3, "回踩型 B+ER>=0.5+偏离0.1~0.3ATR -> S5.1")


if __name__ == '__main__':
    main()
