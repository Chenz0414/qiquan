# -*- coding: utf-8 -*-
"""
止损缓冲跳数参数测试：全部32品种，S1出场，1跳 vs 3跳 vs 5跳
120天数据，趋势EMA20>EMA120
"""

import os
import numpy as np
import pandas as pd
from signal_core import SYMBOL_CONFIGS

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")

TICK_BUFFERS = [1, 3, 5]
MIN_PB_BARS = 4
LAST_DAYS = 120
BARS_PER_DAY = 57


def load_and_prepare(cache_key):
    path = os.path.join(CACHE_DIR, f"{cache_key}_10min_170d.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema120'] = df['close'].ewm(span=120, adjust=False).mean()
    return df


def detect_signals(df, start_idx):
    signals = []
    n = len(df)
    trend_dir = 0
    signal_count = 0
    below_start = -1
    pb_low = pb_high = None
    prev_close = prev_ema10 = None

    for i in range(1, n):
        row = df.iloc[i]
        close, high, low = row['close'], row['high'], row['low']
        ema10, ema20, ema120 = row['ema10'], row['ema20'], row['ema120']

        if prev_close is None or pd.isna(ema120):
            prev_close, prev_ema10 = close, ema10
            continue

        curr_trend = 1 if ema20 > ema120 else (-1 if ema20 < ema120 else 0)
        if curr_trend != trend_dir and curr_trend != 0:
            trend_dir, signal_count = curr_trend, 0
            below_start, pb_low, pb_high = -1, None, None

        if trend_dir == 0:
            prev_close, prev_ema10 = close, ema10
            continue

        if trend_dir == 1:
            if below_start == -1:
                if close < ema10 and prev_close >= prev_ema10:
                    below_start, pb_low = i, low
            else:
                pb_low = min(pb_low, low)
                if close > ema10:
                    pb_bars = i - below_start
                    if pb_bars >= MIN_PB_BARS and i >= start_idx:
                        signal_count += 1
                        signals.append({'idx': i, 'direction': 'long', 'entry_price': close,
                                        'pullback_extreme': pb_low, 'pullback_bars': pb_bars,
                                        'time': row['datetime']})
                    elif pb_bars >= MIN_PB_BARS:
                        signal_count += 1
                    below_start, pb_low = -1, None
        elif trend_dir == -1:
            if below_start == -1:
                if close > ema10 and prev_close <= prev_ema10:
                    below_start, pb_high = i, high
            else:
                pb_high = max(pb_high, high)
                if close < ema10:
                    pb_bars = i - below_start
                    if pb_bars >= MIN_PB_BARS and i >= start_idx:
                        signal_count += 1
                        signals.append({'idx': i, 'direction': 'short', 'entry_price': close,
                                        'pullback_extreme': pb_high, 'pullback_bars': pb_bars,
                                        'time': row['datetime']})
                    elif pb_bars >= MIN_PB_BARS:
                        signal_count += 1
                    below_start, pb_high = -1, None

        prev_close, prev_ema10 = close, ema10
    return signals


def simulate_s1(df, signals, tick_size, stop_ticks):
    trades = []
    n = len(df)
    tick = tick_size * stop_ticks

    for sig in signals:
        entry_idx = sig['idx']
        entry_price = sig['entry_price']
        is_long = sig['direction'] == 'long'
        pb_extreme = sig['pullback_extreme']

        stop = pb_extreme - tick if is_long else pb_extreme + tick
        exit_price = None
        exit_reason = 'backtest_end'
        bars_held = 0

        for j in range(entry_idx + 1, n):
            bar = df.iloc[j]
            prev_bar = df.iloc[j - 1]
            bars_held += 1

            if is_long and bar['low'] <= stop:
                exit_price, exit_reason = stop, 'stop'
                break
            elif not is_long and bar['high'] >= stop:
                exit_price, exit_reason = stop, 'stop'
                break

            if is_long and bar['close'] > prev_bar['close']:
                stop = max(stop, bar['low'] - tick)
            elif not is_long and bar['close'] < prev_bar['close']:
                stop = min(stop, bar['high'] + tick)

        if exit_price is None:
            exit_price = df.iloc[-1]['close']

        pnl = (exit_price - entry_price) / entry_price * 100 if is_long else \
              (entry_price - exit_price) / entry_price * 100

        trades.append({'pnl': round(pnl, 4), 'bars_held': bars_held})

    return trades


# ==================== 主流程 ====================

print("=" * 90)
print("全品种 S1止损跳数测试 | 1跳 vs 3跳 vs 5跳 | 120天")
print("=" * 90)

HIGH_VOL_KEYS = {"GFEX.lc","DCE.jm","SHFE.ag","CZCE.FG","CZCE.SA",
                 "INE.sc","CZCE.MA","CZCE.TA","DCE.eb","DCE.lh"}

all_results = []

for sym_key, cfg in sorted(SYMBOL_CONFIGS.items(), key=lambda x: x[1]['name']):
    cache_key = sym_key.replace(".", "_")
    df = load_and_prepare(cache_key)
    if df is None:
        continue

    tick_size = cfg['tick_size']
    avg_price = df['close'].mean()
    tick_ratio = tick_size / avg_price * 100
    n = len(df)
    start_idx = max(0, n - LAST_DAYS * BARS_PER_DAY)
    group = 'H' if sym_key in HIGH_VOL_KEYS else 'N'

    signals = detect_signals(df, start_idx)
    if not signals:
        continue

    row = {'name': cfg['name'], 'sym': sym_key, 'tick_ratio': tick_ratio, 'group': group, 'N': len(signals)}

    for ticks in TICK_BUFFERS:
        trades = simulate_s1(df, signals, tick_size, ticks)
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        row[f't{ticks}_wr'] = round(len(wins)/len(pnls)*100, 1) if pnls else 0
        row[f't{ticks}_ev'] = round(np.mean(pnls), 4) if pnls else 0
        row[f't{ticks}_sum'] = round(sum(pnls), 2)
        row[f't{ticks}_bars'] = round(np.mean([t['bars_held'] for t in trades]), 1)

    # 最优
    best_tick = max(TICK_BUFFERS, key=lambda t: row[f't{t}_sum'])
    row['best'] = best_tick

    all_results.append(row)

# 输出逐品种
print(f"\n{'':>6} {'':>4} {'tick%':>7} | {'---1tick---':>22} | {'---3tick---':>22} | {'---5tick---':>22} | best")
print(f"{'name':>6} {'grp':>4} {'ratio':>7} | {'N':>4} {'wr%':>5} {'ev%':>8} {'sum%':>8} | {'wr%':>5} {'ev%':>8} {'sum%':>8} | {'wr%':>5} {'ev%':>8} {'sum%':>8} |")
print("-" * 115)

for r in sorted(all_results, key=lambda x: x['tick_ratio'], reverse=True):
    print(f"{r['name']:>6} {r['group']:>4} {r['tick_ratio']:>7.4f} | "
          f"{r['N']:>4} {r['t1_wr']:>5.1f} {r['t1_ev']:>+8.4f} {r['t1_sum']:>+8.2f} | "
          f"{r['t3_wr']:>5.1f} {r['t3_ev']:>+8.4f} {r['t3_sum']:>+8.2f} | "
          f"{r['t5_wr']:>5.1f} {r['t5_ev']:>+8.4f} {r['t5_sum']:>+8.2f} | {r['best']}")

# 汇总
print(f"\n{'='*90}")
print("汇总：")

for label, filt in [("全部", lambda r: True), ("高波动组", lambda r: r['group']=='H'), ("常规组", lambda r: r['group']=='N')]:
    subset = [r for r in all_results if filt(r)]
    if not subset:
        continue
    total_n = sum(r['N'] for r in subset)
    for ticks in TICK_BUFFERS:
        total_sum = sum(r[f't{ticks}_sum'] for r in subset)
        w_ev = sum(r[f't{ticks}_ev'] * r['N'] for r in subset) / total_n
        w_wr = sum(r[f't{ticks}_wr'] * r['N'] for r in subset) / total_n
        w_bars = sum(r[f't{ticks}_bars'] * r['N'] for r in subset) / total_n
        print(f"  {label} {ticks}tick: N={total_n:>5} | wr={w_wr:>5.1f}% | ev={w_ev:>+.4f}% | sum={total_sum:>+8.2f}% | bars={w_bars:.1f}")
    # 最优统计
    cnt = {t: sum(1 for r in subset if r['best']==t) for t in TICK_BUFFERS}
    print(f"  {label} 最优分布: 1tick={cnt[1]}  3tick={cnt[3]}  5tick={cnt[5]}")
    print()
