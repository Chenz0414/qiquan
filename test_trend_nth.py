# -*- coding: utf-8 -*-
"""
趋势内回调序号分析
信号是这轮趋势的第几次回调？不同序号的EV/胜率/盈亏比
32品种120天，A+B类信号，S2出场，5跳止损
"""

import os
import numpy as np
import pandas as pd
from signal_core import SYMBOL_CONFIGS

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")

MIN_PB_BARS = 4
LAST_DAYS = 120
BARS_PER_DAY = 57
STOP_TICKS = 5


def load_and_prepare(cache_key):
    path = os.path.join(CACHE_DIR, f"{cache_key}_10min_170d.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema120'] = df['close'].ewm(span=120, adjust=False).mean()
    net = (df['close'] - df['close'].shift(20)).abs()
    bar_sum = df['close'].diff().abs().rolling(20).sum()
    df['er_20'] = net / bar_sum.replace(0, np.nan)
    return df


def detect_signals_with_nth(df, start_idx):
    """检测A+B类信号，记录是这轮趋势的第几次回调"""
    signals = []
    n = len(df)
    trend_dir = 0
    below_start = -1
    pb_low = pb_high = None
    prev_close = prev_ema10 = None
    # 趋势内回调计数（包含不在start_idx范围内的回调）
    trend_pullback_count = 0

    for i in range(1, n):
        row = df.iloc[i]
        close, high, low = row['close'], row['high'], row['low']
        ema10, ema20, ema120 = row['ema10'], row['ema20'], row['ema120']

        if prev_close is None or pd.isna(ema120) or pd.isna(ema10):
            prev_close, prev_ema10 = close, ema10
            continue

        curr_trend = 1 if ema20 > ema120 else (-1 if ema20 < ema120 else 0)
        if curr_trend != trend_dir and curr_trend != 0:
            trend_dir = curr_trend
            below_start, pb_low, pb_high = -1, None, None
            trend_pullback_count = 0  # 新趋势重置

        if trend_dir == 0:
            prev_close, prev_ema10 = close, ema10
            continue

        er_20 = row.get('er_20', np.nan)

        if trend_dir == 1:
            # A类：影线碰EMA10弹回
            if low <= ema10 and close > ema10 and prev_close > prev_ema10:
                if below_start == -1:
                    trend_pullback_count += 1
                    if i >= start_idx:
                        signals.append({
                            'idx': i, 'type': 'A', 'direction': 'long',
                            'entry_price': close, 'pullback_extreme': low,
                            'nth': trend_pullback_count, 'er_20': er_20,
                        })

            # B类
            if below_start == -1:
                if close < ema10 and prev_close >= prev_ema10:
                    below_start, pb_low = i, low
            else:
                pb_low = min(pb_low, low)
                if close > ema10:
                    pb_bars = i - below_start
                    trend_pullback_count += 1
                    if pb_bars >= MIN_PB_BARS and i >= start_idx:
                        signals.append({
                            'idx': i, 'type': 'B', 'direction': 'long',
                            'entry_price': close, 'pullback_extreme': pb_low,
                            'nth': trend_pullback_count, 'er_20': er_20,
                        })
                    below_start, pb_low = -1, None

        elif trend_dir == -1:
            # A类
            if high >= ema10 and close < ema10 and prev_close < prev_ema10:
                if below_start == -1:
                    trend_pullback_count += 1
                    if i >= start_idx:
                        signals.append({
                            'idx': i, 'type': 'A', 'direction': 'short',
                            'entry_price': close, 'pullback_extreme': high,
                            'nth': trend_pullback_count, 'er_20': er_20,
                        })

            # B类
            if below_start == -1:
                if close > ema10 and prev_close <= prev_ema10:
                    below_start, pb_high = i, high
            else:
                pb_high = max(pb_high, high)
                if close < ema10:
                    pb_bars = i - below_start
                    trend_pullback_count += 1
                    if pb_bars >= MIN_PB_BARS and i >= start_idx:
                        signals.append({
                            'idx': i, 'type': 'B', 'direction': 'short',
                            'entry_price': close, 'pullback_extreme': pb_high,
                            'nth': trend_pullback_count, 'er_20': er_20,
                        })
                    below_start, pb_high = -1, None

        prev_close, prev_ema10 = close, ema10
    return signals


def simulate_s2(df, signals, tick_size):
    trades = []
    n = len(df)
    tick = tick_size * STOP_TICKS

    for sig in signals:
        entry_idx = sig['idx']
        entry_price = sig['entry_price']
        is_long = sig['direction'] == 'long'
        pb_ext = sig['pullback_extreme']

        stop = (pb_ext - tick) if is_long else (pb_ext + tick)
        state = 'normal'
        tracking_extreme = None
        bars_held = 0
        exit_price = None
        mfe = 0.0

        for j in range(entry_idx + 1, n):
            bar = df.iloc[j]
            if pd.isna(bar['ema10']):
                continue
            bars_held += 1
            ma_val = bar['ema10']

            if is_long:
                fav = (bar['high'] - entry_price) / entry_price * 100
            else:
                fav = (entry_price - bar['low']) / entry_price * 100
            mfe = max(mfe, fav)

            # 止损检查
            if is_long and bar['low'] <= stop:
                exit_price = stop
                break
            elif not is_long and bar['high'] >= stop:
                exit_price = stop
                break

            # S2状态机
            if is_long:
                if state == 'normal':
                    if bar['close'] < ma_val:
                        state = 'pullback'
                        tracking_extreme = bar['low']
                elif state == 'pullback':
                    tracking_extreme = min(tracking_extreme, bar['low'])
                    if bar['close'] > ma_val:
                        stop = max(stop, tracking_extreme - tick)
                        state = 'normal'
                        tracking_extreme = None
            else:
                if state == 'normal':
                    if bar['close'] > ma_val:
                        state = 'pullback'
                        tracking_extreme = bar['high']
                elif state == 'pullback':
                    tracking_extreme = max(tracking_extreme, bar['high'])
                    if bar['close'] < ma_val:
                        stop = min(stop, tracking_extreme + tick)
                        state = 'normal'
                        tracking_extreme = None

        if exit_price is None:
            exit_price = df.iloc[-1]['close']

        if is_long:
            pnl = (exit_price - entry_price) / entry_price * 100
        else:
            pnl = (entry_price - exit_price) / entry_price * 100

        trades.append({
            'type': sig['type'],
            'nth': sig['nth'],
            'er_20': sig['er_20'],
            'pnl': round(pnl, 4),
            'bars': bars_held,
            'mfe': round(mfe, 4),
        })

    return trades


def calc_ev(pnls):
    if len(pnls) == 0:
        return 0, 0, 0, 0
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0.001
    pr = avg_win / avg_loss if avg_loss > 0 else 0
    EV = wr * pr - (1 - wr)
    return round(EV, 2), round(wr * 100, 1), round(pr, 2), len(pnls)


# ==================== main ====================

print("=" * 90)
print("trend nth pullback analysis | 32 symbols | 120d | A+B | S2 exit")
print("=" * 90)

all_trades = []

for sym_key, cfg in sorted(SYMBOL_CONFIGS.items(), key=lambda x: x[1]['name']):
    cache_key = sym_key.replace(".", "_")
    df = load_and_prepare(cache_key)
    if df is None:
        continue

    tick_size = cfg['tick_size']
    n = len(df)
    start_idx = max(0, n - LAST_DAYS * BARS_PER_DAY)

    signals = detect_signals_with_nth(df, start_idx)
    if not signals:
        continue

    trades = simulate_s2(df, signals, tick_size)
    for t in trades:
        t['symbol'] = cfg['name']
    all_trades.extend(trades)

df_all = pd.DataFrame(all_trades)
print(f"\ntotal trades: {len(df_all)}")

# ==================== 1. nth vs EV (all) ====================
print(f"\n{'='*90}")
print("1. nth pullback in trend vs EV (all A+B)")
print(f"{'='*90}")

print(f"\n  {'nth':>5} | {'N':>6} {'pct':>6} | {'EV':>6} {'wr':>6} {'pr':>6} | {'avg_pnl':>8} {'avg_mfe':>8}")
print(f"  {'-'*75}")

for nth_val in [1, 2, 3, 4, 5, '6-10', '11+']:
    if nth_val == '6-10':
        sub = df_all[(df_all['nth'] >= 6) & (df_all['nth'] <= 10)]
    elif nth_val == '11+':
        sub = df_all[df_all['nth'] >= 11]
    else:
        sub = df_all[df_all['nth'] == nth_val]
    if len(sub) == 0:
        continue
    ev, wr, pr, N = calc_ev(sub['pnl'].tolist())
    pct = len(sub) / len(df_all) * 100
    print(f"  {str(nth_val):>5} | {N:>6} {pct:>5.1f}% | {ev:>+6.2f} {wr:>5.1f}% {pr:>5.2f} | "
          f"{sub['pnl'].mean():>+8.4f} {sub['mfe'].mean():>8.4f}")


# ==================== 2. nth vs EV, A and B separately ====================
for type_name in ['A', 'B']:
    df_type = df_all[df_all['type'] == type_name]
    print(f"\n{'='*90}")
    print(f"2{type_name}. nth pullback ({type_name} only, N={len(df_type)})")
    print(f"{'='*90}")

    print(f"\n  {'nth':>5} | {'N':>6} {'pct':>6} | {'EV':>6} {'wr':>6} {'pr':>6} | {'avg_pnl':>8} {'avg_mfe':>8}")
    print(f"  {'-'*75}")

    for nth_val in [1, 2, 3, 4, 5, '6-10', '11+']:
        if nth_val == '6-10':
            sub = df_type[(df_type['nth'] >= 6) & (df_type['nth'] <= 10)]
        elif nth_val == '11+':
            sub = df_type[df_type['nth'] >= 11]
        else:
            sub = df_type[df_type['nth'] == nth_val]
        if len(sub) == 0:
            continue
        ev, wr, pr, N = calc_ev(sub['pnl'].tolist())
        pct = len(sub) / len(df_type) * 100
        print(f"  {str(nth_val):>5} | {N:>6} {pct:>5.1f}% | {ev:>+6.2f} {wr:>5.1f}% {pr:>5.2f} | "
              f"{sub['pnl'].mean():>+8.4f} {sub['mfe'].mean():>8.4f}")


# ==================== 3. nth + ER(20) cross ====================
print(f"\n{'='*90}")
print("3. nth x ER(20) cross (all A+B)")
print(f"{'='*90}")

er_bins = [('all', 0, 999), ('<0.5', 0, 0.5), ('>=0.5', 0.5, 999)]

print(f"\n  {'nth':>5} | ", end='')
for bn, _, _ in er_bins:
    print(f"{'--- ' + bn + ' ---':>25} | ", end='')
print()

print(f"  {'':>5} | ", end='')
for _ in er_bins:
    print(f"{'N':>5} {'EV':>6} {'wr':>6} {'pr':>6} | ", end='')
print()
print(f"  {'-'*90}")

for nth_val in [1, 2, 3, 4, 5, '6-10', '11+']:
    if nth_val == '6-10':
        sub_nth = df_all[(df_all['nth'] >= 6) & (df_all['nth'] <= 10)]
    elif nth_val == '11+':
        sub_nth = df_all[df_all['nth'] >= 11]
    else:
        sub_nth = df_all[df_all['nth'] == nth_val]

    print(f"  {str(nth_val):>5} | ", end='')
    for bn, lo, hi in er_bins:
        if bn == 'all':
            sub = sub_nth
        else:
            sub = sub_nth[(sub_nth['er_20'] >= lo) & (sub_nth['er_20'] < hi)]
        ev, wr, pr, N = calc_ev(sub['pnl'].tolist())
        print(f"{N:>5} {ev:>+6.2f} {wr:>5.1f}% {pr:>5.2f} | ", end='')
    print()
