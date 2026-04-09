# -*- coding: utf-8 -*-
"""
回调强度分组 全品种120天
A类：pb_depth = |close - extreme| / close * 100, pb_bars=1, intensity=depth
B类：pb_depth = |close - extreme| / close * 100, intensity=depth/bars
S1/S2/S3，5跳止损
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


def detect_signals_ab(df, start_idx):
    signals = []
    n = len(df)
    trend_dir = 0
    below_start = -1
    pb_low = pb_high = None
    prev_close = prev_ema10 = None

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

        if trend_dir == 0:
            prev_close, prev_ema10 = close, ema10
            continue

        er_20 = row.get('er_20', np.nan)

        if trend_dir == 1:
            # A类
            if i >= start_idx and low <= ema10 and close > ema10 and prev_close > prev_ema10:
                if below_start == -1:
                    pb_depth = (close - low) / close * 100
                    signals.append({
                        'idx': i, 'type': 'A', 'direction': 'long',
                        'entry_price': close, 'pullback_extreme': low,
                        'pb_bars': 1, 'pb_depth': round(pb_depth, 4),
                        'pb_intensity': round(pb_depth, 4),  # A类bars=1
                        'er_20': er_20,
                    })

            # B类
            if below_start == -1:
                if close < ema10 and prev_close >= prev_ema10:
                    below_start, pb_low = i, low
            else:
                pb_low = min(pb_low, low)
                if close > ema10:
                    pb_bars = i - below_start
                    if pb_bars >= MIN_PB_BARS:
                        pb_depth = (close - pb_low) / close * 100
                        pb_intensity = pb_depth / pb_bars
                        if i >= start_idx:
                            signals.append({
                                'idx': i, 'type': 'B', 'direction': 'long',
                                'entry_price': close, 'pullback_extreme': pb_low,
                                'pb_bars': pb_bars, 'pb_depth': round(pb_depth, 4),
                                'pb_intensity': round(pb_intensity, 4),
                                'er_20': er_20,
                            })
                    below_start, pb_low = -1, None

        elif trend_dir == -1:
            # A类
            if i >= start_idx and high >= ema10 and close < ema10 and prev_close < prev_ema10:
                if below_start == -1:
                    pb_depth = (high - close) / close * 100
                    signals.append({
                        'idx': i, 'type': 'A', 'direction': 'short',
                        'entry_price': close, 'pullback_extreme': high,
                        'pb_bars': 1, 'pb_depth': round(pb_depth, 4),
                        'pb_intensity': round(pb_depth, 4),
                        'er_20': er_20,
                    })

            # B类
            if below_start == -1:
                if close > ema10 and prev_close <= prev_ema10:
                    below_start, pb_high = i, high
            else:
                pb_high = max(pb_high, high)
                if close < ema10:
                    pb_bars = i - below_start
                    if pb_bars >= MIN_PB_BARS:
                        pb_depth = (pb_high - close) / close * 100
                        pb_intensity = pb_depth / pb_bars
                        if i >= start_idx:
                            signals.append({
                                'idx': i, 'type': 'B', 'direction': 'short',
                                'entry_price': close, 'pullback_extreme': pb_high,
                                'pb_bars': pb_bars, 'pb_depth': round(pb_depth, 4),
                                'pb_intensity': round(pb_intensity, 4),
                                'er_20': er_20,
                            })
                    below_start, pb_high = -1, None

        prev_close, prev_ema10 = close, ema10
    return signals


def simulate_all_exits(df, signals, tick_size):
    trades = []
    n = len(df)
    tick = tick_size * STOP_TICKS

    for sig in signals:
        entry_idx = sig['idx']
        entry_price = sig['entry_price']
        is_long = sig['direction'] == 'long'
        pb_ext = sig['pullback_extreme']

        s1_stop = s2_stop = s3_stop = (pb_ext - tick) if is_long else (pb_ext + tick)
        s2_state = 'normal'
        s2_tracking = None
        s1_done = s2_done = s3_done = False
        s1_pnl = s2_pnl = s3_pnl = 0.0
        s1_bars = s2_bars = s3_bars = 0
        mfe = 0.0

        for j in range(entry_idx + 1, n):
            bar = df.iloc[j]
            prev_bar = df.iloc[j - 1]
            if pd.isna(bar['ema10']):
                continue

            close, high, low = bar['close'], bar['high'], bar['low']
            ma_val = bar['ema10']
            bars_j = j - entry_idx

            if is_long:
                fav = (high - entry_price) / entry_price * 100
            else:
                fav = (entry_price - low) / entry_price * 100
            mfe = max(mfe, fav)

            # S1
            if not s1_done:
                if (is_long and low <= s1_stop) or (not is_long and high >= s1_stop):
                    s1_done, s1_bars = True, bars_j
                    ep = s1_stop
                    s1_pnl = (ep - entry_price) / entry_price * 100 if is_long else (entry_price - ep) / entry_price * 100
                else:
                    if is_long and close > prev_bar['close']:
                        s1_stop = max(s1_stop, low - tick)
                    elif not is_long and close < prev_bar['close']:
                        s1_stop = min(s1_stop, high + tick)

            # S2
            if not s2_done:
                if (is_long and low <= s2_stop) or (not is_long and high >= s2_stop):
                    s2_done, s2_bars = True, bars_j
                    ep = s2_stop
                    s2_pnl = (ep - entry_price) / entry_price * 100 if is_long else (entry_price - ep) / entry_price * 100
                else:
                    if is_long:
                        if s2_state == 'normal' and close < ma_val:
                            s2_state, s2_tracking = 'pullback', low
                        elif s2_state == 'pullback':
                            s2_tracking = min(s2_tracking, low)
                            if close > ma_val:
                                s2_stop = max(s2_stop, s2_tracking - tick)
                                s2_state, s2_tracking = 'normal', None
                    else:
                        if s2_state == 'normal' and close > ma_val:
                            s2_state, s2_tracking = 'pullback', high
                        elif s2_state == 'pullback':
                            s2_tracking = max(s2_tracking, high)
                            if close < ma_val:
                                s2_stop = min(s2_stop, s2_tracking + tick)
                                s2_state, s2_tracking = 'normal', None

            # S3
            if not s3_done:
                if (is_long and low <= s3_stop) or (not is_long and high >= s3_stop):
                    s3_done, s3_bars = True, bars_j
                    ep = s3_stop
                    s3_pnl = (ep - entry_price) / entry_price * 100 if is_long else (entry_price - ep) / entry_price * 100
                else:
                    if is_long and close > prev_bar['close']:
                        s3_stop = max(s3_stop, prev_bar['low'] - tick)
                    elif not is_long and close < prev_bar['close']:
                        s3_stop = min(s3_stop, prev_bar['high'] + tick)

            if s1_done and s2_done and s3_done:
                break

        last_close = df.iloc[-1]['close']
        if not s1_done:
            s1_pnl = (last_close - entry_price) / entry_price * 100 if is_long else (entry_price - last_close) / entry_price * 100
            s1_bars = n - entry_idx - 1
        if not s2_done:
            s2_pnl = (last_close - entry_price) / entry_price * 100 if is_long else (entry_price - last_close) / entry_price * 100
            s2_bars = n - entry_idx - 1
        if not s3_done:
            s3_pnl = (last_close - entry_price) / entry_price * 100 if is_long else (entry_price - last_close) / entry_price * 100
            s3_bars = n - entry_idx - 1

        trades.append({
            'type': sig['type'],
            'pb_bars': sig['pb_bars'],
            'pb_depth': sig['pb_depth'],
            'pb_intensity': sig['pb_intensity'],
            'er_20': sig['er_20'],
            's1_pnl': round(s1_pnl, 4), 's1_bars': s1_bars,
            's2_pnl': round(s2_pnl, 4), 's2_bars': s2_bars,
            's3_pnl': round(s3_pnl, 4), 's3_bars': s3_bars,
            'mfe': round(mfe, 4),
        })

    return trades


def calc_ev(pnls):
    if len(pnls) == 0:
        return {'N': 0, 'EV': 0, 'wr': 0, 'pr': 0}
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0.001
    pr = avg_win / avg_loss if avg_loss > 0 else 0
    EV = wr * pr - (1 - wr)
    return {'N': len(pnls), 'EV': round(EV, 2), 'wr': round(wr * 100, 1), 'pr': round(pr, 2)}


def print_group_table(df_sub, title, split_col):
    """按中位数和三分位打印S1/S2/S3"""
    median_val = df_sub[split_col].median()
    q33 = df_sub[split_col].quantile(0.33)
    q66 = df_sub[split_col].quantile(0.66)

    print(f"\n{'='*100}")
    print(f"  {title}  (N={len(df_sub)}, median={median_val:.4f})")
    print(f"{'='*100}")

    # 中位数分组
    df_lo = df_sub[df_sub[split_col] <= median_val]
    df_hi = df_sub[df_sub[split_col] > median_val]

    print(f"\n  median split:")
    print(f"  {'group':>12} | {'exit':>4} | {'N':>5} {'EV':>6} {'wr':>6} {'pr':>6} | {'avg_pnl':>8} {'sum_pnl':>9} | {'avg_mfe':>8}")
    print(f"  {'-'*85}")

    for label, sub in [('mild(<=med)', df_lo), ('sharp(>med)', df_hi)]:
        for sx in ['s1', 's2', 's3']:
            pnls = sub[f'{sx}_pnl'].tolist()
            st = calc_ev(pnls)
            avg_pnl = np.mean(pnls) if pnls else 0
            sum_pnl = sum(pnls) if pnls else 0
            avg_mfe = sub['mfe'].mean() if len(sub) > 0 else 0
            print(f"  {label:>12} | {sx.upper():>4} | {st['N']:>5} {st['EV']:>+6.2f} {st['wr']:>5.1f}% {st['pr']:>5.2f} | "
                  f"{avg_pnl:>+8.4f} {sum_pnl:>+9.2f} | {avg_mfe:>8.4f}")
        print(f"  {'-'*85}")

    # 三分位
    g1 = df_sub[df_sub[split_col] <= q33]
    g2 = df_sub[(df_sub[split_col] > q33) & (df_sub[split_col] <= q66)]
    g3 = df_sub[df_sub[split_col] > q66]

    print(f"\n  tertile split (33%={q33:.4f}, 66%={q66:.4f}):")
    print(f"  {'group':>12} | {'exit':>4} | {'N':>5} {'EV':>6} {'wr':>6} {'pr':>6} | {'avg_pnl':>8} {'sum_pnl':>9} | {'avg_mfe':>8}")
    print(f"  {'-'*85}")

    for label, sub in [('gentle', g1), ('moderate', g2), ('sharp', g3)]:
        for sx in ['s1', 's2', 's3']:
            pnls = sub[f'{sx}_pnl'].tolist()
            st = calc_ev(pnls)
            avg_pnl = np.mean(pnls) if pnls else 0
            sum_pnl = sum(pnls) if pnls else 0
            avg_mfe = sub['mfe'].mean() if len(sub) > 0 else 0
            print(f"  {label:>12} | {sx.upper():>4} | {st['N']:>5} {st['EV']:>+6.2f} {st['wr']:>5.1f}% {st['pr']:>5.2f} | "
                  f"{avg_pnl:>+8.4f} {sum_pnl:>+9.2f} | {avg_mfe:>8.4f}")
        print(f"  {'-'*85}")


# ==================== main ====================

print("=" * 100)
print("pullback intensity | ALL 32 symbols | 120d | A+B separate | S1/S2/S3")
print("=" * 100)

all_trades = []

for sym_key, cfg in sorted(SYMBOL_CONFIGS.items(), key=lambda x: x[1]['name']):
    cache_key = sym_key.replace(".", "_")
    df = load_and_prepare(cache_key)
    if df is None:
        continue

    tick_size = cfg['tick_size']
    n = len(df)
    start_idx = max(0, n - LAST_DAYS * BARS_PER_DAY)

    signals = detect_signals_ab(df, start_idx)
    if not signals:
        continue

    trades = simulate_all_exits(df, signals, tick_size)
    for t in trades:
        t['symbol'] = cfg['name']
    all_trades.extend(trades)

    na = sum(1 for s in signals if s['type'] == 'A')
    nb = sum(1 for s in signals if s['type'] == 'B')
    print(f"  {cfg['name']:>6}: A={na:>4} B={nb:>4}")

df_all = pd.DataFrame(all_trades)
df_a = df_all[df_all['type'] == 'A'].copy()
df_b = df_all[df_all['type'] == 'B'].copy()

print(f"\n  TOTAL: A={len(df_a)} B={len(df_b)} all={len(df_all)}")

# A类：用 pb_depth 分（因为bars=1，intensity=depth）
print_group_table(df_a, "A class - by pb_depth (shadow depth %)", 'pb_depth')

# B类：用 pb_intensity 分（depth/bars）
print_group_table(df_b, "B class - by pb_intensity (depth%/bars)", 'pb_intensity')

# B类额外：单独看 pb_depth 和 pb_bars
print_group_table(df_b, "B class - by pb_depth (depth % only)", 'pb_depth')
print_group_table(df_b, "B class - by pb_bars (bar count only)", 'pb_bars')
