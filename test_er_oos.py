# -*- coding: utf-8 -*-
"""
ER(20) 样本外验证：前60天 vs 后60天
32品种，A+B类信号，S2出场，5跳止损
"""

import os
import numpy as np
import pandas as pd
from signal_core import SYMBOL_CONFIGS, ExitTracker

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")

MIN_PB_BARS = 4
BARS_PER_DAY = 57
STOP_TICKS = 5
TOTAL_DAYS = 120
SPLIT_DAY = 60  # 前60天训练，后60天验证


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


def detect_signals_ab(df, start_idx, end_idx):
    signals = []
    n = len(df)
    trend_dir = 0
    below_start = -1
    pb_low = pb_high = None
    prev_close = prev_ema10 = None

    for i in range(1, min(n, end_idx + 200)):  # 跑到end_idx之后留出场空间
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
        if pd.isna(er_20):
            er_20 = np.nan

        in_range = start_idx <= i <= end_idx

        if trend_dir == 1:
            if in_range and low <= ema10 and close > ema10 and prev_close > prev_ema10:
                if below_start == -1:
                    signals.append({
                        'idx': i, 'type': 'A', 'direction': 'long',
                        'entry_price': close, 'pullback_extreme': low,
                        'pullback_bars': 0, 'er_20': er_20,
                    })
            if below_start == -1:
                if close < ema10 and prev_close >= prev_ema10:
                    below_start, pb_low = i, low
            else:
                pb_low = min(pb_low, low)
                if close > ema10:
                    pb_bars = i - below_start
                    if pb_bars >= MIN_PB_BARS and in_range:
                        signals.append({
                            'idx': i, 'type': 'B', 'direction': 'long',
                            'entry_price': close, 'pullback_extreme': pb_low,
                            'pullback_bars': pb_bars, 'er_20': er_20,
                        })
                    below_start, pb_low = -1, None

        elif trend_dir == -1:
            if in_range and high >= ema10 and close < ema10 and prev_close < prev_ema10:
                if below_start == -1:
                    signals.append({
                        'idx': i, 'type': 'A', 'direction': 'short',
                        'entry_price': close, 'pullback_extreme': high,
                        'pullback_bars': 0, 'er_20': er_20,
                    })
            if below_start == -1:
                if close > ema10 and prev_close <= prev_ema10:
                    below_start, pb_high = i, high
            else:
                pb_high = max(pb_high, high)
                if close < ema10:
                    pb_bars = i - below_start
                    if pb_bars >= MIN_PB_BARS and in_range:
                        signals.append({
                            'idx': i, 'type': 'B', 'direction': 'short',
                            'entry_price': close, 'pullback_extreme': pb_high,
                            'pullback_bars': pb_bars, 'er_20': er_20,
                        })
                    below_start, pb_high = -1, None

        prev_close, prev_ema10 = close, ema10
    return signals


def simulate_s2(df, signals, tick_size):
    """只跑S2出场"""
    trades = []
    n = len(df)

    for sig in signals:
        entry_idx = sig['idx']
        entry_price = sig['entry_price']
        is_long = sig['direction'] == 'long'

        tracker = ExitTracker(
            direction=sig['direction'],
            entry_price=entry_price,
            pullback_extreme=sig['pullback_extreme'],
            tick_size=tick_size,
            stop_ticks=STOP_TICKS,
        )

        s2_result = None
        for j in range(entry_idx + 1, n):
            bar = df.iloc[j]
            prev_bar = df.iloc[j - 1]
            if pd.isna(bar['ema10']):
                continue

            exit_events, _ = tracker.process_bar(
                close=bar['close'], high=bar['high'], low=bar['low'],
                ema10=bar['ema10'], prev_close=prev_bar['close'],
                prev_high=prev_bar['high'], prev_low=prev_bar['low'],
            )
            for ev in exit_events:
                if ev.strategy == 'S2' and s2_result is None:
                    s2_result = {'pnl': ev.pnl_pct, 'bars': ev.bars_held}

            if s2_result is not None:
                break

        if s2_result is None:
            last_close = df.iloc[-1]['close']
            if is_long:
                pnl = (last_close - entry_price) / entry_price * 100
            else:
                pnl = (entry_price - last_close) / entry_price * 100
            s2_result = {'pnl': pnl, 'bars': n - entry_idx - 1}

        trades.append({
            'type': sig['type'],
            'er_20': sig['er_20'],
            'pnl': round(s2_result['pnl'], 4),
            'bars': s2_result['bars'],
        })

    return trades


def calc_ev(trades_list):
    if not trades_list:
        return {'N': 0, 'EV': 0, 'wr': 0, 'pr': 0}
    pnls = [t['pnl'] for t in trades_list]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0.001
    pr = avg_win / avg_loss if avg_loss > 0 else 0
    EV = wr * pr - (1 - wr)
    return {'N': len(pnls), 'EV': round(EV, 2), 'wr': round(wr * 100, 1), 'pr': round(pr, 2)}


# ==================== 主流程 ====================

print("=" * 90)
print("ER(20) 样本外验证 | 前60天(训练) vs 后60天(验证) | S2出场 | 32品种")
print("=" * 90)

train_trades = []
test_trades = []

for sym_key, cfg in sorted(SYMBOL_CONFIGS.items(), key=lambda x: x[1]['name']):
    cache_key = sym_key.replace(".", "_")
    df = load_and_prepare(cache_key)
    if df is None:
        continue

    tick_size = cfg['tick_size']
    n = len(df)

    total_bars = TOTAL_DAYS * BARS_PER_DAY
    data_start = max(0, n - total_bars)
    split_point = data_start + SPLIT_DAY * BARS_PER_DAY

    # 前60天信号
    sigs_train = detect_signals_ab(df, data_start, split_point - 1)
    trades_train = simulate_s2(df, sigs_train, tick_size)
    for t in trades_train:
        t['symbol'] = cfg['name']
    train_trades.extend(trades_train)

    # 后60天信号
    sigs_test = detect_signals_ab(df, split_point, n - 1)
    trades_test = simulate_s2(df, sigs_test, tick_size)
    for t in trades_test:
        t['symbol'] = cfg['name']
    test_trades.extend(trades_test)

    print(f"  {cfg['name']:>6}: train={len(trades_train):>4} test={len(trades_test):>4}")


# ==================== 输出 ====================

ER_BINS = [
    ('all',      0,    999),
    ('0~0.25',   0,    0.25),
    ('0.25~0.5', 0.25, 0.5),
    ('0.5+',     0.5,  999),
]

print(f"\n{'='*90}")
print(f"{'':>12} | {'--- TRAIN (前60天) ---':>30} | {'--- TEST (后60天) ---':>30}")
print(f"{'ER(20)':>12} | {'N':>5} {'EV':>6} {'wr':>6} {'pr':>6} | {'N':>5} {'EV':>6} {'wr':>6} {'pr':>6}")
print(f"{'-'*70}")

for bin_name, lo, hi in ER_BINS:
    if bin_name == 'all':
        tr = train_trades
        te = test_trades
    else:
        tr = [t for t in train_trades if not pd.isna(t['er_20']) and lo <= t['er_20'] < hi]
        te = [t for t in test_trades if not pd.isna(t['er_20']) and lo <= t['er_20'] < hi]

    s_tr = calc_ev(tr)
    s_te = calc_ev(te)
    print(f"  {bin_name:>10} | {s_tr['N']:>5} {s_tr['EV']:>+6.2f} {s_tr['wr']:>5.1f}% {s_tr['pr']:>5.2f} |"
          f" {s_te['N']:>5} {s_te['EV']:>+6.2f} {s_te['wr']:>5.1f}% {s_te['pr']:>5.2f}")

# A/B分别看
for type_name, type_filter in [('A only', 'A'), ('B only', 'B')]:
    print(f"\n  {type_name}:")
    print(f"{'ER(20)':>12} | {'N':>5} {'EV':>6} {'wr':>6} {'pr':>6} | {'N':>5} {'EV':>6} {'wr':>6} {'pr':>6}")
    print(f"{'-'*70}")
    for bin_name, lo, hi in ER_BINS:
        if bin_name == 'all':
            tr = [t for t in train_trades if t['type'] == type_filter]
            te = [t for t in test_trades if t['type'] == type_filter]
        else:
            tr = [t for t in train_trades if t['type'] == type_filter and not pd.isna(t['er_20']) and lo <= t['er_20'] < hi]
            te = [t for t in test_trades if t['type'] == type_filter and not pd.isna(t['er_20']) and lo <= t['er_20'] < hi]
        s_tr = calc_ev(tr)
        s_te = calc_ev(te)
        print(f"  {bin_name:>10} | {s_tr['N']:>5} {s_tr['EV']:>+6.2f} {s_tr['wr']:>5.1f}% {s_tr['pr']:>5.2f} |"
              f" {s_te['N']:>5} {s_te['EV']:>+6.2f} {s_te['wr']:>5.1f}% {s_te['pr']:>5.2f}")
