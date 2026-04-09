# -*- coding: utf-8 -*-
"""
ER(20)>=0.5 过滤后 S1/S2/S3 画像
对比过滤前后：回调轮次分布、0轮止损占比、盈亏特征、利润回吐
32品种120天，A+B类信号，5跳止损
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
            if i >= start_idx and low <= ema10 and close > ema10 and prev_close > prev_ema10:
                if below_start == -1:
                    signals.append({
                        'idx': i, 'type': 'A', 'direction': 'long',
                        'entry_price': close, 'pullback_extreme': low,
                        'er_20': er_20,
                    })
            if below_start == -1:
                if close < ema10 and prev_close >= prev_ema10:
                    below_start, pb_low = i, low
            else:
                pb_low = min(pb_low, low)
                if close > ema10:
                    pb_bars = i - below_start
                    if pb_bars >= MIN_PB_BARS and i >= start_idx:
                        signals.append({
                            'idx': i, 'type': 'B', 'direction': 'long',
                            'entry_price': close, 'pullback_extreme': pb_low,
                            'er_20': er_20,
                        })
                    below_start, pb_low = -1, None

        elif trend_dir == -1:
            if i >= start_idx and high >= ema10 and close < ema10 and prev_close < prev_ema10:
                if below_start == -1:
                    signals.append({
                        'idx': i, 'type': 'A', 'direction': 'short',
                        'entry_price': close, 'pullback_extreme': high,
                        'er_20': er_20,
                    })
            if below_start == -1:
                if close > ema10 and prev_close <= prev_ema10:
                    below_start, pb_high = i, high
            else:
                pb_high = max(pb_high, high)
                if close < ema10:
                    pb_bars = i - below_start
                    if pb_bars >= MIN_PB_BARS and i >= start_idx:
                        signals.append({
                            'idx': i, 'type': 'B', 'direction': 'short',
                            'entry_price': close, 'pullback_extreme': pb_high,
                            'er_20': er_20,
                        })
                    below_start, pb_high = -1, None

        prev_close, prev_ema10 = close, ema10
    return signals


def simulate_all_exits(df, signals, tick_size):
    """S1/S2/S3 并行模拟，记录详细画像数据"""
    trades = []
    n = len(df)
    tick = tick_size * STOP_TICKS

    for sig in signals:
        entry_idx = sig['idx']
        entry_price = sig['entry_price']
        is_long = sig['direction'] == 'long'
        pb_ext = sig['pullback_extreme']

        # 三种出场各自独立状态
        # S1: 当根新高追踪
        s1_stop = (pb_ext - tick) if is_long else (pb_ext + tick)
        s1_done = False
        s1_pnl = s1_bars = 0
        s1_mfe = s1_max_unreal = 0.0
        s1_exit_reason = 'end'

        # S2: 回调追踪(EMA10)
        s2_stop = (pb_ext - tick) if is_long else (pb_ext + tick)
        s2_state = 'normal'
        s2_tracking = None
        s2_rounds = 0
        s2_done = False
        s2_pnl = s2_bars = 0
        s2_mfe = s2_max_unreal = 0.0
        s2_exit_reason = 'end'

        # S3: 前根新高追踪
        s3_stop = (pb_ext - tick) if is_long else (pb_ext + tick)
        s3_done = False
        s3_pnl = s3_bars = 0
        s3_mfe = s3_max_unreal = 0.0
        s3_exit_reason = 'end'

        for j in range(entry_idx + 1, n):
            bar = df.iloc[j]
            prev_bar = df.iloc[j - 1]
            if pd.isna(bar['ema10']):
                continue

            close, high, low = bar['close'], bar['high'], bar['low']
            prev_close_val = prev_bar['close']
            prev_high_val = prev_bar['high']
            prev_low_val = prev_bar['low']
            ma_val = bar['ema10']

            if is_long:
                fav = (high - entry_price) / entry_price * 100
                unreal = (close - entry_price) / entry_price * 100
            else:
                fav = (entry_price - low) / entry_price * 100
                unreal = (entry_price - close) / entry_price * 100

            bars_from_entry = j - entry_idx

            # === S1 ===
            if not s1_done:
                s1_mfe = max(s1_mfe, fav)
                s1_max_unreal = max(s1_max_unreal, unreal)
                # 止损检查
                hit = (is_long and low <= s1_stop) or (not is_long and high >= s1_stop)
                if hit:
                    s1_done = True
                    ep = s1_stop
                    s1_pnl = (ep - entry_price) / entry_price * 100 if is_long else (entry_price - ep) / entry_price * 100
                    s1_bars = bars_from_entry
                    s1_exit_reason = 'stop'
                else:
                    # S1追踪：当根创新高/新低
                    if is_long and close > prev_close_val:
                        s1_stop = max(s1_stop, low - tick)
                    elif not is_long and close < prev_close_val:
                        s1_stop = min(s1_stop, high + tick)

            # === S2 ===
            if not s2_done:
                s2_mfe = max(s2_mfe, fav)
                s2_max_unreal = max(s2_max_unreal, unreal)
                hit = (is_long and low <= s2_stop) or (not is_long and high >= s2_stop)
                if hit:
                    s2_done = True
                    ep = s2_stop
                    s2_pnl = (ep - entry_price) / entry_price * 100 if is_long else (entry_price - ep) / entry_price * 100
                    s2_bars = bars_from_entry
                    s2_exit_reason = 'stop'
                else:
                    if is_long:
                        if s2_state == 'normal':
                            if close < ma_val:
                                s2_state = 'pullback'
                                s2_tracking = low
                        elif s2_state == 'pullback':
                            s2_tracking = min(s2_tracking, low)
                            if close > ma_val:
                                s2_rounds += 1
                                s2_stop = max(s2_stop, s2_tracking - tick)
                                s2_state = 'normal'
                                s2_tracking = None
                    else:
                        if s2_state == 'normal':
                            if close > ma_val:
                                s2_state = 'pullback'
                                s2_tracking = high
                        elif s2_state == 'pullback':
                            s2_tracking = max(s2_tracking, high)
                            if close < ma_val:
                                s2_rounds += 1
                                s2_stop = min(s2_stop, s2_tracking + tick)
                                s2_state = 'normal'
                                s2_tracking = None

            # === S3 ===
            if not s3_done:
                s3_mfe = max(s3_mfe, fav)
                s3_max_unreal = max(s3_max_unreal, unreal)
                hit = (is_long and low <= s3_stop) or (not is_long and high >= s3_stop)
                if hit:
                    s3_done = True
                    ep = s3_stop
                    s3_pnl = (ep - entry_price) / entry_price * 100 if is_long else (entry_price - ep) / entry_price * 100
                    s3_bars = bars_from_entry
                    s3_exit_reason = 'stop'
                else:
                    # S3追踪：前根创新高/新低
                    if is_long and close > prev_close_val:
                        s3_stop = max(s3_stop, prev_low_val - tick)
                    elif not is_long and close < prev_close_val:
                        s3_stop = min(s3_stop, prev_high_val + tick)

            if s1_done and s2_done and s3_done:
                break

        # 强制平仓
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

        # S2利润捕获率
        s2_capture = (s2_pnl / s2_mfe * 100) if s2_mfe > 0 and s2_pnl > 0 else 0

        trades.append({
            'type': sig['type'],
            'er_20': sig['er_20'],
            # S1
            's1_pnl': round(s1_pnl, 4), 's1_bars': s1_bars,
            's1_mfe': round(s1_mfe, 4), 's1_max_unreal': round(s1_max_unreal, 4),
            's1_exit': s1_exit_reason,
            # S2
            's2_pnl': round(s2_pnl, 4), 's2_bars': s2_bars,
            's2_mfe': round(s2_mfe, 4), 's2_max_unreal': round(s2_max_unreal, 4),
            's2_rounds': s2_rounds, 's2_exit': s2_exit_reason,
            's2_capture': round(s2_capture, 1),
            # S3
            's3_pnl': round(s3_pnl, 4), 's3_bars': s3_bars,
            's3_mfe': round(s3_mfe, 4), 's3_max_unreal': round(s3_max_unreal, 4),
            's3_exit': s3_exit_reason,
        })

    return trades


def calc_ev(pnls):
    if len(pnls) == 0:
        return {'N': 0, 'EV': 0, 'wr': 0, 'pr': 0, 'avg_pnl': 0, 'sum_pnl': 0}
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0.001
    pr = avg_win / avg_loss if avg_loss > 0 else 0
    EV = wr * pr - (1 - wr)
    return {
        'N': len(pnls), 'EV': round(EV, 2), 'wr': round(wr * 100, 1),
        'pr': round(pr, 2), 'avg_pnl': round(np.mean(pnls), 4),
        'sum_pnl': round(sum(pnls), 2),
    }


def print_exit_profile(df_sub, label):
    """打印一组交易的S1/S2/S3画像"""
    print(f"\n  {label} (N={len(df_sub)})")
    print(f"  {'-'*95}")
    print(f"  {'exit':>4} | {'N':>5} {'EV':>6} {'wr':>6} {'pr':>6} | {'avg_pnl':>8} {'sum_pnl':>9} | {'avg_bars':>8} {'avg_mfe':>8}")
    print(f"  {'-'*95}")

    for sx in ['s1', 's2', 's3']:
        pnls = df_sub[f'{sx}_pnl'].tolist()
        st = calc_ev(pnls)
        avg_bars = df_sub[f'{sx}_bars'].mean()
        avg_mfe = df_sub[f'{sx}_mfe'].mean()
        print(f"  {sx.upper():>4} | {st['N']:>5} {st['EV']:>+6.2f} {st['wr']:>5.1f}% {st['pr']:>5.2f} | "
              f"{st['avg_pnl']:>+8.4f} {st['sum_pnl']:>+9.2f} | {avg_bars:>8.1f} {avg_mfe:>8.4f}")


# ==================== main ====================

print("=" * 100)
print("ER(20) filtered profile | S1/S2/S3 | 32 symbols | 120d | A+B")
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

df_all = pd.DataFrame(all_trades)
df_er = df_all[df_all['er_20'] >= 0.5].copy()

print(f"\n  all trades: {len(df_all)},  ER(20)>=0.5: {len(df_er)}")


# ==================== 1. S1/S2/S3 overview ====================
print(f"\n{'='*100}")
print("1. S1/S2/S3 overview: no filter vs ER(20)>=0.5")
print(f"{'='*100}")

print_exit_profile(df_all, "NO FILTER")
print_exit_profile(df_er, "ER(20)>=0.5")


# ==================== 2. S2 pullback rounds ====================
print(f"\n{'='*100}")
print("2. S2 pullback rounds: no filter vs ER(20)>=0.5")
print(f"{'='*100}")

print(f"\n  {'rounds':>6} | {'--- NO FILTER ---':>35} | {'--- ER(20)>=0.5 ---':>35}")
print(f"  {'':>6} | {'N':>5} {'pct':>5} {'EV':>6} {'wr':>6} {'pr':>6} | {'N':>5} {'pct':>5} {'EV':>6} {'wr':>6} {'pr':>6}")
print(f"  {'-'*85}")

for rounds in [0, 1, 2, 3, '4+']:
    for label, df_sub in [('all', df_all), ('er', df_er)]:
        if rounds == '4+':
            sub = df_sub[df_sub['s2_rounds'] >= 4]
        else:
            sub = df_sub[df_sub['s2_rounds'] == rounds]

        if label == 'all':
            pnls_all = sub['s2_pnl'].tolist()
            st_all = calc_ev(pnls_all)
            pct_all = len(sub) / len(df_sub) * 100 if len(df_sub) > 0 else 0
        else:
            pnls_er = sub['s2_pnl'].tolist()
            st_er = calc_ev(pnls_er)
            pct_er = len(sub) / len(df_sub) * 100 if len(df_sub) > 0 else 0

    print(f"  {str(rounds):>6} | {st_all['N']:>5} {pct_all:>4.1f}% {st_all['EV']:>+6.2f} {st_all['wr']:>5.1f}% {st_all['pr']:>5.2f} | "
          f"{st_er['N']:>5} {pct_er:>4.1f}% {st_er['EV']:>+6.2f} {st_er['wr']:>5.1f}% {st_er['pr']:>5.2f}")


# ==================== 3. Win/Loss profile ====================
print(f"\n{'='*100}")
print("3. Win/Loss profile per exit (ER(20)>=0.5 only)")
print(f"{'='*100}")

for sx in ['s1', 's2', 's3']:
    pnl_col = f'{sx}_pnl'
    bars_col = f'{sx}_bars'
    mfe_col = f'{sx}_mfe'

    wins = df_er[df_er[pnl_col] > 0]
    losses = df_er[df_er[pnl_col] <= 0]

    print(f"\n  {sx.upper()}:")
    print(f"  {'':>15} | {'WIN':>12} | {'LOSS':>12}")
    print(f"  {'-'*45}")
    print(f"  {'N':>15} | {len(wins):>12} | {len(losses):>12}")
    print(f"  {'avg_pnl':>15} | {wins[pnl_col].mean():>+12.4f} | {losses[pnl_col].mean():>+12.4f}")
    print(f"  {'median_pnl':>15} | {wins[pnl_col].median():>+12.4f} | {losses[pnl_col].median():>+12.4f}")
    print(f"  {'avg_bars':>15} | {wins[bars_col].mean():>12.1f} | {losses[bars_col].mean():>12.1f}")
    print(f"  {'avg_mfe':>15} | {wins[mfe_col].mean():>12.4f} | {losses[mfe_col].mean():>12.4f}")


# ==================== 4. S2 profit capture (ER filtered) ====================
print(f"\n{'='*100}")
print("4. S2 profit capture (ER(20)>=0.5, wins only)")
print(f"{'='*100}")

s2_wins = df_er[df_er['s2_pnl'] > 0]
if len(s2_wins) > 0:
    print(f"\n  avg MFE: {s2_wins['s2_mfe'].mean():.4f}%")
    print(f"  avg realized: {s2_wins['s2_pnl'].mean():.4f}%")
    print(f"  avg capture: {s2_wins['s2_capture'].mean():.1f}%")
    print(f"  median capture: {s2_wins['s2_capture'].median():.1f}%")

    print(f"\n  {'capture%':>10} | {'N':>5} | {'avg_pnl':>8} {'avg_mfe':>8} {'avg_bars':>8}")
    print(f"  {'-'*55}")
    for lo, hi, label in [(0, 25, '0-25%'), (25, 50, '25-50%'), (50, 75, '50-75%'), (75, 101, '75-100%')]:
        sub = s2_wins[(s2_wins['s2_capture'] >= lo) & (s2_wins['s2_capture'] < hi)]
        if len(sub) == 0:
            continue
        print(f"  {label:>10} | {len(sub):>5} | {sub['s2_pnl'].mean():>+8.4f} {sub['s2_mfe'].mean():>8.4f} {sub['s2_bars'].mean():>8.1f}")


# ==================== 5. A vs B per exit (ER filtered) ====================
print(f"\n{'='*100}")
print("5. A vs B per exit (ER(20)>=0.5)")
print(f"{'='*100}")

for type_name in ['A', 'B']:
    df_type = df_er[df_er['type'] == type_name]
    print_exit_profile(df_type, f"{type_name} only")


# ==================== 6. S1 vs S2 vs S3 bars distribution (ER filtered) ====================
print(f"\n{'='*100}")
print("6. Hold time distribution per exit (ER(20)>=0.5)")
print(f"{'='*100}")

bar_bins = [(1, 5, '1-5'), (6, 10, '6-10'), (11, 20, '11-20'), (21, 50, '21-50'), (51, 999, '51+')]

for sx in ['s1', 's2', 's3']:
    bars_col = f'{sx}_bars'
    pnl_col = f'{sx}_pnl'
    print(f"\n  {sx.upper()}:")
    print(f"  {'bars':>8} | {'N':>5} {'pct':>5} | {'EV':>6} {'wr':>6} {'avg_pnl':>8}")
    print(f"  {'-'*55}")
    for lo, hi, label in bar_bins:
        sub = df_er[(df_er[bars_col] >= lo) & (df_er[bars_col] <= hi)]
        pnls = sub[pnl_col].tolist()
        st = calc_ev(pnls)
        pct = len(sub) / len(df_er) * 100 if len(df_er) > 0 else 0
        print(f"  {label:>8} | {st['N']:>5} {pct:>4.1f}% | {st['EV']:>+6.2f} {st['wr']:>5.1f}% {st['avg_pnl']:>+8.4f}")
