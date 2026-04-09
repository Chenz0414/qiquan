# -*- coding: utf-8 -*-
"""
S2出场深度分析
- 回调轮次统计
- 持仓时间分布与盈亏关系
- 回调判断线对比（EMA10 vs EMA20）
- 盈亏交易特征对比
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

        if trend_dir == 1:
            if i >= start_idx and low <= ema10 and close > ema10 and prev_close > prev_ema10:
                if below_start == -1:
                    signals.append({
                        'idx': i, 'type': 'A', 'direction': 'long',
                        'entry_price': close, 'pullback_extreme': low,
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
                        })
                    below_start, pb_low = -1, None

        elif trend_dir == -1:
            if i >= start_idx and high >= ema10 and close < ema10 and prev_close < prev_ema10:
                if below_start == -1:
                    signals.append({
                        'idx': i, 'type': 'A', 'direction': 'short',
                        'entry_price': close, 'pullback_extreme': high,
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
                        })
                    below_start, pb_high = -1, None

        prev_close, prev_ema10 = close, ema10
    return signals


def simulate_s2_detailed(df, signals, tick_size, ma_col='ema10'):
    """S2出场详细模拟，记录回调轮次等细节"""
    trades = []
    n = len(df)
    tick = tick_size * STOP_TICKS

    for sig in signals:
        entry_idx = sig['idx']
        entry_price = sig['entry_price']
        is_long = sig['direction'] == 'long'
        pb_ext = sig['pullback_extreme']

        # 初始止损
        stop = (pb_ext - tick) if is_long else (pb_ext + tick)

        state = 'normal'
        tracking_extreme = None
        pullback_rounds = 0  # 回调轮次
        bars_held = 0
        exit_price = None
        exit_reason = 'backtest_end'
        mfe = 0.0
        mae = 0.0
        max_unrealized = 0.0  # 最大浮盈
        stop_updates = []  # 止损更新历史

        for j in range(entry_idx + 1, n):
            bar = df.iloc[j]
            if pd.isna(bar[ma_col]):
                continue
            bars_held += 1
            ma_val = bar[ma_col]

            # MFE/MAE
            if is_long:
                fav = (bar['high'] - entry_price) / entry_price * 100
                adv = (entry_price - bar['low']) / entry_price * 100
                unrealized = (bar['close'] - entry_price) / entry_price * 100
            else:
                fav = (entry_price - bar['low']) / entry_price * 100
                adv = (bar['high'] - entry_price) / entry_price * 100
                unrealized = (entry_price - bar['close']) / entry_price * 100
            mfe = max(mfe, fav)
            mae = max(mae, adv)
            max_unrealized = max(max_unrealized, unrealized)

            # 止损检查
            if is_long and bar['low'] <= stop:
                exit_price = stop
                exit_reason = 'stop'
                break
            elif not is_long and bar['high'] >= stop:
                exit_price = stop
                exit_reason = 'stop'
                break

            # 状态机
            if is_long:
                if state == 'normal':
                    if bar['close'] < ma_val:
                        state = 'pullback'
                        tracking_extreme = bar['low']
                elif state == 'pullback':
                    tracking_extreme = min(tracking_extreme, bar['low'])
                    if bar['close'] > ma_val:
                        pullback_rounds += 1
                        candidate = tracking_extreme - tick
                        old_stop = stop
                        stop = max(stop, candidate)
                        if stop != old_stop:
                            stop_updates.append({'round': pullback_rounds, 'bar': bars_held, 'stop': stop})
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
                        pullback_rounds += 1
                        candidate = tracking_extreme + tick
                        old_stop = stop
                        stop = min(stop, candidate)
                        if stop != old_stop:
                            stop_updates.append({'round': pullback_rounds, 'bar': bars_held, 'stop': stop})
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
            'direction': sig['direction'],
            'pnl': round(pnl, 4),
            'bars': bars_held,
            'pullback_rounds': pullback_rounds,
            'mfe': round(mfe, 4),
            'mae': round(mae, 4),
            'max_unrealized': round(max_unrealized, 4),
            'exit_reason': exit_reason,
            'n_stop_updates': len(stop_updates),
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


# ==================== 主流程 ====================

print("=" * 90)
print("S2出场深度分析 | 32品种 | 120天 | A+B类")
print("=" * 90)

all_trades_ema10 = []
all_trades_ema20 = []

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

    # EMA10版本
    trades10 = simulate_s2_detailed(df, signals, tick_size, ma_col='ema10')
    for t in trades10:
        t['symbol'] = cfg['name']
    all_trades_ema10.extend(trades10)

    # EMA20版本
    trades20 = simulate_s2_detailed(df, signals, tick_size, ma_col='ema20')
    for t in trades20:
        t['symbol'] = cfg['name']
    all_trades_ema20.extend(trades20)

df10 = pd.DataFrame(all_trades_ema10)
df20 = pd.DataFrame(all_trades_ema20)


# ==================== 1. 回调轮次分析 ====================
print(f"\n{'='*90}")
print("1. S2回调轮次分析（EMA10版本）")
print(f"{'='*90}")

print(f"\n  {'rounds':>8} | {'N':>6} {'pct':>6} | {'EV':>6} {'wr':>6} {'pr':>6} | {'avg_bars':>8} {'avg_mfe':>8} {'avg_mae':>8}")
print(f"  {'-'*85}")

for rounds in [0, 1, 2, 3, '4+']:
    if rounds == '4+':
        sub = df10[df10['pullback_rounds'] >= 4]
    else:
        sub = df10[df10['pullback_rounds'] == rounds]
    if len(sub) == 0:
        continue
    ev, wr, pr, N = calc_ev(sub['pnl'].tolist())
    pct = len(sub) / len(df10) * 100
    print(f"  {str(rounds):>8} | {N:>6} {pct:>5.1f}% | {ev:>+6.2f} {wr:>5.1f}% {pr:>5.2f} | "
          f"{sub['bars'].mean():>8.1f} {sub['mfe'].mean():>8.4f} {sub['mae'].mean():>8.4f}")


# ==================== 2. 持仓时间分析 ====================
print(f"\n{'='*90}")
print("2. 持仓时间分布与盈亏（EMA10版本）")
print(f"{'='*90}")

time_bins = [(1, 5), (6, 10), (11, 20), (21, 30), (31, 50), (51, 100), (101, 999)]
print(f"\n  {'bars':>10} | {'N':>6} {'pct':>6} | {'EV':>6} {'wr':>6} {'pr':>6} | {'avg_pnl':>8} {'avg_mfe':>8}")
print(f"  {'-'*80}")

for lo, hi in time_bins:
    sub = df10[(df10['bars'] >= lo) & (df10['bars'] <= hi)]
    if len(sub) == 0:
        continue
    ev, wr, pr, N = calc_ev(sub['pnl'].tolist())
    pct = len(sub) / len(df10) * 100
    label = f"{lo}-{hi}" if hi < 999 else f"{lo}+"
    print(f"  {label:>10} | {N:>6} {pct:>5.1f}% | {ev:>+6.2f} {wr:>5.1f}% {pr:>5.2f} | "
          f"{sub['pnl'].mean():>+8.4f} {sub['mfe'].mean():>8.4f}")


# ==================== 3. EMA10 vs EMA20 ====================
print(f"\n{'='*90}")
print("3. 回调判断线对比：EMA10 vs EMA20")
print(f"{'='*90}")

ev10, wr10, pr10, n10 = calc_ev(df10['pnl'].tolist())
ev20, wr20, pr20, n20 = calc_ev(df20['pnl'].tolist())
print(f"\n  {'version':>10} | {'N':>6} | {'EV':>6} {'wr':>6} {'pr':>6} | {'avg_bars':>8} {'sum_pnl':>10}")
print(f"  {'-'*70}")
print(f"  {'EMA10':>10} | {n10:>6} | {ev10:>+6.2f} {wr10:>5.1f}% {pr10:>5.2f} | {df10['bars'].mean():>8.1f} {df10['pnl'].sum():>+10.2f}")
print(f"  {'EMA20':>10} | {n20:>6} | {ev20:>+6.2f} {wr20:>5.1f}% {pr20:>5.2f} | {df20['bars'].mean():>8.1f} {df20['pnl'].sum():>+10.2f}")


# ==================== 4. 盈亏交易特征对比 ====================
print(f"\n{'='*90}")
print("4. 盈亏交易特征对比（EMA10版本）")
print(f"{'='*90}")

wins = df10[df10['pnl'] > 0]
losses = df10[df10['pnl'] <= 0]

print(f"\n  {'':>12} | {'WIN':>20} | {'LOSS':>20}")
print(f"  {'-'*55}")
print(f"  {'N':>12} | {len(wins):>20} | {len(losses):>20}")
print(f"  {'avg_pnl':>12} | {wins['pnl'].mean():>+20.4f} | {losses['pnl'].mean():>+20.4f}")
print(f"  {'median_pnl':>12} | {wins['pnl'].median():>+20.4f} | {losses['pnl'].median():>+20.4f}")
print(f"  {'avg_bars':>12} | {wins['bars'].mean():>20.1f} | {losses['bars'].mean():>20.1f}")
print(f"  {'avg_mfe':>12} | {wins['mfe'].mean():>20.4f} | {losses['mfe'].mean():>20.4f}")
print(f"  {'avg_mae':>12} | {wins['mae'].mean():>20.4f} | {losses['mae'].mean():>20.4f}")
print(f"  {'avg_rounds':>12} | {wins['pullback_rounds'].mean():>20.1f} | {losses['pullback_rounds'].mean():>20.1f}")
print(f"  {'max_unreal':>12} | {wins['max_unrealized'].mean():>20.4f} | {losses['max_unrealized'].mean():>20.4f}")


# ==================== 5. 最大浮盈回吐分析 ====================
print(f"\n{'='*90}")
print("5. 浮盈回吐分析（赢的交易）")
print(f"{'='*90}")

wins_copy = wins.copy()
wins_copy['captured'] = wins_copy['pnl'] / wins_copy['mfe'] * 100  # 捕获率

print(f"\n  平均MFE: {wins['mfe'].mean():.4f}%")
print(f"  平均实际盈利: {wins['pnl'].mean():.4f}%")
print(f"  平均捕获率: {wins_copy['captured'].mean():.1f}%")
print(f"  中位捕获率: {wins_copy['captured'].median():.1f}%")

# 按捕获率分组
cap_bins = [(0, 25), (25, 50), (50, 75), (75, 100)]
print(f"\n  {'capture%':>10} | {'N':>6} | {'avg_pnl':>8} {'avg_mfe':>8} {'avg_bars':>8}")
print(f"  {'-'*55}")
for lo, hi in cap_bins:
    sub = wins_copy[(wins_copy['captured'] >= lo) & (wins_copy['captured'] < hi)]
    if len(sub) == 0:
        continue
    print(f"  {f'{lo}-{hi}%':>10} | {len(sub):>6} | {sub['pnl'].mean():>+8.4f} {sub['mfe'].mean():>8.4f} {sub['bars'].mean():>8.1f}")


# ==================== 6. 加个时间止损测试 ====================
print(f"\n{'='*90}")
print("6. 假设加时间止损（超过N根强制平仓）对EV的影响")
print(f"{'='*90}")

print(f"\n  {'max_bars':>10} | {'N_affected':>10} | {'EV':>6} {'wr':>6} {'pr':>6} | {'sum_pnl':>10}")
print(f"  {'-'*65}")

for max_bars in [10, 15, 20, 30, 50, 999]:
    # 模拟：超过max_bars的交易，用max_bars时刻的pnl（简化：用原始pnl打折）
    # 实际需要重跑，这里用近似：bars>max_bars的交易，标记为按比例缩减
    # 更准确的做法：直接看bars<=max_bars的交易是否自然更好
    if max_bars == 999:
        sub = df10
        label = 'no limit'
    else:
        # 自然出场在max_bars内的 + 超时的（后者需要实际数据，这里无法精确模拟）
        # 简化：只看自然出场在max_bars内的
        sub = df10[df10['bars'] <= max_bars]
        label = f'<={max_bars}'

    ev, wr, pr, N = calc_ev(sub['pnl'].tolist())
    n_cut = len(df10) - len(sub)
    print(f"  {label:>10} | {n_cut:>10} | {ev:>+6.2f} {wr:>5.1f}% {pr:>5.2f} | {sub['pnl'].sum():>+10.2f}")
