# -*- coding: utf-8 -*-
"""
止损缓冲跳数参数测试：常规品种组，S1出场，1跳 vs 5跳
最近30天数据，趋势EMA20>EMA120
"""

import os
import numpy as np
import pandas as pd
from signal_core import SYMBOL_CONFIGS

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")

# 常规品种组（22个，排除高波动组10个）
HIGH_VOL_KEYS = {"GFEX.lc","DCE.jm","SHFE.ag","CZCE.FG","CZCE.SA",
                 "INE.sc","CZCE.MA","CZCE.TA","DCE.eb","DCE.lh"}

NORMAL_SYMBOLS = []
for sym_key, cfg in SYMBOL_CONFIGS.items():
    if sym_key in HIGH_VOL_KEYS:
        continue
    cache_key = sym_key.replace(".", "_")
    path = os.path.join(CACHE_DIR, f"{cache_key}_10min_170d.parquet")
    if os.path.exists(path):
        NORMAL_SYMBOLS.append((cache_key, sym_key, cfg['name']))

NORMAL_SYMBOLS.sort(key=lambda x: x[2])

TICK_BUFFERS = [1, 5]
MIN_PB_BARS = 4
LAST_DAYS = 30
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

        trades.append({'pnl': round(pnl, 4), 'bars_held': bars_held, 'exit_reason': exit_reason})

    return trades


def calc_stats(trades):
    if not trades:
        return {'N': 0, 'wr': 0, 'ev': 0, 'total': 0, 'avg_bars': 0}
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    return {
        'N': len(trades),
        'wr': round(len(wins) / len(pnls) * 100, 1),
        'ev': round(np.mean(pnls), 4),
        'total': round(sum(pnls), 2),
        'avg_bars': round(np.mean([t['bars_held'] for t in trades]), 1),
    }


# ==================== 主流程 ====================

print("=" * 85)
print("常规品种组 S1止损跳数测试 | 1跳 vs 5跳 | 最近30天")
print("=" * 85)

all_results = []

for cache_key, symbol_key, name in NORMAL_SYMBOLS:
    df = load_and_prepare(cache_key)
    if df is None:
        continue

    tick_size = SYMBOL_CONFIGS[symbol_key]['tick_size']
    avg_price = df['close'].mean()
    tick_ratio = tick_size / avg_price * 100
    n = len(df)
    start_idx = max(0, n - LAST_DAYS * BARS_PER_DAY)

    signals = detect_signals(df, start_idx)
    if not signals:
        continue

    row_data = {'品种': name, '代码': symbol_key, 'tick/价格%': round(tick_ratio, 4)}

    for ticks in TICK_BUFFERS:
        trades = simulate_s1(df, signals, tick_size, ticks)
        stats = calc_stats(trades)
        row_data[f'{ticks}跳_N'] = stats['N']
        row_data[f'{ticks}跳_胜率'] = stats['wr']
        row_data[f'{ticks}跳_期望'] = stats['ev']
        row_data[f'{ticks}跳_累计'] = stats['total']
        row_data[f'{ticks}跳_持仓'] = stats['avg_bars']

    # 判断最优
    if row_data.get('1跳_累计', 0) >= row_data.get('5跳_累计', 0):
        row_data['最优'] = '1跳'
    else:
        row_data['最优'] = '5跳'

    # 阈值预测
    row_data['阈值预测'] = '1跳' if tick_ratio >= 0.042 else '5跳'
    row_data['命中'] = 'Y' if row_data['最优'] == row_data['阈值预测'] else 'N'

    all_results.append(row_data)

# 输出
print(f"\n{'品种':>6} | {'tick/价格%':>9} | {'1跳累计':>8} | {'5跳累计':>8} | {'最优':>4} | {'阈值预测':>6} | 命中")
print("-" * 75)
for r in sorted(all_results, key=lambda x: x['tick/价格%'], reverse=True):
    print(f"{r['品种']:>6} | {r['tick/价格%']:>9.4f} | {r['1跳_累计']:>+8.2f} | {r['5跳_累计']:>+8.2f} | {r['最优']:>4} | {r['阈值预测']:>6} | {r['命中']}")

# 命中率
hits = sum(1 for r in all_results if r['命中'] == 'Y')
total = len(all_results)
print(f"\n阈值0.042%命中率: {hits}/{total} ({hits/total*100:.1f}%)")

# 按阈值分组统计
above = [r for r in all_results if r['tick/价格%'] >= 0.042]
below = [r for r in all_results if r['tick/价格%'] < 0.042]
if above:
    a1 = sum(r['1跳_累计'] for r in above)
    a5 = sum(r['5跳_累计'] for r in above)
    print(f"\ntick/价格>=0.042% ({len(above)}品种): 1跳总累计={a1:+.2f}% | 5跳总累计={a5:+.2f}%")
if below:
    b1 = sum(r['1跳_累计'] for r in below)
    b5 = sum(r['5跳_累计'] for r in below)
    print(f"tick/价格<0.042%  ({len(below)}品种): 1跳总累计={b1:+.2f}% | 5跳总累计={b5:+.2f}%")
