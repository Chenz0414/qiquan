# -*- coding: utf-8 -*-
"""
止损缓冲跳数参数测试：S1出场，分别跑1跳/3跳/5跳
高波动组10品种，最近30天数据
趋势判断：EMA20 > EMA120
信号：B类回调(close跌破EMA10后收回)，回调>=4根
"""

import os
import numpy as np
import pandas as pd
from signal_core import SYMBOL_CONFIGS

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")

# 高波动组
HIGH_VOL_SYMBOLS = [
    ("GFEX_lc", "GFEX.lc", "碳酸锂"),
    ("DCE_jm",  "DCE.jm",  "焦煤"),
    ("SHFE_ag", "SHFE.ag",  "白银"),
    ("CZCE_FG", "CZCE.FG",  "玻璃"),
    ("CZCE_SA", "CZCE.SA",  "纯碱"),
    ("INE_sc",  "INE.sc",   "原油"),
    ("CZCE_MA", "CZCE.MA",  "甲醇"),
    ("CZCE_TA", "CZCE.TA",  "PTA"),
    ("DCE_eb",  "DCE.eb",   "苯乙烯"),
    ("DCE_lh",  "DCE.lh",   "生猪"),
]

TICK_BUFFERS = [1, 3, 5]
MIN_PB_BARS = 4
LAST_DAYS = 30
BARS_PER_DAY = 57  # 10min K线，约9.5小时


def load_and_prepare(cache_key):
    """加载缓存数据，计算EMA10/20/120"""
    path = os.path.join(CACHE_DIR, f"{cache_key}_10min_170d.parquet")
    if not os.path.exists(path):
        print(f"  [跳过] {cache_key} 无缓存")
        return None

    df = pd.read_parquet(path)
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema120'] = df['close'].ewm(span=120, adjust=False).mean()
    return df


def detect_signals(df, start_idx):
    """
    B类信号检测，趋势用EMA20>EMA120，回调用EMA10
    只检测start_idx之后的信号
    """
    signals = []
    n = len(df)

    trend_dir = 0
    signal_count = 0
    below_start = -1
    pb_low = None
    pb_high = None
    prev_close = None
    prev_ema10 = None

    for i in range(1, n):
        row = df.iloc[i]
        close = row['close']
        high = row['high']
        low = row['low']
        ema10 = row['ema10']
        ema20 = row['ema20']
        ema120 = row['ema120']

        if prev_close is None or pd.isna(ema120):
            prev_close = close
            prev_ema10 = ema10
            continue

        # 趋势判断：EMA20 > EMA120
        if ema20 > ema120:
            curr_trend = 1
        elif ema20 < ema120:
            curr_trend = -1
        else:
            curr_trend = 0

        # 趋势翻转重置
        if curr_trend != trend_dir and curr_trend != 0:
            trend_dir = curr_trend
            signal_count = 0
            below_start = -1
            pb_low = None
            pb_high = None

        if trend_dir == 0:
            prev_close = close
            prev_ema10 = ema10
            continue

        # 多头信号
        if trend_dir == 1:
            if below_start == -1:
                if close < ema10 and prev_close >= prev_ema10:
                    below_start = i
                    pb_low = low
            else:
                pb_low = min(pb_low, low)
                if close > ema10:
                    pb_bars = i - below_start
                    if pb_bars >= MIN_PB_BARS and i >= start_idx:
                        signal_count += 1
                        signals.append({
                            'idx': i,
                            'direction': 'long',
                            'entry_price': close,
                            'pullback_extreme': pb_low,
                            'pullback_bars': pb_bars,
                            'signal_seq': signal_count,
                            'time': row['datetime'],
                        })
                    elif pb_bars >= MIN_PB_BARS:
                        signal_count += 1  # 计数但不记录（在start_idx之前）
                    below_start = -1
                    pb_low = None

        # 空头信号
        elif trend_dir == -1:
            if below_start == -1:
                if close > ema10 and prev_close <= prev_ema10:
                    below_start = i
                    pb_high = high
            else:
                pb_high = max(pb_high, high)
                if close < ema10:
                    pb_bars = i - below_start
                    if pb_bars >= MIN_PB_BARS and i >= start_idx:
                        signal_count += 1
                        signals.append({
                            'idx': i,
                            'direction': 'short',
                            'entry_price': close,
                            'pullback_extreme': pb_high,
                            'pullback_bars': pb_bars,
                            'signal_seq': signal_count,
                            'time': row['datetime'],
                        })
                    elif pb_bars >= MIN_PB_BARS:
                        signal_count += 1
                    below_start = -1
                    pb_high = None

        prev_close = close
        prev_ema10 = ema10

    return signals


def simulate_s1(df, signals, tick_size, stop_ticks):
    """S1出场模拟：当根K线新高追踪止损"""
    trades = []
    n = len(df)
    tick = tick_size * stop_ticks

    for sig in signals:
        entry_idx = sig['idx']
        entry_price = sig['entry_price']
        is_long = sig['direction'] == 'long'
        pb_extreme = sig['pullback_extreme']

        # 初始止损：回调极值 - N跳
        if is_long:
            stop = pb_extreme - tick
        else:
            stop = pb_extreme + tick

        exit_price = None
        exit_reason = 'backtest_end'
        bars_held = 0
        max_favorable = 0  # MFE
        max_adverse = 0    # MAE

        for j in range(entry_idx + 1, n):
            bar = df.iloc[j]
            prev_bar = df.iloc[j - 1]
            bars_held += 1

            # MFE/MAE追踪
            if is_long:
                fav = (bar['high'] - entry_price) / entry_price * 100
                adv = (entry_price - bar['low']) / entry_price * 100
            else:
                fav = (entry_price - bar['low']) / entry_price * 100
                adv = (bar['high'] - entry_price) / entry_price * 100
            max_favorable = max(max_favorable, fav)
            max_adverse = max(max_adverse, adv)

            # 检查止损触发
            if is_long and bar['low'] <= stop:
                exit_price = stop
                exit_reason = 'stop'
                break
            elif not is_long and bar['high'] >= stop:
                exit_price = stop
                exit_reason = 'stop'
                break

            # 追踪更新：当根新高时更新止损
            if is_long and bar['close'] > prev_bar['close']:
                candidate = bar['low'] - tick
                stop = max(stop, candidate)
            elif not is_long and bar['close'] < prev_bar['close']:
                candidate = bar['high'] + tick
                stop = min(stop, candidate)

        if exit_price is None:
            exit_price = df.iloc[-1]['close']

        if is_long:
            pnl = (exit_price - entry_price) / entry_price * 100
        else:
            pnl = (entry_price - exit_price) / entry_price * 100

        trades.append({
            'direction': sig['direction'],
            'entry_time': sig['time'],
            'entry_price': entry_price,
            'exit_price': round(exit_price, 4),
            'pnl': round(pnl, 4),
            'bars_held': bars_held,
            'exit_reason': exit_reason,
            'mfe': round(max_favorable, 4),
            'mae': round(max_adverse, 4),
            'pullback_bars': sig['pullback_bars'],
        })

    return trades


def calc_stats(trades):
    """计算统计指标"""
    if not trades:
        return {'N': 0, '胜率': 0, '期望': 0, '盈亏比': 0, '累计': 0,
                'MFE中位': 0, 'MAE中位': 0, '平均持仓': 0}

    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls) * 100 if pnls else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0.001
    profit_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

    mfes = [t['mfe'] for t in trades]
    maes = [t['mae'] for t in trades]
    bars = [t['bars_held'] for t in trades]

    return {
        'N': len(trades),
        '胜率': round(win_rate, 1),
        '期望': round(np.mean(pnls), 4),
        '盈亏比': round(profit_ratio, 2),
        '累计': round(sum(pnls), 2),
        'MFE中位': round(np.median(mfes), 4),
        'MAE中位': round(np.median(maes), 4),
        '平均持仓': round(np.mean(bars), 1),
    }


# ==================== 主流程 ====================

print("=" * 80)
print("S1止损缓冲跳数参数测试 | 高波动组 | 最近30天")
print("趋势：EMA20>EMA120 | 信号：B类回调>=4根 | 出场：S1当根新高追踪")
print("=" * 80)

all_results = []

for cache_key, symbol_key, name in HIGH_VOL_SYMBOLS:
    df = load_and_prepare(cache_key)
    if df is None:
        continue

    tick_size = SYMBOL_CONFIGS[symbol_key]['tick_size']
    n = len(df)
    start_idx = max(0, n - LAST_DAYS * BARS_PER_DAY)

    # 信号检测（在全量数据上跑状态机，只记录最近30天的信号）
    signals = detect_signals(df, start_idx)

    print(f"\n{'='*60}")
    print(f"{name}({symbol_key}) | tick={tick_size} | 信号数={len(signals)}")
    print(f"  数据范围: {df.iloc[start_idx]['datetime']} ~ {df.iloc[-1]['datetime']}")
    print(f"{'='*60}")

    if not signals:
        print("  无信号，跳过")
        continue

    for ticks in TICK_BUFFERS:
        trades = simulate_s1(df, signals, tick_size, ticks)
        stats = calc_stats(trades)
        print(f"  {ticks}跳: N={stats['N']:>3} | 胜率={stats['胜率']:>5.1f}% | "
              f"期望={stats['期望']:>+7.4f}% | 盈亏比={stats['盈亏比']:>5.2f} | "
              f"累计={stats['累计']:>+7.2f}% | MFE中位={stats['MFE中位']:.4f}% | "
              f"MAE中位={stats['MAE中位']:.4f}% | 持仓={stats['平均持仓']:.1f}根")

        all_results.append({
            '品种': name,
            '跳数': ticks,
            **stats,
        })

# ==================== 汇总 ====================
print("\n\n" + "=" * 80)
print("汇总：按跳数分组聚合（全品种）")
print("=" * 80)

df_results = pd.DataFrame(all_results)
if len(df_results) > 0:
    for ticks in TICK_BUFFERS:
        subset = df_results[df_results['跳数'] == ticks]
        if len(subset) == 0:
            continue
        total_n = subset['N'].sum()
        total_pnl = subset['累计'].sum()
        avg_wr = (subset['胜率'] * subset['N']).sum() / total_n if total_n > 0 else 0
        avg_exp = (subset['期望'] * subset['N']).sum() / total_n if total_n > 0 else 0
        avg_mfe = (subset['MFE中位'] * subset['N']).sum() / total_n if total_n > 0 else 0
        avg_mae = (subset['MAE中位'] * subset['N']).sum() / total_n if total_n > 0 else 0
        avg_bars = (subset['平均持仓'] * subset['N']).sum() / total_n if total_n > 0 else 0
        print(f"  {ticks}跳: 总N={total_n:>4} | 加权胜率={avg_wr:>5.1f}% | "
              f"加权期望={avg_exp:>+7.4f}% | 总累计={total_pnl:>+8.2f}% | "
              f"加权MFE中位={avg_mfe:.4f}% | 加权MAE中位={avg_mae:.4f}% | "
              f"加权持仓={avg_bars:.1f}根")

    # 逐品种对比表
    print("\n\n逐品种对比（期望值）：")
    print(f"{'品种':>6} | {'1跳':>10} | {'3跳':>10} | {'5跳':>10} | {'最优':>4}")
    print("-" * 55)
    symbols_done = df_results['品种'].unique()
    for sym in symbols_done:
        row_data = {}
        for ticks in TICK_BUFFERS:
            r = df_results[(df_results['品种'] == sym) & (df_results['跳数'] == ticks)]
            if len(r) > 0:
                row_data[ticks] = r.iloc[0]['期望']
            else:
                row_data[ticks] = float('nan')
        best = max(row_data, key=lambda k: row_data[k]) if row_data else '-'
        print(f"{sym:>6} | {row_data.get(1,0):>+10.4f} | {row_data.get(3,0):>+10.4f} | "
              f"{row_data.get(5,0):>+10.4f} | {best}跳")
