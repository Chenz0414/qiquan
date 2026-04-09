# -*- coding: utf-8 -*-
"""
ER(20) 门槛对比回测: 0.5 vs 0.4
用本地缓存170天数据，对比两个门槛下的信号数量、胜率、盈亏比。
"""

import os
import logging
import pandas as pd
import numpy as np
from collections import defaultdict

from signal_core import (
    SignalDetector, ExitTracker, Signal,
    SYMBOL_CONFIGS, DEFAULT_STOP_TICKS,
    SCENARIO_EXIT,
)
from data_loader import add_indicators, sym_name

logging.basicConfig(level=logging.INFO, format='%(message)s')

CACHE_DIR = "data_cache"
THRESHOLDS = [0.5, 0.45, 0.4]


def classify_with_threshold(sig_type, er20, deviation_atr, threshold):
    """classify_scenario 但 ER 门槛可调"""
    if er20 >= 0.7:
        return None
    if sig_type == 'A' and er20 >= threshold and deviation_atr >= 1.0:
        return 1
    if sig_type == 'C' and deviation_atr >= 2.0:
        return 2
    if sig_type == 'B' and er20 >= threshold and 0.1 <= deviation_atr < 0.3:
        return 3
    return None


def run_backtest(threshold):
    """跑一遍全品种回测，返回所有交易结果"""
    results = []

    for fname in sorted(os.listdir(CACHE_DIR)):
        if not fname.endswith("_10min_170d.parquet"):
            continue
        parts = fname.replace("_10min_170d.parquet", "").split("_", 1)
        sym_key = f"{parts[0]}.{parts[1]}"
        if sym_key not in SYMBOL_CONFIGS:
            continue
        cfg = SYMBOL_CONFIGS[sym_key]

        df = pd.read_parquet(os.path.join(CACHE_DIR, fname))
        if len(df) < 200:
            continue
        df = add_indicators(df, emas=(10, 20, 120),
                            er_periods=(5, 20, 40), atr_period=14)
        if 'er_5' in df.columns:
            df['er5_delta_6'] = df['er_5'] - df['er_5'].shift(6)
        if 'er_40' in df.columns:
            df['er40_delta_12'] = df['er_40'] - df['er_40'].shift(12)

        detector = SignalDetector()
        tracker = None
        tracker_meta = None

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]
            close, high, low = row['close'], row['high'], row['low']
            ema10, ema20, ema120 = row['ema10'], row['ema20'], row['ema120']

            # 出场追踪
            if tracker is not None:
                exit_events, stop_updates = tracker.process_bar(
                    close=close, high=high, low=low, ema10=ema10,
                    prev_close=prev_row['close'],
                    prev_high=prev_row['high'],
                    prev_low=prev_row['low'],
                )
                exit_strategy = tracker_meta['exit_strategy']
                tracker_meta['bars_held'] += 1

                for ev in exit_events:
                    if ev.strategy == exit_strategy:
                        results.append({
                            'sym': sym_key,
                            'name': sym_name(sym_key),
                            'scenario': tracker_meta['scenario'],
                            'er20': tracker_meta['er20'],
                            'pnl_pct': ev.pnl_pct,
                            'bars_held': ev.bars_held,
                            'direction': tracker_meta['direction'],
                        })
                        tracker = None
                        tracker_meta = None
                        break

            # 新信号
            signal = detector.process_bar(close, high, low, ema10, ema20, ema120)
            if signal is not None and tracker is None:
                er20 = float(row.get('er_20', 0) or 0)
                atr = float(row.get('atr', 0) or 0)
                if atr <= 0:
                    continue
                deviation_atr = abs(signal.entry_price - ema10) / atr
                scenario = classify_with_threshold(
                    signal.signal_type, er20, deviation_atr, threshold)
                if scenario is None:
                    continue

                er5_delta_6 = float(row.get('er5_delta_6', 0) or 0)
                if scenario == 1 and er5_delta_6 <= -0.41:
                    continue

                exit_strategy = SCENARIO_EXIT[scenario]
                tick = cfg['tick_size'] * DEFAULT_STOP_TICKS
                if signal.direction == 'long':
                    initial_stop = signal.pullback_extreme - tick
                else:
                    initial_stop = signal.pullback_extreme + tick

                tracker = ExitTracker(
                    direction=signal.direction,
                    entry_price=signal.entry_price,
                    pullback_extreme=signal.pullback_extreme,
                    tick_size=cfg['tick_size'],
                    stop_ticks=DEFAULT_STOP_TICKS,
                )
                tracker_meta = {
                    'scenario': scenario,
                    'exit_strategy': exit_strategy,
                    'entry_price': signal.entry_price,
                    'bars_held': 0,
                    'er20': er20,
                    'direction': signal.direction,
                }

    return results


print("=" * 70)
print("ER(20) 门槛对比回测 — 170天全品种")
print("=" * 70)

all_results = {}
for th in THRESHOLDS:
    all_results[th] = run_backtest(th)

# 输出汇总
print(f"\n{'门槛':>6} | {'信号数':>6} | {'已平仓':>6} | {'胜率':>6} | {'平均盈':>8} | {'平均亏':>8} | {'盈亏比':>6} | {'总PnL':>8}")
print("-" * 75)

for th in THRESHOLDS:
    trades = all_results[th]
    n_total = len(trades)
    if n_total == 0:
        print(f"  {th:.2f} | {'无交易':>6}")
        continue

    pnls = [t['pnl_pct'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / n_total * 100
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    total_pnl = sum(pnls)

    print(f"  {th:.2f} | {n_total:>6} | {n_total:>6} | {win_rate:>5.1f}% | {avg_win:>+7.2f}% | {avg_loss:>+7.2f}% | {profit_factor:>5.2f} | {total_pnl:>+7.1f}%")

# 按场景拆分
for sc in [1, 2, 3]:
    print(f"\n--- 场景{sc} ---")
    print(f"{'门槛':>6} | {'信号数':>6} | {'胜率':>6} | {'平均盈':>8} | {'平均亏':>8} | {'盈亏比':>6} | {'总PnL':>8}")
    print("-" * 70)
    for th in THRESHOLDS:
        trades = [t for t in all_results[th] if t['scenario'] == sc]
        if not trades:
            print(f"  {th:.2f} | {0:>6}")
            continue
        pnls = [t['pnl_pct'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(trades) * 100
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        pf = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        print(f"  {th:.2f} | {len(trades):>6} | {win_rate:>5.1f}% | {avg_win:>+7.2f}% | {avg_loss:>+7.2f}% | {pf:>5.2f} | {sum(pnls):>+7.1f}%")

# 新增信号明细（0.4有但0.5没有的）
print(f"\n{'='*70}")
print("0.4门槛新增交易明细（0.5门槛下不存在的）:")
print(f"{'='*70}")

# 找出0.4新增的
set_05 = set()
for t in all_results[0.5]:
    set_05.add((t['sym'], t['er20'], t['pnl_pct']))

new_trades = []
for t in all_results[0.4]:
    key = (t['sym'], t['er20'], t['pnl_pct'])
    if key not in set_05:
        new_trades.append(t)

if new_trades:
    wins_new = [t for t in new_trades if t['pnl_pct'] > 0]
    losses_new = [t for t in new_trades if t['pnl_pct'] <= 0]
    print(f"新增 {len(new_trades)} 笔: {len(wins_new)}盈 {len(losses_new)}亏")
    print()
    for t in sorted(new_trades, key=lambda x: x['pnl_pct'], reverse=True):
        sign = '+' if t['pnl_pct'] >= 0 else ''
        print(f"  {t['name']:>6} 场景{t['scenario']} ER={t['er20']:.2f} "
              f"{t['direction']:>5} {sign}{t['pnl_pct']:.2f}% "
              f"持{t['bars_held']}根")
else:
    print("  无新增交易")
