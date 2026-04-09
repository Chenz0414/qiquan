# -*- coding: utf-8 -*-
"""
用本地缓存的历史K线数据回填信号到 SQLite，让仪表盘有初始内容。
只取最近一周的数据回放。
运行: python seed_history.py
"""

import os
import logging
import pandas as pd
from datetime import datetime, timedelta

from signal_core import (
    SignalDetector, ExitTracker, Signal,
    SYMBOL_CONFIGS, DEFAULT_STOP_TICKS,
    classify_scenario, SCENARIO_EXIT,
)
from data_loader import add_indicators, sym_name
from signal_db import SignalDB

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "state/signals.db"
CACHE_DIR = "data_cache"


def cache_to_sym_key(filename):
    """SHFE_au_10min_170d.parquet -> SHFE.au"""
    parts = filename.replace("_10min_170d.parquet", "").split("_", 1)
    return f"{parts[0]}.{parts[1]}"


def run():
    # 清空旧数据
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    db = SignalDB(DB_PATH)

    total_signals = 0
    total_closed = 0

    for fname in sorted(os.listdir(CACHE_DIR)):
        if not fname.endswith("_10min_170d.parquet"):
            continue

        sym_key = cache_to_sym_key(fname)
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

        # 只用最后 ~200 根（约一周多）回放信号
        start_idx = max(130, len(df) - 200)

        detector = SignalDetector()
        # 先预热 detector 到 start_idx
        for i in range(1, start_idx):
            row = df.iloc[i]
            detector.process_bar(
                row['close'], row['high'], row['low'],
                row['ema10'], row['ema20'], row['ema120'])

        tracker = None
        tracker_meta = None

        for i in range(start_idx, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]
            close = row['close']
            high, low = row['high'], row['low']
            ema10, ema20, ema120 = row['ema10'], row['ema20'], row['ema120']

            bar_time = row['datetime'] if 'datetime' in df.columns else df.index[i]
            if hasattr(bar_time, 'strftime'):
                # 缓存数据是UTC，转北京时间 +8h
                bar_time_bj = bar_time + timedelta(hours=8)
                time_str = bar_time_bj.strftime('%Y-%m-%d %H:%M')
            else:
                time_str = str(bar_time)

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

                for su in stop_updates:
                    if su.strategy == exit_strategy or (
                        exit_strategy == 'S5.1' and
                        su.strategy in ('S2', 'S3.1')
                    ):
                        db.record_stop_update(
                            tracker_meta['signal_id'],
                            su.old_stop, su.new_stop,
                            su.strategy, time_str)

                for ev in exit_events:
                    if ev.strategy == exit_strategy:
                        db.record_exit(
                            tracker_meta['signal_id'],
                            ev.exit_price, ev.exit_reason,
                            ev.pnl_pct, ev.bars_held, time_str)
                        total_closed += 1
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
                scenario = classify_scenario(
                    signal.signal_type, er20, deviation_atr)
                if scenario is None:
                    continue

                position_multiplier = 1
                er40 = float(row.get('er_40', 0) or 0)
                er5_delta_6 = float(row.get('er5_delta_6', 0) or 0)
                er40_delta_12 = float(row.get('er40_delta_12', 0) or 0)

                if scenario == 1:
                    if er5_delta_6 <= -0.41:
                        continue
                    if er40 >= 0.42 or er5_delta_6 >= 0.50:
                        position_multiplier = 2
                elif scenario == 2:
                    if er40_delta_12 >= 0.14:
                        position_multiplier = 2

                exit_strategy = SCENARIO_EXIT[scenario]
                tick = cfg['tick_size'] * DEFAULT_STOP_TICKS
                if signal.direction == 'long':
                    initial_stop = signal.pullback_extreme - tick
                else:
                    initial_stop = signal.pullback_extreme + tick

                signal_id = db.record_entry(
                    sym_key=sym_key, sym_name=sym_name(sym_key),
                    direction=signal.direction,
                    signal_type=signal.signal_type,
                    scenario=scenario,
                    entry_price=signal.entry_price,
                    initial_stop=initial_stop,
                    pullback_bars=signal.pullback_bars,
                    deviation_atr=deviation_atr,
                    er20=er20,
                    position_multiplier=position_multiplier,
                    exit_strategy=exit_strategy,
                    entry_time=time_str,
                )

                tracker = ExitTracker(
                    direction=signal.direction,
                    entry_price=signal.entry_price,
                    pullback_extreme=signal.pullback_extreme,
                    tick_size=cfg['tick_size'],
                    stop_ticks=DEFAULT_STOP_TICKS,
                )
                tracker_meta = {
                    'signal_id': signal_id,
                    'exit_strategy': exit_strategy,
                    'entry_price': signal.entry_price,
                    'bars_held': 0,
                }

                total_signals += 1
                logger.info(
                    f"  {sym_name(sym_key)} {signal.direction} "
                    f"{signal.signal_type} 场景{scenario} "
                    f"{position_multiplier}x @ {time_str}")

    logger.info(f"\n完成！{total_signals} 个信号，{total_closed} 个已平仓")


if __name__ == '__main__':
    run()
