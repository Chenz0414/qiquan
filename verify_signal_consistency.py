# -*- coding: utf-8 -*-
"""
交叉验证：signal_core.py (SignalDetector) vs test_er20_detail.py (detect_signals_abc)
确认两套引擎在 C 类信号上的输出完全一致。
"""

import os
import numpy as np
import pandas as pd
from signal_core import SignalDetector, SYMBOL_CONFIGS

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
LAST_DAYS = 120
BARS_PER_DAY = 57
MIN_PB_BARS_C = 4


def load_and_prepare(cache_key):
    path = os.path.join(CACHE_DIR, f"{cache_key}_10min_170d.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema120'] = df['close'].ewm(span=120, adjust=False).mean()
    return df


def detect_old_abc(df, start_idx):
    """test_er20_detail.py 原始逻辑（仅提取 C 类）"""
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
            if below_start == -1:
                if close < ema10 and prev_close >= prev_ema10:
                    below_start, pb_low = i, low
            else:
                pb_low = min(pb_low, low)
                if close > ema10:
                    pb_bars = i - below_start
                    if i >= start_idx and pb_bars >= MIN_PB_BARS_C:
                        signals.append({
                            'idx': i, 'type': 'C', 'direction': 'long',
                            'entry_price': close, 'pullback_extreme': pb_low,
                            'pb_bars': pb_bars,
                        })
                    below_start, pb_low = -1, None

        elif trend_dir == -1:
            if below_start == -1:
                if close > ema10 and prev_close <= prev_ema10:
                    below_start, pb_high = i, high
            else:
                pb_high = max(pb_high, high)
                if close < ema10:
                    pb_bars = i - below_start
                    if i >= start_idx and pb_bars >= MIN_PB_BARS_C:
                        signals.append({
                            'idx': i, 'type': 'C', 'direction': 'short',
                            'entry_price': close, 'pullback_extreme': pb_high,
                            'pb_bars': pb_bars,
                        })
                    below_start, pb_high = -1, None

        prev_close, prev_ema10 = close, ema10
    return signals


def detect_new_core(df, start_idx):
    """signal_core.py SignalDetector 仅取 C 类"""
    detector = SignalDetector(signal_types='C')
    signals = []
    n = len(df)

    for i in range(n):
        row = df.iloc[i]
        if pd.isna(row.get('ema10')) or pd.isna(row.get('ema120')):
            detector.process_bar(
                close=row['close'], high=row['high'], low=row['low'],
                ema10=row.get('ema10', row['close']),
                ema20=row.get('ema20', row['close']),
                ema120=row.get('ema120', row['close']),
            )
            continue

        result = detector.process_bar(
            close=row['close'], high=row['high'], low=row['low'],
            ema10=row['ema10'], ema20=row['ema20'], ema120=row['ema120'],
        )

        if result is not None and i >= start_idx:
            signals.append({
                'idx': result.bar_index,
                'type': result.signal_type,
                'direction': result.direction,
                'entry_price': result.entry_price,
                'pullback_extreme': result.pullback_extreme,
                'pb_bars': result.pullback_bars,
            })

    return signals


# ==================== 主流程 ====================

total_match = 0
total_mismatch = 0
total_old_only = 0
total_new_only = 0
problem_symbols = []

for sym_key, cfg in sorted(SYMBOL_CONFIGS.items(), key=lambda x: x[1]['name']):
    cache_key = sym_key.replace(".", "_")
    df = load_and_prepare(cache_key)
    if df is None:
        continue

    n = len(df)
    start_idx = max(0, n - LAST_DAYS * BARS_PER_DAY)

    old_signals = detect_old_abc(df, start_idx)
    new_signals = detect_new_core(df, start_idx)

    # 比较：以 (direction, entry_price, pullback_extreme, pb_bars) 为key
    def sig_key(s):
        return (s['direction'], round(s['entry_price'], 6),
                round(s['pullback_extreme'], 6), s['pb_bars'])

    old_set = {sig_key(s) for s in old_signals}
    new_set = {sig_key(s) for s in new_signals}

    matched = old_set & new_set
    old_only = old_set - new_set
    new_only = new_set - old_set

    total_match += len(matched)
    total_old_only += len(old_only)
    total_new_only += len(new_only)

    if old_only or new_only:
        total_mismatch += 1
        problem_symbols.append(cfg['name'])
        print(f"  ❌ {cfg['name']:6s} | old={len(old_signals)} new={len(new_signals)} "
              f"| match={len(matched)} old_only={len(old_only)} new_only={len(new_only)}")
        # 打印差异详情（最多5条）
        for s in list(old_only)[:3]:
            print(f"       old_only: dir={s[0]} price={s[1]} extreme={s[2]} bars={s[3]}")
        for s in list(new_only)[:3]:
            print(f"       new_only: dir={s[0]} price={s[1]} extreme={s[2]} bars={s[3]}")
    else:
        print(f"  ✅ {cfg['name']:6s} | {len(old_signals)} signals matched")

print()
print("=" * 60)
print(f"总计: {total_match} 信号匹配")
if total_old_only == 0 and total_new_only == 0:
    print("✅ 两套引擎的 C 类信号完全一致！")
else:
    print(f"❌ 差异: old_only={total_old_only} new_only={total_new_only}")
    print(f"   问题品种: {problem_symbols}")
