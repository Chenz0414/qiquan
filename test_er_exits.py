# -*- coding: utf-8 -*-
"""
ER(20) + 5种出场方式验证
========================
32品种全量数据，A+B类信号。
每个信号用5种出场方式模拟实际PnL，
按ER分段统计各出场的累计收益、胜率、平均盈亏。
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import timedelta
from signal_core import ExitTracker, SYMBOL_CONFIGS

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MA_FAST = 10
MA_SLOW = 20
ATR_PERIOD = 14
STOP_TICKS = 5

ER_RANGES = [
    ('全部',     0,   999),
    ('ER<0.2',   0,   0.2),
    ('0.2~0.3',  0.2, 0.3),
    ('0.3~0.4',  0.3, 0.4),
    ('0.4~0.5',  0.4, 0.5),
    ('0.5~0.6',  0.5, 0.6),
    ('0.6~0.7',  0.6, 0.7),
    ('ER>=0.7',  0.7, 999),
]

EXIT_NAMES = ['S1_newhigh', 'S2_pullback', 'newhigh_prev', 'ma10', 'ma20']


def get_all_10min_symbols():
    symbols = []
    for f in os.listdir(CACHE_DIR):
        if f.endswith('_10min_170d.parquet'):
            symbols.append(f.replace('_10min_170d.parquet', ''))
    return sorted(symbols)


def sym_key_to_config_key(sym_key):
    """SHFE_ag -> SHFE.ag"""
    parts = sym_key.split('_', 1)
    if len(parts) == 2:
        return f"{parts[0]}.{parts[1]}"
    return sym_key


def get_tick_size(sym_key):
    config_key = sym_key_to_config_key(sym_key)
    if config_key in SYMBOL_CONFIGS:
        return SYMBOL_CONFIGS[config_key]['tick_size']
    return 1.0  # fallback


def get_sym_name(sym_key):
    config_key = sym_key_to_config_key(sym_key)
    if config_key in SYMBOL_CONFIGS:
        return SYMBOL_CONFIGS[config_key]['name']
    return sym_key


def load_cached(symbol_key):
    path = os.path.join(CACHE_DIR, f"{symbol_key}_10min_170d.parquet")
    return pd.read_parquet(path) if os.path.exists(path) else None


def prepare_data(df):
    if 'ma_fast' not in df.columns:
        df['ma_fast'] = df['close'].rolling(MA_FAST).mean()
    if 'ma_slow' not in df.columns:
        df['ma_slow'] = df['close'].rolling(MA_SLOW).mean()
    df['trend'] = 0
    df.loc[df['ma_fast'] > df['ma_slow'], 'trend'] = 1
    df.loc[df['ma_fast'] < df['ma_slow'], 'trend'] = -1

    # ER(20)
    bar_moves = df['close'].diff().abs()
    net = (df['close'] - df['close'].shift(20)).abs()
    bar_sum = bar_moves.rolling(20).sum()
    df['er_20'] = net / bar_sum.replace(0, np.nan)

    # ER(20) 跳空修正版：cap单根变化为3倍滚动中位数
    median_move = bar_moves.rolling(20).median()
    capped_moves = bar_moves.clip(upper=3 * median_move)
    capped_sum = capped_moves.rolling(20).sum()
    df['er_20_adj'] = net / capped_sum.replace(0, np.nan)

    # ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(ATR_PERIOD).mean()

    return df


def detect_ab_signals(df):
    """A+B类信号检测，记录回调极值"""
    signals = []
    n = len(df)
    warmup = max(MA_SLOW + ATR_PERIOD + 5, 30)

    b_below_start = -1
    b_pullback_low = None
    b_pullback_high = None
    prev_trend = 0

    for i in range(warmup, n):
        row = df.iloc[i]
        if pd.isna(row['ma_fast']) or pd.isna(row['ma_slow']):
            continue
        close = row['close']
        high = row['high']
        low = row['low']
        ma_f = row['ma_fast']
        trend = row['trend']
        er = row['er_20']
        atr = row['atr']
        if pd.isna(er) or pd.isna(atr) or atr <= 0:
            continue

        if trend != prev_trend and trend != 0:
            b_below_start = -1
            b_pullback_low = None
            b_pullback_high = None
        prev_trend = trend
        if trend == 0:
            continue

        prev = df.iloc[i - 1]
        if pd.isna(prev['ma_fast']):
            continue
        prev_close = prev['close']
        prev_ma_f = prev['ma_fast']

        found = []  # (type, direction, pullback_extreme)

        if trend == 1:
            # A类：影线触及MA但收盘仍在上方
            if low <= ma_f and close > ma_f and prev_close > prev_ma_f:
                found.append(('A', 'long', low))
            # B类：实体跌破MA后收回
            if b_below_start == -1:
                if close < ma_f and prev_close >= prev_ma_f:
                    b_below_start = i
                    b_pullback_low = low
            else:
                b_pullback_low = min(b_pullback_low, low)
                if close > ma_f:
                    pb_bars = i - b_below_start
                    if pb_bars >= 4:  # 至少4根
                        found.append(('B', 'long', b_pullback_low))
                    b_below_start = -1
                    b_pullback_low = None

        elif trend == -1:
            if high >= ma_f and close < ma_f and prev_close < prev_ma_f:
                found.append(('A', 'short', high))
            if b_below_start == -1:
                if close > ma_f and prev_close <= prev_ma_f:
                    b_below_start = i
                    b_pullback_high = high
            else:
                b_pullback_high = max(b_pullback_high, high)
                if close < ma_f:
                    pb_bars = i - b_below_start
                    if pb_bars >= 4:
                        found.append(('B', 'short', b_pullback_high))
                    b_below_start = -1
                    b_pullback_high = None

        for sig_type, direction, pb_extreme in found:
            ma_dist_atr = abs(close - ma_f) / atr
            er_adj = row['er_20_adj'] if not pd.isna(row.get('er_20_adj', np.nan)) else er
            signals.append({
                'type': sig_type,
                'direction': direction,
                'idx': i,
                'time': row['datetime'],
                'entry_price': close,
                'pullback_extreme': pb_extreme,
                'er_20': er,
                'er_20_adj': er_adj,
                'atr': atr,
                'ma_dist_atr': round(ma_dist_atr, 3),
            })

    return signals


def simulate_5_exits(df, sig, tick_size):
    """
    模拟5种出场，返回各出场的PnL%。
    与backtest_engine.py simulate_exits()完全一致。
    """
    entry_idx = sig['idx']
    entry_price = sig['entry_price']
    direction = sig['direction']
    pb_extreme = sig['pullback_extreme']
    is_long = direction == 'long'
    n = len(df)
    tick = tick_size * STOP_TICKS

    # S1 & S2 用ExitTracker
    tracker = ExitTracker(
        direction=direction,
        entry_price=entry_price,
        pullback_extreme=pb_extreme,
        tick_size=tick_size,
        stop_ticks=STOP_TICKS,
    )

    s1_result = None
    s2_result = None

    # newhigh_prev, ma10, ma20
    exits_extra = {
        'newhigh_prev': {'done': False, 'stop': None, 'price': None, 'bars': 0},
        'ma10': {'done': False, 'price': None, 'bars': 0},
        'ma20': {'done': False, 'price': None, 'bars': 0},
    }

    if is_long:
        exits_extra['newhigh_prev']['stop'] = pb_extreme - tick
    else:
        exits_extra['newhigh_prev']['stop'] = pb_extreme + tick

    for j in range(entry_idx + 1, n):
        bar = df.iloc[j]
        prev_bar = df.iloc[j - 1]
        if pd.isna(bar['ma_fast']):
            continue

        # S1 & S2
        if not tracker.all_done():
            exit_events, _ = tracker.process_bar(
                close=bar['close'], high=bar['high'], low=bar['low'],
                ma_fast=bar['ma_fast'], prev_close=prev_bar['close'],
            )
            for ev in exit_events:
                if ev.strategy == 'S1' and s1_result is None:
                    s1_result = {'price': ev.exit_price, 'pnl': ev.pnl_pct, 'bars': ev.bars_held}
                elif ev.strategy == 'S2' and s2_result is None:
                    s2_result = {'price': ev.exit_price, 'pnl': ev.pnl_pct, 'bars': ev.bars_held}

        # newhigh_prev
        if not exits_extra['newhigh_prev']['done']:
            exits_extra['newhigh_prev']['bars'] += 1
            stop = exits_extra['newhigh_prev']['stop']
            if is_long and bar['low'] <= stop:
                exits_extra['newhigh_prev']['done'] = True
                exits_extra['newhigh_prev']['price'] = stop
            elif not is_long and bar['high'] >= stop:
                exits_extra['newhigh_prev']['done'] = True
                exits_extra['newhigh_prev']['price'] = stop
            else:
                if is_long and bar['close'] > prev_bar['close']:
                    candidate = prev_bar['low'] - tick
                    exits_extra['newhigh_prev']['stop'] = max(stop, candidate)
                elif not is_long and bar['close'] < prev_bar['close']:
                    candidate = prev_bar['high'] + tick
                    exits_extra['newhigh_prev']['stop'] = min(stop, candidate)

        # ma10
        if not exits_extra['ma10']['done']:
            exits_extra['ma10']['bars'] += 1
            if is_long and bar['close'] < bar['ma_fast']:
                exits_extra['ma10']['done'] = True
                exits_extra['ma10']['price'] = bar['close']
            elif not is_long and bar['close'] > bar['ma_fast']:
                exits_extra['ma10']['done'] = True
                exits_extra['ma10']['price'] = bar['close']

        # ma20
        if not exits_extra['ma20']['done']:
            exits_extra['ma20']['bars'] += 1
            if not pd.isna(bar['ma_slow']):
                if is_long and bar['close'] < bar['ma_slow']:
                    exits_extra['ma20']['done'] = True
                    exits_extra['ma20']['price'] = bar['close']
                elif not is_long and bar['close'] > bar['ma_slow']:
                    exits_extra['ma20']['done'] = True
                    exits_extra['ma20']['price'] = bar['close']

        all_extra_done = all(e['done'] for e in exits_extra.values())
        if tracker.all_done() and all_extra_done:
            break

    # 强制平仓
    last_close = df.iloc[-1]['close']
    if s1_result is None:
        forced = tracker.force_close(last_close)
        for ev in forced:
            if ev.strategy == 'S1':
                s1_result = {'price': ev.exit_price, 'pnl': ev.pnl_pct, 'bars': ev.bars_held}
            elif ev.strategy == 'S2' and s2_result is None:
                s2_result = {'price': ev.exit_price, 'pnl': ev.pnl_pct, 'bars': ev.bars_held}
    elif s2_result is None:
        forced = tracker.force_close(last_close)
        for ev in forced:
            if ev.strategy == 'S2':
                s2_result = {'price': ev.exit_price, 'pnl': ev.pnl_pct, 'bars': ev.bars_held}

    for key, ex in exits_extra.items():
        if not ex['done']:
            ex['price'] = last_close

    # 计算extra PnL
    for key in exits_extra:
        ex = exits_extra[key]
        if is_long:
            ex['pnl'] = (ex['price'] - entry_price) / entry_price * 100
        else:
            ex['pnl'] = (entry_price - ex['price']) / entry_price * 100

    return {
        'S1_newhigh': s1_result['pnl'],
        'S2_pullback': s2_result['pnl'],
        'newhigh_prev': exits_extra['newhigh_prev']['pnl'],
        'ma10': exits_extra['ma10']['pnl'],
        'ma20': exits_extra['ma20']['pnl'],
        'S1_bars': s1_result['bars'],
        'S2_bars': s2_result['bars'],
    }


def main():
    all_symbols = get_all_10min_symbols()
    print("=" * 120)
    print(f"  ER(20) + 5种出场验证 | {len(all_symbols)}品种 | A+B类")
    print("=" * 120)

    all_trades = []
    for sym_key in all_symbols:
        df = load_cached(sym_key)
        if df is None:
            continue
        df = prepare_data(df)
        cutoff = df['datetime'].iloc[-1] - timedelta(days=120)
        df_120 = df[df['datetime'] >= cutoff].copy().reset_index(drop=True)
        if len(df_120) < 200:
            continue

        tick_size = get_tick_size(sym_key)
        name = get_sym_name(sym_key)
        sigs = detect_ab_signals(df_120)
        print(f"  {name}({sym_key}): {len(sigs)}个信号, 模拟出场...", end="", flush=True)

        for s in sigs:
            pnls = simulate_5_exits(df_120, s, tick_size)
            s['symbol'] = sym_key
            s['sym_name'] = name
            s.update(pnls)

        all_trades.extend(sigs)
        print(f" done")

    if not all_trades:
        print("没有信号!")
        return

    a_all = [t for t in all_trades if t['type'] == 'A']
    b_all = [t for t in all_trades if t['type'] == 'B']
    print(f"\n  总信号: {len(all_trades)} | A类: {len(a_all)} | B类: {len(b_all)}")

    # ============================================================
    # 0. MA距离分桶分析（期望 = 胜率×盈亏比 - (1-胜率)）
    # ============================================================
    MA_DIST_BINS = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 999)]
    MA_DIST_LABELS = ['0~0.5', '0.5~1.0', '1.0~1.5', '1.5~2.0', '2.0+']
    exit_labels_cn = ['新高K线', '回调低点', '新高前根', '破10MA', '破20MA']

    for sig_label, sig_list in [('A+B类', all_trades), ('A类', a_all), ('B类', b_all)]:
        if len(sig_list) < 10:
            continue
        print(f"\n{'='*120}")
        print(f"  {sig_label} | MA距离(ATR) 分桶 x 5种出场 期望")
        print(f"{'='*120}")
        header = f"  {'MA距离':<10} {'N':>5} |"
        for el in exit_labels_cn:
            header += f" {el:>8}(期望) {'胜率':>6} {'盈亏比':>6} |"
        print(header)
        print(f"  {'-'*110}")

        for (lo, hi), dlabel in zip(MA_DIST_BINS, MA_DIST_LABELS):
            group = [t for t in sig_list if lo <= t['ma_dist_atr'] < hi]
            if len(group) < 10:
                continue
            n = len(group)
            line = f"  {dlabel:<10} {n:>5} |"
            for ename in EXIT_NAMES:
                pnls = np.array([t[ename] for t in group])
                wins = pnls[pnls > 0]
                losses = pnls[pnls <= 0]
                wr = len(wins) / n if n > 0 else 0
                avg_w = np.mean(wins) if len(wins) > 0 else 0
                avg_l = abs(np.mean(losses)) if len(losses) > 0 else 0.001
                ratio = avg_w / avg_l
                ev = wr * ratio - (1 - wr)
                line += f" {ev:>+8.2f} {wr*100:>5.1f}% {ratio:>5.2f}  |"
            print(line)

    # ============================================================
    # 0b. ER_adj vs ER 对比（B类，期望）
    # ============================================================
    print(f"\n{'='*120}")
    print(f"  ER原版 vs ER修正(跳空cap) | B类 | 期望")
    print(f"{'='*120}")
    for er_col, er_label in [('er_20', 'ER原版'), ('er_20_adj', 'ER修正')]:
        print(f"\n  --- {er_label} ---")
        er_ranges_here = [
            ('全部',     0,   999),
            ('<0.2',     0,   0.2),
            ('0.2~0.3',  0.2, 0.3),
            ('0.3~0.4',  0.3, 0.4),
            ('0.4~0.5',  0.4, 0.5),
            ('>=0.5',    0.5, 999),
        ]
        header = f"  {'区间':<10} {'N':>5} |"
        for el in exit_labels_cn:
            header += f" {el:>8}(期望) {'胜率':>6} |"
        print(header)
        print(f"  {'-'*100}")

        for label, lo, hi in er_ranges_here:
            group = [t for t in b_all if lo <= t.get(er_col, 0) < hi]
            if len(group) < 5:
                continue
            n = len(group)
            line = f"  {label:<10} {n:>5} |"
            for ename in EXIT_NAMES:
                pnls = np.array([t[ename] for t in group])
                wins = pnls[pnls > 0]
                losses = pnls[pnls <= 0]
                wr = len(wins) / n if n > 0 else 0
                avg_w = np.mean(wins) if len(wins) > 0 else 0
                avg_l = abs(np.mean(losses)) if len(losses) > 0 else 0.001
                ratio = avg_w / avg_l
                ev = wr * ratio - (1 - wr)
                line += f" {ev:>+8.2f} {wr*100:>5.1f}% |"
            print(line)

    # ============================================================
    # 1. 总表：各ER段 x 各出场方式（期望）
    # ============================================================
    for sig_label, sig_list in [('A+B类', all_trades), ('B类', b_all)]:
        if len(sig_list) < 10:
            continue
        print(f"\n{'='*120}")
        print(f"  {sig_label} | 各ER区间 x 5种出场 期望")
        print(f"{'='*120}")
        header = f"  {'ER区间':<10} {'N':>5} |"
        for el in exit_labels_cn:
            header += f" {el:>8}(期望) {'胜率':>6} {'盈亏比':>6} |"
        print(header)
        print(f"  {'-'*110}")

        for label, lo, hi in ER_RANGES:
            group = [t for t in sig_list if lo <= t['er_20'] < hi]
            if len(group) < 10:
                continue
            n = len(group)
            line = f"  {label:<10} {n:>5} |"
            for ename in EXIT_NAMES:
                pnls = np.array([t[ename] for t in group])
                wins = pnls[pnls > 0]
                losses = pnls[pnls <= 0]
                wr = len(wins) / n if n > 0 else 0
                avg_w = np.mean(wins) if len(wins) > 0 else 0
                avg_l = abs(np.mean(losses)) if len(losses) > 0 else 0.001
                ratio = avg_w / avg_l
                ev = wr * ratio - (1 - wr)
                line += f" {ev:>+8.2f} {wr*100:>5.1f}% {ratio:>5.2f}  |"
            print(line)

    # ============================================================
    # 2. 分品种: ER>=0.4 vs <0.4, S2_pullback PnL
    # ============================================================
    print(f"\n{'='*120}")
    print(f"  分品种: ER>=0.4 vs <0.4 | S2回调出场 & S1新高出场")
    print(f"{'='*120}")
    print(f"  {'品种':<10} {'N':>4} | {'ER>=0.4':>7} {'S1均':>8} {'S2均':>8} {'S1胜':>6} {'S2胜':>6} | "
          f"{'ER<0.4':>7} {'S1均':>8} {'S2均':>8} {'S1胜':>6} {'S2胜':>6} | {'S2差':>7}")
    print(f"  {'-'*110}")

    sym_diffs = []
    for sym_key in sorted(set(t['symbol'] for t in all_trades)):
        sym_trades = [t for t in all_trades if t['symbol'] == sym_key]
        above = [t for t in sym_trades if t['er_20'] >= 0.4]
        below = [t for t in sym_trades if t['er_20'] < 0.4]
        if len(above) < 3 or len(below) < 3:
            continue
        name = sym_trades[0]['sym_name']

        a_s1 = np.mean([t['S1_newhigh'] for t in above])
        a_s2 = np.mean([t['S2_pullback'] for t in above])
        a_s1w = sum(1 for t in above if t['S1_newhigh'] > 0) / len(above) * 100
        a_s2w = sum(1 for t in above if t['S2_pullback'] > 0) / len(above) * 100

        b_s1 = np.mean([t['S1_newhigh'] for t in below])
        b_s2 = np.mean([t['S2_pullback'] for t in below])
        b_s1w = sum(1 for t in below if t['S1_newhigh'] > 0) / len(below) * 100
        b_s2w = sum(1 for t in below if t['S2_pullback'] > 0) / len(below) * 100

        diff = a_s2 - b_s2
        sym_diffs.append({'sym': sym_key, 'name': name, 'diff': diff, 'n': len(sym_trades),
                          'a_s2': a_s2, 'b_s2': b_s2})

        print(f"  {name:<10} {len(sym_trades):>4} | "
              f"{len(above):>6} {a_s1:>+7.3f}% {a_s2:>+7.3f}% {a_s1w:>5.1f}% {a_s2w:>5.1f}% | "
              f"{len(below):>6} {b_s1:>+7.3f}% {b_s2:>+7.3f}% {b_s1w:>5.1f}% {b_s2w:>5.1f}% | "
              f"{diff:>+6.3f}%")

    pos = sum(1 for d in sym_diffs if d['diff'] > 0)
    print(f"\n  S2回调出场 ER>=0.4 优于 <0.4: {pos}/{len(sym_diffs)} 品种")

    # ============================================================
    # 3. 多阈值对比：ER>=0.3/0.4/0.5/0.6 各出场累计PnL
    # ============================================================
    print(f"\n{'='*120}")
    print(f"  ER阈值累计筛选: ER>=X 时各出场累计PnL")
    print(f"{'='*120}")
    thresholds = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    header = f"  {'阈值':<10} {'N':>5} |"
    for ename in EXIT_NAMES:
        header += f" {ename:>13}(累计) {'avg':>8} {'胜率':>6} |"
    print(header)
    print(f"  {'-'*120}")

    for th in thresholds:
        label = f"ER>={th:.1f}" if th > 0 else "全部"
        group = [t for t in all_trades if t['er_20'] >= th]
        if len(group) < 10:
            continue
        n = len(group)
        line = f"  {label:<10} {n:>5} |"
        for ename in EXIT_NAMES:
            pnls = [t[ename] for t in group]
            total = sum(pnls)
            avg = np.mean(pnls)
            win = sum(1 for p in pnls if p > 0) / n * 100
            line += f" {total:>+9.1f}% {avg:>+7.3f}% {win:>5.1f}% |"
        print(line)

    # ============================================================
    # 4. 图表
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    # 图1: 各ER段 平均PnL（5种出场）
    ax = axes[0, 0]
    er_labels = ['<0.2', '0.2~0.3', '0.3~0.4', '0.4~0.5', '0.5~0.6', '0.6~0.7', '>=0.7']
    er_bounds = [(0,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,999)]
    colors = ['#2196f3', '#4CAF50', '#ff9800', '#f44336', '#9C27B0']

    for ename, color in zip(EXIT_NAMES, colors):
        avgs = []
        valid_labels = []
        for label, (lo, hi) in zip(er_labels, er_bounds):
            group = [t for t in all_trades if lo <= t['er_20'] < hi]
            if len(group) >= 10:
                avgs.append(np.mean([t[ename] for t in group]))
                valid_labels.append(label)
        if avgs:
            ax.plot(valid_labels, avgs, 'o-', color=color, label=ename, linewidth=2)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_ylabel('平均PnL (%)')
    ax.set_title('A+B: 各ER段 x 5种出场 平均PnL', fontweight='bold')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(True, alpha=0.3)

    # 图2: 各ER段 胜率（5种出场）
    ax = axes[0, 1]
    for ename, color in zip(EXIT_NAMES, colors):
        wins = []
        valid_labels = []
        for label, (lo, hi) in zip(er_labels, er_bounds):
            group = [t for t in all_trades if lo <= t['er_20'] < hi]
            if len(group) >= 10:
                w = sum(1 for t in group if t[ename] > 0) / len(group) * 100
                wins.append(w)
                valid_labels.append(label)
        if wins:
            ax.plot(valid_labels, wins, 'o-', color=color, label=ename, linewidth=2)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_ylabel('胜率 (%)')
    ax.set_title('A+B: 各ER段 x 5种出场 胜率', fontweight='bold')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(True, alpha=0.3)

    # 图3: ER>=X 累计筛选 S1 & S2 累计PnL
    ax = axes[1, 0]
    for ename, color in zip(['S1_newhigh', 'S2_pullback'], ['#2196f3', '#4CAF50']):
        totals = []
        ns = []
        for th in thresholds:
            group = [t for t in all_trades if t['er_20'] >= th]
            if len(group) >= 10:
                totals.append(sum(t[ename] for t in group))
                ns.append(len(group))
        labels_th = [f">={th:.1f}" for th in thresholds[:len(totals)]]
        ax.plot(labels_th, totals, 'o-', color=color, label=ename, linewidth=2, markersize=8)
        # 标注信号数
        for x_i, (total_v, n_v) in enumerate(zip(totals, ns)):
            ax.annotate(f'n={n_v}', (x_i, total_v), textcoords="offset points",
                       xytext=(0, 10), fontsize=7, ha='center')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    ax.legend()
    ax.set_ylabel('累计PnL (%)')
    ax.set_title('ER>=X: S1/S2 累计PnL', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 图4: 分品种S2 ER>=0.4 vs <0.4 差值
    ax = axes[1, 1]
    if sym_diffs:
        sym_diffs_sorted = sorted(sym_diffs, key=lambda x: x['diff'], reverse=True)
        names = [d['name'] for d in sym_diffs_sorted]
        diffs = [d['diff'] for d in sym_diffs_sorted]
        bar_colors = ['#4CAF50' if d > 0 else '#f44336' for d in diffs]
        y_pos = range(len(names))
        ax.barh(y_pos, diffs, color=bar_colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=7)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('S2 PnL差 (ER>=0.4 minus <0.4)')
        ax.set_title(f'分品种: ER>=0.4 S2优势 ({pos}/{len(sym_diffs)}正向)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'er_exits.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图表: {out_path}")

    # 保存原始数据
    csv_path = os.path.join(OUTPUT_DIR, 'er_exits_trades.csv')
    df_out = pd.DataFrame(all_trades)
    cols_out = ['symbol', 'sym_name', 'type', 'direction', 'time', 'entry_price',
                'er_20', 'er_20_adj', 'ma_dist_atr'] + EXIT_NAMES + ['S1_bars', 'S2_bars']
    df_out[cols_out].to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"原始数据: {csv_path}")


if __name__ == '__main__':
    main()
