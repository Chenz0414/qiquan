# -*- coding: utf-8 -*-
"""
ER(20) 简单直接验证
==================
32品种全量数据，不分训练/验证。
ER>=0.3/0.4/0.5/0.6/0.7 各组的表现。
多个观察窗口（1h/2h/4h/8h）。
A类B类分开看。
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import timedelta

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MA_FAST = 10
MA_SLOW = 20
MFE_WINDOWS = {'1h': 6, '2h': 12, '4h': 24, '8h': 48}
ER_CUTS = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def get_all_10min_symbols():
    symbols = []
    for f in os.listdir(CACHE_DIR):
        if f.endswith('_10min_170d.parquet'):
            symbols.append(f.replace('_10min_170d.parquet', ''))
    return sorted(symbols)


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

    net = (df['close'] - df['close'].shift(20)).abs()
    bar_sum = df['close'].diff().abs().rolling(20).sum()
    df['er_20'] = net / bar_sum.replace(0, np.nan)
    return df


def detect_ab_signals(df):
    """A+B类信号检测，多窗口MFE/MAE"""
    signals = []
    n = len(df)
    warmup = 30
    max_window = max(MFE_WINDOWS.values())

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
        if pd.isna(er):
            continue

        if trend != prev_trend and trend != 0:
            b_below_start = -1
            b_pullback_low = None
            b_pullback_high = None
        prev_trend = trend
        if trend == 0 or i == warmup:
            continue

        prev = df.iloc[i - 1]
        if pd.isna(prev['ma_fast']):
            continue
        prev_close = prev['close']
        prev_ma_f = prev['ma_fast']

        found = []  # (type, direction)

        if trend == 1:
            if low <= ma_f and close > ma_f and prev_close > prev_ma_f:
                found.append(('A', 'long'))
            if b_below_start == -1:
                if close < ma_f and prev_close >= prev_ma_f:
                    b_below_start = i
                    b_pullback_low = low
            else:
                b_pullback_low = min(b_pullback_low, low)
                if close > ma_f:
                    found.append(('B', 'long'))
                    b_below_start = -1
                    b_pullback_low = None

        elif trend == -1:
            if high >= ma_f and close < ma_f and prev_close < prev_ma_f:
                found.append(('A', 'short'))
            if b_below_start == -1:
                if close > ma_f and prev_close <= prev_ma_f:
                    b_below_start = i
                    b_pullback_high = high
            else:
                b_pullback_high = max(b_pullback_high, high)
                if close < ma_f:
                    found.append(('B', 'short'))
                    b_below_start = -1
                    b_pullback_high = None

        for sig_type, direction in found:
            if i + 2 > len(df) - 1:
                continue

            sig = {
                'type': sig_type, 'direction': direction,
                'idx': i, 'time': row['datetime'],
                'entry_price': close, 'er_20': er,
            }

            # 多窗口MFE/MAE
            for wname, wsize in MFE_WINDOWS.items():
                end_idx = min(i + wsize, len(df) - 1)
                if i + 1 > end_idx:
                    sig[f'mfe_{wname}'] = 0
                    sig[f'mae_{wname}'] = 0
                    sig[f'ratio_{wname}'] = 1
                    continue
                segment = df.iloc[i + 1: end_idx + 1]
                if direction == 'short':
                    mfe = max((close - segment['low'].min()) / close * 100, 0)
                    mae = max((segment['high'].max() - close) / close * 100, 0)
                else:
                    mfe = max((segment['high'].max() - close) / close * 100, 0)
                    mae = max((close - segment['low'].min()) / close * 100, 0)
                sig[f'mfe_{wname}'] = mfe
                sig[f'mae_{wname}'] = mae
                sig[f'ratio_{wname}'] = mfe / mae if mae > 0 else 10

            signals.append(sig)
    return signals


def main():
    all_symbols = get_all_10min_symbols()
    print("=" * 110)
    print(f"  ER(20) 直接验证 | {len(all_symbols)}品种 | A+B类 | 多窗口")
    print("=" * 110)

    all_signals = []
    for sym_key in all_symbols:
        df = load_cached(sym_key)
        if df is None:
            continue
        df = prepare_data(df)
        cutoff = df['datetime'].iloc[-1] - timedelta(days=120)
        df_120 = df[df['datetime'] >= cutoff].copy().reset_index(drop=True)
        if len(df_120) < 200:
            continue
        sigs = detect_ab_signals(df_120)
        for s in sigs:
            s['symbol'] = sym_key
        all_signals.extend(sigs)

    a_all = [s for s in all_signals if s['type'] == 'A']
    b_all = [s for s in all_signals if s['type'] == 'B']
    print(f"  A类: {len(a_all)} | B类: {len(b_all)} | 总计: {len(all_signals)}")

    # ============================================================
    # 1. ER分段 x 窗口 x 信号类型：完整大表
    # ============================================================
    er_ranges = [
        ('全部', 0, 999),
        ('ER<0.2', 0, 0.2),
        ('ER 0.2~0.3', 0.2, 0.3),
        ('ER 0.3~0.4', 0.3, 0.4),
        ('ER 0.4~0.5', 0.4, 0.5),
        ('ER 0.5~0.6', 0.5, 0.6),
        ('ER 0.6~0.7', 0.6, 0.7),
        ('ER>=0.7', 0.7, 999),
    ]

    for sig_label, sig_list in [('A+B类', all_signals), ('A类', a_all), ('B类', b_all)]:
        print(f"\n{'='*110}")
        print(f"  {sig_label} | 各ER区间 x 各观察窗口")
        print(f"{'='*110}")

        for wname in ['1h', '2h', '4h', '8h']:
            print(f"\n  观察窗口: {wname}")
            print(f"  {'ER区间':<14} {'N':>5} | {'MFE':>7} {'MAE':>7} {'MFE/MAE':>8} | {'胜率':>6} {'MFE>2%':>7} {'MFE>3%':>7}")
            print(f"  {'-'*75}")

            for label, lo, hi in er_ranges:
                group = [s for s in sig_list if lo <= s['er_20'] < hi]
                if len(group) < 10:
                    continue
                n = len(group)
                mfe = np.mean([s[f'mfe_{wname}'] for s in group])
                mae = np.mean([s[f'mae_{wname}'] for s in group])
                ratio = np.median([s[f'ratio_{wname}'] for s in group])
                win = sum(1 for s in group if s[f'ratio_{wname}'] > 1) / n * 100
                mfe2 = sum(1 for s in group if s[f'mfe_{wname}'] >= 2) / n * 100
                mfe3 = sum(1 for s in group if s[f'mfe_{wname}'] >= 3) / n * 100
                print(f"  {label:<14} {n:>5} | {mfe:>6.2f}% {mae:>6.2f}% {ratio:>7.2f}  | "
                      f"{win:>5.1f}% {mfe2:>6.1f}% {mfe3:>6.1f}%")

    # ============================================================
    # 2. 分品种：每个品种内ER>=0.3 vs <0.3（2h窗口）
    # ============================================================
    print(f"\n{'='*110}")
    print(f"  分品种: ER>=0.3 vs <0.3 (A+B, 2h窗口)")
    print(f"{'='*110}")

    sym_results = []
    print(f"  {'品种':<12} {'N':>4} | {'ER>=0.3 N':>8} {'MFE/MAE':>8} {'胜率':>6} | {'ER<0.3 N':>8} {'MFE/MAE':>8} {'胜率':>6} | {'差':>6}")
    print(f"  {'-'*90}")

    for sym_key in sorted(set(s['symbol'] for s in all_signals)):
        sym_sigs = [s for s in all_signals if s['symbol'] == sym_key]
        above = [s for s in sym_sigs if s['er_20'] >= 0.3]
        below = [s for s in sym_sigs if s['er_20'] < 0.3]
        if len(above) < 5 or len(below) < 5:
            continue

        a_ratio = np.median([s['ratio_2h'] for s in above])
        b_ratio = np.median([s['ratio_2h'] for s in below])
        a_win = sum(1 for s in above if s['ratio_2h'] > 1) / len(above) * 100
        b_win = sum(1 for s in below if s['ratio_2h'] > 1) / len(below) * 100
        diff = a_ratio - b_ratio

        sym_results.append({'sym': sym_key, 'diff': diff, 'n': len(sym_sigs)})

        print(f"  {sym_key:<12} {len(sym_sigs):>4} | "
              f"{len(above):>7} {a_ratio:>7.2f} {a_win:>5.1f}% | "
              f"{len(below):>7} {b_ratio:>7.2f} {b_win:>5.1f}% | "
              f"{diff:>+5.2f}")

    pos = sum(1 for r in sym_results if r['diff'] > 0)
    print(f"\n  正向: {pos}/{len(sym_results)}")

    # ============================================================
    # 3. 同样的表用4h窗口再来一遍
    # ============================================================
    print(f"\n{'='*110}")
    print(f"  分品种: ER>=0.3 vs <0.3 (A+B, 4h窗口)")
    print(f"{'='*110}")

    sym_results_4h = []
    print(f"  {'品种':<12} {'N':>4} | {'ER>=0.3 N':>8} {'MFE/MAE':>8} {'胜率':>6} | {'ER<0.3 N':>8} {'MFE/MAE':>8} {'胜率':>6} | {'差':>6}")
    print(f"  {'-'*90}")

    for sym_key in sorted(set(s['symbol'] for s in all_signals)):
        sym_sigs = [s for s in all_signals if s['symbol'] == sym_key]
        above = [s for s in sym_sigs if s['er_20'] >= 0.3]
        below = [s for s in sym_sigs if s['er_20'] < 0.3]
        if len(above) < 5 or len(below) < 5:
            continue

        a_ratio = np.median([s['ratio_4h'] for s in above])
        b_ratio = np.median([s['ratio_4h'] for s in below])
        a_win = sum(1 for s in above if s['ratio_4h'] > 1) / len(above) * 100
        b_win = sum(1 for s in below if s['ratio_4h'] > 1) / len(below) * 100
        diff = a_ratio - b_ratio

        sym_results_4h.append({'sym': sym_key, 'diff': diff})

        print(f"  {sym_key:<12} {len(sym_sigs):>4} | "
              f"{len(above):>7} {a_ratio:>7.2f} {a_win:>5.1f}% | "
              f"{len(below):>7} {b_ratio:>7.2f} {b_win:>5.1f}% | "
              f"{diff:>+5.2f}")

    pos_4h = sum(1 for r in sym_results_4h if r['diff'] > 0)
    print(f"\n  正向: {pos_4h}/{len(sym_results_4h)}")

    # ============================================================
    # 图表
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # 图1: A+B各ER段 x 各窗口 MFE/MAE中位数
    ax = axes[0, 0]
    er_labels = ['<0.2', '0.2~0.3', '0.3~0.4', '0.4~0.5', '0.5~0.6', '0.6~0.7', '>=0.7']
    er_bounds = [(0,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,999)]
    colors_w = {'1h': '#2196f3', '2h': '#4CAF50', '4h': '#ff9800', '8h': '#f44336'}

    for wname, color in colors_w.items():
        ratios = []
        valid_labels = []
        for label, (lo, hi) in zip(er_labels, er_bounds):
            group = [s for s in all_signals if lo <= s['er_20'] < hi]
            if len(group) >= 10:
                ratios.append(np.median([s[f'ratio_{wname}'] for s in group]))
                valid_labels.append(label)
        if ratios:
            ax.plot(valid_labels, ratios, 'o-', color=color, label=wname, linewidth=2)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.3)
    ax.legend()
    ax.set_xlabel('ER(20) 区间')
    ax.set_ylabel('MFE/MAE 中位数')
    ax.set_title('A+B: 各ER段 x 各窗口', fontweight='bold')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(True, alpha=0.3)

    # 图2: A类单独
    ax = axes[0, 1]
    for wname, color in colors_w.items():
        ratios = []
        valid_labels = []
        for label, (lo, hi) in zip(er_labels, er_bounds):
            group = [s for s in a_all if lo <= s['er_20'] < hi]
            if len(group) >= 10:
                ratios.append(np.median([s[f'ratio_{wname}'] for s in group]))
                valid_labels.append(label)
        if ratios:
            ax.plot(valid_labels, ratios, 'o-', color=color, label=wname, linewidth=2)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.3)
    ax.legend()
    ax.set_xlabel('ER(20) 区间')
    ax.set_ylabel('MFE/MAE 中位数')
    ax.set_title('A类: 各ER段 x 各窗口', fontweight='bold')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(True, alpha=0.3)

    # 图3: B类单独
    ax = axes[1, 0]
    for wname, color in colors_w.items():
        ratios = []
        valid_labels = []
        for label, (lo, hi) in zip(er_labels, er_bounds):
            group = [s for s in b_all if lo <= s['er_20'] < hi]
            if len(group) >= 10:
                ratios.append(np.median([s[f'ratio_{wname}'] for s in group]))
                valid_labels.append(label)
        if ratios:
            ax.plot(valid_labels, ratios, 'o-', color=color, label=wname, linewidth=2)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.3)
    ax.legend()
    ax.set_xlabel('ER(20) 区间')
    ax.set_ylabel('MFE/MAE 中位数')
    ax.set_title('B类: 各ER段 x 各窗口', fontweight='bold')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(True, alpha=0.3)

    # 图4: 各ER段信号数量 + 胜率
    ax = axes[1, 1]
    counts_a = []
    counts_b = []
    win_rates = []
    for lo, hi in er_bounds:
        a_c = len([s for s in a_all if lo <= s['er_20'] < hi])
        b_c = len([s for s in b_all if lo <= s['er_20'] < hi])
        all_c = [s for s in all_signals if lo <= s['er_20'] < hi]
        w = sum(1 for s in all_c if s['ratio_2h'] > 1) / len(all_c) * 100 if all_c else 0
        counts_a.append(a_c)
        counts_b.append(b_c)
        win_rates.append(w)

    x = np.arange(len(er_labels))
    w_bar = 0.35
    ax.bar(x - w_bar/2, counts_a, w_bar, label='A类', color='#ff9800', alpha=0.7)
    ax.bar(x + w_bar/2, counts_b, w_bar, label='B类', color='#2196f3', alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(x, win_rates, 'D-', color='#4CAF50', linewidth=2, label='2h胜率')
    ax.set_xticks(x)
    ax.set_xticklabels(er_labels, rotation=20)
    ax.set_ylabel('信号数量')
    ax2.set_ylabel('2h胜率 (%)', color='#4CAF50')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax.set_title('各ER段: 信号数量 + 胜率', fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'er_simple.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图表: {out_path}")


if __name__ == '__main__':
    main()
