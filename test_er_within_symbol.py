# -*- coding: utf-8 -*-
"""
品种内部ER(20)拆分：同一品种，干净时做 vs 不干净时不做
=====================================================
不是比品种之间的差异，而是看同一品种内部，
ER(20)高的信号和ER(20)低的信号表现差多少。
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

SYMBOLS = {
    'SHFE_ag': '白银',
    'SHFE_au': '黄金',
    'INE_sc':  '原油',
    'GFEX_lc': '碳酸锂',
    'DCE_v':   'PVC',
    'DCE_lh':  '生猪',
}

MA_FAST = 10
MA_SLOW = 20
MFE_WINDOW = 48


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

    # ER多周期
    for p in [10, 15, 20, 25, 30]:
        net = (df['close'] - df['close'].shift(p)).abs()
        bar_sum = df['close'].diff().abs().rolling(p).sum()
        df[f'er_{p}'] = net / bar_sum.replace(0, np.nan)
    return df


def detect_signals_with_mfe(df):
    signals = []
    n = len(df)
    warmup = 55
    b_below_start = -1
    b_pullback_low = None
    b_pullback_high = None
    prev_trend = 0

    for i in range(warmup, n):
        row = df.iloc[i]
        if pd.isna(row['ma_fast']) or pd.isna(row['ma_slow']):
            continue
        close = row['close']
        ma_f = row['ma_fast']
        trend = row['trend']
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

        sig = None
        if trend == 1:
            if b_below_start == -1:
                if close < ma_f and prev['close'] >= prev['ma_fast']:
                    b_below_start = i
                    b_pullback_low = row['low']
            else:
                b_pullback_low = min(b_pullback_low, row['low'])
                if close > ma_f:
                    sig = {'direction': 'long', 'idx': i, 'time': row['datetime'], 'entry_price': close}
                    b_below_start = -1
        elif trend == -1:
            if b_below_start == -1:
                if close > ma_f and prev['close'] <= prev['ma_fast']:
                    b_below_start = i
                    b_pullback_high = row['high']
            else:
                b_pullback_high = max(b_pullback_high, row['high'])
                if close < ma_f:
                    sig = {'direction': 'short', 'idx': i, 'time': row['datetime'], 'entry_price': close}
                    b_below_start = -1

        if sig is not None:
            for p in [10, 15, 20, 25, 30]:
                val = row[f'er_{p}']
                sig[f'er_{p}'] = val if not pd.isna(val) else np.nan

            end_idx = min(i + MFE_WINDOW, len(df) - 1)
            if i + 1 <= end_idx:
                segment = df.iloc[i + 1: end_idx + 1]
                if sig['direction'] == 'short':
                    sig['mfe'] = max((close - segment['low'].min()) / close * 100, 0)
                    sig['mae'] = max((segment['high'].max() - close) / close * 100, 0)
                else:
                    sig['mfe'] = max((segment['high'].max() - close) / close * 100, 0)
                    sig['mae'] = max((close - segment['low'].min()) / close * 100, 0)
                sig['mfe_mae'] = sig['mfe'] / sig['mae'] if sig['mae'] > 0 else 10
            else:
                continue
            signals.append(sig)
    return signals


def main():
    print("=" * 100)
    print("  品种内部 ER(20) 拆分：干净时做 vs 不干净时不做")
    print("=" * 100)

    all_signals = {}

    for sym_key, sym_name in SYMBOLS.items():
        df = load_cached(sym_key)
        if df is None:
            continue
        df = prepare_data(df)
        cutoff = df['datetime'].iloc[-1] - timedelta(days=120)
        df_120 = df[df['datetime'] >= cutoff].copy().reset_index(drop=True)
        signals = detect_signals_with_mfe(df_120)
        all_signals[sym_name] = {'signals': signals, 'df': df_120}

    # ============================================================
    # 每个品种内部按ER(20) P50拆分
    # ============================================================
    print(f"\n{'='*100}")
    print(f"  每品种内部：ER(20) 高组 vs 低组")
    print(f"{'='*100}")

    for sym_name, data in all_signals.items():
        sigs = data['signals']
        valid = [s for s in sigs if not np.isnan(s['er_20'])]
        if len(valid) < 20:
            continue

        p50 = np.percentile([s['er_20'] for s in valid], 50)
        p33 = np.percentile([s['er_20'] for s in valid], 33)
        p67 = np.percentile([s['er_20'] for s in valid], 67)

        groups = [
            ('低ER', [s for s in valid if s['er_20'] <= p33]),
            ('中ER', [s for s in valid if p33 < s['er_20'] <= p67]),
            ('高ER', [s for s in valid if s['er_20'] > p67]),
        ]

        print(f"\n  {sym_name} (N={len(valid)}, ER20 P33={p33:.3f} P67={p67:.3f})")
        print(f"  {'组':<8} {'N':>4} | {'MFE':>7} {'MAE':>7} {'MFE/MAE':>8} | {'MFE>3%':>7} {'MFE>2%':>7} | {'盈亏比>2':>8}")
        print(f"  {'-'*75}")

        for label, group_sigs in groups:
            if not group_sigs:
                continue
            n = len(group_sigs)
            mfe = np.mean([s['mfe'] for s in group_sigs])
            mae = np.mean([s['mae'] for s in group_sigs])
            ratio = np.median([s['mfe_mae'] for s in group_sigs])
            burst3 = sum(1 for s in group_sigs if s['mfe'] >= 3.0) / n * 100
            burst2 = sum(1 for s in group_sigs if s['mfe'] >= 2.0) / n * 100
            good_ratio = sum(1 for s in group_sigs if s['mfe_mae'] >= 2.0) / n * 100
            print(f"  {label:<8} {n:>4} | {mfe:>6.2f}% {mae:>6.2f}% {ratio:>7.2f}  | "
                  f"{burst3:>6.1f}% {burst2:>6.1f}%  | {good_ratio:>7.1f}%")

    # ============================================================
    # ER最优周期选择
    # ============================================================
    print(f"\n{'='*100}")
    print(f"  ER周期选择：哪个周期的ER最能区分MFE/MAE？（品种内P50拆分）")
    print(f"{'='*100}")

    er_periods = [10, 15, 20, 25, 30]
    print(f"  {'周期':<6}", end='')
    for name in SYMBOLS.values():
        print(f" {name:>8}", end='')
    print("  | 均值  正向")
    print(f"  {'-'*75}")

    for p in er_periods:
        label = f"ER({p})"
        print(f"  {label:<6}", end='')
        diffs = []
        pos = 0
        for sym_name, data in all_signals.items():
            sigs = data['signals']
            valid = [s for s in sigs if not np.isnan(s[f'er_{p}'])]
            if len(valid) < 20:
                print(f"     N/A", end='')
                continue
            med = np.percentile([s[f'er_{p}'] for s in valid], 50)
            low = [s for s in valid if s[f'er_{p}'] <= med]
            high = [s for s in valid if s[f'er_{p}'] > med]
            low_r = np.median([s['mfe_mae'] for s in low])
            high_r = np.median([s['mfe_mae'] for s in high])
            diff = high_r - low_r
            diffs.append(diff)
            if diff > 0:
                pos += 1
            print(f" {diff:>+7.2f}", end='')
        avg = np.mean(diffs) if diffs else 0
        print(f"  | {avg:>+5.2f}  {pos}/6")

    # ============================================================
    # 具体案例：同品种内ER高 vs ER低的信号
    # ============================================================
    print(f"\n{'='*100}")
    print(f"  碳酸锂：ER高信号 vs ER低信号（具体案例）")
    print(f"{'='*100}")

    lc_data = all_signals.get('碳酸锂')
    if lc_data:
        lc_sigs = [s for s in lc_data['signals'] if not np.isnan(s['er_20'])]
        lc_p50 = np.percentile([s['er_20'] for s in lc_sigs], 50)

        print(f"\n  高ER信号（干净阶段，ER>{lc_p50:.3f}）前5名:")
        high_er = sorted([s for s in lc_sigs if s['er_20'] > lc_p50], key=lambda x: -x['mfe_mae'])
        for s in high_er[:5]:
            d = '多' if s['direction'] == 'long' else '空'
            print(f"    {s['time']} {d} ER={s['er_20']:.3f} | MFE={s['mfe']:.2f}% MAE={s['mae']:.2f}% MFE/MAE={s['mfe_mae']:.2f}")

        print(f"\n  低ER信号（震荡阶段，ER<={lc_p50:.3f}）前5名:")
        low_er = sorted([s for s in lc_sigs if s['er_20'] <= lc_p50], key=lambda x: x['mfe_mae'])
        for s in low_er[:5]:
            d = '多' if s['direction'] == 'long' else '空'
            print(f"    {s['time']} {d} ER={s['er_20']:.3f} | MFE={s['mfe']:.2f}% MAE={s['mae']:.2f}% MFE/MAE={s['mfe_mae']:.2f}")

    # 生猪也看一下
    print(f"\n{'='*100}")
    print(f"  生猪：ER高信号 vs ER低信号（具体案例）")
    print(f"{'='*100}")

    zhu_data = all_signals.get('生猪')
    if zhu_data:
        zhu_sigs = [s for s in zhu_data['signals'] if not np.isnan(s['er_20'])]
        zhu_p50 = np.percentile([s['er_20'] for s in zhu_sigs], 50)

        print(f"\n  高ER信号（干净阶段，ER>{zhu_p50:.3f}）前5名:")
        high_er = sorted([s for s in zhu_sigs if s['er_20'] > zhu_p50], key=lambda x: -x['mfe_mae'])
        for s in high_er[:5]:
            d = '多' if s['direction'] == 'long' else '空'
            print(f"    {s['time']} {d} ER={s['er_20']:.3f} | MFE={s['mfe']:.2f}% MAE={s['mae']:.2f}% MFE/MAE={s['mfe_mae']:.2f}")

        print(f"\n  低ER信号（震荡阶段，ER<={zhu_p50:.3f}）前5名:")
        low_er = sorted([s for s in zhu_sigs if s['er_20'] <= zhu_p50], key=lambda x: x['mfe_mae'])
        for s in low_er[:5]:
            d = '多' if s['direction'] == 'long' else '空'
            print(f"    {s['time']} {d} ER={s['er_20']:.3f} | MFE={s['mfe']:.2f}% MAE={s['mae']:.2f}% MFE/MAE={s['mfe_mae']:.2f}")

    # ============================================================
    # 图表
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    sym_list = list(all_signals.keys())

    for idx, sym_name in enumerate(sym_list):
        ax = axes[idx // 3, idx % 3]
        sigs = all_signals[sym_name]['signals']
        valid = [s for s in sigs if not np.isnan(s['er_20'])]
        if len(valid) < 20:
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center')
            ax.set_title(sym_name)
            continue

        p33 = np.percentile([s['er_20'] for s in valid], 33)
        p67 = np.percentile([s['er_20'] for s in valid], 67)
        groups = [
            ('低ER', [s for s in valid if s['er_20'] <= p33], '#f44336'),
            ('中ER', [s for s in valid if p33 < s['er_20'] <= p67], '#FFC107'),
            ('高ER', [s for s in valid if s['er_20'] > p67], '#4CAF50'),
        ]

        box_data = []
        box_labels = []
        box_colors = []
        for label, g, color in groups:
            if g:
                box_data.append([min(s['mfe_mae'], 10) for s in g])
                med_ratio = np.median([s['mfe_mae'] for s in g])
                box_labels.append(f"{label}\nN={len(g)}\nmed={med_ratio:.2f}")
                box_colors.append(color)

        if box_data:
            bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, showmeans=True,
                           meanprops=dict(marker='D', markerfacecolor='white', markersize=5))
            for patch, c in zip(bp['boxes'], box_colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.5)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.4)
        ax.set_title(f'{sym_name}  ER(20)三分组', fontweight='bold')
        ax.set_ylabel('MFE/MAE')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'er_within_symbol.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图表: {out_path}")


if __name__ == '__main__':
    main()
