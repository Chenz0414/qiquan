# -*- coding: utf-8 -*-
"""
ER(20) 样本外验证 + 最优阈值扫描
=================================
1. 前60天训练 / 后60天验证（时间维度）
2. 原6品种 vs 新26品种（品种维度）
3. ER阈值从0.05到0.70扫描，看MFE/MAE和信号数量的平衡
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import timedelta
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MA_FAST = 10
MA_SLOW = 20
MFE_WINDOW = 48

# 原6品种（训练集）
OLD_SYMBOLS = ['SHFE_ag', 'SHFE_au', 'INE_sc', 'GFEX_lc', 'DCE_v', 'DCE_lh']

# ER阈值扫描范围
ER_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def get_all_10min_symbols():
    """扫描缓存目录，获取所有有10min数据的品种"""
    symbols = []
    for f in os.listdir(CACHE_DIR):
        if f.endswith('_10min_170d.parquet'):
            sym = f.replace('_10min_170d.parquet', '')
            symbols.append(sym)
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


def detect_signals(df):
    signals = []
    n = len(df)
    warmup = 30
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
            er = row['er_20']
            if pd.isna(er):
                continue
            sig['er_20'] = er

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


def eval_threshold(signals, threshold):
    """评估某个ER阈值"""
    above = [s for s in signals if s['er_20'] >= threshold]
    below = [s for s in signals if s['er_20'] < threshold]

    if not above:
        return None

    n_total = len(signals)
    n_above = len(above)
    pass_rate = n_above / n_total * 100

    above_ratio = np.median([s['mfe_mae'] for s in above])
    below_ratio = np.median([s['mfe_mae'] for s in below]) if below else 0
    above_mfe = np.mean([s['mfe'] for s in above])
    above_mae = np.mean([s['mae'] for s in above])

    # MFE/MAE > 1的比例（实际能盈利的信号占比）
    win_rate = sum(1 for s in above if s['mfe_mae'] > 1) / n_above * 100

    return {
        'threshold': threshold,
        'n_total': n_total,
        'n_pass': n_above,
        'pass_rate': pass_rate,
        'above_ratio': above_ratio,
        'below_ratio': below_ratio,
        'diff': above_ratio - below_ratio,
        'above_mfe': above_mfe,
        'above_mae': above_mae,
        'win_rate': win_rate,
    }


def main():
    all_symbols = get_all_10min_symbols()
    new_symbols = [s for s in all_symbols if s not in OLD_SYMBOLS]

    print("=" * 100)
    print(f"  ER(20) 样本外验证 + 最优阈值扫描")
    print(f"  原6品种: {len(OLD_SYMBOLS)}个 | 新品种: {len(new_symbols)}个 | 共{len(all_symbols)}个")
    print("=" * 100)

    # 收集所有信号
    groups = {
        'old_first60': [],   # 原品种前60天
        'old_last60': [],    # 原品种后60天
        'new_first60': [],   # 新品种前60天
        'new_last60': [],    # 新品种后60天
    }

    symbol_stats = {}

    for sym_key in all_symbols:
        df = load_cached(sym_key)
        if df is None:
            continue
        df = prepare_data(df)

        # 120天窗口
        cutoff_120 = df['datetime'].iloc[-1] - timedelta(days=120)
        df_120 = df[df['datetime'] >= cutoff_120].copy().reset_index(drop=True)

        if len(df_120) < 200:
            continue

        # 前60天 / 后60天
        mid_date = df_120['datetime'].iloc[0] + timedelta(days=60)
        df_first = df_120[df_120['datetime'] < mid_date].copy().reset_index(drop=True)
        df_last = df_120[df_120['datetime'] >= mid_date].copy().reset_index(drop=True)

        sigs_first = detect_signals(df_first)
        sigs_last = detect_signals(df_last)

        for s in sigs_first:
            s['symbol'] = sym_key
        for s in sigs_last:
            s['symbol'] = sym_key

        is_old = sym_key in OLD_SYMBOLS
        if is_old:
            groups['old_first60'].extend(sigs_first)
            groups['old_last60'].extend(sigs_last)
        else:
            groups['new_first60'].extend(sigs_first)
            groups['new_last60'].extend(sigs_last)

        # 品种级统计
        all_sigs = sigs_first + sigs_last
        if all_sigs:
            symbol_stats[sym_key] = {
                'n': len(all_sigs),
                'er_mean': np.mean([s['er_20'] for s in all_sigs]),
                'mfe_mae_med': np.median([s['mfe_mae'] for s in all_sigs]),
            }

    print(f"\n  信号数量:")
    for key, sigs in groups.items():
        print(f"    {key}: {len(sigs)}")

    # ============================================================
    # 1. 阈值扫描：全维度
    # ============================================================
    scan_groups = {
        '原品种-前60天(训练)': groups['old_first60'],
        '原品种-后60天(验证)': groups['old_last60'],
        '新品种-前60天(验证)': groups['new_first60'],
        '新品种-后60天(验证)': groups['new_last60'],
        '全部品种-全时段':     groups['old_first60'] + groups['old_last60'] + groups['new_first60'] + groups['new_last60'],
    }

    print(f"\n{'='*100}")
    print(f"  ER(20) 阈值扫描")
    print(f"{'='*100}")

    for group_name, sigs in scan_groups.items():
        if not sigs:
            continue
        print(f"\n  --- {group_name} (N={len(sigs)}) ---")
        print(f"  {'阈值':>6} | {'通过数':>5} {'通过率':>6} | {'达标MFE/MAE':>11} {'未达标':>8} {'差':>7} | {'达标胜率':>8} {'达标MFE':>8} {'达标MAE':>8}")
        print(f"  {'-'*90}")

        for t in ER_THRESHOLDS:
            r = eval_threshold(sigs, t)
            if r and r['n_pass'] >= 5:
                print(f"  {t:>6.2f} | {r['n_pass']:>5} {r['pass_rate']:>5.1f}% | "
                      f"{r['above_ratio']:>10.2f} {r['below_ratio']:>7.2f} {r['diff']:>+6.2f}  | "
                      f"{r['win_rate']:>7.1f}% {r['above_mfe']:>7.2f}% {r['above_mae']:>7.2f}%")

    # ============================================================
    # 2. 分品种看ER>0.3的效果（所有32个品种）
    # ============================================================
    print(f"\n{'='*100}")
    print(f"  分品种：ER>=0.3 vs ER<0.3 的 MFE/MAE中位数差值")
    print(f"{'='*100}")

    all_sigs_flat = []
    for sigs in groups.values():
        all_sigs_flat.extend(sigs)

    # 按品种分
    sym_diffs = {}
    for sym_key in all_symbols:
        sym_sigs = [s for s in all_sigs_flat if s['symbol'] == sym_key]
        if len(sym_sigs) < 15:
            continue
        above = [s for s in sym_sigs if s['er_20'] >= 0.3]
        below = [s for s in sym_sigs if s['er_20'] < 0.3]
        if not above or not below:
            continue
        a_ratio = np.median([s['mfe_mae'] for s in above])
        b_ratio = np.median([s['mfe_mae'] for s in below])
        diff = a_ratio - b_ratio
        is_old = sym_key in OLD_SYMBOLS
        tag = '*' if is_old else ' '
        sym_diffs[sym_key] = {'diff': diff, 'is_old': is_old, 'n': len(sym_sigs),
                              'n_above': len(above), 'a_ratio': a_ratio, 'b_ratio': b_ratio}

    # 排序输出
    sorted_syms = sorted(sym_diffs.items(), key=lambda x: -x[1]['diff'])
    pos_count_old = 0
    pos_count_new = 0
    total_old = 0
    total_new = 0

    print(f"  {'品种':<12} {'标记':>4} {'N':>4} {'达标N':>5} | {'ER>=0.3':>8} {'ER<0.3':>8} {'差':>8}")
    print(f"  {'-'*60}")
    for sym_key, data in sorted_syms:
        tag = '原' if data['is_old'] else '新'
        print(f"  {sym_key:<12} {tag:>4} {data['n']:>4} {data['n_above']:>5} | "
              f"{data['a_ratio']:>7.2f} {data['b_ratio']:>7.2f} {data['diff']:>+7.2f}")
        if data['is_old']:
            total_old += 1
            if data['diff'] > 0:
                pos_count_old += 1
        else:
            total_new += 1
            if data['diff'] > 0:
                pos_count_new += 1

    print(f"\n  原品种正向: {pos_count_old}/{total_old}")
    print(f"  新品种正向: {pos_count_new}/{total_new}")
    print(f"  全部正向:   {pos_count_old + pos_count_new}/{total_old + total_new}")

    # ============================================================
    # 图表
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # 图1: 阈值 vs MFE/MAE（训练 vs 各验证集）
    ax = axes[0, 0]
    line_groups = {
        '训练(原-前60天)': groups['old_first60'],
        '验证(原-后60天)': groups['old_last60'],
        '验证(新-前60天)': groups['new_first60'],
        '验证(新-后60天)': groups['new_last60'],
    }
    colors_line = ['#2196f3', '#4CAF50', '#f44336', '#ff9800']
    for (name, sigs), color in zip(line_groups.items(), colors_line):
        if not sigs:
            continue
        thresholds = []
        ratios = []
        for t in ER_THRESHOLDS:
            above = [s for s in sigs if s['er_20'] >= t]
            if len(above) >= 5:
                thresholds.append(t)
                ratios.append(np.median([s['mfe_mae'] for s in above]))
        if thresholds:
            ax.plot(thresholds, ratios, 'o-', color=color, label=name, linewidth=2)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlabel('ER(20) 阈值')
    ax.set_ylabel('MFE/MAE 中位数')
    ax.set_title('ER阈值 vs MFE/MAE（训练 vs 验证）', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 图2: 阈值 vs 通过率（信号数量）
    ax = axes[0, 1]
    all_flat = groups['old_first60'] + groups['old_last60'] + groups['new_first60'] + groups['new_last60']
    thresholds_plot = []
    pass_rates = []
    win_rates = []
    for t in ER_THRESHOLDS:
        r = eval_threshold(all_flat, t)
        if r:
            thresholds_plot.append(t)
            pass_rates.append(r['pass_rate'])
            win_rates.append(r['win_rate'])

    ax2 = ax.twinx()
    ax.bar(thresholds_plot, pass_rates, width=0.04, color='#2196f3', alpha=0.5, label='信号通过率')
    ax2.plot(thresholds_plot, win_rates, 'o-', color='#4CAF50', linewidth=2, label='MFE/MAE>1胜率')
    ax.set_xlabel('ER(20) 阈值')
    ax.set_ylabel('信号通过率 (%)', color='#2196f3')
    ax2.set_ylabel('MFE/MAE>1胜率 (%)', color='#4CAF50')
    ax.set_title('阈值 vs 通过率 & 胜率', fontweight='bold')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax.grid(True, alpha=0.3)

    # 图3: 分品种差值（ER>=0.3高组-低组）
    ax = axes[1, 0]
    sym_names_plot = [k for k, _ in sorted_syms]
    diffs_plot = [v['diff'] for _, v in sorted_syms]
    colors_bar = ['#2196f3' if sym_diffs[k]['is_old'] else '#ff9800' for k in sym_names_plot]
    bars = ax.barh(sym_names_plot[::-1], diffs_plot[::-1], color=colors_bar[::-1], alpha=0.7)
    ax.axvline(x=0, color='red', linewidth=0.8)
    ax.set_title(f'分品种: ER>=0.3 MFE/MAE差值\n蓝=原品种 橙=新品种 (正向{pos_count_old+pos_count_new}/{total_old+total_new})', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 图4: 期望收益 = MFE/MAE * 通过率（综合得分）
    ax = axes[1, 1]
    # 综合得分 = 达标MFE/MAE中位数 * 通过的信号数（越多越好，但质量也要好）
    thresholds_score = []
    scores = []
    mfe_mae_vals = []
    n_pass_vals = []
    for t in ER_THRESHOLDS:
        r = eval_threshold(all_flat, t)
        if r and r['n_pass'] >= 10:
            thresholds_score.append(t)
            mfe_mae_vals.append(r['above_ratio'])
            n_pass_vals.append(r['n_pass'])
            # 综合得分：MFE/MAE中位数 * ln(信号数)
            score = r['above_ratio'] * np.log(r['n_pass'] + 1)
            scores.append(score)

    if thresholds_score:
        ax.plot(thresholds_score, scores, 'o-', color='#9C27B0', linewidth=2, label='综合得分')
        best_idx = np.argmax(scores)
        best_t = thresholds_score[best_idx]
        ax.axvline(x=best_t, color='green', linestyle='--', alpha=0.7,
                   label=f'最优阈值={best_t:.2f}')
        ax.scatter([best_t], [scores[best_idx]], color='green', s=150, zorder=5)

        # 标注几个关键点
        for t_mark in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
            if t_mark in thresholds_score:
                idx = thresholds_score.index(t_mark)
                ax.annotate(f'{t_mark}\nN={n_pass_vals[idx]}\nR={mfe_mae_vals[idx]:.2f}',
                           (t_mark, scores[idx]), textcoords="offset points",
                           xytext=(0, 15), fontsize=7, ha='center')

    ax.legend()
    ax.set_xlabel('ER(20) 阈值')
    ax.set_ylabel('综合得分 (MFE/MAE * ln(N))')
    ax.set_title('阈值综合得分（质量 x 数量平衡）', fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'er_validate.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图表: {out_path}")


if __name__ == '__main__':
    main()
