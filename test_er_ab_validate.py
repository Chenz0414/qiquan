# -*- coding: utf-8 -*-
"""
ER(20) x A/B类信号 完整验证
============================
核心假说：高ER环境（干净趋势）更多产生A类信号，低ER更多产生B类。
之前只测B类导致高ER样本不够。

维度：
1. A类 vs B类 在不同ER区间的表现
2. 高ER+A类 vs 低ER+B类 的对比
3. 32品种 x 前后60天验证
4. ER阈值扫描（A类和B类分别做）
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
MFE_WINDOW = 48

OLD_SYMBOLS = ['SHFE_ag', 'SHFE_au', 'INE_sc', 'GFEX_lc', 'DCE_v', 'DCE_lh']
ER_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


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
    """同时检测A类和B类信号"""
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
        prev_close = prev['close']
        prev_ma_f = prev['ma_fast']
        if pd.isna(prev_ma_f):
            continue

        # ====== 多头 ======
        if trend == 1:
            # A类：影线碰10MA但收盘未破，且前一根也在10MA上方
            if low <= ma_f and close > ma_f and prev_close > prev_ma_f:
                sig = _build_signal(df, i, 'A', 'long', close, er)
                if sig:
                    signals.append(sig)

            # B类
            if b_below_start == -1:
                if close < ma_f and prev_close >= prev_ma_f:
                    b_below_start = i
                    b_pullback_low = low
            else:
                b_pullback_low = min(b_pullback_low, low)
                if close > ma_f:
                    pb_bars = i - b_below_start
                    sig = _build_signal(df, i, 'B', 'long', close, er)
                    if sig:
                        sig['pb_bars'] = pb_bars
                        signals.append(sig)
                    b_below_start = -1
                    b_pullback_low = None

        # ====== 空头 ======
        elif trend == -1:
            # A类
            if high >= ma_f and close < ma_f and prev_close < prev_ma_f:
                sig = _build_signal(df, i, 'A', 'short', close, er)
                if sig:
                    signals.append(sig)

            # B类
            if b_below_start == -1:
                if close > ma_f and prev_close <= prev_ma_f:
                    b_below_start = i
                    b_pullback_high = high
            else:
                b_pullback_high = max(b_pullback_high, high)
                if close < ma_f:
                    pb_bars = i - b_below_start
                    sig = _build_signal(df, i, 'B', 'short', close, er)
                    if sig:
                        sig['pb_bars'] = pb_bars
                        signals.append(sig)
                    b_below_start = -1
                    b_pullback_high = None

    return signals


def _build_signal(df, i, sig_type, direction, close, er):
    """构建信号，计算MFE/MAE"""
    end_idx = min(i + MFE_WINDOW, len(df) - 1)
    if i + 1 > end_idx:
        return None

    segment = df.iloc[i + 1: end_idx + 1]
    if direction == 'short':
        mfe = max((close - segment['low'].min()) / close * 100, 0)
        mae = max((segment['high'].max() - close) / close * 100, 0)
    else:
        mfe = max((segment['high'].max() - close) / close * 100, 0)
        mae = max((close - segment['low'].min()) / close * 100, 0)

    return {
        'type': sig_type,
        'direction': direction,
        'idx': i,
        'time': df.iloc[i]['datetime'],
        'entry_price': close,
        'er_20': er,
        'mfe': mfe,
        'mae': mae,
        'mfe_mae': mfe / mae if mae > 0 else 10,
    }


def group_stats(sigs, label=''):
    """统计一组信号"""
    if not sigs:
        return None
    n = len(sigs)
    return {
        'label': label,
        'n': n,
        'mfe': np.mean([s['mfe'] for s in sigs]),
        'mae': np.mean([s['mae'] for s in sigs]),
        'mfe_mae': np.median([s['mfe_mae'] for s in sigs]),
        'win_rate': sum(1 for s in sigs if s['mfe_mae'] > 1) / n * 100,
        'er_mean': np.mean([s['er_20'] for s in sigs]),
    }


def print_stats(stats):
    if stats is None:
        return
    s = stats
    print(f"  {s['label']:<22} {s['n']:>5} | "
          f"MFE={s['mfe']:>5.2f}% MAE={s['mae']:>5.2f}% "
          f"MFE/MAE={s['mfe_mae']:>5.2f} "
          f"win={s['win_rate']:>5.1f}% "
          f"ER={s['er_mean']:>.3f}")


def main():
    all_symbols = get_all_10min_symbols()
    new_symbols = [s for s in all_symbols if s not in OLD_SYMBOLS]

    print("=" * 100)
    print(f"  ER(20) x A/B类信号 完整验证 ({len(all_symbols)}品种)")
    print("=" * 100)

    # 收集信号，按时间段拆分
    all_first = []  # 前60天
    all_last = []   # 后60天
    symbol_signals = {}

    for sym_key in all_symbols:
        df = load_cached(sym_key)
        if df is None:
            continue
        df = prepare_data(df)

        cutoff_120 = df['datetime'].iloc[-1] - timedelta(days=120)
        df_120 = df[df['datetime'] >= cutoff_120].copy().reset_index(drop=True)
        if len(df_120) < 200:
            continue

        mid_date = df_120['datetime'].iloc[0] + timedelta(days=60)
        df_first = df_120[df_120['datetime'] < mid_date].copy().reset_index(drop=True)
        df_last = df_120[df_120['datetime'] >= mid_date].copy().reset_index(drop=True)

        sigs_first = detect_ab_signals(df_first)
        sigs_last = detect_ab_signals(df_last)

        for s in sigs_first + sigs_last:
            s['symbol'] = sym_key
            s['is_old'] = sym_key in OLD_SYMBOLS

        all_first.extend(sigs_first)
        all_last.extend(sigs_last)
        symbol_signals[sym_key] = sigs_first + sigs_last

    all_sigs = all_first + all_last
    print(f"  前60天: {len(all_first)}信号 | 后60天: {len(all_last)}信号 | 总计: {len(all_sigs)}")

    a_sigs = [s for s in all_sigs if s['type'] == 'A']
    b_sigs = [s for s in all_sigs if s['type'] == 'B']
    print(f"  A类: {len(a_sigs)} | B类: {len(b_sigs)}")

    # ============================================================
    # 1. 基础问题：高ER环境产生什么信号？
    # ============================================================
    print(f"\n{'='*100}")
    print(f"  高ER环境vs低ER环境，A/B类信号的分布")
    print(f"{'='*100}")

    er_p50 = np.percentile([s['er_20'] for s in all_sigs], 50)
    er_p33 = np.percentile([s['er_20'] for s in all_sigs], 33)
    er_p67 = np.percentile([s['er_20'] for s in all_sigs], 67)

    for label, lo, hi in [('低ER', 0, er_p33), ('中ER', er_p33, er_p67), ('高ER', er_p67, 1.1)]:
        group = [s for s in all_sigs if lo <= s['er_20'] < hi]
        a_count = sum(1 for s in group if s['type'] == 'A')
        b_count = sum(1 for s in group if s['type'] == 'B')
        total = a_count + b_count
        if total > 0:
            print(f"  {label} (ER {lo:.3f}~{hi:.3f}): "
                  f"A类={a_count}({a_count/total*100:.0f}%) B类={b_count}({b_count/total*100:.0f}%)")

    # ============================================================
    # 2. A/B类分别按ER拆分
    # ============================================================
    print(f"\n{'='*100}")
    print(f"  A类和B类分别按ER(20)三分组")
    print(f"{'='*100}")

    print(f"\n  {'分组':<22} {'N':>5} | {'MFE':>7} {'MAE':>7} {'MFE/MAE':>8} {'胜率':>6} {'ER均值':>7}")
    print(f"  {'-'*75}")

    for sig_type in ['A', 'B']:
        type_sigs = [s for s in all_sigs if s['type'] == sig_type]
        for label, lo, hi in [('低ER', 0, er_p33), ('中ER', er_p33, er_p67), ('高ER', er_p67, 1.1)]:
            group = [s for s in type_sigs if lo <= s['er_20'] < hi]
            print_stats(group_stats(group, f"{sig_type}类+{label}"))
        print(f"  {'-'*75}")

    # ============================================================
    # 3. 核心4组对比
    # ============================================================
    print(f"\n{'='*100}")
    print(f"  核心对比：ER x 信号类型")
    print(f"{'='*100}")

    core_groups = {
        '低ER+B类': [s for s in all_sigs if s['er_20'] < er_p33 and s['type'] == 'B'],
        '低ER+A类': [s for s in all_sigs if s['er_20'] < er_p33 and s['type'] == 'A'],
        '高ER+B类': [s for s in all_sigs if s['er_20'] >= er_p67 and s['type'] == 'B'],
        '高ER+A类': [s for s in all_sigs if s['er_20'] >= er_p67 and s['type'] == 'A'],
    }

    print(f"\n  {'分组':<22} {'N':>5} | {'MFE':>7} {'MAE':>7} {'MFE/MAE':>8} {'胜率':>6} {'ER均值':>7}")
    print(f"  {'-'*75}")
    for name, sigs in core_groups.items():
        print_stats(group_stats(sigs, name))

    # ============================================================
    # 4. 样本外验证：前60天 vs 后60天
    # ============================================================
    print(f"\n{'='*100}")
    print(f"  样本外验证：前60天(训练) vs 后60天(验证)")
    print(f"{'='*100}")

    for period_name, period_sigs in [('前60天', all_first), ('后60天', all_last)]:
        print(f"\n  --- {period_name} (N={len(period_sigs)}) ---")
        print(f"  {'分组':<22} {'N':>5} | {'MFE':>7} {'MAE':>7} {'MFE/MAE':>8} {'胜率':>6}")
        print(f"  {'-'*70}")

        p33 = np.percentile([s['er_20'] for s in period_sigs], 33)
        p67 = np.percentile([s['er_20'] for s in period_sigs], 67)

        for sig_type in ['A', 'B', 'A+B']:
            if sig_type == 'A+B':
                type_sigs = period_sigs
            else:
                type_sigs = [s for s in period_sigs if s['type'] == sig_type]

            for label, lo, hi in [('低ER', 0, p33), ('高ER', p67, 1.1)]:
                group = [s for s in type_sigs if lo <= s['er_20'] < hi]
                print_stats(group_stats(group, f"{sig_type}+{label}"))
            print_stats(group_stats(type_sigs, f"{sig_type}(全部)"))
            print(f"  {'-'*70}")

    # ============================================================
    # 5. 分品种：ER高组(A+B) vs ER低组(A+B) 的MFE/MAE差
    # ============================================================
    print(f"\n{'='*100}")
    print(f"  分品种验证：A+B合并后 ER高低组差值")
    print(f"{'='*100}")

    sym_results = []
    for sym_key in sorted(symbol_signals.keys()):
        sigs = symbol_signals[sym_key]
        if len(sigs) < 30:
            continue

        sym_p33 = np.percentile([s['er_20'] for s in sigs], 33)
        sym_p67 = np.percentile([s['er_20'] for s in sigs], 67)

        low_all = [s for s in sigs if s['er_20'] < sym_p33]
        high_all = [s for s in sigs if s['er_20'] >= sym_p67]

        low_a = [s for s in low_all if s['type'] == 'A']
        low_b = [s for s in low_all if s['type'] == 'B']
        high_a = [s for s in high_all if s['type'] == 'A']
        high_b = [s for s in high_all if s['type'] == 'B']

        low_ratio = np.median([s['mfe_mae'] for s in low_all]) if low_all else 0
        high_ratio = np.median([s['mfe_mae'] for s in high_all]) if high_all else 0
        diff_all = high_ratio - low_ratio

        # A类单独
        low_a_ratio = np.median([s['mfe_mae'] for s in low_a]) if low_a else 0
        high_a_ratio = np.median([s['mfe_mae'] for s in high_a]) if high_a else 0
        diff_a = high_a_ratio - low_a_ratio if low_a and high_a else np.nan

        # B类单独
        low_b_ratio = np.median([s['mfe_mae'] for s in low_b]) if low_b else 0
        high_b_ratio = np.median([s['mfe_mae'] for s in high_b]) if high_b else 0
        diff_b = high_b_ratio - low_b_ratio if low_b and high_b else np.nan

        is_old = sym_key in OLD_SYMBOLS
        sym_results.append({
            'sym': sym_key, 'is_old': is_old, 'n': len(sigs),
            'diff_all': diff_all, 'diff_a': diff_a, 'diff_b': diff_b,
            'high_a_n': len(high_a), 'high_b_n': len(high_b),
        })

    sym_results.sort(key=lambda x: -x['diff_all'])

    print(f"  {'品种':<12} {'标记':>3} {'N':>4} | {'A+B差':>7} {'A类差':>7} {'B类差':>7} | {'高ER:A':>6} {'高ER:B':>6}")
    print(f"  {'-'*70}")

    pos_all = pos_a = pos_b = 0
    total_valid_a = total_valid_b = 0

    for r in sym_results:
        tag = '原' if r['is_old'] else '新'
        da = f"{r['diff_a']:>+6.2f}" if not np.isnan(r['diff_a']) else '   N/A'
        db = f"{r['diff_b']:>+6.2f}" if not np.isnan(r['diff_b']) else '   N/A'
        print(f"  {r['sym']:<12} {tag:>3} {r['n']:>4} | "
              f"{r['diff_all']:>+6.2f} {da} {db}  | "
              f"{r['high_a_n']:>5} {r['high_b_n']:>5}")

        if r['diff_all'] > 0:
            pos_all += 1
        if not np.isnan(r['diff_a']):
            total_valid_a += 1
            if r['diff_a'] > 0:
                pos_a += 1
        if not np.isnan(r['diff_b']):
            total_valid_b += 1
            if r['diff_b'] > 0:
                pos_b += 1

    n_syms = len(sym_results)
    print(f"\n  A+B合并 正向: {pos_all}/{n_syms}")
    print(f"  A类单独 正向: {pos_a}/{total_valid_a}")
    print(f"  B类单独 正向: {pos_b}/{total_valid_b}")

    # ============================================================
    # 6. ER阈值扫描（A类单独、B类单独、A+B合并）
    # ============================================================
    print(f"\n{'='*100}")
    print(f"  ER阈值扫描：A类 vs B类 vs A+B")
    print(f"{'='*100}")

    for sig_label, sig_filter in [('A类', lambda s: s['type'] == 'A'),
                                   ('B类', lambda s: s['type'] == 'B'),
                                   ('A+B', lambda s: True)]:
        filtered = [s for s in all_sigs if sig_filter(s)]
        print(f"\n  --- {sig_label} (N={len(filtered)}) ---")
        print(f"  {'阈值':>6} | {'达标N':>5} {'通过率':>6} | {'达标MFE/MAE':>11} {'未达标':>8} {'差':>7} | {'胜率':>6}")
        print(f"  {'-'*70}")

        for t in ER_THRESHOLDS:
            above = [s for s in filtered if s['er_20'] >= t]
            below = [s for s in filtered if s['er_20'] < t]
            if len(above) < 10:
                continue
            a_ratio = np.median([s['mfe_mae'] for s in above])
            b_ratio = np.median([s['mfe_mae'] for s in below]) if below else 0
            win = sum(1 for s in above if s['mfe_mae'] > 1) / len(above) * 100
            print(f"  {t:>6.2f} | {len(above):>5} {len(above)/len(filtered)*100:>5.1f}% | "
                  f"{a_ratio:>10.2f} {b_ratio:>7.2f} {a_ratio-b_ratio:>+6.2f}  | {win:>5.1f}%")

    # ============================================================
    # 图表
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))

    # 图1: ER三分组 x A/B类 MFE/MAE箱线图
    ax = axes[0, 0]
    box_data = []
    box_labels = []
    box_colors = []
    for sig_type, color in [('A', '#ff9800'), ('B', '#2196f3')]:
        for label, lo, hi in [('低ER', 0, er_p33), ('中ER', er_p33, er_p67), ('高ER', er_p67, 1.1)]:
            group = [s for s in all_sigs if s['type'] == sig_type and lo <= s['er_20'] < hi]
            if group:
                box_data.append([min(s['mfe_mae'], 10) for s in group])
                box_labels.append(f"{sig_type}+{label}\nN={len(group)}")
                box_colors.append(color)

    if box_data:
        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, showmeans=True,
                       meanprops=dict(marker='D', markerfacecolor='white', markersize=4))
        for patch, c in zip(bp['boxes'], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.4)
    ax.set_title('ER三分组 x A/B类\nMFE/MAE分布', fontweight='bold')
    ax.set_ylabel('MFE/MAE')
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.grid(True, alpha=0.3)

    # 图2: 高ER环境中A/B类信号数量比
    ax = axes[0, 1]
    er_bins = np.arange(0, 0.85, 0.05)
    a_counts = []
    b_counts = []
    bin_centers = []
    for j in range(len(er_bins) - 1):
        lo, hi = er_bins[j], er_bins[j+1]
        a_c = sum(1 for s in all_sigs if s['type'] == 'A' and lo <= s['er_20'] < hi)
        b_c = sum(1 for s in all_sigs if s['type'] == 'B' and lo <= s['er_20'] < hi)
        a_counts.append(a_c)
        b_counts.append(b_c)
        bin_centers.append((lo + hi) / 2)

    w = 0.02
    ax.bar([x - w/2 for x in bin_centers], a_counts, w, label='A类', color='#ff9800', alpha=0.7)
    ax.bar([x + w/2 for x in bin_centers], b_counts, w, label='B类', color='#2196f3', alpha=0.7)
    ax.legend()
    ax.set_xlabel('ER(20)')
    ax.set_ylabel('信号数量')
    ax.set_title('不同ER区间的A/B类信号数量', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 图3: ER阈值扫描 - A类vs B类 MFE/MAE
    ax = axes[0, 2]
    for sig_label, sig_filter, color, marker in [
        ('A类', lambda s: s['type'] == 'A', '#ff9800', 'o'),
        ('B类', lambda s: s['type'] == 'B', '#2196f3', 's'),
        ('A+B', lambda s: True, '#4CAF50', 'D'),
    ]:
        filtered = [s for s in all_sigs if sig_filter(s)]
        ts = []
        rs = []
        for t in ER_THRESHOLDS:
            above = [s for s in filtered if s['er_20'] >= t]
            if len(above) >= 10:
                ts.append(t)
                rs.append(np.median([s['mfe_mae'] for s in above]))
        if ts:
            ax.plot(ts, rs, f'{marker}-', color=color, label=sig_label, linewidth=2)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.3)
    ax.legend()
    ax.set_xlabel('ER(20) 阈值')
    ax.set_ylabel('MFE/MAE 中位数')
    ax.set_title('ER阈值扫描: A vs B vs A+B', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 图4: 前60天vs后60天验证
    ax = axes[1, 0]
    for period_name, period_sigs, ls in [('前60天', all_first, '-'), ('后60天', all_last, '--')]:
        for sig_label, sig_filter, color in [
            ('A类', lambda s: s['type'] == 'A', '#ff9800'),
            ('B类', lambda s: s['type'] == 'B', '#2196f3'),
        ]:
            filtered = [s for s in period_sigs if sig_filter(s)]
            ts = []
            rs = []
            for t in ER_THRESHOLDS:
                above = [s for s in filtered if s['er_20'] >= t]
                if len(above) >= 10:
                    ts.append(t)
                    rs.append(np.median([s['mfe_mae'] for s in above]))
            if ts:
                ax.plot(ts, rs, f'o{ls}', color=color,
                       label=f'{sig_label}-{period_name}', linewidth=1.5)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xlabel('ER(20) 阈值')
    ax.set_ylabel('MFE/MAE 中位数')
    ax.set_title('前后60天验证: A/B分别', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 图5: 分品种差值 A+B
    ax = axes[1, 1]
    syms_plot = [r['sym'] for r in sym_results]
    diffs_plot = [r['diff_all'] for r in sym_results]
    colors_bar = ['#2196f3' if r['is_old'] else '#ff9800' for r in sym_results]
    ax.barh(syms_plot[::-1], diffs_plot[::-1], color=colors_bar[::-1], alpha=0.7)
    ax.axvline(x=0, color='red', linewidth=0.8)
    ax.set_title(f'分品种: ER高-低 MFE/MAE差 (A+B)\n正向{pos_all}/{n_syms}', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 图6: 分品种 A类差值 vs B类差值
    ax = axes[1, 2]
    for r in sym_results:
        if np.isnan(r['diff_a']) or np.isnan(r['diff_b']):
            continue
        color = '#2196f3' if r['is_old'] else '#ff9800'
        ax.scatter(r['diff_b'], r['diff_a'], color=color, s=40, alpha=0.7)
        ax.annotate(r['sym'].split('_')[-1], (r['diff_b'], r['diff_a']),
                   fontsize=6, alpha=0.7)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.3)
    ax.set_xlabel('B类: ER高-低差')
    ax.set_ylabel('A类: ER高-低差')
    ax.set_title('A类 vs B类 ER区分力散点', fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'er_ab_validate.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图表: {out_path}")


if __name__ == '__main__':
    main()
