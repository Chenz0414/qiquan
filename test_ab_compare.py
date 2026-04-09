# -*- coding: utf-8 -*-
"""
A类 vs B类回调 MFE对比
=====================
A类: 影线碰10MA但收盘未破（low <= ma_fast but close > ma_fast）
B类: 收盘跌破10MA后重新站回（现有策略）

按距60MA远近拆分，看两种信号在不同市场环境下的表现差异。
直接读缓存数据，不连天勤。
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

# 品种列表（用缓存中有的）
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
MA_60 = 60
MFE_WINDOWS = [6, 12, 24, 48]  # 1h/2h/4h/8h (10min K线)
BURST_THRESHOLD = 3.0  # MFE_8h > 3% 算爆发


def load_cached(symbol_key):
    """从缓存读取数据"""
    path = os.path.join(CACHE_DIR, f"{symbol_key}_10min_170d.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    return df


def prepare_data(df):
    """计算所需指标"""
    # 确保有ma_fast和ma_slow
    if 'ma_fast' not in df.columns:
        df['ma_fast'] = df['close'].rolling(MA_FAST).mean()
    if 'ma_slow' not in df.columns:
        df['ma_slow'] = df['close'].rolling(MA_SLOW).mean()

    # 60MA
    df['ma_60'] = df['close'].rolling(MA_60).mean()

    # 距60MA距离（%）
    df['dist_60ma_pct'] = ((df['close'] - df['ma_60']) / df['ma_60'] * 100).abs()

    # 趋势方向
    df['trend'] = 0
    df.loc[df['ma_fast'] > df['ma_slow'], 'trend'] = 1
    df.loc[df['ma_fast'] < df['ma_slow'], 'trend'] = -1

    return df


def detect_signals_ab(df):
    """
    同时检测A类和B类信号。

    A类（多头）: 趋势中 low <= ma_fast 但 close > ma_fast（影线碰MA弹回）
    B类（多头）: 收盘跌破ma_fast，之后某根K线收盘站回ma_fast

    空头信号镜像对称。
    """
    signals = []
    n = len(df)
    warmup = MA_60 + 5

    # B类状态追踪
    b_below_start = -1  # 跌破开始的index
    b_pullback_low = None  # 多头回调最低
    b_pullback_high = None  # 空头回调最高
    prev_trend = 0

    for i in range(warmup, n):
        row = df.iloc[i]
        if pd.isna(row['ma_fast']) or pd.isna(row['ma_slow']) or pd.isna(row['ma_60']):
            continue

        close = row['close']
        high = row['high']
        low = row['low']
        ma_f = row['ma_fast']
        ma_s = row['ma_slow']
        trend = row['trend']

        # 趋势翻转，重置B类状态
        if trend != prev_trend and trend != 0:
            b_below_start = -1
            b_pullback_low = None
            b_pullback_high = None
        prev_trend = trend

        if trend == 0:
            continue

        # 前一根
        if i == warmup:
            continue
        prev = df.iloc[i - 1]
        prev_close = prev['close']
        prev_ma_f = prev['ma_fast']

        if pd.isna(prev_ma_f):
            continue

        # ====== 多头 ======
        if trend == 1:
            # --- A类检测 ---
            # 当前K线影线碰到或穿过ma_fast，但收盘在ma_fast上方
            # 且前一根收盘也在ma_fast上方（排除刚从B类回调中恢复的情况）
            if low <= ma_f and close > ma_f and prev_close > prev_ma_f:
                signals.append({
                    'type': 'A',
                    'idx': i,
                    'time': row['datetime'],
                    'direction': 'long',
                    'entry_price': close,
                    'dist_60ma_pct': row['dist_60ma_pct'],
                })

            # --- B类检测 ---
            if b_below_start == -1:
                # 未在回调中：检查是否开始跌破
                if close < ma_f and prev_close >= prev_ma_f:
                    b_below_start = i
                    b_pullback_low = low
            else:
                # 在回调中
                b_pullback_low = min(b_pullback_low, low)
                if close > ma_f:
                    # 收回：记录B类信号
                    pb_bars = i - b_below_start
                    signals.append({
                        'type': 'B',
                        'idx': i,
                        'time': row['datetime'],
                        'direction': 'long',
                        'entry_price': close,
                        'pullback_bars': pb_bars,
                        'dist_60ma_pct': row['dist_60ma_pct'],
                    })
                    b_below_start = -1
                    b_pullback_low = None

        # ====== 空头 ======
        elif trend == -1:
            # --- A类检测 ---
            if high >= ma_f and close < ma_f and prev_close < prev_ma_f:
                signals.append({
                    'type': 'A',
                    'idx': i,
                    'time': row['datetime'],
                    'direction': 'short',
                    'entry_price': close,
                    'dist_60ma_pct': row['dist_60ma_pct'],
                })

            # --- B类检测 ---
            if b_below_start == -1:
                if close > ma_f and prev_close <= prev_ma_f:
                    b_below_start = i
                    b_pullback_high = high
            else:
                b_pullback_high = max(b_pullback_high, high)
                if close < ma_f:
                    pb_bars = i - b_below_start
                    signals.append({
                        'type': 'B',
                        'idx': i,
                        'time': row['datetime'],
                        'direction': 'short',
                        'entry_price': close,
                        'pullback_bars': pb_bars,
                        'dist_60ma_pct': row['dist_60ma_pct'],
                    })
                    b_below_start = -1
                    b_pullback_high = None

    return signals


def calc_mfe(df, sig_idx, direction, windows):
    """计算信号后各窗口的MFE"""
    entry_price = df.iloc[sig_idx]['close']
    results = {}
    for w in windows:
        end_idx = min(sig_idx + w, len(df) - 1)
        if sig_idx + 1 > end_idx:
            results[f'mfe_{w}'] = 0
            continue
        segment = df.iloc[sig_idx + 1: end_idx + 1]
        if direction == 'short':
            mfe = (entry_price - segment['low'].min()) / entry_price * 100
        else:
            mfe = (segment['high'].max() - entry_price) / entry_price * 100
        results[f'mfe_{w}'] = max(mfe, 0)
    return results


def analyze_group(signals, label):
    """统计一组信号的MFE"""
    n = len(signals)
    if n == 0:
        return None

    mfe_48 = [s['mfe_48'] for s in signals]
    burst_count = sum(1 for m in mfe_48 if m >= BURST_THRESHOLD)

    return {
        'label': label,
        'n': n,
        'mfe_1h': np.mean([s['mfe_6'] for s in signals]),
        'mfe_2h': np.mean([s['mfe_12'] for s in signals]),
        'mfe_4h': np.mean([s['mfe_24'] for s in signals]),
        'mfe_8h': np.mean(mfe_48),
        'mfe_8h_median': np.median(mfe_48),
        'burst_count': burst_count,
        'burst_rate': burst_count / n * 100,
    }


def main():
    print("=" * 80)
    print("  A类 vs B类回调 MFE对比（按距60MA远近拆分）")
    print("  A类: 影线碰10MA但收盘未破")
    print("  B类: 收盘跌破10MA后重新站回")
    print("=" * 80)

    # 收集所有品种数据，用于汇总
    all_signals_by_type = {'A': [], 'B': []}
    symbol_results = {}

    for sym_key, sym_name in SYMBOLS.items():
        df = load_cached(sym_key)
        if df is None:
            print(f"\n  {sym_name}: 无缓存数据，跳过")
            continue

        df = prepare_data(df)

        # 取最近120天
        cutoff = df['datetime'].iloc[-1] - timedelta(days=120)
        df_120 = df[df['datetime'] >= cutoff].copy().reset_index(drop=True)

        # 检测信号
        signals = detect_signals_ab(df_120)

        # 计算MFE
        for sig in signals:
            mfe = calc_mfe(df_120, sig['idx'], sig['direction'], MFE_WINDOWS)
            sig.update(mfe)

        # 按类型拆分
        a_sigs = [s for s in signals if s['type'] == 'A']
        b_sigs = [s for s in signals if s['type'] == 'B']

        # 距60MA分位数（用全部信号算）
        all_dist = [s['dist_60ma_pct'] for s in signals]
        if len(all_dist) < 10:
            print(f"\n  {sym_name}: 信号太少({len(all_dist)}个)，跳过")
            continue

        p50 = np.percentile(all_dist, 50)

        print(f"\n{'='*70}")
        print(f"  {sym_name} | A类={len(a_sigs)}个, B类={len(b_sigs)}个 | 60MA距离P50={p50:.2f}%")
        print(f"{'='*70}")

        # 4组拆分
        groups = {
            'A_near': [s for s in a_sigs if s['dist_60ma_pct'] <= p50],
            'A_far':  [s for s in a_sigs if s['dist_60ma_pct'] > p50],
            'B_near': [s for s in b_sigs if s['dist_60ma_pct'] <= p50],
            'B_far':  [s for s in b_sigs if s['dist_60ma_pct'] > p50],
        }

        # 同时输出A/B不分远近的整体
        groups['A_all'] = a_sigs
        groups['B_all'] = b_sigs

        labels = {
            'A_all':  'A类(全部)',
            'B_all':  'B类(全部)',
            'A_near': 'A类+近60MA',
            'A_far':  'A类+远60MA',
            'B_near': 'B类+近60MA',
            'B_far':  'B类+远60MA',
        }

        thresh = BURST_THRESHOLD
        print(f"\n  {'分组':<14} {'N':>4} | {'MFE_1h':>7} {'MFE_2h':>7} {'MFE_4h':>7} {'MFE_8h':>7} {'中位数':>7} | 爆发率({thresh}%+)")
        print(f"  {'-'*80}")

        sym_groups = {}
        for key in ['A_all', 'B_all', 'A_near', 'A_far', 'B_near', 'B_far']:
            result = analyze_group(groups[key], labels[key])
            sym_groups[key] = result
            if result:
                print(f"  {result['label']:<14} {result['n']:>4} | "
                      f"{result['mfe_1h']:>6.2f}% {result['mfe_2h']:>6.2f}% "
                      f"{result['mfe_4h']:>6.2f}% {result['mfe_8h']:>6.2f}% "
                      f"{result['mfe_8h_median']:>6.2f}% | "
                      f"{result['burst_rate']:>5.1f}% ({result['burst_count']}/{result['n']})")
            else:
                print(f"  {labels[key]:<14}    0 | {'-- 无信号 --'}")

        symbol_results[sym_key] = {'name': sym_name, 'groups': sym_groups, 'p50': p50}

        # 汇总
        for s in a_sigs:
            s['symbol'] = sym_name
        for s in b_sigs:
            s['symbol'] = sym_name
        all_signals_by_type['A'].extend(a_sigs)
        all_signals_by_type['B'].extend(b_sigs)

    # ============================================================
    # 全品种汇总
    # ============================================================
    print(f"\n\n{'='*80}")
    print(f"  全品种汇总")
    print(f"{'='*80}")

    all_sigs = all_signals_by_type['A'] + all_signals_by_type['B']
    if not all_sigs:
        print("  无数据")
        return

    global_p50 = np.percentile([s['dist_60ma_pct'] for s in all_sigs], 50)
    print(f"  全局距60MA P50 = {global_p50:.2f}%")
    print(f"  A类总计: {len(all_signals_by_type['A'])}个")
    print(f"  B类总计: {len(all_signals_by_type['B'])}个")

    all_groups = {
        'A_all':  all_signals_by_type['A'],
        'B_all':  all_signals_by_type['B'],
        'A_near': [s for s in all_signals_by_type['A'] if s['dist_60ma_pct'] <= global_p50],
        'A_far':  [s for s in all_signals_by_type['A'] if s['dist_60ma_pct'] > global_p50],
        'B_near': [s for s in all_signals_by_type['B'] if s['dist_60ma_pct'] <= global_p50],
        'B_far':  [s for s in all_signals_by_type['B'] if s['dist_60ma_pct'] > global_p50],
    }

    labels = {
        'A_all':  'A类(全部)',
        'B_all':  'B类(全部)',
        'A_near': 'A类+近60MA',
        'A_far':  'A类+远60MA',
        'B_near': 'B类+近60MA',
        'B_far':  'B类+远60MA',
    }

    print(f"\n  {'分组':<14} {'N':>5} | {'MFE_1h':>7} {'MFE_2h':>7} {'MFE_4h':>7} {'MFE_8h':>7} {'中位数':>7} | {'爆发率':>6}")
    print(f"  {'-'*80}")

    for key in ['A_all', 'B_all', '', 'A_near', 'A_far', '', 'B_near', 'B_far']:
        if key == '':
            print(f"  {'-'*80}")
            continue
        result = analyze_group(all_groups[key], labels[key])
        if result:
            print(f"  {result['label']:<14} {result['n']:>5} | "
                  f"{result['mfe_1h']:>6.2f}% {result['mfe_2h']:>6.2f}% "
                  f"{result['mfe_4h']:>6.2f}% {result['mfe_8h']:>6.2f}% "
                  f"{result['mfe_8h_median']:>6.2f}% | "
                  f"{result['burst_rate']:>5.1f}%")

    # ============================================================
    # B类按回调根数进一步拆分（看>=4根过滤的效果）
    # ============================================================
    print(f"\n\n{'='*80}")
    print(f"  B类回调根数拆分（看>=4根过滤效果）")
    print(f"{'='*80}")

    b_all = all_signals_by_type['B']
    b_short = [s for s in b_all if s.get('pullback_bars', 0) < 4]
    b_long = [s for s in b_all if s.get('pullback_bars', 0) >= 4]

    b_groups = {
        'B_pb<4':        b_short,
        'B_pb>=4':       b_long,
        'B_pb>=4+远60MA': [s for s in b_long if s['dist_60ma_pct'] > global_p50],
    }

    print(f"\n  {'分组':<18} {'N':>5} | {'MFE_8h':>7} {'中位数':>7} | {'爆发率':>6}")
    print(f"  {'-'*60}")

    for key, sigs in b_groups.items():
        result = analyze_group(sigs, key)
        if result:
            print(f"  {key:<18} {result['n']:>5} | "
                  f"{result['mfe_8h']:>6.2f}% {result['mfe_8h_median']:>6.2f}% | "
                  f"{result['burst_rate']:>5.1f}%")

    # ============================================================
    # 图表
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 图1: 全品种 A vs B MFE_8h 箱线图
    ax = axes[0, 0]
    box_data = []
    box_labels = []
    for key in ['A_near', 'A_far', 'B_near', 'B_far']:
        sigs = all_groups[key]
        if sigs:
            box_data.append([s['mfe_48'] for s in sigs])
            box_labels.append(f"{labels[key]}\n(N={len(sigs)})")
    if box_data:
        colors = ['#ff9800', '#f44336', '#2196f3', '#4CAF50']
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='white', markersize=6))
        for patch, c in zip(bp['boxes'], colors[:len(box_data)]):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
    ax.set_title('全品种 MFE_8h 分布', fontweight='bold')
    ax.set_ylabel('MFE_8h (%)')
    ax.grid(True, alpha=0.3)

    # 图2: 各品种爆发率对比 A_far vs B_far
    ax = axes[0, 1]
    sym_names = []
    a_far_rates = []
    b_far_rates = []
    for sym_key, data in symbol_results.items():
        g = data['groups']
        if g.get('A_far') and g.get('B_far'):
            sym_names.append(data['name'])
            a_far_rates.append(g['A_far']['burst_rate'])
            b_far_rates.append(g['B_far']['burst_rate'])

    if sym_names:
        x = np.arange(len(sym_names))
        w = 0.35
        ax.bar(x - w/2, a_far_rates, w, label='A类+远60MA', color='#f44336', alpha=0.7)
        ax.bar(x + w/2, b_far_rates, w, label='B类+远60MA', color='#4CAF50', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(sym_names)
        ax.legend()
    ax.set_title('远离60MA时：A类 vs B类 爆发率', fontweight='bold')
    ax.set_ylabel('爆发率 (%)')
    ax.grid(True, alpha=0.3)

    # 图3: MFE_8h均值对比 (4组)
    ax = axes[1, 0]
    group_keys = ['A_near', 'A_far', 'B_near', 'B_far']
    mfe_vals = []
    mfe_labels = []
    for key in group_keys:
        r = analyze_group(all_groups[key], key)
        if r:
            mfe_vals.append(r['mfe_8h'])
            mfe_labels.append(labels[key])
    if mfe_vals:
        colors = ['#ff9800', '#f44336', '#2196f3', '#4CAF50']
        bars = ax.bar(mfe_labels, mfe_vals, color=colors[:len(mfe_vals)], alpha=0.7)
        for bar, val in zip(bars, mfe_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{val:.2f}%', ha='center', fontweight='bold')
    ax.set_title('全品种 MFE_8h 均值', fontweight='bold')
    ax.set_ylabel('MFE_8h (%)')
    ax.grid(True, alpha=0.3)

    # 图4: 信号数量分布
    ax = axes[1, 1]
    for key in ['A_near', 'A_far', 'B_near', 'B_far']:
        sigs = all_groups[key]
        if sigs:
            mfe_48 = [s['mfe_48'] for s in sigs]
            ax.hist(mfe_48, bins=30, alpha=0.4, label=f"{labels[key]} (N={len(sigs)})")
    ax.axvline(x=BURST_THRESHOLD, color='red', linestyle='--', label=f'爆发阈值({BURST_THRESHOLD}%)')
    ax.legend(fontsize=8)
    ax.set_title('MFE_8h 分布直方图', fontweight='bold')
    ax.set_xlabel('MFE_8h (%)')
    ax.set_ylabel('频次')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'ab_compare.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图表已保存: {out_path}")


if __name__ == '__main__':
    main()
