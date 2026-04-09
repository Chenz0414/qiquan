# -*- coding: utf-8 -*-
"""
因子归因测试：60MA的特殊性 vs 趋势/震荡的特殊性
================================================
问题：MFE的区分能力到底来自"距60MA远近"还是"趋势vs震荡"？

实验：
1. 测多个MA周期距离（20/40/60/80/100MA）→ 都有效说明不是60特殊
2. 测效率比ER（完全不同的趋势度量）→ 也有效说明是市场状态在起作用
3. 测ADX → 另一个独立的趋势强度指标
4. 看这些指标之间的相关性 → 如果高度相关说明测的是同一个东西

直接读缓存数据。
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
MFE_WINDOW = 48  # 8小时
BURST_THRESHOLD = 3.0


def load_cached(symbol_key):
    path = os.path.join(CACHE_DIR, f"{symbol_key}_10min_170d.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def prepare_data(df):
    """计算所有候选因子"""
    if 'ma_fast' not in df.columns:
        df['ma_fast'] = df['close'].rolling(MA_FAST).mean()
    if 'ma_slow' not in df.columns:
        df['ma_slow'] = df['close'].rolling(MA_SLOW).mean()

    # 多周期MA距离
    for period in [20, 40, 60, 80, 100]:
        ma_col = f'ma_{period}'
        dist_col = f'dist_ma{period}_pct'
        df[ma_col] = df['close'].rolling(period).mean()
        df[dist_col] = ((df['close'] - df[ma_col]) / df[ma_col] * 100).abs()

    # 效率比 (Kaufman Efficiency Ratio)
    # ER = abs(net_change) / sum(abs(bar_to_bar_changes)) over N bars
    # 接近1 = 干净趋势，接近0 = 来回震荡
    for er_period in [20, 40, 60]:
        net_change = (df['close'] - df['close'].shift(er_period)).abs()
        bar_changes = df['close'].diff().abs()
        sum_changes = bar_changes.rolling(er_period).sum()
        df[f'er_{er_period}'] = net_change / sum_changes.replace(0, np.nan)

    # ADX (Average Directional Index)
    # 简化版：用14周期
    adx_period = 14
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(adx_period).mean()
    plus_di = 100 * (plus_dm.rolling(adx_period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(adx_period).mean() / atr.replace(0, np.nan))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    df['adx'] = dx.rolling(adx_period).mean()

    # 线性回归R² (过去30根K线拟合直线的R²)
    r2_period = 30
    r2_values = []
    closes = df['close'].values
    for i in range(len(df)):
        if i < r2_period:
            r2_values.append(np.nan)
            continue
        y = closes[i - r2_period + 1: i + 1]
        x = np.arange(r2_period)
        if np.std(y) == 0:
            r2_values.append(0)
            continue
        corr = np.corrcoef(x, y)[0, 1]
        r2_values.append(corr ** 2)
    df['r2_30'] = r2_values

    # 趋势方向
    df['trend'] = 0
    df.loc[df['ma_fast'] > df['ma_slow'], 'trend'] = 1
    df.loc[df['ma_fast'] < df['ma_slow'], 'trend'] = -1

    return df


def detect_all_signals(df):
    """检测所有B类信号（不限回调根数），记录所有因子值"""
    signals = []
    n = len(df)
    warmup = 105  # 100MA + 5

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

        if trend != prev_trend and trend != 0:
            b_below_start = -1
            b_pullback_low = None
            b_pullback_high = None
        prev_trend = trend

        if trend == 0:
            continue
        if i == warmup:
            continue

        prev = df.iloc[i - 1]
        prev_close = prev['close']
        prev_ma_f = prev['ma_fast']
        if pd.isna(prev_ma_f):
            continue

        sig_base = None

        # 多头B类
        if trend == 1:
            if b_below_start == -1:
                if close < ma_f and prev_close >= prev_ma_f:
                    b_below_start = i
                    b_pullback_low = low
            else:
                b_pullback_low = min(b_pullback_low, low)
                if close > ma_f:
                    sig_base = {'direction': 'long', 'idx': i, 'time': row['datetime'],
                                'entry_price': close}
                    b_below_start = -1
                    b_pullback_low = None

        # 空头B类
        elif trend == -1:
            if b_below_start == -1:
                if close > ma_f and prev_close <= prev_ma_f:
                    b_below_start = i
                    b_pullback_high = high
            else:
                b_pullback_high = max(b_pullback_high, high)
                if close < ma_f:
                    sig_base = {'direction': 'short', 'idx': i, 'time': row['datetime'],
                                'entry_price': close}
                    b_below_start = -1
                    b_pullback_high = None

        if sig_base is not None:
            # 记录所有因子
            for period in [20, 40, 60, 80, 100]:
                sig_base[f'dist_ma{period}'] = row[f'dist_ma{period}_pct'] if not pd.isna(row[f'dist_ma{period}_pct']) else np.nan
            for er_p in [20, 40, 60]:
                sig_base[f'er_{er_p}'] = row[f'er_{er_p}'] if not pd.isna(row[f'er_{er_p}']) else np.nan
            sig_base['adx'] = row['adx'] if not pd.isna(row['adx']) else np.nan
            sig_base['r2_30'] = row['r2_30'] if not pd.isna(row['r2_30']) else np.nan

            # 计算MFE
            end_idx = min(i + MFE_WINDOW, len(df) - 1)
            if i + 1 <= end_idx:
                segment = df.iloc[i + 1: end_idx + 1]
                if sig_base['direction'] == 'short':
                    mfe = (close - segment['low'].min()) / close * 100
                else:
                    mfe = (segment['high'].max() - close) / close * 100
                sig_base['mfe_8h'] = max(mfe, 0)
            else:
                sig_base['mfe_8h'] = 0

            sig_base['is_burst'] = sig_base['mfe_8h'] >= BURST_THRESHOLD
            signals.append(sig_base)

    return signals


def evaluate_factor(signals, factor_name):
    """评估单个因子的区分能力：按P50切分，对比爆发率"""
    values = [s[factor_name] for s in signals if not np.isnan(s[factor_name])]
    valid_signals = [s for s in signals if not np.isnan(s[factor_name])]

    if len(values) < 20:
        return None

    p50 = np.percentile(values, 50)

    low_group = [s for s in valid_signals if s[factor_name] <= p50]
    high_group = [s for s in valid_signals if s[factor_name] > p50]

    if not low_group or not high_group:
        return None

    low_burst = sum(1 for s in low_group if s['is_burst']) / len(low_group) * 100
    high_burst = sum(1 for s in high_group if s['is_burst']) / len(high_group) * 100

    low_mfe = np.mean([s['mfe_8h'] for s in low_group])
    high_mfe = np.mean([s['mfe_8h'] for s in high_group])

    return {
        'factor': factor_name,
        'p50': p50,
        'low_n': len(low_group),
        'high_n': len(high_group),
        'low_burst': low_burst,
        'high_burst': high_burst,
        'burst_diff': high_burst - low_burst,
        'low_mfe': low_mfe,
        'high_mfe': high_mfe,
        'mfe_ratio': high_mfe / low_mfe if low_mfe > 0 else 0,
    }


def main():
    print("=" * 90)
    print("  因子归因测试：60MA特殊 还是 趋势/震荡特殊？")
    print("=" * 90)

    all_signals = []

    for sym_key, sym_name in SYMBOLS.items():
        df = load_cached(sym_key)
        if df is None:
            continue

        df = prepare_data(df)
        cutoff = df['datetime'].iloc[-1] - timedelta(days=120)
        df_120 = df[df['datetime'] >= cutoff].copy().reset_index(drop=True)

        signals = detect_all_signals(df_120)
        for s in signals:
            s['symbol'] = sym_name
        all_signals.extend(signals)
        print(f"  {sym_name}: {len(signals)} signals")

    print(f"\n  Total signals: {len(all_signals)}")

    # ============================================================
    # 1. 各因子区分能力对比
    # ============================================================
    factors = [
        'dist_ma20', 'dist_ma40', 'dist_ma60', 'dist_ma80', 'dist_ma100',
        'er_20', 'er_40', 'er_60',
        'adx', 'r2_30',
    ]

    factor_labels = {
        'dist_ma20': '距20MA%', 'dist_ma40': '距40MA%', 'dist_ma60': '距60MA%',
        'dist_ma80': '距80MA%', 'dist_ma100': '距100MA%',
        'er_20': 'ER(20)', 'er_40': 'ER(40)', 'er_60': 'ER(60)',
        'adx': 'ADX(14)', 'r2_30': 'R2(30)',
    }

    print(f"\n{'='*90}")
    print(f"  全品种汇总：各因子按P50切分后的爆发率差异")
    print(f"{'='*90}")
    header = f"  {'因子':<12} {'P50':>8} | {'低组N':>5} {'爆发率':>7} {'MFE_8h':>8} | {'高组N':>5} {'爆发率':>7} {'MFE_8h':>8} | {'爆发率差':>8} {'MFE比':>6}"
    print(header)
    print(f"  {'-'*95}")

    results = []
    for f in factors:
        r = evaluate_factor(all_signals, f)
        if r:
            results.append(r)
            label = factor_labels.get(f, f)
            print(f"  {label:<12} {r['p50']:>8.3f} | "
                  f"{r['low_n']:>5} {r['low_burst']:>6.1f}% {r['low_mfe']:>7.2f}% | "
                  f"{r['high_n']:>5} {r['high_burst']:>6.1f}% {r['high_mfe']:>7.2f}% | "
                  f"{r['burst_diff']:>+7.1f}% {r['mfe_ratio']:>5.2f}x")

    # ============================================================
    # 2. 分品种稳健性检查
    # ============================================================
    print(f"\n{'='*90}")
    print(f"  分品种稳健性：各因子的爆发率差(高组-低组)，正数=因子有效")
    print(f"{'='*90}")

    sym_names = list(SYMBOLS.values())
    header = f"  {'因子':<12}"
    for name in sym_names:
        header += f" {name:>6}"
    header += " | 正向数"
    print(header)
    print(f"  {'-'*75}")

    for f in factors:
        label = factor_labels.get(f, f)
        line = f"  {label:<12}"
        positive_count = 0
        for sym_name in sym_names:
            sym_sigs = [s for s in all_signals if s['symbol'] == sym_name]
            r = evaluate_factor(sym_sigs, f)
            if r:
                line += f" {r['burst_diff']:>+5.1f}%"
                if r['burst_diff'] > 0:
                    positive_count += 1
            else:
                line += f"   {'N/A':>4}"
        line += f" |   {positive_count}/6"
        print(line)

    # ============================================================
    # 3. 因子间相关性
    # ============================================================
    print(f"\n{'='*90}")
    print(f"  因子间相关性（Pearson）")
    print(f"{'='*90}")

    valid = [s for s in all_signals if all(not np.isnan(s[f]) for f in factors)]
    if len(valid) > 50:
        df_corr = pd.DataFrame({f: [s[f] for s in valid] for f in factors})
        corr = df_corr.corr()

        # 打印MA距离之间的相关性
        print(f"\n  MA距离因子之间:")
        ma_factors = ['dist_ma20', 'dist_ma40', 'dist_ma60', 'dist_ma80', 'dist_ma100']
        for i, f1 in enumerate(ma_factors):
            for f2 in ma_factors[i+1:]:
                print(f"    {factor_labels[f1]} vs {factor_labels[f2]}: {corr.loc[f1, f2]:.3f}")

        # MA距离 vs 其他因子
        print(f"\n  距60MA vs 其他因子:")
        other_factors = ['er_20', 'er_40', 'er_60', 'adx', 'r2_30']
        for f in other_factors:
            print(f"    距60MA% vs {factor_labels[f]}: {corr.loc['dist_ma60', f]:.3f}")

        # ER vs ADX vs R2
        print(f"\n  趋势指标之间:")
        trend_factors = ['er_20', 'er_40', 'er_60', 'adx', 'r2_30']
        for i, f1 in enumerate(trend_factors):
            for f2 in trend_factors[i+1:]:
                print(f"    {factor_labels[f1]} vs {factor_labels[f2]}: {corr.loc[f1, f2]:.3f}")

    # ============================================================
    # 4. 关键问题：ER高+距60MA近 vs ER低+距60MA远
    # ============================================================
    print(f"\n{'='*90}")
    print(f"  交叉验证：ER和距60MA是不是测同一个东西？")
    print(f"{'='*90}")

    # 用ER_40和dist_ma60
    er_col = 'er_40'
    dist_col = 'dist_ma60'
    valid_sigs = [s for s in all_signals if not np.isnan(s[er_col]) and not np.isnan(s[dist_col])]

    if len(valid_sigs) > 50:
        er_p50 = np.percentile([s[er_col] for s in valid_sigs], 50)
        dist_p50 = np.percentile([s[dist_col] for s in valid_sigs], 50)

        groups = {
            'ER低+60MA近': [s for s in valid_sigs if s[er_col] <= er_p50 and s[dist_col] <= dist_p50],
            'ER低+60MA远': [s for s in valid_sigs if s[er_col] <= er_p50 and s[dist_col] > dist_p50],
            'ER高+60MA近': [s for s in valid_sigs if s[er_col] > er_p50 and s[dist_col] <= dist_p50],
            'ER高+60MA远': [s for s in valid_sigs if s[er_col] > er_p50 and s[dist_col] > dist_p50],
        }

        print(f"  ER(40) P50 = {er_p50:.3f}, 距60MA P50 = {dist_p50:.3f}%")
        print(f"\n  {'组合':<16} {'N':>5} | {'爆发率':>7} {'MFE_8h':>8} {'中位数':>8}")
        print(f"  {'-'*55}")

        for name, sigs in groups.items():
            if sigs:
                burst = sum(1 for s in sigs if s['is_burst']) / len(sigs) * 100
                mfe = np.mean([s['mfe_8h'] for s in sigs])
                med = np.median([s['mfe_8h'] for s in sigs])
                print(f"  {name:<16} {len(sigs):>5} | {burst:>6.1f}% {mfe:>7.2f}% {med:>7.2f}%")

    # ============================================================
    # 图表
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 图1: 各因子区分能力（爆发率差）
    ax = axes[0, 0]
    if results:
        names = [factor_labels.get(r['factor'], r['factor']) for r in results]
        diffs = [r['burst_diff'] for r in results]
        colors = ['#4CAF50' if d > 0 else '#f44336' for d in diffs]
        bars = ax.barh(names, diffs, color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linewidth=0.5)
        for bar, val in zip(bars, diffs):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{val:+.1f}%', va='center', fontsize=9)
    ax.set_title('各因子区分能力（高组-低组爆发率差）', fontweight='bold')
    ax.set_xlabel('爆发率差 (%)')
    ax.grid(True, alpha=0.3)

    # 图2: 交叉验证 2x2
    ax = axes[0, 1]
    if groups:
        group_names = list(groups.keys())
        burst_rates = []
        for name in group_names:
            sigs = groups[name]
            if sigs:
                burst_rates.append(sum(1 for s in sigs if s['is_burst']) / len(sigs) * 100)
            else:
                burst_rates.append(0)
        colors = ['#f44336', '#ff9800', '#2196f3', '#4CAF50']
        bars = ax.bar(group_names, burst_rates, color=colors, alpha=0.7)
        for bar, val in zip(bars, burst_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', fontweight='bold')
    ax.set_title('ER(40) x 距60MA 交叉验证', fontweight='bold')
    ax.set_ylabel('爆发率 (%)')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(True, alpha=0.3)

    # 图3: MFE比（高组/低组MFE）
    ax = axes[1, 0]
    if results:
        names = [factor_labels.get(r['factor'], r['factor']) for r in results]
        ratios = [r['mfe_ratio'] for r in results]
        colors = ['#4CAF50' if r > 1 else '#f44336' for r in ratios]
        bars = ax.barh(names, ratios, color=colors, alpha=0.7)
        ax.axvline(x=1, color='black', linewidth=0.5, linestyle='--')
        for bar, val in zip(bars, ratios):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}x', va='center', fontsize=9)
    ax.set_title('各因子 MFE比（高组/低组）', fontweight='bold')
    ax.set_xlabel('MFE比')
    ax.grid(True, alpha=0.3)

    # 图4: 分品种稳健性热力图
    ax = axes[1, 1]
    heat_data = []
    heat_factors = []
    for f in factors:
        row = []
        for sym_name in sym_names:
            sym_sigs = [s for s in all_signals if s['symbol'] == sym_name]
            r = evaluate_factor(sym_sigs, f)
            row.append(r['burst_diff'] if r else 0)
        heat_data.append(row)
        heat_factors.append(factor_labels.get(f, f))

    if heat_data:
        heat_arr = np.array(heat_data)
        im = ax.imshow(heat_arr, cmap='RdYlGn', aspect='auto',
                       vmin=-20, vmax=20)
        ax.set_xticks(range(len(sym_names)))
        ax.set_xticklabels(sym_names)
        ax.set_yticks(range(len(heat_factors)))
        ax.set_yticklabels(heat_factors)
        for i in range(len(heat_factors)):
            for j in range(len(sym_names)):
                ax.text(j, i, f'{heat_data[i][j]:+.0f}', ha='center', va='center', fontsize=8)
        plt.colorbar(im, ax=ax, label='爆发率差%')
    ax.set_title('分品种稳健性（绿=因子有效）', fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'factor_attribution.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图表已保存: {out_path}")


if __name__ == '__main__':
    main()
