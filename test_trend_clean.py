# -*- coding: utf-8 -*-
"""
趋势干净度指标对比
==================
核心问题：什么指标能区分"生猪型干净趋势"和"碳酸锂型震荡"？

候选指标：
1. new_extreme: 最近20根里有多少根在创50根新高/新低（顺趋势方向）
2. range_breakout: 最近10根的极值超出了前40根范围多少
3. range_position: 收盘价在50根范围里的位置（0~1，越靠边越趋势）
4. dir_consistency: 最近20根里有多少比例收盘在趋势方向
5. no_overlap: 最近10根的范围和前40根范围不重叠的比例
6. er_20: 效率比（对照组）

用MFE/MAE作为目标，找能区分走势干净程度的指标。
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

    # ER(20) 对照
    net = (df['close'] - df['close'].shift(20)).abs()
    bar_sum = df['close'].diff().abs().rolling(20).sum()
    df['er_20'] = net / bar_sum.replace(0, np.nan)

    return df


def calc_indicators_at(df, i, trend_dir):
    """在第i根K线处计算所有候选指标"""
    if i < 55:
        return None

    close = df.iloc[i]['close']
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    # --- 指标1: new_extreme ---
    # 最近20根里，有多少根在创50根新高或新低（顺趋势方向）
    new_ext_count = 0
    for j in range(i - 19, i + 1):
        if j < 50:
            continue
        lookback_start = j - 49
        if trend_dir == 1:  # 多头看新高
            prev_high = np.max(highs[lookback_start:j])
            if highs[j] > prev_high:
                new_ext_count += 1
        else:  # 空头看新低
            prev_low = np.min(lows[lookback_start:j])
            if lows[j] < prev_low:
                new_ext_count += 1
    new_extreme = new_ext_count / 20

    # --- 指标2: range_breakout ---
    # 最近10根的极值超出前40根范围多少（占前40根范围%）
    recent_high = np.max(highs[i-9:i+1])
    recent_low = np.min(lows[i-9:i+1])
    older_high = np.max(highs[i-49:i-9])
    older_low = np.min(lows[i-49:i-9])
    older_range = older_high - older_low

    if older_range > 0:
        breakout_up = max(0, recent_high - older_high)
        breakout_down = max(0, older_low - recent_low)
        if trend_dir == 1:
            range_breakout = breakout_up / older_range
        else:
            range_breakout = breakout_down / older_range
    else:
        range_breakout = 0

    # --- 指标3: range_position ---
    # 收盘价在50根范围里的位置
    full_high = np.max(highs[i-49:i+1])
    full_low = np.min(lows[i-49:i+1])
    full_range = full_high - full_low
    if full_range > 0:
        raw_pos = (close - full_low) / full_range  # 0=最低, 1=最高
        # 顺趋势方向的"边缘度"：多头close靠近高点好，空头靠近低点好
        if trend_dir == 1:
            range_position = raw_pos  # 越高越好
        else:
            range_position = 1 - raw_pos  # 越低越好（转化成越大越好）
    else:
        range_position = 0.5

    # --- 指标4: dir_consistency ---
    # 最近20根里收盘方向一致性
    up_count = 0
    for j in range(i - 19, i + 1):
        if j < 1:
            continue
        if trend_dir == 1 and closes[j] > closes[j-1]:
            up_count += 1
        elif trend_dir == -1 and closes[j] < closes[j-1]:
            up_count += 1
    dir_consistency = up_count / 20

    # --- 指标5: no_overlap ---
    # 最近10根范围超出前40根范围的比例
    # 重叠区间
    overlap_high = min(recent_high, older_high)
    overlap_low = max(recent_low, older_low)
    overlap = max(0, overlap_high - overlap_low)
    recent_range = recent_high - recent_low
    if recent_range > 0:
        no_overlap = 1 - (overlap / recent_range)
    else:
        no_overlap = 0

    # --- 指标6: trend_run ---
    # 连续顺趋势收盘的最长连续根数（最近30根内）
    max_run = 0
    current_run = 0
    for j in range(i - 29, i + 1):
        if j < 1:
            continue
        if (trend_dir == 1 and closes[j] >= closes[j-1]) or \
           (trend_dir == -1 and closes[j] <= closes[j-1]):
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    trend_run = max_run

    er_20 = df.iloc[i]['er_20']

    return {
        'new_extreme': new_extreme,
        'range_breakout': range_breakout,
        'range_position': range_position,
        'dir_consistency': dir_consistency,
        'no_overlap': no_overlap,
        'trend_run': trend_run,
        'er_20': er_20 if not pd.isna(er_20) else np.nan,
    }


def detect_signals(df):
    """B类信号检测 + 所有指标"""
    signals = []
    n = len(df)
    warmup = 65

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

        sig_base = None
        if trend == 1:
            if b_below_start == -1:
                if close < ma_f and prev['close'] >= prev['ma_fast']:
                    b_below_start = i
                    b_pullback_low = row['low']
            else:
                b_pullback_low = min(b_pullback_low, row['low'])
                if close > ma_f:
                    sig_base = {'direction': 'long', 'idx': i, 'time': row['datetime'],
                                'entry_price': close, 'trend_dir': 1}
                    b_below_start = -1
                    b_pullback_low = None
        elif trend == -1:
            if b_below_start == -1:
                if close > ma_f and prev['close'] <= prev['ma_fast']:
                    b_below_start = i
                    b_pullback_high = row['high']
            else:
                b_pullback_high = max(b_pullback_high, row['high'])
                if close < ma_f:
                    sig_base = {'direction': 'short', 'idx': i, 'time': row['datetime'],
                                'entry_price': close, 'trend_dir': -1}
                    b_below_start = -1
                    b_pullback_high = None

        if sig_base is not None:
            indicators = calc_indicators_at(df, i, sig_base['trend_dir'])
            if indicators is None:
                continue
            sig_base.update(indicators)

            # MFE / MAE
            end_idx = min(i + MFE_WINDOW, len(df) - 1)
            if i + 1 <= end_idx:
                segment = df.iloc[i + 1: end_idx + 1]
                if sig_base['direction'] == 'short':
                    mfe = (close - segment['low'].min()) / close * 100
                    mae = (segment['high'].max() - close) / close * 100
                else:
                    mfe = (segment['high'].max() - close) / close * 100
                    mae = (close - segment['low'].min()) / close * 100
                sig_base['mfe'] = max(mfe, 0)
                sig_base['mae'] = max(mae, 0)
                sig_base['mfe_mae'] = sig_base['mfe'] / sig_base['mae'] if sig_base['mae'] > 0 else 10
            else:
                continue

            signals.append(sig_base)

    return signals


def eval_factor(signals, factor):
    """P50切分，对比MFE/MAE中位数"""
    valid = [s for s in signals if not np.isnan(s.get(factor, np.nan))]
    if len(valid) < 20:
        return None

    p50 = np.percentile([s[factor] for s in valid], 50)
    low = [s for s in valid if s[factor] <= p50]
    high = [s for s in valid if s[factor] > p50]
    if not low or not high:
        return None

    low_ratio = np.median([s['mfe_mae'] for s in low])
    high_ratio = np.median([s['mfe_mae'] for s in high])
    low_mfe = np.mean([s['mfe'] for s in low])
    high_mfe = np.mean([s['mfe'] for s in high])
    low_mae = np.mean([s['mae'] for s in low])
    high_mae = np.mean([s['mae'] for s in high])

    return {
        'p50': p50,
        'low_n': len(low), 'high_n': len(high),
        'low_ratio': low_ratio, 'high_ratio': high_ratio,
        'diff': high_ratio - low_ratio,
        'low_mfe': low_mfe, 'high_mfe': high_mfe,
        'low_mae': low_mae, 'high_mae': high_mae,
    }


def main():
    print("=" * 100)
    print("  趋势干净度指标对比")
    print("=" * 100)

    all_signals = []
    for sym_key, sym_name in SYMBOLS.items():
        df = load_cached(sym_key)
        if df is None:
            continue
        df = prepare_data(df)
        cutoff = df['datetime'].iloc[-1] - timedelta(days=120)
        df_120 = df[df['datetime'] >= cutoff].copy().reset_index(drop=True)
        signals = detect_signals(df_120)
        for s in signals:
            s['symbol'] = sym_name
        all_signals.extend(signals)
        print(f"  {sym_name}: {len(signals)} signals")

    print(f"\n  Total: {len(all_signals)}")

    factors = [
        ('new_extreme', '新极值率'),
        ('range_breakout', '区间突破度'),
        ('range_position', '区间位置'),
        ('dir_consistency', '方向一致性'),
        ('no_overlap', '不重叠率'),
        ('trend_run', '最长连涨/跌'),
        ('er_20', 'ER(20)'),
    ]

    # ============================================================
    # 1. 全品种汇总
    # ============================================================
    print(f"\n{'='*100}")
    print(f"  全品种：各指标预测MFE/MAE的能力（P50切分）")
    print(f"{'='*100}")
    print(f"  {'指标':<14} {'P50':>7} | {'低组MFE/MAE':>11} {'高组MFE/MAE':>11} {'差':>7} | "
          f"{'低MFE':>7} {'高MFE':>7} {'低MAE':>7} {'高MAE':>7}")
    print(f"  {'-'*95}")

    global_results = {}
    for fkey, flabel in factors:
        r = eval_factor(all_signals, fkey)
        if r:
            global_results[fkey] = r
            print(f"  {flabel:<14} {r['p50']:>7.3f} | "
                  f"{r['low_ratio']:>10.2f} {r['high_ratio']:>10.2f} {r['diff']:>+6.2f}  | "
                  f"{r['low_mfe']:>6.2f}% {r['high_mfe']:>6.2f}% "
                  f"{r['low_mae']:>6.2f}% {r['high_mae']:>6.2f}%")

    # ============================================================
    # 2. 分品种稳健性
    # ============================================================
    print(f"\n{'='*100}")
    print(f"  分品种稳健性（MFE/MAE差值，正=有效）")
    print(f"{'='*100}")

    sym_names = list(SYMBOLS.values())
    print(f"  {'指标':<14}", end='')
    for name in sym_names:
        print(f" {name:>7}", end='')
    print("  | 正向")
    print(f"  {'-'*75}")

    robustness = {}
    for fkey, flabel in factors:
        print(f"  {flabel:<14}", end='')
        pos = 0
        for sym_name in sym_names:
            sym_sigs = [s for s in all_signals if s['symbol'] == sym_name]
            r = eval_factor(sym_sigs, fkey)
            if r:
                print(f" {r['diff']:>+6.2f}", end='')
                if r['diff'] > 0:
                    pos += 1
            else:
                print(f"    N/A", end='')
        print(f"  |  {pos}/6")
        robustness[fkey] = pos

    # ============================================================
    # 3. 最佳指标三分组详细
    # ============================================================
    # 找正向最多的
    best_factors = sorted(robustness.items(), key=lambda x: -x[1])[:3]

    for fkey, pos_count in best_factors:
        flabel = dict(factors)[fkey]
        print(f"\n{'='*100}")
        print(f"  {flabel} 三分组详细 (正向{pos_count}/6)")
        print(f"{'='*100}")

        vals = [s[fkey] for s in all_signals if not np.isnan(s.get(fkey, np.nan))]
        valid = [s for s in all_signals if not np.isnan(s.get(fkey, np.nan))]
        p33 = np.percentile(vals, 33)
        p67 = np.percentile(vals, 67)

        groups = [
            (f'低(<{p33:.3f})', [s for s in valid if s[fkey] <= p33]),
            (f'中({p33:.3f}-{p67:.3f})', [s for s in valid if p33 < s[fkey] <= p67]),
            (f'高(>{p67:.3f})', [s for s in valid if s[fkey] > p67]),
        ]

        print(f"  {'组':<22} {'N':>4} | {'MFE':>7} {'MAE':>7} {'MFE/MAE':>8} | {'MFE>3%':>7}")
        print(f"  {'-'*65}")

        for label, sigs in groups:
            if not sigs:
                continue
            n = len(sigs)
            mfe = np.mean([s['mfe'] for s in sigs])
            mae = np.mean([s['mae'] for s in sigs])
            ratio = np.median([s['mfe_mae'] for s in sigs])
            burst = sum(1 for s in sigs if s['mfe'] >= 3.0) / n * 100
            print(f"  {label:<22} {n:>4} | {mfe:>6.2f}% {mae:>6.2f}% {ratio:>7.2f}  | {burst:>6.1f}%")

    # ============================================================
    # 4. 对照：看生猪和碳酸锂在最佳指标上的分布差异
    # ============================================================
    print(f"\n{'='*100}")
    print(f"  生猪 vs 碳酸锂：各指标均值对比")
    print(f"{'='*100}")
    print(f"  {'指标':<14} {'生猪':>8} {'碳酸锂':>8} {'差距':>8}")
    print(f"  {'-'*45}")

    for fkey, flabel in factors:
        zhu = [s[fkey] for s in all_signals if s['symbol'] == '生猪' and not np.isnan(s.get(fkey, np.nan))]
        lc = [s[fkey] for s in all_signals if s['symbol'] == '碳酸锂' and not np.isnan(s.get(fkey, np.nan))]
        if zhu and lc:
            z_mean = np.mean(zhu)
            l_mean = np.mean(lc)
            print(f"  {flabel:<14} {z_mean:>8.3f} {l_mean:>8.3f} {z_mean - l_mean:>+7.3f}")

    # ============================================================
    # 图表
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 图1: 因子区分能力排名
    ax = axes[0, 0]
    fnames = [dict(factors)[fkey] for fkey, _ in best_factors] + \
             [dict(factors)[fkey] for fkey, _ in sorted(robustness.items(), key=lambda x: -x[1])[3:]]
    fdiffs = []
    for fkey in [bf[0] for bf in best_factors] + [bf[0] for bf in sorted(robustness.items(), key=lambda x: -x[1])[3:]]:
        r = global_results.get(fkey)
        fdiffs.append(r['diff'] if r else 0)
    colors = ['#4CAF50' if d > 0 else '#f44336' for d in fdiffs]
    bars = ax.barh(fnames[::-1], fdiffs[::-1], color=colors[::-1], alpha=0.7)
    ax.axvline(x=0, color='black', linewidth=0.5)
    for bar, val in zip(bars, fdiffs[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:+.2f}', va='center', fontsize=9)
    ax.set_title('各指标预测MFE/MAE差值\n(P50切分，高组-低组)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 图2: 稳健性（品种正向数）
    ax = axes[0, 1]
    rob_names = [dict(factors)[k] for k, _ in sorted(robustness.items(), key=lambda x: -x[1])]
    rob_vals = [v for _, v in sorted(robustness.items(), key=lambda x: -x[1])]
    colors = ['#4CAF50' if v >= 5 else '#FFC107' if v >= 4 else '#f44336' for v in rob_vals]
    ax.barh(rob_names[::-1], rob_vals[::-1], color=colors[::-1], alpha=0.7)
    ax.axvline(x=3, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('品种正向数 (/6)')
    ax.set_title('分品种稳健性', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 图3: 生猪 vs 碳酸锂 雷达图
    ax = axes[0, 2]
    radar_factors = ['new_extreme', 'range_breakout', 'range_position', 'dir_consistency', 'no_overlap', 'er_20']
    radar_labels = ['新极值率', '区间突破度', '区间位置', '方向一致性', '不重叠率', 'ER(20)']

    zhu_vals = []
    lc_vals = []
    for fk in radar_factors:
        z = [s[fk] for s in all_signals if s['symbol'] == '生猪' and not np.isnan(s.get(fk, np.nan))]
        l = [s[fk] for s in all_signals if s['symbol'] == '碳酸锂' and not np.isnan(s.get(fk, np.nan))]
        zhu_vals.append(np.mean(z) if z else 0)
        lc_vals.append(np.mean(l) if l else 0)

    x = np.arange(len(radar_labels))
    w = 0.35
    ax.barh(x - w/2, zhu_vals, w, label='生猪', color='#4CAF50', alpha=0.7)
    ax.barh(x + w/2, lc_vals, w, label='碳酸锂', color='#f44336', alpha=0.7)
    ax.set_yticks(x)
    ax.set_yticklabels(radar_labels)
    ax.legend()
    ax.set_title('生猪 vs 碳酸锂 指标对比', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 图4-6: 前3个最佳因子的三分组箱线图
    for plot_idx, (fkey, pos_count) in enumerate(best_factors[:3]):
        ax = axes[1, plot_idx]
        flabel = dict(factors)[fkey]
        vals = [s[fkey] for s in all_signals if not np.isnan(s.get(fkey, np.nan))]
        valid = [s for s in all_signals if not np.isnan(s.get(fkey, np.nan))]
        p33 = np.percentile(vals, 33)
        p67 = np.percentile(vals, 67)

        groups = [
            ('低', [s for s in valid if s[fkey] <= p33]),
            ('中', [s for s in valid if p33 < s[fkey] <= p67]),
            ('高', [s for s in valid if s[fkey] > p67]),
        ]

        box_data = []
        box_labels = []
        for label, sigs in groups:
            if sigs:
                box_data.append([min(s['mfe_mae'], 10) for s in sigs])
                box_labels.append(f"{label}\n(N={len(sigs)})")

        if box_data:
            bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, showmeans=True)
            colors_box = ['#f44336', '#FFC107', '#4CAF50']
            for patch, c in zip(bp['boxes'], colors_box[:len(box_data)]):
                patch.set_facecolor(c)
                patch.set_alpha(0.5)
        ax.set_title(f'{flabel}\nMFE/MAE分布 (正向{pos_count}/6)', fontweight='bold')
        ax.set_ylabel('MFE/MAE')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'trend_clean.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图表: {out_path}")


if __name__ == '__main__':
    main()
