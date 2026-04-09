# -*- coding: utf-8 -*-
"""
路径质量研究：ER能否预测MFE/MAE比？
====================================
核心问题：用效率比(ER)预测"走势干净程度"(MFE/MAE)，
区分"生猪型干净趋势"和"碳酸锂型震荡趋势"。

目标变量换成MFE/MAE比，而非MFE本身。
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
MFE_WINDOW = 48  # 8h


def load_cached(symbol_key):
    path = os.path.join(CACHE_DIR, f"{symbol_key}_10min_170d.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def prepare_data(df):
    if 'ma_fast' not in df.columns:
        df['ma_fast'] = df['close'].rolling(MA_FAST).mean()
    if 'ma_slow' not in df.columns:
        df['ma_slow'] = df['close'].rolling(MA_SLOW).mean()

    # 多周期ER
    for p in [10, 20, 30, 40, 60]:
        net = (df['close'] - df['close'].shift(p)).abs()
        bar_sum = df['close'].diff().abs().rolling(p).sum()
        df[f'er_{p}'] = net / bar_sum.replace(0, np.nan)

    # ATR
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100

    # 60MA距离
    df['ma_60'] = df['close'].rolling(60).mean()
    df['dist_60ma_pct'] = ((df['close'] - df['ma_60']) / df['ma_60'] * 100).abs()

    # 趋势
    df['trend'] = 0
    df.loc[df['ma_fast'] > df['ma_slow'], 'trend'] = 1
    df.loc[df['ma_fast'] < df['ma_slow'], 'trend'] = -1

    # 路径粗糙度: 过去N根K线的 (最高-最低范围) / 净位移
    # 越大越粗糙（来回震荡），越接近1越干净
    for p in [20, 40]:
        roll_high = df['high'].rolling(p).max()
        roll_low = df['low'].rolling(p).min()
        net_move = (df['close'] - df['close'].shift(p)).abs()
        range_move = roll_high - roll_low
        df[f'roughness_{p}'] = range_move / net_move.replace(0, np.nan)
        # clip极端值
        df[f'roughness_{p}'] = df[f'roughness_{p}'].clip(upper=50)

    return df


def detect_signals(df):
    """B类信号检测"""
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
        high = row['high']
        low = row['low']
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
        prev_close = prev['close']
        prev_ma_f = prev['ma_fast']
        if pd.isna(prev_ma_f):
            continue

        sig_base = None

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
            for col in ['er_10', 'er_20', 'er_30', 'er_40', 'er_60',
                         'atr_pct', 'dist_60ma_pct', 'roughness_20', 'roughness_40']:
                val = row.get(col, np.nan)
                sig_base[col] = val if not pd.isna(val) else np.nan

            # MFE和MAE
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

                # 路径到达MFE前的最大回撤
                running_mfe = 0
                max_dd_before_peak = 0
                for j in range(i + 1, end_idx + 1):
                    bar = df.iloc[j]
                    if sig_base['direction'] == 'long':
                        curr_pnl_high = (bar['high'] - close) / close * 100
                        curr_pnl_low = (bar['low'] - close) / close * 100
                    else:
                        curr_pnl_high = (close - bar['low']) / close * 100
                        curr_pnl_low = (close - bar['high']) / close * 100
                    if curr_pnl_high > running_mfe:
                        running_mfe = curr_pnl_high
                    dd = running_mfe - curr_pnl_low
                    if dd > max_dd_before_peak:
                        max_dd_before_peak = dd
                sig_base['path_dd'] = max_dd_before_peak  # 到达MFE过程中经历的最大回撤
                sig_base['capturable'] = sig_base['mfe'] - sig_base['path_dd']  # 可捕获利润
            else:
                sig_base['mfe'] = 0
                sig_base['mae'] = 0
                sig_base['path_dd'] = 0
                sig_base['capturable'] = 0

            # MFE/MAE比
            if sig_base['mae'] > 0:
                sig_base['mfe_mae'] = sig_base['mfe'] / sig_base['mae']
            else:
                sig_base['mfe_mae'] = sig_base['mfe'] * 10 if sig_base['mfe'] > 0 else 1

            signals.append(sig_base)

    return signals


def eval_factor_on_target(signals, factor, target, label):
    """用factor的P50切分，对比target"""
    valid = [s for s in signals if not np.isnan(s.get(factor, np.nan)) and not np.isnan(s.get(target, np.nan))]
    if len(valid) < 20:
        return None

    p50 = np.percentile([s[factor] for s in valid], 50)
    low = [s for s in valid if s[factor] <= p50]
    high = [s for s in valid if s[factor] > p50]

    if not low or not high:
        return None

    low_target = np.median([s[target] for s in low])
    high_target = np.median([s[target] for s in high])

    return {
        'factor': label,
        'p50': p50,
        'low_n': len(low),
        'high_n': len(high),
        'low_val': low_target,
        'high_val': high_target,
        'diff': high_target - low_target,
        'low_mfe': np.mean([s['mfe'] for s in low]),
        'high_mfe': np.mean([s['mfe'] for s in high]),
        'low_mae': np.mean([s['mae'] for s in low]),
        'high_mae': np.mean([s['mae'] for s in high]),
        'low_capturable': np.mean([s['capturable'] for s in low]),
        'high_capturable': np.mean([s['capturable'] for s in high]),
    }


def main():
    print("=" * 100)
    print("  路径质量研究：什么因子能预测'走势干净程度'？")
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

        # 品种级统计
        if signals:
            mfe_mae_med = np.median([s['mfe_mae'] for s in signals])
            cap_mean = np.mean([s['capturable'] for s in signals])
            mfe_mean = np.mean([s['mfe'] for s in signals])
            mae_mean = np.mean([s['mae'] for s in signals])
            er20_mean = np.mean([s['er_20'] for s in signals if not np.isnan(s['er_20'])])
            print(f"  {sym_name:<6} N={len(signals):>3} | MFE={mfe_mean:>5.2f}% MAE={mae_mean:>5.2f}% "
                  f"MFE/MAE中位={mfe_mae_med:>5.2f} | ER20={er20_mean:.3f} | "
                  f"可捕获={cap_mean:>+5.2f}%")

    print(f"\n  Total: {len(all_signals)} signals")

    # ============================================================
    # 1. 各因子预测MFE/MAE比的能力
    # ============================================================
    factors = [
        ('er_10', 'ER(10)'), ('er_20', 'ER(20)'), ('er_30', 'ER(30)'),
        ('er_40', 'ER(40)'), ('er_60', 'ER(60)'),
        ('roughness_20', '粗糙度(20)'), ('roughness_40', '粗糙度(40)'),
        ('atr_pct', 'ATR%'), ('dist_60ma_pct', '距60MA%'),
    ]

    print(f"\n{'='*100}")
    print(f"  各因子预测MFE/MAE比（中位数）的能力")
    print(f"{'='*100}")
    print(f"  {'因子':<12} {'P50':>8} | {'低组MFE/MAE':>11} {'高组MFE/MAE':>11} {'差值':>8} | {'低组可捕获':>10} {'高组可捕获':>10}")
    print(f"  {'-'*90}")

    results = []
    for fkey, flabel in factors:
        r = eval_factor_on_target(all_signals, fkey, 'mfe_mae', flabel)
        if r:
            results.append(r)
            print(f"  {flabel:<12} {r['p50']:>8.3f} | "
                  f"{r['low_val']:>10.2f} {r['high_val']:>10.2f} {r['diff']:>+7.2f}  | "
                  f"{r['low_capturable']:>+9.2f}% {r['high_capturable']:>+9.2f}%")

    # ============================================================
    # 2. 各因子预测"可捕获利润"的能力
    # ============================================================
    print(f"\n{'='*100}")
    print(f"  各因子预测'可捕获利润'(MFE-路径回撤)的能力")
    print(f"{'='*100}")
    print(f"  {'因子':<12} | {'低组':>8} {'高组':>8} {'差值':>8} | {'低组MFE':>8} {'高组MFE':>8} | {'低组MAE':>8} {'高组MAE':>8}")
    print(f"  {'-'*90}")

    for fkey, flabel in factors:
        r = eval_factor_on_target(all_signals, fkey, 'capturable', flabel)
        if r:
            print(f"  {flabel:<12} | "
                  f"{r['low_val']:>+7.2f}% {r['high_val']:>+7.2f}% {r['diff']:>+7.2f}% | "
                  f"{r['low_mfe']:>7.2f}% {r['high_mfe']:>7.2f}% | "
                  f"{r['low_mae']:>7.2f}% {r['high_mae']:>7.2f}%")

    # ============================================================
    # 3. 分品种稳健性：ER(20)预测MFE/MAE
    # ============================================================
    print(f"\n{'='*100}")
    print(f"  分品种稳健性：各因子预测MFE/MAE")
    print(f"{'='*100}")

    sym_names = list(SYMBOLS.values())
    print(f"  {'因子':<12}", end='')
    for name in sym_names:
        print(f" {name:>8}", end='')
    print(" | 正向")
    print(f"  {'-'*75}")

    for fkey, flabel in factors:
        print(f"  {flabel:<12}", end='')
        pos_count = 0
        for sym_name in sym_names:
            sym_sigs = [s for s in all_signals if s['symbol'] == sym_name]
            r = eval_factor_on_target(sym_sigs, fkey, 'mfe_mae', flabel)
            if r:
                print(f" {r['diff']:>+7.2f}", end='')
                if r['diff'] > 0:
                    pos_count += 1
            else:
                print(f"     N/A", end='')
        print(f" |  {pos_count}/6")

    # ============================================================
    # 4. 粗糙度(roughness)详细看
    # ============================================================
    print(f"\n{'='*100}")
    print(f"  粗糙度(20)三分组详细")
    print(f"{'='*100}")

    rough_vals = [s['roughness_20'] for s in all_signals if not np.isnan(s['roughness_20'])]
    valid_sigs = [s for s in all_signals if not np.isnan(s['roughness_20'])]
    p33 = np.percentile(rough_vals, 33)
    p67 = np.percentile(rough_vals, 67)

    groups = {
        f'干净(<{p33:.1f})': [s for s in valid_sigs if s['roughness_20'] <= p33],
        f'中等({p33:.1f}-{p67:.1f})': [s for s in valid_sigs if p33 < s['roughness_20'] <= p67],
        f'粗糙(>{p67:.1f})': [s for s in valid_sigs if s['roughness_20'] > p67],
    }

    print(f"\n  {'组':<18} {'N':>4} | {'MFE':>7} {'MAE':>7} {'MFE/MAE':>8} | {'可捕获':>8} {'路径回撤':>8} | {'ER20':>6}")
    print(f"  {'-'*85}")

    for label, sigs in groups.items():
        if not sigs:
            continue
        n = len(sigs)
        mfe = np.mean([s['mfe'] for s in sigs])
        mae = np.mean([s['mae'] for s in sigs])
        ratio = np.median([s['mfe_mae'] for s in sigs])
        cap = np.mean([s['capturable'] for s in sigs])
        dd = np.mean([s['path_dd'] for s in sigs])
        er = np.mean([s['er_20'] for s in sigs if not np.isnan(s['er_20'])])
        print(f"  {label:<18} {n:>4} | {mfe:>6.2f}% {mae:>6.2f}% {ratio:>7.2f}  | "
              f"{cap:>+7.2f}% {dd:>7.2f}%  | {er:>5.3f}")

    # ============================================================
    # 5. ER高+粗糙度低 的组合
    # ============================================================
    print(f"\n{'='*100}")
    print(f"  ER(20) x 粗糙度(20) 交叉验证")
    print(f"{'='*100}")

    er_valid = [s for s in all_signals if not np.isnan(s['er_20']) and not np.isnan(s['roughness_20'])]
    er_p50 = np.percentile([s['er_20'] for s in er_valid], 50)
    rough_p50 = np.percentile([s['roughness_20'] for s in er_valid], 50)

    cross_groups = {
        'ER低+粗糙':   [s for s in er_valid if s['er_20'] <= er_p50 and s['roughness_20'] > rough_p50],
        'ER低+干净':   [s for s in er_valid if s['er_20'] <= er_p50 and s['roughness_20'] <= rough_p50],
        'ER高+粗糙':   [s for s in er_valid if s['er_20'] > er_p50 and s['roughness_20'] > rough_p50],
        'ER高+干净':   [s for s in er_valid if s['er_20'] > er_p50 and s['roughness_20'] <= rough_p50],
    }

    print(f"  ER(20) P50={er_p50:.3f}, 粗糙度(20) P50={rough_p50:.1f}")
    print(f"\n  {'组合':<14} {'N':>4} | {'MFE':>7} {'MAE':>7} {'MFE/MAE':>8} | {'可捕获':>8} | {'ATR%':>7}")
    print(f"  {'-'*70}")

    for label, sigs in cross_groups.items():
        if not sigs:
            continue
        n = len(sigs)
        mfe = np.mean([s['mfe'] for s in sigs])
        mae = np.mean([s['mae'] for s in sigs])
        ratio = np.median([s['mfe_mae'] for s in sigs])
        cap = np.mean([s['capturable'] for s in sigs])
        atr = np.mean([s['atr_pct'] for s in sigs])
        print(f"  {label:<14} {n:>4} | {mfe:>6.2f}% {mae:>6.2f}% {ratio:>7.2f}  | {cap:>+7.2f}% | {atr:>6.3f}%")

    # ============================================================
    # 图表
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 图1: 各因子预测MFE/MAE差值
    ax = axes[0, 0]
    if results:
        names = [r['factor'] for r in results]
        diffs = [r['diff'] for r in results]
        colors = ['#4CAF50' if d > 0 else '#f44336' for d in diffs]
        bars = ax.barh(names, diffs, color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linewidth=0.5)
        for bar, val in zip(bars, diffs):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:+.2f}', va='center', fontsize=9)
    ax.set_title('各因子预测MFE/MAE比的能力\n(高组-低组中位数差)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 图2: 品种MFE/MAE中位数 vs ER20
    ax = axes[0, 1]
    for sym_name in sym_names:
        sym_sigs = [s for s in all_signals if s['symbol'] == sym_name]
        if sym_sigs:
            er = np.mean([s['er_20'] for s in sym_sigs if not np.isnan(s['er_20'])])
            ratio = np.median([s['mfe_mae'] for s in sym_sigs])
            ax.scatter(er, ratio, s=100, zorder=5)
            ax.annotate(sym_name, (er, ratio), textcoords="offset points",
                       xytext=(5, 5), fontsize=10, fontweight='bold')
    ax.set_xlabel('ER(20) 均值')
    ax.set_ylabel('MFE/MAE 中位数')
    ax.set_title('品种级别: ER vs MFE/MAE', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 图3: 粗糙度三分组箱线图
    ax = axes[0, 2]
    box_data = []
    box_labels = []
    for label, sigs in groups.items():
        if sigs:
            box_data.append([min(s['mfe_mae'], 10) for s in sigs])
            short_label = label.split('(')[0]
            box_labels.append(f"{short_label}\n(N={len(sigs)})")
    if box_data:
        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, showmeans=True)
        colors = ['#4CAF50', '#FFC107', '#f44336']
        for patch, c in zip(bp['boxes'], colors[:len(box_data)]):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
    ax.set_title('粗糙度分组 MFE/MAE分布', fontweight='bold')
    ax.set_ylabel('MFE/MAE')
    ax.grid(True, alpha=0.3)

    # 图4: ER x 粗糙度 交叉验证
    ax = axes[1, 0]
    cross_labels = list(cross_groups.keys())
    cross_ratios = []
    cross_caps = []
    for label in cross_labels:
        sigs = cross_groups[label]
        if sigs:
            cross_ratios.append(np.median([s['mfe_mae'] for s in sigs]))
            cross_caps.append(np.mean([s['capturable'] for s in sigs]))
        else:
            cross_ratios.append(0)
            cross_caps.append(0)

    x = np.arange(len(cross_labels))
    w = 0.35
    ax.bar(x - w/2, cross_ratios, w, label='MFE/MAE中位数', color='#2196f3', alpha=0.7)
    ax.bar(x + w/2, cross_caps, w, label='可捕获利润%', color='#4CAF50', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(cross_labels, fontsize=9)
    ax.legend()
    ax.set_title('ER x 粗糙度 交叉验证', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 图5: 各品种MFE vs MAE散点
    ax = axes[1, 1]
    for sym_name in sym_names:
        sym_sigs = [s for s in all_signals if s['symbol'] == sym_name]
        if sym_sigs:
            mfes = [s['mfe'] for s in sym_sigs]
            maes = [s['mae'] for s in sym_sigs]
            ax.scatter(maes, mfes, alpha=0.3, s=15, label=sym_name)
    ax.plot([0, 10], [0, 10], 'r--', alpha=0.5, label='MFE=MAE')
    ax.legend(fontsize=8)
    ax.set_xlabel('MAE (%)')
    ax.set_ylabel('MFE (%)')
    ax.set_title('MFE vs MAE 散点图', fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3)

    # 图6: 可捕获利润分布
    ax = axes[1, 2]
    for sym_name in ['生猪', '碳酸锂', '白银', '原油']:
        sym_sigs = [s for s in all_signals if s['symbol'] == sym_name]
        if sym_sigs:
            caps = [s['capturable'] for s in sym_sigs]
            ax.hist(caps, bins=30, alpha=0.4, label=f'{sym_name} (med={np.median(caps):+.2f}%)')
    ax.axvline(x=0, color='red', linestyle='--')
    ax.legend(fontsize=9)
    ax.set_title('可捕获利润分布', fontweight='bold')
    ax.set_xlabel('可捕获利润 (%)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'path_quality.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图表已保存: {out_path}")


if __name__ == '__main__':
    main()
