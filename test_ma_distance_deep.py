# -*- coding: utf-8 -*-
"""
距60MA深挖：是真信号还是波动率幻觉？
=====================================
1. MFE用ATR标准化后，距60MA的优势是否消失？
2. MAE（最大不利波动）在远近组之间的差异
3. MFE/MAE比值（风险调整后的收益）
4. 挑出具体case在K线图上标注

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
MA_60 = 60
ATR_PERIOD = 14
MFE_WINDOW = 48  # 8h
BURST_THRESHOLD = 3.0


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
    df['ma_60'] = df['close'].rolling(MA_60).mean()
    df['dist_60ma_pct'] = ((df['close'] - df['ma_60']) / df['ma_60'] * 100).abs()

    # ATR
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(ATR_PERIOD).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100  # ATR占价格百分比

    df['trend'] = 0
    df.loc[df['ma_fast'] > df['ma_slow'], 'trend'] = 1
    df.loc[df['ma_fast'] < df['ma_slow'], 'trend'] = -1
    return df


def detect_signals(df):
    """检测B类信号，记录全部所需数据"""
    signals = []
    n = len(df)
    warmup = MA_60 + ATR_PERIOD + 5

    b_below_start = -1
    b_pullback_low = None
    b_pullback_high = None
    prev_trend = 0

    for i in range(warmup, n):
        row = df.iloc[i]
        if pd.isna(row['ma_fast']) or pd.isna(row['ma_slow']) or pd.isna(row['ma_60']) or pd.isna(row['atr']):
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
            sig_base['dist_60ma_pct'] = row['dist_60ma_pct']
            sig_base['atr_pct'] = row['atr_pct']
            sig_base['atr'] = row['atr']

            # 计算MFE和MAE
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
                # ATR标准化版本
                if row['atr_pct'] > 0:
                    sig_base['mfe_norm'] = sig_base['mfe'] / row['atr_pct']
                    sig_base['mae_norm'] = sig_base['mae'] / row['atr_pct']
                else:
                    sig_base['mfe_norm'] = 0
                    sig_base['mae_norm'] = 0
            else:
                sig_base['mfe'] = 0
                sig_base['mae'] = 0
                sig_base['mfe_norm'] = 0
                sig_base['mae_norm'] = 0

            sig_base['is_burst'] = sig_base['mfe'] >= BURST_THRESHOLD
            # MFE/MAE比
            sig_base['mfe_mae_ratio'] = sig_base['mfe'] / sig_base['mae'] if sig_base['mae'] > 0 else sig_base['mfe'] * 10
            signals.append(sig_base)

    return signals


def print_group_stats(label, sigs):
    """打印一组信号的完整统计"""
    if not sigs:
        print(f"  {label:<16}    0 |  -- N/A --")
        return

    n = len(sigs)
    mfe_vals = [s['mfe'] for s in sigs]
    mae_vals = [s['mae'] for s in sigs]
    mfe_norm = [s['mfe_norm'] for s in sigs]
    mae_norm = [s['mae_norm'] for s in sigs]
    ratios = [s['mfe_mae_ratio'] for s in sigs]
    burst = sum(1 for s in sigs if s['is_burst']) / n * 100
    atr_pct = np.mean([s['atr_pct'] for s in sigs])

    print(f"  {label:<16} {n:>4} | "
          f"MFE={np.mean(mfe_vals):>5.2f}% MAE={np.mean(mae_vals):>5.2f}% "
          f"MFE/MAE={np.median(ratios):>4.2f} | "
          f"norm_MFE={np.mean(mfe_norm):>5.2f} norm_MAE={np.mean(mae_norm):>5.2f} | "
          f"ATR%={atr_pct:>5.3f}% burst={burst:>5.1f}%")


def main():
    print("=" * 100)
    print("  距60MA深挖：真信号 还是 波动率幻觉？")
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
            s['sym_key'] = sym_key
            s['_df_ref'] = df_120  # 保留引用用于画图
        all_signals.extend(signals)
        print(f"  {sym_name}: {len(signals)} signals, ATR%={np.mean([s['atr_pct'] for s in signals]):.3f}%")

    print(f"\n  Total: {len(all_signals)} signals")

    # ============================================================
    # 1. 全品种汇总：原始 vs ATR标准化
    # ============================================================
    p50 = np.percentile([s['dist_60ma_pct'] for s in all_signals], 50)
    near = [s for s in all_signals if s['dist_60ma_pct'] <= p50]
    far = [s for s in all_signals if s['dist_60ma_pct'] > p50]

    print(f"\n{'='*100}")
    print(f"  问题1: MFE用ATR标准化后，距60MA的优势是否消失？")
    print(f"  (P50={p50:.3f}%)")
    print(f"{'='*100}")
    print(f"  {'分组':<16} {'N':>4} | {'MFE':>7} {'MAE':>7} {'MFE/MAE':>8} | {'norm_MFE':>9} {'norm_MAE':>9} | {'ATR%':>7} {'爆发率':>7}")
    print(f"  {'-'*95}")
    print_group_stats("近60MA", near)
    print_group_stats("远60MA", far)
    print_group_stats("全部", all_signals)

    # 标准化后的"爆发"：norm_MFE > 某阈值
    # 先看norm_MFE的分布来定阈值
    near_norm = [s['mfe_norm'] for s in near]
    far_norm = [s['mfe_norm'] for s in far]
    norm_p75 = np.percentile([s['mfe_norm'] for s in all_signals], 75)

    print(f"\n  标准化后的爆发（norm_MFE > {norm_p75:.2f}，即全局P75）：")
    near_norm_burst = sum(1 for s in near if s['mfe_norm'] > norm_p75) / len(near) * 100
    far_norm_burst = sum(1 for s in far if s['mfe_norm'] > norm_p75) / len(far) * 100
    print(f"    近60MA: {near_norm_burst:.1f}%")
    print(f"    远60MA: {far_norm_burst:.1f}%")
    print(f"    差异: {far_norm_burst - near_norm_burst:+.1f}%")

    # ============================================================
    # 2. 分品种验证
    # ============================================================
    print(f"\n{'='*100}")
    print(f"  分品种验证")
    print(f"{'='*100}")

    for sym_name in SYMBOLS.values():
        sym_sigs = [s for s in all_signals if s['symbol'] == sym_name]
        if len(sym_sigs) < 20:
            continue

        sym_p50 = np.percentile([s['dist_60ma_pct'] for s in sym_sigs], 50)
        sym_near = [s for s in sym_sigs if s['dist_60ma_pct'] <= sym_p50]
        sym_far = [s for s in sym_sigs if s['dist_60ma_pct'] > sym_p50]

        # 原始MFE差
        near_mfe = np.mean([s['mfe'] for s in sym_near])
        far_mfe = np.mean([s['mfe'] for s in sym_far])
        raw_ratio = far_mfe / near_mfe if near_mfe > 0 else 0

        # 标准化MFE差
        near_norm_mfe = np.mean([s['mfe_norm'] for s in sym_near])
        far_norm_mfe = np.mean([s['mfe_norm'] for s in sym_far])
        norm_ratio = far_norm_mfe / near_norm_mfe if near_norm_mfe > 0 else 0

        # ATR差
        near_atr = np.mean([s['atr_pct'] for s in sym_near])
        far_atr = np.mean([s['atr_pct'] for s in sym_far])

        # MAE
        near_mae = np.mean([s['mae'] for s in sym_near])
        far_mae = np.mean([s['mae'] for s in sym_far])

        # MFE/MAE中位数
        near_ratio = np.median([s['mfe_mae_ratio'] for s in sym_near])
        far_ratio = np.median([s['mfe_mae_ratio'] for s in sym_far])

        print(f"\n  {sym_name} (N={len(sym_sigs)}, P50={sym_p50:.2f}%):")
        print(f"    {'':>12} {'MFE':>7} {'MAE':>7} {'MFE/MAE':>8} | {'norm_MFE':>9} | {'ATR%':>7}")
        print(f"    {'近60MA':<12} {near_mfe:>6.2f}% {near_mae:>6.2f}% {near_ratio:>7.2f}  | {near_norm_mfe:>8.2f}  | {near_atr:>6.3f}%")
        print(f"    {'远60MA':<12} {far_mfe:>6.2f}% {far_mae:>6.2f}% {far_ratio:>7.2f}  | {far_norm_mfe:>8.2f}  | {far_atr:>6.3f}%")
        print(f"    {'远/近比':<12} {raw_ratio:>6.2f}x {far_mae/near_mae if near_mae > 0 else 0:>6.2f}x {'':>8}  | {norm_ratio:>8.2f}x | {far_atr/near_atr if near_atr > 0 else 0:>6.2f}x")

    # ============================================================
    # 3. 关键问题：按ATR分位数控制后再看60MA
    # ============================================================
    print(f"\n{'='*100}")
    print(f"  控制变量：固定ATR水平后，距60MA还有没有区分能力？")
    print(f"{'='*100}")

    atr_p33 = np.percentile([s['atr_pct'] for s in all_signals], 33)
    atr_p67 = np.percentile([s['atr_pct'] for s in all_signals], 67)

    atr_groups = {
        '低ATR': [s for s in all_signals if s['atr_pct'] <= atr_p33],
        '中ATR': [s for s in all_signals if atr_p33 < s['atr_pct'] <= atr_p67],
        '高ATR': [s for s in all_signals if s['atr_pct'] > atr_p67],
    }

    print(f"  ATR% 分位: P33={atr_p33:.3f}%, P67={atr_p67:.3f}%")
    print(f"\n  {'ATR组':<8} {'60MA组':<10} {'N':>4} | {'MFE':>7} {'MAE':>7} {'MFE/MAE':>8} | {'爆发率':>7} {'norm_MFE':>9}")
    print(f"  {'-'*80}")

    for atr_label, atr_sigs in atr_groups.items():
        if len(atr_sigs) < 20:
            continue
        local_p50 = np.percentile([s['dist_60ma_pct'] for s in atr_sigs], 50)
        local_near = [s for s in atr_sigs if s['dist_60ma_pct'] <= local_p50]
        local_far = [s for s in atr_sigs if s['dist_60ma_pct'] > local_p50]

        for ma_label, ma_sigs in [('近60MA', local_near), ('远60MA', local_far)]:
            if not ma_sigs:
                continue
            n = len(ma_sigs)
            mfe = np.mean([s['mfe'] for s in ma_sigs])
            mae = np.mean([s['mae'] for s in ma_sigs])
            ratio = np.median([s['mfe_mae_ratio'] for s in ma_sigs])
            burst = sum(1 for s in ma_sigs if s['is_burst']) / n * 100
            norm_mfe = np.mean([s['mfe_norm'] for s in ma_sigs])
            print(f"  {atr_label:<8} {ma_label:<10} {n:>4} | "
                  f"{mfe:>6.2f}% {mae:>6.2f}% {ratio:>7.2f}  | "
                  f"{burst:>6.1f}% {norm_mfe:>8.2f}")

    # ============================================================
    # 图表
    # ============================================================
    fig = plt.figure(figsize=(20, 20))

    # --- 图1: 原始MFE vs 标准化MFE 对比 ---
    ax = fig.add_subplot(3, 2, 1)
    labels_list = ['近60MA\n原始MFE', '远60MA\n原始MFE', '近60MA\nnorm_MFE', '远60MA\nnorm_MFE']
    vals = [np.mean([s['mfe'] for s in near]), np.mean([s['mfe'] for s in far]),
            np.mean(near_norm), np.mean(far_norm)]
    colors = ['#2196f3', '#4CAF50', '#2196f3', '#4CAF50']
    hatches = ['', '', '//', '//']
    bars = ax.bar(labels_list, vals, color=colors, alpha=0.7)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', fontweight='bold')
    ax.set_title('原始MFE vs ATR标准化MFE', fontweight='bold')
    ax.set_ylabel('值')
    ax.grid(True, alpha=0.3)

    # --- 图2: MFE和MAE的箱线图 ---
    ax = fig.add_subplot(3, 2, 2)
    box_data = [
        [s['mfe'] for s in near], [s['mfe'] for s in far],
        [s['mae'] for s in near], [s['mae'] for s in far],
    ]
    bp = ax.boxplot(box_data,
                    tick_labels=['近60MA\nMFE', '远60MA\nMFE', '近60MA\nMAE', '远60MA\nMAE'],
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='white', markersize=5))
    box_colors = ['#2196f3', '#4CAF50', '#ff9800', '#f44336']
    for patch, c in zip(bp['boxes'], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    ax.set_title('MFE与MAE分布对比', fontweight='bold')
    ax.set_ylabel('%')
    ax.grid(True, alpha=0.3)

    # --- 图3: MFE/MAE比值分布 ---
    ax = fig.add_subplot(3, 2, 3)
    near_ratios = [min(s['mfe_mae_ratio'], 10) for s in near]  # clip极端值
    far_ratios = [min(s['mfe_mae_ratio'], 10) for s in far]
    ax.hist(near_ratios, bins=40, alpha=0.5, label=f'近60MA (med={np.median(near_ratios):.2f})', color='#2196f3')
    ax.hist(far_ratios, bins=40, alpha=0.5, label=f'远60MA (med={np.median(far_ratios):.2f})', color='#4CAF50')
    ax.axvline(x=1, color='red', linestyle='--', label='MFE=MAE')
    ax.legend()
    ax.set_title('MFE/MAE比值分布（>1越好）', fontweight='bold')
    ax.set_xlabel('MFE/MAE')
    ax.grid(True, alpha=0.3)

    # --- 图4: 控制ATR后的爆发率 ---
    ax = fig.add_subplot(3, 2, 4)
    atr_labels_plot = []
    near_bursts = []
    far_bursts = []
    for atr_label, atr_sigs in atr_groups.items():
        if len(atr_sigs) < 20:
            continue
        local_p50 = np.percentile([s['dist_60ma_pct'] for s in atr_sigs], 50)
        local_near = [s for s in atr_sigs if s['dist_60ma_pct'] <= local_p50]
        local_far = [s for s in atr_sigs if s['dist_60ma_pct'] > local_p50]
        if local_near and local_far:
            atr_labels_plot.append(atr_label)
            near_bursts.append(sum(1 for s in local_near if s['is_burst']) / len(local_near) * 100)
            far_bursts.append(sum(1 for s in local_far if s['is_burst']) / len(local_far) * 100)

    if atr_labels_plot:
        x = np.arange(len(atr_labels_plot))
        w = 0.35
        ax.bar(x - w/2, near_bursts, w, label='近60MA', color='#2196f3', alpha=0.7)
        ax.bar(x + w/2, far_bursts, w, label='远60MA', color='#4CAF50', alpha=0.7)
        for i, (nb, fb) in enumerate(zip(near_bursts, far_bursts)):
            ax.text(i - w/2, nb + 0.5, f'{nb:.0f}%', ha='center', fontsize=9)
            ax.text(i + w/2, fb + 0.5, f'{fb:.0f}%', ha='center', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(atr_labels_plot)
        ax.legend()
    ax.set_title('控制ATR后：距60MA仍有区分力？', fontweight='bold')
    ax.set_ylabel('爆发率 (%)')
    ax.grid(True, alpha=0.3)

    # --- 图5-6: 挑ER低+远60MA的具体案例画K线 ---
    # 找爆发最大的几个信号
    er_40_vals = []
    for s in all_signals:
        # 重新算ER_40
        df_ref = s['_df_ref']
        idx = s['idx']
        if idx >= 40:
            net = abs(df_ref.iloc[idx]['close'] - df_ref.iloc[idx - 40]['close'])
            bar_sum = df_ref['close'].diff().abs().iloc[idx - 39: idx + 1].sum()
            s['er_40'] = net / bar_sum if bar_sum > 0 else 0
        else:
            s['er_40'] = np.nan

    er_p50 = np.percentile([s['er_40'] for s in all_signals if not np.isnan(s['er_40'])], 50)

    # 找案例：远60MA + 爆发的
    cases_far_burst = sorted(
        [s for s in all_signals if s['dist_60ma_pct'] > p50 and s['mfe'] >= BURST_THRESHOLD],
        key=lambda x: -x['mfe']
    )

    # 找案例：近60MA + 没爆发的
    cases_near_noburst = sorted(
        [s for s in all_signals if s['dist_60ma_pct'] <= p50 and s['mfe'] < 1.0],
        key=lambda x: x['mfe']
    )

    # 画2个案例
    case_sets = [
        (cases_far_burst[:3], '远60MA + 爆发案例'),
        (cases_near_noburst[:3], '近60MA + 未爆发案例'),
    ]

    for plot_idx, (cases, title) in enumerate(case_sets):
        ax = fig.add_subplot(3, 2, 5 + plot_idx)
        if not cases:
            ax.text(0.5, 0.5, 'No cases', transform=ax.transAxes, ha='center')
            ax.set_title(title)
            continue

        # 画第一个案例的K线
        case = cases[0]
        df_ref = case['_df_ref']
        idx = case['idx']
        start = max(0, idx - 60)
        end = min(len(df_ref), idx + MFE_WINDOW + 10)
        segment = df_ref.iloc[start:end]

        ax.plot(range(len(segment)), segment['close'].values, color='#333', linewidth=0.8)
        ax.plot(range(len(segment)), segment['ma_fast'].values, color='#2196f3', linewidth=0.6, alpha=0.7, label='10MA')
        ax.plot(range(len(segment)), segment['ma_60'].values, color='#f44336', linewidth=0.8, alpha=0.7, label='60MA')

        # 标注入场点
        entry_x = idx - start
        ax.axvline(x=entry_x, color='green', linewidth=1.5, linestyle='--', alpha=0.7)
        d_str = '多' if case['direction'] == 'long' else '空'
        ax.scatter([entry_x], [case['entry_price']], color='green', s=100, zorder=5, marker='^' if case['direction'] == 'long' else 'v')

        # 标注信息
        info = (f"{case['symbol']} {case['time'].strftime('%m-%d %H:%M')} {d_str}\n"
                f"MFE={case['mfe']:.2f}% MAE={case['mae']:.2f}%\n"
                f"dist60MA={case['dist_60ma_pct']:.2f}% ATR%={case['atr_pct']:.3f}%")
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.legend(fontsize=8, loc='lower right')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'ma_distance_deep.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图表已保存: {out_path}")

    # ============================================================
    # 打印案例详情
    # ============================================================
    print(f"\n{'='*100}")
    print(f"  典型案例：远60MA + 爆发 TOP5")
    print(f"{'='*100}")
    for i, case in enumerate(cases_far_burst[:5]):
        d_str = '多' if case['direction'] == 'long' else '空'
        er_str = f"{case['er_40']:.3f}" if not np.isnan(case.get('er_40', np.nan)) else 'N/A'
        print(f"  {i+1}. {case['symbol']} {case['time']} {d_str} | "
              f"MFE={case['mfe']:.2f}% MAE={case['mae']:.2f}% MFE/MAE={case['mfe_mae_ratio']:.2f} | "
              f"dist60MA={case['dist_60ma_pct']:.2f}% ATR%={case['atr_pct']:.3f}% ER40={er_str}")


if __name__ == '__main__':
    main()
