# -*- coding: utf-8 -*-
"""
MFE止盈时机分析：
1. MFE到达速度：爆发性行情多久到达峰值？
2. MFE到达后回撤多少？
3. 最优止盈点在哪？（不同止盈水平的收益对比）
"""

import os, sys, time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import timedelta
from tqsdk import TqApi, TqAuth
import config as cfg
from signal_core import SignalDetector, SYMBOL_CONFIGS
from data_cache import get_klines

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

SYMBOLS = {
    'SHFE.ag': {'tq': 'KQ.m@SHFE.ag', 'name': '白银'},
    'SHFE.au': {'tq': 'KQ.m@SHFE.au', 'name': '黄金'},
    'INE.sc':  {'tq': 'KQ.m@INE.sc',  'name': '原油'},
    'GFEX.lc': {'tq': 'KQ.m@GFEX.lc', 'name': '碳酸锂'},
    'DCE.lh':  {'tq': 'KQ.m@DCE.lh',  'name': '生猪'},
    'DCE.v':   {'tq': 'KQ.m@DCE.v',   'name': 'PVC'},
}

MAX_BARS = 72  # 观察窗口12小时
TP_LEVELS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]  # 止盈水平（%）


def get_mfe_path(df, idx, direction):
    """
    返回信号后每根K线的累计MFE路径
    和每根K线的当前浮盈
    """
    entry = df.iloc[idx]['close']
    mfe_path = []   # 到第N根时的累计MFE
    pnl_path = []   # 到第N根时的当前浮盈
    running_mfe = 0

    for j in range(1, MAX_BARS + 1):
        if idx + j >= len(df):
            break
        row = df.iloc[idx + j]
        if direction == 'short':
            bar_mfe = (entry - row['low']) / entry * 100
            bar_pnl = (entry - row['close']) / entry * 100
        else:
            bar_mfe = (row['high'] - entry) / entry * 100
            bar_pnl = (row['close'] - entry) / entry * 100

        running_mfe = max(running_mfe, bar_mfe)
        mfe_path.append(running_mfe)
        pnl_path.append(bar_pnl)

    return mfe_path, pnl_path


def simulate_tp(mfe_path, pnl_path, tp_level):
    """
    模拟固定止盈：MFE达到tp_level时平仓
    返回：是否触发止盈、触发时的bar序号、实际平仓盈亏
    """
    for i, mfe in enumerate(mfe_path):
        if mfe >= tp_level:
            # 触发止盈，用当根收盘价近似
            return True, i + 1, pnl_path[i]
    # 未触发，用最后一根的浮盈
    return False, len(pnl_path), pnl_path[-1] if pnl_path else 0


def main():
    api = TqApi(auth=TqAuth(cfg.TQ_ACCOUNT, cfg.TQ_PASSWORD))

    all_signals = []

    for sym_key, sym_info in SYMBOLS.items():
        print(f"加载 {sym_info['name']}...", end=' ')
        df = get_klines(api, sym_info['tq'], sym_key, period_min=10, days=170)

        # 计算60MA和距离
        df['ma60'] = df['close'].rolling(60).mean()
        df['dist_ma60'] = ((df['close'] - df['ma60']) / df['ma60'] * 100).abs()

        cutoff = df['datetime'].iloc[-1] - timedelta(days=120)
        df = df[df['datetime'] >= cutoff].copy().reset_index(drop=True)

        detector = SignalDetector(min_pb_bars=2)
        warmup = cfg.MA_SLOW + 5

        for i in range(warmup, len(df)):
            row = df.iloc[i]
            if pd.isna(row['ma_fast']) or pd.isna(row['ma_slow']):
                continue
            result = detector.process_bar(
                close=row['close'], high=row['high'], low=row['low'],
                ma_fast=row['ma_fast'], ma_slow=row['ma_slow'],
            )
            if result is None:
                continue

            mfe_path, pnl_path = get_mfe_path(df, i, result.direction)
            if not mfe_path:
                continue

            # MFE峰值和到达时间
            peak_mfe = max(mfe_path)
            peak_bar = mfe_path.index(peak_mfe) + 1  # 第几根到达峰值

            # 峰值后的回撤
            if peak_bar < len(pnl_path):
                post_peak_pnl = pnl_path[peak_bar:]  # 峰值后的浮盈序列
                min_post_pnl = min(post_peak_pnl) if post_peak_pnl else pnl_path[peak_bar - 1]
                drawdown = peak_mfe - min_post_pnl  # 从峰值回撤了多少
            else:
                drawdown = 0

            # 各止盈水平模拟
            tp_results = {}
            for tp in TP_LEVELS:
                hit, bar_n, realized = simulate_tp(mfe_path, pnl_path, tp)
                tp_results[tp] = {'hit': hit, 'bars': bar_n, 'realized': realized}

            dist = row['dist_ma60'] if not pd.isna(row['dist_ma60']) else 0

            all_signals.append({
                'symbol': sym_key, 'name': sym_info['name'],
                'time': row['datetime'], 'direction': result.direction,
                'entry': result.entry_price,
                'peak_mfe': peak_mfe, 'peak_bar': peak_bar,
                'drawdown': drawdown,
                'mfe_path': mfe_path, 'pnl_path': pnl_path,
                'tp': tp_results,
                'dist_ma60': dist,
                'is_far': dist >= 1.11,  # 距60MA"达标"
            })

        print(f"{sum(1 for s in all_signals if s['symbol']==sym_key)}信号")

    api.close()

    # ============================================================
    # 1. MFE到达速度
    # ============================================================
    print(f"\n\n{'='*80}")
    print(f"  MFE到达速度（信号后多久到达峰值？）")
    print(f"{'='*80}")

    # 只看爆发信号（peak_mfe >= 3%）
    boom = [s for s in all_signals if s['peak_mfe'] >= 3.0]
    non_boom = [s for s in all_signals if s['peak_mfe'] < 3.0]

    print(f"\n  爆发信号（MFE>=3%）共{len(boom)}个:")
    if boom:
        peak_bars = [s['peak_bar'] for s in boom]
        peak_hours = [b * 10 / 60 for b in peak_bars]
        print(f"    到达峰值时间: 平均={np.mean(peak_hours):.1f}h 中位={np.median(peak_hours):.1f}h "
              f"P25={np.percentile(peak_hours,25):.1f}h P75={np.percentile(peak_hours,75):.1f}h")
        print(f"    峰值大小: 平均={np.mean([s['peak_mfe'] for s in boom]):.2f}% "
              f"中位={np.median([s['peak_mfe'] for s in boom]):.2f}%")
        print(f"    峰值后回撤: 平均={np.mean([s['drawdown'] for s in boom]):.2f}% "
              f"中位={np.median([s['drawdown'] for s in boom]):.2f}%")

    # 分品种
    print(f"\n  分品种（爆发信号）:")
    print(f"  {'品种':<8} {'数量':>4} {'平均峰值':>8} {'到达(h)':>8} {'回撤':>8}")
    for sym_key, sym_info in SYMBOLS.items():
        sym_boom = [s for s in boom if s['symbol'] == sym_key]
        if not sym_boom:
            print(f"  {sym_info['name']:<8} {0:>4}")
            continue
        avg_peak = np.mean([s['peak_mfe'] for s in sym_boom])
        avg_time = np.mean([s['peak_bar'] * 10 / 60 for s in sym_boom])
        avg_dd = np.mean([s['drawdown'] for s in sym_boom])
        print(f"  {sym_info['name']:<8} {len(sym_boom):>4} {avg_peak:>7.2f}% {avg_time:>7.1f}h {avg_dd:>7.2f}%")

    # 距60MA远 vs 近
    far_boom = [s for s in boom if s['is_far']]
    near_boom = [s for s in boom if not s['is_far']]
    if far_boom and near_boom:
        print(f"\n  距60MA远(>{1.11}%) vs 近:")
        print(f"    远: {len(far_boom)}个 | 峰值={np.mean([s['peak_mfe'] for s in far_boom]):.2f}% | "
              f"到达={np.mean([s['peak_bar']*10/60 for s in far_boom]):.1f}h | "
              f"回撤={np.mean([s['drawdown'] for s in far_boom]):.2f}%")
        print(f"    近: {len(near_boom)}个 | 峰值={np.mean([s['peak_mfe'] for s in near_boom]):.2f}% | "
              f"到达={np.mean([s['peak_bar']*10/60 for s in near_boom]):.1f}h | "
              f"回撤={np.mean([s['drawdown'] for s in near_boom]):.2f}%")

    # ============================================================
    # 2. MFE时间曲线（平均路径）
    # ============================================================
    print(f"\n\n{'='*80}")
    print(f"  MFE平均路径（信号后每小时的累计MFE）")
    print(f"{'='*80}")

    hours = [1, 2, 3, 4, 6, 8, 10, 12]
    bar_indices = [int(h * 6) - 1 for h in hours]  # 10min bars

    print(f"  {'品种':<8}", end='')
    for h in hours:
        print(f" {h:>4}h", end='')
    print()

    for sym_key, sym_info in SYMBOLS.items():
        sym_sigs = [s for s in all_signals if s['symbol'] == sym_key]
        if not sym_sigs:
            continue
        print(f"  {sym_info['name']:<8}", end='')
        for bi in bar_indices:
            vals = [s['mfe_path'][bi] for s in sym_sigs if len(s['mfe_path']) > bi]
            if vals:
                print(f" {np.mean(vals):>4.1f}%", end='')
            else:
                print(f"   N/A", end='')
        print()

    # ============================================================
    # 3. 止盈水平对比
    # ============================================================
    print(f"\n\n{'='*80}")
    print(f"  止盈水平对比：不同止盈目标的效果")
    print(f"{'='*80}")

    print(f"\n  全品种汇总:")
    print(f"  {'止盈':>6} {'触发率':>7} {'平均等待':>8} {'触发时盈':>8} {'未触发盈':>8} {'综合期望':>8}")
    for tp in TP_LEVELS:
        hits = [s for s in all_signals if s['tp'][tp]['hit']]
        misses = [s for s in all_signals if not s['tp'][tp]['hit']]
        hit_rate = len(hits) / len(all_signals) * 100
        avg_wait = np.mean([s['tp'][tp]['bars'] * 10 / 60 for s in hits]) if hits else 0
        avg_hit_pnl = np.mean([s['tp'][tp]['realized'] for s in hits]) if hits else 0
        avg_miss_pnl = np.mean([s['tp'][tp]['realized'] for s in misses]) if misses else 0
        # 综合期望 = 触发率*触发盈亏 + (1-触发率)*未触发盈亏
        expected = hit_rate/100 * avg_hit_pnl + (1 - hit_rate/100) * avg_miss_pnl
        print(f"  {tp:>5.1f}% {hit_rate:>6.1f}% {avg_wait:>7.1f}h {avg_hit_pnl:>7.2f}% {avg_miss_pnl:>7.2f}% {expected:>7.2f}%")

    # 分品种最优止盈
    print(f"\n  分品种最优止盈:")
    print(f"  {'品种':<8} {'最优止盈':>8} {'触发率':>7} {'综合期望':>8} {'平均等待':>8}")
    for sym_key, sym_info in SYMBOLS.items():
        sym_sigs = [s for s in all_signals if s['symbol'] == sym_key]
        if not sym_sigs:
            continue
        best_tp = None
        best_ev = -999
        for tp in TP_LEVELS:
            hits = [s for s in sym_sigs if s['tp'][tp]['hit']]
            misses = [s for s in sym_sigs if not s['tp'][tp]['hit']]
            hr = len(hits) / len(sym_sigs)
            h_pnl = np.mean([s['tp'][tp]['realized'] for s in hits]) if hits else 0
            m_pnl = np.mean([s['tp'][tp]['realized'] for s in misses]) if misses else 0
            ev = hr * h_pnl + (1 - hr) * m_pnl
            if ev > best_ev:
                best_ev = ev
                best_tp = tp
                best_hr = hr * 100
                best_wait = np.mean([s['tp'][tp]['bars']*10/60 for s in hits]) if hits else 0

        print(f"  {sym_info['name']:<8} {best_tp:>7.1f}% {best_hr:>6.1f}% {best_ev:>7.2f}% {best_wait:>7.1f}h")

    # ============================================================
    # 4. 距60MA远的信号：止盈对比
    # ============================================================
    far_sigs = [s for s in all_signals if s['is_far']]
    print(f"\n\n{'='*80}")
    print(f"  距60MA远(>1.11%)的信号止盈对比 ({len(far_sigs)}个)")
    print(f"{'='*80}")
    print(f"  {'止盈':>6} {'触发率':>7} {'平均等待':>8} {'综合期望':>8}")
    for tp in TP_LEVELS:
        hits = [s for s in far_sigs if s['tp'][tp]['hit']]
        misses = [s for s in far_sigs if not s['tp'][tp]['hit']]
        hr = len(hits) / len(far_sigs) * 100
        avg_wait = np.mean([s['tp'][tp]['bars']*10/60 for s in hits]) if hits else 0
        h_pnl = np.mean([s['tp'][tp]['realized'] for s in hits]) if hits else 0
        m_pnl = np.mean([s['tp'][tp]['realized'] for s in misses]) if misses else 0
        ev = hr/100 * h_pnl + (1-hr/100) * m_pnl
        print(f"  {tp:>5.1f}% {hr:>6.1f}% {avg_wait:>7.1f}h {ev:>7.2f}%")

    # 分品种
    print(f"\n  分品种（距60MA远）最优止盈:")
    for sym_key, sym_info in SYMBOLS.items():
        sym_far = [s for s in far_sigs if s['symbol'] == sym_key]
        if len(sym_far) < 5:
            continue
        best_tp = None
        best_ev = -999
        for tp in TP_LEVELS:
            hits = [s for s in sym_far if s['tp'][tp]['hit']]
            misses = [s for s in sym_far if not s['tp'][tp]['hit']]
            hr = len(hits) / len(sym_far)
            h_pnl = np.mean([s['tp'][tp]['realized'] for s in hits]) if hits else 0
            m_pnl = np.mean([s['tp'][tp]['realized'] for s in misses]) if misses else 0
            ev = hr * h_pnl + (1 - hr) * m_pnl
            if ev > best_ev:
                best_ev = ev
                best_tp = tp
                best_hr = hr * 100
                best_wait = np.mean([s['tp'][tp]['bars']*10/60 for s in hits]) if hits else 0

        print(f"  {sym_info['name']:<8} 最优止盈={best_tp:.1f}% 触发率={best_hr:.1f}% 期望={best_ev:.2f}% 等待={best_wait:.1f}h")

    # ============================================================
    # 图表
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 图1: 各品种MFE平均路径
    ax = axes[0, 0]
    colors_map = {'白银': '#2196F3', '黄金': '#FFC107', '原油': '#333',
                  '碳酸锂': '#4CAF50', '生猪': '#f44336', 'PVC': '#9C27B0'}
    for sym_key, sym_info in SYMBOLS.items():
        sym_sigs = [s for s in all_signals if s['symbol'] == sym_key]
        if not sym_sigs:
            continue
        max_len = max(len(s['mfe_path']) for s in sym_sigs)
        avg_path = []
        for j in range(min(max_len, MAX_BARS)):
            vals = [s['mfe_path'][j] for s in sym_sigs if len(s['mfe_path']) > j]
            avg_path.append(np.mean(vals) if vals else 0)
        x_hours = [(j+1) * 10 / 60 for j in range(len(avg_path))]
        ax.plot(x_hours, avg_path, label=sym_info['name'],
               color=colors_map.get(sym_info['name'], '#999'), linewidth=1.5)
    ax.set_xlabel('小时')
    ax.set_ylabel('累计MFE %')
    ax.set_title('各品种平均MFE路径', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 图2: 爆发信号的峰值到达时间分布
    ax = axes[0, 1]
    if boom:
        peak_hours = [s['peak_bar'] * 10 / 60 for s in boom]
        ax.hist(peak_hours, bins=24, color='#4CAF50', edgecolor='white', alpha=0.7)
        ax.axvline(np.median(peak_hours), color='red', linestyle='--', label=f'中位={np.median(peak_hours):.1f}h')
        ax.axvline(np.mean(peak_hours), color='blue', linestyle='--', label=f'平均={np.mean(peak_hours):.1f}h')
    ax.set_xlabel('小时')
    ax.set_ylabel('信号数')
    ax.set_title('爆发信号峰值到达时间分布', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图3: 峰值后回撤分布
    ax = axes[0, 2]
    if boom:
        dds = [s['drawdown'] for s in boom]
        ax.hist(dds, bins=20, color='#f44336', edgecolor='white', alpha=0.7)
        ax.axvline(np.median(dds), color='blue', linestyle='--', label=f'中位={np.median(dds):.1f}%')
    ax.set_xlabel('回撤 %')
    ax.set_ylabel('信号数')
    ax.set_title('峰值后回撤分布（爆发信号）', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图4: 止盈水平 vs 综合期望（全品种）
    ax = axes[1, 0]
    evs_all = []
    evs_far = []
    for tp in TP_LEVELS:
        hits = [s for s in all_signals if s['tp'][tp]['hit']]
        misses = [s for s in all_signals if not s['tp'][tp]['hit']]
        hr = len(hits) / len(all_signals)
        h_pnl = np.mean([s['tp'][tp]['realized'] for s in hits]) if hits else 0
        m_pnl = np.mean([s['tp'][tp]['realized'] for s in misses]) if misses else 0
        evs_all.append(hr * h_pnl + (1-hr) * m_pnl)

        f_hits = [s for s in far_sigs if s['tp'][tp]['hit']]
        f_misses = [s for s in far_sigs if not s['tp'][tp]['hit']]
        f_hr = len(f_hits) / len(far_sigs) if far_sigs else 0
        f_h_pnl = np.mean([s['tp'][tp]['realized'] for s in f_hits]) if f_hits else 0
        f_m_pnl = np.mean([s['tp'][tp]['realized'] for s in f_misses]) if f_misses else 0
        evs_far.append(f_hr * f_h_pnl + (1-f_hr) * f_m_pnl)

    ax.plot(TP_LEVELS, evs_all, 'o-', color='#999', label='全部信号', linewidth=2)
    ax.plot(TP_LEVELS, evs_far, 's-', color='#4CAF50', label='距60MA远', linewidth=2)
    ax.set_xlabel('止盈目标 %')
    ax.set_ylabel('综合期望收益 %')
    ax.set_title('止盈水平 vs 期望收益', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图5: 分品种最优止盈热力图
    ax = axes[1, 1]
    sym_names = [SYMBOLS[k]['name'] for k in SYMBOLS]
    ev_matrix = np.zeros((len(SYMBOLS), len(TP_LEVELS)))
    for i, sym_key in enumerate(SYMBOLS):
        sym_sigs = [s for s in all_signals if s['symbol'] == sym_key]
        for j, tp in enumerate(TP_LEVELS):
            hits = [s for s in sym_sigs if s['tp'][tp]['hit']]
            misses = [s for s in sym_sigs if not s['tp'][tp]['hit']]
            if not sym_sigs:
                continue
            hr = len(hits) / len(sym_sigs)
            h_pnl = np.mean([s['tp'][tp]['realized'] for s in hits]) if hits else 0
            m_pnl = np.mean([s['tp'][tp]['realized'] for s in misses]) if misses else 0
            ev_matrix[i, j] = hr * h_pnl + (1-hr) * m_pnl

    im = ax.imshow(ev_matrix, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(TP_LEVELS)))
    ax.set_xticklabels([f'{tp}%' for tp in TP_LEVELS])
    ax.set_yticks(range(len(sym_names)))
    ax.set_yticklabels(sym_names)
    for i in range(len(sym_names)):
        for j in range(len(TP_LEVELS)):
            ax.text(j, i, f'{ev_matrix[i,j]:.2f}', ha='center', va='center', fontsize=7,
                   color='white' if ev_matrix[i,j] > 1.5 else 'black')
    ax.set_title('品种 x 止盈水平 期望收益(%)', fontweight='bold')
    plt.colorbar(im, ax=ax)

    # 图6: 距60MA远 MFE路径 vs 近
    ax = axes[1, 2]
    for label, sigs, color in [('远(>1.11%)', far_sigs, '#4CAF50'),
                                ('近', [s for s in all_signals if not s['is_far']], '#f44336')]:
        if not sigs:
            continue
        max_len = max(len(s['mfe_path']) for s in sigs)
        avg_path = []
        for j in range(min(max_len, MAX_BARS)):
            vals = [s['mfe_path'][j] for s in sigs if len(s['mfe_path']) > j]
            avg_path.append(np.mean(vals) if vals else 0)
        x_hours = [(j+1)*10/60 for j in range(len(avg_path))]
        ax.plot(x_hours, avg_path, label=label, color=color, linewidth=2)
    ax.set_xlabel('小时')
    ax.set_ylabel('累计MFE %')
    ax.set_title('距60MA远 vs 近 — MFE路径', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'output', 'mfe_timing.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图表: {out_path}")


if __name__ == '__main__':
    main()
