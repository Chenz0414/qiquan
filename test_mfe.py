# -*- coding: utf-8 -*-
"""
期权视角回测：MFE（最大有利波动）分析
——信号出现后，价格朝有利方向最多跑了多远？
——强趋势期的信号是否产生"爆发性"行情？
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

# ============================================================
# 测试品种
# ============================================================
SYMBOLS = {
    'DCE.lh':  {'tq': 'KQ.m@DCE.lh',  'name': '生猪'},
    'SHFE.ag': {'tq': 'KQ.m@SHFE.ag', 'name': '白银'},
    'SHFE.au': {'tq': 'KQ.m@SHFE.au', 'name': '黄金'},
    'INE.sc':  {'tq': 'KQ.m@INE.sc',  'name': '原油'},
    'DCE.v':   {'tq': 'KQ.m@DCE.v',   'name': 'PVC'},
    'GFEX.lc': {'tq': 'KQ.m@GFEX.lc', 'name': '碳酸锂'},
}

# MFE观察窗口：信号后看多少根K线
MFE_WINDOWS = [6, 12, 24, 48, 72]  # 10min: 1h/2h/4h/8h/12h


def classify_phase(df):
    """给df每根K线标注阶段（strong/normal/chop）和趋势方向"""
    ma10 = df['ma_fast']
    ma20 = df['ma_slow']
    atr = df['high'].rolling(cfg.ATR_PERIOD).max() - df['low'].rolling(cfg.ATR_PERIOD).min()
    atr = atr / cfg.ATR_PERIOD  # 简化ATR
    atr = atr.replace(0, np.nan)

    ma_slope = (ma10 - ma10.shift(5)) / atr
    ma_width = (ma10 - ma20) / atr

    # 百分位阈值
    abs_slope = ma_slope.abs().dropna()
    abs_width = ma_width.abs().dropna()
    s80 = abs_slope.quantile(0.80)
    s40 = abs_slope.quantile(0.40)
    w80 = abs_width.quantile(0.80)
    w40 = abs_width.quantile(0.40)

    phase = []
    trend_dir = []
    for i in range(len(df)):
        s = abs(ma_slope.iloc[i]) if not pd.isna(ma_slope.iloc[i]) else 0
        w = abs(ma_width.iloc[i]) if not pd.isna(ma_width.iloc[i]) else 0
        wd = ma_width.iloc[i] if not pd.isna(ma_width.iloc[i]) else 0

        if s >= s80 and w >= w80:
            phase.append('strong')
        elif s >= s40 or w >= w40:
            phase.append('normal')
        else:
            phase.append('chop')
        trend_dir.append('up' if wd > 0 else 'down')

    df['phase'] = phase
    df['trend_dir'] = trend_dir
    df['ma_slope'] = ma_slope
    df['ma_width'] = ma_width
    return df


def calc_mfe(df, sig_idx, direction, windows):
    """
    计算信号后各窗口内的MFE（最大有利偏移%）和MAE（最大不利偏移%）
    MFE: 价格朝有利方向最远跑了多少%
    MAE: 价格朝不利方向最远跑了多少%
    """
    entry_price = df.iloc[sig_idx]['close']
    results = {}

    for w in windows:
        end_idx = min(sig_idx + w, len(df) - 1)
        if sig_idx + 1 > end_idx:
            results[w] = {'mfe': 0, 'mae': 0, 'end_pnl': 0}
            continue

        segment = df.iloc[sig_idx + 1: end_idx + 1]

        if direction == 'short':
            # 做空：价格下跌有利
            mfe = (entry_price - segment['low'].min()) / entry_price * 100
            mae = (segment['high'].max() - entry_price) / entry_price * 100
            end_pnl = (entry_price - segment.iloc[-1]['close']) / entry_price * 100
        else:
            # 做多：价格上涨有利
            mfe = (segment['high'].max() - entry_price) / entry_price * 100
            mae = (entry_price - segment['low'].min()) / entry_price * 100
            end_pnl = (segment.iloc[-1]['close'] - entry_price) / entry_price * 100

        results[w] = {'mfe': max(mfe, 0), 'mae': max(mae, 0), 'end_pnl': end_pnl}

    return results


def main():
    api = TqApi(auth=TqAuth(cfg.TQ_ACCOUNT, cfg.TQ_PASSWORD))

    all_results = {}

    for sym_key, sym_info in SYMBOLS.items():
        print(f"\n{'='*60}")
        print(f"  {sym_info['name']} ({sym_key})")
        print(f"{'='*60}")

        df = get_klines(api, sym_info['tq'], sym_key, period_min=10, days=170)
        print(f"  数据: {len(df)}根, {df['datetime'].iloc[0].date()} ~ {df['datetime'].iloc[-1].date()}")

        # 阶段分类
        df = classify_phase(df)

        # 120天窗口
        cutoff = df['datetime'].iloc[-1] - timedelta(days=120)
        df_120 = df[df['datetime'] >= cutoff].copy().reset_index(drop=True)

        # 信号检测（>=2根，最宽松）
        detector = SignalDetector(min_pb_bars=2)
        warmup = cfg.MA_SLOW + 5

        signals = []
        for i in range(warmup, len(df_120)):
            row = df_120.iloc[i]
            if pd.isna(row['ma_fast']) or pd.isna(row['ma_slow']):
                continue
            result = detector.process_bar(
                close=row['close'], high=row['high'], low=row['low'],
                ma_fast=row['ma_fast'], ma_slow=row['ma_slow'],
            )
            if result is not None:
                signals.append({
                    'idx': i,
                    'time': row['datetime'],
                    'direction': result.direction,
                    'entry_price': result.entry_price,
                    'pb_bars': result.pullback_bars,
                    'phase': row['phase'],
                    'trend_dir': row['trend_dir'],
                    'with_trend': (result.direction == 'short' and row['trend_dir'] == 'down') or
                                  (result.direction == 'long' and row['trend_dir'] == 'up'),
                })

        # 计算每个信号的MFE
        for sig in signals:
            mfe_data = calc_mfe(df_120, sig['idx'], sig['direction'], MFE_WINDOWS)
            sig['mfe'] = mfe_data

        all_results[sym_key] = {
            'name': sym_info['name'],
            'signals': signals,
            'df': df_120,
        }

        # ---- 按阶段汇总MFE ----
        phase_names = ['strong', 'normal', 'chop']
        print(f"\n  {'阶段':<8} {'信号':>4} {'顺势':>4} | ", end='')
        for w in MFE_WINDOWS:
            print(f"  MFE_{w}根", end='')
        print()
        print(f"  {'-'*75}")

        for phase in phase_names:
            phase_sigs = [s for s in signals if s['phase'] == phase]
            trend_sigs = [s for s in phase_sigs if s['with_trend']]
            n = len(phase_sigs)
            nt = len(trend_sigs)

            print(f"  {phase:<8} {n:>4} {nt:>4} | ", end='')
            for w in MFE_WINDOWS:
                if phase_sigs:
                    avg_mfe = np.mean([s['mfe'][w]['mfe'] for s in phase_sigs])
                    print(f"  {avg_mfe:>6.2f}%", end='')
                else:
                    print(f"  {'N/A':>7}", end='')
            print()

            # 顺势子集
            if trend_sigs and len(trend_sigs) != len(phase_sigs):
                print(f"    ↳顺势 {nt:>4}      | ", end='')
                for w in MFE_WINDOWS:
                    avg_mfe = np.mean([s['mfe'][w]['mfe'] for s in trend_sigs])
                    print(f"  {avg_mfe:>6.2f}%", end='')
                print()

        # ---- 爆发性信号统计 ----
        print(f"\n  爆发性信号（MFE_48根 > 2%）:")
        big_moves = [s for s in signals if s['mfe'][48]['mfe'] > 2.0]
        if big_moves:
            for s in sorted(big_moves, key=lambda x: -x['mfe'][48]['mfe']):
                d = "空" if s['direction'] == 'short' else "多"
                t = "顺" if s['with_trend'] else "逆"
                print(f"    {s['time']} {d}{t} | {s['phase']:<6} | "
                      f"MFE: 2h={s['mfe'][12]['mfe']:.2f}% 4h={s['mfe'][24]['mfe']:.2f}% "
                      f"8h={s['mfe'][48]['mfe']:.2f}% 12h={s['mfe'][72]['mfe']:.2f}%")
        else:
            big_moves = [s for s in signals if s['mfe'][48]['mfe'] > 1.0]
            print(f"    (无>2%信号，降低到>1%:)")
            for s in sorted(big_moves, key=lambda x: -x['mfe'][48]['mfe'])[:8]:
                d = "空" if s['direction'] == 'short' else "多"
                t = "顺" if s['with_trend'] else "逆"
                print(f"    {s['time']} {d}{t} | {s['phase']:<6} | "
                      f"MFE: 2h={s['mfe'][12]['mfe']:.2f}% 4h={s['mfe'][24]['mfe']:.2f}% "
                      f"8h={s['mfe'][48]['mfe']:.2f}% 12h={s['mfe'][72]['mfe']:.2f}%")

        # ---- MFE分布：强趋势顺势 vs 其他 ----
        strong_trend = [s for s in signals if s['phase'] == 'strong' and s['with_trend']]
        others = [s for s in signals if s['phase'] != 'strong' or not s['with_trend']]
        if strong_trend:
            st_mfe48 = [s['mfe'][48]['mfe'] for s in strong_trend]
            ot_mfe48 = [s['mfe'][48]['mfe'] for s in others] if others else [0]
            print(f"\n  强趋势顺势 vs 其他（MFE_48根=8小时）:")
            print(f"    强趋势顺势: N={len(st_mfe48):>3} | 平均={np.mean(st_mfe48):.2f}% | "
                  f"中位={np.median(st_mfe48):.2f}% | P75={np.percentile(st_mfe48, 75):.2f}% | "
                  f"最大={max(st_mfe48):.2f}%")
            print(f"    其他信号:   N={len(ot_mfe48):>3} | 平均={np.mean(ot_mfe48):.2f}% | "
                  f"中位={np.median(ot_mfe48):.2f}% | P75={np.percentile(ot_mfe48, 75):.2f}% | "
                  f"最大={max(ot_mfe48):.2f}%")

    api.close()

    # ============================================================
    # 汇总表
    # ============================================================
    print(f"\n\n{'='*80}")
    print(f"  汇  总：各品种 强趋势顺势 MFE（期权视角）")
    print(f"{'='*80}")
    print(f"  {'品种':<8} {'信号':>4} | {'MFE_1h':>7} {'MFE_2h':>7} {'MFE_4h':>7} {'MFE_8h':>7} {'MFE_12h':>8} | {'爆发>2%':>6} {'爆发>1%':>6}")
    print(f"  {'-'*85}")

    for sym_key, data in all_results.items():
        signals = data['signals']
        strong_trend = [s for s in signals if s['phase'] == 'strong' and s['with_trend']]
        n = len(strong_trend)
        if n == 0:
            print(f"  {data['name']:<8} {n:>4} |  {'-- 无强趋势顺势信号 --'}")
            continue

        boom2 = sum(1 for s in strong_trend if s['mfe'][48]['mfe'] > 2.0)
        boom1 = sum(1 for s in strong_trend if s['mfe'][48]['mfe'] > 1.0)
        avgs = []
        for w in MFE_WINDOWS:
            avgs.append(np.mean([s['mfe'][w]['mfe'] for s in strong_trend]))

        print(f"  {data['name']:<8} {n:>4} | {avgs[0]:>6.2f}% {avgs[1]:>6.2f}% {avgs[2]:>6.2f}% "
              f"{avgs[3]:>6.2f}% {avgs[4]:>7.2f}% | {boom2:>5}个 {boom1:>5}个")

    # 对比：全部信号的MFE vs 强趋势顺势的MFE
    print(f"\n  {'品种':<8} | {'全部信号 MFE_8h':>15} | {'强趋势顺势 MFE_8h':>18} | {'倍数':>5}")
    print(f"  {'-'*65}")
    for sym_key, data in all_results.items():
        signals = data['signals']
        strong_trend = [s for s in signals if s['phase'] == 'strong' and s['with_trend']]
        all_mfe = np.mean([s['mfe'][48]['mfe'] for s in signals]) if signals else 0
        st_mfe = np.mean([s['mfe'][48]['mfe'] for s in strong_trend]) if strong_trend else 0
        ratio = st_mfe / all_mfe if all_mfe > 0 else 0
        print(f"  {data['name']:<8} | {all_mfe:>14.2f}% | {st_mfe:>17.2f}% | {ratio:>4.1f}x")

    # ============================================================
    # 图表
    # ============================================================
    n_sym = len(all_results)
    fig, axes = plt.subplots(n_sym, 2, figsize=(20, 5 * n_sym))
    if n_sym == 1:
        axes = axes.reshape(1, -1)

    for row_idx, (sym_key, data) in enumerate(all_results.items()):
        signals = data['signals']
        df_plot = data['df']

        # 左图：MFE箱线图 按阶段
        ax = axes[row_idx, 0]
        phase_data = {}
        for phase in ['strong', 'normal', 'chop']:
            phase_sigs = [s for s in signals if s['phase'] == phase and s['with_trend']]
            if phase_sigs:
                phase_data[phase] = [s['mfe'][48]['mfe'] for s in phase_sigs]

        if phase_data:
            labels = list(phase_data.keys())
            box_data = [phase_data[l] for l in labels]
            colors = {'strong': '#4CAF50', 'normal': '#FFC107', 'chop': '#f44336'}
            bp = ax.boxplot(box_data, labels=[f"{l}\n(N={len(d)})" for l, d in zip(labels, box_data)],
                           patch_artist=True, showmeans=True)
            for patch, label in zip(bp['boxes'], labels):
                patch.set_facecolor(colors.get(label, '#999'))
                patch.set_alpha(0.6)
        ax.set_title(f"{data['name']} — MFE_8h分布（顺势信号）", fontweight='bold')
        ax.set_ylabel('MFE %')
        ax.grid(True, alpha=0.3)

        # 右图：价格+阶段背景+爆发信号标注
        ax = axes[row_idx, 1]
        # 画价格
        ax.plot(df_plot['datetime'], df_plot['close'], color='#333', linewidth=0.6)

        # 阶段背景色
        phase_colors = {'strong': '#4CAF5020', 'normal': '#FFC10710', 'chop': '#f4433610'}
        prev_phase = None
        start_dt = None
        for i in range(len(df_plot)):
            curr_phase = df_plot.iloc[i]['phase']
            if curr_phase != prev_phase:
                if prev_phase is not None and start_dt is not None:
                    ax.axvspan(start_dt, df_plot.iloc[i]['datetime'],
                              color=phase_colors.get(prev_phase, '#00000005'), linewidth=0)
                start_dt = df_plot.iloc[i]['datetime']
                prev_phase = curr_phase
        # 最后一段
        if prev_phase and start_dt:
            ax.axvspan(start_dt, df_plot.iloc[-1]['datetime'],
                      color=phase_colors.get(prev_phase, '#00000005'), linewidth=0)

        # 标注爆发信号（MFE_48 > 1%）
        big = [s for s in signals if s['mfe'][48]['mfe'] > 1.0 and s['with_trend']]
        for s in big:
            color = '#f44336' if s['direction'] == 'short' else '#2196f3'
            marker = 'v' if s['direction'] == 'short' else '^'
            ax.scatter(s['time'], s['entry_price'], color=color, marker=marker, s=80, zorder=5)
            ax.annotate(f"{s['mfe'][48]['mfe']:.1f}%",
                       xy=(s['time'], s['entry_price']), fontsize=7,
                       textcoords="offset points", xytext=(5, -15 if s['direction'] == 'short' else 10),
                       color=color, fontweight='bold')

        ax.set_title(f"{data['name']} — 爆发信号标注（MFE>1%，绿色背景=强趋势）", fontweight='bold')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'output', 'mfe_analysis.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图表已保存: {out_path}")


if __name__ == '__main__':
    main()
