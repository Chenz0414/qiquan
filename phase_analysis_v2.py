# -*- coding: utf-8 -*-
"""
市场阶段分类 v2 — 分位数自适应阈值
===================================
v1问题：固定阈值太严，45%的K线被归为"震荡"但其实是正常趋势。
v2改进：先统计ma_slope/ma_width的实际分布，用分位数定阈值。
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
from signal_core import SignalDetector, ExitTracker, SYMBOL_CONFIGS
from data_cache import get_klines

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

SYMBOLS = [
    ("KQ.m@SHFE.au", "SHFE.au", "黄金"),
    ("KQ.m@SHFE.ag", "SHFE.ag", "白银"),
    ("KQ.m@INE.sc",  "INE.sc",  "原油"),
    ("KQ.m@DCE.v",   "DCE.v",   "PVC"),
    ("KQ.m@GFEX.lc", "GFEX.lc", "碳酸锂"),
    ("KQ.m@CZCE.MA", "CZCE.MA", "甲醇"),
    ("KQ.m@CZCE.TA", "CZCE.TA", "PTA"),
    ("KQ.m@DCE.lh",  "DCE.lh",  "生猪"),
]

PERIOD_MIN = 10
SCAN_DAYS = 170


def compute_atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_indicators(df):
    """计算阶段指标（不分类，只计算原始值）"""
    df = df.copy()
    df['atr'] = compute_atr(df, 14)
    df['ma_slope'] = (df['ma_fast'] - df['ma_fast'].shift(5)) / df['atr']
    df['ma_width'] = (df['ma_fast'] - df['ma_slow']) / df['atr']
    # 趋势方向
    df['trend_dir'] = np.where(df['ma_width'] > 0, 'up', 'down')
    return df


def classify_by_percentile(df, slope_strong_pct=80, slope_chop_pct=40):
    """用分位数分类阶段"""
    df = df.copy()
    abs_slope = df['ma_slope'].abs()
    abs_width = df['ma_width'].abs()

    # 只用有效值算分位数
    valid = abs_slope.dropna()
    if len(valid) == 0:
        df['phase'] = 'unknown'
        return df

    slope_strong_th = np.percentile(valid, slope_strong_pct)
    slope_chop_th = np.percentile(valid, slope_chop_pct)

    valid_w = abs_width.dropna()
    width_strong_th = np.percentile(valid_w, slope_strong_pct)
    width_chop_th = np.percentile(valid_w, slope_chop_pct)

    def classify(row):
        if pd.isna(row['ma_slope']) or pd.isna(row['ma_width']):
            return 'unknown'
        s = abs(row['ma_slope'])
        w = abs(row['ma_width'])
        if s >= slope_strong_th and w >= width_strong_th:
            return 'strong'
        elif s >= slope_chop_th or w >= width_chop_th:
            return 'normal'
        else:
            return 'chop'

    df['phase'] = df.apply(classify, axis=1)
    return df, {
        'slope_strong': slope_strong_th, 'slope_chop': slope_chop_th,
        'width_strong': width_strong_th, 'width_chop': width_chop_th,
    }


def run_backtest_all(df, symbol_key):
    """>=2根全量信号 + 双止损"""
    sym_cfg = SYMBOL_CONFIGS.get(symbol_key, {"tick_size": 1.0})
    tick_size = sym_cfg["tick_size"]
    detector = SignalDetector(min_pb_bars=2)
    warmup = max(cfg.MA_SLOW, 20) + 5

    signals = []
    for i in range(warmup, len(df)):
        row = df.iloc[i]
        if pd.isna(row['ma_fast']) or pd.isna(row['ma_slow']):
            continue
        result = detector.process_bar(
            close=row['close'], high=row['high'], low=row['low'],
            ma_fast=row['ma_fast'], ma_slow=row['ma_slow'],
        )
        if result is not None:
            signals.append({'idx': i, 'signal': result,
                           'phase': row['phase'], 'trend_dir': row['trend_dir'],
                           'ma_slope': row['ma_slope']})

    cutoff = df['datetime'].iloc[-1] - timedelta(days=120)
    signals = [s for s in signals if df.iloc[s['idx']]['datetime'] >= cutoff]

    trades = []
    for sig_info in signals:
        idx = sig_info['idx']
        sig = sig_info['signal']
        for stop_ticks in [3, 5]:
            tracker = ExitTracker(
                direction=sig.direction, entry_price=sig.entry_price,
                pullback_extreme=sig.pullback_extreme,
                tick_size=tick_size, stop_ticks=stop_ticks,
            )
            s1_exit = s2_exit = None
            for j in range(idx + 1, len(df)):
                bar = df.iloc[j]
                prev_bar = df.iloc[j - 1]
                if pd.isna(bar['ma_fast']):
                    continue
                exits, _ = tracker.process_bar(
                    close=bar['close'], high=bar['high'], low=bar['low'],
                    ma_fast=bar['ma_fast'], prev_close=prev_bar['close'],
                )
                for ev in exits:
                    if ev.strategy == 'S1' and s1_exit is None:
                        s1_exit = ev
                    elif ev.strategy == 'S2' and s2_exit is None:
                        s2_exit = ev
                if tracker.all_done():
                    break
            if s1_exit is None or s2_exit is None:
                forced = tracker.force_close(df.iloc[-1]['close'])
                for ev in forced:
                    if ev.strategy == 'S1' and s1_exit is None:
                        s1_exit = ev
                    if ev.strategy == 'S2' and s2_exit is None:
                        s2_exit = ev
            trades.append({
                'time': df.iloc[idx]['datetime'],
                'direction': sig.direction,
                'pb_bars': sig.pullback_bars,
                'entry_price': sig.entry_price,
                'phase': sig_info['phase'],
                'trend_dir': sig_info['trend_dir'],
                'ma_slope': sig_info['ma_slope'],
                'stop_ticks': stop_ticks,
                's1_pnl': s1_exit.pnl_pct if s1_exit else 0,
                's2_pnl': s2_exit.pnl_pct if s2_exit else 0,
                's1_reason': s1_exit.exit_reason if s1_exit else 'open',
                's2_reason': s2_exit.exit_reason if s2_exit else 'open',
            })
    return trades


def filter_fixed(trades):
    return [t for t in trades if t['pb_bars'] >= 4 and t['stop_ticks'] == 5]


def filter_adaptive(trades):
    result = []
    for t in trades:
        phase = t['phase']
        if phase == 'strong':
            if t['stop_ticks'] != 3 or t['pb_bars'] < 2:
                continue
            is_wt = ((t['direction'] == 'long' and t['trend_dir'] == 'up') or
                     (t['direction'] == 'short' and t['trend_dir'] == 'down'))
            if not is_wt:
                continue
            result.append(t)
        elif phase == 'normal':
            if t['stop_ticks'] != 5 or t['pb_bars'] < 4:
                continue
            result.append(t)
        # chop: skip
    return result


def filter_adaptive_v2(trades):
    """v2: 震荡期也做，但要求>=5根+只做顺势"""
    result = []
    for t in trades:
        phase = t['phase']
        if phase == 'strong':
            if t['stop_ticks'] != 3 or t['pb_bars'] < 2:
                continue
            is_wt = ((t['direction'] == 'long' and t['trend_dir'] == 'up') or
                     (t['direction'] == 'short' and t['trend_dir'] == 'down'))
            if not is_wt:
                continue
            result.append(t)
        elif phase == 'normal':
            if t['stop_ticks'] != 5 or t['pb_bars'] < 4:
                continue
            result.append(t)
        elif phase == 'chop':
            # 震荡期：>=5根、5ticks、只做顺势
            if t['stop_ticks'] != 5 or t['pb_bars'] < 5:
                continue
            is_wt = ((t['direction'] == 'long' and t['trend_dir'] == 'up') or
                     (t['direction'] == 'short' and t['trend_dir'] == 'down'))
            if not is_wt:
                continue
            result.append(t)
    return result


def calc_stats(pnls):
    if not pnls:
        return None
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls) * 100
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0.001
    pf = avg_win / avg_loss if avg_loss > 0 else 999
    ev = win_rate / 100 * pf - (1 - win_rate / 100)
    total = sum(pnls)
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    max_dd = (cum - peak).min()
    return {'count': len(pnls), 'win_rate': win_rate, 'pf': pf, 'ev': ev,
            'total': total, 'max_dd': max_dd, 'cum': cum}


def main():
    print("=" * 70)
    print("  阶段自适应策略 v2 — 分位数阈值")
    print("=" * 70)

    api = TqApi(auth=TqAuth(cfg.TQ_ACCOUNT, cfg.TQ_PASSWORD))

    # Step 1: 先拉所有数据，看指标分布
    all_dfs = {}
    for tq_sym, sym_key, name in SYMBOLS:
        print(f"  加载 {name}...")
        df = get_klines(api, tq_sym, sym_key, period_min=PERIOD_MIN, days=SCAN_DAYS)
        df = compute_indicators(df)
        all_dfs[name] = (df, sym_key)

    api.close()

    # Step 2: 统计所有品种的 ma_slope 分布
    print(f"\n{'='*70}")
    print(f"  Step 1: 指标分布分析")
    print(f"{'='*70}")
    all_slopes = []
    all_widths = []
    for name, (df, _) in all_dfs.items():
        cutoff = df['datetime'].iloc[-1] - timedelta(days=120)
        recent = df[df['datetime'] >= cutoff]
        slopes = recent['ma_slope'].dropna().abs()
        widths = recent['ma_width'].dropna().abs()
        all_slopes.extend(slopes.tolist())
        all_widths.extend(widths.tolist())
        print(f"  {name:<8} ma_slope: P25={slopes.quantile(0.25):.3f} P50={slopes.quantile(0.5):.3f} "
              f"P75={slopes.quantile(0.75):.3f} P90={slopes.quantile(0.9):.3f} P95={slopes.quantile(0.95):.3f}")

    all_slopes = np.array(all_slopes)
    all_widths = np.array(all_widths)
    print(f"\n  全品种汇总:")
    for p in [25, 40, 50, 60, 75, 80, 90, 95]:
        print(f"    P{p}: slope={np.percentile(all_slopes, p):.3f}  width={np.percentile(all_widths, p):.3f}")

    # Step 3: 用分位数分类
    print(f"\n{'='*70}")
    print(f"  Step 2: 分位数分类 (strong=P80, chop=P40)")
    print(f"{'='*70}")

    all_results = {}

    for name, (df, sym_key) in all_dfs.items():
        # 每个品种用自己的分位数
        df, thresholds = classify_by_percentile(df, slope_strong_pct=80, slope_chop_pct=40)

        cutoff = df['datetime'].iloc[-1] - timedelta(days=120)
        recent = df[df['datetime'] >= cutoff]
        phase_dist = recent['phase'].value_counts(normalize=True)
        print(f"\n  {name}: 强{phase_dist.get('strong',0)*100:.0f}% "
              f"正常{phase_dist.get('normal',0)*100:.0f}% "
              f"震荡{phase_dist.get('chop',0)*100:.0f}%"
              f"  (slope阈值: >{thresholds['slope_strong']:.3f} / >{thresholds['slope_chop']:.3f})")

        all_trades = run_backtest_all(df, sym_key)

        fixed = filter_fixed(all_trades)
        adaptive = filter_adaptive(all_trades)
        adaptive_v2 = filter_adaptive_v2(all_trades)

        # 分阶段表现
        print(f"  {'阶段':<8} {'信号':>4} {'胜率':>7} {'EV':>8} {'累计':>8}")
        phase_stats = {}
        for ph in ['strong', 'normal', 'chop']:
            pt = [t for t in all_trades if t['phase'] == ph and t['stop_ticks'] == 5 and t['pb_bars'] >= 2]
            pnls = [t['s1_pnl'] for t in pt]
            stats = calc_stats(pnls)
            phase_stats[ph] = stats
            if stats:
                flag = " *" if stats['count'] < 15 else ""
                print(f"  {ph:<8} {stats['count']:>4} {stats['win_rate']:>6.1f}% {stats['ev']:>+8.2f} {stats['total']:>+7.2f}%{flag}")
            else:
                print(f"  {ph:<8}    0       -        -        -")

        f_s1 = calc_stats([t['s1_pnl'] for t in fixed])
        f_s2 = calc_stats([t['s2_pnl'] for t in fixed])
        a_s1 = calc_stats([t['s1_pnl'] for t in adaptive])
        a_s2 = calc_stats([t['s2_pnl'] for t in adaptive])
        a2_s1 = calc_stats([t['s1_pnl'] for t in adaptive_v2])
        a2_s2 = calc_stats([t['s2_pnl'] for t in adaptive_v2])

        def fmt(stats, label):
            if stats:
                return f"{label}: {stats['count']}笔 EV={stats['ev']:+.2f} 累计{stats['total']:+.2f}%"
            return f"{label}: 无信号"

        print(f"  {fmt(f_s1, '固定')}")
        print(f"  {fmt(a_s1, '自适应A(震荡跳过)')}")
        print(f"  {fmt(a2_s1, '自适应B(震荡收紧)')}")

        all_results[name] = {
            'df': df, 'fixed': fixed, 'adaptive': adaptive, 'adaptive_v2': adaptive_v2,
            'f_s1': f_s1, 'f_s2': f_s2,
            'a_s1': a_s1, 'a_s2': a_s2,
            'a2_s1': a2_s1, 'a2_s2': a2_s2,
            'phase_dist': phase_dist, 'phase_stats': phase_stats,
        }

    # ============================================================
    #  汇总表
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  汇总对比 S1 (固定 vs 自适应A vs 自适应B)")
    print(f"{'='*70}")
    print(f"  {'品种':<6} | {'固定':^22} | {'A(震荡跳过)':^22} | {'B(震荡收紧)':^22}")
    print(f"  {'':6} | {'笔数':>4} {'EV':>7} {'累计':>8} | {'笔数':>4} {'EV':>7} {'累计':>8} | {'笔数':>4} {'EV':>7} {'累计':>8}")
    print(f"  {'-'*76}")

    for name, r in all_results.items():
        def g(s):
            return (s['count'], s['ev'], s['total']) if s else (0, 0, 0)
        fc, fe, ft = g(r['f_s1'])
        ac, ae, at_ = g(r['a_s1'])
        bc, be, bt = g(r['a2_s1'])
        print(f"  {name:<6} | {fc:>4} {fe:>+7.2f} {ft:>+7.2f}% | {ac:>4} {ae:>+7.2f} {at_:>+7.2f}% | {bc:>4} {be:>+7.2f} {bt:>+7.2f}%")

    print(f"\n  汇总对比 S2:")
    print(f"  {'-'*76}")
    for name, r in all_results.items():
        def g(s):
            return (s['count'], s['ev'], s['total']) if s else (0, 0, 0)
        fc, fe, ft = g(r['f_s2'])
        ac, ae, at_ = g(r['a_s2'])
        bc, be, bt = g(r['a2_s2'])
        print(f"  {name:<6} | {fc:>4} {fe:>+7.2f} {ft:>+7.2f}% | {ac:>4} {ae:>+7.2f} {at_:>+7.2f}% | {bc:>4} {be:>+7.2f} {bt:>+7.2f}%")

    # ============================================================
    #  绘图
    # ============================================================
    n = len(all_results)
    fig, axes = plt.subplots(n, 2, figsize=(18, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for row_idx, (name, r) in enumerate(all_results.items()):
        df_p = r['df']
        cutoff = df_p['datetime'].iloc[-1] - timedelta(days=120)
        seg = df_p[df_p['datetime'] >= cutoff]

        # 左图：价格+阶段背景
        ax = axes[row_idx][0]
        ax.plot(seg['datetime'], seg['close'], color='#333', linewidth=0.7)
        ax.plot(seg['datetime'], seg['ma_fast'], color='#00bcd4', linewidth=0.7, alpha=0.6)
        ax.plot(seg['datetime'], seg['ma_slow'], color='#e91e63', linewidth=0.7, alpha=0.6)

        prev_phase = None
        start_dt = None
        cmap = {'strong': '#4caf50', 'normal': '#ff9800', 'chop': '#f44336', 'unknown': '#9e9e9e'}
        for _, bar in seg.iterrows():
            if bar['phase'] != prev_phase:
                if prev_phase and start_dt:
                    ax.axvspan(start_dt, bar['datetime'], alpha=0.12, color=cmap.get(prev_phase, '#9e9e9e'))
                start_dt = bar['datetime']
                prev_phase = bar['phase']
        if prev_phase and start_dt:
            ax.axvspan(start_dt, seg['datetime'].iloc[-1], alpha=0.12, color=cmap.get(prev_phase, '#9e9e9e'))

        # 标注自适应B信号
        for t in r['adaptive_v2']:
            color = '#f44336' if t['direction'] == 'short' else '#2196f3'
            marker = 'v' if t['direction'] == 'short' else '^'
            ax.scatter(t['time'], t['entry_price'], color=color, marker=marker, s=30, zorder=5)

        pd_str = r['phase_dist']
        ax.set_title(f"{name} | 强{pd_str.get('strong',0)*100:.0f}% "
                    f"正常{pd_str.get('normal',0)*100:.0f}% 震荡{pd_str.get('chop',0)*100:.0f}%",
                    fontsize=10, fontweight='bold')
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.15)

        # 右图：三条累计收益线
        ax = axes[row_idx][1]
        if r['f_s1'] and len(r['f_s1']['cum']) > 0:
            ax.plot(r['f_s1']['cum'], color='#9e9e9e', linewidth=1.5,
                   label=f"固定 {r['f_s1']['count']}笔 EV={r['f_s1']['ev']:+.2f}")
        if r['a_s1'] and len(r['a_s1']['cum']) > 0:
            ax.plot(r['a_s1']['cum'], color='#f44336', linewidth=1.2, linestyle='--',
                   label=f"A跳过 {r['a_s1']['count']}笔 EV={r['a_s1']['ev']:+.2f}")
        if r['a2_s1'] and len(r['a2_s1']['cum']) > 0:
            ax.plot(r['a2_s1']['cum'], color='#1565c0', linewidth=2,
                   label=f"B收紧 {r['a2_s1']['count']}笔 EV={r['a2_s1']['ev']:+.2f}")
        ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
        ax.set_title(f"{name} S1 累计收益", fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15)

    plt.suptitle('阶段自适应策略 v2 (分位数阈值) | 固定 vs A(跳过震荡) vs B(震荡收紧)',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'phase_analysis_v2.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图表: {path}")

    # 结论
    print(f"\n{'='*70}")
    print(f"  结论")
    print(f"{'='*70}")
    for label, key in [("自适应A(震荡跳过)", 'a_s1'), ("自适应B(震荡收紧)", 'a2_s1')]:
        improved = sum(1 for r in all_results.values()
                       if r[key] and r['f_s1'] and r[key]['ev'] > r['f_s1']['ev'])
        print(f"  {label}: {improved}/{len(all_results)} 个品种EV优于固定策略")


if __name__ == '__main__':
    main()
