# -*- coding: utf-8 -*-
"""
市场阶段分类 + 阶段自适应策略验证
==================================
核心假设：策略跟着阶段走，不跟着品种走。
- 强趋势阶段：>=2根回调、只做顺势、3ticks止损
- 正常趋势阶段：>=4根回调、双向、5ticks止损
- 震荡阶段：不交易

对比：固定策略 vs 自适应策略，8个品种120天回测
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

# ============================================================
#  品种列表
# ============================================================
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
SCAN_DAYS = 170  # 120天回测 + 50天预热

# ============================================================
#  阶段分类
# ============================================================

def compute_atr(df, period=14):
    """计算ATR"""
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_phases(df):
    """计算阶段指标并分类"""
    df = df.copy()
    df['atr'] = compute_atr(df, 14)

    # 指标A: MA斜率 = (MA10 - MA10[5根前]) / ATR
    df['ma_slope'] = (df['ma_fast'] - df['ma_fast'].shift(5)) / df['atr']

    # 指标B: 均线间距 = (MA10 - MA20) / ATR (带符号)
    df['ma_width'] = (df['ma_fast'] - df['ma_slow']) / df['atr']

    # 阶段分类
    def classify(row):
        if pd.isna(row['ma_slope']) or pd.isna(row['ma_width']) or pd.isna(row['atr']) or row['atr'] == 0:
            return 'unknown'
        slope = abs(row['ma_slope'])
        width = abs(row['ma_width'])
        if slope > 1.5 and width > 1.0:
            return 'strong'
        elif slope > 0.5 and width > 0.3:
            return 'normal'
        else:
            return 'chop'

    df['phase'] = df.apply(classify, axis=1)

    # 趋势方向（用ma_width符号判断）
    df['trend_dir'] = np.where(df['ma_width'] > 0, 'up', 'down')

    return df


# ============================================================
#  回测引擎
# ============================================================

def run_backtest_all_signals(df, symbol_key):
    """用最宽松参数(>=2根)跑全量信号，返回所有交易+阶段标记"""
    sym_cfg = SYMBOL_CONFIGS.get(symbol_key, {"tick_size": 1.0})
    tick_size = sym_cfg["tick_size"]

    detector = SignalDetector(min_pb_bars=2)  # 最宽松
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
            signals.append({
                'idx': i,
                'signal': result,
                'phase': row['phase'],
                'trend_dir': row['trend_dir'],
                'ma_slope': row['ma_slope'],
                'ma_width': row['ma_width'],
            })

    # 只保留最近120天
    cutoff = df['datetime'].iloc[-1] - timedelta(days=120)
    signals = [s for s in signals if df.iloc[s['idx']]['datetime'] >= cutoff]

    # 模拟出场（对每个信号分别用不同止损跑）
    trades = []
    for sig_info in signals:
        idx = sig_info['idx']
        sig = sig_info['signal']

        # 跑两套止损：3ticks（强趋势用）和 5ticks（正常用）
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
    """固定策略：>=4根、5ticks、双向"""
    return [t for t in trades if t['pb_bars'] >= 4 and t['stop_ticks'] == 5]


def filter_adaptive(trades):
    """自适应策略：按阶段选参数"""
    result = []
    for t in trades:
        phase = t['phase']
        if phase == 'strong':
            # 强趋势：>=2根、3ticks、只做顺势
            if t['stop_ticks'] != 3:
                continue
            if t['pb_bars'] < 2:
                continue
            # 顺势过滤：多头信号+上升趋势，或空头信号+下降趋势
            is_with_trend = (
                (t['direction'] == 'long' and t['trend_dir'] == 'up') or
                (t['direction'] == 'short' and t['trend_dir'] == 'down')
            )
            if not is_with_trend:
                continue
            result.append(t)
        elif phase == 'normal':
            # 正常趋势：>=4根、5ticks、双向
            if t['stop_ticks'] != 5:
                continue
            if t['pb_bars'] < 4:
                continue
            result.append(t)
        # 震荡期：丢弃
    return result


# ============================================================
#  统计
# ============================================================

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
    streak = max_streak = 0
    for p in pnls:
        if p <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return {
        'count': len(pnls), 'win_rate': win_rate,
        'avg_win': avg_win, 'avg_loss': avg_loss,
        'pf': pf, 'ev': ev, 'total': total,
        'max_dd': max_dd, 'max_streak': max_streak,
        'cum': cum,
    }


# ============================================================
#  主流程
# ============================================================

def main():
    print("=" * 70)
    print("  市场阶段分类 + 阶段自适应策略验证")
    print("=" * 70)

    api = TqApi(auth=TqAuth(cfg.TQ_ACCOUNT, cfg.TQ_PASSWORD))

    all_results = {}

    for tq_sym, sym_key, name in SYMBOLS:
        print(f"\n--- {name} ({sym_key}) ---")
        df = get_klines(api, tq_sym, sym_key, period_min=PERIOD_MIN, days=SCAN_DAYS)
        df = compute_phases(df)

        # 阶段分布
        cutoff = df['datetime'].iloc[-1] - timedelta(days=120)
        recent = df[df['datetime'] >= cutoff]
        phase_dist = recent['phase'].value_counts(normalize=True)
        print(f"  阶段分布: 强趋势{phase_dist.get('strong',0)*100:.1f}% | "
              f"正常{phase_dist.get('normal',0)*100:.1f}% | "
              f"震荡{phase_dist.get('chop',0)*100:.1f}%")

        # 全量信号
        all_trades = run_backtest_all_signals(df, sym_key)
        print(f"  全量信号(>=2根): {len([t for t in all_trades if t['stop_ticks']==3])}笔")

        # 固定策略
        fixed = filter_fixed(all_trades)
        # 自适应策略
        adaptive = filter_adaptive(all_trades)

        # 分阶段统计（用5ticks的交易）
        print(f"\n  分阶段表现 (S1):")
        print(f"  {'阶段':<8} {'信号':>4} {'胜率':>8} {'EV':>8} {'累计':>8}")
        phase_stats = {}
        for phase_name in ['strong', 'normal', 'chop']:
            # 用最宽松(>=2, 5ticks)看该阶段的信号表现
            phase_trades = [t for t in all_trades
                           if t['phase'] == phase_name and t['stop_ticks'] == 5 and t['pb_bars'] >= 2]
            pnls = [t['s1_pnl'] for t in phase_trades]
            stats = calc_stats(pnls)
            phase_stats[phase_name] = stats
            if stats:
                flag = " *" if stats['count'] < 15 else ""
                print(f"  {phase_name:<8} {stats['count']:>4} {stats['win_rate']:>7.1f}% "
                      f"{stats['ev']:>+8.2f} {stats['total']:>+7.2f}%{flag}")
            else:
                print(f"  {phase_name:<8}    0      -        -        -")

        # 固定 vs 自适应
        fixed_s1 = calc_stats([t['s1_pnl'] for t in fixed])
        fixed_s2 = calc_stats([t['s2_pnl'] for t in fixed])
        adapt_s1 = calc_stats([t['s1_pnl'] for t in adaptive])
        adapt_s2 = calc_stats([t['s2_pnl'] for t in adaptive])

        print(f"\n  固定 vs 自适应:")
        if fixed_s1:
            print(f"    固定:   S1 {fixed_s1['count']}笔 EV={fixed_s1['ev']:+.2f} 累计{fixed_s1['total']:+.2f}% "
                  f"| S2 EV={fixed_s2['ev']:+.2f} 累计{fixed_s2['total']:+.2f}%")
        else:
            print(f"    固定:   无信号")
        if adapt_s1:
            print(f"    自适应: S1 {adapt_s1['count']}笔 EV={adapt_s1['ev']:+.2f} 累计{adapt_s1['total']:+.2f}% "
                  f"| S2 EV={adapt_s2['ev']:+.2f} 累计{adapt_s2['total']:+.2f}%")
        else:
            print(f"    自适应: 无信号")

        all_results[name] = {
            'df': df, 'all_trades': all_trades,
            'fixed': fixed, 'adaptive': adaptive,
            'fixed_s1': fixed_s1, 'fixed_s2': fixed_s2,
            'adapt_s1': adapt_s1, 'adapt_s2': adapt_s2,
            'phase_dist': phase_dist, 'phase_stats': phase_stats,
        }

    api.close()

    # ============================================================
    #  汇总对比表
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  汇总对比: 固定策略 vs 阶段自适应策略 (S1)")
    print(f"{'='*70}")
    print(f"  {'品种':<8} {'固定EV':>8} {'固定累计':>8} {'固定笔数':>8} | "
          f"{'自适应EV':>8} {'自适应累计':>10} {'自适应笔数':>10} | {'EV改善':>8}")
    print(f"  {'-'*85}")

    for name, r in all_results.items():
        f_ev = r['fixed_s1']['ev'] if r['fixed_s1'] else 0
        f_tot = r['fixed_s1']['total'] if r['fixed_s1'] else 0
        f_cnt = r['fixed_s1']['count'] if r['fixed_s1'] else 0
        a_ev = r['adapt_s1']['ev'] if r['adapt_s1'] else 0
        a_tot = r['adapt_s1']['total'] if r['adapt_s1'] else 0
        a_cnt = r['adapt_s1']['count'] if r['adapt_s1'] else 0
        diff = a_ev - f_ev
        marker = ">>>" if diff > 0.1 else ("===" if abs(diff) <= 0.1 else "<<<")
        print(f"  {name:<8} {f_ev:>+8.2f} {f_tot:>+7.2f}% {f_cnt:>8} | "
              f"{a_ev:>+8.2f} {a_tot:>+9.2f}% {a_cnt:>10} | {diff:>+7.2f} {marker}")

    # S2
    print(f"\n  汇总对比: 固定策略 vs 阶段自适应策略 (S2)")
    print(f"  {'-'*85}")
    for name, r in all_results.items():
        f_ev = r['fixed_s2']['ev'] if r['fixed_s2'] else 0
        f_tot = r['fixed_s2']['total'] if r['fixed_s2'] else 0
        f_cnt = r['fixed_s2']['count'] if r['fixed_s2'] else 0
        a_ev = r['adapt_s2']['ev'] if r['adapt_s2'] else 0
        a_tot = r['adapt_s2']['total'] if r['adapt_s2'] else 0
        a_cnt = r['adapt_s2']['count'] if r['adapt_s2'] else 0
        diff = a_ev - f_ev
        marker = ">>>" if diff > 0.1 else ("===" if abs(diff) <= 0.1 else "<<<")
        print(f"  {name:<8} {f_ev:>+8.2f} {f_tot:>+7.2f}% {f_cnt:>8} | "
              f"{a_ev:>+8.2f} {a_tot:>+9.02f}% {a_cnt:>10} | {diff:>+7.2f} {marker}")

    # ============================================================
    #  阈值敏感性检查
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  阈值敏感性检查 (ma_slope阈值 ±25%)")
    print(f"{'='*70}")

    # 拿白银做敏感性测试
    ag_data = all_results.get('白银')
    if ag_data:
        for slope_mult in [0.75, 1.0, 1.25]:
            slope_thresh_strong = 1.5 * slope_mult
            slope_thresh_normal = 0.5 * slope_mult

            # 重新分类
            def reclassify(t):
                slope = abs(t['ma_slope']) if not pd.isna(t['ma_slope']) else 0
                width = abs(df.iloc[0]['ma_fast'] - df.iloc[0]['ma_slow']) if len(df) > 0 else 0
                # 用原始数据重新判断
                if slope > slope_thresh_strong:
                    return 'strong'
                elif slope > slope_thresh_normal:
                    return 'normal'
                return 'chop'

            # 用白银的全量交易重新过滤
            ag_trades = ag_data['all_trades']
            adapted = []
            for t in ag_trades:
                slope = abs(t['ma_slope']) if not pd.isna(t['ma_slope']) else 0
                # 简单重新分类
                if slope > slope_thresh_strong:
                    phase = 'strong'
                elif slope > slope_thresh_normal:
                    phase = 'normal'
                else:
                    phase = 'chop'

                if phase == 'strong':
                    if t['stop_ticks'] != 3 or t['pb_bars'] < 2:
                        continue
                    is_wt = ((t['direction'] == 'long' and t['trend_dir'] == 'up') or
                             (t['direction'] == 'short' and t['trend_dir'] == 'down'))
                    if not is_wt:
                        continue
                    adapted.append(t)
                elif phase == 'normal':
                    if t['stop_ticks'] != 5 or t['pb_bars'] < 4:
                        continue
                    adapted.append(t)

            stats = calc_stats([t['s1_pnl'] for t in adapted])
            if stats:
                print(f"  slope阈值x{slope_mult:.2f} (强>{slope_thresh_strong:.2f}, 正常>{slope_thresh_normal:.2f}): "
                      f"{stats['count']}笔 EV={stats['ev']:+.2f} 累计{stats['total']:+.2f}%")

    # ============================================================
    #  绘图
    # ============================================================
    n = len(all_results)
    fig, axes = plt.subplots(n, 2, figsize=(18, 4.5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    phase_colors = {'strong': '#4caf5040', 'normal': '#ff980040', 'chop': '#f4433640', 'unknown': '#9e9e9e20'}

    for row_idx, (name, r) in enumerate(all_results.items()):
        df_plot = r['df']
        cutoff = df_plot['datetime'].iloc[-1] - timedelta(days=120)
        seg = df_plot[df_plot['datetime'] >= cutoff].copy()

        # 左图：价格+阶段背景色
        ax = axes[row_idx][0]
        ax.plot(seg['datetime'], seg['close'], color='#333', linewidth=0.7)
        ax.plot(seg['datetime'], seg['ma_fast'], color='#00bcd4', linewidth=0.8, alpha=0.7)
        ax.plot(seg['datetime'], seg['ma_slow'], color='#e91e63', linewidth=0.8, alpha=0.7)

        # 画阶段背景色（每天一个色块）
        prev_phase = None
        start_dt = None
        for _, bar in seg.iterrows():
            if bar['phase'] != prev_phase:
                if prev_phase and start_dt:
                    ax.axvspan(start_dt, bar['datetime'], alpha=0.15,
                              color={'strong': '#4caf50', 'normal': '#ff9800', 'chop': '#f44336'}.get(prev_phase, '#9e9e9e'))
                start_dt = bar['datetime']
                prev_phase = bar['phase']
        # 最后一段
        if prev_phase and start_dt:
            ax.axvspan(start_dt, seg['datetime'].iloc[-1], alpha=0.15,
                      color={'strong': '#4caf50', 'normal': '#ff9800', 'chop': '#f44336'}.get(prev_phase, '#9e9e9e'))

        # 标注自适应信号
        for t in r['adaptive']:
            color = '#f44336' if t['direction'] == 'short' else '#2196f3'
            marker = 'v' if t['direction'] == 'short' else '^'
            ax.scatter(t['time'], t['entry_price'], color=color, marker=marker, s=40, zorder=5)

        pd_str = r['phase_dist']
        ax.set_title(f"{name} | 强{pd_str.get('strong',0)*100:.0f}% 正常{pd_str.get('normal',0)*100:.0f}% 震荡{pd_str.get('chop',0)*100:.0f}%",
                    fontsize=11, fontweight='bold')
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.15)

        # 右图：累计收益对比
        ax = axes[row_idx][1]
        if r['fixed_s1'] and len(r['fixed_s1']['cum']) > 0:
            ax.plot(r['fixed_s1']['cum'], color='#9e9e9e', linewidth=1.5, label=f"固定 EV={r['fixed_s1']['ev']:+.2f}")
        if r['adapt_s1'] and len(r['adapt_s1']['cum']) > 0:
            ax.plot(r['adapt_s1']['cum'], color='#1565c0', linewidth=2, label=f"自适应 EV={r['adapt_s1']['ev']:+.2f}")
        ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
        ax.set_title(f"{name} S1 累计收益对比", fontsize=11, fontweight='bold')
        ax.set_xlabel('交易序号', fontsize=9)
        ax.set_ylabel('累计 %', fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.15)

    plt.suptitle('阶段自适应策略 vs 固定策略 | 120天回测 | 8品种', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'phase_analysis.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图表: {path}")

    # 最终结论
    print(f"\n{'='*70}")
    print(f"  结论")
    print(f"{'='*70}")
    improved = sum(1 for r in all_results.values()
                   if r['adapt_s1'] and r['fixed_s1'] and r['adapt_s1']['ev'] > r['fixed_s1']['ev'])
    total_syms = len(all_results)
    print(f"  自适应策略在 {improved}/{total_syms} 个品种上EV优于固定策略")

    # 好品种是否退步
    good_syms = ['黄金', '白银', '原油', 'PVC', '碳酸锂']
    degraded = []
    for name in good_syms:
        r = all_results.get(name)
        if r and r['fixed_s1'] and r['adapt_s1']:
            if r['adapt_s1']['ev'] < r['fixed_s1']['ev'] - 0.1:
                degraded.append(name)
    if degraded:
        print(f"  WARNING: 好品种退步: {', '.join(degraded)}")
    else:
        print(f"  好品种无明显退步")

    # 差品种是否改善
    bad_syms = ['甲醇', 'PTA', '生猪']
    for name in bad_syms:
        r = all_results.get(name)
        if r and r['fixed_s1'] and r['adapt_s1']:
            diff = r['adapt_s1']['ev'] - r['fixed_s1']['ev']
            if diff > 0.1:
                print(f"  {name}: EV改善 {diff:+.2f}")
            else:
                print(f"  {name}: 未明显改善 ({diff:+.2f})")


if __name__ == '__main__':
    main()
