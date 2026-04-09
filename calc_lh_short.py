# -*- coding: utf-8 -*-
"""生猪：只做空 vs 多空都做 vs 只顺势做 对比"""

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


def run_strategy(df, symbol_key, min_pb_bars=4, stop_ticks=5):
    sym_cfg = SYMBOL_CONFIGS.get(symbol_key, {"tick_size": 5.0})
    tick_size = sym_cfg["tick_size"]
    detector = SignalDetector(min_pb_bars=min_pb_bars)
    warmup = cfg.MA_SLOW + 5

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
            signals.append({'idx': i, 'signal': result})

    trades = []
    for sig_info in signals:
        idx = sig_info['idx']
        sig = sig_info['signal']
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
            's1_pnl': s1_exit.pnl_pct if s1_exit else 0,
            's2_pnl': s2_exit.pnl_pct if s2_exit else 0,
            's1_reason': s1_exit.exit_reason if s1_exit else 'open',
            's2_reason': s2_exit.exit_reason if s2_exit else 'open',
        })
    return trades


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
    return {
        'count': len(pnls), 'win_rate': win_rate,
        'avg_win': avg_win, 'avg_loss': avg_loss,
        'pf': pf, 'ev': ev, 'total': total,
        'max_dd': max_dd, 'cum': cum,
    }


def print_stats(label, pnls):
    stats = calc_stats(pnls)
    if not stats:
        print(f"  {label}: 无数据")
        return stats
    verdict = "有效" if stats['ev'] > 0.3 else ("边缘" if stats['ev'] > 0 else "无效")
    print(f"  {label}: {stats['count']}笔 | 胜率{stats['win_rate']:.1f}% | 盈亏比{stats['pf']:.2f} | EV={stats['ev']:+.2f} | 累计{stats['total']:+.2f}% | 回撤{stats['max_dd']:.2f}% -> {verdict}")
    return stats


def main():
    api = TqApi(auth=TqAuth(cfg.TQ_ACCOUNT, cfg.TQ_PASSWORD))
    df = get_klines(api, "KQ.m@DCE.lh", "DCE.lh", period_min=10, days=170)
    api.close()
    print(f"数据: {len(df)}根K线, {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

    trades = run_strategy(df, "DCE.lh", min_pb_bars=4, stop_ticks=5)

    # 120天交易
    cutoff_120 = df['datetime'].iloc[-1] - timedelta(days=120)
    trades_120 = [t for t in trades if t['time'] >= cutoff_120]

    # 最近30天交易
    cutoff_30 = df['datetime'].iloc[-1] - timedelta(days=30)
    trades_30 = [t for t in trades_120 if t['time'] >= cutoff_30]

    # 最近60天
    cutoff_60 = df['datetime'].iloc[-1] - timedelta(days=60)
    trades_60 = [t for t in trades_120 if t['time'] >= cutoff_60]

    print(f"\n{'='*70}")
    print(f"  生猪 — 不同过滤方式对比")
    print(f"{'='*70}")

    for period_label, period_trades in [
        ("120天全部", trades_120),
        ("60天", trades_60),
        ("30天", trades_30),
    ]:
        print(f"\n--- {period_label} ---")
        all_s1 = [t['s1_pnl'] for t in period_trades]
        short_s1 = [t['s1_pnl'] for t in period_trades if t['direction'] == 'short']
        long_s1 = [t['s1_pnl'] for t in period_trades if t['direction'] == 'long']
        all_s2 = [t['s2_pnl'] for t in period_trades]
        short_s2 = [t['s2_pnl'] for t in period_trades if t['direction'] == 'short']
        long_s2 = [t['s2_pnl'] for t in period_trades if t['direction'] == 'long']

        print(f"  [S1 快刀]")
        print_stats("    多空都做", all_s1)
        print_stats("    只做空  ", short_s1)
        print_stats("    只做多  ", long_s1)
        print(f"  [S2 波段]")
        print_stats("    多空都做", all_s2)
        print_stats("    只做空  ", short_s2)
        print_stats("    只做多  ", long_s2)

    # 逐笔打印30天的交易
    print(f"\n{'='*70}")
    print(f"  最近30天逐笔明细")
    print(f"{'='*70}")
    print(f"  {'时间':<20} {'方向':>4} {'入场':>8} {'回调':>4} {'S1':>8} {'S2':>8}")
    for t in trades_30:
        d = "多" if t['direction'] == 'long' else "空"
        print(f"  {str(t['time']):<20} {d:>4} {t['entry_price']:>8.1f} {t['pb_bars']:>4} {t['s1_pnl']:>+8.2f}% {t['s2_pnl']:>+8.2f}%")

    # 核心问题：顺势过滤 — 用60MA判断大趋势方向，只做顺势
    print(f"\n{'='*70}")
    print(f"  顺势过滤测试：用不同方法判断大方向，只做顺势信号")
    print(f"{'='*70}")

    # 方法1: 20MA方向（已有的 ma_fast > ma_slow）
    # 当前策略已经内置了这个：多头信号要求 ma10 > ma20，空头要求 ma10 < ma20
    # 所以这个已经在用了

    # 方法2: 加60MA过滤
    df['ma60'] = df['close'].rolling(window=60).mean()

    # 重新标记每个信号的60MA方向
    for t in trades_120:
        idx = df[df['datetime'] == t['time']].index
        if len(idx) > 0:
            i = idx[0]
            ma60 = df.iloc[i]['ma60'] if not pd.isna(df.iloc[i]['ma60']) else None
            close = df.iloc[i]['close']
            t['trend_60'] = 'down' if (ma60 and close < ma60) else 'up'
        else:
            t['trend_60'] = 'unknown'

    # 方法3: 加100MA过滤
    df['ma100'] = df['close'].rolling(window=100).mean()
    for t in trades_120:
        idx = df[df['datetime'] == t['time']].index
        if len(idx) > 0:
            i = idx[0]
            ma100 = df.iloc[i]['ma100'] if not pd.isna(df.iloc[i]['ma100']) else None
            close = df.iloc[i]['close']
            t['trend_100'] = 'down' if (ma100 and close < ma100) else 'up'
        else:
            t['trend_100'] = 'unknown'

    print(f"\n  方法1: 只做顺势（价格在60MA下方只做空，上方只做多）")
    trend60_trades = [t for t in trades_120
                      if (t['direction'] == 'short' and t['trend_60'] == 'down')
                      or (t['direction'] == 'long' and t['trend_60'] == 'up')]
    print_stats("    S1", [t['s1_pnl'] for t in trend60_trades])
    print_stats("    S2", [t['s2_pnl'] for t in trend60_trades])

    print(f"\n  方法2: 只做顺势（价格在100MA下方只做空，上方只做多）")
    trend100_trades = [t for t in trades_120
                       if (t['direction'] == 'short' and t['trend_100'] == 'down')
                       or (t['direction'] == 'long' and t['trend_100'] == 'up')]
    print_stats("    S1", [t['s1_pnl'] for t in trend100_trades])
    print_stats("    S2", [t['s2_pnl'] for t in trend100_trades])

    print(f"\n  方法3: 只做顺势（60MA+100MA同向才做）")
    strict_trades = [t for t in trades_120
                     if (t['direction'] == 'short' and t['trend_60'] == 'down' and t['trend_100'] == 'down')
                     or (t['direction'] == 'long' and t['trend_60'] == 'up' and t['trend_100'] == 'up')]
    print_stats("    S1", [t['s1_pnl'] for t in strict_trades])
    print_stats("    S2", [t['s2_pnl'] for t in strict_trades])

    # 对比：对所有已验证品种也做一下顺势过滤
    print(f"\n{'='*70}")
    print(f"  思考：如果所有品种都加顺势过滤会怎样？")
    print(f"{'='*70}")
    print(f"  当前策略已经有10MA>20MA的趋势过滤。")
    print(f"  问题是：在大趋势震荡时，10MA和20MA频繁交叉，")
    print(f"  导致多空信号都出，互相抵消。")
    print(f"  加60MA/100MA大趋势过滤可以解决这个问题。")
    print(f"  但代价是：在趋势转折初期会错过信号。")

    print(f"\n  结论：")
    base_s1 = calc_stats([t['s1_pnl'] for t in trades_120])
    t60_s1 = calc_stats([t['s1_pnl'] for t in trend60_trades])
    t100_s1 = calc_stats([t['s1_pnl'] for t in trend100_trades])
    if base_s1 and t60_s1:
        print(f"  无过滤:   {base_s1['count']}笔 EV={base_s1['ev']:+.2f} 累计{base_s1['total']:+.2f}%")
        print(f"  60MA过滤: {t60_s1['count']}笔 EV={t60_s1['ev']:+.2f} 累计{t60_s1['total']:+.2f}%")
    if t100_s1:
        print(f"  100MA过滤: {t100_s1['count']}笔 EV={t100_s1['ev']:+.2f} 累计{t100_s1['total']:+.2f}%")


if __name__ == '__main__':
    main()
