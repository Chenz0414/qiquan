# -*- coding: utf-8 -*-
"""
生猪(DCE.lh) 详细策略分析
- 基准策略: 回调>=4根
- 对比: 回调>=3根、>=5根、>=6根
- 对比: 不同止损档位(3/5/7/10 ticks)
- 多空分别统计
- 按月收益分布
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

SYMBOL_KEY = "DCE.lh"
TQ_SYMBOL = "KQ.m@DCE.lh"
NAME = "生猪"
PERIOD_MIN = 10
SCAN_DAYS = 170  # 多取一些数据


def fetch_data(api):
    df = get_klines(api, TQ_SYMBOL, SYMBOL_KEY, period_min=PERIOD_MIN, days=SCAN_DAYS)
    print(f"  共 {len(df)} 根K线, {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    return df


def run_strategy(df, min_pb_bars=4, stop_ticks=5):
    """跑一组参数"""
    sym_cfg = SYMBOL_CONFIGS.get(SYMBOL_KEY, {"tick_size": 5.0})
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

    cutoff = df['datetime'].iloc[-1] - timedelta(days=120)
    signals = [s for s in signals if df.iloc[s['idx']]['datetime'] >= cutoff]

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
            's1_bars': s1_exit.bars_held if s1_exit else 0,
            's2_bars': s2_exit.bars_held if s2_exit else 0,
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


def main():
    print("=" * 65)
    print(f"  {NAME} 详细策略分析报告")
    print("=" * 65)

    api = TqApi(auth=TqAuth(cfg.TQ_ACCOUNT, cfg.TQ_PASSWORD))
    df = fetch_data(api)
    api.close()

    # =============================================
    # 1. 基准策略: 回调>=4根, 止损5ticks
    # =============================================
    print(f"\n{'='*65}")
    print(f"  一、基准策略 (回调>=4根, 止损5ticks)")
    print(f"{'='*65}")
    trades_base = run_strategy(df, min_pb_bars=4, stop_ticks=5)
    print(f"  信号数: {len(trades_base)}")

    if trades_base:
        longs = [t for t in trades_base if t['direction'] == 'long']
        shorts = [t for t in trades_base if t['direction'] == 'short']
        print(f"  做多: {len(longs)}笔 | 做空: {len(shorts)}笔")

        for strat, label in [('s1', 'S1 快刀'), ('s2', 'S2 波段')]:
            pnls = [t[f'{strat}_pnl'] for t in trades_base if t[f'{strat}_reason'] != 'open']
            stats = calc_stats(pnls)
            if stats:
                verdict = "有效" if stats['ev'] > 0.3 else ("边缘" if stats['ev'] > 0 else "无效")
                print(f"\n  [{label}]")
                print(f"    样本: {stats['count']} | 胜率: {stats['win_rate']:.1f}% | 盈亏比: {stats['pf']:.2f}")
                print(f"    EV: {stats['ev']:+.2f} | 累计: {stats['total']:+.2f}% | 最大回撤: {stats['max_dd']:.2f}% | 连亏: {stats['max_streak']}")
                print(f"    结论: {verdict}")

        # 多空分别统计
        print(f"\n  --- 多空分别 ---")
        for direction, dir_label, dir_trades in [('long', '做多', longs), ('short', '做空', shorts)]:
            if not dir_trades:
                continue
            for strat, sl in [('s1', 'S1'), ('s2', 'S2')]:
                pnls = [t[f'{strat}_pnl'] for t in dir_trades if t[f'{strat}_reason'] != 'open']
                stats = calc_stats(pnls)
                if stats:
                    verdict = "有效" if stats['ev'] > 0.3 else ("边缘" if stats['ev'] > 0 else "无效")
                    print(f"    {dir_label} {sl}: {stats['count']}笔 胜率{stats['win_rate']:.1f}% EV={stats['ev']:+.2f} 累计{stats['total']:+.2f}% -> {verdict}")

        # 按月分布
        print(f"\n  --- 按月收益分布 (S1) ---")
        for t in trades_base:
            t['month'] = t['time'].strftime('%Y-%m')
        months = sorted(set(t['month'] for t in trades_base))
        for m in months:
            m_trades = [t for t in trades_base if t['month'] == m]
            m_s1 = sum(t['s1_pnl'] for t in m_trades)
            m_wins = sum(1 for t in m_trades if t['s1_pnl'] > 0)
            print(f"    {m}: {len(m_trades)}笔 S1累计{m_s1:+.2f}% 胜率{m_wins/len(m_trades)*100:.0f}%")

        # 逐笔明细
        print(f"\n  --- 逐笔交易明细 ---")
        print(f"  {'时间':<20} {'方向':>4} {'入场':>8} {'回调':>4} {'S1收益':>8} {'S2收益':>8} {'S1出场':>10} {'S2出场':>10}")
        for t in trades_base:
            d = "多" if t['direction'] == 'long' else "空"
            print(f"  {str(t['time']):<20} {d:>4} {t['entry_price']:>8.1f} {t['pb_bars']:>4} {t['s1_pnl']:>+8.2f}% {t['s2_pnl']:>+8.2f}% {t['s1_reason']:>10} {t['s2_reason']:>10}")

    # =============================================
    # 2. 回调根数对比: 3/4/5/6
    # =============================================
    print(f"\n{'='*65}")
    print(f"  二、回调根数对比")
    print(f"{'='*65}")
    print(f"  {'回调':>6} {'信号数':>6} {'S1胜率':>8} {'S1 EV':>8} {'S1累计':>8} {'S2胜率':>8} {'S2 EV':>8} {'S2累计':>8}")
    print(f"  {'-'*62}")

    pb_results = {}
    for pb in [3, 4, 5, 6]:
        trades = run_strategy(df, min_pb_bars=pb, stop_ticks=5)
        s1_pnls = [t['s1_pnl'] for t in trades if t['s1_reason'] != 'open']
        s2_pnls = [t['s2_pnl'] for t in trades if t['s2_reason'] != 'open']
        s1 = calc_stats(s1_pnls)
        s2 = calc_stats(s2_pnls)
        pb_results[pb] = {'trades': trades, 's1': s1, 's2': s2}

        s1_wr = s1['win_rate'] if s1 else 0
        s1_ev = s1['ev'] if s1 else 0
        s1_tot = s1['total'] if s1 else 0
        s2_wr = s2['win_rate'] if s2 else 0
        s2_ev = s2['ev'] if s2 else 0
        s2_tot = s2['total'] if s2 else 0
        print(f"  >={pb}根 {len(trades):>6} {s1_wr:>7.1f}% {s1_ev:>+8.2f} {s1_tot:>+7.2f}% {s2_wr:>7.1f}% {s2_ev:>+8.2f} {s2_tot:>+7.2f}%")

    # =============================================
    # 3. 止损档位对比: 3/5/7/10 ticks
    # =============================================
    print(f"\n{'='*65}")
    print(f"  三、止损档位对比 (回调>=4根)")
    print(f"{'='*65}")
    print(f"  {'止损':>8} {'信号数':>6} {'S1胜率':>8} {'S1 EV':>8} {'S1累计':>8} {'S2胜率':>8} {'S2 EV':>8} {'S2累计':>8}")
    print(f"  {'-'*62}")

    stop_results = {}
    for st in [3, 5, 7, 10]:
        trades = run_strategy(df, min_pb_bars=4, stop_ticks=st)
        s1_pnls = [t['s1_pnl'] for t in trades if t['s1_reason'] != 'open']
        s2_pnls = [t['s2_pnl'] for t in trades if t['s2_reason'] != 'open']
        s1 = calc_stats(s1_pnls)
        s2 = calc_stats(s2_pnls)
        stop_results[st] = {'s1': s1, 's2': s2}

        s1_wr = s1['win_rate'] if s1 else 0
        s1_ev = s1['ev'] if s1 else 0
        s1_tot = s1['total'] if s1 else 0
        s2_wr = s2['win_rate'] if s2 else 0
        s2_ev = s2['ev'] if s2 else 0
        s2_tot = s2['total'] if s2 else 0
        label = f"{st}ticks({st*5}元)"
        print(f"  {label:>8} {len(trades):>6} {s1_wr:>7.1f}% {s1_ev:>+8.2f} {s1_tot:>+7.02f}% {s2_wr:>7.1f}% {s2_ev:>+8.2f} {s2_tot:>+7.2f}%")

    # =============================================
    # 4. 绘图
    # =============================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (0,0) 基准策略 S1+S2 累计收益
    base_s1 = calc_stats([t['s1_pnl'] for t in trades_base if t['s1_reason'] != 'open'])
    base_s2 = calc_stats([t['s2_pnl'] for t in trades_base if t['s2_reason'] != 'open'])
    ax = axes[0][0]
    if base_s1:
        ax.plot(base_s1['cum'], label=f"S1 EV={base_s1['ev']:+.2f}", color='#1565c0', linewidth=1.5)
        ax.fill_between(range(len(base_s1['cum'])), 0, base_s1['cum'], alpha=0.1, color='#1565c0')
    if base_s2:
        ax.plot(base_s2['cum'], label=f"S2 EV={base_s2['ev']:+.2f}", color='#e65100', linewidth=1.5)
        ax.fill_between(range(len(base_s2['cum'])), 0, base_s2['cum'], alpha=0.1, color='#e65100')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'{NAME} 基准策略 (>=4根, 5ticks)', fontweight='bold')
    ax.set_xlabel('交易序号')
    ax.set_ylabel('累计收益 %')
    ax.legend()
    ax.grid(True, alpha=0.2)

    # (0,1) 多空分别
    ax = axes[0][1]
    for direction, label, color in [('long', '做多', '#1565c0'), ('short', '做空', '#e65100')]:
        pnls = [t['s1_pnl'] for t in trades_base if t['direction'] == direction and t['s1_reason'] != 'open']
        if pnls:
            cum = np.cumsum(pnls)
            stats = calc_stats(pnls)
            ax.plot(cum, label=f"{label} S1 EV={stats['ev']:+.2f} ({len(pnls)}笔)", color=color, linewidth=1.5)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'{NAME} 多空S1分别', fontweight='bold')
    ax.set_xlabel('交易序号')
    ax.set_ylabel('累计收益 %')
    ax.legend()
    ax.grid(True, alpha=0.2)

    # (1,0) 不同回调根数对比
    ax = axes[1][0]
    colors_pb = {3: '#f44336', 4: '#1565c0', 5: '#4caf50', 6: '#ff9800'}
    for pb, res in pb_results.items():
        if res['s1'] and res['s1']['cum'] is not None:
            ev = res['s1']['ev']
            ax.plot(res['s1']['cum'], label=f">={pb}根 EV={ev:+.2f} ({res['s1']['count']}笔)",
                    color=colors_pb[pb], linewidth=2 if pb == 4 else 1)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'{NAME} 回调根数对比 (S1)', fontweight='bold')
    ax.set_xlabel('交易序号')
    ax.set_ylabel('累计收益 %')
    ax.legend()
    ax.grid(True, alpha=0.2)

    # (1,1) 不同止损档位对比
    ax = axes[1][1]
    colors_st = {3: '#f44336', 5: '#1565c0', 7: '#4caf50', 10: '#ff9800'}
    for st, res in stop_results.items():
        if res['s1'] and res['s1']['cum'] is not None:
            ev = res['s1']['ev']
            ax.plot(res['s1']['cum'], label=f"{st}ticks({st*5}元) EV={ev:+.2f}",
                    color=colors_st[st], linewidth=2 if st == 5 else 1)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'{NAME} 止损档位对比 (S1, >=4根)', fontweight='bold')
    ax.set_xlabel('交易序号')
    ax.set_ylabel('累计收益 %')
    ax.legend()
    ax.grid(True, alpha=0.2)

    plt.suptitle(f'{NAME}(DCE.lh) 策略详细分析 | 120天 10min', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'lh_detail_analysis.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图表已保存: {path}")

    # =============================================
    # 5. 最终建议
    # =============================================
    print(f"\n{'='*65}")
    print(f"  四、分析总结与建议")
    print(f"{'='*65}")

    # 找最佳回调根数
    best_pb = max(pb_results.items(), key=lambda x: x[1]['s1']['ev'] if x[1]['s1'] else -999)
    best_st = max(stop_results.items(), key=lambda x: x[1]['s1']['ev'] if x[1]['s1'] else -999)

    print(f"  最佳回调根数: >={best_pb[0]}根 (S1 EV={best_pb[1]['s1']['ev']:+.2f})")
    print(f"  最佳止损档位: {best_st[0]}ticks (S1 EV={best_st[1]['s1']['ev']:+.2f})")

    # 综合判断
    base_ev = base_s1['ev'] if base_s1 else 0
    if base_ev > 0.3:
        print(f"\n  >>> 基准策略(>=4根/5ticks): 有效，可以交易")
    elif base_ev > 0:
        print(f"\n  >>> 基准策略(>=4根/5ticks): 边缘，谨慎交易")
        if best_pb[1]['s1']['ev'] > 0.3:
            print(f"  >>> 调整为 >={best_pb[0]}根 可能改善表现")
        if best_st[1]['s1']['ev'] > 0.3:
            print(f"  >>> 调整止损为 {best_st[0]}ticks 可能改善表现")
    else:
        print(f"\n  >>> 基准策略(>=4根/5ticks): 无效，不建议交易")
        any_good = False
        for pb, res in pb_results.items():
            if res['s1'] and res['s1']['ev'] > 0.3:
                print(f"  >>> 但 >={pb}根 表现有效 (EV={res['s1']['ev']:+.2f})，可考虑")
                any_good = True
        if not any_good:
            print(f"  >>> 所有参数组合均不理想，建议放弃该品种")


if __name__ == '__main__':
    main()
