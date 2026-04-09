# -*- coding: utf-8 -*-
"""
验证策略在指定品种上的表现（120天回测）
用法: python scan_verify.py
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

# 要验证的品种
VERIFY_SYMBOLS = [
    ("KQ.m@CZCE.MA", "CZCE.MA", "甲醇"),
    ("KQ.m@CZCE.TA", "CZCE.TA", "PTA"),
    ("KQ.m@SHFE.fu", "SHFE.fu", "燃料油"),
    ("KQ.m@DCE.v",   "DCE.v",   "PVC"),
]

PERIOD_MIN = 10
SCAN_DAYS = 120 + 50  # 120天回测 + 50天预热


def fetch_data(api, tq_symbol, symbol_key, period_min, scan_days):
    return get_klines(api, tq_symbol, symbol_key, period_min=period_min, days=scan_days)


def scan_and_simulate(df, symbol_key):
    sym_cfg = SYMBOL_CONFIGS.get(symbol_key, {"tick_size": 1.0})
    tick_size = sym_cfg["tick_size"]

    detector = SignalDetector(min_pb_bars=4)
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

    # 只保留最近120天
    cutoff = df['datetime'].iloc[-1] - timedelta(days=120)
    signals = [s for s in signals if df.iloc[s['idx']]['datetime'] >= cutoff]

    trades = []
    for sig_info in signals:
        idx = sig_info['idx']
        sig = sig_info['signal']

        tracker = ExitTracker(
            direction=sig.direction, entry_price=sig.entry_price,
            pullback_extreme=sig.pullback_extreme,
            tick_size=tick_size, stop_ticks=5,
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
    """计算统计指标"""
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

    # 最大回撤
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = dd.min()

    # 最大连亏
    streak = 0
    max_streak = 0
    for p in pnls:
        if p <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    return {
        'count': len(pnls),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'pf': pf,
        'ev': ev,
        'total': total,
        'max_dd': max_dd,
        'max_streak': max_streak,
        'cum': np.cumsum(pnls),
    }


def print_result(name, trades):
    print(f"\n{'='*60}")
    print(f"  {name} | 120天回测 | 10min B类+回调>=4根")
    print(f"{'='*60}")
    print(f"  信号总数: {len(trades)}")

    if not trades:
        print("  ⚠ 无信号")
        return None, None

    longs = sum(1 for t in trades if t['direction'] == 'long')
    shorts = sum(1 for t in trades if t['direction'] == 'short')
    print(f"  做多: {longs}笔 | 做空: {shorts}笔")
    print(f"  平均每天: {len(trades)/120:.1f}笔")

    results = {}
    for strat, label in [('s1', 'S1 快刀(大仓位)'), ('s2', 'S2 波段(小仓位)')]:
        pnls = [t[f'{strat}_pnl'] for t in trades if t[f'{strat}_reason'] != 'open']
        stats = calc_stats(pnls)
        if stats is None:
            continue
        results[strat] = stats

        verdict = "有效" if stats['ev'] > 0.3 else ("边缘" if stats['ev'] > 0 else "无效")
        emoji = "OK" if verdict == "有效" else ("--" if verdict == "边缘" else "NO")

        print(f"\n  [{label}]")
        print(f"    样本: {stats['count']}笔 | 胜率: {stats['win_rate']:.1f}%")
        print(f"    平均盈利: +{stats['avg_win']:.2f}% | 平均亏损: -{stats['avg_loss']:.2f}%")
        print(f"    盈亏比: {stats['pf']:.2f} | 期望值(EV): {stats['ev']:+.2f}")
        print(f"    累计收益: {stats['total']:+.2f}% | 最大回撤: {stats['max_dd']:.2f}%")
        print(f"    最大连亏: {stats['max_streak']}笔")
        print(f"    >>> 结论: [{emoji}] {verdict}")

    return results


def plot_results(all_results, output_dir):
    """绘制所有品种的对比图"""
    n_symbols = len(all_results)
    fig, axes = plt.subplots(n_symbols, 2, figsize=(16, 5 * n_symbols))
    if n_symbols == 1:
        axes = axes.reshape(1, -1)

    colors = {'s1': '#1565c0', 's2': '#e65100'}

    for row_idx, (name, results) in enumerate(all_results.items()):
        for col_idx, (strat, label) in enumerate([('s1', 'S1 快刀'), ('s2', 'S2 波段')]):
            ax = axes[row_idx][col_idx]
            if strat not in results:
                ax.text(0.5, 0.5, '无数据', transform=ax.transAxes, ha='center', fontsize=14)
                ax.set_title(f'{name} - {label}')
                continue

            stats = results[strat]
            cum = stats['cum']
            x = range(1, len(cum) + 1)

            # 累计收益曲线
            ax.fill_between(x, 0, cum, alpha=0.15, color=colors[strat])
            ax.plot(x, cum, color=colors[strat], linewidth=1.5)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

            # 标注最终值
            final = cum[-1]
            ax.annotate(f'{final:+.1f}%', xy=(len(cum), final),
                       fontsize=11, fontweight='bold',
                       color='green' if final > 0 else 'red')

            verdict = "有效" if stats['ev'] > 0.3 else ("边缘" if stats['ev'] > 0 else "无效")
            ax.set_title(f"{name} - {label} | EV={stats['ev']:+.2f} | "
                        f"胜率{stats['win_rate']:.0f}% | 盈亏比{stats['pf']:.1f} | {verdict}",
                        fontsize=11, fontweight='bold',
                        color='#2e7d32' if verdict == '有效' else ('#f57c00' if verdict == '边缘' else '#d32f2f'))
            ax.set_xlabel('交易序号')
            ax.set_ylabel('累计收益 %')
            ax.grid(True, alpha=0.2)

    plt.suptitle('策略验证: 10min B类+回调>=4根 双出场 (120天)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'verify_result.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def main():
    print("连接天勤拉取数据...")
    api = TqApi(auth=TqAuth(cfg.TQ_ACCOUNT, cfg.TQ_PASSWORD))

    all_results = {}

    for tq_symbol, symbol_key, name in VERIFY_SYMBOLS:
        print(f"\n处理 {name} ({tq_symbol}) ...")
        df = fetch_data(api, tq_symbol, symbol_key, PERIOD_MIN, SCAN_DAYS)
        print(f"  共 {len(df)} 根K线, {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

        trades = scan_and_simulate(df, symbol_key)
        results = print_result(name, trades)
        if results:
            all_results[name] = results

    api.close()

    # 绘图
    if all_results:
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(output_dir, exist_ok=True)
        path = plot_results(all_results, output_dir)
        print(f"\n图表: {path}")

    # 最终汇总对比表
    print(f"\n{'='*60}")
    print(f"  汇总对比（含已验证品种参考）")
    print(f"{'='*60}")
    print(f"  {'品种':<8} {'S1 EV':>8} {'S1胜率':>8} {'S1累计':>8} {'S2 EV':>8} {'S2胜率':>8} {'S2累计':>8} {'结论':>6}")
    print(f"  {'-'*62}")
    # 参考数据
    ref = [
        ("白银*", "+0.94", "42.9%", "+37.0%", "+1.05", "30.1%", "+79.4%", "有效"),
        ("黄金*", "+1.05", "43.1%", "+19.3%", "+0.67", "33.3%", "+23.3%", "有效"),
        ("原油*", "+0.93", "36.2%", "+29.9%", "+1.23", "35.5%", "+54.7%", "有效"),
    ]
    for r in ref:
        print(f"  {r[0]:<8} {r[1]:>8} {r[2]:>8} {r[3]:>8} {r[4]:>8} {r[5]:>8} {r[6]:>8} {r[7]:>6}")

    for name, results in all_results.items():
        s1 = results.get('s1', {})
        s2 = results.get('s2', {})
        s1_v = "有效" if s1.get('ev', 0) > 0.3 else ("边缘" if s1.get('ev', 0) > 0 else "无效")
        s2_v = "有效" if s2.get('ev', 0) > 0.3 else ("边缘" if s2.get('ev', 0) > 0 else "无效")
        verdict = "有效" if s1_v == "有效" and s2_v == "有效" else (
            "边缘" if s1.get('ev', 0) > 0 and s2.get('ev', 0) > 0 else "无效")
        print(f"  {name:<8} {s1.get('ev',0):>+8.2f} {s1.get('win_rate',0):>7.1f}% {s1.get('total',0):>+7.1f}% "
              f"{s2.get('ev',0):>+8.2f} {s2.get('win_rate',0):>7.1f}% {s2.get('total',0):>+7.1f}% {verdict:>6}")

    print(f"\n  * 为此前已验证品种（参考值）")


if __name__ == '__main__':
    main()
