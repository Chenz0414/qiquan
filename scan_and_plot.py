# -*- coding: utf-8 -*-
"""
扫描白银过去30天的10min信号，绘制K线+入场出场标注
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch
from datetime import datetime, timedelta
from tqsdk import TqApi, TqAuth
import config as cfg
from signal_core import SignalDetector, ExitTracker

# 中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def fetch_and_scan():
    """拉取白银10min数据，检测信号+模拟出场"""
    symbol = "KQ.m@SHFE.ag"
    tick_size = 1.0
    period_min = 10
    scan_days = 30 + 50  # 30天数据 + 50天预热

    print("连接天勤拉取数据...")
    api = TqApi(auth=TqAuth(cfg.TQ_ACCOUNT, cfg.TQ_PASSWORD))

    bars_per_day = int(9.5 * 60 / period_min) + 5
    total_bars = min(bars_per_day * scan_days, 8964)

    klines = api.get_kline_serial(symbol, duration_seconds=period_min * 60, data_length=total_bars)
    api.wait_update()

    df = klines.copy()
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ns')
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.dropna(subset=['close'])
    df = df.reset_index(drop=True)

    api.close()
    print(f"共 {len(df)} 根K线, {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

    # 计算均线
    df['ma_fast'] = df['close'].rolling(window=cfg.MA_FAST).mean()
    df['ma_slow'] = df['close'].rolling(window=cfg.MA_SLOW).mean()

    # 信号检测
    detector = SignalDetector(min_pb_bars=4)
    signals = []

    warmup = cfg.MA_SLOW + 5
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
            })

    # 只保留最近30天内的信号
    cutoff = df['datetime'].iloc[-1] - timedelta(days=30)
    signals = [s for s in signals if df.iloc[s['idx']]['datetime'] >= cutoff]
    print(f"最近30天共 {len(signals)} 个信号")

    # 模拟出场
    trades = []
    for sig_info in signals:
        idx = sig_info['idx']
        sig = sig_info['signal']

        tracker = ExitTracker(
            direction=sig.direction,
            entry_price=sig.entry_price,
            pullback_extreme=sig.pullback_extreme,
            tick_size=tick_size,
            stop_ticks=5,
        )

        s1_exit = None
        s2_exit = None
        # 追踪S1/S2止损变化用于标注
        s1_stops = [(idx, tracker.s1_stop)]
        s2_stops = [(idx, tracker.s2_stop)]

        for j in range(idx + 1, len(df)):
            bar = df.iloc[j]
            prev_bar = df.iloc[j - 1]
            if pd.isna(bar['ma_fast']):
                continue

            exits, updates = tracker.process_bar(
                close=bar['close'], high=bar['high'], low=bar['low'],
                ma_fast=bar['ma_fast'], prev_close=prev_bar['close'],
            )

            s1_stops.append((j, tracker.s1_stop))
            s2_stops.append((j, tracker.s2_stop))

            for ev in exits:
                if ev.strategy == 'S1' and s1_exit is None:
                    s1_exit = {'idx': j, 'price': ev.exit_price, 'pnl': ev.pnl_pct,
                               'bars': ev.bars_held, 'reason': ev.exit_reason}
                elif ev.strategy == 'S2' and s2_exit is None:
                    s2_exit = {'idx': j, 'price': ev.exit_price, 'pnl': ev.pnl_pct,
                               'bars': ev.bars_held, 'reason': ev.exit_reason}

            if tracker.all_done():
                break

        # 未平仓的用最后价格
        if s1_exit is None:
            s1_exit = {'idx': len(df)-1, 'price': df.iloc[-1]['close'], 'pnl': 0, 'bars': 0, 'reason': 'open'}
        if s2_exit is None:
            s2_exit = {'idx': len(df)-1, 'price': df.iloc[-1]['close'], 'pnl': 0, 'bars': 0, 'reason': 'open'}

        trades.append({
            'entry_idx': idx,
            'signal': sig,
            's1_exit': s1_exit,
            's2_exit': s2_exit,
            's1_stops': s1_stops,
            's2_stops': s2_stops,
        })

    return df, trades


def plot_trade(df, trade, trade_num, output_dir):
    """绘制单笔交易的K线图"""
    entry_idx = trade['entry_idx']
    sig = trade['signal']
    s1 = trade['s1_exit']
    s2 = trade['s2_exit']

    # 显示范围：入场前20根 ~ 最远出场后10根
    last_exit_idx = max(s1['idx'], s2['idx'])
    start = max(0, entry_idx - 20)
    end = min(len(df), last_exit_idx + 10)
    sub = df.iloc[start:end].copy()
    sub = sub.reset_index(drop=True)

    # 相对index
    entry_rel = entry_idx - start
    s1_rel = s1['idx'] - start
    s2_rel = s2['idx'] - start

    fig, ax = plt.subplots(1, 1, figsize=(16, 7))

    # 绘制K线
    for i in range(len(sub)):
        row = sub.iloc[i]
        color = '#d32f2f' if row['close'] < row['open'] else '#2e7d32'
        # K线实体
        body_bottom = min(row['open'], row['close'])
        body_height = abs(row['close'] - row['open'])
        if body_height < 0.01:
            body_height = 0.01
        ax.bar(i, body_height, bottom=body_bottom, width=0.6, color=color, edgecolor=color, linewidth=0.5)
        # 上下影线
        ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=0.8)

    # 绘制10MA和20MA
    if 'ma_fast' in sub.columns:
        ma_fast_vals = sub['ma_fast'].values
        ma_slow_vals = sub['ma_slow'].values
        valid_fast = ~np.isnan(ma_fast_vals)
        valid_slow = ~np.isnan(ma_slow_vals)
        x_range = np.arange(len(sub))
        ax.plot(x_range[valid_fast], ma_fast_vals[valid_fast], color='#1976d2', linewidth=1.2, label='10MA', alpha=0.8)
        ax.plot(x_range[valid_slow], ma_slow_vals[valid_slow], color='#f57c00', linewidth=1.2, label='20MA', alpha=0.8)

    # 标注入场点
    entry_price = sig.entry_price
    dir_label = "做多" if sig.direction == "long" else "做空"
    marker = '^' if sig.direction == 'long' else 'v'
    ax.scatter(entry_rel, entry_price, marker=marker, s=200, color='#1565c0', zorder=10, edgecolors='white', linewidths=1.5)
    ax.annotate(f'入场 {entry_price:.0f}\n{dir_label} 回调{sig.pullback_bars}根',
                xy=(entry_rel, entry_price),
                xytext=(entry_rel + 1.5, entry_price + (sub['high'].max() - sub['low'].min()) * 0.08),
                fontsize=9, fontweight='bold', color='#1565c0',
                arrowprops=dict(arrowstyle='->', color='#1565c0', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#e3f2fd', edgecolor='#1565c0', alpha=0.9))

    # 标注S1出场
    s1_color = '#2e7d32' if s1['pnl'] >= 0 else '#d32f2f'
    s1_label = f"S1出场 {s1['price']:.0f}\n{s1['pnl']:+.2f}% ({s1['bars']}根)"
    if s1['reason'] == 'open':
        s1_label = f"S1持仓中 {s1['price']:.0f}"
    ax.scatter(s1_rel, s1['price'], marker='x', s=180, color=s1_color, zorder=10, linewidths=2.5)
    offset_y = -(sub['high'].max() - sub['low'].min()) * 0.08
    ax.annotate(s1_label,
                xy=(s1_rel, s1['price']),
                xytext=(s1_rel + 1.2, s1['price'] + offset_y),
                fontsize=8, fontweight='bold', color=s1_color,
                arrowprops=dict(arrowstyle='->', color=s1_color, lw=1),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5e9' if s1['pnl'] >= 0 else '#ffebee',
                          edgecolor=s1_color, alpha=0.9))

    # 标注S2出场
    s2_color = '#2e7d32' if s2['pnl'] >= 0 else '#d32f2f'
    s2_label = f"S2出场 {s2['price']:.0f}\n{s2['pnl']:+.2f}% ({s2['bars']}根)"
    if s2['reason'] == 'open':
        s2_label = f"S2持仓中 {s2['price']:.0f}"
    ax.scatter(s2_rel, s2['price'], marker='D', s=120, color=s2_color, zorder=10, linewidths=1.5, edgecolors='white')
    offset_y2 = (sub['high'].max() - sub['low'].min()) * 0.12
    ax.annotate(s2_label,
                xy=(s2_rel, s2['price']),
                xytext=(s2_rel + 1.2, s2['price'] + offset_y2),
                fontsize=8, fontweight='bold', color=s2_color,
                arrowprops=dict(arrowstyle='->', color=s2_color, lw=1),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5e9' if s2['pnl'] >= 0 else '#ffebee',
                          edgecolor=s2_color, alpha=0.9))

    # 绘制初始止损线
    init_stop = sig.pullback_extreme - 5 * 1.0 if sig.direction == 'long' else sig.pullback_extreme + 5 * 1.0
    ax.axhline(y=init_stop, xmin=0, xmax=1, color='#d32f2f', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.text(len(sub) - 1, init_stop, f'  初始止损 {init_stop:.0f}', fontsize=7, color='#d32f2f', alpha=0.7,
            verticalalignment='bottom')

    # 回调区间高亮
    pb_start_rel = entry_rel - sig.pullback_bars
    if pb_start_rel >= 0:
        ax.axvspan(pb_start_rel, entry_rel, alpha=0.06, color='#1565c0')
        ax.text((pb_start_rel + entry_rel) / 2, sub['low'].min(), '回调区间',
                fontsize=7, color='#1565c0', alpha=0.6, ha='center')

    # X轴标签：显示日期时间
    tick_positions = list(range(0, len(sub), max(1, len(sub) // 8)))
    tick_labels = [sub.iloc[i]['datetime'].strftime('%m/%d\n%H:%M') for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=7)

    # 入场时间
    entry_time = df.iloc[entry_idx]['datetime'].strftime('%Y-%m-%d %H:%M')

    ax.set_title(f'白银 10min | 交易#{trade_num} | {entry_time} | {dir_label} | '
                 f'S1: {s1["pnl"]:+.2f}%  S2: {s2["pnl"]:+.2f}%',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_ylabel('价格', fontsize=10)

    plt.tight_layout()
    path = os.path.join(output_dir, f'trade_{trade_num:02d}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def main():
    df, trades = fetch_and_scan()

    if not trades:
        print("没有检测到信号！")
        return

    output_dir = os.path.join(os.path.dirname(__file__), 'output', 'ag_trades_chart')
    os.makedirs(output_dir, exist_ok=True)

    # 只画前10个
    n = min(10, len(trades))
    print(f"\n绘制前 {n} 笔交易K线图...")

    for i in range(n):
        t = trades[i]
        sig = t['signal']
        entry_time = df.iloc[t['entry_idx']]['datetime'].strftime('%m/%d %H:%M')
        dir_label = "多" if sig.direction == "long" else "空"
        print(f"  #{i+1} {entry_time} {dir_label} 回调{sig.pullback_bars}根 | "
              f"S1:{t['s1_exit']['pnl']:+.2f}% S2:{t['s2_exit']['pnl']:+.2f}%")
        path = plot_trade(df, t, i + 1, output_dir)
        print(f"    → {path}")

    # 汇总统计
    print(f"\n===== 前{n}笔汇总 =====")
    s1_pnls = [t['s1_exit']['pnl'] for t in trades[:n] if t['s1_exit']['reason'] != 'open']
    s2_pnls = [t['s2_exit']['pnl'] for t in trades[:n] if t['s2_exit']['reason'] != 'open']
    if s1_pnls:
        wins = sum(1 for p in s1_pnls if p > 0)
        print(f"  S1: {len(s1_pnls)}笔完成 | 胜率{wins/len(s1_pnls)*100:.0f}% | 累计{sum(s1_pnls):+.2f}%")
    if s2_pnls:
        wins = sum(1 for p in s2_pnls if p > 0)
        print(f"  S2: {len(s2_pnls)}笔完成 | 胜率{wins/len(s2_pnls)*100:.0f}% | 累计{sum(s2_pnls):+.2f}%")

    print(f"\n图表已保存到: {output_dir}")


if __name__ == '__main__':
    main()
