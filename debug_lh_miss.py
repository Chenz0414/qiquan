# -*- coding: utf-8 -*-
"""
诊断：生猪3月12日-4月2日这段下跌，10min策略为什么没抓住？
"""

import os, sys, time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqsdk import TqApi, TqAuth
import config as cfg
from signal_core import SignalDetector, SYMBOL_CONFIGS

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def main():
    api = TqApi(auth=TqAuth(cfg.TQ_ACCOUNT, cfg.TQ_PASSWORD))

    # 拉取10min数据
    klines = api.get_kline_serial("KQ.m@DCE.lh", duration_seconds=600, data_length=8964)
    deadline = time.time() + 60
    while True:
        api.wait_update(deadline=time.time() + 5)
        if len(klines) > 0 and not pd.isna(klines.iloc[-1]['close']):
            break
        if time.time() > deadline:
            break

    df = klines.copy()
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ns')
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.dropna(subset=['close']).reset_index(drop=True)
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()

    # 同时拉60min
    klines60 = api.get_kline_serial("KQ.m@DCE.lh", duration_seconds=3600, data_length=2000)
    api.wait_update(deadline=time.time() + 10)
    df60 = klines60.copy()
    df60['datetime'] = pd.to_datetime(df60['datetime'], unit='ns')
    df60 = df60[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
    df60 = df60.dropna(subset=['close']).reset_index(drop=True)
    df60['ma10'] = df60['close'].rolling(10).mean()
    df60['ma20'] = df60['close'].rolling(20).mean()

    api.close()

    # 只看3月12日-4月2日
    start = pd.Timestamp('2026-03-12')
    end = pd.Timestamp('2026-04-03')
    mask = (df['datetime'] >= start) & (df['datetime'] <= end)
    seg = df[mask].copy().reset_index(drop=True)

    mask60 = (df60['datetime'] >= start) & (df60['datetime'] <= end)
    seg60 = df60[mask60].copy().reset_index(drop=True)

    print(f"10min数据: {len(seg)}根 ({seg['datetime'].iloc[0]} ~ {seg['datetime'].iloc[-1]})")
    print(f"60min数据: {len(seg60)}根 ({seg60['datetime'].iloc[0]} ~ {seg60['datetime'].iloc[-1]})")
    print(f"价格范围: {seg['high'].max():.0f} ~ {seg['low'].min():.0f}")

    # ============================================================
    # 1. 用现有策略(10min, >=4根)扫描
    # ============================================================
    print(f"\n{'='*65}")
    print(f"  10min >=4根 (现有策略)")
    print(f"{'='*65}")
    detector4 = SignalDetector(min_pb_bars=4)
    sigs4 = []
    for i in range(len(seg)):
        row = seg.iloc[i]
        if pd.isna(row['ma10']) or pd.isna(row['ma20']):
            continue
        r = detector4.process_bar(close=row['close'], high=row['high'], low=row['low'],
                                   ma_fast=row['ma10'], ma_slow=row['ma20'])
        if r:
            sigs4.append({'i': i, 'time': row['datetime'], 'dir': r.direction,
                         'price': r.entry_price, 'pb': r.pullback_bars})

    for s in sigs4:
        d = "空" if s['dir'] == 'short' else "多"
        print(f"  {s['time']}  {d}  入场{s['price']:.0f}  回调{s['pb']}根")
    print(f"  共 {len(sigs4)} 个信号 (空: {sum(1 for s in sigs4 if s['dir']=='short')})")

    # ============================================================
    # 2. 放宽到 >=2根
    # ============================================================
    print(f"\n{'='*65}")
    print(f"  10min >=2根 (放宽)")
    print(f"{'='*65}")
    detector2 = SignalDetector(min_pb_bars=2)
    sigs2 = []
    for i in range(len(seg)):
        row = seg.iloc[i]
        if pd.isna(row['ma10']) or pd.isna(row['ma20']):
            continue
        r = detector2.process_bar(close=row['close'], high=row['high'], low=row['low'],
                                   ma_fast=row['ma10'], ma_slow=row['ma20'])
        if r:
            sigs2.append({'i': i, 'time': row['datetime'], 'dir': r.direction,
                         'price': r.entry_price, 'pb': r.pullback_bars})

    for s in sigs2:
        d = "空" if s['dir'] == 'short' else "多"
        print(f"  {s['time']}  {d}  入场{s['price']:.0f}  回调{s['pb']}根")
    print(f"  共 {len(sigs2)} 个信号 (空: {sum(1 for s in sigs2 if s['dir']=='short')})")

    # ============================================================
    # 3. 60min >=2根
    # ============================================================
    print(f"\n{'='*65}")
    print(f"  60min >=2根")
    print(f"{'='*65}")
    detector60_2 = SignalDetector(min_pb_bars=2)
    sigs60_2 = []
    for i in range(len(seg60)):
        row = seg60.iloc[i]
        if pd.isna(row['ma10']) or pd.isna(row['ma20']):
            continue
        r = detector60_2.process_bar(close=row['close'], high=row['high'], low=row['low'],
                                      ma_fast=row['ma10'], ma_slow=row['ma20'])
        if r:
            sigs60_2.append({'i': i, 'time': row['datetime'], 'dir': r.direction,
                            'price': r.entry_price, 'pb': r.pullback_bars})

    for s in sigs60_2:
        d = "空" if s['dir'] == 'short' else "多"
        print(f"  {s['time']}  {d}  入场{s['price']:.0f}  回调{s['pb']}根")
    print(f"  共 {len(sigs60_2)} 个信号 (空: {sum(1 for s in sigs60_2 if s['dir']=='short')})")

    # ============================================================
    # 4. 60min >=4根
    # ============================================================
    print(f"\n{'='*65}")
    print(f"  60min >=4根")
    print(f"{'='*65}")
    detector60_4 = SignalDetector(min_pb_bars=4)
    sigs60_4 = []
    for i in range(len(seg60)):
        row = seg60.iloc[i]
        if pd.isna(row['ma10']) or pd.isna(row['ma20']):
            continue
        r = detector60_4.process_bar(close=row['close'], high=row['high'], low=row['low'],
                                      ma_fast=row['ma10'], ma_slow=row['ma20'])
        if r:
            sigs60_4.append({'i': i, 'time': row['datetime'], 'dir': r.direction,
                            'price': r.entry_price, 'pb': r.pullback_bars})

    for s in sigs60_4:
        d = "空" if s['dir'] == 'short' else "多"
        print(f"  {s['time']}  {d}  入场{s['price']:.0f}  回调{s['pb']}根")
    print(f"  共 {len(sigs60_4)} 个信号 (空: {sum(1 for s in sigs60_4 if s['dir']=='short')})")

    # ============================================================
    # 5. 诊断: 10min图上这段为什么信号少？
    # ============================================================
    print(f"\n{'='*65}")
    print(f"  诊断：10min图上价格与MA的关系")
    print(f"{'='*65}")

    # 每天统计收盘价vs MA关系
    seg['date'] = seg['datetime'].dt.date
    for date, group in seg.groupby('date'):
        last = group.iloc[-1]
        close = last['close']
        ma10 = last['ma10']
        ma20 = last['ma20']
        if pd.isna(ma10) or pd.isna(ma20):
            continue
        # 统计这天有多少根K线 close > ma10（即回调到MA上方的根数）
        above_ma10 = (group['close'] > group['ma10']).sum()
        total = len(group)
        diff = close - ma10
        print(f"  {date} | 收:{close:.0f} MA10:{ma10:.0f} 差:{diff:+.0f} | "
              f"收盘>MA10: {above_ma10}/{total}根 | MA10{'>' if ma10>ma20 else '<'}MA20")

    # ============================================================
    # 6. 画图
    # ============================================================
    fig, axes = plt.subplots(2, 1, figsize=(18, 12))

    # 10min图
    ax = axes[0]
    ax.plot(seg['datetime'], seg['close'], color='#333', linewidth=0.8, label='收盘价')
    ax.plot(seg['datetime'], seg['ma10'], color='#00bcd4', linewidth=1, label='MA10', alpha=0.8)
    ax.plot(seg['datetime'], seg['ma20'], color='#e91e63', linewidth=1, label='MA20', alpha=0.8)

    for s in sigs4:
        color = '#f44336' if s['dir'] == 'short' else '#2196f3'
        marker = 'v' if s['dir'] == 'short' else '^'
        ax.scatter(s['time'], s['price'], color=color, marker=marker, s=100, zorder=5)
        ax.annotate(f"{'空' if s['dir']=='short' else '多'}{s['pb']}根\n{s['price']:.0f}",
                   xy=(s['time'], s['price']), fontsize=8,
                   textcoords="offset points", xytext=(10, 10 if s['dir']=='long' else -20))

    # 标注 >=2根的额外信号（用小标记）
    extra2 = [s for s in sigs2 if not any(abs((s['time']-s4['time']).total_seconds()) < 600 for s4 in sigs4)]
    for s in extra2:
        color = '#ff9800' if s['dir'] == 'short' else '#9c27b0'
        marker = 'v' if s['dir'] == 'short' else '^'
        ax.scatter(s['time'], s['price'], color=color, marker=marker, s=50, zorder=4, alpha=0.7)

    ax.set_title('生猪 10min (3/12-4/2) | 大三角=>=4根信号 | 小三角=>=2根额外信号', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.2)

    # 60min图
    ax = axes[1]
    ax.plot(seg60['datetime'], seg60['close'], color='#333', linewidth=1, label='收盘价')
    ax.plot(seg60['datetime'], seg60['ma10'], color='#00bcd4', linewidth=1.2, label='MA10')
    ax.plot(seg60['datetime'], seg60['ma20'], color='#e91e63', linewidth=1.2, label='MA20')

    for s in sigs60_2:
        color = '#f44336' if s['dir'] == 'short' else '#2196f3'
        marker = 'v' if s['dir'] == 'short' else '^'
        size = 120 if s['pb'] >= 4 else 60
        ax.scatter(s['time'], s['price'], color=color, marker=marker, s=size, zorder=5)
        ax.annotate(f"{'空' if s['dir']=='short' else '多'}{s['pb']}根\n{s['price']:.0f}",
                   xy=(s['time'], s['price']), fontsize=9, fontweight='bold',
                   textcoords="offset points", xytext=(10, 10 if s['dir']=='long' else -20))

    ax.set_title('生猪 60min (3/12-4/2) | 信号标注(>=2根)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(os.path.dirname(__file__), 'output', 'lh_miss_diagnosis.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图表: {path}")


if __name__ == '__main__':
    main()
