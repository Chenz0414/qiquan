# -*- coding: utf-8 -*-
"""
对比：有20MA趋势过滤 vs 只看10MA（无趋势过滤）
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import timedelta
from tqsdk import TqApi, TqAuth
import config as cfg
from signal_core import SignalDetector, ExitTracker

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class SignalDetectorNoTrend:
    """
    只看10MA版本：去掉20MA趋势过滤。
    只要 close跌破10MA -> 收回10MA，回调>=4根，就触发。
    方向由回调前价格在10MA哪边决定（跌破=多头回调，突破=空头回调）。
    """
    def __init__(self, min_pb_bars=4):
        self.min_pb_bars = min_pb_bars
        self.signal_count = 0
        # 多头回调追踪（价格从上方跌破10MA）
        self.long_below_start = -1
        self.long_pullback_low = None
        # 空头回调追踪（价格从下方突破10MA）
        self.short_above_start = -1
        self.short_pullback_high = None
        self._prev_close = None
        self._prev_ma_fast = None
        self._bar_index = -1

    def process_bar(self, close, high, low, ma_fast, ma_slow=None):
        self._bar_index += 1
        signal = None

        if self._prev_close is None:
            self._prev_close = close
            self._prev_ma_fast = ma_fast
            return None

        # === 多头回调检测（价格从10MA上方跌到下方，再收回） ===
        if self.long_below_start == -1:
            # 开始回调：前一根在10MA上方，当前跌到下方
            if close < ma_fast and self._prev_close >= self._prev_ma_fast:
                self.long_below_start = self._bar_index
                self.long_pullback_low = low
        else:
            self.long_pullback_low = min(self.long_pullback_low, low)
            if close > ma_fast:
                # 收回！
                pb_bars = self._bar_index - self.long_below_start
                if pb_bars >= self.min_pb_bars:
                    self.signal_count += 1
                    signal = type('Signal', (), {
                        'direction': 'long',
                        'entry_price': close,
                        'pullback_extreme': self.long_pullback_low,
                        'pullback_bars': pb_bars,
                        'signal_seq': self.signal_count,
                        'bar_index': self._bar_index,
                    })()
                self.long_below_start = -1
                self.long_pullback_low = None

        # === 空头回调检测（价格从10MA下方涨到上方，再跌回） ===
        if self.short_above_start == -1:
            if close > ma_fast and self._prev_close <= self._prev_ma_fast:
                self.short_above_start = self._bar_index
                self.short_pullback_high = high
        else:
            self.short_pullback_high = max(self.short_pullback_high, high)
            if close < ma_fast:
                pb_bars = self._bar_index - self.short_above_start
                if pb_bars >= self.min_pb_bars:
                    self.signal_count += 1
                    signal = type('Signal', (), {
                        'direction': 'short',
                        'entry_price': close,
                        'pullback_extreme': self.short_pullback_high,
                        'pullback_bars': pb_bars,
                        'signal_seq': self.signal_count,
                        'bar_index': self._bar_index,
                    })()
                self.short_above_start = -1
                self.short_pullback_high = None

        self._prev_close = close
        self._prev_ma_fast = ma_fast
        return signal


def run_scan(df, detector_class, label):
    """用指定的检测器扫描信号并模拟出场"""
    detector = detector_class(min_pb_bars=4)
    tick_size = 1.0
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

    # 只保留最近30天
    cutoff = df['datetime'].iloc[-1] - timedelta(days=30)
    signals = [s for s in signals if df.iloc[s['idx']]['datetime'] >= cutoff]

    # 模拟出场
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

        if s1_exit is None:
            forced = tracker.force_close(df.iloc[-1]['close'])
            for ev in forced:
                if ev.strategy == 'S1': s1_exit = ev
                elif ev.strategy == 'S2' and s2_exit is None: s2_exit = ev
        elif s2_exit is None:
            forced = tracker.force_close(df.iloc[-1]['close'])
            for ev in forced:
                if ev.strategy == 'S2': s2_exit = ev

        trades.append({
            'time': df.iloc[idx]['datetime'],
            'direction': sig.direction,
            'pb_bars': sig.pullback_bars,
            's1_pnl': s1_exit.pnl_pct if s1_exit else 0,
            's2_pnl': s2_exit.pnl_pct if s2_exit else 0,
            's1_reason': s1_exit.exit_reason if s1_exit else 'open',
            's2_reason': s2_exit.exit_reason if s2_exit else 'open',
            's1_bars': s1_exit.bars_held if s1_exit else 0,
            's2_bars': s2_exit.bars_held if s2_exit else 0,
        })

    return trades


def print_stats(trades, label):
    """打印统计"""
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  信号总数: {len(trades)}")

    if not trades:
        return {}

    # 按方向统计
    longs = [t for t in trades if t['direction'] == 'long']
    shorts = [t for t in trades if t['direction'] == 'short']
    print(f"  做多: {len(longs)}笔 | 做空: {len(shorts)}笔")

    stats = {}
    for strat in ['s1', 's2']:
        pnls = [t[f'{strat}_pnl'] for t in trades if t[f'{strat}_reason'] != 'open']
        if not pnls:
            continue
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) * 100
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0.001
        pf = avg_win / avg_loss if avg_loss > 0 else 999
        ev = win_rate / 100 * pf - (1 - win_rate / 100)
        total = sum(pnls)

        strat_label = "S1快刀" if strat == "s1" else "S2波段"
        print(f"\n  [{strat_label}]")
        print(f"    完成: {len(pnls)}笔 | 胜率: {win_rate:.1f}%")
        print(f"    平均盈利: +{avg_win:.2f}% | 平均亏损: -{avg_loss:.2f}%")
        print(f"    盈亏比: {pf:.2f} | 期望值(EV): {ev:+.2f}")
        print(f"    累计收益: {total:+.2f}%")

        # 累计曲线数据
        stats[strat] = {
            'pnls': pnls, 'win_rate': win_rate, 'pf': pf,
            'ev': ev, 'total': total, 'count': len(pnls),
        }

    return stats


def plot_comparison(stats_with_trend, stats_no_trend, output_dir):
    """绘制对比图"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    for strat, strat_label in [('s1', 'S1 快刀(大仓位)'), ('s2', 'S2 波段(小仓位)')]:
        ax = axes[0] if strat == 's1' else axes[1]

        # 有趋势过滤
        if strat in stats_with_trend:
            pnls = stats_with_trend[strat]['pnls']
            cum = np.cumsum(pnls)
            ax.plot(range(1, len(cum)+1), cum, 'b-o', markersize=4, linewidth=1.5,
                    label=f"有20MA过滤 ({len(pnls)}笔, 胜率{stats_with_trend[strat]['win_rate']:.0f}%, "
                          f"EV={stats_with_trend[strat]['ev']:+.2f})")

        # 无趋势过滤
        if strat in stats_no_trend:
            pnls = stats_no_trend[strat]['pnls']
            cum = np.cumsum(pnls)
            ax.plot(range(1, len(cum)+1), cum, 'r-s', markersize=4, linewidth=1.5,
                    label=f"只看10MA ({len(pnls)}笔, 胜率{stats_no_trend[strat]['win_rate']:.0f}%, "
                          f"EV={stats_no_trend[strat]['ev']:+.2f})")

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(strat_label, fontsize=13, fontweight='bold')
        ax.set_xlabel('交易序号')
        ax.set_ylabel('累计收益 %')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('白银 30天对比: 有20MA趋势过滤 vs 只看10MA', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, 'compare_trend_filter.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def main():
    # 拉数据
    symbol = "KQ.m@SHFE.ag"
    period_min = 10
    scan_days = 80

    print("连接天勤拉取白银10min数据...")
    api = TqApi(auth=TqAuth(cfg.TQ_ACCOUNT, cfg.TQ_PASSWORD))
    bars_per_day = int(9.5 * 60 / period_min) + 5
    total_bars = min(bars_per_day * scan_days, 8964)

    klines = api.get_kline_serial(symbol, duration_seconds=period_min * 60, data_length=total_bars)
    api.wait_update()

    df = klines.copy()
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ns')
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.dropna(subset=['close']).reset_index(drop=True)
    api.close()
    print(f"共 {len(df)} 根K线")

    df['ma_fast'] = df['close'].rolling(window=cfg.MA_FAST).mean()
    df['ma_slow'] = df['close'].rolling(window=cfg.MA_SLOW).mean()

    # 方案A：有20MA趋势过滤（原策略）
    trades_a = run_scan(df, SignalDetector, "有20MA趋势过滤")
    stats_a = print_stats(trades_a, "方案A: 10MA > 20MA 趋势过滤（原策略）")

    # 方案B：只看10MA
    trades_b = run_scan(df, SignalDetectorNoTrend, "只看10MA")
    stats_b = print_stats(trades_b, "方案B: 只看10MA（无趋势过滤）")

    # 对比
    print(f"\n{'='*55}")
    print(f"  对比总结")
    print(f"{'='*55}")
    print(f"  信号数量: A={len(trades_a)} vs B={len(trades_b)} (多{len(trades_b)-len(trades_a)}笔)")

    for strat, label in [('s1', 'S1'), ('s2', 'S2')]:
        ev_a = stats_a.get(strat, {}).get('ev', 0)
        ev_b = stats_b.get(strat, {}).get('ev', 0)
        total_a = stats_a.get(strat, {}).get('total', 0)
        total_b = stats_b.get(strat, {}).get('total', 0)
        print(f"  {label}: EV {ev_a:+.2f} vs {ev_b:+.2f} | 累计 {total_a:+.2f}% vs {total_b:+.2f}%")

    # 绘图
    output_dir = os.path.join(os.path.dirname(__file__), 'output', 'ag_trades_chart')
    os.makedirs(output_dir, exist_ok=True)
    path = plot_comparison(stats_a, stats_b, output_dir)
    print(f"\n对比图: {path}")

    # 额外：列出B方案中新增的那些信号（A没有B有的）
    a_times = set(t['time'] for t in trades_a)
    extra = [t for t in trades_b if t['time'] not in a_times]
    if extra:
        print(f"\n方案B比A多出的 {len(extra)} 笔信号:")
        for t in extra:
            dir_label = "多" if t['direction'] == "long" else "空"
            print(f"  {t['time'].strftime('%m/%d %H:%M')} {dir_label} 回调{t['pb_bars']}根 | "
                  f"S1:{t['s1_pnl']:+.2f}% S2:{t['s2_pnl']:+.2f}%")


if __name__ == '__main__':
    main()
