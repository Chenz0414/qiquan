# -*- coding: utf-8 -*-
"""
多周期策略测试：10min定方向 + 2min找入场
=========================================
当10min处于强趋势阶段时，降到2min级别做顺势回调。
用生猪3/12-4/2那段瀑布行情验证。
同时对比其他品种的强趋势段。
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

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

SYMBOLS = [
    ("KQ.m@DCE.lh",  "DCE.lh",  "生猪"),
    ("KQ.m@SHFE.ag", "SHFE.ag", "白银"),
    ("KQ.m@SHFE.au", "SHFE.au", "黄金"),
    ("KQ.m@INE.sc",  "INE.sc",  "原油"),
]


def compute_atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def fetch_klines(api, tq_symbol, period_sec, n_bars):
    """拉取K线"""
    klines = api.get_kline_serial(tq_symbol, duration_seconds=period_sec, data_length=n_bars)
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
    return df


def prepare_10min(df):
    """10min数据加指标+阶段分类"""
    df = df.copy()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['atr'] = compute_atr(df, 14)
    df['ma_slope'] = (df['ma10'] - df['ma10'].shift(5)) / df['atr']
    df['ma_width'] = (df['ma10'] - df['ma20']) / df['atr']

    # 用分位数分类（P80=强，P40=震荡）
    abs_slope = df['ma_slope'].abs().dropna()
    abs_width = df['ma_width'].abs().dropna()
    if len(abs_slope) > 0:
        s80 = np.percentile(abs_slope, 80)
        s40 = np.percentile(abs_slope, 40)
        w80 = np.percentile(abs_width, 80)
        w40 = np.percentile(abs_width, 40)
    else:
        s80, s40, w80, w40 = 1.5, 0.5, 1.0, 0.3

    def classify(row):
        if pd.isna(row['ma_slope']) or pd.isna(row['ma_width']):
            return 'unknown'
        s = abs(row['ma_slope'])
        w = abs(row['ma_width'])
        if s >= s80 and w >= w80:
            return 'strong'
        elif s >= s40 or w >= w40:
            return 'normal'
        return 'chop'

    df['phase'] = df.apply(classify, axis=1)
    df['trend_dir'] = np.where(df['ma_width'] > 0, 'up', 'down')
    return df


def prepare_2min(df):
    """2min数据加均线"""
    df = df.copy()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    return df


def get_10min_phase_at(df10, dt):
    """查询某时刻的10min阶段"""
    # 找到<=dt的最后一根10min K线
    mask = df10['datetime'] <= dt
    if mask.sum() == 0:
        return 'unknown', 'unknown'
    row = df10[mask].iloc[-1]
    return row['phase'], row['trend_dir']


def run_2min_in_strong(df10, df2, symbol_key, min_pb_bars=3):
    """在10min强趋势期间，用2min找回调信号"""
    sym_cfg = SYMBOL_CONFIGS.get(symbol_key, {"tick_size": 1.0})
    tick_size = sym_cfg["tick_size"]

    detector = SignalDetector(min_pb_bars=min_pb_bars)
    warmup = 25

    signals = []
    for i in range(warmup, len(df2)):
        row = df2.iloc[i]
        if pd.isna(row['ma10']) or pd.isna(row['ma20']):
            continue

        # 查10min阶段
        phase, trend_dir = get_10min_phase_at(df10, row['datetime'])

        result = detector.process_bar(
            close=row['close'], high=row['high'], low=row['low'],
            ma_fast=row['ma10'], ma_slow=row['ma20'],
        )
        if result is not None:
            signals.append({
                'idx': i, 'signal': result,
                'phase_10m': phase, 'trend_dir_10m': trend_dir,
                'time': row['datetime'],
            })

    # 分类过滤
    all_trades = []  # 所有2min信号
    strong_trades = []  # 只在10min强趋势期+顺势的

    for sig_info in signals:
        idx = sig_info['idx']
        sig = sig_info['signal']

        tracker = ExitTracker(
            direction=sig.direction, entry_price=sig.entry_price,
            pullback_extreme=sig.pullback_extreme,
            tick_size=tick_size, stop_ticks=5,
        )
        s1_exit = s2_exit = None
        for j in range(idx + 1, min(idx + 200, len(df2))):  # 2min最多看200根(~7小时)
            bar = df2.iloc[j]
            prev_bar = df2.iloc[j - 1]
            if pd.isna(bar['ma10']):
                continue
            exits, _ = tracker.process_bar(
                close=bar['close'], high=bar['high'], low=bar['low'],
                ma_fast=bar['ma10'], prev_close=prev_bar['close'],
            )
            for ev in exits:
                if ev.strategy == 'S1' and s1_exit is None:
                    s1_exit = ev
                elif ev.strategy == 'S2' and s2_exit is None:
                    s2_exit = ev
            if tracker.all_done():
                break

        if s1_exit is None or s2_exit is None:
            forced = tracker.force_close(df2.iloc[min(idx + 199, len(df2) - 1)]['close'])
            for ev in forced:
                if ev.strategy == 'S1' and s1_exit is None:
                    s1_exit = ev
                if ev.strategy == 'S2' and s2_exit is None:
                    s2_exit = ev

        trade = {
            'time': sig_info['time'],
            'direction': sig.direction,
            'pb_bars': sig.pullback_bars,
            'entry_price': sig.entry_price,
            'phase_10m': sig_info['phase_10m'],
            'trend_dir_10m': sig_info['trend_dir_10m'],
            's1_pnl': s1_exit.pnl_pct if s1_exit else 0,
            's2_pnl': s2_exit.pnl_pct if s2_exit else 0,
            's1_reason': s1_exit.exit_reason if s1_exit else 'open',
        }
        all_trades.append(trade)

        # 强趋势+顺势
        if sig_info['phase_10m'] == 'strong':
            is_wt = ((sig.direction == 'long' and sig_info['trend_dir_10m'] == 'up') or
                     (sig.direction == 'short' and sig_info['trend_dir_10m'] == 'down'))
            if is_wt:
                strong_trades.append(trade)

    return all_trades, strong_trades


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


def fmt_stats(stats, label):
    if not stats:
        return f"  {label}: 无信号"
    v = "有效" if stats['ev'] > 0.3 else ("边缘" if stats['ev'] > 0 else "无效")
    flag = " (样本少)" if stats['count'] < 15 else ""
    return (f"  {label}: {stats['count']}笔 胜率{stats['win_rate']:.1f}% "
            f"盈亏比{stats['pf']:.2f} EV={stats['ev']:+.2f} "
            f"累计{stats['total']:+.2f}% 回撤{stats['max_dd']:.2f}% -> {v}{flag}")


def main():
    print("=" * 70)
    print("  多周期策略：10min定方向 + 2min找入场")
    print("=" * 70)

    api = TqApi(auth=TqAuth(cfg.TQ_ACCOUNT, cfg.TQ_PASSWORD))

    all_results = {}

    for tq_sym, sym_key, name in SYMBOLS:
        print(f"\n{'='*70}")
        print(f"  {name} ({sym_key})")
        print(f"{'='*70}")

        # 拉10min (120天)
        print(f"  拉取10min...")
        df10 = fetch_klines(api, tq_sym, 600, 8964)
        df10 = prepare_10min(df10)

        # 拉2min (天勤限制，最多8964根 ≈ 约35天的2min数据)
        print(f"  拉取2min...")
        df2 = fetch_klines(api, tq_sym, 120, 8964)
        df2 = prepare_2min(df2)

        print(f"  10min: {len(df10)}根 ({df10['datetime'].iloc[0]} ~ {df10['datetime'].iloc[-1]})")
        print(f"  2min:  {len(df2)}根 ({df2['datetime'].iloc[0]} ~ {df2['datetime'].iloc[-1]})")

        # 10min阶段分布（2min数据覆盖的时间段内）
        t_start = df2['datetime'].iloc[0]
        t_end = df2['datetime'].iloc[-1]
        overlap = df10[(df10['datetime'] >= t_start) & (df10['datetime'] <= t_end)]
        if len(overlap) > 0:
            pd_str = overlap['phase'].value_counts(normalize=True)
            strong_bars = (overlap['phase'] == 'strong').sum()
            total_bars = len(overlap)
            print(f"  2min覆盖范围内10min阶段: 强{pd_str.get('strong',0)*100:.0f}% "
                  f"正常{pd_str.get('normal',0)*100:.0f}% "
                  f"震荡{pd_str.get('chop',0)*100:.0f}% "
                  f"(强趋势{strong_bars}/{total_bars}根)")

        # 跑2min策略（不同回调根数）
        for pb in [2, 3, 4]:
            print(f"\n  --- 2min 回调>={pb}根 ---")
            all_trades, strong_trades = run_2min_in_strong(df10, df2, sym_key, min_pb_bars=pb)

            # 全部2min信号
            s1_all = calc_stats([t['s1_pnl'] for t in all_trades])
            print(fmt_stats(s1_all, f"全部2min信号 S1"))

            # 只在10min强趋势+顺势
            s1_strong = calc_stats([t['s1_pnl'] for t in strong_trades])
            print(fmt_stats(s1_strong, f"10min强趋势顺势 S1"))

            # 10min正常趋势期的2min信号
            normal_wt = [t for t in all_trades if t['phase_10m'] == 'normal' and
                        ((t['direction'] == 'long' and t['trend_dir_10m'] == 'up') or
                         (t['direction'] == 'short' and t['trend_dir_10m'] == 'down'))]
            s1_normal = calc_stats([t['s1_pnl'] for t in normal_wt])
            print(fmt_stats(s1_normal, f"10min正常趋势顺势 S1"))

            # 震荡期
            chop_trades = [t for t in all_trades if t['phase_10m'] == 'chop']
            s1_chop = calc_stats([t['s1_pnl'] for t in chop_trades])
            print(fmt_stats(s1_chop, f"10min震荡期 S1"))

            if pb == 3:  # 主力参数，存详细结果
                all_results[name] = {
                    'df10': df10, 'df2': df2,
                    'all_trades': all_trades, 'strong_trades': strong_trades,
                    's1_strong': s1_strong,
                    'normal_wt': normal_wt, 's1_normal': s1_normal,
                }

                # 逐笔打印强趋势信号
                if strong_trades:
                    print(f"\n  强趋势顺势信号逐笔:")
                    print(f"  {'时间':<20} {'方向':>4} {'入场':>8} {'回调':>4} {'S1':>8}")
                    for t in strong_trades:
                        d = "空" if t['direction'] == 'short' else "多"
                        print(f"  {str(t['time']):<20} {d:>4} {t['entry_price']:>8.1f} "
                              f"{t['pb_bars']:>4} {t['s1_pnl']:>+8.2f}%")

    api.close()

    # ============================================================
    #  汇总
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  汇总: 2min回调>=3根 各阶段表现 (S1)")
    print(f"{'='*70}")
    print(f"  {'品种':<6} | {'强趋势顺势':^26} | {'正常趋势顺势':^26} | {'震荡期':^26}")
    print(f"  {'':6} | {'笔':>3} {'胜率':>6} {'EV':>7} {'累计':>7} | {'笔':>3} {'胜率':>6} {'EV':>7} {'累计':>7} | {'笔':>3} {'胜率':>6} {'EV':>7} {'累计':>7}")
    print(f"  {'-'*86}")
    for name, r in all_results.items():
        def g(s):
            if s:
                return s['count'], s['win_rate'], s['ev'], s['total']
            return 0, 0, 0, 0
        sc, sw, se, st = g(r['s1_strong'])
        nc, nw, ne, nt = g(r['s1_normal'])
        cc, cw, ce, ct = g(calc_stats([t['s1_pnl'] for t in r['all_trades'] if t['phase_10m'] == 'chop']))
        print(f"  {name:<6} | {sc:>3} {sw:>5.1f}% {se:>+7.2f} {st:>+6.2f}% | "
              f"{nc:>3} {nw:>5.1f}% {ne:>+7.2f} {nt:>+6.2f}% | "
              f"{cc:>3} {cw:>5.1f}% {ce:>+7.2f} {ct:>+6.2f}%")

    # ============================================================
    #  绘图
    # ============================================================
    n = len(all_results)
    fig, axes = plt.subplots(n, 2, figsize=(18, 5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for row_idx, (name, r) in enumerate(all_results.items()):
        df10 = r['df10']
        df2 = r['df2']
        t_start = df2['datetime'].iloc[0]
        seg10 = df10[df10['datetime'] >= t_start]

        # 左图：10min价格+阶段+2min强趋势信号
        ax = axes[row_idx][0]
        ax.plot(seg10['datetime'], seg10['close'], color='#333', linewidth=0.8)
        ax.plot(seg10['datetime'], seg10['ma10'], color='#00bcd4', linewidth=0.7, alpha=0.7)
        ax.plot(seg10['datetime'], seg10['ma20'], color='#e91e63', linewidth=0.7, alpha=0.7)

        cmap = {'strong': '#4caf50', 'normal': '#ff9800', 'chop': '#f44336', 'unknown': '#9e9e9e'}
        prev_ph = None
        start_dt = None
        for _, bar in seg10.iterrows():
            if bar['phase'] != prev_ph:
                if prev_ph and start_dt:
                    ax.axvspan(start_dt, bar['datetime'], alpha=0.12, color=cmap.get(prev_ph, '#9e9e9e'))
                start_dt = bar['datetime']
                prev_ph = bar['phase']
        if prev_ph and start_dt:
            ax.axvspan(start_dt, seg10['datetime'].iloc[-1], alpha=0.12, color=cmap.get(prev_ph, '#9e9e9e'))

        for t in r['strong_trades']:
            c = '#f44336' if t['direction'] == 'short' else '#2196f3'
            m = 'v' if t['direction'] == 'short' else '^'
            ax.scatter(t['time'], t['entry_price'], color=c, marker=m, s=80, zorder=5, edgecolors='white', linewidths=0.5)

        ax.set_title(f"{name} 10min + 2min强趋势信号(>=3根)", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.15)
        ax.tick_params(labelsize=8)

        # 右图：三条累计收益线
        ax = axes[row_idx][1]
        if r['s1_strong'] and len(r['s1_strong']['cum']) > 0:
            ax.plot(r['s1_strong']['cum'], color='#4caf50', linewidth=2,
                   label=f"强趋势顺势 {r['s1_strong']['count']}笔 EV={r['s1_strong']['ev']:+.2f}")
        if r['s1_normal'] and len(r['s1_normal']['cum']) > 0:
            ax.plot(r['s1_normal']['cum'], color='#ff9800', linewidth=1.5,
                   label=f"正常趋势顺势 {r['s1_normal']['count']}笔 EV={r['s1_normal']['ev']:+.2f}")
        chop_pnls = [t['s1_pnl'] for t in r['all_trades'] if t['phase_10m'] == 'chop']
        chop_stats = calc_stats(chop_pnls)
        if chop_stats and len(chop_stats['cum']) > 0:
            ax.plot(chop_stats['cum'], color='#f44336', linewidth=1, alpha=0.7,
                   label=f"震荡期 {chop_stats['count']}笔 EV={chop_stats['ev']:+.2f}")

        ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
        ax.set_title(f"{name} 2min S1 各阶段累计收益", fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.15)

    plt.suptitle('多周期策略: 10min定阶段 + 2min回调入场 (>=3根)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'test_2min_strong.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图表: {path}")


if __name__ == '__main__':
    main()
