# -*- coding: utf-8 -*-
"""
ER过滤 + 三种出场 全品种回测
=============================
32品种120天，A+B类信号，ER(10/20/40)三档过滤，S1/S2/S3三种出场
趋势：EMA20 > EMA120
"""

import os
import numpy as np
import pandas as pd
from signal_core import SYMBOL_CONFIGS, ExitTracker

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")

MIN_PB_BARS = 4
LAST_DAYS = 120
BARS_PER_DAY = 57
STOP_TICKS = 5
ER_PERIODS = [10, 20, 40]
ER_BINS = [
    ('all',      0,    999),
    ('0~0.25',   0,    0.25),
    ('0.25~0.5', 0.25, 0.5),
    ('0.5+',     0.5,  999),
]

HIGH_VOL_KEYS = {"GFEX.lc","DCE.jm","SHFE.ag","CZCE.FG","CZCE.SA",
                 "INE.sc","CZCE.MA","CZCE.TA","DCE.eb","DCE.lh"}


def load_and_prepare(cache_key):
    path = os.path.join(CACHE_DIR, f"{cache_key}_10min_170d.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema120'] = df['close'].ewm(span=120, adjust=False).mean()
    for p in ER_PERIODS:
        net = (df['close'] - df['close'].shift(p)).abs()
        bar_sum = df['close'].diff().abs().rolling(p).sum()
        df[f'er_{p}'] = net / bar_sum.replace(0, np.nan)
    return df


def detect_signals_ab(df, start_idx):
    """检测A类+B类信号，返回信号列表"""
    signals = []
    n = len(df)
    trend_dir = 0
    # B类状态
    below_start = -1
    pb_low = pb_high = None
    prev_close = prev_ema10 = None

    for i in range(1, n):
        row = df.iloc[i]
        close, high, low = row['close'], row['high'], row['low']
        ema10, ema20, ema120 = row['ema10'], row['ema20'], row['ema120']

        if prev_close is None or pd.isna(ema120) or pd.isna(ema10):
            prev_close, prev_ema10 = close, ema10
            continue

        # 趋势判断
        curr_trend = 1 if ema20 > ema120 else (-1 if ema20 < ema120 else 0)
        if curr_trend != trend_dir and curr_trend != 0:
            trend_dir = curr_trend
            below_start, pb_low, pb_high = -1, None, None

        if trend_dir == 0:
            prev_close, prev_ema10 = close, ema10
            continue

        # ER值
        er_vals = {}
        for p in ER_PERIODS:
            v = row.get(f'er_{p}', np.nan)
            er_vals[p] = v if not pd.isna(v) else np.nan

        # ===== 多头 =====
        if trend_dir == 1:
            # A类：影线碰EMA10弹回（单根）
            if i >= start_idx and low <= ema10 and close > ema10 and prev_close > prev_ema10:
                # 确保不是B类回调中
                if below_start == -1:
                    signals.append({
                        'idx': i, 'type': 'A', 'direction': 'long',
                        'entry_price': close, 'pullback_extreme': low,
                        'pullback_bars': 0, 'time': row['datetime'],
                        'er_10': er_vals[10], 'er_20': er_vals[20], 'er_40': er_vals[40],
                    })

            # B类：实体跌破EMA10后收回
            if below_start == -1:
                if close < ema10 and prev_close >= prev_ema10:
                    below_start, pb_low = i, low
            else:
                pb_low = min(pb_low, low)
                if close > ema10:
                    pb_bars = i - below_start
                    if pb_bars >= MIN_PB_BARS and i >= start_idx:
                        signals.append({
                            'idx': i, 'type': 'B', 'direction': 'long',
                            'entry_price': close, 'pullback_extreme': pb_low,
                            'pullback_bars': pb_bars, 'time': row['datetime'],
                            'er_10': er_vals[10], 'er_20': er_vals[20], 'er_40': er_vals[40],
                        })
                    below_start, pb_low = -1, None

        # ===== 空头 =====
        elif trend_dir == -1:
            # A类
            if i >= start_idx and high >= ema10 and close < ema10 and prev_close < prev_ema10:
                if below_start == -1:
                    signals.append({
                        'idx': i, 'type': 'A', 'direction': 'short',
                        'entry_price': close, 'pullback_extreme': high,
                        'pullback_bars': 0, 'time': row['datetime'],
                        'er_10': er_vals[10], 'er_20': er_vals[20], 'er_40': er_vals[40],
                    })

            # B类
            if below_start == -1:
                if close > ema10 and prev_close <= prev_ema10:
                    below_start, pb_high = i, high
            else:
                pb_high = max(pb_high, high)
                if close < ema10:
                    pb_bars = i - below_start
                    if pb_bars >= MIN_PB_BARS and i >= start_idx:
                        signals.append({
                            'idx': i, 'type': 'B', 'direction': 'short',
                            'entry_price': close, 'pullback_extreme': pb_high,
                            'pullback_bars': pb_bars, 'time': row['datetime'],
                            'er_10': er_vals[10], 'er_20': er_vals[20], 'er_40': er_vals[40],
                        })
                    below_start, pb_high = -1, None

        prev_close, prev_ema10 = close, ema10
    return signals


def simulate_exits(df, signals, tick_size):
    """模拟S1/S2/S3出场，返回交易列表"""
    trades = []
    n = len(df)

    for sig in signals:
        entry_idx = sig['idx']
        entry_price = sig['entry_price']
        is_long = sig['direction'] == 'long'
        pb_extreme = sig['pullback_extreme']

        tracker = ExitTracker(
            direction=sig['direction'],
            entry_price=entry_price,
            pullback_extreme=pb_extreme,
            tick_size=tick_size,
            stop_ticks=STOP_TICKS,
        )

        results = {'S1': None, 'S2': None, 'S3': None}
        mfe = 0.0
        mae = 0.0

        for j in range(entry_idx + 1, n):
            bar = df.iloc[j]
            prev_bar = df.iloc[j - 1]

            if pd.isna(bar['ema10']):
                continue

            # MFE/MAE
            if is_long:
                fav = (bar['high'] - entry_price) / entry_price * 100
                adv = (entry_price - bar['low']) / entry_price * 100
            else:
                fav = (entry_price - bar['low']) / entry_price * 100
                adv = (bar['high'] - entry_price) / entry_price * 100
            mfe = max(mfe, fav)
            mae = max(mae, adv)

            if not tracker.all_done():
                exit_events, _ = tracker.process_bar(
                    close=bar['close'],
                    high=bar['high'],
                    low=bar['low'],
                    ema10=bar['ema10'],
                    prev_close=prev_bar['close'],
                    prev_high=prev_bar['high'],
                    prev_low=prev_bar['low'],
                )
                for ev in exit_events:
                    if results[ev.strategy] is None:
                        results[ev.strategy] = {
                            'pnl': ev.pnl_pct, 'bars': ev.bars_held, 'reason': ev.exit_reason,
                        }

            if tracker.all_done():
                break

        # 强制平仓
        last_bar = df.iloc[-1]
        forced = tracker.force_close(last_bar['close'])
        for ev in forced:
            if results[ev.strategy] is None:
                results[ev.strategy] = {
                    'pnl': ev.pnl_pct, 'bars': ev.bars_held, 'reason': ev.exit_reason,
                }

        trade = {
            'type': sig['type'],
            'direction': sig['direction'],
            'er_10': sig['er_10'],
            'er_20': sig['er_20'],
            'er_40': sig['er_40'],
            'mfe': round(mfe, 4),
            'mae': round(mae, 4),
        }
        for s in ['S1', 'S2', 'S3']:
            r = results[s]
            trade[f'{s}_pnl'] = round(r['pnl'], 4)
            trade[f'{s}_bars'] = r['bars']

        trades.append(trade)

    return trades


def calc_group_stats(trades_df, exit_name):
    """计算单个出场方式的统计"""
    pnl_col = f'{exit_name}_pnl'
    bars_col = f'{exit_name}_bars'
    if len(trades_df) == 0:
        return {'N': 0, 'wr': 0, 'EV': 0, 'sum': 0, 'pr': 0, 'bars': 0}

    pnls = trades_df[pnl_col]
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    wr = len(wins) / len(pnls)  # 0~1
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.001
    pr = avg_win / avg_loss if avg_loss > 0 else 0
    # EV = 胜率 × 盈亏比 - 败率
    EV = wr * pr - (1 - wr)

    return {
        'N': len(pnls),
        'wr': round(wr * 100, 1),
        'EV': round(EV, 2),
        'sum': round(pnls.sum(), 2),
        'pr': round(pr, 2),
        'bars': round(trades_df[bars_col].mean(), 1),
    }


# ==================== 主流程 ====================

print("=" * 100)
print("ER过滤 + 三种出场 全品种回测 | 32品种 | 120天 | A+B类信号")
print("=" * 100)

all_trades = []

for sym_key, cfg in sorted(SYMBOL_CONFIGS.items(), key=lambda x: x[1]['name']):
    cache_key = sym_key.replace(".", "_")
    df = load_and_prepare(cache_key)
    if df is None:
        continue

    tick_size = cfg['tick_size']
    n = len(df)
    start_idx = max(0, n - LAST_DAYS * BARS_PER_DAY)

    signals = detect_signals_ab(df, start_idx)
    if not signals:
        print(f"  {cfg['name']}: no signals")
        continue

    trades = simulate_exits(df, signals, tick_size)

    n_a = sum(1 for s in signals if s['type'] == 'A')
    n_b = sum(1 for s in signals if s['type'] == 'B')
    print(f"  {cfg['name']:>6}: A={n_a:>4} B={n_b:>4} total={len(trades):>4}")

    for t in trades:
        t['symbol'] = cfg['name']
        t['group'] = 'H' if sym_key in HIGH_VOL_KEYS else 'N'
    all_trades.extend(trades)

df_all = pd.DataFrame(all_trades)
n_total = len(df_all)
n_a_total = len(df_all[df_all['type'] == 'A'])
n_b_total = len(df_all[df_all['type'] == 'B'])
print(f"\n  TOTAL: A={n_a_total} B={n_b_total} all={n_total}")


# ==================== 输出统计表 ====================

def print_table(df_sub, title):
    """打印一张 ER分档 x 出场方式 的表"""
    print(f"\n{'='*100}")
    print(f"  {title}  (N={len(df_sub)})")
    print(f"{'='*100}")

    for er_p in ER_PERIODS:
        er_col = f'er_{er_p}'
        print(f"\n  --- ER({er_p}) ---")
        print(f"  {'ER_bin':>10} | {'N':>5} | {'S1_EV':>6} {'S1_wr':>6} {'S1_pr':>6} {'S1_bar':>6} |"
              f" {'S2_EV':>6} {'S2_wr':>6} {'S2_pr':>6} {'S2_bar':>6} |"
              f" {'S3_EV':>6} {'S3_wr':>6} {'S3_pr':>6} {'S3_bar':>6}")
        print(f"  {'-'*100}")

        for bin_name, lo, hi in ER_BINS:
            if bin_name == 'all':
                subset = df_sub
            else:
                subset = df_sub[(df_sub[er_col] >= lo) & (df_sub[er_col] < hi)]

            parts = [f"  {bin_name:>10} |"]
            parts.append(f" {len(subset):>5} |")

            for s in ['S1', 'S2', 'S3']:
                st = calc_group_stats(subset, s)
                parts.append(f" {st['EV']:>+6.2f} {st['wr']:>5.1f}% {st['pr']:>5.2f} {st['bars']:>5.1f} |")

            print(''.join(parts))


# 全部信号
print_table(df_all, "ALL (A+B)")

# A类
df_a = df_all[df_all['type'] == 'A']
print_table(df_a, "A only")

# B类
df_b = df_all[df_all['type'] == 'B']
print_table(df_b, "B only")

# 高波动 vs 常规
df_h = df_all[df_all['group'] == 'H']
df_n = df_all[df_all['group'] == 'N']
print_table(df_h, "HIGH VOL group")
print_table(df_n, "NORMAL group")
