# -*- coding: utf-8 -*-
"""
入场偏离度过滤测试
如果信号K线收盘价距EMA10超过0.5%，放弃本次机会
对比：保留 vs 放弃 的收益差异
A+B类 | S1/S2/S4 | 全品种120天
"""

import os, json
import numpy as np
import pandas as pd
from signal_core import SYMBOL_CONFIGS

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

LAST_DAYS = 120
BARS_PER_DAY = 57
STOP_TICKS = 5
MIN_PB_BARS_C = 4

DEV_THRESHOLD = 0.5  # 偏离度阈值 %

ER_FINE_BINS = [
    ('all',       0,    999),
    ('<0.5',      0,    0.5),
    ('0.5~0.7',   0.5,  0.7),
]


def load_and_prepare(cache_key):
    path = os.path.join(CACHE_DIR, f"{cache_key}_10min_170d.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema120'] = df['close'].ewm(span=120, adjust=False).mean()
    net = (df['close'] - df['close'].shift(20)).abs()
    bar_sum = df['close'].diff().abs().rolling(20).sum()
    df['er_20'] = net / bar_sum.replace(0, np.nan)
    return df


def detect_signals_with_deviation(df, start_idx):
    """A+B类信号检测，额外记录收盘价对EMA10的偏离度"""
    signals = []
    n = len(df)
    trend_dir = 0
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

        curr_trend = 1 if ema20 > ema120 else (-1 if ema20 < ema120 else 0)
        if curr_trend != trend_dir and curr_trend != 0:
            trend_dir = curr_trend
            below_start, pb_low, pb_high = -1, None, None

        if trend_dir == 0:
            prev_close, prev_ema10 = close, ema10
            continue

        er_20 = row.get('er_20', np.nan)
        # 偏离度 = |close - ema10| / ema10 * 100
        deviation = abs(close - ema10) / ema10 * 100

        if trend_dir == 1:
            if i >= start_idx and low <= ema10 and close > ema10 and prev_close > prev_ema10:
                if below_start == -1:
                    signals.append({
                        'idx': i, 'type': 'A', 'direction': 'long',
                        'entry_price': close, 'pullback_extreme': low,
                        'er_20': er_20, 'deviation': round(deviation, 4),
                    })
            if below_start == -1:
                if close < ema10 and prev_close >= prev_ema10:
                    below_start, pb_low = i, low
            else:
                pb_low = min(pb_low, low)
                if close > ema10:
                    pb_bars = i - below_start
                    if i >= start_idx and 1 <= pb_bars < MIN_PB_BARS_C:
                        signals.append({
                            'idx': i, 'type': 'B', 'direction': 'long',
                            'entry_price': close, 'pullback_extreme': pb_low,
                            'er_20': er_20, 'deviation': round(deviation, 4),
                        })
                    below_start, pb_low = -1, None

        elif trend_dir == -1:
            if i >= start_idx and high >= ema10 and close < ema10 and prev_close < prev_ema10:
                if below_start == -1:
                    signals.append({
                        'idx': i, 'type': 'A', 'direction': 'short',
                        'entry_price': close, 'pullback_extreme': high,
                        'er_20': er_20, 'deviation': round(deviation, 4),
                    })
            if below_start == -1:
                if close > ema10 and prev_close <= prev_ema10:
                    below_start, pb_high = i, high
            else:
                pb_high = max(pb_high, high)
                if close < ema10:
                    pb_bars = i - below_start
                    if i >= start_idx and 1 <= pb_bars < MIN_PB_BARS_C:
                        signals.append({
                            'idx': i, 'type': 'B', 'direction': 'short',
                            'entry_price': close, 'pullback_extreme': pb_high,
                            'er_20': er_20, 'deviation': round(deviation, 4),
                        })
                    below_start, pb_high = -1, None

        prev_close, prev_ema10 = close, ema10
    return signals


def simulate_exits(df, signals, tick_size):
    trades = []
    n = len(df)
    tick = tick_size * STOP_TICKS

    for sig in signals:
        entry_idx = sig['idx']
        entry_price = sig['entry_price']
        is_long = sig['direction'] == 'long'
        pb_ext = sig['pullback_extreme']

        init_stop = (pb_ext - tick) if is_long else (pb_ext + tick)
        s1_stop = s2_stop = init_stop
        s2_state = 'normal'
        s2_tracking = None
        s1_done = s2_done = False
        s1_pnl = s2_pnl = 0.0

        for j in range(entry_idx + 1, n):
            bar = df.iloc[j]
            prev_bar = df.iloc[j - 1]
            if pd.isna(bar['ema10']):
                continue
            close, high, low = bar['close'], bar['high'], bar['low']
            ma_val = bar['ema10']

            # S1
            if not s1_done:
                if (is_long and low <= s1_stop) or (not is_long and high >= s1_stop):
                    s1_done = True
                    ep = s1_stop
                    s1_pnl = (ep - entry_price) / entry_price * 100 if is_long else (entry_price - ep) / entry_price * 100
                else:
                    if is_long and close > prev_bar['close']:
                        s1_stop = max(s1_stop, low - tick)
                    elif not is_long and close < prev_bar['close']:
                        s1_stop = min(s1_stop, high + tick)

            # S2
            if not s2_done:
                if (is_long and low <= s2_stop) or (not is_long and high >= s2_stop):
                    s2_done = True
                    ep = s2_stop
                    s2_pnl = (ep - entry_price) / entry_price * 100 if is_long else (entry_price - ep) / entry_price * 100
                else:
                    if is_long:
                        if s2_state == 'normal' and close < ma_val:
                            s2_state, s2_tracking = 'pullback', low
                        elif s2_state == 'pullback':
                            s2_tracking = min(s2_tracking, low)
                            if close > ma_val:
                                s2_stop = max(s2_stop, s2_tracking - tick)
                                s2_state, s2_tracking = 'normal', None
                    else:
                        if s2_state == 'normal' and close > ma_val:
                            s2_state, s2_tracking = 'pullback', high
                        elif s2_state == 'pullback':
                            s2_tracking = max(s2_tracking, high)
                            if close < ma_val:
                                s2_stop = min(s2_stop, s2_tracking + tick)
                                s2_state, s2_tracking = 'normal', None

            if s1_done and s2_done:
                break

        last_close = df.iloc[-1]['close']
        if not s1_done:
            s1_pnl = (last_close - entry_price) / entry_price * 100 if is_long else (entry_price - last_close) / entry_price * 100
        if not s2_done:
            s2_pnl = (last_close - entry_price) / entry_price * 100 if is_long else (entry_price - last_close) / entry_price * 100

        s4_pnl = (s1_pnl + s2_pnl) / 2

        trades.append({
            'type': sig['type'], 'direction': sig['direction'],
            'er_20': sig['er_20'], 'deviation': sig['deviation'],
            's1_pnl': round(s1_pnl, 4),
            's2_pnl': round(s2_pnl, 4),
            's4_pnl': round(s4_pnl, 4),
        })
    return trades


def calc_ev(pnls):
    if len(pnls) == 0:
        return {'N': 0, 'EV': 0, 'wr': 0, 'pr': 0, 'avg_w': 0, 'avg_l': 0, 'sum': 0}
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0.001
    pr = avg_win / avg_loss if avg_loss > 0 else 0
    EV = wr * pr - (1 - wr)
    return {'N': len(pnls), 'EV': round(EV, 2), 'wr': round(wr * 100, 1),
            'pr': round(pr, 2), 'avg_w': round(avg_win, 4), 'avg_l': round(avg_loss, 4),
            'sum': round(sum(pnls), 2)}


# ==================== main ====================
all_trades = []

for sym_key, cfg in sorted(SYMBOL_CONFIGS.items(), key=lambda x: x[1]['name']):
    cache_key = sym_key.replace(".", "_")
    df = load_and_prepare(cache_key)
    if df is None:
        continue
    tick_size = cfg['tick_size']
    n = len(df)
    start_idx = max(0, n - LAST_DAYS * BARS_PER_DAY)
    signals = detect_signals_with_deviation(df, start_idx)
    if not signals:
        continue
    trades = simulate_exits(df, signals, tick_size)
    for t in trades:
        t['symbol'] = cfg['name']
    all_trades.extend(trades)

df_all = pd.DataFrame(all_trades)

# 只看A+B类
df_ab = df_all[df_all['type'].isin(['A', 'B'])]

print(f"A+B total: {len(df_ab)}")

# 偏离度分布
dev = df_ab['deviation']
print(f"\n=== 偏离度分布 ===")
for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]:
    n_above = len(df_ab[dev > th])
    pct = n_above / len(df_ab) * 100
    print(f"  > {th}%: {n_above} ({pct:.1f}%)")

# 核心对比：保留(<=0.5%) vs 放弃(>0.5%)
print(f"\n{'='*100}")
print(f"核心对比：偏离度阈值 {DEV_THRESHOLD}%")
print(f"{'='*100}")

df_keep = df_ab[df_ab['deviation'] <= DEV_THRESHOLD]
df_skip = df_ab[df_ab['deviation'] > DEV_THRESHOLD]

# 写结果到文件
with open(os.path.join(OUTPUT_DIR, '_deviation_result.txt'), 'w', encoding='utf-8') as out:
    out.write(f"入场偏离度过滤测试 | A+B类 | S1/S2/S4 | 32品种120天\n")
    out.write(f"阈值: 收盘价距EMA10 > {DEV_THRESHOLD}% 则放弃\n")
    out.write(f"{'='*100}\n\n")

    # 偏离度分布
    out.write("=== 偏离度分布 ===\n")
    for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]:
        n_above = len(df_ab[dev > th])
        pct = n_above / len(df_ab) * 100
        out.write(f"  > {th}%: {n_above:>6} ({pct:.1f}%)\n")

    out.write(f"\n{'='*100}\n")
    out.write(f"核心对比：偏离度阈值 {DEV_THRESHOLD}%\n")
    out.write(f"{'='*100}\n\n")

    # 全量对比
    for label, subset_label, df_sub in [
        ('全量', 'A+B全部', df_ab),
        ('保留', f'偏离 <= {DEV_THRESHOLD}%', df_keep),
        ('放弃', f'偏离 > {DEV_THRESHOLD}%', df_skip),
    ]:
        out.write(f"\n--- {label} ({subset_label}) ---\n")
        for sx_label, sx in [('S1', 's1_pnl'), ('S2', 's2_pnl'), ('S4', 's4_pnl')]:
            ev = calc_ev(df_sub[sx].tolist())
            out.write(f"  {sx_label}: N={ev['N']:>6} EV={ev['EV']:>+.2f} wr={ev['wr']:>5.1f}% pr={ev['pr']:.2f} sum={ev['sum']:>+.1f}%\n")

    # 按ER分层对比
    out.write(f"\n{'='*100}\n")
    out.write(f"按ER(20)分层对比 (S2出场)\n")
    out.write(f"{'='*100}\n\n")

    for er_label, er_lo, er_hi in ER_FINE_BINS:
        out.write(f"\n--- ER(20) {er_label} ---\n")
        for group_label, df_sub in [('全量', df_ab), ('保留', df_keep), ('放弃', df_skip)]:
            if er_label == 'all':
                sub = df_sub
            else:
                sub = df_sub[(df_sub['er_20'] >= er_lo) & (df_sub['er_20'] < er_hi)]
            ev = calc_ev(sub['s2_pnl'].tolist())
            ev4 = calc_ev(sub['s4_pnl'].tolist())
            out.write(f"  {group_label:>4}: N={ev['N']:>6} S2 EV={ev['EV']:>+.2f} sum={ev['sum']:>+8.1f}% | S4 EV={ev4['EV']:>+.2f} sum={ev4['sum']:>+8.1f}%\n")

    # 按信号类型分
    out.write(f"\n{'='*100}\n")
    out.write(f"按信号类型分 (S2出场, 全ER)\n")
    out.write(f"{'='*100}\n\n")

    for sig_type in ['A', 'B']:
        out.write(f"\n--- {sig_type}类 ---\n")
        for group_label, df_sub in [('全量', df_ab), ('保留', df_keep), ('放弃', df_skip)]:
            sub = df_sub[df_sub['type'] == sig_type]
            ev = calc_ev(sub['s2_pnl'].tolist())
            out.write(f"  {group_label:>4}: N={ev['N']:>6} EV={ev['EV']:>+.2f} wr={ev['wr']:>5.1f}% pr={ev['pr']:.2f} sum={ev['sum']:>+.1f}%\n")

    # ER>=0.5 + 偏离度分层 细分
    out.write(f"\n{'='*100}\n")
    out.write(f"ER(20) 0.5~0.7 甜点区 偏离度细分\n")
    out.write(f"{'='*100}\n\n")

    df_sweet = df_ab[(df_ab['er_20'] >= 0.5) & (df_ab['er_20'] < 0.7)]
    dev_bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                (0.5, 0.7), (0.7, 1.0), (1.0, 2.0), (2.0, 99)]
    out.write(f"{'偏离度':>12} {'N':>5} {'S2 EV':>7} {'S2 wr':>6} {'S2 pr':>6} {'S2 sum':>9} {'S4 EV':>7} {'S4 sum':>9}\n")
    out.write("-" * 80 + "\n")
    for lo, hi in dev_bins:
        sub = df_sweet[(df_sweet['deviation'] > lo) & (df_sweet['deviation'] <= hi)]
        if len(sub) == 0:
            continue
        label = f"{lo}~{hi}%" if hi < 90 else f">{lo}%"
        ev2 = calc_ev(sub['s2_pnl'].tolist())
        ev4 = calc_ev(sub['s4_pnl'].tolist())
        out.write(f"{label:>12} {ev2['N']:>5} {ev2['EV']:>+7.2f} {ev2['wr']:>5.1f}% {ev2['pr']:>6.2f} {ev2['sum']:>+9.1f} {ev4['EV']:>+7.2f} {ev4['sum']:>+9.1f}\n")

    # 汇总：过滤后整体效果
    out.write(f"\n{'='*100}\n")
    out.write(f"最终汇总：加入偏离度过滤后的整体效果\n")
    out.write(f"{'='*100}\n\n")

    # ER 0.5~0.7 + 偏离<=0.5%
    df_final = df_ab[(df_ab['er_20'] >= 0.5) & (df_ab['er_20'] < 0.7) & (df_ab['deviation'] <= DEV_THRESHOLD)]
    df_before = df_ab[(df_ab['er_20'] >= 0.5) & (df_ab['er_20'] < 0.7)]

    out.write(f"ER 0.5~0.7 不过滤偏离度:\n")
    for sx_l, sx in [('S1','s1_pnl'),('S2','s2_pnl'),('S4','s4_pnl')]:
        ev = calc_ev(df_before[sx].tolist())
        out.write(f"  {sx_l}: N={ev['N']:>5} EV={ev['EV']:>+.2f} wr={ev['wr']:.1f}% sum={ev['sum']:>+.1f}%\n")

    out.write(f"\nER 0.5~0.7 + 偏离<=0.5%:\n")
    for sx_l, sx in [('S1','s1_pnl'),('S2','s2_pnl'),('S4','s4_pnl')]:
        ev = calc_ev(df_final[sx].tolist())
        out.write(f"  {sx_l}: N={ev['N']:>5} EV={ev['EV']:>+.2f} wr={ev['wr']:.1f}% sum={ev['sum']:>+.1f}%\n")

    skipped = df_ab[(df_ab['er_20'] >= 0.5) & (df_ab['er_20'] < 0.7) & (df_ab['deviation'] > DEV_THRESHOLD)]
    out.write(f"\n被过滤掉的交易 (ER 0.5~0.7, 偏离>{DEV_THRESHOLD}%):\n")
    for sx_l, sx in [('S1','s1_pnl'),('S2','s2_pnl'),('S4','s4_pnl')]:
        ev = calc_ev(skipped[sx].tolist())
        out.write(f"  {sx_l}: N={ev['N']:>5} EV={ev['EV']:>+.2f} wr={ev['wr']:.1f}% sum={ev['sum']:>+.1f}%\n")

    # 多个阈值对比
    out.write(f"\n{'='*100}\n")
    out.write(f"不同阈值对比 (ER 0.5~0.7, S2出场)\n")
    out.write(f"{'='*100}\n\n")
    out.write(f"{'阈值':>8} {'保留N':>6} {'保留EV':>7} {'保留sum':>9} {'放弃N':>6} {'放弃EV':>7} {'放弃sum':>9}\n")
    out.write("-" * 70 + "\n")

    df_er = df_ab[(df_ab['er_20'] >= 0.5) & (df_ab['er_20'] < 0.7)]
    for th in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 999]:
        keep = df_er[df_er['deviation'] <= th]
        skip = df_er[df_er['deviation'] > th]
        ev_k = calc_ev(keep['s2_pnl'].tolist())
        ev_s = calc_ev(skip['s2_pnl'].tolist())
        th_label = f"<={th}%" if th < 900 else "不过滤"
        out.write(f"{th_label:>8} {ev_k['N']:>6} {ev_k['EV']:>+7.2f} {ev_k['sum']:>+9.1f} {ev_s['N']:>6} {ev_s['EV']:>+7.2f} {ev_s['sum']:>+9.1f}\n")

print("Done. Results in output/_deviation_result.txt")
