# -*- coding: utf-8 -*-
"""
入场偏离度研究（ATR标准化版）
=============================
信号K线收盘价离EMA10有多远（用ATR标准化），对后续走势有什么影响？

指标：entry_deviation_atr = |entry_price - EMA10| / ATR(14)
品种：32品种全量 | 高波动10 | 常规22 | 动态Top5/Top10
信号：A + B + C 三类
出场：S1/S2/S3/S4
"""

import os, sys
import numpy as np
import pandas as pd
from signal_core import SYMBOL_CONFIGS

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

LAST_DAYS = 120
BARS_PER_DAY = 57
STOP_TICKS = 5
MIN_PB_BARS_C = 4  # >=4根归C类，1~3根归B类
ATR_PERIOD = 14

# 偏离度分档
DEV_BINS = [
    ('0~0.5',   0,   0.5),
    ('0.5~1.0', 0.5, 1.0),
    ('1.0~1.5', 1.0, 1.5),
    ('1.5~2.0', 1.5, 2.0),
    ('2.0+',    2.0, 999),
]

# 高波动品种（固定）
HIGH_VOL = {
    "GFEX.lc", "DCE.jm", "SHFE.ag", "CZCE.FG", "CZCE.SA",
    "INE.sc", "CZCE.MA", "CZCE.TA", "DCE.eb", "DCE.lh",
}


def load_and_prepare(cache_key):
    path = os.path.join(CACHE_DIR, f"{cache_key}_10min_170d.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema120'] = df['close'].ewm(span=120, adjust=False).mean()
    # ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs(),
        )
    )
    df['atr'] = df['tr'].rolling(window=ATR_PERIOD).mean()
    # ER(20)
    net = (df['close'] - df['close'].shift(20)).abs()
    bar_sum = df['close'].diff().abs().rolling(20).sum()
    df['er_20'] = net / bar_sum.replace(0, np.nan)
    return df


def detect_signals(df, start_idx):
    """A+B+C三类信号，记录ATR标准化偏离度"""
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
        atr = row.get('atr', np.nan)

        if prev_close is None or pd.isna(ema120) or pd.isna(ema10) or pd.isna(atr) or atr <= 0:
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
        dev_atr = abs(close - ema10) / atr

        if trend_dir == 1:
            # A类
            if i >= start_idx and low <= ema10 and close > ema10 and prev_close > prev_ema10:
                if below_start == -1:
                    signals.append({
                        'idx': i, 'type': 'A', 'direction': 'long',
                        'entry_price': close, 'pullback_extreme': low,
                        'er_20': er_20, 'dev_atr': round(dev_atr, 4),
                    })
            # B/C类
            if below_start == -1:
                if close < ema10 and prev_close >= prev_ema10:
                    below_start, pb_low = i, low
            else:
                pb_low = min(pb_low, low)
                if close > ema10:
                    pb_bars = i - below_start
                    if i >= start_idx and pb_bars >= 1:
                        sig_type = 'B' if pb_bars < MIN_PB_BARS_C else 'C'
                        signals.append({
                            'idx': i, 'type': sig_type, 'direction': 'long',
                            'entry_price': close, 'pullback_extreme': pb_low,
                            'er_20': er_20, 'dev_atr': round(dev_atr, 4),
                        })
                    below_start, pb_low = -1, None

        elif trend_dir == -1:
            # A类
            if i >= start_idx and high >= ema10 and close < ema10 and prev_close < prev_ema10:
                if below_start == -1:
                    signals.append({
                        'idx': i, 'type': 'A', 'direction': 'short',
                        'entry_price': close, 'pullback_extreme': high,
                        'er_20': er_20, 'dev_atr': round(dev_atr, 4),
                    })
            # B/C类
            if below_start == -1:
                if close > ema10 and prev_close <= prev_ema10:
                    below_start, pb_high = i, high
            else:
                pb_high = max(pb_high, high)
                if close < ema10:
                    pb_bars = i - below_start
                    if i >= start_idx and pb_bars >= 1:
                        sig_type = 'B' if pb_bars < MIN_PB_BARS_C else 'C'
                        signals.append({
                            'idx': i, 'type': sig_type, 'direction': 'short',
                            'entry_price': close, 'pullback_extreme': pb_high,
                            'er_20': er_20, 'dev_atr': round(dev_atr, 4),
                        })
                    below_start, pb_high = -1, None

        prev_close, prev_ema10 = close, ema10
    return signals


def simulate_exits(df, signals, tick_size):
    """S1/S2/S3/S4 四种出场"""
    trades = []
    n = len(df)
    tick = tick_size * STOP_TICKS

    for sig in signals:
        entry_idx = sig['idx']
        entry_price = sig['entry_price']
        is_long = sig['direction'] == 'long'
        pb_ext = sig['pullback_extreme']

        init_stop = (pb_ext - tick) if is_long else (pb_ext + tick)
        s1_stop = s2_stop = s3_stop = init_stop
        s2_state = 'normal'
        s2_tracking = None
        s1_done = s2_done = s3_done = False
        s1_pnl = s2_pnl = s3_pnl = 0.0

        prev_bar = df.iloc[entry_idx]

        for j in range(entry_idx + 1, n):
            bar = df.iloc[j]
            if pd.isna(bar['ema10']):
                prev_bar = bar
                continue
            close, high, low = bar['close'], bar['high'], bar['low']
            ma_val = bar['ema10']
            p_close = prev_bar['close']
            p_low, p_high = prev_bar['low'], prev_bar['high']

            # S1 当根新高追踪
            if not s1_done:
                if (is_long and low <= s1_stop) or (not is_long and high >= s1_stop):
                    s1_done = True
                    ep = s1_stop
                    s1_pnl = (ep - entry_price) / entry_price * 100 if is_long else (entry_price - ep) / entry_price * 100
                else:
                    if is_long and close > p_close:
                        s1_stop = max(s1_stop, low - tick)
                    elif not is_long and close < p_close:
                        s1_stop = min(s1_stop, high + tick)

            # S2 回调追踪
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

            # S3 前根新高追踪
            if not s3_done:
                if (is_long and low <= s3_stop) or (not is_long and high >= s3_stop):
                    s3_done = True
                    ep = s3_stop
                    s3_pnl = (ep - entry_price) / entry_price * 100 if is_long else (entry_price - ep) / entry_price * 100
                else:
                    if is_long and close > p_close:
                        s3_stop = max(s3_stop, p_low - tick)
                    elif not is_long and close < p_close:
                        s3_stop = min(s3_stop, p_high + tick)

            prev_bar = bar
            if s1_done and s2_done and s3_done:
                break

        last_close = df.iloc[-1]['close']
        if not s1_done:
            s1_pnl = (last_close - entry_price) / entry_price * 100 if is_long else (entry_price - last_close) / entry_price * 100
        if not s2_done:
            s2_pnl = (last_close - entry_price) / entry_price * 100 if is_long else (entry_price - last_close) / entry_price * 100
        if not s3_done:
            s3_pnl = (last_close - entry_price) / entry_price * 100 if is_long else (entry_price - last_close) / entry_price * 100

        s4_pnl = (s1_pnl + s2_pnl) / 2

        trades.append({
            'type': sig['type'], 'direction': sig['direction'],
            'er_20': sig['er_20'], 'dev_atr': sig['dev_atr'],
            's1_pnl': round(s1_pnl, 4),
            's2_pnl': round(s2_pnl, 4),
            's3_pnl': round(s3_pnl, 4),
            's4_pnl': round(s4_pnl, 4),
        })
    return trades


def calc_ev(pnls):
    if len(pnls) == 0:
        return {'N': 0, 'EV': 0, 'wr': 0, 'pr': 0, 'sum': 0}
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0.001
    pr = avg_win / avg_loss if avg_loss > 0 else 0
    EV = wr * pr - (1 - wr)
    return {'N': len(pnls), 'EV': round(EV, 2), 'wr': round(wr * 100, 1),
            'pr': round(pr, 2), 'sum': round(sum(pnls), 2)}


def ev_line(label, pnls):
    ev = calc_ev(pnls)
    return f"  {label}: N={ev['N']:>6} EV={ev['EV']:>+.2f} wr={ev['wr']:>5.1f}% pr={ev['pr']:.2f} sum={ev['sum']:>+.1f}%"


def write_group_table(out, df_sub, title):
    """按偏离度分档输出S1/S2/S3/S4的EV表"""
    out.write(f"\n{'='*110}\n")
    out.write(f"{title}\n")
    out.write(f"{'='*110}\n\n")

    header = f"{'偏离度ATR':>10} {'N':>6}  |  {'S1 EV':>6} {'S1 wr':>6} {'S1 pr':>6}  |  {'S2 EV':>6} {'S2 wr':>6} {'S2 pr':>6}  |  {'S3 EV':>6} {'S3 wr':>6} {'S3 pr':>6}  |  {'S4 EV':>6} {'S4 sum':>9}"
    out.write(header + "\n")
    out.write("-" * 110 + "\n")

    for label, lo, hi in DEV_BINS:
        sub = df_sub[(df_sub['dev_atr'] >= lo) & (df_sub['dev_atr'] < hi)]
        if len(sub) == 0:
            continue
        e1 = calc_ev(sub['s1_pnl'].tolist())
        e2 = calc_ev(sub['s2_pnl'].tolist())
        e3 = calc_ev(sub['s3_pnl'].tolist())
        e4 = calc_ev(sub['s4_pnl'].tolist())
        out.write(f"{label:>10} {e1['N']:>6}  |  {e1['EV']:>+6.2f} {e1['wr']:>5.1f}% {e1['pr']:>6.2f}  |  {e2['EV']:>+6.2f} {e2['wr']:>5.1f}% {e2['pr']:>6.2f}  |  {e3['EV']:>+6.2f} {e3['wr']:>5.1f}% {e3['pr']:>6.2f}  |  {e4['EV']:>+6.2f} {e4['sum']:>+9.1f}\n")

    # 全量行
    e1 = calc_ev(df_sub['s1_pnl'].tolist())
    e2 = calc_ev(df_sub['s2_pnl'].tolist())
    e3 = calc_ev(df_sub['s3_pnl'].tolist())
    e4 = calc_ev(df_sub['s4_pnl'].tolist())
    out.write("-" * 110 + "\n")
    out.write(f"{'全量':>10} {e1['N']:>6}  |  {e1['EV']:>+6.2f} {e1['wr']:>5.1f}% {e1['pr']:>6.2f}  |  {e2['EV']:>+6.2f} {e2['wr']:>5.1f}% {e2['pr']:>6.2f}  |  {e3['EV']:>+6.2f} {e3['wr']:>5.1f}% {e3['pr']:>6.2f}  |  {e4['EV']:>+6.2f} {e4['sum']:>+9.1f}\n")


def write_continuity(out, df_sub, title):
    """连续性检验：0.1步长滑动窗口，找EV拐点"""
    out.write(f"\n{'='*110}\n")
    out.write(f"{title} — 连续性检验（0.1步长，>=该阈值的累计EV）\n")
    out.write(f"{'='*110}\n\n")

    out.write(f"{'>=阈值':>8} {'N':>6}  {'S1 EV':>7}  {'S2 EV':>7}  {'S3 EV':>7}  {'S4 EV':>7}\n")
    out.write("-" * 60 + "\n")

    thresholds = [round(x * 0.1, 1) for x in range(0, 31)]  # 0.0 ~ 3.0
    for th in thresholds:
        sub = df_sub[df_sub['dev_atr'] >= th]
        if len(sub) < 10:
            break
        e1 = calc_ev(sub['s1_pnl'].tolist())
        e2 = calc_ev(sub['s2_pnl'].tolist())
        e3 = calc_ev(sub['s3_pnl'].tolist())
        e4 = calc_ev(sub['s4_pnl'].tolist())
        out.write(f"  >={th:.1f}  {e1['N']:>6}  {e1['EV']:>+7.2f}  {e2['EV']:>+7.2f}  {e3['EV']:>+7.2f}  {e4['EV']:>+7.2f}\n")


# ==================== main ====================
print("加载数据并检测信号...")
all_trades = []
symbol_trades = {}  # sym_key -> trades list

for sym_key, cfg in sorted(SYMBOL_CONFIGS.items(), key=lambda x: x[1]['name']):
    cache_key = sym_key.replace(".", "_")
    df = load_and_prepare(cache_key)
    if df is None:
        continue
    tick_size = cfg['tick_size']
    n = len(df)
    start_idx = max(0, n - LAST_DAYS * BARS_PER_DAY)
    signals = detect_signals(df, start_idx)
    if not signals:
        continue
    trades = simulate_exits(df, signals, tick_size)
    for t in trades:
        t['symbol'] = cfg['name']
        t['sym_key'] = sym_key
    all_trades.extend(trades)
    symbol_trades[sym_key] = trades

df_all = pd.DataFrame(all_trades)
print(f"总交易数: {len(df_all)} (A={len(df_all[df_all['type']=='A'])}, B={len(df_all[df_all['type']=='B'])}, C={len(df_all[df_all['type']=='C'])})")

# 按S2 EV排名取动态Top5/Top10
sym_ev = {}
for sym_key, trades in symbol_trades.items():
    pnls = [t['s2_pnl'] for t in trades]
    if len(pnls) >= 10:
        sym_ev[sym_key] = calc_ev(pnls)['EV']
top_syms = sorted(sym_ev, key=sym_ev.get, reverse=True)
top5_set = set(top_syms[:5])
top10_set = set(top_syms[:10])

print(f"\n动态Top5 (S2 EV): {[SYMBOL_CONFIGS[s]['name'] for s in top_syms[:5]]}")
print(f"动态Top10 (S2 EV): {[SYMBOL_CONFIGS[s]['name'] for s in top_syms[:10]]}")

# 品种分组
groups = {
    '全量32品种': df_all,
    '高波动10': df_all[df_all['sym_key'].isin(HIGH_VOL)],
    '常规22': df_all[~df_all['sym_key'].isin(HIGH_VOL)],
    '动态Top5': df_all[df_all['sym_key'].isin(top5_set)],
    '动态Top10': df_all[df_all['sym_key'].isin(top10_set)],
}

# 写报告
out_path = os.path.join(OUTPUT_DIR, '_deviation_atr_result.txt')
with open(out_path, 'w', encoding='utf-8') as out:
    out.write("入场偏离度研究（ATR标准化版）\n")
    out.write(f"指标: |entry_price - EMA10| / ATR(14)\n")
    out.write(f"32品种 x 120天 | A+B+C三类 | S1/S2/S3/S4 | 5跳止损\n")
    out.write(f"总交易: {len(df_all)}\n\n")

    # Top品种列表
    out.write("=== 动态排名（S2 EV）===\n")
    for i, s in enumerate(top_syms[:10]):
        name = SYMBOL_CONFIGS[s]['name']
        ev = sym_ev[s]
        n_trades = len(symbol_trades[s])
        tag = " ★Top5" if s in top5_set else ""
        out.write(f"  {i+1:>2}. {name:<6} EV={ev:>+.2f} N={n_trades}{tag}\n")

    # ===== 第一步：描述性统计 =====
    out.write(f"\n{'='*110}\n")
    out.write(f"第一步：偏离度ATR分布\n")
    out.write(f"{'='*110}\n\n")

    dev = df_all['dev_atr']
    for p in [25, 50, 75, 90, 95, 99]:
        out.write(f"  P{p}: {np.percentile(dev, p):.2f} ATR\n")
    out.write(f"  均值: {dev.mean():.2f} ATR\n")

    # 按品种看分布（确认标准化后是否收敛）
    out.write(f"\n按品种分布（P50 / P90）:\n")
    for sym_key in sorted(symbol_trades.keys(), key=lambda s: SYMBOL_CONFIGS[s]['name']):
        trs = symbol_trades[sym_key]
        devs = [t['dev_atr'] for t in trs]
        if len(devs) < 10:
            continue
        name = SYMBOL_CONFIGS[sym_key]['name']
        p50 = np.percentile(devs, 50)
        p90 = np.percentile(devs, 90)
        out.write(f"  {name:<6} P50={p50:.2f} P90={p90:.2f} N={len(devs)}\n")

    # ===== 第二步：分组对比 =====
    for group_name, group_df in groups.items():
        # 全信号类型
        write_group_table(out, group_df, f"第二步：分组对比 — {group_name} — 全类型(A+B+C)")

        # 分信号类型
        for sig_type in ['A', 'B', 'C']:
            sub = group_df[group_df['type'] == sig_type]
            if len(sub) >= 20:
                write_group_table(out, sub, f"第二步：分组对比 — {group_name} — {sig_type}类")

        # ER(20)>=0.5过滤后
        sub_er = group_df[group_df['er_20'] >= 0.5]
        if len(sub_er) >= 20:
            write_group_table(out, sub_er, f"第二步：分组对比 — {group_name} — ER(20)>=0.5")

        # ER(20)>=0.5 + 分信号类型
        for sig_type in ['A', 'B', 'C']:
            sub_er_sig = sub_er[sub_er['type'] == sig_type]
            if len(sub_er_sig) >= 15:
                write_group_table(out, sub_er_sig, f"第二步：分组对比 — {group_name} — ER(20)>=0.5 — {sig_type}类")

    # ===== 第三步：连续性检验 =====
    write_continuity(out, df_all, "全量32品种 — 全类型")

    # 分信号类型
    for sig_type in ['A', 'B', 'C']:
        sub = df_all[df_all['type'] == sig_type]
        if len(sub) >= 50:
            write_continuity(out, sub, f"全量32品种 — {sig_type}类")

    # ER过滤后
    sub_er = df_all[df_all['er_20'] >= 0.5]
    if len(sub_er) >= 50:
        write_continuity(out, sub_er, "全量32品种 — ER(20)>=0.5")

    # ER过滤后 + 分信号类型
    for sig_type in ['A', 'B', 'C']:
        sub_er_sig = sub_er[sub_er['type'] == sig_type]
        if len(sub_er_sig) >= 30:
            write_continuity(out, sub_er_sig, f"全量32品种 — ER(20)>=0.5 — {sig_type}类")

    # 品种组连续性
    for group_name in ['高波动10', '动态Top10']:
        if len(groups[group_name]) >= 50:
            write_continuity(out, groups[group_name], group_name)

    # ===== 第四步：品种内部验证 =====
    out.write(f"\n{'='*110}\n")
    out.write(f"第四步：品种内部验证（高波动品种单独看）\n")
    out.write(f"{'='*110}\n")

    for sym_key in ["GFEX.lc", "SHFE.ag", "INE.sc", "DCE.jm", "CZCE.SA"]:
        if sym_key not in symbol_trades:
            continue
        name = SYMBOL_CONFIGS[sym_key]['name']
        trs = pd.DataFrame(symbol_trades[sym_key])
        if len(trs) < 30:
            continue

        out.write(f"\n--- {name} (N={len(trs)}) ---\n")
        header = f"{'偏离度ATR':>10} {'N':>5}  {'S1 EV':>7}  {'S2 EV':>7}  {'S3 EV':>7}  {'S4 EV':>7}"
        out.write(header + "\n")
        out.write("-" * 60 + "\n")

        for label, lo, hi in DEV_BINS:
            sub = trs[(trs['dev_atr'] >= lo) & (trs['dev_atr'] < hi)]
            if len(sub) < 3:
                continue
            e1 = calc_ev(sub['s1_pnl'].tolist())
            e2 = calc_ev(sub['s2_pnl'].tolist())
            e3 = calc_ev(sub['s3_pnl'].tolist())
            e4 = calc_ev(sub['s4_pnl'].tolist())
            out.write(f"{label:>10} {e1['N']:>5}  {e1['EV']:>+7.2f}  {e2['EV']:>+7.2f}  {e3['EV']:>+7.2f}  {e4['EV']:>+7.2f}\n")

    # ===== 汇总结论 =====
    out.write(f"\n{'='*110}\n")
    out.write(f"汇总：各偏离度档的S2 sum对比\n")
    out.write(f"{'='*110}\n\n")

    out.write(f"{'偏离度ATR':>10} {'全量32':>10} {'高波动10':>10} {'常规22':>10} {'Top5':>10} {'Top10':>10}\n")
    out.write("-" * 70 + "\n")
    for label, lo, hi in DEV_BINS:
        vals = []
        for gname in ['全量32品种', '高波动10', '常规22', '动态Top5', '动态Top10']:
            sub = groups[gname]
            sub_bin = sub[(sub['dev_atr'] >= lo) & (sub['dev_atr'] < hi)]
            e = calc_ev(sub_bin['s2_pnl'].tolist())
            vals.append(f"{e['sum']:>+10.1f}")
        out.write(f"{label:>10} {''.join(vals)}\n")

print(f"\n报告已写入: {out_path}")
print("Done.")
