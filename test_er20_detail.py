# -*- coding: utf-8 -*-
"""
ER(20) 细分档位 A/B/C分开 S1/S2/S3 全维度
A类：影线碰EMA10弹回（单根）
B类：跌破EMA10后1~3根收回
C类：跌破EMA10后>=4根收回
32品种120天，5跳止损
"""

import os
import numpy as np
import pandas as pd
from signal_core import SYMBOL_CONFIGS

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")

HIGH_VOL_KEYS = {"GFEX.lc", "DCE.jm", "SHFE.ag", "CZCE.FG", "CZCE.SA",
                 "INE.sc", "CZCE.MA", "CZCE.TA", "DCE.eb", "DCE.lh"}

MIN_PB_BARS_C = 4  # C类>=4根
LAST_DAYS = 120
BARS_PER_DAY = 57
STOP_TICKS = 5

ER_FINE_BINS = [
    ('all',       0,    999),
    ('0~0.1',     0,    0.1),
    ('0.1~0.2',   0.1,  0.2),
    ('0.2~0.3',   0.2,  0.3),
    ('0.3~0.4',   0.3,  0.4),
    ('0.4~0.5',   0.4,  0.5),
    ('0.5~0.6',   0.5,  0.6),
    ('0.6~0.7',   0.6,  0.7),
    ('0.7+',      0.7,  999),
]

ER_COARSE_BINS = [
    ('all',       0,    999),
    ('<0.25',     0,    0.25),
    ('0.25~0.5',  0.25, 0.5),
    ('>=0.5',     0.5,  999),
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


def detect_signals_abc(df, start_idx):
    """A/B/C三类信号检测
    A类：影线碰EMA10弹回（单根，close不跌破）
    B类：实体跌破EMA10后1~3根收回
    C类：实体跌破EMA10后>=4根收回
    """
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

        if trend_dir == 1:
            # A类：影线碰EMA10弹回
            if i >= start_idx and low <= ema10 and close > ema10 and prev_close > prev_ema10:
                if below_start == -1:
                    signals.append({
                        'idx': i, 'type': 'A', 'direction': 'long',
                        'entry_price': close, 'pullback_extreme': low,
                        'er_20': er_20,
                    })

            # B类/C类：实体跌破EMA10后收回
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
                            'er_20': er_20,
                        })
                    below_start, pb_low = -1, None

        elif trend_dir == -1:
            # A类
            if i >= start_idx and high >= ema10 and close < ema10 and prev_close < prev_ema10:
                if below_start == -1:
                    signals.append({
                        'idx': i, 'type': 'A', 'direction': 'short',
                        'entry_price': close, 'pullback_extreme': high,
                        'er_20': er_20,
                    })

            # B类/C类
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
                            'er_20': er_20,
                        })
                    below_start, pb_high = -1, None

        prev_close, prev_ema10 = close, ema10
    return signals


def simulate_all_exits(df, signals, tick_size):
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
        s2_rounds = 0
        s1_done = s2_done = s3_done = False
        s1_pnl = s2_pnl = s3_pnl = 0.0
        s1_bars = s2_bars = s3_bars = 0
        s1_reason = s2_reason = s3_reason = 'stop'
        mfe = 0.0
        mae = 0.0

        for j in range(entry_idx + 1, n):
            bar = df.iloc[j]
            prev_bar = df.iloc[j - 1]
            if pd.isna(bar['ema10']):
                continue

            close, high, low = bar['close'], bar['high'], bar['low']
            ma_val = bar['ema10']
            bars_j = j - entry_idx

            if is_long:
                fav = (high - entry_price) / entry_price * 100
                adv = (entry_price - low) / entry_price * 100
            else:
                fav = (entry_price - low) / entry_price * 100
                adv = (high - entry_price) / entry_price * 100
            mfe = max(mfe, fav)
            mae = max(mae, adv)

            # S1
            if not s1_done:
                if (is_long and low <= s1_stop) or (not is_long and high >= s1_stop):
                    s1_done, s1_bars = True, bars_j
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
                    s2_done, s2_bars = True, bars_j
                    ep = s2_stop
                    s2_pnl = (ep - entry_price) / entry_price * 100 if is_long else (entry_price - ep) / entry_price * 100
                else:
                    if is_long:
                        if s2_state == 'normal' and close < ma_val:
                            s2_state, s2_tracking = 'pullback', low
                        elif s2_state == 'pullback':
                            s2_tracking = min(s2_tracking, low)
                            if close > ma_val:
                                s2_rounds += 1
                                s2_stop = max(s2_stop, s2_tracking - tick)
                                s2_state, s2_tracking = 'normal', None
                    else:
                        if s2_state == 'normal' and close > ma_val:
                            s2_state, s2_tracking = 'pullback', high
                        elif s2_state == 'pullback':
                            s2_tracking = max(s2_tracking, high)
                            if close < ma_val:
                                s2_rounds += 1
                                s2_stop = min(s2_stop, s2_tracking + tick)
                                s2_state, s2_tracking = 'normal', None

            # S3
            if not s3_done:
                if (is_long and low <= s3_stop) or (not is_long and high >= s3_stop):
                    s3_done, s3_bars = True, bars_j
                    ep = s3_stop
                    s3_pnl = (ep - entry_price) / entry_price * 100 if is_long else (entry_price - ep) / entry_price * 100
                else:
                    if is_long and close > prev_bar['close']:
                        s3_stop = max(s3_stop, prev_bar['low'] - tick)
                    elif not is_long and close < prev_bar['close']:
                        s3_stop = min(s3_stop, prev_bar['high'] + tick)

            if s1_done and s2_done and s3_done:
                break

        last_close = df.iloc[-1]['close']
        if not s1_done:
            s1_pnl = (last_close - entry_price) / entry_price * 100 if is_long else (entry_price - last_close) / entry_price * 100
            s1_bars = n - entry_idx - 1
            s1_reason = 'backtest_end'
        if not s2_done:
            s2_pnl = (last_close - entry_price) / entry_price * 100 if is_long else (entry_price - last_close) / entry_price * 100
            s2_bars = n - entry_idx - 1
            s2_reason = 'backtest_end'
        if not s3_done:
            s3_pnl = (last_close - entry_price) / entry_price * 100 if is_long else (entry_price - last_close) / entry_price * 100
            s3_bars = n - entry_idx - 1
            s3_reason = 'backtest_end'

        # S4 = 半仓S1 + 半仓S2
        s4_pnl = (s1_pnl + s2_pnl) / 2
        s4_bars = max(s1_bars, s2_bars)  # 以最晚出场的为准

        trades.append({
            'type': sig['type'], 'direction': sig['direction'],
            'er_20': sig['er_20'],
            's1_pnl': round(s1_pnl, 4), 's1_bars': s1_bars, 's1_reason': s1_reason,
            's2_pnl': round(s2_pnl, 4), 's2_bars': s2_bars, 's2_rounds': s2_rounds, 's2_reason': s2_reason,
            's3_pnl': round(s3_pnl, 4), 's3_bars': s3_bars, 's3_reason': s3_reason,
            's4_pnl': round(s4_pnl, 4), 's4_bars': s4_bars,
            'mfe': round(mfe, 4), 'mae': round(mae, 4),
        })

    return trades


def calc_ev(pnls):
    if len(pnls) == 0:
        return {'N': 0, 'EV': 0, 'wr': 0, 'pr': 0, 'avg_w': 0, 'avg_l': 0}
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0.001
    pr = avg_win / avg_loss if avg_loss > 0 else 0
    EV = wr * pr - (1 - wr)
    return {'N': len(pnls), 'EV': round(EV, 2), 'wr': round(wr * 100, 1),
            'pr': round(pr, 2), 'avg_w': round(avg_win, 4), 'avg_l': round(avg_loss, 4)}


# ==================== main ====================

import json

all_trades = []

for sym_key, cfg in sorted(SYMBOL_CONFIGS.items(), key=lambda x: x[1]['name']):
    cache_key = sym_key.replace(".", "_")
    df = load_and_prepare(cache_key)
    if df is None:
        continue
    tick_size = cfg['tick_size']
    n = len(df)
    start_idx = max(0, n - LAST_DAYS * BARS_PER_DAY)
    signals = detect_signals_abc(df, start_idx)
    if not signals:
        continue
    trades = simulate_all_exits(df, signals, tick_size)
    grp = 'H' if sym_key in HIGH_VOL_KEYS else 'N'
    for t in trades:
        t['symbol'] = cfg['name']
        t['group'] = grp
    all_trades.extend(trades)

df_all = pd.DataFrame(all_trades)

# 输出汇总
na = len(df_all[df_all['type'] == 'A'])
nb = len(df_all[df_all['type'] == 'B'])
nc = len(df_all[df_all['type'] == 'C'])
print(f"total: A={na} B={nb} C={nc} all={len(df_all)}")

# 生成JSON给HTML用
results = {}

# 先确定top5/top10品种列表（基于全量排名）
sym_stats_tmp = []
for sym in sorted(df_all['symbol'].unique()):
    sym_df = df_all[df_all['symbol'] == sym]
    sym_er = sym_df[(sym_df['er_20'] >= 0.5) & (sym_df['s2_reason'] != 'backtest_end')]
    st_er = calc_ev(sym_er['s2_pnl'].tolist())
    sym_stats_tmp.append({'symbol': sym, 'ev_er': st_er['EV']})
sym_stats_tmp.sort(key=lambda x: x['ev_er'], reverse=True)
TOP5_SYMS = set(s['symbol'] for s in sym_stats_tmp[:5])
TOP10_SYMS = set(s['symbol'] for s in sym_stats_tmp[:10])

print(f"Top5: {sorted(TOP5_SYMS)}")
print(f"Top10: {sorted(TOP10_SYMS)}")

# 1. 各类型细分档 — 按 品种组 x 方向 生成所有组合
SYM_GROUPS = {'all': None, 'top5': TOP5_SYMS, 'top10': TOP10_SYMS}
DIR_GROUPS = {'all': None, 'long': 'long', 'short': 'short'}

for type_name in ['A', 'B', 'C']:
    df_type = df_all[df_all['type'] == type_name]
    for sg_key, sg_syms in SYM_GROUPS.items():
        for dg_key, dg_dir in DIR_GROUPS.items():
            # 过滤
            sub_base = df_type
            if sg_syms is not None:
                sub_base = sub_base[sub_base['symbol'].isin(sg_syms)]
            if dg_dir is not None:
                sub_base = sub_base[sub_base['direction'] == dg_dir]

            key = f'{type_name}_fine_{sg_key}_{dg_key}'
            results[key] = []
            for bin_name, lo, hi in ER_FINE_BINS:
                if bin_name == 'all':
                    sub = sub_base
                else:
                    sub = sub_base[(sub_base['er_20'] >= lo) & (sub_base['er_20'] < hi)]
                row_data = {'er': bin_name}
                for sx in ['s1', 's2', 's3', 's4']:
                    # 过滤掉 backtest_end 的交易
                    if sx == 's4':
                        valid = sub[(sub['s1_reason'] != 'backtest_end') & (sub['s2_reason'] != 'backtest_end')]
                    else:
                        valid = sub[sub[f'{sx}_reason'] != 'backtest_end']
                    pnls = valid[f'{sx}_pnl'].tolist()
                    st = calc_ev(pnls)
                    avg_pnl = round(np.mean(pnls), 4) if pnls else 0
                    sum_pnl = round(sum(pnls), 2) if pnls else 0
                    avg_bars = round(valid[f'{sx}_bars'].mean(), 1) if len(valid) > 0 else 0
                    avg_mfe = round(valid['mfe'].mean(), 4) if len(valid) > 0 else 0
                    avg_mae = round(valid['mae'].mean(), 4) if len(valid) > 0 else 0
                    n_excluded = len(sub) - len(valid)
                    row_data[sx] = {**st, 'avg_pnl': avg_pnl, 'sum_pnl': sum_pnl,
                                   'avg_bars': avg_bars, 'avg_mfe': avg_mfe, 'avg_mae': avg_mae,
                                   'n_excluded': n_excluded}
                results[key].append(row_data)
    # 保留兼容key（all_all = 原来的默认）
    results[f'{type_name}_fine'] = results[f'{type_name}_fine_all_all']

# 2. 粗分档全景 (全部 + A/B/C分开)
for label, df_sub in [('all', df_all), ('A', df_all[df_all['type']=='A']),
                       ('B', df_all[df_all['type']=='B']), ('C', df_all[df_all['type']=='C'])]:
    results[f'coarse_{label}'] = []
    for bin_name, lo, hi in ER_COARSE_BINS:
        if bin_name == 'all':
            sub = df_sub
        else:
            sub = df_sub[(df_sub['er_20'] >= lo) & (df_sub['er_20'] < hi)]
        row_data = {'er': bin_name, 'n_total': len(sub)}
        for sx in ['s1', 's2', 's3', 's4']:
            if sx == 's4':
                valid = sub[(sub['s1_reason'] != 'backtest_end') & (sub['s2_reason'] != 'backtest_end')]
            else:
                valid = sub[sub[f'{sx}_reason'] != 'backtest_end']
            pnls = valid[f'{sx}_pnl'].tolist()
            st = calc_ev(pnls)
            avg_pnl = round(np.mean(pnls), 4) if pnls else 0
            sum_pnl = round(sum(pnls), 2) if pnls else 0
            avg_bars = round(valid[f'{sx}_bars'].mean(), 1) if len(valid) > 0 else 0
            row_data[sx] = {**st, 'avg_pnl': avg_pnl, 'sum_pnl': sum_pnl, 'avg_bars': avg_bars,
                           'n_excluded': len(sub) - len(valid)}
        results[f'coarse_{label}'].append(row_data)

# 3. 逐品种排名（过滤 backtest_end）
sym_stats = []
for sym in sorted(df_all['symbol'].unique()):
    sym_df = df_all[df_all['symbol'] == sym]
    sym_valid = sym_df[sym_df['s2_reason'] != 'backtest_end']
    sym_er = sym_valid[sym_valid['er_20'] >= 0.5]
    grp = sym_df.iloc[0]['group']
    st_all = calc_ev(sym_valid['s2_pnl'].tolist())
    st_er = calc_ev(sym_er['s2_pnl'].tolist())
    sym_stats.append({
        'symbol': sym, 'group': grp,
        'n_all': st_all['N'], 'n_er': st_er['N'],
        'pct': round(st_er['N'] / st_all['N'] * 100, 1) if st_all['N'] > 0 else 0,
        'ev_all': st_all['EV'], 'ev_er': st_er['EV'],
        'wr_all': st_all['wr'], 'wr_er': st_er['wr'],
        'sum_all': round(sym_valid['s2_pnl'].sum(), 2),
        'sum_er': round(sym_er['s2_pnl'].sum(), 2) if len(sym_er) > 0 else 0,
    })

# 按ER>=0.5 S2 EV排序
sym_stats.sort(key=lambda x: x['ev_er'], reverse=True)
results['sym_ranked'] = sym_stats

# top5/top10 聚合
for topn_label, topn in [('top5', 5), ('top10', 10), ('all32', len(sym_stats))]:
    top_syms = [s['symbol'] for s in sym_stats[:topn]]
    df_top = df_all[df_all['symbol'].isin(top_syms)]
    results[f'top_{topn_label}'] = []
    for bin_name, lo, hi in ER_COARSE_BINS:
        if bin_name == 'all':
            sub = df_top
        else:
            sub = df_top[(df_top['er_20'] >= lo) & (df_top['er_20'] < hi)]
        row_data = {'er': bin_name, 'n_total': len(sub)}
        for sx in ['s1', 's2', 's3', 's4']:
            if sx == 's4':
                valid = sub[(sub['s1_reason'] != 'backtest_end') & (sub['s2_reason'] != 'backtest_end')]
            else:
                valid = sub[sub[f'{sx}_reason'] != 'backtest_end']
            pnls = valid[f'{sx}_pnl'].tolist()
            st = calc_ev(pnls)
            sum_pnl = round(sum(pnls), 2) if pnls else 0
            row_data[sx] = {**st, 'sum_pnl': sum_pnl}
        results[f'top_{topn_label}'].append(row_data)

# 4. 多空
for direction in ['long', 'short']:
    df_dir = df_all[df_all['direction'] == direction]
    results[f'dir_{direction}'] = []
    for bin_name, lo, hi in ER_COARSE_BINS:
        if bin_name == 'all':
            sub = df_dir
        else:
            sub = df_dir[(df_dir['er_20'] >= lo) & (df_dir['er_20'] < hi)]
        row_data = {'er': bin_name}
        for sx in ['s1', 's2', 's3', 's4']:
            if sx == 's4':
                valid = sub[(sub['s1_reason'] != 'backtest_end') & (sub['s2_reason'] != 'backtest_end')]
            else:
                valid = sub[sub[f'{sx}_reason'] != 'backtest_end']
            pnls = valid[f'{sx}_pnl'].tolist()
            st = calc_ev(pnls)
            sum_pnl = round(sum(pnls), 2) if pnls else 0
            row_data[sx] = {**st, 'sum_pnl': sum_pnl}
        results[f'dir_{direction}'].append(row_data)

# 保存JSON
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'er20_data.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"JSON saved to {out_path}")
print(f"top5 symbols: {[s['symbol'] for s in sym_stats[:5]]}")
print(f"top10 symbols: {[s['symbol'] for s in sym_stats[:10]]}")
