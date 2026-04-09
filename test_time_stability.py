# -*- coding: utf-8 -*-
"""
时间稳定性检验：120天分4个30天窗口
策略：A+B类信号 | S2出场 | ER(20) 0.5~0.7 | 5跳止损
检验Top3/5/10品种在各窗口的表现是否稳定
"""

import os, json
import numpy as np
import pandas as pd
from signal_core import SYMBOL_CONFIGS

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
LAST_DAYS = 120
BARS_PER_DAY = 57
STOP_TICKS = 5
MIN_PB_BARS_C = 4
N_WINDOWS = 4
WINDOW_DAYS = LAST_DAYS // N_WINDOWS  # 30天

ER_LO, ER_HI = 0.5, 0.7  # 只看甜点区


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


def detect_signals_ab(df, start_idx, end_idx):
    """A+B类信号检测（不含C类），限定在[start_idx, end_idx)范围"""
    signals = []
    n = min(len(df), end_idx)
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
            if i >= start_idx and low <= ema10 and close > ema10 and prev_close > prev_ema10:
                if below_start == -1:
                    signals.append({
                        'idx': i, 'type': 'A', 'direction': 'long',
                        'entry_price': close, 'pullback_extreme': low, 'er_20': er_20,
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
                            'entry_price': close, 'pullback_extreme': pb_low, 'er_20': er_20,
                        })
                    below_start, pb_low = -1, None

        elif trend_dir == -1:
            if i >= start_idx and high >= ema10 and close < ema10 and prev_close < prev_ema10:
                if below_start == -1:
                    signals.append({
                        'idx': i, 'type': 'A', 'direction': 'short',
                        'entry_price': close, 'pullback_extreme': high, 'er_20': er_20,
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
                            'entry_price': close, 'pullback_extreme': pb_high, 'er_20': er_20,
                        })
                    below_start, pb_high = -1, None

        prev_close, prev_ema10 = close, ema10
    return signals


def simulate_s2(df, signals, tick_size):
    """只跑S2出场"""
    trades = []
    n = len(df)
    tick = tick_size * STOP_TICKS

    for sig in signals:
        entry_idx = sig['idx']
        entry_price = sig['entry_price']
        is_long = sig['direction'] == 'long'
        pb_ext = sig['pullback_extreme']
        init_stop = (pb_ext - tick) if is_long else (pb_ext + tick)

        s2_stop = init_stop
        s2_state = 'normal'
        s2_tracking = None
        s2_done = False
        s2_pnl = 0.0

        for j in range(entry_idx + 1, n):
            bar = df.iloc[j]
            if pd.isna(bar['ema10']):
                continue
            close, high, low = bar['close'], bar['high'], bar['low']
            ma_val = bar['ema10']

            if is_long and low <= s2_stop:
                s2_pnl = (s2_stop - entry_price) / entry_price * 100
                s2_done = True
                break
            elif not is_long and high >= s2_stop:
                s2_pnl = (entry_price - s2_stop) / entry_price * 100
                s2_done = True
                break

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

        if not s2_done:
            last_close = df.iloc[-1]['close']
            s2_pnl = (last_close - entry_price) / entry_price * 100 if is_long else (entry_price - last_close) / entry_price * 100

        trades.append({
            'type': sig['type'], 'direction': sig['direction'],
            'er_20': sig['er_20'], 'pnl': round(s2_pnl, 4),
        })
    return trades


def calc_ev(pnls):
    if len(pnls) == 0:
        return {'N': 0, 'EV': 0, 'wr': 0, 'sum': 0}
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0.001
    pr = avg_win / avg_loss if avg_loss > 0 else 0
    EV = wr * pr - (1 - wr)
    return {'N': len(pnls), 'EV': round(EV, 2), 'wr': round(wr * 100, 1),
            'sum': round(sum(pnls), 2)}


# ==================== main ====================

# 逐品种、逐窗口收集数据
# window_data[sym_name][window_idx] = list of pnls (已过滤ER 0.5~0.7, A+B)
window_data = {}
full_data = {}  # 120天汇总

for sym_key, cfg in sorted(SYMBOL_CONFIGS.items(), key=lambda x: x[1]['name']):
    cache_key = sym_key.replace(".", "_")
    df = load_and_prepare(cache_key)
    if df is None:
        continue

    tick_size = cfg['tick_size']
    name = cfg['name']
    n = len(df)
    total_bars = LAST_DAYS * BARS_PER_DAY
    global_start = max(0, n - total_bars)

    # 全120天信号检测（需要从数据起点开始检测趋势状态）
    signals = detect_signals_ab(df, global_start, n)
    trades = simulate_s2(df, signals, tick_size)

    # ER 0.5~0.7 过滤
    filtered = [t for t in trades if not pd.isna(t['er_20']) and ER_LO <= t['er_20'] < ER_HI]

    # 全量汇总
    full_data[name] = [t['pnl'] for t in filtered]

    # 按窗口分组（根据信号的idx判断属于哪个窗口）
    window_data[name] = {w: [] for w in range(N_WINDOWS)}

    # 重新跑一遍带idx的来分窗口
    sig_by_idx = {sig['idx']: sig for sig in signals}
    for trade, sig in zip(trades, signals):
        if pd.isna(trade['er_20']) or not (ER_LO <= trade['er_20'] < ER_HI):
            continue
        # 信号idx在哪个窗口
        idx = sig['idx']
        bar_offset = idx - global_start
        window_idx = min(bar_offset // (WINDOW_DAYS * BARS_PER_DAY), N_WINDOWS - 1)
        window_data[name][window_idx].append(trade['pnl'])

# 按120天EV排名
ranked = []
for name, pnls in full_data.items():
    ev = calc_ev(pnls)
    ranked.append({'name': name, **ev})
ranked.sort(key=lambda x: x['EV'], reverse=True)

# 输出结果
results = {
    'ranked': ranked,
    'window_detail': {},
}

for item in ranked:
    name = item['name']
    wd = {}
    for w in range(N_WINDOWS):
        pnls = window_data[name][w]
        wd[f'W{w+1}'] = calc_ev(pnls)
    results['window_detail'][name] = wd

# 保存JSON
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'time_stability.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# 打印结果
print(f"{'':>4} {'品种':>6}  {'120天':>8} {'N':>4}  {'W1(早)':>8} {'N1':>3}  {'W2':>8} {'N2':>3}  {'W3':>8} {'N3':>3}  {'W4(近)':>8} {'N4':>3}  稳定性")
print("-" * 120)

for i, item in enumerate(ranked[:15]):
    name = item['name']
    wd = results['window_detail'][name]

    evs = [wd[f'W{w+1}']['EV'] for w in range(N_WINDOWS)]
    ns = [wd[f'W{w+1}']['N'] for w in range(N_WINDOWS)]

    # 稳定性指标：正EV窗口数 / 总窗口数
    pos_windows = sum(1 for e in evs if e > 0)
    # 如果某窗口N=0，不计入
    valid_windows = sum(1 for n in ns if n > 0)
    stability = f"{pos_windows}/{valid_windows}" if valid_windows > 0 else "-"

    tag = ""
    if i < 3:
        tag = " <-- Top3"
    elif i < 5:
        tag = " <-- Top5"
    elif i < 10:
        tag = " <-- Top10"

    line = f"{i+1:>3}. {name:>6}  EV={item['EV']:>+5.2f} {item['N']:>4}"
    for w in range(N_WINDOWS):
        e, n = evs[w], ns[w]
        line += f"  {e:>+5.2f} {n:>3}"
    line += f"  {stability}{tag}"
    print(line)

# Top组合汇总
print()
print("=== 组合汇总 ===")
for topn_label, topn in [('Top3', 3), ('Top5', 5), ('Top10', 10)]:
    top_names = [r['name'] for r in ranked[:topn]]
    print(f"\n{topn_label}: {', '.join(top_names)}")
    for w in range(N_WINDOWS):
        pool = []
        for name in top_names:
            pool.extend(window_data[name][w])
        ev = calc_ev(pool)
        print(f"  W{w+1}: N={ev['N']:>3} EV={ev['EV']:>+.2f} wr={ev['wr']:.1f}% sum={ev['sum']:>+.2f}%")
    # 120天汇总
    pool_all = []
    for name in top_names:
        pool_all.extend(full_data[name])
    ev_all = calc_ev(pool_all)
    print(f"  120天: N={ev_all['N']:>3} EV={ev_all['EV']:>+.2f} wr={ev_all['wr']:.1f}% sum={ev_all['sum']:>+.2f}%")

print(f"\nJSON saved to {out_path}")
