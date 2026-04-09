# -*- coding: utf-8 -*-
"""同一120天窗口，旧规则旧出场 vs 新规则新出场 全量对比"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd
from data_loader import load_all, sym_name, tick_size
from signal_core import SignalDetector, ExitTracker, SYMBOL_CONFIGS, DEFAULT_STOP_TICKS

def classify_old(t, er, dev):
    """早上的旧规则：无ER上限，场景3<0.5"""
    if t == 'A' and er >= 0.5 and dev >= 1.0: return 1
    if t == 'C' and dev >= 2.0: return 2
    if t == 'B' and er >= 0.5 and dev < 0.5: return 3
    return None

def classify_new(t, er, dev):
    """现在的新规则：ER<0.7，场景3 0.1~0.3"""
    if er >= 0.7: return None
    if t == 'A' and er >= 0.5 and dev >= 1.0: return 1
    if t == 'C' and dev >= 2.0: return 2
    if t == 'B' and er >= 0.5 and dev >= 0.1 and dev < 0.3: return 3
    return None

all_data = load_all(period_min=10, days=170, last_days=None,
                    emas=(10, 20, 120), er_periods=(20, 40), atr_period=14)
BARS_PER_DAY = 57
LAST_DAYS = 120

old_records = []
new_records = []

for sym_key, df in sorted(all_data.items()):
    n = len(df)
    signal_start = max(130, n - LAST_DAYS * BARS_PER_DAY)
    ts = tick_size(sym_key)
    detector = SignalDetector(signal_types='ABC')

    for i in range(max(2, 130), n):
        row = df.iloc[i]
        if pd.isna(row['ema10']) or pd.isna(row['ema20']) or pd.isna(row['ema120']):
            continue
        if pd.isna(row['atr']) or row['atr'] <= 0:
            continue
        result = detector.process_bar(
            close=row['close'], high=row['high'], low=row['low'],
            ema10=row['ema10'], ema20=row['ema20'], ema120=row['ema120'])
        if result is None or i < signal_start:
            continue

        er20 = row.get('er_20', 0)
        if pd.isna(er20):
            er20 = 0
        atr = row['atr']
        dev = abs(result.entry_price - row['ema10']) / atr if atr > 0 else 0

        sc_old = classify_old(result.signal_type, er20, dev)
        sc_new = classify_new(result.signal_type, er20, dev)
        if sc_old is None and sc_new is None:
            continue

        is_long = result.direction == 'long'
        tick = ts * DEFAULT_STOP_TICKS
        init_stop = result.pullback_extreme - tick if is_long else result.pullback_extreme + tick

        # 手动算旧S1(盘中触损)
        s1_stop = init_stop
        s1_done = False
        s1_pnl = 0

        # 手动算旧S3(盘中触损)
        s3_stop = init_stop
        s3_done = False
        s3_pnl = 0

        # 新出场用ExitTracker
        tracker = ExitTracker(
            direction=result.direction, entry_price=result.entry_price,
            pullback_extreme=result.pullback_extreme, tick_size=ts, stop_ticks=DEFAULT_STOP_TICKS)
        new_ex = {s: None for s in ['S1.1', 'S2', 'S3.1', 'S5.1']}

        for j in range(i + 1, n):
            bar = df.iloc[j]
            prev = df.iloc[j - 1]
            if pd.isna(bar['ema10']):
                continue
            c = bar['close']
            h = bar['high']
            lo = bar['low']
            pc = prev['close']
            ph = prev['high']
            pl = prev['low']

            # 旧S1
            if not s1_done:
                if is_long and lo <= s1_stop:
                    s1_done = True
                    s1_pnl = (s1_stop - result.entry_price) / result.entry_price * 100
                elif not is_long and h >= s1_stop:
                    s1_done = True
                    s1_pnl = (result.entry_price - s1_stop) / result.entry_price * 100
                else:
                    if is_long and c > pc:
                        s1_stop = max(s1_stop, lo - tick)
                    elif not is_long and c < pc:
                        s1_stop = min(s1_stop, h + tick)

            # 旧S3
            if not s3_done:
                if is_long and lo <= s3_stop:
                    s3_done = True
                    s3_pnl = (s3_stop - result.entry_price) / result.entry_price * 100
                elif not is_long and h >= s3_stop:
                    s3_done = True
                    s3_pnl = (result.entry_price - s3_stop) / result.entry_price * 100
                else:
                    if is_long and c > pc:
                        s3_stop = max(s3_stop, pl - tick)
                    elif not is_long and c < pc:
                        s3_stop = min(s3_stop, ph + tick)

            # 新出场
            if not tracker.all_done():
                evts, _ = tracker.process_bar(
                    close=c, high=h, low=lo, ema10=bar['ema10'],
                    prev_close=pc, prev_high=ph, prev_low=pl)
                for ev in evts:
                    if ev.strategy in new_ex and new_ex[ev.strategy] is None:
                        new_ex[ev.strategy] = ev

            if s1_done and s3_done and tracker.all_done():
                break

        # force close
        last_c = df.iloc[-1]['close']
        if not s1_done:
            if is_long:
                s1_pnl = (last_c - result.entry_price) / result.entry_price * 100
            else:
                s1_pnl = (result.entry_price - last_c) / result.entry_price * 100
        if not s3_done:
            if is_long:
                s3_pnl = (last_c - result.entry_price) / result.entry_price * 100
            else:
                s3_pnl = (result.entry_price - last_c) / result.entry_price * 100
        forced = tracker.force_close(last_c)
        for ev in forced:
            if ev.strategy in new_ex and new_ex[ev.strategy] is None:
                new_ex[ev.strategy] = ev

        s2_pnl = new_ex['S2'].pnl_pct
        s4_pnl = (s1_pnl + s2_pnl) / 2

        rec = {
            'S1': s1_pnl, 'S2': s2_pnl, 'S3': s3_pnl, 'S4': s4_pnl,
            'S1.1': new_ex['S1.1'].pnl_pct,
            'S3.1': new_ex['S3.1'].pnl_pct,
            'S5.1': new_ex['S5.1'].pnl_pct,
        }

        if sc_old is not None:
            r = rec.copy()
            r['sc'] = sc_old
            old_records.append(r)
        if sc_new is not None:
            r = rec.copy()
            r['sc'] = sc_new
            new_records.append(r)

old_df = pd.DataFrame(old_records)
new_df = pd.DataFrame(new_records)

print('=' * 110)
print('同一120天窗口 | 旧规则+旧出场 vs 新规则+新出场 | 全量对比')
print('=' * 110)

for s in [1, 2, 3, '合计']:
    if s == '合计':
        sub_old = old_df
        sub_new = new_df
        label = '合计'
    else:
        sub_old = old_df[old_df['sc'] == s]
        sub_new = new_df[new_df['sc'] == s]
        label = f'场景{s}'

    no = len(sub_old)
    nn = len(sub_new)
    print(f'\n  {label}: 旧{no}笔 -> 新{nn}笔 ({nn - no:+d})')
    print(f'  {"":>6} | {"旧EV":>9} {"旧累计":>10} {"旧胜率":>6} | {"新EV":>9} {"新累计":>10} {"新胜率":>6} | {"EV变化":>8}')
    print('  ' + '-' * 85)

    strats = ['S1', 'S2', 'S3', 'S4', 'S1.1', 'S3.1', 'S5.1']
    for c in strats:
        if no > 0:
            o_ev = sub_old[c].mean()
            o_sum = sub_old[c].sum()
            o_wr = (sub_old[c] > 0).mean() * 100
        else:
            o_ev = o_sum = o_wr = 0
        if nn > 0:
            n_ev = sub_new[c].mean()
            n_sum = sub_new[c].sum()
            n_wr = (sub_new[c] > 0).mean() * 100
        else:
            n_ev = n_sum = n_wr = 0
        d_ev = n_ev - o_ev
        print(f'  {c:>6} | {o_ev:>+8.4f}% {o_sum:>+9.2f}% {o_wr:>5.0f}% | '
              f'{n_ev:>+8.4f}% {n_sum:>+9.2f}% {n_wr:>5.0f}% | {d_ev:>+7.4f}%')

print()
print('=' * 110)
print('核心对比：早上的最优 vs 现在的最优（同一120天窗口）')
print('=' * 110)
print()

configs = [
    (1, 'S4', 'S2'),
    (2, 'S4', 'S2'),
    (3, 'S3', 'S5.1'),
]
total_old_sum = 0
total_new_sum = 0
total_old_n = 0
total_new_n = 0

for s, old_best, new_best in configs:
    sub_old = old_df[old_df['sc'] == s]
    sub_new = new_df[new_df['sc'] == s]
    no = len(sub_old)
    nn = len(sub_new)
    o_ev = sub_old[old_best].mean() if no > 0 else 0
    n_ev = sub_new[new_best].mean() if nn > 0 else 0
    o_sum = sub_old[old_best].sum() if no > 0 else 0
    n_sum = sub_new[new_best].sum() if nn > 0 else 0
    o_wr = (sub_old[old_best] > 0).mean() * 100 if no > 0 else 0
    n_wr = (sub_new[new_best] > 0).mean() * 100 if nn > 0 else 0
    total_old_sum += o_sum
    total_new_sum += n_sum
    total_old_n += no
    total_new_n += nn
    print(f'  场景{s}: 旧={old_best}({no}笔) EV={o_ev:+.4f}% 累计={o_sum:+.2f}% 胜率={o_wr:.0f}%')
    print(f'         新={new_best}({nn}笔) EV={n_ev:+.4f}% 累计={n_sum:+.2f}% 胜率={n_wr:.0f}%')
    print(f'         EV变化: {n_ev - o_ev:+.4f}%  累计变化: {n_sum - o_sum:+.2f}%')
    print()

print(f'  总计: 旧{total_old_n}笔 累计={total_old_sum:+.2f}%  EV={total_old_sum/total_old_n:+.4f}%')
print(f'        新{total_new_n}笔 累计={total_new_sum:+.2f}%  EV={total_new_sum/total_new_n:+.4f}%')
print(f'        累计变化: {total_new_sum - total_old_sum:+.2f}%')
