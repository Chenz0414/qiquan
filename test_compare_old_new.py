# -*- coding: utf-8 -*-
"""对比修改前后60天效果变化"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd
from data_loader import load_all, sym_name, tick_size
from signal_core import SignalDetector, ExitTracker, SYMBOL_CONFIGS, DEFAULT_STOP_TICKS

LAST_DAYS = 60; BARS_PER_DAY = 57

def classify_old(t, er, dev):
    """原始版本"""
    if t == 'A' and er >= 0.5 and dev >= 1.0: return 1
    if t == 'C' and dev >= 2.0: return 2
    if t == 'B' and er >= 0.5 and dev < 0.5: return 3
    return None

def classify_new(t, er, dev):
    """最终版本"""
    if er >= 0.7: return None
    if t == 'A' and er >= 0.5 and dev >= 1.0: return 1
    if t == 'C' and dev >= 2.0: return 2
    if t == 'B' and er >= 0.5 and dev >= 0.1 and dev < 0.3: return 3
    return None

all_data = load_all(period_min=10, days=170, last_days=None,
                    emas=(10, 20, 120), er_periods=(20, 40), atr_period=14)

old_records = []
new_records = []
cut_records = []  # 被砍掉的

for sym_key, df in sorted(all_data.items()):
    n = len(df)
    signal_start = max(130, n - LAST_DAYS * BARS_PER_DAY)
    ts = tick_size(sym_key)
    name = sym_name(sym_key)
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
        if pd.isna(er20): er20 = 0
        atr = row['atr']
        dev = abs(result.entry_price - row['ema10']) / atr if atr > 0 else 0

        sc_old = classify_old(result.signal_type, er20, dev)
        sc_new = classify_new(result.signal_type, er20, dev)

        if sc_old is None and sc_new is None:
            continue

        # 模拟出场
        tracker = ExitTracker(
            direction=result.direction, entry_price=result.entry_price,
            pullback_extreme=result.pullback_extreme,
            tick_size=ts, stop_ticks=DEFAULT_STOP_TICKS)
        ex = {s: None for s in ['S1', 'S2', 'S3']}
        for j in range(i + 1, n):
            bar = df.iloc[j]; prev = df.iloc[j - 1]
            if pd.isna(bar['ema10']): continue
            if not tracker.all_done():
                evts, _ = tracker.process_bar(
                    close=bar['close'], high=bar['high'], low=bar['low'],
                    ema10=bar['ema10'], prev_close=prev['close'],
                    prev_high=prev['high'], prev_low=prev['low'])
                for ev in evts:
                    if ex[ev.strategy] is None: ex[ev.strategy] = ev
            if tracker.all_done(): break
        forced = tracker.force_close(df.iloc[-1]['close'])
        for ev in forced:
            if ex[ev.strategy] is None: ex[ev.strategy] = ev
        s1 = ex['S1'].pnl_pct; s2 = ex['S2'].pnl_pct
        s3 = ex['S3'].pnl_pct; s4 = (s1 + s2) / 2

        rec = {'S1': s1, 'S2': s2, 'S3': s3, 'S4': s4}

        if sc_old is not None:
            r = rec.copy(); r['sc'] = sc_old
            old_records.append(r)
        if sc_new is not None:
            r = rec.copy(); r['sc'] = sc_new
            new_records.append(r)

        # 被砍掉的
        if sc_old is not None and sc_new is None:
            if er20 >= 0.7:
                reason = 'ER>=0.7'
            elif dev < 0.1:
                reason = 'dev<0.1'
            elif dev >= 0.3:
                reason = 'dev0.3~0.5'
            else:
                reason = 'other'
            r = rec.copy()
            r['reason'] = reason
            r['sc_old'] = sc_old
            r['symbol'] = name
            cut_records.append(r)

old = pd.DataFrame(old_records)
new = pd.DataFrame(new_records)
cut = pd.DataFrame(cut_records)

print('=' * 100)
print('最近60天 修改前后对比')
print('=' * 100)
print()
print('修改内容:')
print('  1. 全局增加 ER(20)>=0.7 禁止开仓（正期望消失）')
print('  2. 场景3偏离度从 <0.5ATR 收紧为 0.1~0.3ATR（砍掉两端噪音）')
print()

# 逐场景对比
for s in [1, 2, 3]:
    sub_old = old[old['sc'] == s]
    sub_new = new[new['sc'] == s]
    no = len(sub_old); nn = len(sub_new)
    diff_n = nn - no
    print(f'--- 场景{s} ---')
    print(f'  笔数: {no} -> {nn} ({diff_n:+d})')
    for col in ['S1', 'S2', 'S3', 'S4']:
        o_sum = sub_old[col].sum()
        n_sum = sub_new[col].sum()
        o_avg = sub_old[col].mean() if no > 0 else 0
        n_avg = sub_new[col].mean() if nn > 0 else 0
        o_wr = (sub_old[col] > 0).mean() * 100 if no > 0 else 0
        n_wr = (sub_new[col] > 0).mean() * 100 if nn > 0 else 0
        d_sum = n_sum - o_sum
        d_avg = n_avg - o_avg
        print(f'  {col}: 累计 {o_sum:>+8.2f}% -> {n_sum:>+8.2f}% ({d_sum:>+7.2f}%) '
              f'| 均值 {o_avg:>+.4f} -> {n_avg:>+.4f} ({d_avg:>+.4f}) '
              f'| 胜率 {o_wr:.0f}% -> {n_wr:.0f}%')
    print()

# 合计
print(f'--- 合计 ---')
no = len(old); nn = len(new)
print(f'  笔数: {no} -> {nn} ({nn - no:+d})')
for col in ['S1', 'S2', 'S3', 'S4']:
    o_sum = old[col].sum(); n_sum = new[col].sum()
    o_avg = old[col].mean(); n_avg = new[col].mean()
    o_wr = (old[col] > 0).mean() * 100; n_wr = (new[col] > 0).mean() * 100
    d_sum = n_sum - o_sum; d_avg = n_avg - o_avg
    print(f'  {col}: 累计 {o_sum:>+8.2f}% -> {n_sum:>+8.2f}% ({d_sum:>+7.2f}%) '
          f'| 均值 {o_avg:>+.4f} -> {n_avg:>+.4f} ({d_avg:>+.4f}) '
          f'| 胜率 {o_wr:.0f}% -> {n_wr:.0f}%')
print()

# 被砍掉的信号分析
print('=' * 100)
print('被砍掉的信号分析')
print('=' * 100)
print()
for reason in ['ER>=0.7', 'dev<0.1', 'dev0.3~0.5']:
    sub = cut[cut['reason'] == reason]
    if len(sub) == 0:
        continue
    sc_dist = sub['sc_old'].value_counts().to_dict()
    print(f'{reason}: {len(sub)}笔 (原属场景: {sc_dist})')
    for col in ['S1', 'S2', 'S3', 'S4']:
        wr = (sub[col] > 0).mean() * 100
        print(f'  {col}: 累计={sub[col].sum():>+7.2f}% 均值={sub[col].mean():>+.4f}% 胜率={wr:.0f}%')
    print()
