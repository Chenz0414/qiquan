# -*- coding: utf-8 -*-
"""
最近60天全品种 - 3场景完整统计HTML报告
场景3边界已更新为<0.3ATR
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from data_loader import load_all, sym_name, tick_size
from signal_core import SignalDetector, ExitTracker, SYMBOL_CONFIGS, DEFAULT_STOP_TICKS
from report_engine import Report

LAST_DAYS = 60
BARS_PER_DAY = 57


def classify(t, er, dev):
    if er >= 0.7: return None  # 全局过滤：ER>=0.7正期望消失
    if t == 'A' and er >= 0.5 and dev >= 1.0: return 1
    if t == 'C' and dev >= 2.0: return 2
    if t == 'B' and er >= 0.5 and dev >= 0.1 and dev < 0.3: return 3
    return None


def make_stats_row(label, sub):
    """生成一行完整统计: 累计/均值/胜率/盈亏比 x 4种出场"""
    nn = len(sub)
    if nn == 0:
        return [label, 0] + ['-'] * 16
    row = [label, nn]
    for col in ['S1', 'S2', 'S3', 'S4']:
        total = sub[col].sum()
        avg = sub[col].mean()
        wins = (sub[col] > 0).sum()
        losses = (sub[col] < 0).sum()
        wr = wins / nn * 100
        w_avg = sub[sub[col] > 0][col].mean() if wins > 0 else 0
        l_avg = abs(sub[sub[col] < 0][col].mean()) if losses > 0 else 0.001
        pr = w_avg / l_avg if l_avg > 0 else 0
        row.extend([f'{total:+.2f}%', f'{avg:+.4f}%', f'{wr:.0f}%', f'{pr:.2f}'])
    return row


STATS_HEADERS = [
    '分组', '笔数',
    'S1累计', 'S1均值', 'S1胜率', 'S1盈亏比',
    'S2累计', 'S2均值', 'S2胜率', 'S2盈亏比',
    'S3累计', 'S3均值', 'S3胜率', 'S3盈亏比',
    'S4累计', 'S4均值', 'S4胜率', 'S4盈亏比',
]


def run():
    all_data = load_all(period_min=10, days=170, last_days=None,
                        emas=(10, 20, 120), er_periods=(20, 40), atr_period=14)
    print(f"加载品种数: {len(all_data)}")

    records = []
    for sym_key, df in sorted(all_data.items()):
        n = len(df)
        signal_start = max(130, n - LAST_DAYS * BARS_PER_DAY)
        name = sym_name(sym_key)
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
            er40 = row.get('er_40', 0)
            if pd.isna(er20): er20 = 0
            if pd.isna(er40): er40 = 0
            atr = row['atr']
            dev = abs(result.entry_price - row['ema10']) / atr if atr > 0 else 0
            sc = classify(result.signal_type, er20, dev)
            if sc is None:
                continue

            tracker = ExitTracker(
                direction=result.direction, entry_price=result.entry_price,
                pullback_extreme=result.pullback_extreme,
                tick_size=ts, stop_ticks=DEFAULT_STOP_TICKS)
            ex = {s: None for s in ['S1', 'S2', 'S3']}
            for j in range(i + 1, n):
                bar = df.iloc[j]
                prev = df.iloc[j - 1]
                if pd.isna(bar['ema10']):
                    continue
                if not tracker.all_done():
                    evts, _ = tracker.process_bar(
                        close=bar['close'], high=bar['high'], low=bar['low'],
                        ema10=bar['ema10'], prev_close=prev['close'],
                        prev_high=prev['high'], prev_low=prev['low'])
                    for ev in evts:
                        if ex[ev.strategy] is None:
                            ex[ev.strategy] = ev
                if tracker.all_done():
                    break
            forced = tracker.force_close(df.iloc[-1]['close'])
            for ev in forced:
                if ex[ev.strategy] is None:
                    ex[ev.strategy] = ev

            s1 = ex['S1'].pnl_pct
            s2 = ex['S2'].pnl_pct
            s3 = ex['S3'].pnl_pct
            s4 = (s1 + s2) / 2
            holding = (ex['S1'].exit_reason == 'backtest_end'
                       or ex['S2'].exit_reason == 'backtest_end')
            records.append({
                'symbol': name, 'sym_key': sym_key,
                'time': str(row['datetime'])[:16],
                'dir': result.direction,
                'type': result.signal_type,
                'er20': er20, 'er40': er40, 'dev': dev,
                'sc': sc,
                'S1': s1, 'S2': s2, 'S3': s3, 'S4': s4,
                'holding': holding,
            })

    data = pd.DataFrame(records)
    print(f"命中信号: {len(data)}笔")

    # ===== HTML 报告 =====
    rpt = Report(f'最近{LAST_DAYS}天全品种 - 3场景完整统计')

    # 一、总汇总
    rpt.add_section('一、总汇总')
    rows = []
    for s in [1, 2, 3]:
        rows.append(make_stats_row(f'场景{s}', data[data['sc'] == s]))
    rows.append(make_stats_row('合计', data))
    rpt.add_table(STATS_HEADERS, rows, highlight_pnl_cols=[2, 6, 10, 14])

    # 二、场景1 偏离度分档
    rpt.add_section('二、场景1 偏离度分档 (A类+ER>=0.5+偏离>=1.0ATR)')
    s1d = data[data['sc'] == 1]

    rpt.add_text('粗分档 (1.0ATR一档)')
    rows = []
    for lo, hi, label in [(1.0, 1.5, '1.0~1.5'), (1.5, 2.0, '1.5~2.0'),
                           (2.0, 3.0, '2.0~3.0'), (3.0, 99, '3.0+')]:
        sub = s1d[(s1d['dev'] >= lo) & (s1d['dev'] < hi)]
        rows.append(make_stats_row(f'{label}ATR', sub))
    rows.append(make_stats_row('场景1合计', s1d))
    rpt.add_table(STATS_HEADERS, rows, highlight_pnl_cols=[2, 6, 10, 14])

    rpt.add_text('细分档 (0.5ATR一档)')
    rows = []
    for lo in np.arange(1.0, 5.5, 0.5):
        hi = lo + 0.5
        sub = s1d[(s1d['dev'] >= lo) & (s1d['dev'] < hi)]
        if len(sub) > 0:
            rows.append(make_stats_row(f'{lo:.1f}~{hi:.1f}ATR', sub))
    sub_5p = s1d[s1d['dev'] >= 5.0]
    if len(sub_5p) > 0:
        rows.append(make_stats_row('5.0+ATR', sub_5p))
    rows.append(make_stats_row('场景1合计', s1d))
    rpt.add_table(STATS_HEADERS, rows, highlight_pnl_cols=[2, 6, 10, 14])

    # 场景1 按ER(20)分档
    rpt.add_text('场景1 按ER(20)分档')
    rows = []
    for lo, hi, label in [(0.5, 0.6, '0.5~0.6'), (0.6, 0.7, '0.6~0.7'), (0.7, 1.0, '0.7+')]:
        sub = s1d[(s1d['er20'] >= lo) & (s1d['er20'] < hi)]
        if len(sub) > 0:
            rows.append(make_stats_row(f'ER {label}', sub))
    rows.append(make_stats_row('场景1合计', s1d))
    rpt.add_table(STATS_HEADERS, rows, highlight_pnl_cols=[2, 6, 10, 14])

    # 三、场景2 分档
    rpt.add_section('三、场景2 分档 (C类+偏离>=2.0ATR)')
    s2d = data[data['sc'] == 2]

    rpt.add_text('按ER(20)分档')
    rows = []
    for lo, hi, label in [(0, 0.2, '<0.2'), (0.2, 0.3, '0.2~0.3'),
                           (0.3, 0.5, '0.3~0.5'), (0.5, 0.7, '0.5~0.7'), (0.7, 1.0, '0.7+')]:
        sub = s2d[(s2d['er20'] >= lo) & (s2d['er20'] < hi)]
        if len(sub) > 0:
            rows.append(make_stats_row(f'ER {label}', sub))
    rows.append(make_stats_row('场景2合计', s2d))
    rpt.add_table(STATS_HEADERS, rows, highlight_pnl_cols=[2, 6, 10, 14])

    rpt.add_text('按偏离度分档')
    rows = []
    for lo, hi, label in [(2.0, 2.5, '2.0~2.5'), (2.5, 3.0, '2.5~3.0'),
                           (3.0, 4.0, '3.0~4.0'), (4.0, 99, '4.0+')]:
        sub = s2d[(s2d['dev'] >= lo) & (s2d['dev'] < hi)]
        if len(sub) > 0:
            rows.append(make_stats_row(f'{label}ATR', sub))
    rows.append(make_stats_row('场景2合计', s2d))
    rpt.add_table(STATS_HEADERS, rows, highlight_pnl_cols=[2, 6, 10, 14])

    # 四、场景3 分档
    rpt.add_section('四、场景3 分档 (B类+ER>=0.5+偏离<0.3ATR)')
    s3d = data[data['sc'] == 3]

    rpt.add_text('按偏离度细分')
    rows = []
    for lo, hi, label in [(0, 0.1, '<0.1'), (0.1, 0.2, '0.1~0.2'), (0.2, 0.3, '0.2~0.3')]:
        sub = s3d[(s3d['dev'] >= lo) & (s3d['dev'] < hi)]
        if len(sub) > 0:
            rows.append(make_stats_row(f'{label}ATR', sub))
    rows.append(make_stats_row('场景3合计', s3d))
    rpt.add_table(STATS_HEADERS, rows, highlight_pnl_cols=[2, 6, 10, 14])

    rpt.add_text('按ER(20)分档')
    rows = []
    for lo, hi, label in [(0.5, 0.6, '0.5~0.6'), (0.6, 0.7, '0.6~0.7'), (0.7, 1.0, '0.7+')]:
        sub = s3d[(s3d['er20'] >= lo) & (s3d['er20'] < hi)]
        if len(sub) > 0:
            rows.append(make_stats_row(f'ER {label}', sub))
    rows.append(make_stats_row('场景3合计', s3d))
    rpt.add_table(STATS_HEADERS, rows, highlight_pnl_cols=[2, 6, 10, 14])

    # 五、各场景最优出场
    rpt.add_section('五、各场景最优出场策略')
    best_headers = ['场景', '笔数',
                    '第1名', '累计', '均值', '胜率', '盈亏比',
                    '第2名', '累计', '均值', '胜率', '盈亏比',
                    '第3名', '累计', '均值', '胜率', '盈亏比',
                    '第4名', '累计', '均值', '胜率', '盈亏比']
    best_rows = []
    for s in [1, 2, 3]:
        sub = data[data['sc'] == s]
        nn = len(sub)
        results = {}
        for col in ['S1', 'S2', 'S3', 'S4']:
            total = sub[col].sum()
            avg = sub[col].mean()
            wr = (sub[col] > 0).mean() * 100
            wins = (sub[col] > 0).sum()
            losses = (sub[col] < 0).sum()
            w_avg = sub[sub[col] > 0][col].mean() if wins > 0 else 0
            l_avg = abs(sub[sub[col] < 0][col].mean()) if losses > 0 else 0.001
            pr = w_avg / l_avg if l_avg > 0 else 0
            results[col] = (total, avg, wr, pr)
        ranked = sorted(results.items(), key=lambda x: -x[1][0])
        row = [f'场景{s}', nn]
        for name, (total, avg, wr, pr) in ranked:
            row.extend([name, f'{total:+.2f}%', f'{avg:+.4f}%', f'{wr:.0f}%', f'{pr:.2f}'])
        best_rows.append(row)
    # 合计
    sub = data
    nn = len(sub)
    results = {}
    for col in ['S1', 'S2', 'S3', 'S4']:
        total = sub[col].sum()
        avg = sub[col].mean()
        wr = (sub[col] > 0).mean() * 100
        wins = (sub[col] > 0).sum()
        losses = (sub[col] < 0).sum()
        w_avg = sub[sub[col] > 0][col].mean() if wins > 0 else 0
        l_avg = abs(sub[sub[col] < 0][col].mean()) if losses > 0 else 0.001
        pr = w_avg / l_avg if l_avg > 0 else 0
        results[col] = (total, avg, wr, pr)
    ranked = sorted(results.items(), key=lambda x: -x[1][0])
    row = ['合计', nn]
    for name, (total, avg, wr, pr) in ranked:
        row.extend([name, f'{total:+.2f}%', f'{avg:+.4f}%', f'{wr:.0f}%', f'{pr:.2f}'])
    best_rows.append(row)
    rpt.add_table(best_headers, best_rows)

    # 六、品种排行
    rpt.add_section('六、品种排行 (按S4累计排序)')
    sym_headers = ['品种', '笔数',
                   'S1累计', 'S1均值', 'S1胜率',
                   'S2累计', 'S2均值', 'S2胜率',
                   'S3累计', 'S3均值', 'S3胜率',
                   'S4累计', 'S4均值', 'S4胜率']
    sym_rows = []
    for sym in data.groupby('symbol')['S4'].sum().sort_values(ascending=False).index:
        sub = data[data['symbol'] == sym]
        nn = len(sub)
        row = [sym, nn]
        for col in ['S1', 'S2', 'S3', 'S4']:
            total = sub[col].sum()
            avg = sub[col].mean()
            wr = (sub[col] > 0).mean() * 100
            row.extend([f'{total:+.2f}%', f'{avg:+.3f}%', f'{wr:.0f}%'])
        sym_rows.append(row)
    rpt.add_table(sym_headers, sym_rows, highlight_pnl_cols=[2, 5, 8, 11])

    # 场景定义
    rpt.add_section('场景定义')
    rpt.add_text('场景1: A类 + ER(20)>=0.5 + 偏离>=1.0ATR (强趋势远离EMA10影线弹回)')
    rpt.add_text('场景2: C类 + 偏离>=2.0ATR (长回调蓄力爆发，不要求ER)')
    rpt.add_text('场景3: B类 + ER(20)>=0.5 + 偏离<0.3ATR (强趋势小幅回踩)')
    rpt.add_text(f'数据范围: 最近{LAST_DAYS}天, 全{len(all_data)}品种, 10min K线, 共{len(data)}笔命中信号')
    rpt.add_text('出场: S1=当根新高追踪 / S2=回调追踪 / S3=前根新高追踪 / S4=半仓S1+半仓S2')
    rpt.add_text('偏离度 = |entry_price - EMA10| / ATR(14) | 止损: 统一5跳')

    rpt.save('output/60day_full_stats.html')
    print('HTML报告已保存: output/60day_full_stats.html')


if __name__ == '__main__':
    run()
