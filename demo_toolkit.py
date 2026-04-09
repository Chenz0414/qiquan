# -*- coding: utf-8 -*-
"""
工具链 Demo：演示 data_loader + signal_core + stats_utils + chart_engine + report_engine 协同。

场景：对白银(ag)做C类信号回测，找出ER>=0.5中亏钱的交易渲染K线图，
     同时找出ER<0.5中赚钱的交易渲染K线图。生成对比报告。
"""

from data_loader import load_symbol, tick_size, sym_name, get_start_idx
from signal_core import SignalDetector, ExitTracker, DEFAULT_STOP_TICKS
from stats_utils import calc_ev, ev_line
from chart_engine import render_chart
from report_engine import Report

# ============ 1. 加载数据 ============
SYM = 'SHFE.ag'
df = load_symbol(SYM, emas=(10, 20, 120), er_periods=(20,))
if df is None:
    print(f"缓存不存在: {SYM}")
    exit()
print(f"加载 {sym_name(SYM)}: {len(df)} 根K线")

# ============ 2. 检测信号 ============
detector = SignalDetector(signal_types='ABC')
start_idx = get_start_idx(df, last_days=120)

signals = []
for i in range(len(df)):
    row = df.iloc[i]
    if any(row.get(c) is None for c in ['ema10', 'ema20', 'ema120']):
        continue
    result = detector.process_bar(
        close=row['close'], high=row['high'], low=row['low'],
        ema10=row['ema10'], ema20=row['ema20'], ema120=row['ema120'],
    )
    if result is not None and i >= start_idx:
        signals.append(result)

print(f"检测到 {len(signals)} 个信号 (A={sum(1 for s in signals if s.signal_type=='A')}, "
      f"B={sum(1 for s in signals if s.signal_type=='B')}, "
      f"C={sum(1 for s in signals if s.signal_type=='C')})")

# ============ 3. 模拟出场 ============
ts = tick_size(SYM)
trades = []

for sig in signals:
    tracker = ExitTracker(
        direction=sig.direction,
        entry_price=sig.entry_price,
        pullback_extreme=sig.pullback_extreme,
        tick_size=ts,
        stop_ticks=DEFAULT_STOP_TICKS,
    )
    entry_idx = sig.bar_index
    s2_result = None

    for j in range(entry_idx + 1, len(df)):
        bar = df.iloc[j]
        prev_bar = df.iloc[j - 1]
        if bar.get('ema10') is None:
            continue

        exits, _ = tracker.process_bar(
            close=bar['close'], high=bar['high'], low=bar['low'],
            ema10=bar['ema10'], prev_close=prev_bar['close'],
            prev_high=prev_bar['high'], prev_low=prev_bar['low'],
        )
        for ev in exits:
            if ev.strategy == 'S2' and s2_result is None:
                s2_result = {'idx': j, 'price': ev.exit_price, 'pnl': ev.pnl_pct, 'reason': ev.exit_reason}
        if tracker.all_done():
            break

    if s2_result is None:
        forced = tracker.force_close(df.iloc[-1]['close'])
        for ev in forced:
            if ev.strategy == 'S2':
                s2_result = {'idx': len(df)-1, 'price': ev.exit_price, 'pnl': ev.pnl_pct, 'reason': 'backtest_end'}

    er_val = df.iloc[entry_idx].get('er_20', 0) or 0
    trades.append({
        'signal': sig,
        'entry_idx': entry_idx,
        'er_20': er_val,
        's2': s2_result,
    })

# 过滤掉 backtest_end
valid_trades = [t for t in trades if t['s2']['reason'] != 'backtest_end']
print(f"有效交易: {len(valid_trades)} (排除 {len(trades) - len(valid_trades)} 笔 backtest_end)")

# ============ 4. 统计 ============
all_pnls = [t['s2']['pnl'] for t in valid_trades]
er_high = [t for t in valid_trades if t['er_20'] >= 0.5]
er_low = [t for t in valid_trades if t['er_20'] < 0.5]

print(ev_line(f'{sym_name(SYM)} 全部', all_pnls))
print(ev_line(f'{sym_name(SYM)} ER>=0.5', [t['s2']['pnl'] for t in er_high]))
print(ev_line(f'{sym_name(SYM)} ER<0.5', [t['s2']['pnl'] for t in er_low]))

# ============ 5. 生成报告 ============
rpt = Report(f'{sym_name(SYM)} S2 案例分析')

# 5a. 统计表
rpt.add_section('S2 统计摘要')
rpt.add_table(
    headers=['分组', 'N', 'EV', 'WR%', 'PR', '累计%'],
    rows=[
        ['全部'] + [calc_ev(all_pnls)[k] for k in ('N', 'EV', 'wr', 'pr', 'sum_pnl')],
        ['ER>=0.5'] + [calc_ev([t['s2']['pnl'] for t in er_high])[k] for k in ('N', 'EV', 'wr', 'pr', 'sum_pnl')],
        ['ER<0.5'] + [calc_ev([t['s2']['pnl'] for t in er_low])[k] for k in ('N', 'EV', 'wr', 'pr', 'sum_pnl')],
    ],
    highlight_pnl_cols=[2, 5],
)

# 5b. ER>=0.5 中亏钱的 — 为什么好信号也亏了？
losses_in_high_er = sorted([t for t in er_high if t['s2']['pnl'] <= 0],
                            key=lambda x: x['s2']['pnl'])
rpt.add_section(f'ER>=0.5 亏损交易 ({len(losses_in_high_er)} 笔)',
                '好趋势中为什么亏了？看K线找原因')
for t in losses_in_high_er[:8]:  # 最多8个案例
    sig = t['signal']
    s2 = t['s2']
    init_stop = sig.pullback_extreme - ts * DEFAULT_STOP_TICKS if sig.direction == 'long' \
                else sig.pullback_extreme + ts * DEFAULT_STOP_TICKS
    chart = render_chart(
        df, entry_idx=t['entry_idx'], direction=sig.direction,
        exits=[{'name': 'S2', 'idx': s2['idx'], 'price': s2['price']}],
        stop_price=init_stop,
        title=f"{sym_name(SYM)} {sig.signal_type} {sig.direction} | ER={t['er_20']:.2f}",
        extra_info={
            'PnL': f"{s2['pnl']:+.2f}%",
            'Type': sig.signal_type,
            'PB_bars': sig.pullback_bars,
        },
    )
    rpt.add_chart(chart)

# 5c. ER<0.5 中赚钱的 — 差趋势中为什么赚了？
wins_in_low_er = sorted([t for t in er_low if t['s2']['pnl'] > 0],
                         key=lambda x: x['s2']['pnl'], reverse=True)
rpt.add_section(f'ER<0.5 盈利交易 ({len(wins_in_low_er)} 笔)',
                '弱趋势中为什么赚了？看K线找机会')
for t in wins_in_low_er[:8]:
    sig = t['signal']
    s2 = t['s2']
    init_stop = sig.pullback_extreme - ts * DEFAULT_STOP_TICKS if sig.direction == 'long' \
                else sig.pullback_extreme + ts * DEFAULT_STOP_TICKS
    chart = render_chart(
        df, entry_idx=t['entry_idx'], direction=sig.direction,
        exits=[{'name': 'S2', 'idx': s2['idx'], 'price': s2['price']}],
        stop_price=init_stop,
        title=f"{sym_name(SYM)} {sig.signal_type} {sig.direction} | ER={t['er_20']:.2f}",
        extra_info={
            'PnL': f"{s2['pnl']:+.2f}%",
            'Type': sig.signal_type,
            'PB_bars': sig.pullback_bars,
        },
    )
    rpt.add_chart(chart)

# 5d. 保存
import os
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'demo_toolkit.html')
rpt.save(out_path)
print(f"\n报告已生成: {out_path}")
