# -*- coding: utf-8 -*-
"""
ER(20) 策略案例可视化
====================
从各场景随机抽取真实交易案例，画K线图标注入场/出场点，
生成HTML页面。
"""

import os
import random
import base64
import io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from datetime import timedelta
from signal_core import ExitTracker, SYMBOL_CONFIGS

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MA_FAST = 10
MA_SLOW = 20
ATR_PERIOD = 14
STOP_TICKS = 5

# 出场颜色
EXIT_COLORS = {
    'S1_newhigh':   '#e74c3c',  # 红
    'S2_pullback':  '#2ecc71',  # 绿
    'newhigh_prev': '#f39c12',  # 橙
    'ma10':         '#3498db',  # 蓝
    'ma20':         '#9b59b6',  # 紫
}
EXIT_LABELS_LONG = {
    'S1_newhigh':   '新高K线止损',
    'S2_pullback':  '回调低点止损',
    'newhigh_prev': '新高前根止损',
    'ma10':         '破10MA',
    'ma20':         '破20MA',
}
EXIT_LABELS_SHORT = {
    'S1_newhigh':   '新低K线止损',
    'S2_pullback':  '回调高点止损',
    'newhigh_prev': '新低前根止损',
    'ma10':         '破10MA',
    'ma20':         '破20MA',
}


def get_exit_label(ename, direction):
    if direction == 'short':
        return EXIT_LABELS_SHORT[ename]
    return EXIT_LABELS_LONG[ename]


def sym_key_to_config_key(sym_key):
    parts = sym_key.split('_', 1)
    if len(parts) == 2:
        return f"{parts[0]}.{parts[1]}"
    return sym_key


def get_tick_size(sym_key):
    config_key = sym_key_to_config_key(sym_key)
    if config_key in SYMBOL_CONFIGS:
        return SYMBOL_CONFIGS[config_key]['tick_size']
    return 1.0


def get_sym_name(sym_key):
    config_key = sym_key_to_config_key(sym_key)
    if config_key in SYMBOL_CONFIGS:
        return SYMBOL_CONFIGS[config_key]['name']
    return sym_key


def load_cached(symbol_key):
    path = os.path.join(CACHE_DIR, f"{symbol_key}_10min_170d.parquet")
    return pd.read_parquet(path) if os.path.exists(path) else None


def prepare_data(df):
    if 'ma_fast' not in df.columns:
        df['ma_fast'] = df['close'].rolling(MA_FAST).mean()
    if 'ma_slow' not in df.columns:
        df['ma_slow'] = df['close'].rolling(MA_SLOW).mean()
    df['trend'] = 0
    df.loc[df['ma_fast'] > df['ma_slow'], 'trend'] = 1
    df.loc[df['ma_fast'] < df['ma_slow'], 'trend'] = -1
    bar_moves = df['close'].diff().abs()
    net = (df['close'] - df['close'].shift(20)).abs()
    bar_sum = bar_moves.rolling(20).sum()
    df['er_20'] = net / bar_sum.replace(0, np.nan)
    median_move = bar_moves.rolling(20).median()
    capped_moves = bar_moves.clip(upper=3 * median_move)
    capped_sum = capped_moves.rolling(20).sum()
    df['er_20_adj'] = net / capped_sum.replace(0, np.nan)
    df['tr'] = np.maximum(df['high'] - df['low'],
        np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(ATR_PERIOD).mean()
    return df


def detect_ab_signals(df):
    signals = []
    n = len(df)
    warmup = max(MA_SLOW + ATR_PERIOD + 5, 30)
    b_below_start = -1
    b_pullback_low = None
    b_pullback_high = None
    prev_trend = 0

    for i in range(warmup, n):
        row = df.iloc[i]
        if pd.isna(row['ma_fast']) or pd.isna(row['ma_slow']):
            continue
        close, high, low, ma_f = row['close'], row['high'], row['low'], row['ma_fast']
        trend, er, atr = row['trend'], row['er_20'], row['atr']
        if pd.isna(er) or pd.isna(atr) or atr <= 0:
            continue
        if trend != prev_trend and trend != 0:
            b_below_start = -1
            b_pullback_low = None
            b_pullback_high = None
        prev_trend = trend
        if trend == 0:
            continue
        prev = df.iloc[i - 1]
        if pd.isna(prev['ma_fast']):
            continue
        prev_close, prev_ma_f = prev['close'], prev['ma_fast']
        found = []

        if trend == 1:
            if low <= ma_f and close > ma_f and prev_close > prev_ma_f:
                found.append(('A', 'long', low))
            if b_below_start == -1:
                if close < ma_f and prev_close >= prev_ma_f:
                    b_below_start = i
                    b_pullback_low = low
            else:
                b_pullback_low = min(b_pullback_low, low)
                if close > ma_f:
                    pb_bars = i - b_below_start
                    if pb_bars >= 4:
                        found.append(('B', 'long', b_pullback_low))
                    b_below_start = -1
                    b_pullback_low = None
        elif trend == -1:
            if high >= ma_f and close < ma_f and prev_close < prev_ma_f:
                found.append(('A', 'short', high))
            if b_below_start == -1:
                if close > ma_f and prev_close <= prev_ma_f:
                    b_below_start = i
                    b_pullback_high = high
            else:
                b_pullback_high = max(b_pullback_high, high)
                if close < ma_f:
                    pb_bars = i - b_below_start
                    if pb_bars >= 4:
                        found.append(('B', 'short', b_pullback_high))
                    b_below_start = -1
                    b_pullback_high = None

        for sig_type, direction, pb_extreme in found:
            ma_dist_atr = abs(close - ma_f) / atr
            er_adj = row['er_20_adj'] if not pd.isna(row.get('er_20_adj', np.nan)) else er
            signals.append({
                'type': sig_type, 'direction': direction,
                'idx': i, 'time': row['datetime'],
                'entry_price': close, 'pullback_extreme': pb_extreme,
                'er_20': er, 'er_20_adj': er_adj, 'atr': atr,
                'ma_dist_atr': round(ma_dist_atr, 3),
            })
    return signals


def simulate_5_exits_detail(df, sig, tick_size):
    """模拟5种出场，返回每种出场的idx、price、pnl"""
    entry_idx = sig['idx']
    entry_price = sig['entry_price']
    direction = sig['direction']
    pb_extreme = sig['pullback_extreme']
    is_long = direction == 'long'
    n = len(df)
    tick = tick_size * STOP_TICKS

    tracker = ExitTracker(direction=direction, entry_price=entry_price,
                          pullback_extreme=pb_extreme, tick_size=tick_size, stop_ticks=STOP_TICKS)
    s1_result = None
    s2_result = None

    exits_extra = {
        'newhigh_prev': {'done': False, 'stop': pb_extreme - tick if is_long else pb_extreme + tick,
                         'price': None, 'idx': None},
        'ma10': {'done': False, 'price': None, 'idx': None},
        'ma20': {'done': False, 'price': None, 'idx': None},
    }

    for j in range(entry_idx + 1, n):
        bar = df.iloc[j]
        prev_bar = df.iloc[j - 1]
        if pd.isna(bar['ma_fast']):
            continue

        if not tracker.all_done():
            exit_events, _ = tracker.process_bar(
                close=bar['close'], high=bar['high'], low=bar['low'],
                ma_fast=bar['ma_fast'], prev_close=prev_bar['close'])
            for ev in exit_events:
                if ev.strategy == 'S1' and s1_result is None:
                    s1_result = {'price': ev.exit_price, 'pnl': ev.pnl_pct, 'idx': j}
                elif ev.strategy == 'S2' and s2_result is None:
                    s2_result = {'price': ev.exit_price, 'pnl': ev.pnl_pct, 'idx': j}

        if not exits_extra['newhigh_prev']['done']:
            stop = exits_extra['newhigh_prev']['stop']
            if is_long and bar['low'] <= stop:
                exits_extra['newhigh_prev']['done'] = True
                exits_extra['newhigh_prev']['price'] = stop
                exits_extra['newhigh_prev']['idx'] = j
            elif not is_long and bar['high'] >= stop:
                exits_extra['newhigh_prev']['done'] = True
                exits_extra['newhigh_prev']['price'] = stop
                exits_extra['newhigh_prev']['idx'] = j
            else:
                if is_long and bar['close'] > prev_bar['close']:
                    exits_extra['newhigh_prev']['stop'] = max(stop, prev_bar['low'] - tick)
                elif not is_long and bar['close'] < prev_bar['close']:
                    exits_extra['newhigh_prev']['stop'] = min(stop, prev_bar['high'] + tick)

        if not exits_extra['ma10']['done']:
            if is_long and bar['close'] < bar['ma_fast']:
                exits_extra['ma10'] = {'done': True, 'price': bar['close'], 'idx': j}
            elif not is_long and bar['close'] > bar['ma_fast']:
                exits_extra['ma10'] = {'done': True, 'price': bar['close'], 'idx': j}

        if not exits_extra['ma20']['done']:
            if not pd.isna(bar['ma_slow']):
                if is_long and bar['close'] < bar['ma_slow']:
                    exits_extra['ma20'] = {'done': True, 'price': bar['close'], 'idx': j}
                elif not is_long and bar['close'] > bar['ma_slow']:
                    exits_extra['ma20'] = {'done': True, 'price': bar['close'], 'idx': j}

        all_done = tracker.all_done() and all(e['done'] for e in exits_extra.values())
        if all_done:
            break

    last_close = df.iloc[-1]['close']
    last_idx = len(df) - 1
    if s1_result is None:
        forced = tracker.force_close(last_close)
        for ev in forced:
            if ev.strategy == 'S1':
                s1_result = {'price': ev.exit_price, 'pnl': ev.pnl_pct, 'idx': last_idx}
            elif ev.strategy == 'S2' and s2_result is None:
                s2_result = {'price': ev.exit_price, 'pnl': ev.pnl_pct, 'idx': last_idx}
    elif s2_result is None:
        forced = tracker.force_close(last_close)
        for ev in forced:
            if ev.strategy == 'S2':
                s2_result = {'price': ev.exit_price, 'pnl': ev.pnl_pct, 'idx': last_idx}

    for key in exits_extra:
        if not exits_extra[key]['done']:
            pnl = (last_close - entry_price) / entry_price * 100 if is_long else (entry_price - last_close) / entry_price * 100
            exits_extra[key] = {'done': True, 'price': last_close, 'idx': last_idx, 'pnl': pnl}
        else:
            p = exits_extra[key]['price']
            exits_extra[key]['pnl'] = (p - entry_price) / entry_price * 100 if is_long else (entry_price - p) / entry_price * 100

    return {
        'S1_newhigh':   s1_result,
        'S2_pullback':  s2_result,
        'newhigh_prev': exits_extra['newhigh_prev'],
        'ma10':         exits_extra['ma10'],
        'ma20':         exits_extra['ma20'],
    }


def draw_trade_chart(df, sig, exits, sym_name):
    """画单笔交易K线图，返回base64 PNG"""
    entry_idx = sig['idx']
    is_long = sig['direction'] == 'long'

    # 确定显示范围：入场前10根到最远出场后5根
    max_exit_idx = max(ex['idx'] for ex in exits.values() if ex and ex['idx'] is not None)
    start = max(0, entry_idx - 15)
    end = min(len(df) - 1, max_exit_idx + 8)
    segment = df.iloc[start:end + 1].copy().reset_index(drop=True)
    entry_pos = entry_idx - start

    fig, ax = plt.subplots(figsize=(16, 7))

    # 画K线
    for i in range(len(segment)):
        row = segment.iloc[i]
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        color = '#ef5350' if c >= o else '#26a69a'  # 红涨绿跌
        # 影线
        ax.plot([i, i], [l, h], color=color, linewidth=0.8)
        # 实体
        body_bottom = min(o, c)
        body_height = abs(c - o)
        if body_height < (h - l) * 0.01:
            body_height = (h - l) * 0.01
        rect = plt.Rectangle((i - 0.35, body_bottom), 0.7, body_height,
                              facecolor=color, edgecolor=color, linewidth=0.5)
        ax.add_patch(rect)

    # 画MA
    if 'ma_fast' in segment.columns:
        ma10_vals = segment['ma_fast'].values
        valid = ~np.isnan(ma10_vals)
        xs = np.arange(len(segment))
        ax.plot(xs[valid], ma10_vals[valid], color='#FF9800', linewidth=1.2, alpha=0.8, label='MA10')
    if 'ma_slow' in segment.columns:
        ma20_vals = segment['ma_slow'].values
        valid = ~np.isnan(ma20_vals)
        ax.plot(xs[valid], ma20_vals[valid], color='#2196F3', linewidth=1.2, alpha=0.8, label='MA20')

    # 标注入场
    entry_price = sig['entry_price']
    marker = '^' if is_long else 'v'
    marker_color = '#FF1744'
    ax.scatter(entry_pos, entry_price, marker=marker, s=200, color=marker_color,
               zorder=10, edgecolors='black', linewidths=0.8)
    ax.annotate(f'入场 {entry_price:.1f}', (entry_pos, entry_price),
                textcoords="offset points", xytext=(-40, 20 if is_long else -25),
                fontsize=9, fontweight='bold', color=marker_color,
                arrowprops=dict(arrowstyle='->', color=marker_color, lw=1.2))

    # 标注5种出场
    y_offsets = [25, -25, 35, -35, 45]  # 避免重叠
    for idx_e, (ename, edata) in enumerate(exits.items()):
        if edata is None or edata['idx'] is None:
            continue
        exit_pos = edata['idx'] - start
        exit_price = edata['price']
        pnl = edata['pnl']
        color = EXIT_COLORS[ename]
        label = get_exit_label(ename, sig['direction'])

        if exit_pos < 0 or exit_pos >= len(segment):
            continue

        # 出场标记
        ax.scatter(exit_pos, exit_price, marker='x', s=120, color=color, zorder=9, linewidths=2)

        # 入场到出场连线
        ax.plot([entry_pos, exit_pos], [entry_price, exit_price],
                color=color, linewidth=1, linestyle='--', alpha=0.5)

        # 标注文字
        pnl_str = f'{pnl:+.2f}%'
        y_off = y_offsets[idx_e % len(y_offsets)]
        ax.annotate(f'{label} {exit_price:.1f} ({pnl_str})',
                    (exit_pos, exit_price),
                    textcoords="offset points", xytext=(10, y_off),
                    fontsize=7.5, color=color, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.8),
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.85))

    # X轴时间标签
    tick_step = max(1, len(segment) // 10)
    tick_positions = list(range(0, len(segment), tick_step))
    tick_labels = []
    for tp in tick_positions:
        dt = segment.iloc[tp]['datetime']
        if hasattr(dt, 'strftime'):
            tick_labels.append(dt.strftime('%m/%d %H:%M'))
        else:
            tick_labels.append(str(dt)[:16])
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30, fontsize=7)

    dir_str = '做多' if is_long else '做空'
    er_val = sig['er_20']
    er_adj = sig.get('er_20_adj', er_val)
    ma_dist = sig.get('ma_dist_atr', 0)
    ax.set_title(f"{sym_name} 10min | {sig['type']}类{dir_str} | ER={er_val:.3f} ER_adj={er_adj:.3f} | MA距离={ma_dist:.2f}ATR | {str(sig['time'])[:16]}",
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def main():
    random.seed(42)

    # 加载所有品种数据和信号
    print("加载数据和检测信号...")
    all_data = {}  # sym_key -> df
    all_signals = []

    for f in sorted(os.listdir(CACHE_DIR)):
        if not f.endswith('_10min_170d.parquet'):
            continue
        sym_key = f.replace('_10min_170d.parquet', '')
        df = pd.read_parquet(os.path.join(CACHE_DIR, f))
        df = prepare_data(df)
        cutoff = df['datetime'].iloc[-1] - timedelta(days=120)
        df_120 = df[df['datetime'] >= cutoff].copy().reset_index(drop=True)
        if len(df_120) < 200:
            continue
        all_data[sym_key] = df_120
        sigs = detect_ab_signals(df_120)
        for s in sigs:
            s['symbol'] = sym_key
            s['sym_name'] = get_sym_name(sym_key)
        all_signals.extend(sigs)

    print(f"  总信号: {len(all_signals)}")

    # 定义场景
    scenarios = [
        ('ER>=0.3 B类', lambda s: s['er_20'] >= 0.3 and s['type'] == 'B'),
        ('ER<0.3 B类',  lambda s: s['er_20'] < 0.3 and s['type'] == 'B'),
        ('ER>=0.3 A类', lambda s: s['er_20'] >= 0.3 and s['type'] == 'A'),
        ('ER<0.3 A类',  lambda s: s['er_20'] < 0.3 and s['type'] == 'A'),
        ('ER>=0.5 B类', lambda s: s['er_20'] >= 0.5 and s['type'] == 'B'),
        ('ER>=0.5 A类', lambda s: s['er_20'] >= 0.5 and s['type'] == 'A'),
        ('MA距离>1.5ATR (入场偏离大)', lambda s: s.get('ma_dist_atr', 0) > 1.5),
        ('MA距离<=0.5ATR (入场贴近MA)', lambda s: s.get('ma_dist_atr', 0) <= 0.5 and s['type'] == 'B'),
    ]

    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>ER(20) 策略案例可视化</title>
<style>
body { background: #1a1a2e; color: #e0e0e0; font-family: 'Microsoft YaHei', sans-serif; margin: 20px; }
h1 { color: #00d4ff; text-align: center; border-bottom: 2px solid #00d4ff; padding-bottom: 10px; }
h2 { color: #ffd700; margin-top: 40px; border-left: 4px solid #ffd700; padding-left: 12px; }
.scenario { margin-bottom: 50px; }
.trade-card { background: #16213e; border-radius: 10px; padding: 15px; margin: 15px 0;
              border: 1px solid #0f3460; }
.trade-card img { width: 100%; border-radius: 6px; }
.trade-info { display: flex; gap: 20px; flex-wrap: wrap; margin: 10px 0; }
.info-tag { background: #0f3460; padding: 4px 12px; border-radius: 15px; font-size: 13px; }
.pnl-table { width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 13px; }
.pnl-table th { background: #0f3460; padding: 6px 10px; text-align: center; }
.pnl-table td { padding: 5px 10px; text-align: center; border-bottom: 1px solid #1a1a3e; }
.win { color: #4CAF50; font-weight: bold; }
.loss { color: #f44336; font-weight: bold; }
.legend { display: flex; gap: 15px; flex-wrap: wrap; margin: 10px 0; justify-content: center; }
.legend-item { display: flex; align-items: center; gap: 5px; font-size: 12px; }
.legend-dot { width: 12px; height: 12px; border-radius: 50%; }
</style></head><body>
<h1>ER(20) 策略案例可视化 - 10min K线 32品种</h1>
<div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#e74c3c"></div>新高K线止损</div>
    <div class="legend-item"><div class="legend-dot" style="background:#2ecc71"></div>回调低点止损</div>
    <div class="legend-item"><div class="legend-dot" style="background:#f39c12"></div>新高前根止损</div>
    <div class="legend-item"><div class="legend-dot" style="background:#3498db"></div>破10MA</div>
    <div class="legend-item"><div class="legend-dot" style="background:#9b59b6"></div>破20MA</div>
</div>
""")

    for scenario_name, filter_fn in scenarios:
        candidates = [s for s in all_signals if filter_fn(s)]
        if len(candidates) < 3:
            continue

        # 抽3个：尽量不同品种，混合盈亏
        random.shuffle(candidates)
        selected = []
        used_syms = set()
        for s in candidates:
            if len(selected) >= 3:
                break
            if s['symbol'] not in used_syms and s['idx'] + 20 < len(all_data[s['symbol']]):
                selected.append(s)
                used_syms.add(s['symbol'])
        # 不够3个就放宽品种限制
        if len(selected) < 3:
            for s in candidates:
                if len(selected) >= 3:
                    break
                if s not in selected and s['idx'] + 20 < len(all_data[s['symbol']]):
                    selected.append(s)

        print(f"\n场景: {scenario_name} ({len(candidates)}个候选, 选{len(selected)}个)")
        html_parts.append(f'<div class="scenario"><h2>{scenario_name} ({len(candidates)}个信号)</h2>')

        for trade_i, sig in enumerate(selected):
            sym_key = sig['symbol']
            sym_name = sig['sym_name']
            df = all_data[sym_key]
            tick_size = get_tick_size(sym_key)

            ma_dist = sig.get('ma_dist_atr', 0)
            er_adj = sig.get('er_20_adj', sig['er_20'])
            print(f"  [{trade_i+1}] {sym_name} {sig['type']}类 {sig['direction']} ER={sig['er_20']:.3f} ER_adj={er_adj:.3f} MA距={ma_dist:.2f}")

            exits = simulate_5_exits_detail(df, sig, tick_size)
            img_b64 = draw_trade_chart(df, sig, exits, sym_name)

            dir_str = '做多' if sig['direction'] == 'long' else '做空'
            html_parts.append(f'''<div class="trade-card">
<div class="trade-info">
    <span class="info-tag">{sym_name}</span>
    <span class="info-tag">{sig['type']}类 {dir_str}</span>
    <span class="info-tag">ER(20) = {sig['er_20']:.3f}</span>
    <span class="info-tag">ER_adj = {er_adj:.3f}</span>
    <span class="info-tag">MA距离 = {ma_dist:.2f} ATR</span>
    <span class="info-tag">入场 {str(sig['time'])[:16]}</span>
    <span class="info-tag">入场价 {sig['entry_price']:.1f}</span>
</div>
<img src="data:image/png;base64,{img_b64}">
<table class="pnl-table">
<tr><th>出场方式</th><th>出场价</th><th>PnL</th><th>持仓K线数</th></tr>''')
            for ename in ['S1_newhigh', 'S2_pullback', 'newhigh_prev', 'ma10', 'ma20']:
                ex = exits[ename]
                if ex is None:
                    continue
                pnl = ex['pnl']
                bars = ex['idx'] - sig['idx']
                pnl_class = 'win' if pnl > 0 else 'loss'
                html_parts.append(
                    f'<tr><td style="color:{EXIT_COLORS[ename]}">{get_exit_label(ename, sig["direction"])}</td>'
                    f'<td>{ex["price"]:.1f}</td>'
                    f'<td class="{pnl_class}">{pnl:+.2f}%</td>'
                    f'<td>{bars}根</td></tr>')
            html_parts.append('</table></div>')

        html_parts.append('</div>')

    html_parts.append('</body></html>')

    out_path = os.path.join(OUTPUT_DIR, 'er_exits_visual.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))
    print(f"\n输出: {out_path}")


if __name__ == '__main__':
    main()
