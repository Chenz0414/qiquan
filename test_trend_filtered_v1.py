# -*- coding: utf-8 -*-
"""
被MA20>MA120过滤掉的信号 K线可视化
===================================
找出 MA10>MA20 认为有趋势、但 MA20>MA120 不认为有趋势的信号，
随机抽10个画K线图，看过滤是否合理。
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
from datetime import timedelta
from signal_core import SYMBOL_CONFIGS

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STOP_LOSS_PCT = 2.0
TAKE_PROFIT_PCT = 5.0

MA_COLORS = {
    'ema_10': ('#FF9800', 'EMA10'),
    'ema_20': ('#2196F3', 'EMA20'),
    'ema_60': ('#E91E63', 'EMA60'),
    'ema_120': ('#9C27B0', 'EMA120'),
}


def sym_key_to_config_key(sym_key):
    parts = sym_key.split('_', 1)
    return f"{parts[0]}.{parts[1]}" if len(parts) == 2 else sym_key


def get_sym_name(sym_key):
    ck = sym_key_to_config_key(sym_key)
    return SYMBOL_CONFIGS[ck]['name'] if ck in SYMBOL_CONFIGS else sym_key


def load_cached(symbol_key):
    path = os.path.join(CACHE_DIR, f"{symbol_key}_10min_170d.parquet")
    return pd.read_parquet(path) if os.path.exists(path) else None


def prepare_data(df):
    for p in [10, 20, 60, 120]:
        df[f'ema_{p}'] = df['close'].ewm(span=p, adjust=False).mean()
    df['tr'] = np.maximum(df['high'] - df['low'],
        np.maximum(abs(df['high'] - df['close'].shift(1)),
                   abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
    bar_moves = df['close'].diff().abs()
    net = (df['close'] - df['close'].shift(20)).abs()
    bar_sum = bar_moves.rolling(20).sum()
    df['er_20'] = net / bar_sum.replace(0, np.nan)
    return df


def trend_ma10_ma20(df, i):
    row = df.iloc[i]
    m10, m20 = row['ema_10'], row['ema_20']
    if pd.isna(m10) or pd.isna(m20):
        return 0
    return 1 if m10 > m20 else (-1 if m10 < m20 else 0)


def trend_ma20_ma120(df, i):
    row = df.iloc[i]
    m20, m120 = row['ema_20'], row['ema_120']
    if pd.isna(m20) or pd.isna(m120):
        return 0
    return 1 if m20 > m120 else (-1 if m20 < m120 else 0)


def detect_filtered_signals(df):
    """检测 MA10>MA20 有信号但 MA20>MA120 无信号的情况"""
    signals = []
    n = len(df)
    warmup = 130

    # MA10>MA20 的 B类状态
    b_below_start_1 = -1
    b_pullback_low_1 = None
    b_pullback_high_1 = None
    prev_trend_1 = 0

    for i in range(warmup, n):
        row = df.iloc[i]
        ma_f = row['ema_10']
        if pd.isna(ma_f):
            continue

        close = row['close']
        high = row['high']
        low = row['low']
        atr = row['atr']
        er = row['er_20']

        trend_1 = trend_ma10_ma20(df, i)  # MA10>MA20 的趋势
        trend_2 = trend_ma20_ma120(df, i)  # MA20>MA120 的趋势

        if trend_1 != prev_trend_1 and trend_1 != 0:
            b_below_start_1 = -1
            b_pullback_low_1 = None
            b_pullback_high_1 = None
        prev_trend_1 = trend_1
        if trend_1 == 0:
            continue

        if i < 1:
            continue
        prev = df.iloc[i - 1]
        prev_close = prev['close']
        prev_ma_f = prev['ema_10']
        if pd.isna(prev_ma_f):
            continue

        found = []

        if trend_1 == 1:
            if low <= ma_f and close > ma_f and prev_close > prev_ma_f:
                found.append(('A', 'long'))
            if b_below_start_1 == -1:
                if close < ma_f and prev_close >= prev_ma_f:
                    b_below_start_1 = i
                    b_pullback_low_1 = low
            else:
                b_pullback_low_1 = min(b_pullback_low_1, low)
                if close > ma_f:
                    pb_bars = i - b_below_start_1
                    if pb_bars >= 4:
                        found.append(('B', 'long'))
                    b_below_start_1 = -1
                    b_pullback_low_1 = None

        elif trend_1 == -1:
            if high >= ma_f and close < ma_f and prev_close < prev_ma_f:
                found.append(('A', 'short'))
            if b_below_start_1 == -1:
                if close > ma_f and prev_close <= prev_ma_f:
                    b_below_start_1 = i
                    b_pullback_high_1 = high
            else:
                b_pullback_high_1 = max(b_pullback_high_1, high)
                if close < ma_f:
                    pb_bars = i - b_below_start_1
                    if pb_bars >= 4:
                        found.append(('B', 'short'))
                    b_below_start_1 = -1
                    b_pullback_high_1 = None

        for sig_type, direction in found:
            # 关键：只保留 MA20>MA120 不认同趋势方向的信号
            # 即 trend_2 != trend_1（要么反向，要么无趋势）
            if trend_2 == trend_1:
                continue  # MA20>MA120 也认同，不是被过滤的

            ma_dist_atr = abs(close - ma_f) / atr if (not pd.isna(atr) and atr > 0) else 0

            # 记录MA20>MA120此时的状态
            if trend_2 == 0:
                filter_reason = 'MA20≈MA120 (无趋势)'
            elif trend_2 == -trend_1:
                filter_reason = 'MA20>MA120 反向'
            else:
                filter_reason = f'trend2={trend_2}'

            signals.append({
                'type': sig_type,
                'direction': direction,
                'idx': i,
                'time': row['datetime'],
                'entry_price': close,
                'er_20': er if not pd.isna(er) else 0,
                'atr': atr if not pd.isna(atr) else 0,
                'ma_dist_atr': round(ma_dist_atr, 3),
                'trend_1020': trend_1,
                'trend_20120': trend_2,
                'filter_reason': filter_reason,
            })

    return signals


def simulate_fixed_exit(df, sig):
    entry_idx = sig['idx']
    entry_price = sig['entry_price']
    is_long = sig['direction'] == 'long'
    n = len(df)

    sl_price = entry_price * (1 - STOP_LOSS_PCT / 100) if is_long else entry_price * (1 + STOP_LOSS_PCT / 100)
    tp_price = entry_price * (1 + TAKE_PROFIT_PCT / 100) if is_long else entry_price * (1 - TAKE_PROFIT_PCT / 100)

    for j in range(entry_idx + 1, n):
        bar = df.iloc[j]
        if is_long:
            if bar['low'] <= sl_price:
                return j, sl_price, -STOP_LOSS_PCT, '止损'
            if bar['high'] >= tp_price:
                return j, tp_price, +TAKE_PROFIT_PCT, '止盈'
        else:
            if bar['high'] >= sl_price:
                return j, sl_price, -STOP_LOSS_PCT, '止损'
            if bar['low'] <= tp_price:
                return j, tp_price, +TAKE_PROFIT_PCT, '止盈'

    last_close = df.iloc[-1]['close']
    if is_long:
        pnl = (last_close - entry_price) / entry_price * 100
    else:
        pnl = (entry_price - last_close) / entry_price * 100
    return n - 1, last_close, pnl, '未触发'


def draw_trade_chart(df, sig, exit_info, sym_name):
    entry_idx = sig['idx']
    is_long = sig['direction'] == 'long'
    exit_idx, exit_price, pnl, exit_reason = exit_info

    start = max(0, entry_idx - 25)
    end = min(len(df) - 1, exit_idx + 8)
    segment = df.iloc[start:end + 1].copy().reset_index(drop=True)
    entry_pos = entry_idx - start
    exit_pos = exit_idx - start

    fig, ax = plt.subplots(figsize=(16, 7))

    # K线
    for i in range(len(segment)):
        row = segment.iloc[i]
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        color = '#ef5350' if c >= o else '#26a69a'
        ax.plot([i, i], [l, h], color=color, linewidth=0.8)
        body_bottom = min(o, c)
        body_height = max(abs(c - o), (h - l) * 0.01)
        rect = plt.Rectangle((i - 0.35, body_bottom), 0.7, body_height,
                              facecolor=color, edgecolor=color, linewidth=0.5)
        ax.add_patch(rect)

    # 画4条EMA
    xs = np.arange(len(segment))
    for ma_col in ['ema_10', 'ema_20', 'ema_120']:
        if ma_col in segment.columns:
            vals = segment[ma_col].values
            valid = ~np.isnan(vals)
            clr, label = MA_COLORS[ma_col]
            lw = 1.8 if ma_col in ['ema_20', 'ema_120'] else 1.0
            style = '--' if ma_col == 'ema_10' else '-'
            ax.plot(xs[valid], vals[valid], color=clr, linewidth=lw,
                    linestyle=style, alpha=0.85, label=label)

    # 入场标注
    marker = '^' if is_long else 'v'
    ax.scatter(entry_pos, sig['entry_price'], marker=marker, s=200,
               color='#FF1744', zorder=10, edgecolors='black', linewidths=0.8)
    ax.annotate(f'入场 {sig["entry_price"]:.1f}',
                (entry_pos, sig['entry_price']),
                textcoords="offset points", xytext=(-40, 20 if is_long else -25),
                fontsize=9, fontweight='bold', color='#FF1744',
                arrowprops=dict(arrowstyle='->', color='#FF1744', lw=1.2))

    # 出场标注
    exit_color = '#4CAF50' if pnl > 0 else '#f44336'
    ax.scatter(exit_pos, exit_price, marker='x', s=180, color=exit_color,
               zorder=10, linewidths=2.5)
    ax.plot([entry_pos, exit_pos], [sig['entry_price'], exit_price],
            color=exit_color, linewidth=1.2, linestyle='--', alpha=0.6)
    ax.annotate(f'{exit_reason} {exit_price:.1f} ({pnl:+.2f}%)',
                (exit_pos, exit_price),
                textcoords="offset points", xytext=(15, -20 if is_long else 20),
                fontsize=9, fontweight='bold', color=exit_color,
                arrowprops=dict(arrowstyle='->', color=exit_color, lw=1),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=exit_color, alpha=0.9))

    # 止损/止盈线
    sl_price = sig['entry_price'] * (1 - STOP_LOSS_PCT / 100) if is_long else sig['entry_price'] * (1 + STOP_LOSS_PCT / 100)
    tp_price = sig['entry_price'] * (1 + TAKE_PROFIT_PCT / 100) if is_long else sig['entry_price'] * (1 - TAKE_PROFIT_PCT / 100)
    ax.axhline(y=sl_price, color='#f44336', linewidth=0.8, linestyle=':', alpha=0.5)
    ax.axhline(y=tp_price, color='#4CAF50', linewidth=0.8, linestyle=':', alpha=0.5)
    ax.text(len(segment) - 1, sl_price, f' SL {sl_price:.1f}', fontsize=7, color='#f44336', va='center')
    ax.text(len(segment) - 1, tp_price, f' TP {tp_price:.1f}', fontsize=7, color='#4CAF50', va='center')

    # X轴
    tick_step = max(1, len(segment) // 10)
    tick_positions = list(range(0, len(segment), tick_step))
    tick_labels = [segment.iloc[tp]['datetime'].strftime('%m/%d %H:%M')
                   if hasattr(segment.iloc[tp]['datetime'], 'strftime')
                   else str(segment.iloc[tp]['datetime'])[:16]
                   for tp in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30, fontsize=7)

    dir_str = '做多' if is_long else '做空'
    ax.set_title(
        f"{sym_name} 10min | 被过滤信号 | {sig['type']}类{dir_str} | "
        f"原因: {sig['filter_reason']} | "
        f"ER={sig['er_20']:.3f} | {str(sig['time'])[:16]}",
        fontsize=11, fontweight='bold', color='#ff6b6b')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def main():
    random.seed(77)

    print("加载数据，检测被过滤的信号...")
    sym_data = {}
    all_filtered = []

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
        sym_data[sym_key] = df_120
        sigs = detect_filtered_signals(df_120)
        for s in sigs:
            s['symbol'] = sym_key
            s['sym_name'] = get_sym_name(sym_key)
            exit_info = simulate_fixed_exit(df_120, s)
            s['exit_info'] = exit_info
        all_filtered.extend(sigs)

    # 统计
    wins = [s for s in all_filtered if s['exit_info'][2] > 0]
    losses = [s for s in all_filtered if s['exit_info'][2] <= 0]
    n_total = len(all_filtered)
    wr = len(wins) / n_total * 100 if n_total > 0 else 0
    avg_w = np.mean([s['exit_info'][2] for s in wins]) if wins else 0
    avg_l = abs(np.mean([s['exit_info'][2] for s in losses])) if losses else 0.001
    ratio = avg_w / avg_l
    ev = wr / 100 * ratio - (1 - wr / 100)

    n_reverse = sum(1 for s in all_filtered if '反向' in s['filter_reason'])
    n_none = sum(1 for s in all_filtered if '无趋势' in s['filter_reason'])

    print(f"\n  被过滤信号总计: {n_total}笔")
    print(f"    MA20>MA120反向: {n_reverse}笔  无趋势: {n_none}笔")
    print(f"    胜率={wr:.1f}%  盈亏比={ratio:.2f}  期望={ev:+.2f}")
    print(f"    (对比: MA10>MA20全部期望=+0.13, MA20>MA120全部期望=+0.16)")
    if ev < 0.13:
        print(f"    >>> 过滤掉的信号期望({ev:+.2f}) < 保留的信号期望(+0.16)，过滤合理！")

    # 分类统计
    for label, subset in [('A类', [s for s in all_filtered if s['type'] == 'A']),
                           ('B类', [s for s in all_filtered if s['type'] == 'B'])]:
        if not subset:
            continue
        sw = [s for s in subset if s['exit_info'][2] > 0]
        sl = [s for s in subset if s['exit_info'][2] <= 0]
        nn = len(subset)
        w = len(sw) / nn * 100
        aw = np.mean([s['exit_info'][2] for s in sw]) if sw else 0
        al = abs(np.mean([s['exit_info'][2] for s in sl])) if sl else 0.001
        r = aw / al
        e = w / 100 * r - (1 - w / 100)
        print(f"    {label}: N={nn} 胜率={w:.1f}% 盈亏比={r:.2f} 期望={e:+.2f}")

    # 按过滤原因分
    for reason_tag, reason_label in [('反向', 'MA20>MA120反向'), ('无趋势', 'MA20≈MA120无趋势')]:
        subset = [s for s in all_filtered if reason_tag in s['filter_reason']]
        if not subset:
            continue
        sw = [s for s in subset if s['exit_info'][2] > 0]
        sl = [s for s in subset if s['exit_info'][2] <= 0]
        nn = len(subset)
        w = len(sw) / nn * 100
        aw = np.mean([s['exit_info'][2] for s in sw]) if sw else 0
        al = abs(np.mean([s['exit_info'][2] for s in sl])) if sl else 0.001
        r = aw / al
        e = w / 100 * r - (1 - w / 100)
        print(f"    {reason_label}: N={nn} 胜率={w:.1f}% 盈亏比={r:.2f} 期望={e:+.2f}")

    # 选10个：混合盈亏、不同品种、两种过滤原因都有
    valid = [s for s in all_filtered if s['idx'] + 15 < len(sym_data[s['symbol']])]
    reverse_sigs = [s for s in valid if '反向' in s['filter_reason']]
    none_sigs = [s for s in valid if '无趋势' in s['filter_reason']]

    selected = []
    used_syms = set()

    # 先选反向的（盈亏各半，不同品种）
    random.shuffle(reverse_sigs)
    rev_wins = [s for s in reverse_sigs if s['exit_info'][2] > 0]
    rev_losses = [s for s in reverse_sigs if s['exit_info'][2] <= 0]
    for s in rev_wins:
        if len(selected) >= 3:
            break
        if s['symbol'] not in used_syms:
            selected.append(s)
            used_syms.add(s['symbol'])
    for s in rev_losses:
        if len(selected) >= 5:
            break
        if s['symbol'] not in used_syms:
            selected.append(s)
            used_syms.add(s['symbol'])

    # 再选无趋势的
    random.shuffle(none_sigs)
    none_wins = [s for s in none_sigs if s['exit_info'][2] > 0]
    none_losses = [s for s in none_sigs if s['exit_info'][2] <= 0]
    for s in none_wins:
        if len(selected) >= 7:
            break
        if s['symbol'] not in used_syms:
            selected.append(s)
            used_syms.add(s['symbol'])
    for s in none_losses:
        if len(selected) >= 10:
            break
        if s['symbol'] not in used_syms:
            selected.append(s)
            used_syms.add(s['symbol'])

    # 不够10个放宽品种限制
    for pool in [reverse_sigs, none_sigs]:
        random.shuffle(pool)
        for s in pool:
            if len(selected) >= 10:
                break
            if s not in selected:
                selected.append(s)

    # HTML
    html_parts = [f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>被MA20>MA120过滤掉的信号 v1</title>
<style>
body {{ background: #1a1a2e; color: #e0e0e0; font-family: 'Microsoft YaHei', sans-serif; margin: 20px; }}
h1 {{ color: #ff6b6b; text-align: center; border-bottom: 2px solid #ff6b6b; padding-bottom: 10px; }}
h2 {{ color: #ffd700; margin-top: 30px; border-left: 4px solid #ffd700; padding-left: 12px; }}
.summary {{ background: #16213e; border-radius: 10px; padding: 15px 20px; margin: 15px 0;
           border: 1px solid #0f3460; font-size: 14px; line-height: 1.8; }}
.summary b {{ color: #00d4ff; }}
.trade-card {{ background: #16213e; border-radius: 10px; padding: 15px; margin: 15px 0;
              border: 1px solid #0f3460; }}
.trade-card img {{ width: 100%; border-radius: 6px; }}
.trade-info {{ display: flex; gap: 15px; flex-wrap: wrap; margin: 10px 0; }}
.info-tag {{ background: #0f3460; padding: 4px 12px; border-radius: 15px; font-size: 13px; }}
.info-tag.win {{ border: 1px solid #4CAF50; color: #4CAF50; }}
.info-tag.loss {{ border: 1px solid #f44336; color: #f44336; }}
.info-tag.filtered {{ border: 1px solid #ff6b6b; color: #ff6b6b; background: #2a1a1a; }}
.legend {{ display: flex; gap: 15px; flex-wrap: wrap; margin: 15px 0; justify-content: center; }}
.legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 12px; }}
.legend-dot {{ width: 12px; height: 12px; border-radius: 50%; }}
.stats-table {{ border-collapse: collapse; margin: 15px auto; font-size: 13px; }}
.stats-table th {{ background: #0f3460; padding: 8px 14px; text-align: center; }}
.stats-table td {{ padding: 6px 14px; text-align: center; border-bottom: 1px solid #1a1a3e; }}
.win {{ color: #4CAF50; font-weight: bold; }}
.loss {{ color: #f44336; font-weight: bold; }}
</style></head><body>
<h1>被 MA20>MA120 过滤掉的信号</h1>
<div class="summary">
    <b>含义:</b> 这些信号在 EMA10>EMA20 下会触发入场，但 EMA20>EMA120 不认同趋势方向，所以被过滤掉。<br>
    <b>目的:</b> 验证过滤是否合理 —— 这些信号应该大部分是不该做的。<br><br>
    <b>被过滤信号总计:</b> {n_total}笔 (反向: {n_reverse}, 无趋势: {n_none})<br>
    <b>被过滤信号期望:</b> <span class="{'win' if ev > 0 else 'loss'}">{ev:+.2f}</span>
    (胜率={wr:.1f}%, 盈亏比={ratio:.2f})<br>
    <b>对比:</b> MA10>MA20保留的全部信号期望=+0.13, MA20>MA120保留的全部信号期望=+0.16<br>
    {'<b style="color:#4CAF50">结论: 过滤掉的信号期望更低，过滤有效！</b>' if ev < 0.16 else '<b style="color:#f44336">注意: 过滤掉的信号期望不低</b>'}
</div>
<div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#FF9800"></div>EMA10 (入场MA，虚线)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#2196F3"></div>EMA20</div>
    <div class="legend-item"><div class="legend-dot" style="background:#9C27B0"></div>EMA120</div>
    <div class="legend-item"><div class="legend-dot" style="background:#4CAF50"></div>止盈线</div>
    <div class="legend-item"><div class="legend-dot" style="background:#f44336"></div>止损线</div>
</div>
"""]

    # 统计表
    html_parts.append("""<table class="stats-table">
<tr><th>分类</th><th>信号数</th><th>胜率</th><th>盈亏比</th><th>期望</th></tr>""")
    for label, subset in [('全部被过滤', all_filtered),
                           ('A类被过滤', [s for s in all_filtered if s['type'] == 'A']),
                           ('B类被过滤', [s for s in all_filtered if s['type'] == 'B']),
                           ('因反向被过滤', [s for s in all_filtered if '反向' in s['filter_reason']]),
                           ('因无趋势被过滤', [s for s in all_filtered if '无趋势' in s['filter_reason']])]:
        sw = [s for s in subset if s['exit_info'][2] > 0]
        sl = [s for s in subset if s['exit_info'][2] <= 0]
        nn = len(subset)
        if nn == 0:
            continue
        w = len(sw) / nn * 100
        aw = np.mean([s['exit_info'][2] for s in sw]) if sw else 0
        al = abs(np.mean([s['exit_info'][2] for s in sl])) if sl else 0.001
        r = aw / al
        e = w / 100 * r - (1 - w / 100)
        ec = 'win' if e > 0 else 'loss'
        html_parts.append(f'<tr><td><b>{label}</b></td><td>{nn}</td>'
                          f'<td>{w:.1f}%</td><td>{r:.2f}</td>'
                          f'<td class="{ec}">{e:+.2f}</td></tr>')
    html_parts.append('</table>')

    # 画10个案例
    html_parts.append(f'<h2>随机抽样 {len(selected)} 笔被过滤信号</h2>')

    for trade_i, sig in enumerate(selected):
        sym_key = sig['symbol']
        df = sym_data[sym_key]
        exit_info = sig['exit_info']
        pnl = exit_info[2]
        exit_reason = exit_info[3]

        print(f"  [{trade_i+1}] {sig['sym_name']} {sig['type']}类 {sig['direction']} "
              f"| {sig['filter_reason']} | ER={sig['er_20']:.3f} -> {exit_reason} {pnl:+.2f}%")

        img_b64 = draw_trade_chart(df, sig, exit_info, sig['sym_name'])

        dir_str = '做多' if sig['direction'] == 'long' else '做空'
        pnl_class = 'win' if pnl > 0 else 'loss'
        html_parts.append(f'''<div class="trade-card">
<div class="trade-info">
    <span class="info-tag">{sig['sym_name']}</span>
    <span class="info-tag">{sig['type']}类 {dir_str}</span>
    <span class="info-tag filtered">{sig['filter_reason']}</span>
    <span class="info-tag">ER = {sig['er_20']:.3f}</span>
    <span class="info-tag">MA距 = {sig['ma_dist_atr']:.2f} ATR</span>
    <span class="info-tag {pnl_class}">{exit_reason} {pnl:+.2f}%</span>
    <span class="info-tag">{str(sig['time'])[:16]}</span>
</div>
<img src="data:image/png;base64,{img_b64}">
</div>''')

    html_parts.append('</body></html>')

    out_path = os.path.join(OUTPUT_DIR, 'trend_filtered_v1.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))
    print(f"\n输出: {out_path}")


if __name__ == '__main__':
    main()
