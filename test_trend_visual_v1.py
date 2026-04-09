# -*- coding: utf-8 -*-
"""
趋势判断方法 K线可视化 v1
========================
EMA三线排列 vs EMA MA20>MA120，各抽5个案例画K线图。
固定出场：2%止损/5%止盈。
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

TEST_SYMBOLS = ['SHFE_au', 'SHFE_ag', 'GFEX_lc', 'DCE_lh']
TEST_DAYS = 60
STOP_LOSS_PCT = 2.0
TAKE_PROFIT_PCT = 5.0
EXAMPLES_PER_SCENARIO = 5


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
    """计算EMA均线"""
    for p in [10, 20, 60, 120]:
        df[f'ema_{p}'] = df['close'].ewm(span=p, adjust=False).mean()
    # ATR
    df['tr'] = np.maximum(df['high'] - df['low'],
        np.maximum(abs(df['high'] - df['close'].shift(1)),
                   abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
    # ER
    bar_moves = df['close'].diff().abs()
    net = (df['close'] - df['close'].shift(20)).abs()
    bar_sum = bar_moves.rolling(20).sum()
    df['er_20'] = net / bar_sum.replace(0, np.nan)
    return df


# ============================================================
# 趋势判断
# ============================================================

def trend_triple_ema(df, i):
    """EMA三线排列: EMA10 > EMA20 > EMA60"""
    row = df.iloc[i]
    m10, m20, m60 = row['ema_10'], row['ema_20'], row['ema_60']
    if pd.isna(m10) or pd.isna(m20) or pd.isna(m60):
        return 0
    if m10 > m20 > m60:
        return 1
    elif m10 < m20 < m60:
        return -1
    return 0


def trend_ma20_ma120_ema(df, i):
    """EMA MA20>MA120"""
    row = df.iloc[i]
    m20, m120 = row['ema_20'], row['ema_120']
    if pd.isna(m20) or pd.isna(m120):
        return 0
    return 1 if m20 > m120 else (-1 if m20 < m120 else 0)


TREND_METHODS = [
    ('EMA三线排列', trend_triple_ema, ['ema_10', 'ema_20', 'ema_60']),
    ('EMA MA20>MA120', trend_ma20_ma120_ema, ['ema_20', 'ema_120']),
]


def detect_signals_with_trend(df, trend_fn):
    """检测A+B类信号，入场基于EMA10回调"""
    signals = []
    n = len(df)
    warmup = 130

    b_below_start = -1
    b_pullback_low = None
    b_pullback_high = None
    prev_trend = 0

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

        trend = trend_fn(df, i)

        if trend != prev_trend and trend != 0:
            b_below_start = -1
            b_pullback_low = None
            b_pullback_high = None
        prev_trend = trend
        if trend == 0:
            continue

        if i < 1:
            continue
        prev = df.iloc[i - 1]
        prev_close = prev['close']
        prev_ma_f = prev['ema_10']
        if pd.isna(prev_ma_f):
            continue

        found = []

        if trend == 1:
            if low <= ma_f and close > ma_f and prev_close > prev_ma_f:
                found.append(('A', 'long'))
            if b_below_start == -1:
                if close < ma_f and prev_close >= prev_ma_f:
                    b_below_start = i
                    b_pullback_low = low
            else:
                b_pullback_low = min(b_pullback_low, low)
                if close > ma_f:
                    pb_bars = i - b_below_start
                    if pb_bars >= 4:
                        found.append(('B', 'long'))
                    b_below_start = -1
                    b_pullback_low = None

        elif trend == -1:
            if high >= ma_f and close < ma_f and prev_close < prev_ma_f:
                found.append(('A', 'short'))
            if b_below_start == -1:
                if close > ma_f and prev_close <= prev_ma_f:
                    b_below_start = i
                    b_pullback_high = high
            else:
                b_pullback_high = max(b_pullback_high, high)
                if close < ma_f:
                    pb_bars = i - b_below_start
                    if pb_bars >= 4:
                        found.append(('B', 'short'))
                    b_below_start = -1
                    b_pullback_high = None

        for sig_type, direction in found:
            ma_dist_atr = abs(close - ma_f) / atr if (not pd.isna(atr) and atr > 0) else 0
            signals.append({
                'type': sig_type,
                'direction': direction,
                'idx': i,
                'time': row['datetime'],
                'entry_price': close,
                'er_20': er if not pd.isna(er) else 0,
                'atr': atr if not pd.isna(atr) else 0,
                'ma_dist_atr': round(ma_dist_atr, 3),
            })

    return signals


def simulate_fixed_exit(df, sig):
    """固定2%止损/5%止盈，返回exit_idx, exit_price, pnl"""
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


# MA颜色
MA_COLORS = {
    'ema_10': ('#FF9800', 'EMA10'),
    'ema_20': ('#2196F3', 'EMA20'),
    'ema_60': ('#E91E63', 'EMA60'),
    'ema_120': ('#9C27B0', 'EMA120'),
}


def draw_trade_chart(df, sig, exit_info, sym_name, method_name, trend_mas):
    """画单笔交易K线图，返回base64 PNG"""
    entry_idx = sig['idx']
    is_long = sig['direction'] == 'long'
    exit_idx, exit_price, pnl, exit_reason = exit_info

    start = max(0, entry_idx - 20)
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

    # 画所有趋势相关MA + EMA10（入场用）
    xs = np.arange(len(segment))
    all_mas = list(set(['ema_10'] + trend_mas))
    for ma_col in sorted(all_mas):
        if ma_col in segment.columns:
            vals = segment[ma_col].values
            valid = ~np.isnan(vals)
            clr, label = MA_COLORS.get(ma_col, ('#888', ma_col))
            lw = 1.8 if ma_col in trend_mas else 1.0
            style = '-' if ma_col in trend_mas else '--'
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
    ax.text(len(segment) - 1, sl_price, f' SL {sl_price:.1f}', fontsize=7,
            color='#f44336', va='center')
    ax.text(len(segment) - 1, tp_price, f' TP {tp_price:.1f}', fontsize=7,
            color='#4CAF50', va='center')

    # X轴
    tick_step = max(1, len(segment) // 10)
    tick_positions = list(range(0, len(segment), tick_step))
    tick_labels = []
    for tp in tick_positions:
        dt = segment.iloc[tp]['datetime']
        tick_labels.append(dt.strftime('%m/%d %H:%M') if hasattr(dt, 'strftime') else str(dt)[:16])
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30, fontsize=7)

    dir_str = '做多' if is_long else '做空'
    ax.set_title(
        f"{sym_name} 10min | {method_name} | {sig['type']}类{dir_str} | "
        f"ER={sig['er_20']:.3f} | MA距={sig['ma_dist_atr']:.2f}ATR | "
        f"{str(sig['time'])[:16]}",
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
    random.seed(123)

    # 加载数据
    print("加载数据...")
    sym_data = {}
    for sym_key in TEST_SYMBOLS:
        df = load_cached(sym_key)
        if df is None:
            print(f"  警告: {sym_key} 无缓存")
            continue
        df = prepare_data(df)
        cutoff = df['datetime'].iloc[-1] - timedelta(days=TEST_DAYS)
        df_test = df[df['datetime'] >= cutoff].copy().reset_index(drop=True)
        if len(df_test) < 200:
            print(f"  警告: {sym_key} 数据不足")
            continue
        sym_data[sym_key] = df_test
        print(f"  {get_sym_name(sym_key)}: {len(df_test)}根K线")

    # HTML头部
    html_parts = ["""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>趋势判断方法 K线验证 v1</title>
<style>
body { background: #1a1a2e; color: #e0e0e0; font-family: 'Microsoft YaHei', sans-serif; margin: 20px; }
h1 { color: #00d4ff; text-align: center; border-bottom: 2px solid #00d4ff; padding-bottom: 10px; }
h2 { color: #ffd700; margin-top: 40px; border-left: 4px solid #ffd700; padding-left: 12px; }
.summary { background: #16213e; border-radius: 10px; padding: 15px 20px; margin: 15px 0;
           border: 1px solid #0f3460; font-size: 14px; line-height: 1.8; }
.summary b { color: #00d4ff; }
.scenario { margin-bottom: 50px; }
.trade-card { background: #16213e; border-radius: 10px; padding: 15px; margin: 15px 0;
              border: 1px solid #0f3460; }
.trade-card img { width: 100%; border-radius: 6px; }
.trade-info { display: flex; gap: 15px; flex-wrap: wrap; margin: 10px 0; }
.info-tag { background: #0f3460; padding: 4px 12px; border-radius: 15px; font-size: 13px; }
.info-tag.win { border: 1px solid #4CAF50; color: #4CAF50; }
.info-tag.loss { border: 1px solid #f44336; color: #f44336; }
.legend { display: flex; gap: 15px; flex-wrap: wrap; margin: 15px 0; justify-content: center; }
.legend-item { display: flex; align-items: center; gap: 5px; font-size: 12px; }
.legend-dot { width: 12px; height: 12px; border-radius: 50%; }
.stats-table { border-collapse: collapse; margin: 15px auto; font-size: 13px; }
.stats-table th { background: #0f3460; padding: 8px 14px; text-align: center; }
.stats-table td { padding: 6px 14px; text-align: center; border-bottom: 1px solid #1a1a3e; }
.win { color: #4CAF50; font-weight: bold; }
.loss { color: #f44336; font-weight: bold; }
</style></head><body>
<h1>趋势判断方法 K线验证 v1</h1>
<div class="summary">
    <b>测试品种:</b> 黄金、白银、碳酸锂、生猪 (4品种 × 60天)<br>
    <b>入场信号:</b> A+B类 EMA10回调<br>
    <b>出场:</b> 固定 2%止损 / 5%止盈<br>
    <b>对比方案:</b> EMA三线排列 (EMA10>EMA20>EMA60) vs EMA MA20>MA120
</div>
<div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#FF9800"></div>EMA10 (入场MA，虚线)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#2196F3"></div>EMA20</div>
    <div class="legend-item"><div class="legend-dot" style="background:#E91E63"></div>EMA60</div>
    <div class="legend-item"><div class="legend-dot" style="background:#9C27B0"></div>EMA120</div>
    <div class="legend-item"><div class="legend-dot" style="background:#4CAF50"></div>止盈线</div>
    <div class="legend-item"><div class="legend-dot" style="background:#f44336"></div>止损线</div>
</div>
"""]

    for method_name, trend_fn, trend_mas in TREND_METHODS:
        # 检测所有信号
        all_sigs = []
        for sym_key, df in sym_data.items():
            sigs = detect_signals_with_trend(df, trend_fn)
            for s in sigs:
                exit_info = simulate_fixed_exit(df, s)
                s['symbol'] = sym_key
                s['sym_name'] = get_sym_name(sym_key)
                s['exit_info'] = exit_info
            all_sigs.extend(sigs)

        wins = [s for s in all_sigs if s['exit_info'][2] > 0]
        losses = [s for s in all_sigs if s['exit_info'][2] <= 0]
        a_sigs = [s for s in all_sigs if s['type'] == 'A']
        b_sigs = [s for s in all_sigs if s['type'] == 'B']

        n_total = len(all_sigs)
        wr = len(wins) / n_total * 100 if n_total > 0 else 0
        avg_w = np.mean([s['exit_info'][2] for s in wins]) if wins else 0
        avg_l = abs(np.mean([s['exit_info'][2] for s in losses])) if losses else 0.001
        ratio = avg_w / avg_l
        ev = wr / 100 * ratio - (1 - wr / 100)

        print(f"\n{'='*60}")
        print(f"  {method_name}: {n_total}笔 | 胜率={wr:.1f}% 盈亏比={ratio:.2f} 期望={ev:+.2f}")
        print(f"  A类={len(a_sigs)} B类={len(b_sigs)} 盈={len(wins)} 亏={len(losses)}")

        # 统计表
        html_parts.append(f'<div class="scenario">')
        html_parts.append(f'<h2>{method_name}</h2>')
        html_parts.append(f'''<table class="stats-table">
<tr><th>指标</th><th>全部</th><th>A类</th><th>B类</th></tr>''')

        for label, subset in [('全部', all_sigs), ('A类', a_sigs), ('B类', b_sigs)]:
            pass  # will build per-row

        for row_label, subset in [('全部', all_sigs), ('A类', a_sigs), ('B类', b_sigs)]:
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

        # 重新组织：按列输出
        stats = {}
        for row_label, subset in [('全部', all_sigs), ('A类', a_sigs), ('B类', b_sigs)]:
            sw = [s for s in subset if s['exit_info'][2] > 0]
            sl = [s for s in subset if s['exit_info'][2] <= 0]
            nn = len(subset)
            if nn == 0:
                stats[row_label] = {'n': 0, 'wr': 0, 'ratio': 0, 'ev': 0}
                continue
            w = len(sw) / nn * 100
            aw = np.mean([s['exit_info'][2] for s in sw]) if sw else 0
            al = abs(np.mean([s['exit_info'][2] for s in sl])) if sl else 0.001
            r = aw / al
            e = w / 100 * r - (1 - w / 100)
            stats[row_label] = {'n': nn, 'wr': w, 'ratio': r, 'ev': e}

        # 清除之前错误的表格，重建
        html_parts[-1] = f'''<table class="stats-table">
<tr><th></th><th>信号数</th><th>胜率</th><th>盈亏比</th><th>期望</th></tr>'''
        for rl in ['全部', 'A类', 'B类']:
            st = stats[rl]
            ec = 'win' if st['ev'] > 0 else 'loss'
            html_parts.append(
                f'<tr><td><b>{rl}</b></td><td>{st["n"]}</td>'
                f'<td>{st["wr"]:.1f}%</td><td>{st["ratio"]:.2f}</td>'
                f'<td class="{ec}">{st["ev"]:+.2f}</td></tr>')
        html_parts.append('</table>')

        # 选5个案例：保证盈亏混合、不同品种、A+B都有
        valid_sigs = [s for s in all_sigs if s['idx'] + 15 < len(sym_data[s['symbol']])]
        win_sigs = [s for s in valid_sigs if s['exit_info'][2] > 0]
        loss_sigs = [s for s in valid_sigs if s['exit_info'][2] <= 0]
        b_sigs_pool = [s for s in valid_sigs if s['type'] == 'B']

        selected = []
        used_syms = set()

        # 先保证有B类（如果有的话）
        random.shuffle(b_sigs_pool)
        for s in b_sigs_pool:
            if len(selected) >= 2:
                break
            if s['symbol'] not in used_syms:
                selected.append(s)
                used_syms.add(s['symbol'])

        # 再补盈利案例（不同品种）
        random.shuffle(win_sigs)
        for s in win_sigs:
            if len(selected) >= 4:
                break
            if s not in selected and s['symbol'] not in used_syms:
                selected.append(s)
                used_syms.add(s['symbol'])

        # 再补亏损案例
        random.shuffle(loss_sigs)
        for s in loss_sigs:
            if len(selected) >= EXAMPLES_PER_SCENARIO:
                break
            if s not in selected:
                selected.append(s)

        # 如果还不够，放宽条件
        random.shuffle(valid_sigs)
        for s in valid_sigs:
            if len(selected) >= EXAMPLES_PER_SCENARIO:
                break
            if s not in selected:
                selected.append(s)

        for trade_i, sig in enumerate(selected):
            sym_key = sig['symbol']
            df = sym_data[sym_key]
            exit_idx, exit_price, pnl, exit_reason = sig['exit_info']

            print(f"  [{trade_i+1}] {sig['sym_name']} {sig['type']}类 "
                  f"{sig['direction']} ER={sig['er_20']:.3f} -> {exit_reason} {pnl:+.2f}%")

            img_b64 = draw_trade_chart(df, sig, sig['exit_info'], sig['sym_name'],
                                       method_name, trend_mas)

            dir_str = '做多' if sig['direction'] == 'long' else '做空'
            pnl_class = 'win' if pnl > 0 else 'loss'
            html_parts.append(f'''<div class="trade-card">
<div class="trade-info">
    <span class="info-tag">{sig['sym_name']}</span>
    <span class="info-tag">{sig['type']}类 {dir_str}</span>
    <span class="info-tag">ER = {sig['er_20']:.3f}</span>
    <span class="info-tag">MA距 = {sig['ma_dist_atr']:.2f} ATR</span>
    <span class="info-tag {pnl_class}">{exit_reason} {pnl:+.2f}%</span>
    <span class="info-tag">{str(sig['time'])[:16]}</span>
    <span class="info-tag">入场价 {sig['entry_price']:.1f}</span>
</div>
<img src="data:image/png;base64,{img_b64}">
</div>''')

        html_parts.append('</div>')

    html_parts.append('</body></html>')

    out_path = os.path.join(OUTPUT_DIR, 'trend_visual_v1.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))
    print(f"\n输出: {out_path}")


if __name__ == '__main__':
    main()
