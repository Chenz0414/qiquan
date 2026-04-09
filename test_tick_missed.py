# -*- coding: utf-8 -*-
"""
被1跳止损扫出但5跳不会被扫、且后续创新高的案例分析
高波动组10品种，最近30天
输出：统计概率 + HTML K线图
"""

import os
import numpy as np
import pandas as pd
from signal_core import SYMBOL_CONFIGS

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HIGH_VOL_SYMBOLS = [
    ("GFEX_lc", "GFEX.lc", "碳酸锂"),
    ("DCE_jm",  "DCE.jm",  "焦煤"),
    ("SHFE_ag", "SHFE.ag",  "白银"),
    ("CZCE_FG", "CZCE.FG",  "玻璃"),
    ("CZCE_SA", "CZCE.SA",  "纯碱"),
    ("INE_sc",  "INE.sc",   "原油"),
    ("CZCE_MA", "CZCE.MA",  "甲醇"),
    ("CZCE_TA", "CZCE.TA",  "PTA"),
    ("DCE_eb",  "DCE.eb",   "苯乙烯"),
    ("DCE_lh",  "DCE.lh",   "生猪"),
]

MIN_PB_BARS = 4
LAST_DAYS = 30
BARS_PER_DAY = 57
LOOK_AHEAD = 30  # 止损后看30根K线是否创新高


def load_data(cache_key):
    path = os.path.join(CACHE_DIR, f"{cache_key}_10min_170d.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema120'] = df['close'].ewm(span=120, adjust=False).mean()
    return df


def detect_signals(df, start_idx):
    signals = []
    n = len(df)
    trend_dir = 0
    signal_count = 0
    below_start = -1
    pb_low = pb_high = None
    prev_close = prev_ema10 = None

    for i in range(1, n):
        row = df.iloc[i]
        close, high, low = row['close'], row['high'], row['low']
        ema10, ema20, ema120 = row['ema10'], row['ema20'], row['ema120']

        if prev_close is None or pd.isna(ema120):
            prev_close, prev_ema10 = close, ema10
            continue

        curr_trend = 1 if ema20 > ema120 else (-1 if ema20 < ema120 else 0)
        if curr_trend != trend_dir and curr_trend != 0:
            trend_dir, signal_count = curr_trend, 0
            below_start, pb_low, pb_high = -1, None, None

        if trend_dir == 0:
            prev_close, prev_ema10 = close, ema10
            continue

        if trend_dir == 1:
            if below_start == -1:
                if close < ema10 and prev_close >= prev_ema10:
                    below_start, pb_low = i, low
            else:
                pb_low = min(pb_low, low)
                if close > ema10:
                    pb_bars = i - below_start
                    if pb_bars >= MIN_PB_BARS and i >= start_idx:
                        signal_count += 1
                        signals.append({'idx': i, 'direction': 'long', 'entry_price': close,
                                        'pullback_extreme': pb_low, 'pullback_bars': pb_bars,
                                        'time': row['datetime']})
                    elif pb_bars >= MIN_PB_BARS:
                        signal_count += 1
                    below_start, pb_low = -1, None
        elif trend_dir == -1:
            if below_start == -1:
                if close > ema10 and prev_close <= prev_ema10:
                    below_start, pb_high = i, high
            else:
                pb_high = max(pb_high, high)
                if close < ema10:
                    pb_bars = i - below_start
                    if pb_bars >= MIN_PB_BARS and i >= start_idx:
                        signal_count += 1
                        signals.append({'idx': i, 'direction': 'short', 'entry_price': close,
                                        'pullback_extreme': pb_high, 'pullback_bars': pb_bars,
                                        'time': row['datetime']})
                    elif pb_bars >= MIN_PB_BARS:
                        signal_count += 1
                    below_start, pb_high = -1, None

        prev_close, prev_ema10 = close, ema10
    return signals


def simulate_compare(df, signals, tick_size):
    """对比1跳和5跳，找出被1跳扫出但5跳不会的案例"""
    cases = []
    n = len(df)

    for sig in signals:
        entry_idx = sig['idx']
        entry_price = sig['entry_price']
        is_long = sig['direction'] == 'long'
        pb_ext = sig['pullback_extreme']

        tick1 = tick_size * 1
        tick5 = tick_size * 5

        if is_long:
            stop1 = pb_ext - tick1
            stop5 = pb_ext - tick5
        else:
            stop1 = pb_ext + tick1
            stop5 = pb_ext + tick5

        stop1_triggered = False
        stop1_bar = None
        stop5_triggered_before_stop1 = False

        # 模拟逐根
        s1_stop = stop1
        s5_stop = stop5
        prev_close = entry_price

        for j in range(entry_idx + 1, min(n, entry_idx + 200)):
            bar = df.iloc[j]
            prev_bar = df.iloc[j - 1]

            # 5跳先触发了 → 不是我们要找的case
            if is_long and bar['low'] <= s5_stop:
                stop5_triggered_before_stop1 = True
                break
            elif not is_long and bar['high'] >= s5_stop:
                stop5_triggered_before_stop1 = True
                break

            # 1跳触发
            if is_long and bar['low'] <= s1_stop:
                stop1_triggered = True
                stop1_bar = j
                break
            elif not is_long and bar['high'] >= s1_stop:
                stop1_triggered = True
                stop1_bar = j
                break

            # 追踪更新（两个止损各自更新）
            if is_long and bar['close'] > prev_bar['close']:
                s1_stop = max(s1_stop, bar['low'] - tick1)
                s5_stop = max(s5_stop, bar['low'] - tick5)
            elif not is_long and bar['close'] < prev_bar['close']:
                s1_stop = min(s1_stop, bar['high'] + tick1)
                s5_stop = min(s5_stop, bar['high'] + tick5)

        if not stop1_triggered or stop5_triggered_before_stop1:
            continue

        # 1跳被扫了，检查5跳在同一根是否也被扫
        bar_at_stop1 = df.iloc[stop1_bar]
        if is_long and bar_at_stop1['low'] <= s5_stop:
            continue  # 同一根也扫了5跳
        if not is_long and bar_at_stop1['high'] >= s5_stop:
            continue

        # 好的，1跳被扫但5跳没有。看后续是否创新高
        # 继续模拟5跳的后续
        made_new_high = False
        max_after = 0
        s5_final_pnl = None
        s5_exit_bar = None

        for j in range(stop1_bar + 1, min(n, stop1_bar + LOOK_AHEAD)):
            bar = df.iloc[j]
            prev_bar = df.iloc[j - 1]

            if is_long:
                move = (bar['high'] - entry_price) / entry_price * 100
            else:
                move = (entry_price - bar['low']) / entry_price * 100
            if move > max_after:
                max_after = move

            # 检查5跳是否也被扫了
            if is_long and bar['low'] <= s5_stop:
                s5_final_pnl = (s5_stop - entry_price) / entry_price * 100
                s5_exit_bar = j
                break
            elif not is_long and bar['high'] >= s5_stop:
                s5_final_pnl = (entry_price - s5_stop) / entry_price * 100
                s5_exit_bar = j
                break

            # 继续追踪
            if is_long and bar['close'] > prev_bar['close']:
                s5_stop = max(s5_stop, bar['low'] - tick5)
            elif not is_long and bar['close'] < prev_bar['close']:
                s5_stop = min(s5_stop, bar['high'] + tick5)

        # 计算1跳的pnl
        if is_long:
            pnl_1tick = (s1_stop - entry_price) / entry_price * 100
        else:
            pnl_1tick = (entry_price - s1_stop) / entry_price * 100

        if s5_final_pnl is None:
            # 5跳在look_ahead内没被扫
            last_j = min(n - 1, stop1_bar + LOOK_AHEAD - 1)
            last_close = df.iloc[last_j]['close']
            if is_long:
                s5_final_pnl = (last_close - entry_price) / entry_price * 100
            else:
                s5_final_pnl = (entry_price - last_close) / entry_price * 100
            s5_exit_bar = last_j

        if max_after > abs(pnl_1tick):
            made_new_high = True

        cases.append({
            'entry_idx': entry_idx,
            'stop1_bar': stop1_bar,
            's5_exit_bar': s5_exit_bar,
            'direction': sig['direction'],
            'entry_price': entry_price,
            'entry_time': sig['time'],
            'pnl_1tick': round(pnl_1tick, 4),
            'pnl_5tick': round(s5_final_pnl, 4),
            'max_after_stop1': round(max_after, 4),
            'made_new_high': made_new_high,
            'pullback_bars': sig['pullback_bars'],
        })

    return cases


def generate_html(all_cases, all_dfs):
    """生成K线图HTML"""
    # 选最具代表性的案例：1跳亏损但5跳盈利且创新高
    best_cases = [c for c in all_cases if c['made_new_high'] and c['pnl_5tick'] > 0]
    best_cases.sort(key=lambda x: x['pnl_5tick'] - x['pnl_1tick'], reverse=True)
    show_cases = best_cases[:8]  # 最多展示8个

    if not show_cases:
        show_cases = [c for c in all_cases if c['made_new_high']][:8]

    html_parts = ["""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>1跳被扫 vs 5跳存活 案例分析</title>
<style>
body { font-family: 'Microsoft YaHei', sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; }
h1 { color: #00d4ff; text-align: center; }
.case { background: #16213e; border-radius: 10px; padding: 20px; margin: 20px 0; }
.case h3 { color: #ffa500; }
.stats { display: flex; gap: 30px; margin: 10px 0; flex-wrap: wrap; }
.stat { background: #0f3460; padding: 8px 16px; border-radius: 5px; }
.stat .label { color: #888; font-size: 12px; }
.stat .value { font-size: 18px; font-weight: bold; }
.win { color: #00ff88; }
.lose { color: #ff4444; }
canvas { width: 100%; height: 400px; background: #0a0a1a; border-radius: 5px; margin-top: 10px; }
.summary { background: #1e3a5f; border-radius: 10px; padding: 20px; margin: 20px 0; }
</style></head><body>
<h1>1跳止损被扫 vs 5跳存活 案例对比</h1>
"""]

    # 汇总统计
    total = len(all_cases)
    new_high_count = sum(1 for c in all_cases if c['made_new_high'])
    profit_with_5 = sum(1 for c in all_cases if c['pnl_5tick'] > 0)
    avg_loss_1 = np.mean([c['pnl_1tick'] for c in all_cases]) if all_cases else 0
    avg_pnl_5 = np.mean([c['pnl_5tick'] for c in all_cases]) if all_cases else 0
    avg_max_after = np.mean([c['max_after_stop1'] for c in all_cases]) if all_cases else 0

    html_parts.append(f"""
<div class="summary">
<h2>汇总统计</h2>
<div class="stats">
<div class="stat"><div class="label">1跳被扫但5跳存活的案例</div><div class="value">{total}笔</div></div>
<div class="stat"><div class="label">其中后续创新高</div><div class="value {'win' if new_high_count/max(total,1)>0.5 else 'lose'}">{new_high_count}笔 ({new_high_count/max(total,1)*100:.1f}%)</div></div>
<div class="stat"><div class="label">5跳最终盈利</div><div class="value {'win' if profit_with_5/max(total,1)>0.5 else ''}">{profit_with_5}笔 ({profit_with_5/max(total,1)*100:.1f}%)</div></div>
<div class="stat"><div class="label">1跳平均PnL</div><div class="value lose">{avg_loss_1:+.4f}%</div></div>
<div class="stat"><div class="label">5跳平均PnL</div><div class="value {'win' if avg_pnl_5>0 else 'lose'}">{avg_pnl_5:+.4f}%</div></div>
<div class="stat"><div class="label">被扫后平均最大有利</div><div class="value win">+{avg_max_after:.4f}%</div></div>
</div></div>
""")

    for ci, case in enumerate(show_cases):
        sym_name = case['symbol']
        df = all_dfs[case['df_key']]
        entry_idx = case['entry_idx']
        stop1_bar = case['stop1_bar']

        # 取K线范围：入场前20根 ~ 止损后30根
        chart_start = max(0, entry_idx - 20)
        chart_end = min(len(df), stop1_bar + LOOK_AHEAD + 5)
        chart_df = df.iloc[chart_start:chart_end].copy()

        # 准备数据
        ohlc_data = []
        ema10_data = []
        ema20_data = []
        for _, r in chart_df.iterrows():
            ts = str(r['datetime'])[:16]
            ohlc_data.append(f'["{ts}",{r["open"]},{r["high"]},{r["low"]},{r["close"]}]')
            ema10_data.append(f'{r["ema10"]:.4f}')
            ema20_data.append(f'{r["ema20"]:.4f}')

        entry_pos = entry_idx - chart_start
        stop1_pos = stop1_bar - chart_start
        direction_label = "做多" if case['direction'] == 'long' else "做空"

        html_parts.append(f"""
<div class="case">
<h3>案例{ci+1}: {sym_name} | {direction_label} | {str(case['entry_time'])[:16]}</h3>
<div class="stats">
<div class="stat"><div class="label">入场价</div><div class="value">{case['entry_price']}</div></div>
<div class="stat"><div class="label">1跳PnL</div><div class="value lose">{case['pnl_1tick']:+.4f}%</div></div>
<div class="stat"><div class="label">5跳PnL</div><div class="value {'win' if case['pnl_5tick']>0 else 'lose'}">{case['pnl_5tick']:+.4f}%</div></div>
<div class="stat"><div class="label">被扫后最大有利</div><div class="value win">+{case['max_after_stop1']:.4f}%</div></div>
<div class="stat"><div class="label">回调根数</div><div class="value">{case['pullback_bars']}</div></div>
</div>
<canvas id="chart{ci}"></canvas>
<script>
(function() {{
var canvas = document.getElementById('chart{ci}');
var ctx = canvas.getContext('2d');
canvas.width = canvas.offsetWidth * 2;
canvas.height = 800;
var W = canvas.width, H = canvas.height;
var data = [{','.join(ohlc_data)}];
var ema10 = [{','.join(ema10_data)}];
var ema20 = [{','.join(ema20_data)}];
var n = data.length;
var pad = {{top:30, bottom:60, left:80, right:30}};
var cw = (W - pad.left - pad.right) / n;

// 价格范围
var allH = data.map(d=>d[2]), allL = data.map(d=>d[3]);
var pMax = Math.max(...allH)*1.001, pMin = Math.min(...allL)*0.999;
var pRange = pMax - pMin;
function y(p) {{ return pad.top + (pMax - p) / pRange * (H - pad.top - pad.bottom); }}
function x(i) {{ return pad.left + i * cw + cw/2; }}

// 背景
ctx.fillStyle = '#0a0a1a';
ctx.fillRect(0, 0, W, H);

// 网格
ctx.strokeStyle = '#1a2a3a';
ctx.lineWidth = 0.5;
for (var g = 0; g < 6; g++) {{
    var gy = pad.top + g * (H - pad.top - pad.bottom) / 5;
    ctx.beginPath(); ctx.moveTo(pad.left, gy); ctx.lineTo(W-pad.right, gy); ctx.stroke();
    var gp = pMax - g * pRange / 5;
    ctx.fillStyle = '#666';
    ctx.font = '20px monospace';
    ctx.fillText(gp.toFixed(2), 5, gy + 6);
}}

// K线
for (var i = 0; i < n; i++) {{
    var o=data[i][1], h=data[i][2], l=data[i][3], c=data[i][4];
    var bull = c >= o;
    ctx.strokeStyle = bull ? '#00ff88' : '#ff4444';
    ctx.fillStyle = bull ? '#00ff88' : '#ff4444';
    // 影线
    ctx.beginPath(); ctx.moveTo(x(i), y(h)); ctx.lineTo(x(i), y(l)); ctx.lineWidth=1; ctx.stroke();
    // 实体
    var bw = cw * 0.6;
    var top = bull ? y(c) : y(o);
    var bot = bull ? y(o) : y(c);
    var bh = Math.max(bot - top, 1);
    if (bull) {{ ctx.strokeRect(x(i)-bw/2, top, bw, bh); }}
    else {{ ctx.fillRect(x(i)-bw/2, top, bw, bh); }}
}}

// EMA10
ctx.strokeStyle = '#ffaa00';
ctx.lineWidth = 2;
ctx.beginPath();
for (var i = 0; i < n; i++) {{
    if (i===0) ctx.moveTo(x(i), y(ema10[i]));
    else ctx.lineTo(x(i), y(ema10[i]));
}}
ctx.stroke();

// EMA20
ctx.strokeStyle = '#00aaff';
ctx.lineWidth = 2;
ctx.beginPath();
for (var i = 0; i < n; i++) {{
    if (i===0) ctx.moveTo(x(i), y(ema20[i]));
    else ctx.lineTo(x(i), y(ema20[i]));
}}
ctx.stroke();

// 入场标记
ctx.fillStyle = '#00ff88';
ctx.beginPath();
ctx.moveTo(x({entry_pos}), y(data[{entry_pos}][3])-5);
ctx.lineTo(x({entry_pos})-10, y(data[{entry_pos}][3])-25);
ctx.lineTo(x({entry_pos})+10, y(data[{entry_pos}][3])-25);
ctx.fill();
ctx.fillStyle = '#fff';
ctx.font = 'bold 20px sans-serif';
ctx.fillText('入场', x({entry_pos})-20, y(data[{entry_pos}][3])-30);

// 1跳止损标记
ctx.fillStyle = '#ff4444';
ctx.font = 'bold 22px sans-serif';
ctx.fillText('1跳止损', x({stop1_pos})-30, y(data[{stop1_pos}][2])+40);
ctx.beginPath();
ctx.arc(x({stop1_pos}), y(data[{stop1_pos}][3]), 8, 0, Math.PI*2);
ctx.fill();

// X轴时间标签
ctx.fillStyle = '#666';
ctx.font = '16px monospace';
var step = Math.max(1, Math.floor(n / 8));
for (var i = 0; i < n; i += step) {{
    ctx.save();
    ctx.translate(x(i), H - 5);
    ctx.rotate(-0.5);
    ctx.fillText(data[i][0].substring(5), 0, 0);
    ctx.restore();
}}

// 图例
ctx.font = '18px sans-serif';
ctx.fillStyle='#ffaa00'; ctx.fillText('— EMA10', W-250, 25);
ctx.fillStyle='#00aaff'; ctx.fillText('— EMA20', W-150, 25);
}})();
</script>
</div>
""")

    html_parts.append("</body></html>")
    html_path = os.path.join(OUTPUT_DIR, "tick_missed_cases.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))
    print(f"\nHTML已生成: {html_path}")
    return html_path


# ==================== 主流程 ====================

print("=" * 70)
print("1跳被扫 vs 5跳存活 案例分析 | 高波动组 | 最近30天")
print("=" * 70)

all_cases = []
all_dfs = {}

for cache_key, symbol_key, name in HIGH_VOL_SYMBOLS:
    df = load_data(cache_key)
    if df is None:
        continue

    tick_size = SYMBOL_CONFIGS[symbol_key]['tick_size']
    n = len(df)
    start_idx = max(0, n - LAST_DAYS * BARS_PER_DAY)

    signals = detect_signals(df, start_idx)
    if not signals:
        print(f"{name}: 无信号")
        continue

    cases = simulate_compare(df, signals, tick_size)

    # 附加品种信息
    df_key = cache_key
    all_dfs[df_key] = df
    for c in cases:
        c['symbol'] = name
        c['df_key'] = df_key

    all_cases.extend(cases)

    total_sigs = len(signals)
    n_cases = len(cases)
    n_new_high = sum(1 for c in cases if c['made_new_high'])
    print(f"{name}: {total_sigs}个信号 → {n_cases}个被1跳扫但5跳没扫 "
          f"({n_cases/total_sigs*100:.1f}%) → 其中{n_new_high}个后续创新高")

# 汇总
print(f"\n{'='*70}")
print("汇总:")
total_cases = len(all_cases)
new_high = sum(1 for c in all_cases if c['made_new_high'])
profit_5 = sum(1 for c in all_cases if c['pnl_5tick'] > 0)
print(f"  总案例: {total_cases}")
print(f"  后续创新高: {new_high} ({new_high/max(total_cases,1)*100:.1f}%)")
print(f"  5跳最终盈利: {profit_5} ({profit_5/max(total_cases,1)*100:.1f}%)")
if all_cases:
    avg1 = np.mean([c['pnl_1tick'] for c in all_cases])
    avg5 = np.mean([c['pnl_5tick'] for c in all_cases])
    avg_max = np.mean([c['max_after_stop1'] for c in all_cases])
    print(f"  1跳平均PnL: {avg1:+.4f}%")
    print(f"  5跳平均PnL: {avg5:+.4f}%")
    print(f"  被扫后平均最大有利: +{avg_max:.4f}%")

# 生成HTML
if all_cases:
    generate_html(all_cases, all_dfs)
