# -*- coding: utf-8 -*-
"""
时间稳定性检验 HTML报告
120天分4个30天窗口 | A+B类 | S2出场 | ER(20) 0.5~0.7
包含白银逐窗口K线案例（每窗口最赚2笔+最亏2笔，<=5笔全展示）
"""

import os, json
import numpy as np
import pandas as pd
from signal_core import SYMBOL_CONFIGS

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
LAST_DAYS = 120
BARS_PER_DAY = 57
STOP_TICKS = 5
MIN_PB_BARS_C = 4
N_WINDOWS = 4
WINDOW_DAYS = LAST_DAYS // N_WINDOWS
ER_LO, ER_HI = 0.5, 0.7


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


def simulate_s2_detail(df, signals, tick_size):
    """S2出场，返回详细交易数据（含exit_idx）"""
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
        exit_idx = n - 1

        for j in range(entry_idx + 1, n):
            bar = df.iloc[j]
            if pd.isna(bar['ema10']):
                continue
            close, high, low = bar['close'], bar['high'], bar['low']
            ma_val = bar['ema10']

            if is_long and low <= s2_stop:
                s2_pnl = (s2_stop - entry_price) / entry_price * 100
                s2_done = True
                exit_idx = j
                break
            elif not is_long and high >= s2_stop:
                s2_pnl = (entry_price - s2_stop) / entry_price * 100
                s2_done = True
                exit_idx = j
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
            exit_idx = n - 1

        trades.append({
            'type': sig['type'], 'direction': sig['direction'],
            'er_20': sig['er_20'], 'pnl': round(s2_pnl, 4),
            'entry_idx': entry_idx, 'exit_idx': exit_idx,
            'entry_price': entry_price, 'stop_price': round(s2_stop, 2),
            'pullback_extreme': pb_ext,
        })
    return trades


def calc_ev(pnls):
    if len(pnls) == 0:
        return {'N': 0, 'EV': 0, 'wr': 0, 'sum': 0, 'avg_w': 0, 'avg_l': 0, 'pr': 0}
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0.001
    pr = avg_win / avg_loss if avg_loss > 0 else 0
    EV = wr * pr - (1 - wr)
    return {'N': len(pnls), 'EV': round(EV, 2), 'wr': round(wr * 100, 1),
            'sum': round(sum(pnls), 2), 'avg_w': round(avg_win, 4),
            'avg_l': round(avg_loss, 4), 'pr': round(pr, 2)}


# ==================== main ====================
print("Loading data...")

all_sym_data = {}  # {name: {trades: [...], windows: {0:[],1:[],2:[],3:[]}}}
all_dfs = {}  # 保存白银的df

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

    signals = detect_signals_ab(df, global_start, n)
    trades = simulate_s2_detail(df, signals, tick_size)

    # ER 0.5~0.7 过滤
    filtered = [t for t in trades if not pd.isna(t['er_20']) and ER_LO <= t['er_20'] < ER_HI]

    # 按窗口分组
    windows = {w: [] for w in range(N_WINDOWS)}
    for t in filtered:
        bar_offset = t['entry_idx'] - global_start
        w = min(bar_offset // (WINDOW_DAYS * BARS_PER_DAY), N_WINDOWS - 1)
        t['window'] = w
        windows[w].append(t)

    all_sym_data[name] = {
        'trades': filtered,
        'windows': windows,
        'sym_key': sym_key,
    }

    # 保存白银df
    if name == '白银':
        all_dfs['白银'] = df

# 按120天EV排名
ranked = []
for name, data in all_sym_data.items():
    pnls = [t['pnl'] for t in data['trades']]
    ev = calc_ev(pnls)
    ranked.append({'name': name, **ev})
ranked.sort(key=lambda x: x['EV'], reverse=True)

print(f"Ranked {len(ranked)} symbols")
for i, r in enumerate(ranked[:10]):
    print(f"  {i+1}. {r['name']} EV={r['EV']:+.2f} N={r['N']}")


# ==================== 生成HTML ====================
print("Generating HTML report...")

# 白银K线案例数据准备
ag_df = all_dfs.get('白银')
ag_data = all_sym_data.get('白银')
ag_cases = {}  # {window_idx: [trade_dicts]}

if ag_data and ag_df is not None:
    for w in range(N_WINDOWS):
        w_trades = ag_data['windows'][w]
        if len(w_trades) <= 5:
            # 全部展示
            ag_cases[w] = w_trades
        else:
            # 排序取最赚2+最亏2
            sorted_by_pnl = sorted(w_trades, key=lambda t: t['pnl'])
            worst2 = sorted_by_pnl[:2]
            best2 = sorted_by_pnl[-2:]
            # 按entry_idx排序展示
            ag_cases[w] = sorted(worst2 + best2, key=lambda t: t['entry_idx'])


def make_chart_js(chart_id, df, trade, before=25, after=35):
    """生成单个K线图的JS代码"""
    entry_idx = trade['entry_idx']
    exit_idx = trade['exit_idx']
    is_long = trade['direction'] == 'long'

    chart_start = max(0, entry_idx - before)
    chart_end = min(len(df), max(exit_idx + 10, entry_idx + after))
    chart_df = df.iloc[chart_start:chart_end]

    ohlc = []
    ema10_arr = []
    ema20_arr = []
    for _, r in chart_df.iterrows():
        ts = str(r['datetime'])[:16] if 'datetime' in r.index else ''
        ohlc.append(f'["{ts}",{r["open"]},{r["high"]},{r["low"]},{r["close"]}]')
        ema10_arr.append(f'{r["ema10"]:.2f}')
        ema20_arr.append(f'{r["ema20"]:.2f}')

    entry_pos = entry_idx - chart_start
    exit_pos = min(exit_idx - chart_start, len(chart_df) - 1)
    dir_label = "LONG" if is_long else "SHORT"
    pnl = trade['pnl']
    pnl_color = '#00ff88' if pnl > 0 else '#ff4444'

    return f"""
(function() {{
var canvas = document.getElementById('{chart_id}');
var ctx = canvas.getContext('2d');
canvas.width = canvas.parentElement.offsetWidth * 2;
canvas.height = 700;
var W = canvas.width, H = canvas.height;
var data = [{','.join(ohlc)}];
var ema10 = [{','.join(ema10_arr)}];
var ema20 = [{','.join(ema20_arr)}];
var n = data.length;
var pad = {{top:30, bottom:60, left:90, right:30}};
var cw = (W - pad.left - pad.right) / n;
var allH = data.map(d=>d[2]), allL = data.map(d=>d[3]);
var pMax = Math.max(...allH,...ema10,...ema20)*1.001, pMin = Math.min(...allL,...ema10,...ema20)*0.999;
var pRange = pMax - pMin;
function y(p) {{ return pad.top + (pMax - p) / pRange * (H - pad.top - pad.bottom); }}
function x(i) {{ return pad.left + i * cw + cw/2; }}

ctx.fillStyle = '#0a0a1a'; ctx.fillRect(0, 0, W, H);

// grid
ctx.strokeStyle = '#1a2a3a'; ctx.lineWidth = 0.5;
for (var g = 0; g < 6; g++) {{
    var gy = pad.top + g * (H - pad.top - pad.bottom) / 5;
    ctx.beginPath(); ctx.moveTo(pad.left, gy); ctx.lineTo(W-pad.right, gy); ctx.stroke();
    ctx.fillStyle = '#666'; ctx.font = '20px monospace';
    ctx.fillText((pMax - g * pRange / 5).toFixed(1), 5, gy + 6);
}}

// K lines
for (var i = 0; i < n; i++) {{
    var o=data[i][1], h=data[i][2], l=data[i][3], c=data[i][4];
    var bull = c >= o;
    ctx.strokeStyle = bull ? '#00b36b' : '#e63946';
    ctx.fillStyle = bull ? '#00b36b' : '#e63946';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x(i), y(h)); ctx.lineTo(x(i), y(l)); ctx.stroke();
    var bw = Math.max(cw * 0.6, 2);
    var top = bull ? y(c) : y(o), bot = bull ? y(o) : y(c);
    var bh = Math.max(bot - top, 1);
    if (bull) ctx.strokeRect(x(i)-bw/2, top, bw, bh);
    else ctx.fillRect(x(i)-bw/2, top, bw, bh);
}}

// EMA10
ctx.strokeStyle = '#ffaa00'; ctx.lineWidth = 2; ctx.beginPath();
for (var i = 0; i < n; i++) {{ if(i===0) ctx.moveTo(x(i),y(ema10[i])); else ctx.lineTo(x(i),y(ema10[i])); }}
ctx.stroke();

// EMA20
ctx.strokeStyle = '#00aaff'; ctx.lineWidth = 1.5; ctx.setLineDash([6,3]); ctx.beginPath();
for (var i = 0; i < n; i++) {{ if(i===0) ctx.moveTo(x(i),y(ema20[i])); else ctx.lineTo(x(i),y(ema20[i])); }}
ctx.stroke(); ctx.setLineDash([]);

// entry marker
var ep = {entry_pos};
ctx.fillStyle = '{pnl_color}';
ctx.beginPath();
if ('{trade["direction"]}' === 'long') {{
    ctx.moveTo(x(ep), y(data[ep][3])+5);
    ctx.lineTo(x(ep)-12, y(data[ep][3])+28);
    ctx.lineTo(x(ep)+12, y(data[ep][3])+28);
}} else {{
    ctx.moveTo(x(ep), y(data[ep][2])-5);
    ctx.lineTo(x(ep)-12, y(data[ep][2])-28);
    ctx.lineTo(x(ep)+12, y(data[ep][2])-28);
}}
ctx.fill();
ctx.fillStyle = '#fff'; ctx.font = 'bold 20px sans-serif';
ctx.fillText('{dir_label} {trade["type"]}', x(ep)-30, '{trade["direction"]}' === 'long' ? y(data[ep][3])+45 : y(data[ep][2])-35);

// exit marker
var xp = {exit_pos};
ctx.fillStyle = '#ff4444';
ctx.beginPath(); ctx.arc(x(xp), y(data[xp][3]), 8, 0, Math.PI*2); ctx.fill();
ctx.fillStyle = '{pnl_color}'; ctx.font = 'bold 22px sans-serif';
ctx.fillText('EXIT {pnl:+.2f}%', x(xp)-40, y(data[xp][2])-15);

// stop line
var stopP = {trade['stop_price']};
if (stopP >= pMin && stopP <= pMax) {{
    ctx.strokeStyle = '#ff444488'; ctx.lineWidth = 1; ctx.setLineDash([4,4]);
    ctx.beginPath(); ctx.moveTo(x(ep), y(stopP)); ctx.lineTo(x(Math.min(xp+2,n-1)), y(stopP)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#ff4444'; ctx.font = '16px monospace';
    ctx.fillText('STOP '+stopP.toFixed(1), x(ep)+15, y(stopP)-5);
}}

// x-axis time
ctx.fillStyle = '#666'; ctx.font = '16px monospace';
var step = Math.max(1, Math.floor(n / 8));
for (var i = 0; i < n; i += step) {{
    ctx.save(); ctx.translate(x(i), H - 5); ctx.rotate(-0.5);
    ctx.fillText(data[i][0].substring(5), 0, 0); ctx.restore();
}}

// legend
ctx.font = '18px sans-serif';
ctx.fillStyle='#ffaa00'; ctx.fillText('EMA10', W-200, 25);
ctx.fillStyle='#00aaff'; ctx.fillText('EMA20', W-120, 25);
}})();
"""


# ==================== HTML构建 ====================

html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>时间稳定性检验报告</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0d1117; color: #c9d1d9; font-family: 'Consolas', 'Monaco', monospace; padding: 20px; max-width: 1600px; margin: 0 auto; }
h1 { color: #58a6ff; font-size: 22px; margin-bottom: 6px; }
h2 { color: #ff7b72; font-size: 18px; margin: 30px 0 10px; border-bottom: 1px solid #30363d; padding-bottom: 6px; }
h3 { color: #d2a8ff; font-size: 15px; margin: 16px 0 8px; }
.subtitle { color: #8b949e; font-size: 13px; margin-bottom: 20px; }
table { border-collapse: collapse; margin: 8px 0 16px; width: 100%; font-size: 13px; }
th { background: #161b22; color: #8b949e; padding: 8px 10px; text-align: right; border: 1px solid #21262d; font-weight: normal; }
td { padding: 7px 10px; text-align: right; border: 1px solid #21262d; }
th:first-child, td:first-child { text-align: left; }
tr:hover td { background: #1c2333; }
.pos { color: #3fb950; }
.neg { color: #f85149; }
.best { color: #ffd700; font-weight: bold; }
.section { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin: 16px 0; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.grid-4 { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 12px; }
.highlight-row td { background: #1a2332; }
.note { background: #1a1a0d; border-left: 3px solid #d29922; padding: 10px 14px; margin: 12px 0; font-size: 13px; color: #d29922; }
.warn { background: #1a0d0d; border-left: 3px solid #f85149; padding: 10px 14px; margin: 12px 0; font-size: 13px; color: #f85149; }
.stable-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
.stable-4 { background: #1a3a1a; color: #3fb950; }
.stable-3 { background: #1f3a5f; color: #58a6ff; }
.stable-2 { background: #3a2f1a; color: #d29922; }
.stable-1 { background: #3a1a1a; color: #f85149; }
.stable-0 { background: #3a1a1a; color: #f85149; }
.case-card { background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 12px; margin: 12px 0; }
.case-header { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 8px; font-size: 13px; }
.case-tag { padding: 3px 10px; border-radius: 4px; font-weight: bold; }
.tag-win { background: #1a3a1a; color: #3fb950; }
.tag-lose { background: #3a1a1a; color: #f85149; }
canvas { width: 100%; height: 350px; background: #0a0a1a; border-radius: 5px; }
.window-section { border-left: 3px solid #30363d; padding-left: 12px; margin: 16px 0; }
.w-pos { border-left-color: #3fb950; }
.w-neg { border-left-color: #f85149; }
</style>
</head>
<body>

<h1>时间稳定性检验报告</h1>
<p class="subtitle">A+B类信号 | S2出场 | ER(20) 0.5~0.7 | 5跳止损 | 120天分4个30天窗口 | 2026-04-05</p>
"""

# ============ 模块一：全品种排名+分窗口 ============
html += '<h2>一、全品种时间稳定性（按120天EV排名）</h2>\n'
html += '<div class="section">\n'
html += '<table>\n'
html += '<tr><th>排名</th><th>品种</th><th>120天EV</th><th>N</th><th>胜率</th><th>盈亏比</th><th>总PnL%</th>'
for w in range(N_WINDOWS):
    html += f'<th>W{w+1} EV</th><th>N</th><th>PnL%</th>'
html += '<th>正EV窗口</th><th>标签</th></tr>\n'

for i, r in enumerate(ranked):
    name = r['name']
    data = all_sym_data[name]
    evs = []
    for w in range(N_WINDOWS):
        pnls_w = [t['pnl'] for t in data['windows'][w]]
        evs.append(calc_ev(pnls_w))

    pos_w = sum(1 for e in evs if e['EV'] > 0 and e['N'] > 0)
    valid_w = sum(1 for e in evs if e['N'] > 0)
    stability = f"{pos_w}/{valid_w}" if valid_w > 0 else "-"

    tag = 'Top3' if i < 3 else ('Top5' if i < 5 else ('Top10' if i < 10 else ''))
    is_top = i < 10
    row_class = ' class="highlight-row"' if i < 5 else ''

    ev_cls = 'best' if r['EV'] >= 1.0 else ('pos' if r['EV'] > 0 else 'neg')

    html += f'<tr{row_class}>'
    html += f'<td>{i+1}</td>'
    html += f'<td><b>{name}</b></td>' if is_top else f'<td>{name}</td>'
    html += f'<td class="{ev_cls}">{r["EV"]:+.2f}</td>'
    html += f'<td>{r["N"]}</td>'
    html += f'<td>{r["wr"]:.1f}%</td>'
    html += f'<td>{r["pr"]:.2f}</td>'
    html += f'<td class="{"pos" if r["sum"]>0 else "neg"}">{r["sum"]:+.1f}</td>'

    for w in range(N_WINDOWS):
        e = evs[w]
        if e['N'] == 0:
            html += '<td class="neutral">-</td><td>0</td><td>-</td>'
        else:
            ecls = 'pos' if e['EV'] > 0 else 'neg'
            html += f'<td class="{ecls}">{e["EV"]:+.2f}</td><td>{e["N"]}</td>'
            html += f'<td class="{"pos" if e["sum"]>0 else "neg"}">{e["sum"]:+.1f}</td>'

    badge_cls = f'stable-{pos_w}'
    html += f'<td><span class="stable-badge {badge_cls}">{stability}</span></td>'
    html += f'<td><b>{tag}</b></td>' if tag else '<td></td>'
    html += '</tr>\n'

html += '</table></div>\n'

# ============ 模块二：Top3/5/10组合汇总 ============
html += '<h2>二、Top组合分窗口稳定性</h2>\n'
html += '<div class="grid-2">\n'

for topn, topn_label in [(3, 'Top3'), (5, 'Top5'), (10, 'Top10')]:
    top_names = [r['name'] for r in ranked[:topn]]
    html += f'<div class="section"><h3>{topn_label}：{", ".join(top_names)}</h3>\n'
    html += '<table><tr><th>窗口</th><th>N</th><th>EV</th><th>胜率</th><th>盈亏比</th><th>总PnL%</th><th>平均赢%</th><th>平均亏%</th></tr>\n'

    for w in range(N_WINDOWS):
        pool = []
        for name in top_names:
            pool.extend([t['pnl'] for t in all_sym_data[name]['windows'][w]])
        ev = calc_ev(pool)
        ecls = 'pos' if ev['EV'] > 0 else 'neg'
        html += f'<tr><td>W{w+1} (第{w*30+1}~{(w+1)*30}天)</td><td>{ev["N"]}</td>'
        html += f'<td class="{ecls}">{ev["EV"]:+.2f}</td><td>{ev["wr"]:.1f}%</td>'
        html += f'<td>{ev["pr"]:.2f}</td>'
        html += f'<td class="{"pos" if ev["sum"]>0 else "neg"}">{ev["sum"]:+.1f}</td>'
        html += f'<td>{ev["avg_w"]:.4f}</td><td>{ev["avg_l"]:.4f}</td></tr>\n'

    # 120天汇总
    pool_all = []
    for name in top_names:
        pool_all.extend([t['pnl'] for t in all_sym_data[name]['trades']])
    ev_all = calc_ev(pool_all)
    ecls = 'pos' if ev_all['EV'] > 0 else 'neg'
    html += f'<tr class="highlight-row"><td><b>120天汇总</b></td><td><b>{ev_all["N"]}</b></td>'
    html += f'<td class="best">{ev_all["EV"]:+.2f}</td><td><b>{ev_all["wr"]:.1f}%</b></td>'
    html += f'<td><b>{ev_all["pr"]:.2f}</b></td>'
    html += f'<td class="{"pos" if ev_all["sum"]>0 else "neg"}"><b>{ev_all["sum"]:+.1f}</b></td>'
    html += f'<td>{ev_all["avg_w"]:.4f}</td><td>{ev_all["avg_l"]:.4f}</td></tr>\n'
    html += '</table></div>\n'

html += '</div>\n'

# 额外：全32品种组合
html += '<div class="section"><h3>全部32品种汇总</h3>\n'
html += '<table><tr><th>窗口</th><th>N</th><th>EV</th><th>胜率</th><th>盈亏比</th><th>总PnL%</th></tr>\n'
for w in range(N_WINDOWS):
    pool = []
    for name in all_sym_data:
        pool.extend([t['pnl'] for t in all_sym_data[name]['windows'][w]])
    ev = calc_ev(pool)
    ecls = 'pos' if ev['EV'] > 0 else 'neg'
    html += f'<tr><td>W{w+1}</td><td>{ev["N"]}</td><td class="{ecls}">{ev["EV"]:+.2f}</td>'
    html += f'<td>{ev["wr"]:.1f}%</td><td>{ev["pr"]:.2f}</td>'
    html += f'<td class="{"pos" if ev["sum"]>0 else "neg"}">{ev["sum"]:+.1f}</td></tr>\n'
pool_all = []
for name in all_sym_data:
    pool_all.extend([t['pnl'] for t in all_sym_data[name]['trades']])
ev_all = calc_ev(pool_all)
html += f'<tr class="highlight-row"><td><b>120天</b></td><td><b>{ev_all["N"]}</b></td>'
html += f'<td class="best">{ev_all["EV"]:+.2f}</td><td><b>{ev_all["wr"]:.1f}%</b></td>'
html += f'<td><b>{ev_all["pr"]:.2f}</b></td>'
html += f'<td class="pos"><b>{ev_all["sum"]:+.1f}</b></td></tr>\n'
html += '</table></div>\n'

# ============ 模块三：逐品种分窗口详细 ============
html += '<h2>三、Top10逐品种分窗口详细</h2>\n'

for i, r in enumerate(ranked[:10]):
    name = r['name']
    data = all_sym_data[name]
    tag = 'Top3' if i < 3 else ('Top5' if i < 5 else 'Top10')

    html += f'<div class="section"><h3>#{i+1} {name} <span style="color:#8b949e">({tag})</span> — 120天 EV={r["EV"]:+.2f} N={r["N"]} 总PnL={r["sum"]:+.1f}%</h3>\n'
    html += '<div class="grid-4">\n'

    for w in range(N_WINDOWS):
        w_trades = data['windows'][w]
        pnls = [t['pnl'] for t in w_trades]
        ev = calc_ev(pnls)
        w_cls = 'w-pos' if ev['EV'] > 0 else 'w-neg'

        html += f'<div class="window-section {w_cls}">'
        html += f'<b>W{w+1} (第{w*30+1}~{(w+1)*30}天)</b><br>'
        if ev['N'] == 0:
            html += '<span class="neutral">无交易</span>'
        else:
            ecls = 'pos' if ev['EV'] > 0 else 'neg'
            html += f'EV=<span class="{ecls}">{ev["EV"]:+.2f}</span> N={ev["N"]} '
            html += f'胜率={ev["wr"]:.1f}%<br>'
            html += f'PnL=<span class="{"pos" if ev["sum"]>0 else "neg"}">{ev["sum"]:+.1f}%</span> '
            html += f'盈亏比={ev["pr"]:.2f}<br>'
            # 逐笔
            html += '<span style="font-size:11px;color:#8b949e">'
            for t in w_trades:
                cls = 'pos' if t['pnl'] > 0 else 'neg'
                html += f'<span class="{cls}">{t["pnl"]:+.2f}%</span> '
            html += '</span>'
        html += '</div>\n'

    html += '</div></div>\n'

# ============ 模块四：白银K线案例 ============
html += '<h2>四、白银 分窗口交易K线复盘</h2>\n'

if ag_df is not None and ag_data:
    chart_idx = 0
    for w in range(N_WINDOWS):
        w_trades = ag_data['windows'][w]
        ev = calc_ev([t['pnl'] for t in w_trades])
        total_n = len(w_trades)
        show_trades = ag_cases.get(w, [])

        w_cls = 'w-pos' if ev['EV'] > 0 else 'w-neg'
        html += f'<div class="section">\n'
        html += f'<h3 class="{"pos" if ev["EV"]>0 else "neg"}">W{w+1} (第{w*30+1}~{(w+1)*30}天) — EV={ev["EV"]:+.2f} N={total_n} 总PnL={ev["sum"]:+.1f}%</h3>\n'

        if total_n <= 5:
            html += f'<div class="note">本窗口共{total_n}笔交易，全部展示</div>\n'
        else:
            html += f'<div class="note">本窗口共{total_n}笔交易，展示最赚2笔 + 最亏2笔</div>\n'

        # 逐笔PnL一览
        if w_trades:
            html += '<div style="margin:8px 0;font-size:13px">全部交易PnL: '
            for t in sorted(w_trades, key=lambda x: x['entry_idx']):
                cls = 'pos' if t['pnl'] > 0 else 'neg'
                html += f'<span class="{cls}">{t["pnl"]:+.2f}%</span> '
            html += '</div>\n'

        for trade in show_trades:
            pnl = trade['pnl']
            tag_cls = 'tag-win' if pnl > 0 else 'tag-lose'
            dir_label = '做多' if trade['direction'] == 'long' else '做空'
            chart_id = f'ag_chart_{chart_idx}'

            # 获取入场时间
            entry_time = ''
            if 'datetime' in ag_df.columns:
                entry_time = str(ag_df.iloc[trade['entry_idx']]['datetime'])[:16]

            html += f'<div class="case-card">\n'
            html += f'<div class="case-header">'
            html += f'<span class="case-tag {tag_cls}">PnL {pnl:+.2f}%</span>'
            html += f'<span>{dir_label} {trade["type"]}类</span>'
            html += f'<span>入场: {entry_time}</span>'
            html += f'<span>ER={trade["er_20"]:.3f}</span>'
            html += f'<span>入场价={trade["entry_price"]}</span>'
            html += f'</div>\n'
            html += f'<canvas id="{chart_id}"></canvas>\n'
            html += f'<script>\n{make_chart_js(chart_id, ag_df, trade)}\n</script>\n'
            html += '</div>\n'
            chart_idx += 1

        html += '</div>\n'

# ============ 模块五：结论 ============
html += '<h2>五、关键发现</h2>\n'
html += '<div class="section">\n'

# 找稳定品种
stable_4 = [r['name'] for i, r in enumerate(ranked) if sum(1 for w in range(N_WINDOWS) if calc_ev([t['pnl'] for t in all_sym_data[r['name']]['windows'][w]])['EV'] > 0 and calc_ev([t['pnl'] for t in all_sym_data[r['name']]['windows'][w]])['N'] > 0) == 4]
stable_3 = [r['name'] for i, r in enumerate(ranked[:15]) if sum(1 for w in range(N_WINDOWS) if calc_ev([t['pnl'] for t in all_sym_data[r['name']]['windows'][w]])['EV'] > 0 and calc_ev([t['pnl'] for t in all_sym_data[r['name']]['windows'][w]])['N'] > 0) == 3]

html += f'<p><b>4/4稳定品种（每个窗口都正EV）：</b> <span class="pos">{", ".join(stable_4) if stable_4 else "无"}</span></p>\n'
html += f'<p><b>3/4稳定品种（Top15中）：</b> <span style="color:#58a6ff">{", ".join(stable_3) if stable_3 else "无"}</span></p>\n'

html += '</div>\n'

# 数据参数
html += """
<div class="section" style="margin-top:20px">
  <h3>数据参数</h3>
  <table style="width:auto">
    <tr><td style="text-align:left">策略</td><td>A+B类信号 | S2出场 | ER(20) 0.5~0.7 | 5跳止损</td></tr>
    <tr><td style="text-align:left">周期</td><td>10min</td></tr>
    <tr><td style="text-align:left">趋势判断</td><td>EMA20 > EMA120</td></tr>
    <tr><td style="text-align:left">数据范围</td><td>120天 x 32品种</td></tr>
    <tr><td style="text-align:left">窗口划分</td><td>4 x 30天 (W1=最早, W4=最近)</td></tr>
    <tr><td style="text-align:left">K线案例选择</td><td>每窗口最赚2笔+最亏2笔（<=5笔全展示）</td></tr>
  </table>
</div>
"""

html += '</body></html>'

out_path = os.path.join(OUTPUT_DIR, 'time_stability_report.html')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"HTML report saved to {out_path}")
print(f"Total chart cases: {sum(len(v) for v in ag_cases.values())}")
