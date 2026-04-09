# -*- coding: utf-8 -*-
"""
偏离度场景K线示例
================
三个场景各5个命中示例 + 各5个漏掉的示例
"""

import os, sys, random
import numpy as np
import pandas as pd
from signal_core import SYMBOL_CONFIGS

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

LAST_DAYS = 120
BARS_PER_DAY = 57
STOP_TICKS = 5
MIN_PB_BARS_C = 4
ATR_PERIOD = 14

HIGH_VOL = {
    "GFEX.lc", "DCE.jm", "SHFE.ag", "CZCE.FG", "CZCE.SA",
    "INE.sc", "CZCE.MA", "CZCE.TA", "DCE.eb", "DCE.lh",
}


def load_and_prepare(cache_key):
    path = os.path.join(CACHE_DIR, f"{cache_key}_10min_170d.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema120'] = df['close'].ewm(span=120, adjust=False).mean()
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs(),
        )
    )
    df['atr'] = df['tr'].rolling(window=ATR_PERIOD).mean()
    net = (df['close'] - df['close'].shift(20)).abs()
    bar_sum = df['close'].diff().abs().rolling(20).sum()
    df['er_20'] = net / bar_sum.replace(0, np.nan)
    return df


def detect_signals(df, start_idx):
    signals = []
    n = len(df)
    trend_dir = 0
    below_start = -1
    pb_low = pb_high = None
    prev_close = prev_ema10 = None

    for i in range(1, n):
        row = df.iloc[i]
        close, high, low = row['close'], row['high'], row['low']
        ema10, ema20, ema120 = row['ema10'], row['ema20'], row['ema120']
        atr = row.get('atr', np.nan)

        if prev_close is None or pd.isna(ema120) or pd.isna(ema10) or pd.isna(atr) or atr <= 0:
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
        dev_atr = abs(close - ema10) / atr

        if trend_dir == 1:
            if i >= start_idx and low <= ema10 and close > ema10 and prev_close > prev_ema10:
                if below_start == -1:
                    signals.append({
                        'idx': i, 'type': 'A', 'direction': 'long',
                        'entry_price': close, 'pullback_extreme': low,
                        'er_20': er_20, 'dev_atr': round(dev_atr, 4),
                    })
            if below_start == -1:
                if close < ema10 and prev_close >= prev_ema10:
                    below_start, pb_low = i, low
            else:
                pb_low = min(pb_low, low)
                if close > ema10:
                    pb_bars = i - below_start
                    if i >= start_idx and pb_bars >= 1:
                        sig_type = 'B' if pb_bars < MIN_PB_BARS_C else 'C'
                        signals.append({
                            'idx': i, 'type': sig_type, 'direction': 'long',
                            'entry_price': close, 'pullback_extreme': pb_low,
                            'er_20': er_20, 'dev_atr': round(dev_atr, 4),
                        })
                    below_start, pb_low = -1, None

        elif trend_dir == -1:
            if i >= start_idx and high >= ema10 and close < ema10 and prev_close < prev_ema10:
                if below_start == -1:
                    signals.append({
                        'idx': i, 'type': 'A', 'direction': 'short',
                        'entry_price': close, 'pullback_extreme': high,
                        'er_20': er_20, 'dev_atr': round(dev_atr, 4),
                    })
            if below_start == -1:
                if close > ema10 and prev_close <= prev_ema10:
                    below_start, pb_high = i, high
            else:
                pb_high = max(pb_high, high)
                if close < ema10:
                    pb_bars = i - below_start
                    if i >= start_idx and pb_bars >= 1:
                        sig_type = 'B' if pb_bars < MIN_PB_BARS_C else 'C'
                        signals.append({
                            'idx': i, 'type': sig_type, 'direction': 'short',
                            'entry_price': close, 'pullback_extreme': pb_high,
                            'er_20': er_20, 'dev_atr': round(dev_atr, 4),
                        })
                    below_start, pb_high = -1, None

        prev_close, prev_ema10 = close, ema10
    return signals


def simulate_exits(df, signals, tick_size):
    trades = []
    n = len(df)
    tick = tick_size * STOP_TICKS

    for sig in signals:
        entry_idx = sig['idx']
        entry_price = sig['entry_price']
        is_long = sig['direction'] == 'long'
        pb_ext = sig['pullback_extreme']

        init_stop = (pb_ext - tick) if is_long else (pb_ext + tick)
        s1_stop = s2_stop = s3_stop = init_stop
        s2_state = 'normal'
        s2_tracking = None
        s1_done = s2_done = s3_done = False
        s1_pnl = s2_pnl = s3_pnl = 0.0
        s1_exit_idx = s2_exit_idx = s3_exit_idx = n - 1
        s1_exit_price = s2_exit_price = s3_exit_price = 0.0

        prev_bar = df.iloc[entry_idx]

        for j in range(entry_idx + 1, n):
            bar = df.iloc[j]
            if pd.isna(bar['ema10']):
                prev_bar = bar
                continue
            close, high, low = bar['close'], bar['high'], bar['low']
            ma_val = bar['ema10']
            p_close = prev_bar['close']
            p_low, p_high = prev_bar['low'], prev_bar['high']

            if not s1_done:
                if (is_long and low <= s1_stop) or (not is_long and high >= s1_stop):
                    s1_done = True
                    ep = s1_stop
                    s1_pnl = (ep - entry_price) / entry_price * 100 if is_long else (entry_price - ep) / entry_price * 100
                    s1_exit_idx = j
                    s1_exit_price = ep
                else:
                    if is_long and close > p_close:
                        s1_stop = max(s1_stop, low - tick)
                    elif not is_long and close < p_close:
                        s1_stop = min(s1_stop, high + tick)

            if not s2_done:
                if (is_long and low <= s2_stop) or (not is_long and high >= s2_stop):
                    s2_done = True
                    ep = s2_stop
                    s2_pnl = (ep - entry_price) / entry_price * 100 if is_long else (entry_price - ep) / entry_price * 100
                    s2_exit_idx = j
                    s2_exit_price = ep
                else:
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

            if not s3_done:
                if (is_long and low <= s3_stop) or (not is_long and high >= s3_stop):
                    s3_done = True
                    ep = s3_stop
                    s3_pnl = (ep - entry_price) / entry_price * 100 if is_long else (entry_price - ep) / entry_price * 100
                    s3_exit_idx = j
                    s3_exit_price = ep
                else:
                    if is_long and close > p_close:
                        s3_stop = max(s3_stop, p_low - tick)
                    elif not is_long and close < p_close:
                        s3_stop = min(s3_stop, p_high + tick)

            prev_bar = bar
            if s1_done and s2_done and s3_done:
                break

        last_close = df.iloc[-1]['close']
        if not s1_done:
            s1_pnl = (last_close - entry_price) / entry_price * 100 if is_long else (entry_price - last_close) / entry_price * 100
            s1_exit_price = last_close
        if not s2_done:
            s2_pnl = (last_close - entry_price) / entry_price * 100 if is_long else (entry_price - last_close) / entry_price * 100
            s2_exit_price = last_close
        if not s3_done:
            s3_pnl = (last_close - entry_price) / entry_price * 100 if is_long else (entry_price - last_close) / entry_price * 100
            s3_exit_price = last_close

        s4_pnl = (s1_pnl + s2_pnl) / 2

        # 取最长出场的idx作为显示范围
        max_exit_idx = max(s1_exit_idx, s2_exit_idx, s3_exit_idx)

        trades.append({
            'type': sig['type'], 'direction': sig['direction'],
            'er_20': sig['er_20'], 'dev_atr': sig['dev_atr'],
            'entry_idx': entry_idx, 'entry_price': entry_price,
            'init_stop': init_stop,
            's1_pnl': round(s1_pnl, 4), 's1_exit_idx': s1_exit_idx, 's1_exit_price': s1_exit_price,
            's2_pnl': round(s2_pnl, 4), 's2_exit_idx': s2_exit_idx, 's2_exit_price': s2_exit_price,
            's3_pnl': round(s3_pnl, 4), 's3_exit_idx': s3_exit_idx, 's3_exit_price': s3_exit_price,
            's4_pnl': round(s4_pnl, 4),
            'max_exit_idx': max_exit_idx,
        })
    return trades


def classify_scenario(t):
    """返回命中的场景编号，0=未命中"""
    er = t.get('er_20', 0)
    if pd.isna(er): er = 0
    # 全局过滤：ER>=0.7正期望消失
    if er >= 0.7:
        return 0
    if t['type'] == 'A' and er >= 0.5 and t['dev_atr'] >= 1.0:
        return 1
    if t['type'] == 'C' and t['dev_atr'] >= 2.0:
        return 2
    if t['type'] == 'B' and er >= 0.5 and t['dev_atr'] >= 0.1 and t['dev_atr'] < 0.3:
        return 3
    return 0


def render_kline_html(all_examples):
    """生成K线图HTML - 用Canvas+JS渲染，避免巨大SVG"""
    import json

    # 准备JSON数据（精简浮点数）
    charts_data = []
    for ex in all_examples:
        bars_compact = []
        for b in ex['bars']:
            bars_compact.append([
                round(b['open'], 2), round(b['close'], 2),
                round(b['high'], 2), round(b['low'], 2),
                round(b.get('ema10', 0) or 0, 2),
                round(b.get('ema20', 0) or 0, 2),
                round(b.get('ema120', 0) or 0, 2),
            ])
        charts_data.append({
            'b': bars_compact,  # bars: [o,c,h,l,ema10,ema20,ema120]
            'ep': ex['entry_pos'],
            'dir': ex['direction'],
            'stop': round(ex['init_stop'], 2),
            'sn': ex['symbol_name'],
            'tp': ex['type'],
            'da': round(ex['dev_atr'], 2),
            'er': round(ex['er_20'], 2) if not pd.isna(ex['er_20']) else -1,
            'p1': round(ex['s1_pnl'], 2), 'x1': ex['s1_exit_pos'], 'xp1': round(ex['s1_exit_price'], 2),
            'p2': round(ex['s2_pnl'], 2), 'x2': ex['s2_exit_pos'], 'xp2': round(ex['s2_exit_price'], 2),
            'p3': round(ex['s3_pnl'], 2), 'x3': ex['s3_exit_pos'], 'xp3': round(ex['s3_exit_price'], 2),
            'p4': round(ex['s4_pnl'], 2),
            'sc': ex['scenario'],
            'hit': ex['is_hit'],
        })

    data_json = json.dumps(charts_data, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>偏离度场景K线示例</title>
<style>
body {{ background: #1a1a2e; color: #eee; font-family: 'Microsoft YaHei', sans-serif; padding: 20px; }}
h1 {{ color: #e94560; text-align: center; }}
h2 {{ color: #0ff; margin-top: 40px; border-bottom: 1px solid #0ff; padding-bottom: 8px; }}
h3 {{ color: #ffa500; margin-top: 25px; }}
.chart-box {{ background: #16213e; border-radius: 8px; padding: 15px; margin: 15px 0; border: 1px solid #333; }}
.info {{ color: #aaa; font-size: 13px; margin-bottom: 8px; }}
.info span {{ margin-right: 12px; }}
.pos {{ color: #4caf50; }}
.neg {{ color: #f44336; }}
.lbl {{ color: #888; }}
</style></head><body>
<h1>入场偏离度 - K线场景示例</h1>
<div id="root"></div>
<script>
const DATA = {data_json};
const SCENARIOS = {{
  1: '场景1：A类 + ER>=0.5 + 偏离>=1.0ATR',
  2: '场景2：C类 + 不过ER + 偏离>=2.0ATR',
  3: '场景3：B类 + ER>=0.5 + 偏离<0.3ATR'
}};

function drawChart(canvas, d) {{
  const ctx = canvas.getContext('2d');
  const W = 900, H = 320;
  canvas.width = W; canvas.height = H;
  const pl=60, pr=20, pt=25, pb=25;
  const cw = W-pl-pr, ch = H-pt-pb;
  const bars = d.b, n = bars.length;
  const bw = cw / n;

  // price range (include ema10/ema20/ema120)
  let pMax = -1e18, pMin = 1e18;
  for (const b of bars) {{
    if (b[2] > pMax) pMax = b[2];
    if (b[3] < pMin) pMin = b[3];
    for (let ei=4; ei<=6; ei++) {{
      if (b[ei] > 0) {{
        if (b[ei] > pMax) pMax = b[ei];
        if (b[ei] < pMin) pMin = b[ei];
      }}
    }}
  }}
  let mg = (pMax-pMin)*0.05;
  pMax += mg; pMin -= mg;
  if (pMax === pMin) pMax += 1;

  function py(p) {{ return pt + ch * (1 - (p-pMin)/(pMax-pMin)); }}
  function bx(i) {{ return pl + bw*(i+0.5); }}

  // background
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0,0,W,H);

  // grid
  ctx.strokeStyle = '#222'; ctx.lineWidth = 0.5;
  ctx.fillStyle = '#666'; ctx.font = '10px sans-serif'; ctx.textAlign = 'right';
  for (let g=0; g<=5; g++) {{
    let gp = pMin + (pMax-pMin)*g/5;
    let gy = py(gp);
    ctx.beginPath(); ctx.moveTo(pl, gy); ctx.lineTo(W-pr, gy); ctx.stroke();
    ctx.fillText(gp.toFixed(1), pl-4, gy+3);
  }}

  // signal bar highlight
  let ex = bx(d.ep) - bw*0.5;
  ctx.fillStyle = 'rgba(255,165,0,0.15)';
  ctx.fillRect(ex, pt, bw, ch);

  // candles
  for (let i=0; i<n; i++) {{
    const [o,c,h,l] = bars[i];
    const cx = bx(i);
    const up = c >= o;
    ctx.strokeStyle = ctx.fillStyle = up ? '#26a69a' : '#ef5350';

    // wick
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx, py(h)); ctx.lineTo(cx, py(l)); ctx.stroke();

    // body
    const yt = py(Math.max(o,c)), yb = py(Math.min(o,c));
    const bh = Math.max(yb-yt, 1);
    const bodyW = bw * 0.6;
    ctx.fillRect(cx-bodyW/2, yt, bodyW, bh);
  }}

  // EMA10
  ctx.beginPath(); ctx.strokeStyle = '#ffeb3b'; ctx.lineWidth = 1.2; ctx.globalAlpha = 0.8;
  let started = false;
  for (let i=0; i<n; i++) {{
    if (bars[i][4] > 0) {{
      if (!started) {{ ctx.moveTo(bx(i), py(bars[i][4])); started=true; }}
      else ctx.lineTo(bx(i), py(bars[i][4]));
    }}
  }}
  ctx.stroke(); ctx.globalAlpha = 1;

  // EMA20
  ctx.beginPath(); ctx.strokeStyle = '#2196f3'; ctx.lineWidth = 1; ctx.globalAlpha = 0.6;
  started = false;
  for (let i=0; i<n; i++) {{
    if (bars[i][5] > 0) {{
      if (!started) {{ ctx.moveTo(bx(i), py(bars[i][5])); started=true; }}
      else ctx.lineTo(bx(i), py(bars[i][5]));
    }}
  }}
  ctx.stroke(); ctx.globalAlpha = 1;

  // EMA120
  ctx.beginPath(); ctx.strokeStyle = '#e040fb'; ctx.lineWidth = 1.5; ctx.globalAlpha = 0.5;
  started = false;
  for (let i=0; i<n; i++) {{
    if (bars[i][6] > 0) {{
      if (!started) {{ ctx.moveTo(bx(i), py(bars[i][6])); started=true; }}
      else ctx.lineTo(bx(i), py(bars[i][6]));
    }}
  }}
  ctx.stroke(); ctx.globalAlpha = 1;

  // entry marker
  const ecx = bx(d.ep), ey = py(bars[d.ep][1]);
  ctx.fillStyle = d.dir==='long' ? '#4caf50' : '#f44336';
  ctx.beginPath();
  if (d.dir==='long') {{
    ctx.moveTo(ecx, ey+12); ctx.lineTo(ecx-5, ey+22); ctx.lineTo(ecx+5, ey+22);
  }} else {{
    ctx.moveTo(ecx, ey-12); ctx.lineTo(ecx-5, ey-22); ctx.lineTo(ecx+5, ey-22);
  }}
  ctx.fill();

  // stop line
  ctx.setLineDash([4,3]);
  ctx.strokeStyle = '#f44336'; ctx.lineWidth = 1; ctx.globalAlpha = 0.6;
  ctx.beginPath(); ctx.moveTo(ecx, py(d.stop)); ctx.lineTo(W-pr, py(d.stop)); ctx.stroke();
  ctx.setLineDash([]); ctx.globalAlpha = 1;

  // exit markers S1/S2/S3
  const exits = [
    {{pos: d.x1, price: d.xp1, label: 'S1', color: '#ff6b6b'}},
    {{pos: d.x2, price: d.xp2, label: 'S2', color: '#ffd93d'}},
    {{pos: d.x3, price: d.xp3, label: 'S3', color: '#6bcb77'}},
  ];
  ctx.font = 'bold 9px sans-serif'; ctx.textAlign = 'center';
  for (let ei=0; ei<exits.length; ei++) {{
    const e = exits[ei];
    if (e.pos < 0 || e.pos >= n) continue;
    const ex2 = bx(e.pos), epy = py(e.price);
    // X marker
    ctx.strokeStyle = e.color; ctx.lineWidth = 2; ctx.globalAlpha = 0.9;
    ctx.beginPath(); ctx.moveTo(ex2-4, epy-4); ctx.lineTo(ex2+4, epy+4); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(ex2+4, epy-4); ctx.lineTo(ex2-4, epy+4); ctx.stroke();
    // label above/below
    ctx.fillStyle = e.color; ctx.globalAlpha = 1;
    const labelY = d.dir==='long' ? epy+16+ei*11 : epy-10-ei*11;
    ctx.fillText(e.label, ex2, labelY);
  }}
  ctx.globalAlpha = 1;

  // legend
  ctx.font = '10px sans-serif';
  ctx.fillStyle = '#ffeb3b'; ctx.textAlign = 'left';
  ctx.fillText('EMA10', pl+5, pt+12);
  ctx.fillStyle = '#2196f3';
  ctx.fillText('EMA20', pl+50, pt+12);
  ctx.fillStyle = '#e040fb';
  ctx.fillText('EMA120', pl+100, pt+12);
  ctx.fillStyle = '#ff6b6b';
  ctx.fillText('S1', pl+160, pt+12);
  ctx.fillStyle = '#ffd93d';
  ctx.fillText('S2', pl+180, pt+12);
  ctx.fillStyle = '#6bcb77';
  ctx.fillText('S3', pl+200, pt+12);
}}

function pnlSpan(v) {{
  const cls = v >= 0 ? 'pos' : 'neg';
  return '<span class="'+cls+'">'+( v>=0 ? '+' : '')+v.toFixed(2)+'%</span>';
}}

const root = document.getElementById('root');

for (const sid of [1,2,3]) {{
  root.innerHTML += '<h2>' + SCENARIOS[sid] + '</h2>';
  root.innerHTML += '<h3>命中示例（符合条件的交易）</h3>';
  const hits = DATA.filter(d => d.sc===sid && d.hit);
  for (const d of hits) {{
    const id = 'c'+Math.random().toString(36).substr(2,8);
    const er = d.er >= 0 ? d.er.toFixed(2) : 'N/A';
    root.innerHTML += '<div class="chart-box"><div class="info">' +
      '<span><span class="lbl">品种:</span> '+d.sn+'</span>' +
      '<span><span class="lbl">类型:</span> '+d.tp+'类 '+d.dir+'</span>' +
      '<span><span class="lbl">偏离:</span> '+d.da+' ATR</span>' +
      '<span><span class="lbl">ER(20):</span> '+er+'</span>' +
      '<span><span class="lbl">S1:</span> '+pnlSpan(d.p1)+'</span>' +
      '<span><span class="lbl">S2:</span> '+pnlSpan(d.p2)+'</span>' +
      '<span><span class="lbl">S3:</span> '+pnlSpan(d.p3)+'</span>' +
      '<span><span class="lbl">S4:</span> '+pnlSpan(d.p4)+'</span>' +
      '</div><canvas id="'+id+'"></canvas></div>';
    setTimeout(() => drawChart(document.getElementById(id), d), 0);
  }}

  root.innerHTML += '<h3>漏掉示例（同类型但不符合条件）</h3>';
  const misses = DATA.filter(d => d.sc===sid && !d.hit);
  for (const d of misses) {{
    const id = 'c'+Math.random().toString(36).substr(2,8);
    const er = d.er >= 0 ? d.er.toFixed(2) : 'N/A';
    root.innerHTML += '<div class="chart-box"><div class="info">' +
      '<span><span class="lbl">品种:</span> '+d.sn+'</span>' +
      '<span><span class="lbl">类型:</span> '+d.tp+'类 '+d.dir+'</span>' +
      '<span><span class="lbl">偏离:</span> '+d.da+' ATR</span>' +
      '<span><span class="lbl">ER(20):</span> '+er+'</span>' +
      '<span><span class="lbl">S1:</span> '+pnlSpan(d.p1)+'</span>' +
      '<span><span class="lbl">S2:</span> '+pnlSpan(d.p2)+'</span>' +
      '<span><span class="lbl">S3:</span> '+pnlSpan(d.p3)+'</span>' +
      '<span><span class="lbl">S4:</span> '+pnlSpan(d.p4)+'</span>' +
      '</div><canvas id="'+id+'"></canvas></div>';
    setTimeout(() => drawChart(document.getElementById(id), d), 0);
  }}
}}
</script>
</body></html>"""
    return html



def pick_examples(all_trades_with_df, scenario_id, is_hit, count=5):
    """挑选示例，尽量分散品种"""
    candidates = []
    for sym, sym_name, df, trades in all_trades_with_df:
        for t in trades:
            matched = classify_scenario(t)
            if is_hit and matched == scenario_id:
                candidates.append((sym, sym_name, df, t))
            elif not is_hit:
                # 漏掉的：同类型信号但不满足场景条件
                if scenario_id == 1 and t['type'] == 'A' and matched != 1:
                    candidates.append((sym, sym_name, df, t))
                elif scenario_id == 2 and t['type'] == 'C' and matched != 2:
                    candidates.append((sym, sym_name, df, t))
                elif scenario_id == 3 and t['type'] == 'B' and matched != 3:
                    candidates.append((sym, sym_name, df, t))

    if not candidates:
        return []

    # 分散品种选择
    random.seed(42)
    by_sym = {}
    for c in candidates:
        by_sym.setdefault(c[0], []).append(c)

    selected = []
    syms = list(by_sym.keys())
    random.shuffle(syms)

    # 先每个品种选1个，再循环
    sym_idx = 0
    while len(selected) < count and len(selected) < len(candidates):
        sym = syms[sym_idx % len(syms)]
        pool = by_sym[sym]
        if pool:
            pick = random.choice(pool)
            pool.remove(pick)
            selected.append(pick)
        sym_idx += 1
        if sym_idx >= len(syms) * 3:
            break

    # 转为chart数据
    examples = []
    for sym, sym_name, df, t in selected:
        entry_idx = t['entry_idx']
        max_exit = t['max_exit_idx']

        # 显示范围：信号前15根到最后出场后5根
        start = max(0, entry_idx - 15)
        end = min(len(df), max(entry_idx + 30, max_exit + 5))
        # 限制最多120根（确保出场点可见）
        if end - start > 120:
            end = start + 120

        bars = []
        for i in range(start, end):
            row = df.iloc[i]
            bars.append({
                'open': row['open'], 'close': row['close'],
                'high': row['high'], 'low': row['low'],
                'ema10': row.get('ema10', np.nan),
                'ema20': row.get('ema20', np.nan),
                'ema120': row.get('ema120', np.nan),
            })

        # 出场位置转为bars中的相对位置，超出范围标-1
        def rel_pos(idx):
            p = idx - start
            return p if 0 <= p < len(bars) else -1

        examples.append({
            'symbol_name': sym_name,
            'type': t['type'],
            'direction': t['direction'],
            'dev_atr': t['dev_atr'],
            'er_20': t['er_20'],
            's1_pnl': t['s1_pnl'],
            's2_pnl': t['s2_pnl'],
            's3_pnl': t['s3_pnl'],
            's4_pnl': t['s4_pnl'],
            'init_stop': t['init_stop'],
            'entry_pos': entry_idx - start,
            's1_exit_pos': rel_pos(t['s1_exit_idx']),
            's2_exit_pos': rel_pos(t['s2_exit_idx']),
            's3_exit_pos': rel_pos(t['s3_exit_idx']),
            's1_exit_price': t['s1_exit_price'],
            's2_exit_price': t['s2_exit_price'],
            's3_exit_price': t['s3_exit_price'],
            'bars': bars,
            'scenario': scenario_id,
            'is_hit': is_hit,
        })

    return examples


# === 主逻辑 ===
print("加载数据...")
all_trades_with_df = []
for sym, cfg in SYMBOL_CONFIGS.items():
    cache_key = sym.replace(".", "_")
    df = load_and_prepare(cache_key)
    if df is None:
        continue
    n = len(df)
    start_idx = max(0, n - LAST_DAYS * BARS_PER_DAY)

    signals = detect_signals(df, start_idx)
    if not signals:
        continue

    trades = simulate_exits(df, signals, cfg['tick_size'])
    all_trades_with_df.append((sym, cfg['name'], df, trades))
    print(f"  {cfg['name']}: {len(trades)} trades")

# 统计各场景命中数
hit_counts = {1: 0, 2: 0, 3: 0, 0: 0}
for sym, sym_name, df, trades in all_trades_with_df:
    for t in trades:
        s = classify_scenario(t)
        hit_counts[s] += 1

print(f"\n场景命中统计:")
print(f"  场景1 (A+ER>=0.5+>=1.0ATR): {hit_counts[1]}")
print(f"  场景2 (C+>=2.0ATR): {hit_counts[2]}")
print(f"  场景3 (B+ER>=0.5+<0.3ATR): {hit_counts[3]}")
print(f"  未命中: {hit_counts[0]}")

# 挑选示例
print("\n挑选示例...")
all_examples = []
for scenario_id in [1, 2, 3]:
    hits = pick_examples(all_trades_with_df, scenario_id, is_hit=True, count=5)
    misses = pick_examples(all_trades_with_df, scenario_id, is_hit=False, count=5)
    all_examples.extend(hits)
    all_examples.extend(misses)
    print(f"  场景{scenario_id}: {len(hits)} 命中 + {len(misses)} 漏掉")

# 生成HTML
print("生成HTML...")
html = render_kline_html(all_examples)
out_path = os.path.join(OUTPUT_DIR, "deviation_kline_examples.html")
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"输出: {out_path}")
