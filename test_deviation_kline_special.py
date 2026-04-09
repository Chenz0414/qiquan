# -*- coding: utf-8 -*-
"""
特殊案例K线示例
================
命中规则但亏钱 + 没命中但赚钱，各3个有特色的
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


def best_exit_pnl(t):
    """该单最优出场的pnl"""
    return max(t['s1_pnl'], t['s2_pnl'], t['s3_pnl'])


def worst_exit_pnl(t):
    return min(t['s1_pnl'], t['s2_pnl'], t['s3_pnl'])


# === 主逻辑 ===
print("加载数据...")
all_data = []
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
    all_data.append((sym, cfg['name'], df, trades))
    print(f"  {cfg['name']}: {len(trades)} trades")

# 分类
hit_lose = []   # 命中规则但亏钱
miss_win = []   # 没命中但赚钱

for sym, name, df, trades in all_data:
    for t in trades:
        sc = classify_scenario(t)
        if sc > 0:
            # 命中了某个场景
            # 亏钱 = 最优出场都亏（确实无可挽救的亏）
            if best_exit_pnl(t) < -0.05:
                hit_lose.append((sym, name, df, t, sc))
        else:
            # 没命中任何场景
            # 赚钱 = 最优出场赚了不少
            if best_exit_pnl(t) > 0.3:
                miss_win.append((sym, name, df, t, sc))

print(f"\n命中但亏: {len(hit_lose)}")
print(f"没命中但赚: {len(miss_win)}")

# 分场景统计命中但亏
for sc_id in [1, 2, 3]:
    sc_loses = [x for x in hit_lose if x[4] == sc_id]
    print(f"  场景{sc_id}亏损: {len(sc_loses)}")

# 挑选有特色的3个命中但亏
# 策略：每个场景挑1个，选亏损最大的
random.seed(42)
selected_hit_lose = []
for sc_id in [1, 2, 3]:
    sc_loses = [x for x in hit_lose if x[4] == sc_id]
    if sc_loses:
        # 按最优出场pnl排序（最亏的排前面）
        sc_loses.sort(key=lambda x: best_exit_pnl(x[3]))
        selected_hit_lose.append(sc_loses[0])
        print(f"  场景{sc_id}选中: {sc_loses[0][1]} {sc_loses[0][3]['type']}类 "
              f"best_pnl={best_exit_pnl(sc_loses[0][3]):.2f}% "
              f"dev={sc_loses[0][3]['dev_atr']:.2f} er={sc_loses[0][3]['er_20']:.2f}")

# 挑选有特色的3个没命中但赚
# 策略：按赚钱幅度排序，但分散信号类型
miss_win.sort(key=lambda x: -best_exit_pnl(x[3]))

# 每个信号类型各选1个最赚的
selected_miss_win = []
used_types = set()
for x in miss_win:
    tp = x[3]['type']
    if tp not in used_types:
        selected_miss_win.append(x)
        used_types.add(tp)
        print(f"  漏掉赚钱: {x[1]} {tp}类 best_pnl=+{best_exit_pnl(x[3]):.2f}% "
              f"dev={x[3]['dev_atr']:.2f} er={x[3]['er_20']:.2f}")
    if len(selected_miss_win) >= 3:
        break

# 构建chart数据
def make_chart_data(sym, name, df, t, tag, note):
    entry_idx = t['entry_idx']
    max_exit = t['max_exit_idx']
    start = max(0, entry_idx - 15)
    end = min(len(df), max(entry_idx + 30, max_exit + 5))
    if end - start > 120:
        end = start + 120

    def rel_pos(idx):
        p = idx - start
        return p if 0 <= p < (end - start) else -1

    bars = []
    for i in range(start, end):
        row = df.iloc[i]
        bars.append([
            round(row['open'], 2), round(row['close'], 2),
            round(row['high'], 2), round(row['low'], 2),
            round(row['ema10'] if not pd.isna(row['ema10']) else 0, 2),
            round(row['ema20'] if not pd.isna(row['ema20']) else 0, 2),
            round(row['ema120'] if not pd.isna(row['ema120']) else 0, 2),
        ])

    return {
        'b': bars,
        'ep': entry_idx - start,
        'dir': t['direction'],
        'stop': round(t['init_stop'], 2),
        'sn': name,
        'tp': t['type'],
        'da': round(t['dev_atr'], 2),
        'er': round(t['er_20'], 2) if not pd.isna(t['er_20']) else -1,
        'p1': round(t['s1_pnl'], 2), 'x1': rel_pos(t['s1_exit_idx']), 'xp1': round(t['s1_exit_price'], 2),
        'p2': round(t['s2_pnl'], 2), 'x2': rel_pos(t['s2_exit_idx']), 'xp2': round(t['s2_exit_price'], 2),
        'p3': round(t['s3_pnl'], 2), 'x3': rel_pos(t['s3_exit_idx']), 'xp3': round(t['s3_exit_price'], 2),
        'p4': round(t['s4_pnl'], 2),
        'tag': tag,
        'note': note,
    }

import json

charts = []

# 命中但亏
for x in selected_hit_lose:
    sym, name, df, t, sc = x
    scenario_names = {1: 'A+ER>=0.5+>=1.0ATR', 2: 'C+>=2.0ATR', 3: 'B+ER>=0.5+<0.3ATR'}
    note = f"场景{sc}({scenario_names[sc]}) 命中但全出场亏损"
    charts.append(make_chart_data(sym, name, df, t, 'hit_lose', note))

# 没命中但赚
for x in selected_miss_win:
    sym, name, df, t, sc = x
    # 分析为什么没命中
    reasons = []
    if t['type'] == 'A':
        if pd.isna(t['er_20']) or t['er_20'] < 0.5:
            reasons.append(f"ER={t['er_20']:.2f}<0.5")
        if t['dev_atr'] < 1.0:
            reasons.append(f"偏离={t['dev_atr']:.2f}<1.0ATR")
    elif t['type'] == 'B':
        if pd.isna(t['er_20']) or t['er_20'] < 0.5:
            reasons.append(f"ER={t['er_20']:.2f}<0.5")
        if t['dev_atr'] >= 0.5:
            reasons.append(f"偏离={t['dev_atr']:.2f}>=0.5ATR")
    elif t['type'] == 'C':
        if t['dev_atr'] < 2.0:
            reasons.append(f"偏离={t['dev_atr']:.2f}<2.0ATR")
    note = f"{t['type']}类未命中，原因: {', '.join(reasons)}，但最优出场+{best_exit_pnl(t):.2f}%"
    charts.append(make_chart_data(sym, name, df, t, 'miss_win', note))

data_json = json.dumps(charts, ensure_ascii=False)

html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>特殊案例K线</title>
<style>
body {{ background: #1a1a2e; color: #eee; font-family: 'Microsoft YaHei', sans-serif; padding: 20px; }}
h1 {{ color: #e94560; text-align: center; }}
h2 {{ color: #0ff; margin-top: 40px; border-bottom: 1px solid #0ff; padding-bottom: 8px; }}
.chart-box {{ background: #16213e; border-radius: 8px; padding: 15px; margin: 15px 0; border: 1px solid #333; }}
.info {{ color: #aaa; font-size: 13px; margin-bottom: 5px; }}
.info span {{ margin-right: 12px; }}
.note {{ color: #ffa500; font-size: 12px; margin-bottom: 8px; padding: 4px 8px; background: rgba(255,165,0,0.1); border-radius: 4px; border-left: 3px solid #ffa500; }}
.note-miss {{ color: #4caf50; border-left-color: #4caf50; background: rgba(76,175,80,0.1); }}
.pos {{ color: #4caf50; }}
.neg {{ color: #f44336; }}
.lbl {{ color: #888; }}
</style></head><body>
<h1>特殊案例 - 命中但亏 vs 漏掉但赚</h1>

<h2>命中规则但亏钱（3单）</h2>
<p style="color:#aaa;font-size:13px;">符合场景条件，但三种出场方式都亏损的交易</p>
<div id="hit_lose"></div>

<h2>没命中规则但赚钱（3单）</h2>
<p style="color:#aaa;font-size:13px;">不满足任何场景条件，但实际走出大行情的交易</p>
<div id="miss_win"></div>

<script>
const DATA = {data_json};

function drawChart(canvas, d) {{
  const ctx = canvas.getContext('2d');
  const W = 900, H = 350;
  canvas.width = W; canvas.height = H;
  const pl=60, pr=20, pt=25, pb=25;
  const cw = W-pl-pr, ch = H-pt-pb;
  const bars = d.b, n = bars.length;
  const bw = cw / n;

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

  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0,0,W,H);

  ctx.strokeStyle = '#222'; ctx.lineWidth = 0.5;
  ctx.fillStyle = '#666'; ctx.font = '10px sans-serif'; ctx.textAlign = 'right';
  for (let g=0; g<=5; g++) {{
    let gp = pMin + (pMax-pMin)*g/5;
    let gy = py(gp);
    ctx.beginPath(); ctx.moveTo(pl, gy); ctx.lineTo(W-pr, gy); ctx.stroke();
    ctx.fillText(gp.toFixed(1), pl-4, gy+3);
  }}

  let ex = bx(d.ep) - bw*0.5;
  ctx.fillStyle = 'rgba(255,165,0,0.15)';
  ctx.fillRect(ex, pt, bw, ch);

  for (let i=0; i<n; i++) {{
    const [o,c,h,l] = bars[i];
    const cx = bx(i);
    const up = c >= o;
    ctx.strokeStyle = ctx.fillStyle = up ? '#26a69a' : '#ef5350';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx, py(h)); ctx.lineTo(cx, py(l)); ctx.stroke();
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

  // exit markers
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
    ctx.strokeStyle = e.color; ctx.lineWidth = 2; ctx.globalAlpha = 0.9;
    ctx.beginPath(); ctx.moveTo(ex2-4, epy-4); ctx.lineTo(ex2+4, epy+4); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(ex2+4, epy-4); ctx.lineTo(ex2-4, epy+4); ctx.stroke();
    ctx.fillStyle = e.color; ctx.globalAlpha = 1;
    const labelY = d.dir==='long' ? epy+16+ei*11 : epy-10-ei*11;
    ctx.fillText(e.label, ex2, labelY);
  }}
  ctx.globalAlpha = 1;

  // legend
  ctx.font = '10px sans-serif'; ctx.textAlign = 'left';
  ctx.fillStyle = '#ffeb3b'; ctx.fillText('EMA10', pl+5, pt+12);
  ctx.fillStyle = '#2196f3'; ctx.fillText('EMA20', pl+50, pt+12);
  ctx.fillStyle = '#e040fb'; ctx.fillText('EMA120', pl+100, pt+12);
  ctx.fillStyle = '#ff6b6b'; ctx.fillText('S1', pl+160, pt+12);
  ctx.fillStyle = '#ffd93d'; ctx.fillText('S2', pl+180, pt+12);
  ctx.fillStyle = '#6bcb77'; ctx.fillText('S3', pl+200, pt+12);
}}

function pnlSpan(v) {{
  const cls = v >= 0 ? 'pos' : 'neg';
  return '<span class="'+cls+'">'+(v>=0?'+':'')+v.toFixed(2)+'%</span>';
}}

function renderGroup(containerId, tag) {{
  const el = document.getElementById(containerId);
  const items = DATA.filter(d => d.tag === tag);
  for (const d of items) {{
    const id = 'c'+Math.random().toString(36).substr(2,8);
    const er = d.er >= 0 ? d.er.toFixed(2) : 'N/A';
    const noteClass = tag === 'miss_win' ? 'note note-miss' : 'note';
    el.innerHTML += '<div class="chart-box">' +
      '<div class="'+noteClass+'">'+d.note+'</div>' +
      '<div class="info">' +
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

renderGroup('hit_lose', 'hit_lose');
renderGroup('miss_win', 'miss_win');
</script>
</body></html>"""

out_path = os.path.join(OUTPUT_DIR, "deviation_kline_special.html")
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"\n输出: {out_path}")
