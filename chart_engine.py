# -*- coding: utf-8 -*-
"""
K线图渲染引擎（Canvas）
======================
通用 K 线 + 指标 + 入场/出场标注 渲染模块。
生成 HTML 片段，嵌入到 report_engine.py 的报告中。

用法:
    from chart_engine import render_chart, get_chart_js

    html = render_chart(df, entry_idx=150, direction='long',
                        exits=[{'name':'S1','idx':165,'price':5800}],
                        stop_price=5650, title='白银 B long')

原理:
    - get_chart_js() 返回一段 JS，定义全局 drawChart() 函数，页面引入一次
    - render_chart() 返回 <div> 包含 <canvas> + <script>，调用 drawChart()
    - 数据以紧凑 JSON 嵌入，100根K线约 3~5KB
"""

import json
import re
import uuid
import pandas as pd

# ============ 颜色常量 ============

EMA_COLORS = {
    'ema10':  '#FF9800',   # 橙
    'ema20':  '#2196F3',   # 蓝
    'ema60':  '#E91E63',   # 粉
    'ema120': '#9C27B0',   # 紫
}

EXIT_COLORS = {
    'S1': '#e74c3c',
    'S2': '#2ecc71',
    'S3': '#f39c12',
    'S4': '#3498db',
}

BULL_COLOR = '#ef5350'   # 红涨（中国市场惯例）
BEAR_COLOR = '#26a69a'   # 绿跌
BG_COLOR = '#0d1117'


def _auto_detect_emas(df):
    """自动检测 DataFrame 中的 ema 列"""
    emas = []
    for col in df.columns:
        m = re.match(r'^ema(\d+)$', col)
        if m:
            period = int(m.group(1))
            color = EMA_COLORS.get(col, '#888888')
            emas.append((col, f'EMA{period}', color))
    emas.sort(key=lambda x: x[0])
    return emas


def prepare_chart_data(df, entry_idx, direction='long',
                       before_bars=30, after_bars=40,
                       exits=None, stop_price=None,
                       ema_cols=None, title='', extra_info=None,
                       max_bars=200):
    """
    从 DataFrame 提取紧凑的图表数据 dict。

    参数:
      df: 含 datetime/OHLC/ema列 的 DataFrame
      entry_idx: 入场K线在 df 中的 index 位置（iloc索引）
      direction: 'long' or 'short'
      before_bars: 入场前显示多少根K线
      after_bars: 入场后显示多少根K线
      exits: 出场标注列表 [{'name':'S1', 'idx':int, 'price':float, 'color':str(可选)}, ...]
             idx 是在 df 中的 iloc 索引
      stop_price: 止损价（画水平虚线）
      ema_cols: 要画的EMA列名列表，如 ['ema10','ema20','ema120']。None=自动检测。
      title: 图表标题
      extra_info: 右上角显示的 {key: value} 字典
      max_bars: 最多显示的K线数
    返回:
      dict — 可 json 序列化，传给 JS drawChart()
    """
    n = len(df)

    # 计算显示范围
    start = max(0, entry_idx - before_bars)
    end = min(n, entry_idx + after_bars)

    # 如果有出场标注，确保范围能覆盖
    if exits:
        max_exit_idx = max(e['idx'] for e in exits)
        end = min(n, max(end, max_exit_idx + 5))

    # 限制最大K线数
    if end - start > max_bars:
        end = start + max_bars

    segment = df.iloc[start:end]
    entry_pos = entry_idx - start  # 相对位置

    # OHLC 数据
    bars = []
    for _, row in segment.iterrows():
        bars.append([
            round(row['open'], 2),
            round(row['close'], 2),
            round(row['high'], 2),
            round(row['low'], 2),
        ])

    # 时间标签
    times = []
    if 'datetime' in segment.columns:
        for _, row in segment.iterrows():
            dt = row['datetime']
            if hasattr(dt, 'strftime'):
                times.append(dt.strftime('%m-%d %H:%M'))
            else:
                times.append(str(dt)[:16])

    # EMA 数据
    if ema_cols is None:
        ema_info = _auto_detect_emas(df)
    else:
        ema_info = []
        for col in ema_cols:
            m = re.match(r'^ema(\d+)$', col)
            label = f'EMA{m.group(1)}' if m else col
            color = EMA_COLORS.get(col, '#888888')
            ema_info.append((col, label, color))

    emas_data = []
    for col, label, color in ema_info:
        if col in segment.columns:
            vals = []
            for v in segment[col]:
                if pd.isna(v):
                    vals.append(0)
                else:
                    vals.append(round(float(v), 2))
            emas_data.append({'v': vals, 'c': color, 'n': label})

    # 出场标注
    exits_data = []
    if exits:
        for e in exits:
            rel_idx = e['idx'] - start
            if 0 <= rel_idx < len(bars):
                color = e.get('color', EXIT_COLORS.get(e['name'], '#ffffff'))
                exits_data.append({
                    'pos': rel_idx,
                    'price': round(e['price'], 2),
                    'name': e['name'],
                    'color': color,
                })

    data = {
        'bars': bars,
        'times': times,
        'emas': emas_data,
        'ep': entry_pos,
        'dir': direction,
        'stop': round(stop_price, 2) if stop_price is not None else None,
        'exits': exits_data,
        'title': title,
        'info': extra_info or {},
    }
    return data


def render_chart(df, entry_idx, direction='long',
                 before_bars=30, after_bars=40,
                 exits=None, stop_price=None,
                 ema_cols=None, title='', extra_info=None,
                 width=900, height=320, chart_id=None):
    """
    渲染单个K线图，返回 HTML 片段 <div>。

    需要页面已包含 get_chart_js() 的输出。
    """
    data = prepare_chart_data(
        df, entry_idx, direction, before_bars, after_bars,
        exits, stop_price, ema_cols, title, extra_info,
    )

    if chart_id is None:
        chart_id = 'c_' + uuid.uuid4().hex[:8]

    data_json = json.dumps(data, ensure_ascii=False, separators=(',', ':'))

    return (
        f'<div class="chart-box">'
        f'<canvas id="{chart_id}"></canvas>'
        f'<script>drawChart(document.getElementById("{chart_id}"),'
        f'{data_json},{{w:{width},h:{height}}});</script>'
        f'</div>'
    )


def render_batch(charts_data, width=900, height=320):
    """
    批量渲染多个图表（charts_data 是 prepare_chart_data 的返回值列表）。
    使用 setTimeout 避免阻塞。
    """
    parts = []
    ids = []
    for i, data in enumerate(charts_data):
        cid = f'cb_{i}'
        ids.append(cid)
        data_json = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
        parts.append(
            f'<div class="chart-box">'
            f'<canvas id="{cid}"></canvas>'
            f'</div>'
        )
    # 延迟渲染脚本
    script_lines = []
    for i, (cid, data) in enumerate(zip(ids, charts_data)):
        data_json = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
        script_lines.append(
            f'setTimeout(function(){{drawChart(document.getElementById("{cid}"),'
            f'{data_json},{{w:{width},h:{height}}});}},{i * 5});'
        )
    parts.append('<script>' + '\n'.join(script_lines) + '</script>')
    return '\n'.join(parts)


def get_chart_js():
    """
    返回通用 drawChart(canvas, d, opts) 的 JavaScript 代码。
    在 HTML 页面中引入一次即可。
    """
    return r"""
function drawChart(canvas, d, opts) {
    opts = opts || {};
    var W = opts.w || 900, H = opts.h || 320;
    var dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    var ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    var pl = 62, pr = 12, pt = 24, pb = 24;
    var cw = W - pl - pr, ch = H - pt - pb;
    var bars = d.bars, n = bars.length;
    if (n === 0) return;
    var bw = cw / n;

    // 1. 计算价格范围（OHLC + EMA + stop + exits）
    var pMax = -1e18, pMin = 1e18;
    for (var i = 0; i < n; i++) {
        if (bars[i][2] > pMax) pMax = bars[i][2]; // high
        if (bars[i][3] < pMin) pMin = bars[i][3]; // low
    }
    if (d.emas) {
        for (var e = 0; e < d.emas.length; e++) {
            var ev = d.emas[e].v;
            for (var j = 0; j < ev.length; j++) {
                if (ev[j] > 0) {
                    if (ev[j] > pMax) pMax = ev[j];
                    if (ev[j] < pMin) pMin = ev[j];
                }
            }
        }
    }
    if (d.stop != null) {
        if (d.stop > pMax) pMax = d.stop;
        if (d.stop < pMin) pMin = d.stop;
    }
    if (d.exits) {
        for (var x = 0; x < d.exits.length; x++) {
            if (d.exits[x].price > pMax) pMax = d.exits[x].price;
            if (d.exits[x].price < pMin) pMin = d.exits[x].price;
        }
    }
    var pRange = pMax - pMin;
    pMax += pRange * 0.05;
    pMin -= pRange * 0.05;
    pRange = pMax - pMin;
    if (pRange <= 0) pRange = 1;

    function py(p) { return pt + ch * (1 - (p - pMin) / pRange); }
    function bx(i) { return pl + bw * (i + 0.5); }

    // 2. 背景
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, W, H);

    // 3. 网格线 + 价格标签
    ctx.strokeStyle = '#21262d';
    ctx.lineWidth = 0.5;
    ctx.fillStyle = '#6e7681';
    ctx.font = '10px Consolas, monospace';
    ctx.textAlign = 'right';
    for (var g = 0; g <= 4; g++) {
        var gp = pMin + pRange * g / 4;
        var gy = py(gp);
        ctx.beginPath(); ctx.moveTo(pl, gy); ctx.lineTo(W - pr, gy); ctx.stroke();
        ctx.fillText(gp.toFixed(2), pl - 4, gy + 3);
    }

    // 4. 信号K线高亮列
    if (d.ep >= 0 && d.ep < n) {
        ctx.fillStyle = 'rgba(255,152,0,0.08)';
        ctx.fillRect(pl + bw * d.ep, pt, bw, ch);
    }

    // 5. K线
    var bodyW = Math.max(bw * 0.6, 1);
    for (var i = 0; i < n; i++) {
        var o = bars[i][0], c = bars[i][1], h = bars[i][2], l = bars[i][3];
        var bull = c >= o;
        var color = bull ? '#ef5350' : '#26a69a';
        var x = bx(i);
        // 影线
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, py(h)); ctx.lineTo(x, py(l)); ctx.stroke();
        // 实体
        var yTop = py(Math.max(o, c));
        var yBot = py(Math.min(o, c));
        var bh = Math.max(yBot - yTop, 1);
        ctx.fillStyle = color;
        ctx.fillRect(x - bodyW/2, yTop, bodyW, bh);
    }

    // 6. EMA 线
    if (d.emas) {
        for (var e = 0; e < d.emas.length; e++) {
            var ema = d.emas[e];
            ctx.strokeStyle = ema.c;
            ctx.lineWidth = 1.2;
            ctx.globalAlpha = 0.85;
            ctx.beginPath();
            var started = false;
            for (var j = 0; j < ema.v.length; j++) {
                if (ema.v[j] > 0) {
                    var ex = bx(j), ey = py(ema.v[j]);
                    if (!started) { ctx.moveTo(ex, ey); started = true; }
                    else ctx.lineTo(ex, ey);
                }
            }
            ctx.stroke();
            ctx.globalAlpha = 1.0;
        }
    }

    // 7. 入场标记（三角形）
    if (d.ep >= 0 && d.ep < n) {
        var isLong = d.dir === 'long';
        var eBar = bars[d.ep];
        var eColor = isLong ? '#4caf50' : '#f44336';
        var ePrice = isLong ? eBar[3] : eBar[2]; // long在low下方，short在high上方
        var eY = py(ePrice);
        var triH = 10;
        var triBase = 6;
        ctx.fillStyle = eColor;
        ctx.beginPath();
        if (isLong) {
            eY += 4;
            ctx.moveTo(bx(d.ep), eY + triH);
            ctx.lineTo(bx(d.ep) - triBase, eY + triH + triH);
            ctx.lineTo(bx(d.ep) + triBase, eY + triH + triH);
        } else {
            eY -= 4;
            ctx.moveTo(bx(d.ep), eY - triH);
            ctx.lineTo(bx(d.ep) - triBase, eY - triH - triH);
            ctx.lineTo(bx(d.ep) + triBase, eY - triH - triH);
        }
        ctx.closePath();
        ctx.fill();

        // 入场价标注
        ctx.fillStyle = eColor;
        ctx.font = '10px Microsoft YaHei';
        ctx.textAlign = 'left';
        var entryPrice = eBar[1]; // close
        ctx.fillText('▸' + entryPrice.toFixed(2), bx(d.ep) + triBase + 2,
                      isLong ? eY + triH + 8 : eY - triH - 4);
    }

    // 8. 止损虚线
    if (d.stop != null) {
        var sy = py(d.stop);
        ctx.strokeStyle = '#f44336';
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.6;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(d.ep >= 0 ? bx(d.ep) : pl, sy);
        ctx.lineTo(W - pr, sy);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1.0;
        // 标签
        ctx.fillStyle = '#f44336';
        ctx.font = '9px Consolas';
        ctx.textAlign = 'right';
        ctx.fillText('SL ' + d.stop.toFixed(2), W - pr - 2, sy - 3);
    }

    // 9. 出场标记（X + 标签）
    if (d.exits) {
        for (var x = 0; x < d.exits.length; x++) {
            var ex = d.exits[x];
            var xPos = bx(ex.pos);
            var yPos = py(ex.price);
            var sz = 5;
            ctx.strokeStyle = ex.color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(xPos - sz, yPos - sz); ctx.lineTo(xPos + sz, yPos + sz);
            ctx.moveTo(xPos + sz, yPos - sz); ctx.lineTo(xPos - sz, yPos + sz);
            ctx.stroke();
            // 标签（错开避免重叠）
            ctx.fillStyle = ex.color;
            ctx.font = '9px Microsoft YaHei';
            ctx.textAlign = 'center';
            var labelY = yPos - sz - 4 - x * 12;
            ctx.fillText(ex.name + ' ' + ex.price.toFixed(2), xPos, labelY);
        }
    }

    // 10. 时间标签（X轴，间隔显示）
    if (d.times && d.times.length > 0) {
        ctx.fillStyle = '#6e7681';
        ctx.font = '9px Consolas';
        ctx.textAlign = 'center';
        var step = Math.max(1, Math.floor(n / 8));
        for (var t = 0; t < n; t += step) {
            ctx.fillText(d.times[t], bx(t), H - 4);
        }
    }

    // 11. 图例（左上角）
    if (d.emas && d.emas.length > 0) {
        var lx = pl + 6, ly = pt + 12;
        ctx.font = '10px Consolas';
        for (var e = 0; e < d.emas.length; e++) {
            ctx.fillStyle = d.emas[e].c;
            ctx.fillText('— ' + d.emas[e].n, lx + e * 72, ly);
        }
    }

    // 12. 标题（顶部居中）
    if (d.title) {
        ctx.fillStyle = '#e6edf3';
        ctx.font = 'bold 12px Microsoft YaHei';
        ctx.textAlign = 'center';
        ctx.fillText(d.title, W / 2, 14);
    }

    // 13. 信息面板（右上角）
    if (d.info) {
        ctx.font = '10px Microsoft YaHei';
        ctx.textAlign = 'right';
        var ix = W - pr - 4, iy = pt + 14;
        var keys = Object.keys(d.info);
        for (var k = 0; k < keys.length; k++) {
            var val = '' + d.info[keys[k]];
            // PnL 颜色判定
            if (val.indexOf('+') === 0 || parseFloat(val) > 0) ctx.fillStyle = '#3fb950';
            else if (val.indexOf('-') === 0 || parseFloat(val) < 0) ctx.fillStyle = '#f85149';
            else ctx.fillStyle = '#8b949e';
            ctx.fillText(keys[k] + ': ' + val, ix, iy + k * 14);
        }
    }
}
"""
