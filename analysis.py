# -*- coding: utf-8 -*-
"""
统计分析 + HTML报告生成
读取trades.csv，7个课题 × 5种出场交叉分析
输出 report.html
"""

import os
import math
import pandas as pd
import numpy as np
from datetime import datetime
import config as cfg

EXIT_METHODS = ['newhigh', 'newhigh_prev', 'pullback', 'ma10', 'ma20']
EXIT_NAMES = {
    'newhigh': '新高K线止损',
    'newhigh_prev': '新高前根止损',
    'pullback': '回调低点止损',
    'ma10': '破10MA',
    'ma20': '破20MA',
}


def load_data():
    """加载trades.csv和meta.csv"""
    base = os.path.dirname(__file__)
    csv_path = os.path.join(base, cfg.OUTPUT_DIR, cfg.CSV_FILENAME)
    meta_path = os.path.join(base, cfg.OUTPUT_DIR, 'meta.csv')

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df['entry_time'] = pd.to_datetime(df['entry_time'])

    meta = pd.read_csv(meta_path)
    buy_hold = meta['buy_hold_return'].iloc[0]

    return df, buy_hold


def calc_stats(pnl_series):
    """计算统计指标"""
    n = len(pnl_series)
    if n == 0:
        return {'trades': 0, 'cum_pnl': 0, 'win_rate': 0, 'profit_factor': 0, 'max_dd': 0, 'avg_hold': 0}

    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series <= 0]

    cum_pnl = pnl_series.sum()
    win_rate = len(wins) / n * 100 if n > 0 else 0

    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.001
    profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')

    # 最大回撤
    cum_curve = pnl_series.cumsum()
    peak = cum_curve.cummax()
    drawdown = cum_curve - peak
    max_dd = drawdown.min()

    return {
        'trades': n,
        'cum_pnl': round(cum_pnl, 2),
        'win_rate': round(win_rate, 1),
        'profit_factor': round(profit_factor, 2),
        'max_dd': round(max_dd, 2),
    }


def calc_stats_with_hold(df_subset, exit_method):
    """计算统计指标，包含平均持仓K线数"""
    pnl_col = f'exit_{exit_method}_pnl'
    hold_col = f'hold_bars_{exit_method}'
    reason_col = f'exit_{exit_method}_reason'

    # 过滤掉backtest_end的交易
    mask = df_subset[reason_col] != 'backtest_end'
    df_valid = df_subset[mask]

    pnl = df_valid[pnl_col]
    stats = calc_stats(pnl)

    if hold_col in df_valid.columns and len(df_valid) > 0:
        stats['avg_hold'] = round(df_valid[hold_col].mean(), 1)
    else:
        stats['avg_hold'] = 0

    # 也算上backtest_end的（用于参考）
    stats['total_trades'] = len(df_subset)
    stats['backtest_end_count'] = len(df_subset) - len(df_valid)

    return stats


def generate_report(df, buy_hold):
    """生成HTML报告"""
    periods = sorted(df['period'].unique(), key=lambda x: int(x.replace('min', '')))

    # ===== 课题定义 =====
    topics = [
        {
            'name': '信号次序',
            'desc': '本轮趋势中第几次回调信号',
            'col': 'signal_seq',
            'groups': [
                ('第1次', lambda x: x == 1),
                ('第2-3次', lambda x: 2 <= x <= 3),
                ('第4次+', lambda x: x >= 4),
            ]
        },
        {
            'name': '距100MA(ATR)',
            'desc': '价格距100MA的ATR倍数',
            'col': 'dist_100ma_atr',
            'groups': [
                ('≤1倍ATR', lambda x: x <= 1),
                ('1-2倍ATR', lambda x: 1 < x <= 2),
                ('>2倍ATR', lambda x: x > 2),
            ]
        },
        {
            'name': '回调K线数',
            'desc': '价格跌破10MA后几根K线收回',
            'col': 'pullback_bars',
            'groups': [
                ('0根(影线)', lambda x: x == 0),
                ('1-3根', lambda x: 1 <= x <= 3),
                ('4+根', lambda x: x >= 4),
            ]
        },
        {
            'name': '回调强度',
            'desc': '回调幅度%÷回调K线数（越大=越急）。A类信号(回调0根)不参与此课题',
            'col': 'pullback_intensity',
            'groups': 'median',  # 特殊标记：按中位数切
        },
        {
            'name': '均线宽度',
            'desc': '10MA与20MA的距离(%)',
            'col': 'ma_width_pct',
            'groups': [
                ('<0.5%', lambda x: x < 0.5),
                ('≥0.5%', lambda x: x >= 0.5),
            ]
        },
        {
            'name': '入场类型',
            'desc': 'A=影线触碰回弹 / B=实体破位后收回',
            'col': 'entry_type',
            'groups': [
                ('A类(影线)', lambda x: x == 'A'),
                ('B类(实体)', lambda x: x == 'B'),
            ]
        },
    ]

    # 多周期共振单独处理（因为字段不同）
    resonance_topics = []
    if 'tf_10min_aligned' in df.columns:
        resonance_topics.append({
            'name': '共振:10min同向',
            'desc': '信号触发时10min趋势是否同向',
            'col': 'tf_10min_aligned',
            'groups': [
                ('同向', lambda x: x == True or x == 'True'),
                ('不同向', lambda x: x == False or x == 'False'),
            ],
            'filter_period': '2min',
        })
    if 'tf_60min_aligned' in df.columns:
        resonance_topics.append({
            'name': '共振:60min同向',
            'desc': '信号触发时60min趋势是否同向',
            'col': 'tf_60min_aligned',
            'groups': [
                ('同向', lambda x: x == True or x == 'True'),
                ('不同向', lambda x: x == False or x == 'False'),
            ],
        })

    all_topics = topics + resonance_topics

    # ===== 构建HTML =====
    html = _html_head()

    # 总览Tab
    html += '<div class="sec"><h2>总览</h2>'
    html += '<div class="tbar">'
    for i, p in enumerate(periods):
        cls = ' on' if i == 0 else ''
        html += f'<button class="tb{cls}" onclick="showTab(\'{p}\')" id="btn_{p}">{p}</button>'
    html += '<button class="tb" onclick="showTab(\'all\')" id="btn_all">全部</button>'
    html += '</div>'

    for tab_period in periods + ['all']:
        display = 'block' if tab_period == periods[0] else 'none'
        html += f'<div class="tab-panel" id="tab_{tab_period}" style="display:{display}">'

        if tab_period == 'all':
            df_tab = df
        else:
            df_tab = df[df['period'] == tab_period]

        # 5种出场对比表（主指标：期望值）
        html += '<h3>5种出场方式对比</h3>'
        html += '<table><thead><tr><th>出场方式</th><th>交易数</th><th>期望值</th><th>胜率%</th><th>盈亏比</th><th>累计盈亏%</th><th>最大回撤%</th><th>平均持仓</th></tr></thead><tbody>'

        best_ev = -999
        best_method = ''
        for m in EXIT_METHODS:
            stats = calc_stats_with_hold(df_tab, m)
            wr = stats['win_rate'] / 100
            ev = wr * stats['profit_factor'] - (1 - wr) if stats['trades'] > 0 else -999
            if ev > best_ev and stats['trades'] >= 10:
                best_ev = ev
                best_method = m

        for m in EXIT_METHODS:
            stats = calc_stats_with_hold(df_tab, m)
            wr = stats['win_rate'] / 100
            ev = wr * stats['profit_factor'] - (1 - wr) if stats['trades'] > 0 else 0
            bg = ' style="background:#1a3a2a"' if m == best_method else ''
            ev_color = '#10b981' if ev > 0.3 else ('#f59e0b' if ev > 0 else '#ef4444')
            pnl_color = '#10b981' if stats['cum_pnl'] >= 0 else '#ef4444'
            dd_color = '#ef4444' if stats['max_dd'] < -5 else '#94a3b8'
            be_note = f" (+{stats['backtest_end_count']}未平)" if stats['backtest_end_count'] > 0 else ''
            html += f'<tr{bg}><td><b>{EXIT_NAMES[m]}</b></td>'
            html += f'<td>{stats["trades"]}{be_note}</td>'
            html += f'<td style="color:{ev_color};font-weight:700;font-size:16px">{ev:+.2f}</td>'
            html += f'<td>{stats["win_rate"]}</td>'
            html += f'<td>{stats["profit_factor"]}</td>'
            html += f'<td style="color:{pnl_color};font-size:11px">{stats["cum_pnl"]:+.2f}</td>'
            html += f'<td style="color:{dd_color};font-size:11px">{stats["max_dd"]:.2f}</td>'
            html += f'<td>{stats["avg_hold"]}根</td></tr>'

        # 买入持有基准行
        html += f'<tr style="border-top:2px solid #475569"><td><i>买入持有基准</i></td><td>-</td><td>-</td><td>-</td><td>-</td>'
        bh_color = '#10b981' if buy_hold >= 0 else '#ef4444'
        html += f'<td style="color:{bh_color};font-size:11px">{buy_hold:+.2f}</td><td>-</td><td>-</td></tr>'
        html += '</tbody></table>'

        # 去重版对照
        df_no_overlap = df_tab[df_tab['overlap_count'] == 0]
        overlap_pct = (1 - len(df_no_overlap) / len(df_tab)) * 100 if len(df_tab) > 0 else 0
        html += f'<p class="desc" style="margin-top:10px">重叠交易占比: {overlap_pct:.1f}% | 去重版（仅首笔）:</p>'
        html += '<table><thead><tr><th>出场方式</th><th>交易数</th><th>期望值</th><th>胜率%</th><th>盈亏比</th></tr></thead><tbody>'
        for m in EXIT_METHODS:
            stats = calc_stats_with_hold(df_no_overlap, m)
            wr = stats['win_rate'] / 100
            ev = wr * stats['profit_factor'] - (1 - wr) if stats['trades'] > 0 else 0
            ev_color = '#10b981' if ev > 0.3 else ('#f59e0b' if ev > 0 else '#ef4444')
            html += f'<tr><td>{EXIT_NAMES[m]}</td><td>{stats["trades"]}</td>'
            html += f'<td style="color:{ev_color};font-weight:700">{ev:+.2f}</td>'
            html += f'<td>{stats["win_rate"]}</td>'
            html += f'<td>{stats["profit_factor"]}</td></tr>'
        html += '</tbody></table>'

        html += '</div>'  # tab-panel

    html += '</div>'  # sec

    # ===== 课题板块 =====
    for topic in all_topics:
        html += _render_topic(df, topic, periods)

    # ===== 累计盈亏曲线 =====
    html += _render_equity_curve(df)

    html += _html_foot()
    return html


def _render_topic(df, topic, periods):
    """渲染单个课题板块"""
    html = f'<div class="sec"><h2>{topic["name"]}</h2>'
    html += f'<p class="desc">{topic["desc"]}</p>'

    # 如果课题有period过滤
    filter_period = topic.get('filter_period')

    for tab_period in periods + ['all']:
        if tab_period == 'all':
            df_p = df
        else:
            df_p = df[df['period'] == tab_period]

        if filter_period and tab_period != 'all':
            if tab_period != filter_period:
                continue
        if filter_period and tab_period == 'all':
            df_p = df[df['period'] == filter_period]

        if len(df_p) == 0:
            continue

        html += f'<h3 style="margin-top:10px">{tab_period if tab_period != "all" else "全部"}</h3>'

        # 确定分组
        col = topic['col']
        if col not in df_p.columns:
            html += '<p class="desc">该周期无此数据</p>'
            continue

        groups = topic['groups']
        if groups == 'median':
            # 按中位数切
            valid = df_p[col].dropna()
            if len(valid) < 4:
                html += '<p class="desc">样本不足</p>'
                continue
            med = valid.median()
            groups = [
                (f'≤{med:.4f}(温和)', lambda x, m=med: x <= m),
                (f'>{med:.4f}(急跌)', lambda x, m=med: x > m),
            ]

        # 表头
        html += '<table><thead><tr><th>分组</th><th>笔数</th>'
        for m in EXIT_METHODS:
            html += f'<th>{EXIT_NAMES[m]}</th>'
        html += '</tr></thead><tbody>'

        # 找最佳格子（按期望值）
        best_val = -999
        best_pos = None

        group_data = []
        for gname, gfunc in groups:
            mask = df_p[col].apply(lambda x: False if pd.isna(x) else gfunc(x))
            df_g = df_p[mask]
            row_data = {'name': gname, 'count': len(df_g), 'stats': {}}
            for m in EXIT_METHODS:
                stats = calc_stats_with_hold(df_g, m)
                row_data['stats'][m] = stats
                wr = stats['win_rate'] / 100
                ev = wr * stats['profit_factor'] - (1 - wr) if stats['trades'] > 0 else -999
                if ev > best_val and stats['trades'] >= 10:
                    best_val = ev
                    best_pos = (gname, m)
            group_data.append(row_data)

        # 排除笔数说明
        total_in_topic = sum(r['count'] for r in group_data)
        total_in_period = len(df_p)
        if total_in_topic < total_in_period:
            excluded = total_in_period - total_in_topic
            html += f'<p class="desc" style="color:#f59e0b">⚠ {excluded}笔交易因数据缺失未参与此课题（如A类信号无回调强度）</p>'

        for row in group_data:
            html += f'<tr><td><b>{row["name"]}</b></td><td>{row["count"]}</td>'
            for m in EXIT_METHODS:
                s = row['stats'][m]
                is_best = best_pos and best_pos[0] == row['name'] and best_pos[1] == m
                bg = 'background:#1a3a2a;' if is_best else ''
                gray = 'opacity:0.4;' if s['trades'] < 10 else ''
                # 期望值 = 胜率×盈亏比 - (1-胜率)
                wr = s['win_rate'] / 100
                ev = wr * s['profit_factor'] - (1 - wr) if s['trades'] > 0 else 0
                ev_color = '#10b981' if ev > 0.3 else ('#f59e0b' if ev > 0 else '#ef4444')
                title = f"笔数:{s['trades']} | 累计:{s['cum_pnl']:+.2f}%"
                html += f'<td style="{bg}{gray}" title="{title}">'
                html += f'<span style="color:{ev_color};font-weight:700;font-size:15px">{ev:+.2f}</span>'
                html += f'<br><span style="font-size:10px;color:#94a3b8">胜率{s["win_rate"]}% 盈亏比{s["profit_factor"]}</span>'
                html += f'</td>'
            html += '</tr>'

        html += '</tbody></table>'

    html += '</div>'
    return html


def _render_equity_curve(df):
    """渲染累计盈亏曲线（用JS Canvas）"""
    html = '<div class="sec"><h2>累计盈亏曲线</h2>'
    html += '<canvas id="eqChart" width="1160" height="350" style="width:100%;background:#0f172a;border-radius:8px"></canvas>'

    # 准备数据：按entry_time排序
    df_sorted = df.sort_values('entry_time').reset_index(drop=True)

    curves = {}
    for m in EXIT_METHODS:
        reason_col = f'exit_{m}_reason'
        pnl_col = f'exit_{m}_pnl'
        valid = df_sorted[df_sorted[reason_col] != 'backtest_end']
        cum = valid[pnl_col].cumsum().tolist()
        curves[m] = cum

    # 输出为JS数组
    html += '<script>'
    html += 'const curves = {'
    colors = {'newhigh': '#3b82f6', 'newhigh_prev': '#8b5cf6',
              'pullback': '#10b981', 'ma10': '#f59e0b', 'ma20': '#ef4444'}
    for m in EXIT_METHODS:
        data_str = ','.join(f'{v:.2f}' for v in curves[m])
        html += f'"{m}":[{data_str}],'
    html += '};\n'
    color_items = ','.join(f'"{m}":"{c}"' for m, c in colors.items())
    html += f'const colors = {{{color_items}}};\n'
    name_items = ','.join(f'"{m}":"{EXIT_NAMES[m]}"' for m in EXIT_METHODS)
    html += f'const names = {{{name_items}}};\n'

    html += """
    const canvas = document.getElementById('eqChart');
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const pad = {t:30, b:40, l:60, r:20};

    // 找全局min/max
    let allVals = [];
    for(let m in curves) allVals.push(...curves[m]);
    if(allVals.length === 0) allVals = [0];
    let yMin = Math.min(0, ...allVals), yMax = Math.max(0, ...allVals);
    let yRange = yMax - yMin || 1;
    yMin -= yRange*0.05; yMax += yRange*0.05; yRange = yMax - yMin;

    let maxLen = Math.max(...Object.values(curves).map(c=>c.length));
    if(maxLen < 2) maxLen = 2;

    function toX(i){return pad.l + i/(maxLen-1)*(W-pad.l-pad.r)}
    function toY(v){return pad.t + (yMax-v)/yRange*(H-pad.t-pad.b)}

    // 网格
    ctx.strokeStyle='#1e293b'; ctx.lineWidth=1;
    for(let i=0;i<5;i++){
        let y=pad.t+i*(H-pad.t-pad.b)/4;
        ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();
        let val=yMax-i*yRange/4;
        ctx.fillStyle='#64748b';ctx.font='11px sans-serif';ctx.textAlign='right';
        ctx.fillText(val.toFixed(1)+'%',pad.l-5,y+4);
    }

    // 零线
    if(yMin<0 && yMax>0){
        ctx.strokeStyle='#475569';ctx.setLineDash([4,4]);
        ctx.beginPath();ctx.moveTo(pad.l,toY(0));ctx.lineTo(W-pad.r,toY(0));ctx.stroke();
        ctx.setLineDash([]);
    }

    // 画曲线
    for(let m in curves){
        let data=curves[m]; if(data.length<2) continue;
        ctx.strokeStyle=colors[m]; ctx.lineWidth=2; ctx.beginPath();
        for(let i=0;i<data.length;i++){
            let x=toX(i), y=toY(data[i]);
            i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
        }
        ctx.stroke();
    }

    // 图例
    let lx=pad.l+10, ly=pad.t+10;
    for(let m in curves){
        ctx.fillStyle=colors[m];ctx.fillRect(lx,ly,14,3);
        ctx.fillStyle='#e2e8f0';ctx.font='11px sans-serif';ctx.textAlign='left';
        let lastVal=curves[m].length>0?curves[m][curves[m].length-1]:0;
        ctx.fillText(names[m]+' ('+lastVal.toFixed(1)+'%)',lx+18,ly+4);
        ly+=16;
    }
    </script>"""

    html += '</div>'
    return html


def _html_head():
    return """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>回测研究报告</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0f172a;color:#e2e8f0;font-family:-apple-system,"Microsoft YaHei",sans-serif;padding:20px;max-width:1200px;margin:0 auto}
h1{font-size:22px;margin-bottom:18px} h2{font-size:16px;margin-bottom:10px;color:#f8fafc} h3{font-size:14px;margin-bottom:5px;color:#94a3b8}
.sec{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:14px;margin-bottom:14px}
.desc{color:#64748b;font-size:11px;margin-bottom:8px}
table{width:100%;border-collapse:collapse;font-size:13px;margin-bottom:8px}
th{text-align:left;padding:5px 7px;border-bottom:2px solid #334155;color:#94a3b8;font-size:11px}
td{padding:5px 7px;border-bottom:1px solid #252f3f}
tbody tr:hover{background:#334155}
td[title]{cursor:help}
.tbar{display:flex;gap:5px;margin-bottom:12px}
.tb{padding:6px 14px;border-radius:6px;border:1px solid #334155;background:#0f172a;color:#94a3b8;font-size:12px;cursor:pointer;font-weight:600}
.tb.on{background:#3b82f6;color:#fff;border-color:#3b82f6}
</style></head><body>
<h1>回测研究报告 — """ + cfg.SYMBOL_NAME + f" | {datetime.now().strftime('%Y-%m-%d %H:%M')}</h1>\n"


def _html_foot():
    return """
<script>
function showTab(p){
    document.querySelectorAll('.tab-panel').forEach(el=>el.style.display='none');
    document.querySelectorAll('.tb').forEach(el=>el.classList.remove('on'));
    document.getElementById('tab_'+p).style.display='block';
    document.getElementById('btn_'+p).classList.add('on');
}
</script>
</body></html>"""


def run():
    print("加载数据...")
    df, buy_hold = load_data()
    print(f"  共 {len(df)} 笔交易, 买入持有基准: {buy_hold:.2f}%")

    print("生成报告...")
    html = generate_report(df, buy_hold)

    out_path = os.path.join(os.path.dirname(__file__), cfg.OUTPUT_DIR, cfg.REPORT_FILENAME)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  报告已输出: {out_path}")


if __name__ == '__main__':
    run()
