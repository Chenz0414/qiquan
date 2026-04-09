# -*- coding: utf-8 -*-
"""
ER5/ER40走势深度研究：多分位扫描 + 底部减仓
=============================================
方向1: 从P50(前50%)扫到P95(前5%)，看EV是否单调递增
方向2: 底部P5~P25，看是否该减仓或跳过

输出: output/er_trend_deep.html
"""

import sys
import io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from data_loader import load_all, sym_name, tick_size
from signal_core import SignalDetector, ExitTracker, DEFAULT_STOP_TICKS
from stats_utils import calc_ev
from report_engine import Report


# ============================================================
#  复用：场景分类 + 指标定义
# ============================================================

def classify_scenario(sig_type, er20, deviation_atr):
    if er20 >= 0.7:
        return None
    if sig_type == 'A' and er20 >= 0.5 and deviation_atr >= 1.0:
        return 1
    if sig_type == 'C' and deviation_atr >= 2.0:
        return 2
    if sig_type == 'B' and er20 >= 0.5 and deviation_atr >= 0.1 and deviation_atr < 0.3:
        return 3
    return None


# 只研究非二值指标（二值的无法做多分位扫描）
METRICS = [
    {'col': 'er_5',          'name': 'ER(5)'},
    {'col': 'er_40',         'name': 'ER(40)'},
    {'col': 'er5_delta_6',   'name': 'ER5变化(6根)'},
    {'col': 'er40_delta_12', 'name': 'ER40变化(12根)'},
    {'col': 'er5_over_er20', 'name': 'ER5/ER20比值'},
]

SCENARIO_PNL = {1: 's2_pnl', 2: 's2_pnl', 3: 's51_pnl'}
SCENARIO_REASON = {1: 's2_reason', 2: 's2_reason', 3: 's51_reason'}
SCENARIO_NAMES = {1: 'A类 + ER>=0.5 + 偏离>=1.0ATR → S2',
                  2: 'C类 + 偏离>=2.0ATR → S2',
                  3: 'B类 + ER>=0.5 + 偏离0.1~0.3ATR → S5.1'}

# 扫描分位点：顶部（高置信加仓）和底部（低置信减仓）
TOP_PERCENTILES = [50, 60, 70, 75, 80, 85, 90, 95]   # P50=前50%, P95=前5%
BOT_PERCENTILES = [5, 10, 15, 20, 25]                  # P5=后5%, P25=后25%


# ============================================================
#  数据收集（复用 test_er_trend.py 的逻辑）
# ============================================================

def collect_signals():
    print("加载数据...")
    all_data = load_all(period_min=10, days=170, last_days=None,
                        emas=(10, 20, 120), er_periods=(5, 20, 40), atr_period=14)
    print(f"品种数: {len(all_data)}")

    for sym_key, df in all_data.items():
        df['er5_delta_6'] = df['er_5'] - df['er_5'].shift(6)
        df['er40_delta_12'] = df['er_40'] - df['er_40'].shift(12)
        er20_safe = df['er_20'].replace(0, np.nan)
        df['er5_over_er20'] = df['er_5'] / er20_safe

    records = []
    BARS_PER_DAY = 57
    LAST_DAYS = 120

    for sym_key, df in sorted(all_data.items()):
        n = len(df)
        signal_start = max(130, n - LAST_DAYS * BARS_PER_DAY)
        ts = tick_size(sym_key)
        name = sym_name(sym_key)
        detector = SignalDetector(signal_types='ABC')

        for i in range(max(2, 130), n):
            row = df.iloc[i]
            if pd.isna(row['ema10']) or pd.isna(row['ema20']) or pd.isna(row['ema120']):
                continue
            if pd.isna(row['atr']) or row['atr'] <= 0:
                continue

            result = detector.process_bar(
                close=row['close'], high=row['high'], low=row['low'],
                ema10=row['ema10'], ema20=row['ema20'], ema120=row['ema120'],
            )
            if result is None or i < signal_start:
                continue

            er20 = row.get('er_20', 0)
            if pd.isna(er20):
                er20 = 0
            atr = row['atr']
            deviation_atr = abs(result.entry_price - row['ema10']) / atr if atr > 0 else 0
            scenario = classify_scenario(result.signal_type, er20, deviation_atr)
            if scenario is None:
                continue

            tracker = ExitTracker(
                direction=result.direction, entry_price=result.entry_price,
                pullback_extreme=result.pullback_extreme, tick_size=ts,
                stop_ticks=DEFAULT_STOP_TICKS,
            )
            exit_results = {'S2': None, 'S5.1': None}
            for j in range(i + 1, n):
                bar = df.iloc[j]
                prev_bar = df.iloc[j - 1]
                if pd.isna(bar['ema10']):
                    continue
                if not tracker.all_done():
                    exit_events, _ = tracker.process_bar(
                        close=bar['close'], high=bar['high'], low=bar['low'],
                        ema10=bar['ema10'], prev_close=prev_bar['close'],
                        prev_high=prev_bar['high'], prev_low=prev_bar['low'],
                    )
                    for ev in exit_events:
                        if ev.strategy == 'S2' and exit_results['S2'] is None:
                            exit_results['S2'] = ev
                        elif ev.strategy == 'S5.1' and exit_results['S5.1'] is None:
                            exit_results['S5.1'] = ev
                if tracker.all_done():
                    break

            last_bar = df.iloc[-1]
            for ev in tracker.force_close(last_bar['close']):
                if ev.strategy == 'S2' and exit_results['S2'] is None:
                    exit_results['S2'] = ev
                elif ev.strategy == 'S5.1' and exit_results['S5.1'] is None:
                    exit_results['S5.1'] = ev

            s2 = exit_results['S2']
            s51 = exit_results['S5.1']

            def safe_val(key):
                v = row.get(key, np.nan)
                return round(v, 4) if not pd.isna(v) else 0

            records.append({
                'symbol': name, 'sym_key': sym_key, 'time': row['datetime'],
                'direction': result.direction, 'type': result.signal_type,
                'scenario': scenario, 'er_20': round(er20, 4), 'dev_atr': round(deviation_atr, 3),
                'er_5': safe_val('er_5'), 'er_40': safe_val('er_40'),
                'er5_delta_6': safe_val('er5_delta_6'), 'er40_delta_12': safe_val('er40_delta_12'),
                'er5_over_er20': safe_val('er5_over_er20'),
                's2_pnl': round(s2.pnl_pct, 4) if s2 else 0,
                's2_reason': s2.exit_reason if s2 else 'missing',
                's51_pnl': round(s51.pnl_pct, 4) if s51 else 0,
                's51_reason': s51.exit_reason if s51 else 'missing',
            })

    print(f"总信号: {len(records)}笔")
    return pd.DataFrame(records)


# ============================================================
#  多分位扫描核心
# ============================================================

def scan_percentiles(sc_df, metric_col, pnl_col, reason_col, percentiles, direction='top'):
    """
    对单个指标做多分位扫描。

    direction='top': 高于Pxx的为选中组（加仓研究）
    direction='bot': 低于Pxx的为选中组（减仓研究）

    返回: [{percentile, threshold, N_selected, N_rest, pct,
            ev_selected, wr_selected, pr_selected, sum_selected, avg_pnl_selected,
            ev_rest, wr_rest, ev_all, ...}, ...]
    """
    # 过滤backtest_end
    if reason_col in sc_df.columns:
        df = sc_df[sc_df[reason_col] != 'backtest_end'].copy()
    else:
        df = sc_df.copy()

    all_pnls = df[pnl_col].tolist()
    ev_all = calc_ev(all_pnls)
    values = df[metric_col].dropna()

    results = []
    for pct in percentiles:
        if direction == 'top':
            threshold = float(np.percentile(values, pct))
            mask = df[metric_col] >= threshold
        else:
            threshold = float(np.percentile(values, pct))
            mask = df[metric_col] <= threshold

        selected = df[mask]
        rest = df[~mask]

        pnls_sel = selected[pnl_col].tolist()
        pnls_rest = rest[pnl_col].tolist()

        ev_sel = calc_ev(pnls_sel)
        ev_rst = calc_ev(pnls_rest)

        # 最大单笔亏损
        max_loss_sel = min(pnls_sel) if pnls_sel else 0
        max_loss_rest = min(pnls_rest) if pnls_rest else 0

        # 实际占比
        actual_pct = len(pnls_sel) / len(all_pnls) * 100 if all_pnls else 0

        results.append({
            'percentile': pct,
            'threshold': round(threshold, 4),
            'N_sel': ev_sel['N'],
            'N_rest': ev_rst['N'],
            'actual_pct': round(actual_pct, 1),
            'ev_sel': ev_sel['EV'],
            'wr_sel': ev_sel['wr'],
            'pr_sel': ev_sel['pr'],
            'sum_sel': ev_sel['sum_pnl'],
            'avg_sel': ev_sel['avg_pnl'],
            'max_loss_sel': round(max_loss_sel, 4),
            'ev_rest': ev_rst['EV'],
            'wr_rest': ev_rst['wr'],
            'sum_rest': ev_rst['sum_pnl'],
            'ev_all': ev_all['EV'],
            'wr_all': ev_all['wr'],
        })

    return results


def check_monotonic(results, key='ev_sel'):
    """检查是否单调递增（允许1次违反）"""
    vals = [r[key] for r in results]
    violations = 0
    for i in range(1, len(vals)):
        if vals[i] < vals[i-1]:
            violations += 1
    return violations <= 1, violations


# ============================================================
#  仓位模拟（加仓+减仓组合）
# ============================================================

def simulate_combined_sizing(df, metric_col, pnl_col, reason_col,
                             top_threshold, bot_threshold):
    """
    三档仓位模拟：
    - 底部(<=bot_threshold): 0.5x 仓位（减仓）
    - 中间: 1x 仓位
    - 顶部(>=top_threshold): 2x 仓位（加仓）

    返回各项统计
    """
    if reason_col in df.columns:
        df = df[df[reason_col] != 'backtest_end'].copy()

    pnls = df[pnl_col].values
    vals = df[metric_col].values

    base_pnls = []
    sized_pnls = []
    tiers = {'top': 0, 'mid': 0, 'bot': 0}

    for pnl, v in zip(pnls, vals):
        base_pnls.append(pnl)
        if not np.isnan(v) and v >= top_threshold:
            sized_pnls.append(pnl * 2)
            tiers['top'] += 1
        elif not np.isnan(v) and v <= bot_threshold:
            sized_pnls.append(pnl * 0.5)
            tiers['bot'] += 1
        else:
            sized_pnls.append(pnl)
            tiers['mid'] += 1

    sum_base = sum(base_pnls)
    sum_sized = sum(sized_pnls)

    def max_drawdown(pnl_list):
        if not pnl_list:
            return 0
        cum = np.cumsum(pnl_list)
        peak = np.maximum.accumulate(cum)
        return round(float(np.max(peak - cum)), 4)

    mdd_base = max_drawdown(base_pnls)
    mdd_sized = max_drawdown(sized_pnls)
    gain_pct = ((sum_sized - sum_base) / abs(sum_base) * 100) if abs(sum_base) > 0.001 else 0
    dd_ratio = mdd_sized / mdd_base if mdd_base > 0.001 else 0

    n = len(pnls)
    return {
        'sum_base': round(sum_base, 2),
        'sum_sized': round(sum_sized, 2),
        'gain_pct': round(gain_pct, 1),
        'mdd_base': mdd_base,
        'mdd_sized': mdd_sized,
        'dd_ratio': round(dd_ratio, 2),
        'top_n': tiers['top'],
        'mid_n': tiers['mid'],
        'bot_n': tiers['bot'],
        'top_pct': round(tiers['top'] / n * 100, 1) if n > 0 else 0,
        'bot_pct': round(tiers['bot'] / n * 100, 1) if n > 0 else 0,
    }


# ============================================================
#  报告生成
# ============================================================

def build_report(df_all):
    rpt = Report('ER走势深度研究：多分位扫描 + 底部减仓')

    rpt.add_html('''<div style="background:#1c2333;padding:12px 16px;border-radius:8px;
        margin:12px 0;border-left:4px solid #58a6ff;font-size:13px;line-height:1.8">
        <b>研究1 — 顶部扫描</b>：前50%→前5%逐步收严，EV是否单调上升？单调=指标真有用<br>
        <b>研究2 — 底部扫描</b>：后5%→后25%，EV是否显著差于整体？差=该减仓<br>
        <b>研究3 — 三档仓位</b>：顶部2x + 底部0.5x + 中间1x，总收益和回撤如何变化？
    </div>''')

    # 收集所有最终推荐
    recommendations = []

    for sc_num in [1, 2, 3]:
        pnl_col = SCENARIO_PNL[sc_num]
        reason_col = SCENARIO_REASON[sc_num]
        sc_df = df_all[df_all['scenario'] == sc_num].copy()

        # 基线
        base_df = sc_df[sc_df[reason_col] != 'backtest_end'] if reason_col in sc_df.columns else sc_df
        base_st = calc_ev(base_df[pnl_col].tolist())

        rpt.add_section(f'场景{sc_num}. {SCENARIO_NAMES[sc_num]}')
        rpt.add_html(f'''<div style="background:#161b22;padding:10px 16px;border-radius:6px;
            margin:8px 0;border:1px solid #30363d;font-size:13px">
            <span style="color:#8b949e">基线</span> &nbsp;
            <b>N={base_st["N"]}</b> &nbsp;
            <span style="color:#58a6ff">EV={base_st["EV"]}</span> &nbsp;
            胜率={base_st["wr"]}% &nbsp;
            <span style="color:#3fb950">累计={base_st["sum_pnl"]:+.1f}%</span>
        </div>''')

        for m in METRICS:
            col = m['col']
            name = m['name']

            # ---- 顶部扫描 ----
            top_results = scan_percentiles(sc_df, col, pnl_col, reason_col,
                                           TOP_PERCENTILES, direction='top')
            is_mono_ev, violations_ev = check_monotonic(top_results, 'ev_sel')
            is_mono_wr, violations_wr = check_monotonic(top_results, 'wr_sel')

            # ---- 底部扫描 ----
            bot_results = scan_percentiles(sc_df, col, pnl_col, reason_col,
                                           BOT_PERCENTILES, direction='bot')

            # 判定颜色
            if is_mono_ev and top_results[-1]['ev_sel'] > base_st['EV']:
                card_color = '#238636'  # 绿
                tag = 'EV单调递增'
                tag_color = '#3fb950'
            elif top_results[-1]['ev_sel'] > top_results[0]['ev_sel']:
                card_color = '#9e6a03'  # 黄
                tag = f'大致递增(违反{violations_ev}次)'
                tag_color = '#d29922'
            else:
                card_color = '#30363d'  # 灰
                tag = '无单调趋势'
                tag_color = '#8b949e'

            # 底部EV是否差于整体
            bot_worst = bot_results[0]  # P5，最底部5%
            bot_25 = bot_results[-1]    # P25，底部25%
            if bot_25['ev_sel'] < base_st['EV'] * 0.5:
                bot_tag = '底部很差，建议减仓'
                bot_color = '#f85149'
            elif bot_25['ev_sel'] < base_st['EV']:
                bot_tag = '底部偏弱'
                bot_color = '#d29922'
            else:
                bot_tag = '底部不弱，减仓无意义'
                bot_color = '#8b949e'

            # ---- 渲染卡片 ----
            rpt.add_html(f'''
            <div style="background:#161b22;padding:12px 16px;border-radius:8px;margin:12px 0;
                border-left:4px solid {card_color}">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                    <span style="font-size:14px;font-weight:bold;color:#c9d1d9">{name}</span>
                    <span style="color:{tag_color};font-size:12px;font-weight:bold">{tag}</span>
                </div>''')

            # 顶部扫描表
            rpt.add_html(f'''
                <div style="font-size:11px;color:#8b949e;margin:4px 0">顶部扫描（前X%加仓）</div>
                <table style="margin:0;font-size:12px">
                <tr>
                    <th style="background:transparent;color:#8b949e;border:none;padding:4px 8px;text-align:right">分位</th>
                    <th style="background:transparent;color:#8b949e;border:none;padding:4px 8px;text-align:right">阈值</th>
                    <th style="background:transparent;color:#8b949e;border:none;padding:4px 8px;text-align:right">选中N</th>
                    <th style="background:transparent;color:#8b949e;border:none;padding:4px 8px;text-align:right">占比</th>
                    <th style="background:transparent;color:#8b949e;border:none;padding:4px 8px;text-align:right">选中EV</th>
                    <th style="background:transparent;color:#8b949e;border:none;padding:4px 8px;text-align:right">选中WR</th>
                    <th style="background:transparent;color:#8b949e;border:none;padding:4px 8px;text-align:right">其余EV</th>
                    <th style="background:transparent;color:#8b949e;border:none;padding:4px 8px;text-align:right">其余WR</th>
                    <th style="background:transparent;color:#8b949e;border:none;padding:4px 8px;text-align:right">累计PnL</th>
                    <th style="background:transparent;color:#8b949e;border:none;padding:4px 8px;text-align:right">最大单亏</th>
                </tr>''')

            for r in top_results:
                ev_color = '#3fb950' if r['ev_sel'] > base_st['EV'] else ('#f85149' if r['ev_sel'] < 0 else '#c9d1d9')
                n_warn = ' style="color:#f85149"' if r['N_sel'] < 20 else ''
                pct_label = f"前{100-r['percentile']:.0f}%"
                rpt.add_html(f'''
                <tr>
                    <td style="border:none;padding:4px 8px;text-align:right;color:#8b949e">{pct_label}</td>
                    <td style="border:none;padding:4px 8px;text-align:right">{r['threshold']}</td>
                    <td{n_warn} style="border:none;padding:4px 8px;text-align:right">{r['N_sel']}</td>
                    <td style="border:none;padding:4px 8px;text-align:right">{r['actual_pct']}%</td>
                    <td style="border:none;padding:4px 8px;text-align:right;color:{ev_color};font-weight:bold">{r['ev_sel']}</td>
                    <td style="border:none;padding:4px 8px;text-align:right">{r['wr_sel']}%</td>
                    <td style="border:none;padding:4px 8px;text-align:right;color:#8b949e">{r['ev_rest']}</td>
                    <td style="border:none;padding:4px 8px;text-align:right;color:#8b949e">{r['wr_rest']}%</td>
                    <td style="border:none;padding:4px 8px;text-align:right">{r['sum_sel']:+.1f}%</td>
                    <td style="border:none;padding:4px 8px;text-align:right;color:#f85149">{r['max_loss_sel']:+.2f}%</td>
                </tr>''')

            rpt.add_html('</table>')

            # 底部扫描表
            rpt.add_html(f'''
                <div style="font-size:11px;margin:8px 0 4px">
                    <span style="color:#8b949e">底部扫描（后X%减仓） — </span>
                    <span style="color:{bot_color}">{bot_tag}</span>
                </div>
                <table style="margin:0;font-size:12px">
                <tr>
                    <th style="background:transparent;color:#8b949e;border:none;padding:4px 8px;text-align:right">分位</th>
                    <th style="background:transparent;color:#8b949e;border:none;padding:4px 8px;text-align:right">阈值</th>
                    <th style="background:transparent;color:#8b949e;border:none;padding:4px 8px;text-align:right">选中N</th>
                    <th style="background:transparent;color:#8b949e;border:none;padding:4px 8px;text-align:right">占比</th>
                    <th style="background:transparent;color:#8b949e;border:none;padding:4px 8px;text-align:right">底部EV</th>
                    <th style="background:transparent;color:#8b949e;border:none;padding:4px 8px;text-align:right">底部WR</th>
                    <th style="background:transparent;color:#8b949e;border:none;padding:4px 8px;text-align:right">其余EV</th>
                    <th style="background:transparent;color:#8b949e;border:none;padding:4px 8px;text-align:right">累计PnL</th>
                </tr>''')

            for r in bot_results:
                ev_color = '#f85149' if r['ev_sel'] < 0 else ('#d29922' if r['ev_sel'] < base_st['EV'] else '#c9d1d9')
                rpt.add_html(f'''
                <tr>
                    <td style="border:none;padding:4px 8px;text-align:right;color:#8b949e">后{r['percentile']}%</td>
                    <td style="border:none;padding:4px 8px;text-align:right">{r['threshold']}</td>
                    <td style="border:none;padding:4px 8px;text-align:right">{r['N_sel']}</td>
                    <td style="border:none;padding:4px 8px;text-align:right">{r['actual_pct']}%</td>
                    <td style="border:none;padding:4px 8px;text-align:right;color:{ev_color};font-weight:bold">{r['ev_sel']}</td>
                    <td style="border:none;padding:4px 8px;text-align:right">{r['wr_sel']}%</td>
                    <td style="border:none;padding:4px 8px;text-align:right;color:#8b949e">{r['ev_rest']}</td>
                    <td style="border:none;padding:4px 8px;text-align:right">{r['sum_sel']:+.1f}%</td>
                </tr>''')

            rpt.add_html('</table>')

            # ---- 三档仓位模拟 ----
            # 用P75作为加仓阈值，P25作为减仓阈值
            top_thr = top_results[3]['threshold']  # P75 index=3 in TOP_PERCENTILES
            bot_thr = bot_results[-1]['threshold']  # P25
            sizing = simulate_combined_sizing(sc_df, col, pnl_col, reason_col,
                                              top_thr, bot_thr)

            gain_color = '#3fb950' if sizing['gain_pct'] > 10 else ('#d29922' if sizing['gain_pct'] > 0 else '#f85149')
            dd_color = '#3fb950' if sizing['dd_ratio'] < 1.2 else ('#d29922' if sizing['dd_ratio'] < 1.5 else '#f85149')

            rpt.add_html(f'''
                <div style="font-size:11px;color:#8b949e;margin:8px 0 4px">三档仓位模拟 (顶部2x / 中间1x / 底部0.5x)</div>
                <div style="font-size:12px;padding:4px 0">
                    分布: 顶部{sizing['top_pct']}%({sizing['top_n']}笔) |
                    中间{100-sizing['top_pct']-sizing['bot_pct']:.0f}%({sizing['mid_n']}笔) |
                    底部{sizing['bot_pct']}%({sizing['bot_n']}笔) &nbsp;&nbsp;
                    基础累计: {sizing['sum_base']:+.1f}% → 三档累计: {sizing['sum_sized']:+.1f}%
                    &nbsp;&nbsp;
                    <span style="color:{gain_color}">增益: {sizing['gain_pct']:+.1f}%</span>
                    &nbsp;&nbsp;
                    <span style="color:{dd_color}">回撤放大: {sizing['dd_ratio']}x</span>
                </div>
            </div>''')

            # 收集推荐
            if is_mono_ev and top_results[-1]['ev_sel'] > base_st['EV']:
                # 找最优分位：EV最高且N>=15
                best = None
                for r in reversed(top_results):
                    if r['N_sel'] >= 15:
                        if best is None or r['ev_sel'] > best['ev_sel']:
                            best = r
                        break  # 从最严格开始，第一个N>=15的就是最优

                recommendations.append({
                    'scenario': sc_num,
                    'metric': name,
                    'col': col,
                    'mono': True,
                    'best_pct': f"前{100-best['percentile']:.0f}%" if best else '-',
                    'best_thr': best['threshold'] if best else 0,
                    'best_ev': best['ev_sel'] if best else 0,
                    'best_wr': best['wr_sel'] if best else 0,
                    'best_n': best['N_sel'] if best else 0,
                    'base_ev': base_st['EV'],
                    'bot_useful': bot_25['ev_sel'] < base_st['EV'] * 0.5,
                    'sizing': sizing,
                })

    # ---- 最终汇总 ----
    rpt.add_section('汇总：推荐的加仓/减仓指标')

    if recommendations:
        rpt.add_html('''<div style="font-size:12px;color:#8b949e;margin:8px 0">
            只列出顶部EV单调递增的指标（越严格EV越高 = 指标可靠）
        </div>''')

        headers = ['场景', '指标', '最优分位', '阈值', 'N', 'EV', '胜率%',
                   '基线EV', '底部减仓?', '三档增益%', '回撤放大']
        rows = []
        for r in recommendations:
            rows.append([
                f'场景{r["scenario"]}', r['metric'], r['best_pct'],
                r['best_thr'], r['best_n'], r['best_ev'], r['best_wr'],
                r['base_ev'],
                'Y' if r['bot_useful'] else 'N',
                f'{r["sizing"]["gain_pct"]:+.1f}',
                r['sizing']['dd_ratio'],
            ])
        rpt.add_table(headers, rows, highlight_pnl_cols=[5, 9])
    else:
        rpt.add_text('没有指标通过单调递增检验', color='#f85149')

    # 最终结论
    rpt.add_section('结论')
    if recommendations:
        mono_metrics = set(r['metric'] for r in recommendations)
        cross_scene = {}
        for r in recommendations:
            cross_scene.setdefault(r['metric'], []).append(r['scenario'])
        multi = {k: v for k, v in cross_scene.items() if len(v) >= 2}

        if multi:
            rpt.add_text('跨场景验证通过的指标（最可信）:', color='#3fb950')
            for metric, scenes in multi.items():
                sc_str = '、'.join(f'场景{s}' for s in scenes)
                rpt.add_text(f'  {metric} — 在{sc_str}中EV均单调递增', color='#3fb950')
        else:
            rpt.add_text('没有指标在2个以上场景同时单调递增', color='#d29922')

        # 减仓建议
        bot_useful = [r for r in recommendations if r['bot_useful']]
        if bot_useful:
            rpt.add_text('底部减仓有意义的指标:', color='#58a6ff')
            for r in bot_useful:
                rpt.add_text(f'  场景{r["scenario"]} + {r["metric"]}: 底部25% EV远低于基线')

        rpt.add_html('''<div style="background:#1c2333;padding:10px 16px;border-radius:6px;
            margin:12px 0;border-left:4px solid #d29922;font-size:12px;line-height:1.6;color:#d29922">
            <b>重要提醒</b><br>
            1. 场景3仅60笔，前5%=3笔，统计无意义，请忽略场景3的极端分位结论<br>
            2. 单调递增是必要条件，但前5%~10%的样本太少可能是幸存者偏差<br>
            3. 建议以前15%~25%作为实盘加仓阈值，不要用前5%~10%（样本太少不可靠）
        </div>''')

    return rpt


# ============================================================
#  主函数
# ============================================================

def run():
    df_all = collect_signals()
    for sc in [1, 2, 3]:
        n = len(df_all[df_all['scenario'] == sc])
        print(f"场景{sc}: {n}笔")
    rpt = build_report(df_all)
    rpt.save('output/er_trend_deep.html')
    print("完成!")


if __name__ == '__main__':
    run()
