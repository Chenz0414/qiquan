# -*- coding: utf-8 -*-
"""
ER5/ER40走势作为仓位加码依据研究
================================
不过滤信号，只验证：ER5/ER40走势好时标记"高置信"→ 加仓是否值得。
高置信 = 每个指标在该场景内取前25%（P75分位），保证约25%高置信、75%普通。

输出: output/er_trend_research.html
"""

import sys
import io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from data_loader import load_all, sym_name, tick_size
from signal_core import SignalDetector, ExitTracker, DEFAULT_STOP_TICKS
from stats_utils import calc_ev
from chart_engine import render_chart
from report_engine import Report


# ============================================================
#  场景分类
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


# ============================================================
#  7个ER走势指标
# ============================================================

METRICS = [
    {'col': 'er_5',          'name': 'ER(5)',          'higher_better': True},
    {'col': 'er_40',         'name': 'ER(40)',         'higher_better': True},
    {'col': 'er5_delta_6',   'name': 'ER5变化(6根)',   'higher_better': True},
    {'col': 'er40_delta_12', 'name': 'ER40变化(12根)', 'higher_better': True},
    {'col': 'er5_over_er20', 'name': 'ER5/ER20比值',   'higher_better': True},
    {'col': 'er5_gt_er20',   'name': 'ER5>ER20',       'higher_better': True,  'binary': True},
    {'col': 'er5_gt_er40',   'name': 'ER5>ER40',       'higher_better': True,  'binary': True},
]

SCENARIO_PNL = {1: 's2_pnl', 2: 's2_pnl', 3: 's51_pnl'}
SCENARIO_REASON = {1: 's2_reason', 2: 's2_reason', 3: 's51_reason'}
SCENARIO_NAMES = {1: 'A类 + ER>=0.5 + 偏离>=1.0ATR → S2出场',
                  2: 'C类 + 偏离>=2.0ATR → S2出场',
                  3: 'B类 + ER>=0.5 + 偏离0.1~0.3ATR → S5.1出场'}


# ============================================================
#  数据收集（不变）
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
        df['er5_gt_er20'] = (df['er_5'] > df['er_20']).astype(int)
        df['er5_gt_er40'] = (df['er_5'] > df['er_40']).astype(int)

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
                'er5_gt_er20': int(row.get('er5_gt_er20', 0) if not pd.isna(row.get('er5_gt_er20', np.nan)) else 0),
                'er5_gt_er40': int(row.get('er5_gt_er40', 0) if not pd.isna(row.get('er5_gt_er40', np.nan)) else 0),
                's2_pnl': round(s2.pnl_pct, 4) if s2 else 0,
                's2_reason': s2.exit_reason if s2 else 'missing',
                's51_pnl': round(s51.pnl_pct, 4) if s51 else 0,
                's51_reason': s51.exit_reason if s51 else 'missing',
                '_df': df, '_entry_idx': i,
                '_stop_price': (result.pullback_extreme - ts * DEFAULT_STOP_TICKS
                                if result.direction == 'long'
                                else result.pullback_extreme + ts * DEFAULT_STOP_TICKS),
            })

    print(f"总信号: {len(records)}笔")
    df_all = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in records])
    return records, df_all


# ============================================================
#  分析函数
# ============================================================

def compute_threshold(series, metric):
    """计算自适应阈值：取P75分位（前25%为高置信）"""
    if metric.get('binary'):
        return 1  # 二值指标固定用1
    valid = series.dropna()
    if len(valid) < 5:
        return valid.median() if len(valid) > 0 else 0
    return float(np.percentile(valid, 75))


def split_high_normal(df, metric, pnl_col, reason_col):
    """按P75分位分成高置信/普通两组"""
    col = metric['col']
    threshold = compute_threshold(df[col], metric)

    if metric.get('binary'):
        mask_high = df[col] == 1
    else:
        mask_high = df[col] >= threshold

    high_df = df[mask_high]
    norm_df = df[~mask_high]

    def group_stats(sub, label):
        if reason_col in sub.columns:
            sub = sub[sub[reason_col] != 'backtest_end']
        pnls = sub[pnl_col].tolist()
        st = calc_ev(pnls)
        st['label'] = label
        st['max_loss'] = round(min(pnls), 4) if pnls else 0
        st['max_gain'] = round(max(pnls), 4) if pnls else 0
        st['max_consec_loss'] = max_consecutive_losses(pnls)
        return st, pnls

    st_high, pnls_high = group_stats(high_df, '高置信')
    st_norm, pnls_norm = group_stats(norm_df, '普通')
    st_all, pnls_all = group_stats(df, '全部')
    ratio = st_high['N'] / st_all['N'] * 100 if st_all['N'] > 0 else 0

    return {
        'high': st_high, 'normal': st_norm, 'all': st_all,
        'high_pnls': pnls_high, 'normal_pnls': pnls_norm, 'all_pnls': pnls_all,
        'high_ratio': round(ratio, 1), 'threshold': threshold,
    }


def max_consecutive_losses(pnls):
    max_cl = cur = 0
    for p in pnls:
        if p <= 0:
            cur += 1
            max_cl = max(max_cl, cur)
        else:
            cur = 0
    return max_cl


def simulate_sizing(pnls_all, is_high_list):
    base_pnls = list(pnls_all)
    sized_pnls = [p * 2 if h else p for p, h in zip(pnls_all, is_high_list)]
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
    return {
        'sum_base': round(sum_base, 2), 'sum_sized': round(sum_sized, 2),
        'gain_pct': round(gain_pct, 1), 'mdd_base': mdd_base,
        'mdd_sized': mdd_sized, 'dd_ratio': round(dd_ratio, 2),
    }


def leave_one_out(pnls_high, pnls_norm):
    if not pnls_high or not pnls_norm:
        return None
    st_norm = calc_ev(pnls_norm)
    max_idx = int(np.argmax(pnls_high))
    reduced = [p for i, p in enumerate(pnls_high) if i != max_idx]
    if not reduced:
        return None
    st_reduced = calc_ev(reduced)
    return {
        'high_ev_full': calc_ev(pnls_high)['EV'],
        'high_ev_loo': st_reduced['EV'],
        'norm_ev': st_norm['EV'],
        'robust': st_reduced['EV'] > st_norm['EV'],
    }


def compute_correlation(df_all):
    cols = ['er_5', 'er_20', 'er_40', 'er5_delta_6', 'er40_delta_12']
    existing = [c for c in cols if c in df_all.columns]
    sub = df_all[existing].dropna()
    if len(sub) < 10:
        return None, existing
    return sub.corr(method='spearman'), existing


def judge_bonus(st_high, st_norm, high_ratio, sizing, loo):
    checks = {
        'EV更高': st_high['EV'] > st_norm['EV'],
        '胜率更高': st_high['wr'] > st_norm['wr'],
        '占比合理': 15 <= high_ratio <= 45,
        '加仓有增益': sizing['gain_pct'] > 10,
        '回撤可控': sizing['dd_ratio'] < 1.5,
        '稳健': loo['robust'] if loo else False,
    }
    passed = sum(checks.values())
    if passed >= 5:
        verdict = 'PASS'
    elif passed >= 3:
        verdict = 'MARGINAL'
    else:
        verdict = 'FAIL'
    return verdict, checks


# ============================================================
#  报告生成（优化格式）
# ============================================================

def build_report(records, df_all):
    rpt = Report('ER5/ER40走势 — 仓位加码依据研究')

    # 报告说明
    rpt.add_html('''<div style="background:#1c2333;padding:12px 16px;border-radius:8px;
        margin:12px 0;border-left:4px solid #58a6ff;font-size:13px;line-height:1.8">
        <b>研究目的</b>：不砍任何信号，看ER5/ER40走势好的时候加大仓位是否值得<br>
        <b>分组方法</b>：每个指标取该场景内<b>前25%</b>（P75分位）为高置信，其余为普通<br>
        <b>加仓模拟</b>：高置信=2倍仓位，普通=1倍仓位<br>
        <b>判定标准</b>：高EV>普通EV + 高胜率>普通胜率 + 占比15~45% + 增益>10% + 回撤<1.5x + 去最大盈利后仍优
    </div>''')

    # ---- Section 0: 相关性矩阵 ----
    rpt.add_section('0. 指标相关性', '检查指标之间是否重复（|r|>0.7=冗余）')
    corr, corr_cols = compute_correlation(df_all)
    if corr is not None:
        col_names = {'er_5': 'ER(5)', 'er_20': 'ER(20)', 'er_40': 'ER(40)',
                     'er5_delta_6': 'ER5变化', 'er40_delta_12': 'ER40变化'}
        headers = [''] + [col_names.get(c, c) for c in corr_cols]
        rows = []
        for r_col in corr_cols:
            row = [col_names.get(r_col, r_col)]
            for c_col in corr_cols:
                v = corr.loc[r_col, c_col]
                row.append(f'{v:.2f}')
            rows.append(row)
        rpt.add_table(headers, rows)

        # 标注冗余
        redundant = []
        for i, c1 in enumerate(corr_cols):
            for c2 in corr_cols[i+1:]:
                r = abs(corr.loc[c1, c2])
                if r > 0.6:
                    redundant.append(f'{col_names.get(c1,c1)} vs {col_names.get(c2,c2)}: r={r:.2f}')
        if redundant:
            rpt.add_text('注意相关较高: ' + ' | '.join(redundant), color='#d29922')
        else:
            rpt.add_text('所有指标对相关性均<0.6，独立性良好', color='#3fb950')

    # ---- Sections 1-3: 每场景用卡片式展示 ----
    all_results = []  # 收集所有结果用于汇总

    for sc_num in [1, 2, 3]:
        pnl_col = SCENARIO_PNL[sc_num]
        reason_col = SCENARIO_REASON[sc_num]
        sc_df = df_all[df_all['scenario'] == sc_num].copy()

        base_df = sc_df[sc_df[reason_col] != 'backtest_end'] if reason_col in sc_df.columns else sc_df
        base_st = calc_ev(base_df[pnl_col].tolist())

        rpt.add_section(f'场景{sc_num}. {SCENARIO_NAMES[sc_num]}')
        # 基线卡片
        rpt.add_html(f'''<div style="background:#161b22;padding:10px 16px;border-radius:6px;
            margin:8px 0;border:1px solid #30363d;font-size:13px">
            <span style="color:#8b949e">基线（全部信号）</span> &nbsp;
            <b>N={base_st["N"]}</b> &nbsp;
            <span style="color:#58a6ff">EV={base_st["EV"]}</span> &nbsp;
            胜率={base_st["wr"]}% &nbsp; 盈亏比={base_st["pr"]} &nbsp;
            <span style="color:#3fb950">累计={base_st["sum_pnl"]:+.1f}%</span>
        </div>''')

        for m in METRICS:
            result = split_high_normal(sc_df, m, pnl_col, reason_col)
            h = result['high']
            nm = result['normal']
            thr = result['threshold']

            # 加仓模拟
            base_ordered = sc_df[sc_df[reason_col] != 'backtest_end'] if reason_col in sc_df.columns else sc_df
            col = m['col']
            if m.get('binary'):
                is_high = (base_ordered[col] == 1).tolist()
            else:
                is_high = (base_ordered[col] >= thr).tolist()
            pnls_ordered = base_ordered[pnl_col].tolist()
            sizing = simulate_sizing(pnls_ordered, is_high)
            loo = leave_one_out(result['high_pnls'], result['normal_pnls'])
            verdict, checks = judge_bonus(h, nm, result['high_ratio'], sizing, loo)

            # 颜色
            if verdict == 'PASS':
                badge_color = '#3fb950'
                border_color = '#238636'
            elif verdict == 'MARGINAL':
                badge_color = '#d29922'
                border_color = '#9e6a03'
            else:
                badge_color = '#8b949e'
                border_color = '#30363d'

            # EV差和胜率差的箭头
            ev_diff = h['EV'] - nm['EV']
            wr_diff = h['wr'] - nm['wr']
            ev_arrow = f'<span style="color:#3fb950">+{ev_diff:.2f}</span>' if ev_diff > 0 else f'<span style="color:#f85149">{ev_diff:.2f}</span>'
            wr_arrow = f'<span style="color:#3fb950">+{wr_diff:.1f}%</span>' if wr_diff > 0 else f'<span style="color:#f85149">{wr_diff:.1f}%</span>'

            # 检查项标记
            check_html = ' '.join(
                f'<span style="color:{"#3fb950" if v else "#f85149"}">{k}</span>'
                for k, v in checks.items()
            )

            thr_display = f'阈值P75={thr:.3f}' if not m.get('binary') else '二值(True/False)'

            rpt.add_html(f'''
            <div style="background:#161b22;padding:12px 16px;border-radius:8px;margin:10px 0;
                border-left:4px solid {border_color}">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                    <span style="font-size:14px;font-weight:bold;color:#c9d1d9">{m['name']}</span>
                    <span style="background:{badge_color};color:#000;padding:2px 10px;border-radius:4px;
                        font-size:12px;font-weight:bold">{verdict}</span>
                </div>
                <div style="color:#8b949e;font-size:11px;margin-bottom:8px">{thr_display} | 高置信占比 {result["high_ratio"]}%</div>
                <table style="margin:0;font-size:12px">
                <tr><th style="background:transparent;color:#8b949e;border:none;padding:4px 12px 4px 0;text-align:left">组</th>
                    <th style="background:transparent;color:#8b949e;border:none;padding:4px 12px;text-align:right">N</th>
                    <th style="background:transparent;color:#8b949e;border:none;padding:4px 12px;text-align:right">EV</th>
                    <th style="background:transparent;color:#8b949e;border:none;padding:4px 12px;text-align:right">胜率</th>
                    <th style="background:transparent;color:#8b949e;border:none;padding:4px 12px;text-align:right">盈亏比</th>
                    <th style="background:transparent;color:#8b949e;border:none;padding:4px 12px;text-align:right">累计PnL</th>
                    <th style="background:transparent;color:#8b949e;border:none;padding:4px 12px;text-align:right">最大单亏</th>
                    <th style="background:transparent;color:#8b949e;border:none;padding:4px 12px;text-align:right">最大连亏</th>
                </tr>
                <tr style="color:#58a6ff">
                    <td style="border:none;padding:4px 12px 4px 0;text-align:left;font-weight:bold">高置信</td>
                    <td style="border:none;padding:4px 12px;text-align:right">{h['N']}</td>
                    <td style="border:none;padding:4px 12px;text-align:right;font-weight:bold">{h['EV']}</td>
                    <td style="border:none;padding:4px 12px;text-align:right">{h['wr']}%</td>
                    <td style="border:none;padding:4px 12px;text-align:right">{h['pr']}</td>
                    <td style="border:none;padding:4px 12px;text-align:right">{h['sum_pnl']:+.1f}%</td>
                    <td style="border:none;padding:4px 12px;text-align:right;color:#f85149">{h['max_loss']:+.2f}%</td>
                    <td style="border:none;padding:4px 12px;text-align:right">{h['max_consec_loss']}</td>
                </tr>
                <tr style="color:#8b949e">
                    <td style="border:none;padding:4px 12px 4px 0;text-align:left">普通</td>
                    <td style="border:none;padding:4px 12px;text-align:right">{nm['N']}</td>
                    <td style="border:none;padding:4px 12px;text-align:right">{nm['EV']}</td>
                    <td style="border:none;padding:4px 12px;text-align:right">{nm['wr']}%</td>
                    <td style="border:none;padding:4px 12px;text-align:right">{nm['pr']}</td>
                    <td style="border:none;padding:4px 12px;text-align:right">{nm['sum_pnl']:+.1f}%</td>
                    <td style="border:none;padding:4px 12px;text-align:right">{nm['max_loss']:+.2f}%</td>
                    <td style="border:none;padding:4px 12px;text-align:right">{nm['max_consec_loss']}</td>
                </tr>
                <tr><td colspan="8" style="border:none;padding:2px 0"><hr style="border-color:#21262d"></td></tr>
                <tr style="font-size:11px">
                    <td colspan="2" style="border:none;padding:4px 0;text-align:left;color:#c9d1d9">
                        EV差: {ev_arrow} &nbsp; 胜率差: {wr_arrow}</td>
                    <td colspan="3" style="border:none;padding:4px 0;text-align:right;color:#c9d1d9">
                        加仓增益: <b>{sizing['gain_pct']:+.1f}%</b></td>
                    <td colspan="3" style="border:none;padding:4px 0;text-align:right;color:#c9d1d9">
                        回撤放大: <b>{sizing['dd_ratio']}x</b></td>
                </tr>
                </table>
                <div style="font-size:11px;margin-top:6px;color:#8b949e">
                    检查项: {check_html}
                </div>
            </div>''')

            all_results.append({
                'scenario': sc_num, 'metric': m['name'], 'col': col,
                'threshold': thr, 'binary': m.get('binary', False),
                'high': h, 'normal': nm, 'ratio': result['high_ratio'],
                'sizing': sizing, 'loo': loo, 'verdict': verdict, 'checks': checks,
            })

    # ---- Section 4: 最终汇总 ----
    rpt.add_section('汇总：值得加仓的指标')

    passed = [r for r in all_results if r['verdict'] == 'PASS']
    marginal = [r for r in all_results if r['verdict'] == 'MARGINAL']

    if passed:
        rpt.add_html('<div style="margin:8px 0">')
        for r in passed:
            h, nm = r['high'], r['normal']
            thr_str = f'>={r["threshold"]:.3f}' if not r['binary'] else '=True'
            rpt.add_html(f'''
            <div style="background:#0d1117;padding:10px 16px;border-radius:6px;margin:6px 0;
                border:1px solid #238636;font-size:13px">
                <span style="color:#3fb950;font-weight:bold">PASS</span> &nbsp;
                场景{r['scenario']} + <b>{r['metric']}</b> ({thr_str}) &nbsp;|&nbsp;
                高EV=<b>{h['EV']}</b> vs 普EV={nm['EV']} &nbsp;|&nbsp;
                高WR=<b>{h['wr']}%</b> vs 普WR={nm['wr']}% &nbsp;|&nbsp;
                占比={r['ratio']}% &nbsp;|&nbsp;
                增益=<span style="color:#3fb950">{r['sizing']['gain_pct']:+.1f}%</span> &nbsp;
                回撤={r['sizing']['dd_ratio']}x
            </div>''')
        rpt.add_html('</div>')
    else:
        rpt.add_text('没有指标完全通过所有标准', color='#f85149')

    if marginal:
        rpt.add_text('边缘指标（部分通过，仅供参考）:', color='#d29922')
        for r in marginal:
            h, nm = r['high'], r['normal']
            rpt.add_html(f'''
            <div style="font-size:12px;color:#8b949e;margin:4px 0 4px 16px">
                场景{r['scenario']} + {r['metric']}: 高EV={h['EV']} vs 普EV={nm['EV']},
                增益{r['sizing']['gain_pct']:+.1f}%, 回撤{r['sizing']['dd_ratio']}x
            </div>''')

    # ---- K线图 ----
    if passed and records:
        rpt.add_section('典型K线案例')
        shown = set()
        for item in passed[:3]:
            sc_num = item['scenario']
            if sc_num in shown:
                continue
            shown.add(sc_num)
            pnl_col = SCENARIO_PNL[sc_num]
            sc_recs = [r for r in records if r['scenario'] == sc_num]
            sc_recs.sort(key=lambda x: x[pnl_col])
            if len(sc_recs) >= 3:
                picks = [(sc_recs[-1], '最佳'), (sc_recs[len(sc_recs)//2], '中位'), (sc_recs[0], '最差')]
            elif sc_recs:
                picks = [(sc_recs[-1], '示例')]
            else:
                continue
            for rec, label in picks:
                title = f'场景{sc_num} {label}: {rec["symbol"]} {rec["time"]} PnL={rec[pnl_col]:+.4f}%'
                extra = f'ER5={rec["er_5"]} ER20={rec["er_20"]} ER40={rec["er_40"]} dev={rec["dev_atr"]}ATR'
                try:
                    chart_html = render_chart(
                        rec['_df'], rec['_entry_idx'], rec['direction'],
                        stop_price=rec['_stop_price'], title=title, extra_info=extra)
                    rpt.add_chart(chart_html)
                except Exception:
                    pass

    return rpt


# ============================================================
#  主函数
# ============================================================

def run():
    records, df_all = collect_signals()
    for sc in [1, 2, 3]:
        n = len(df_all[df_all['scenario'] == sc])
        print(f"场景{sc}: {n}笔")
    rpt = build_report(records, df_all)
    rpt.save('output/er_trend_research.html')
    print("完成!")


if __name__ == '__main__':
    run()
