# -*- coding: utf-8 -*-
"""
样本外验证：时间切分法
======================
将120天数据切成前60天(IS)和后60天(OOS)，验证3个入场场景的EV是否稳健。
同时做3折交叉验证（40天×3）。

核心问题：前半段发现的规则，后半段还成立吗？

输出: output/oos_validation.html
"""

import sys
import io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from data_loader import load_all, sym_name, tick_size, HIGH_VOL
from signal_core import SignalDetector, ExitTracker, DEFAULT_STOP_TICKS, SYMBOL_CONFIGS
from stats_utils import calc_ev
from report_engine import Report


# ============================================================
#  场景分类（复用已确认规则）
# ============================================================

def classify_scenario(sig_type, er20, deviation_atr):
    """返回场景编号 1/2/3 或 None"""
    if er20 >= 0.7:
        return None
    if sig_type == 'A' and er20 >= 0.5 and deviation_atr >= 1.0:
        return 1
    if sig_type == 'C' and deviation_atr >= 2.0:
        return 2
    if sig_type == 'B' and er20 >= 0.5 and 0.1 <= deviation_atr < 0.3:
        return 3
    return None


SCENARIO_NAMES = {
    1: '场景1: A类+ER≥0.5+偏离≥1.0ATR → S2',
    2: '场景2: C类+偏离≥2.0ATR → S2',
    3: '场景3: B类+ER≥0.5+偏离0.1~0.3ATR → S5.1',
}
SCENARIO_PNL = {1: 's2_pnl', 2: 's2_pnl', 3: 's51_pnl'}
SCENARIO_REASON = {1: 's2_reason', 2: 's2_reason', 3: 's51_reason'}


# ============================================================
#  数据收集：全量120天信号 + 时间戳
# ============================================================

def collect_all_signals():
    """收集全部120天内的可交易信号，带时间戳用于后续切分"""
    print("加载数据...")
    all_data = load_all(period_min=10, days=170, last_days=None,
                        emas=(10, 20, 120), er_periods=(5, 20, 40), atr_period=14)
    print(f"品种数: {len(all_data)}")

    # 添加ER走势指标（仓位加码用）
    for sym_key, df in all_data.items():
        df['er5_delta_6'] = df['er_5'] - df['er_5'].shift(6)
        df['er40_delta_12'] = df['er_40'] - df['er_40'].shift(12)

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
            er40 = row.get('er_40', 0)
            if pd.isna(er40):
                er40 = 0
            atr = row['atr']
            deviation_atr = abs(result.entry_price - row['ema10']) / atr if atr > 0 else 0

            # 场景分类
            scenario = classify_scenario(result.signal_type, er20, deviation_atr)

            # ER5变化过滤（场景1专属）
            er5_delta_6 = row.get('er5_delta_6', np.nan)
            if pd.isna(er5_delta_6):
                er5_delta_6 = 0
            er40_delta_12 = row.get('er40_delta_12', np.nan)
            if pd.isna(er40_delta_12):
                er40_delta_12 = 0

            # 模拟出场
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

            records.append({
                'symbol': name, 'sym_key': sym_key,
                'time': row['datetime'],
                'direction': result.direction,
                'type': result.signal_type,
                'scenario': scenario,  # None = 不符合任何场景
                'er_20': round(er20, 4),
                'er_40': round(er40, 4),
                'dev_atr': round(deviation_atr, 3),
                'er5_delta_6': round(er5_delta_6, 4),
                'er40_delta_12': round(er40_delta_12, 4),
                's2_pnl': round(s2.pnl_pct, 4) if s2 else 0,
                's2_reason': s2.exit_reason if s2 else 'missing',
                's51_pnl': round(s51.pnl_pct, 4) if s51 else 0,
                's51_reason': s51.exit_reason if s51 else 'missing',
            })

    print(f"总信号: {len(records)}笔")
    return pd.DataFrame(records)


# ============================================================
#  分析函数
# ============================================================

def analyze_window(df_window, label, exclude_backtest_end=True):
    """分析一个时间窗口内的3个场景表现，返回结果字典"""
    results = {}
    for sc in [1, 2, 3]:
        pnl_col = SCENARIO_PNL[sc]
        reason_col = SCENARIO_REASON[sc]
        sub = df_window[df_window['scenario'] == sc].copy()

        if exclude_backtest_end and reason_col in sub.columns:
            sub = sub[sub[reason_col] != 'backtest_end']

        pnls = sub[pnl_col].tolist()
        ev = calc_ev(pnls)
        results[sc] = ev
    return results


def analyze_with_filters(df_window, exclude_backtest_end=True):
    """带仓位加码过滤的分析"""
    results = {}

    for sc in [1, 2, 3]:
        pnl_col = SCENARIO_PNL[sc]
        reason_col = SCENARIO_REASON[sc]
        sub = df_window[df_window['scenario'] == sc].copy()

        if exclude_backtest_end and reason_col in sub.columns:
            sub = sub[sub[reason_col] != 'backtest_end']

        # 基础（无额外过滤）
        base_ev = calc_ev(sub[pnl_col].tolist())

        # 场景1专属过滤：ER5变化 > -0.41
        if sc == 1:
            filtered = sub[sub['er5_delta_6'] > -0.41]
            filtered_ev = calc_ev(filtered[pnl_col].tolist())
            # 加码：ER40>=0.42
            boost_er40 = sub[sub['er_40'] >= 0.42]
            boost_er40_ev = calc_ev(boost_er40[pnl_col].tolist())
            # 加码：ER5变化>=0.50
            boost_er5 = sub[sub['er5_delta_6'] >= 0.50]
            boost_er5_ev = calc_ev(boost_er5[pnl_col].tolist())
        else:
            filtered = sub
            filtered_ev = base_ev
            boost_er40_ev = None
            boost_er5_ev = None

        # 场景2加码：ER40变化>=0.14
        if sc == 2:
            boost_er40_chg = sub[sub['er40_delta_12'] >= 0.14]
            boost_er40_chg_ev = calc_ev(boost_er40_chg[pnl_col].tolist())
        else:
            boost_er40_chg_ev = None

        results[sc] = {
            'base': base_ev,
            'filtered': filtered_ev,
            'boost_er40': boost_er40_ev,
            'boost_er5': boost_er5_ev,
            'boost_er40_chg': boost_er40_chg_ev,
        }

    return results


# ============================================================
#  全量未过滤信号的ER/趋势基础规则验证
# ============================================================

def analyze_base_rules(df_window):
    """验证基础规则：ER>=0.5、ER<0.7、趋势EMA20>120等"""
    results = {}

    # 全量ABC信号（不分场景）的S2 EV
    all_s2 = df_window[df_window['s2_reason'] != 'backtest_end']['s2_pnl'].tolist()
    results['all_signals'] = calc_ev(all_s2)

    # ER>=0.5
    er_high = df_window[(df_window['er_20'] >= 0.5) & (df_window['s2_reason'] != 'backtest_end')]
    results['er_gte_0.5'] = calc_ev(er_high['s2_pnl'].tolist())

    # ER<0.5
    er_low = df_window[(df_window['er_20'] < 0.5) & (df_window['s2_reason'] != 'backtest_end')]
    results['er_lt_0.5'] = calc_ev(er_low['s2_pnl'].tolist())

    # ER>=0.7
    er_very_high = df_window[(df_window['er_20'] >= 0.7) & (df_window['s2_reason'] != 'backtest_end')]
    results['er_gte_0.7'] = calc_ev(er_very_high['s2_pnl'].tolist())

    # 0.5~0.7甜点区
    er_sweet = df_window[(df_window['er_20'] >= 0.5) & (df_window['er_20'] < 0.7) & (df_window['s2_reason'] != 'backtest_end')]
    results['er_0.5_0.7'] = calc_ev(er_sweet['s2_pnl'].tolist())

    # 按信号类型
    for sig_type in ['A', 'B', 'C']:
        sub = df_window[(df_window['type'] == sig_type) & (df_window['s2_reason'] != 'backtest_end')]
        results[f'type_{sig_type}'] = calc_ev(sub['s2_pnl'].tolist())

    return results


# ============================================================
#  HTML 报告生成
# ============================================================

def build_report(df_all):
    rpt = Report('样本外验证：时间切分法')

    # 确定时间范围
    t_min = df_all['time'].min()
    t_max = df_all['time'].max()
    t_mid = t_min + (t_max - t_min) / 2

    rpt.add_text(f'数据范围: {t_min.strftime("%Y-%m-%d")} ~ {t_max.strftime("%Y-%m-%d")}')
    rpt.add_text(f'中点切分: {t_mid.strftime("%Y-%m-%d %H:%M")}')
    rpt.add_text(f'总信号数: {len(df_all)} (场景内: {len(df_all[df_all["scenario"].notna()])})')

    # ============================================================
    #  Part 0: 基础规则验证（ER有效性、信号类型）
    # ============================================================
    rpt.add_section('Part 0: 基础规则时间稳定性', '全量ABC信号，不分场景，验证ER过滤和信号分类的基础有效性')

    df_first = df_all[df_all['time'] <= t_mid]
    df_second = df_all[df_all['time'] > t_mid]

    base_full = analyze_base_rules(df_all)
    base_first = analyze_base_rules(df_first)
    base_second = analyze_base_rules(df_second)

    headers = ['规则', '全量N', '全量EV', '全量WR%', '全量Σ',
               '前半N', '前半EV', '前半WR%', '前半Σ',
               '后半N', '后半EV', '后半WR%', '后半Σ', '方向一致']
    rows = []
    rule_names = {
        'all_signals': '全部信号(S2)',
        'er_gte_0.5': 'ER≥0.5',
        'er_lt_0.5': 'ER<0.5',
        'er_gte_0.7': 'ER≥0.7',
        'er_0.5_0.7': 'ER 0.5~0.7(甜点)',
        'type_A': 'A类信号',
        'type_B': 'B类信号',
        'type_C': 'C类信号',
    }
    for key, label in rule_names.items():
        f = base_full[key]
        h1 = base_first[key]
        h2 = base_second[key]
        # 方向一致 = 两半EV同号
        consistent = '✓' if (h1['EV'] > 0 and h2['EV'] > 0) or (h1['EV'] <= 0 and h2['EV'] <= 0) else '✗'
        rows.append([label,
                     f['N'], f['EV'], f['wr'], f['sum_pnl'],
                     h1['N'], h1['EV'], h1['wr'], h1['sum_pnl'],
                     h2['N'], h2['EV'], h2['wr'], h2['sum_pnl'],
                     consistent])
    rpt.add_table(headers, rows, highlight_pnl_cols=[2, 4, 6, 8, 10, 12])

    # ============================================================
    #  Part 1: 前60天 vs 后60天 — 3场景基础
    # ============================================================
    rpt.add_section('Part 1: 前60天 vs 后60天 — 3场景基础EV',
                    f'前半: {t_min.strftime("%m-%d")}~{t_mid.strftime("%m-%d")} | '
                    f'后半: {t_mid.strftime("%m-%d")}~{t_max.strftime("%m-%d")}')

    full_results = analyze_window(df_all, '全量120天')
    first_results = analyze_window(df_first, '前60天')
    second_results = analyze_window(df_second, '后60天')

    headers = ['场景', '全量N', '全量EV', '全量WR%', '全量Σ',
               '前半N', '前半EV', '前半WR%', '前半Σ',
               '后半N', '后半EV', '后半WR%', '后半Σ', '方向一致']
    rows = []
    for sc in [1, 2, 3]:
        f = full_results[sc]
        h1 = first_results[sc]
        h2 = second_results[sc]
        consistent = '✓' if (h1['EV'] > 0 and h2['EV'] > 0) or (h1['EV'] <= 0 and h2['EV'] <= 0) else '✗'
        rows.append([SCENARIO_NAMES[sc],
                     f['N'], f['EV'], f['wr'], f['sum_pnl'],
                     h1['N'], h1['EV'], h1['wr'], h1['sum_pnl'],
                     h2['N'], h2['EV'], h2['wr'], h2['sum_pnl'],
                     consistent])
    # 汇总行
    f_all_n = sum(full_results[sc]['N'] for sc in [1, 2, 3])
    f_all_sum = sum(full_results[sc]['sum_pnl'] for sc in [1, 2, 3])
    h1_all_n = sum(first_results[sc]['N'] for sc in [1, 2, 3])
    h1_all_sum = sum(first_results[sc]['sum_pnl'] for sc in [1, 2, 3])
    h2_all_n = sum(second_results[sc]['N'] for sc in [1, 2, 3])
    h2_all_sum = sum(second_results[sc]['sum_pnl'] for sc in [1, 2, 3])
    rows.append(['合计', f_all_n, '-', '-', f_all_sum,
                 h1_all_n, '-', '-', h1_all_sum,
                 h2_all_n, '-', '-', h2_all_sum, '-'])

    rpt.add_table(headers, rows, highlight_pnl_cols=[2, 4, 6, 8, 10, 12])

    # ============================================================
    #  Part 2: 前60天 vs 后60天 — 带过滤和加码
    # ============================================================
    rpt.add_section('Part 2: 带过滤/加码规则的时间稳定性',
                    '验证仓位加码规则（ER5变化过滤、ER40加码等）在两半段是否一致')

    full_detailed = analyze_with_filters(df_all)
    first_detailed = analyze_with_filters(df_first)
    second_detailed = analyze_with_filters(df_second)

    for sc in [1, 2, 3]:
        rpt.add_section(f'{SCENARIO_NAMES[sc]}')
        fd = full_detailed[sc]
        h1d = first_detailed[sc]
        h2d = second_detailed[sc]

        sub_headers = ['指标', '全量N', '全量EV', '全量WR%', '全量Σ',
                       '前半N', '前半EV', '前半WR%', '前半Σ',
                       '后半N', '后半EV', '后半WR%', '后半Σ', '方向一致']
        sub_rows = []

        # 基础
        def make_row(label, f_ev, h1_ev, h2_ev):
            consistent = '✓' if (h1_ev['EV'] > 0 and h2_ev['EV'] > 0) or (h1_ev['EV'] <= 0 and h2_ev['EV'] <= 0) else '✗'
            return [label,
                    f_ev['N'], f_ev['EV'], f_ev['wr'], f_ev['sum_pnl'],
                    h1_ev['N'], h1_ev['EV'], h1_ev['wr'], h1_ev['sum_pnl'],
                    h2_ev['N'], h2_ev['EV'], h2_ev['wr'], h2_ev['sum_pnl'],
                    consistent]

        sub_rows.append(make_row('基础', fd['base'], h1d['base'], h2d['base']))

        if sc == 1:
            sub_rows.append(make_row('ER5变化>-0.41过滤后', fd['filtered'], h1d['filtered'], h2d['filtered']))
            if fd['boost_er40']:
                sub_rows.append(make_row('ER40≥0.42(2倍仓)', fd['boost_er40'], h1d['boost_er40'], h2d['boost_er40']))
            if fd['boost_er5']:
                sub_rows.append(make_row('ER5变化≥0.50(2倍仓)', fd['boost_er5'], h1d['boost_er5'], h2d['boost_er5']))

        if sc == 2 and fd['boost_er40_chg']:
            sub_rows.append(make_row('ER40变化≥0.14(2倍仓)', fd['boost_er40_chg'], h1d['boost_er40_chg'], h2d['boost_er40_chg']))

        rpt.add_table(sub_headers, sub_rows, highlight_pnl_cols=[2, 4, 6, 8, 10, 12])

    # ============================================================
    #  Part 3: 3折交叉验证
    # ============================================================
    rpt.add_section('Part 3: 3折交叉验证',
                    '将120天切成3段（各~40天），轮流做样本外')

    t_range = t_max - t_min
    t_cut1 = t_min + t_range / 3
    t_cut2 = t_min + t_range * 2 / 3

    folds = [
        ('Fold1(前40天)', df_all[df_all['time'] <= t_cut1]),
        ('Fold2(中40天)', df_all[(df_all['time'] > t_cut1) & (df_all['time'] <= t_cut2)]),
        ('Fold3(后40天)', df_all[df_all['time'] > t_cut2]),
    ]

    rpt.add_text(f'切分: ~{t_cut1.strftime("%m-%d")} | ~{t_cut2.strftime("%m-%d")} | ~{t_max.strftime("%m-%d")}')

    for sc in [1, 2, 3]:
        pnl_col = SCENARIO_PNL[sc]
        reason_col = SCENARIO_REASON[sc]

        headers_3f = ['折段', 'N', 'EV', 'WR%', 'PR', 'Σ', 'avg_pnl%']
        rows_3f = []

        evs = []
        for fold_name, fold_df in folds:
            sub = fold_df[fold_df['scenario'] == sc].copy()
            if reason_col in sub.columns:
                sub = sub[sub[reason_col] != 'backtest_end']
            ev = calc_ev(sub[pnl_col].tolist())
            evs.append(ev['EV'])
            rows_3f.append([fold_name, ev['N'], ev['EV'], ev['wr'], ev['pr'],
                           ev['sum_pnl'], ev['avg_pnl']])

        # 一致性判断
        pos_count = sum(1 for e in evs if e > 0)
        neg_count = sum(1 for e in evs if e <= 0)
        consistency = f'{pos_count}/3折为正' if pos_count > 0 else '全部为负'

        rpt.add_section(f'{SCENARIO_NAMES[sc]} — {consistency}')
        rpt.add_table(headers_3f, rows_3f, highlight_pnl_cols=[2, 5, 6])

    # ============================================================
    #  Part 4: 按品种组验证
    # ============================================================
    rpt.add_section('Part 4: 品种组稳定性',
                    '高波动10个 vs 常规22个，前后半段是否方向一致')

    for group_name, group_filter in [('高波动10', True), ('常规22', False)]:
        if group_filter:
            df_group = df_all[df_all['sym_key'].isin(HIGH_VOL)]
        else:
            df_group = df_all[~df_all['sym_key'].isin(HIGH_VOL)]

        df_g_first = df_group[df_group['time'] <= t_mid]
        df_g_second = df_group[df_group['time'] > t_mid]

        g_full = analyze_window(df_group, f'{group_name}全量')
        g_first = analyze_window(df_g_first, f'{group_name}前半')
        g_second = analyze_window(df_g_second, f'{group_name}后半')

        headers_g = ['场景', '全量N', '全量EV', '前半N', '前半EV', '后半N', '后半EV', '方向一致']
        rows_g = []
        for sc in [1, 2, 3]:
            f = g_full[sc]
            h1 = g_first[sc]
            h2 = g_second[sc]
            consistent = '✓' if (h1['EV'] > 0 and h2['EV'] > 0) or (h1['EV'] <= 0 and h2['EV'] <= 0) else '✗'
            rows_g.append([f'场景{sc}', f['N'], f['EV'], h1['N'], h1['EV'], h2['N'], h2['EV'], consistent])

        rpt.add_section(f'{group_name}')
        rpt.add_table(headers_g, rows_g, highlight_pnl_cols=[2, 4, 6])

    # ============================================================
    #  Part 5: 逐月EV走势
    # ============================================================
    rpt.add_section('Part 5: 逐月EV走势', '按自然月统计，观察EV是否有明显衰减趋势')

    df_all['month'] = df_all['time'].dt.to_period('M')
    months = sorted(df_all['month'].unique())

    for sc in [1, 2, 3]:
        pnl_col = SCENARIO_PNL[sc]
        reason_col = SCENARIO_REASON[sc]

        headers_m = ['月份', 'N', 'EV', 'WR%', 'Σ', 'avg_pnl%']
        rows_m = []

        for m in months:
            sub = df_all[(df_all['month'] == m) & (df_all['scenario'] == sc)].copy()
            if reason_col in sub.columns:
                sub = sub[sub[reason_col] != 'backtest_end']
            ev = calc_ev(sub[pnl_col].tolist())
            if ev['N'] == 0:
                continue
            rows_m.append([str(m), ev['N'], ev['EV'], ev['wr'], ev['sum_pnl'], ev['avg_pnl']])

        if rows_m:
            rpt.add_section(f'{SCENARIO_NAMES[sc]}')
            rpt.add_table(headers_m, rows_m, highlight_pnl_cols=[2, 4, 5])

    # ============================================================
    #  Part 6: 综合判定
    # ============================================================
    rpt.add_section('Part 6: 综合判定')

    # 统计一致性
    summary_lines = []
    for sc in [1, 2, 3]:
        f = full_results[sc]
        h1 = first_results[sc]
        h2 = second_results[sc]
        half_ok = (h1['EV'] > 0 and h2['EV'] > 0)

        # 3折
        fold_evs = []
        for _, fold_df in folds:
            pnl_col = SCENARIO_PNL[sc]
            reason_col = SCENARIO_REASON[sc]
            sub = fold_df[fold_df['scenario'] == sc].copy()
            if reason_col in sub.columns:
                sub = sub[sub[reason_col] != 'backtest_end']
            ev = calc_ev(sub[pnl_col].tolist())
            fold_evs.append(ev['EV'])
        fold_pos = sum(1 for e in fold_evs if e > 0)

        if half_ok and fold_pos >= 2:
            verdict = '✓ 通过（前后一致 + ≥2/3折为正）'
            color = '#3fb950'
        elif half_ok or fold_pos >= 2:
            verdict = '△ 部分通过（需关注）'
            color = '#d29922'
        else:
            verdict = '✗ 未通过（过拟合风险高）'
            color = '#f85149'

        line = f'{SCENARIO_NAMES[sc]}: {verdict} | 全量EV={f["EV"]} | 前半EV={h1["EV"]} 后半EV={h2["EV"]} | 3折={fold_evs}'
        summary_lines.append((line, color))

    for line, color in summary_lines:
        rpt.add_text(line, color=color)

    rpt.add_text('')
    rpt.add_text('注意：120天切分后样本量进一步缩小，尤其场景2/3。'
                 '即使通过也不代表没有过拟合——仅代表在这120天内前后一致。'
                 '真正的样本外验证需要用新的时间段数据（方案2）。', color='#8b949e')

    return rpt


# ============================================================
#  主流程
# ============================================================

if __name__ == '__main__':
    df_all = collect_all_signals()

    print("\n生成报告...")
    rpt = build_report(df_all)

    out_path = 'output/oos_validation.html'
    rpt.save(out_path)

    # 控制台快速摘要
    print("\n" + "=" * 70)
    print("快速摘要")
    print("=" * 70)

    t_mid = df_all['time'].min() + (df_all['time'].max() - df_all['time'].min()) / 2
    df_first = df_all[df_all['time'] <= t_mid]
    df_second = df_all[df_all['time'] > t_mid]

    for sc in [1, 2, 3]:
        pnl_col = SCENARIO_PNL[sc]
        reason_col = SCENARIO_REASON[sc]

        for label, subset in [('全量', df_all), ('前60天', df_first), ('后60天', df_second)]:
            sub = subset[subset['scenario'] == sc].copy()
            if reason_col in sub.columns:
                sub = sub[sub[reason_col] != 'backtest_end']
            ev = calc_ev(sub[pnl_col].tolist())
            print(f"  场景{sc} {label}: N={ev['N']:>4} EV={ev['EV']:>+.2f} WR={ev['wr']:>5.1f}% Σ={ev['sum_pnl']:>+.1f}%")
        print()

    print(f"详细报告: {out_path}")
