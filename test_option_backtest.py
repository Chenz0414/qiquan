# -*- coding: utf-8 -*-
"""
期权购买策略回测
================
基于3个入场场景的期货信号，模拟不同Delta/到期时间的期权购买收益。

核心流程：
1. 收集3场景信号 + MFE路径 + 持仓时间
2. 用Black-76模型模拟期权P&L
3. 3场景 × 5 Delta × 2 T = 30组对比
4. 输出HTML报告

输出: output/option_strategy.html
"""

import sys
import io
import json
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from data_loader import load_all, sym_name, tick_size, HIGH_VOL
from signal_core import (SignalDetector, ExitTracker, SYMBOL_CONFIGS,
                         DEFAULT_STOP_TICKS, classify_scenario,
                         SCENARIO_NAMES, SCENARIO_PNL_COL, SCENARIO_REASON_COL)
from stats_utils import calc_ev
from volatility import add_hv
from option_pricing import option_pnl, black76_greeks, R_DEFAULT
from report_engine import Report


# ============================================================
#  配置
# ============================================================

DELTA_LEVELS = [0.30, 0.40, 0.50, 0.60, 0.70]
T_DAYS = [15, 30]  # 到期时间（交易日）
MFE_WINDOWS = [6, 12, 24, 48, 72]  # 1h/2h/4h/8h/12h
SLIPPAGE_PCT = 5.0  # 买卖滑点%
BARS_PER_DAY = 57
TRADING_DAYS_YEAR = 245


# ============================================================
#  数据收集
# ============================================================

def collect_signals():
    """收集3场景信号 + MFE路径 + 出场信息 + HV"""
    print("加载数据...")
    all_data = load_all(period_min=10, days=170, last_days=None,
                        emas=(10, 20, 120), er_periods=(5, 20, 40), atr_period=14)
    print(f"品种数: {len(all_data)}")

    # 添加HV和ER走势
    for sym_key, df in all_data.items():
        add_hv(df, windows=[20, 40])
        df['er5_delta_6'] = df['er_5'] - df['er_5'].shift(6)

    records = []
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

            # ER5变化过滤（场景1）
            er5_d6 = row.get('er5_delta_6', 0)
            if pd.isna(er5_d6):
                er5_d6 = 0
            if scenario == 1 and er5_d6 <= -0.41:
                continue

            # HV
            hv = row.get('hv_20', np.nan)
            if pd.isna(hv) or hv <= 0:
                hv = row.get('hv_40', np.nan)
            if pd.isna(hv) or hv <= 0:
                continue

            # 模拟出场（S2和S5.1）
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

            # 确定该场景使用的出场
            exit_key = 'S2' if scenario in [1, 2] else 'S5.1'
            ex = exit_results[exit_key]
            if ex is None or ex.exit_reason == 'backtest_end':
                continue

            # MFE路径
            mfe_vals = {}
            entry_price = result.entry_price
            is_long = result.direction == 'long'
            running_mfe = 0
            running_mae = 0
            for w in MFE_WINDOWS:
                if i + w < n:
                    for k in range(1, w + 1):
                        if i + k >= n:
                            break
                        bar_k = df.iloc[i + k]
                        if is_long:
                            bar_mfe = (bar_k['high'] - entry_price) / entry_price * 100
                            bar_mae = (entry_price - bar_k['low']) / entry_price * 100
                        else:
                            bar_mfe = (entry_price - bar_k['low']) / entry_price * 100
                            bar_mae = (bar_k['high'] - entry_price) / entry_price * 100
                        running_mfe = max(running_mfe, bar_mfe)
                        running_mae = max(running_mae, bar_mae)
                    mfe_vals[f'mfe_{w}'] = round(running_mfe, 4)
                    mfe_vals[f'mae_{w}'] = round(running_mae, 4)

            records.append({
                'symbol': name, 'sym_key': sym_key,
                'direction': result.direction,
                'type': result.signal_type,
                'scenario': scenario,
                'entry_price': entry_price,
                'exit_price': ex.exit_price,
                'bars_held': ex.bars_held,
                'futures_pnl': round(ex.pnl_pct, 4),
                'hv': round(hv, 4),
                'atr': round(atr, 4),
                'er_20': round(er20, 4),
                'dev_atr': round(deviation_atr, 3),
                **mfe_vals,
            })

    print(f"可交易信号（排除backtest_end）: {len(records)}笔")
    return pd.DataFrame(records)


# ============================================================
#  期权P&L计算
# ============================================================

def run_option_simulation(df_signals):
    """对每笔信号 × 每个Delta × 每个T，计算期权ROI"""
    results = []

    for _, row in df_signals.iterrows():
        cp = 'call' if row['direction'] == 'long' else 'put'

        for delta in DELTA_LEVELS:
            for t_days in T_DAYS:
                T = t_days / TRADING_DAYS_YEAR

                opt = option_pnl(
                    F_entry=row['entry_price'],
                    F_exit=row['exit_price'],
                    T_entry=T,
                    bars_held=row['bars_held'],
                    sigma=row['hv'],
                    target_delta=delta,
                    cp=cp,
                    r=R_DEFAULT,
                    slippage_pct=SLIPPAGE_PCT,
                )

                if opt is None:
                    continue

                results.append({
                    'scenario': row['scenario'],
                    'symbol': row['symbol'],
                    'direction': row['direction'],
                    'delta': delta,
                    't_days': t_days,
                    'futures_pnl': row['futures_pnl'],
                    'option_roi': opt['roi_pct'],
                    'entry_premium': opt['entry_premium'],
                    'exit_premium': opt['exit_premium'],
                    'K': opt['K'],
                    'actual_delta': opt['delta_entry'],
                    'theta_daily': opt['theta_daily'],
                    'bars_held': opt['bars_held'],
                    'T_exit_days': opt['T_exit_days'],
                })

    return pd.DataFrame(results)


# ============================================================
#  MFE分析
# ============================================================

def mfe_summary(df_signals):
    """按场景汇总MFE分布"""
    summary = {}
    for sc in [1, 2, 3]:
        sub = df_signals[df_signals['scenario'] == sc]
        if len(sub) == 0:
            continue
        stats = {'N': len(sub)}
        for w in MFE_WINDOWS:
            col = f'mfe_{w}'
            if col in sub.columns:
                vals = sub[col].dropna()
                if len(vals) > 0:
                    stats[f'mfe_{w}_p50'] = round(vals.median(), 2)
                    stats[f'mfe_{w}_p75'] = round(vals.quantile(0.75), 2)
                    stats[f'mfe_{w}_p90'] = round(vals.quantile(0.90), 2)
                    stats[f'mfe_{w}_burst3'] = round((vals >= 3.0).mean() * 100, 1)
        # 持仓时间分布
        stats['bars_p50'] = round(sub['bars_held'].median(), 0)
        stats['bars_p75'] = round(sub['bars_held'].quantile(0.75), 0)
        stats['hours_p50'] = round(sub['bars_held'].median() * 10 / 60, 1)
        # HV分布
        stats['hv_p50'] = round(sub['hv'].median() * 100, 1)
        stats['hv_p25'] = round(sub['hv'].quantile(0.25) * 100, 1)
        stats['hv_p75'] = round(sub['hv'].quantile(0.75) * 100, 1)
        summary[sc] = stats
    return summary


# ============================================================
#  HTML报告
# ============================================================

def build_report(df_signals, df_options, mfe_stats):
    rpt = Report('期权购买策略研究')

    rpt.add_text(f'3场景 × 5 Delta × 2 T | 32品种 × 120天 | Black-76模型 | 滑点{SLIPPAGE_PCT}%')

    # ====== Part 1: MFE分布 ======
    rpt.add_section('Part 1: 3场景MFE分布（期货价格路径）',
                    '信号后不同时间窗口的最大有利偏移百分比')

    headers_mfe = ['场景', 'N', '持仓中位(h)',
                   'MFE_1h P50', 'MFE_2h P50', 'MFE_4h P50', 'MFE_8h P50',
                   'MFE_8h P90', '爆发率(>3%)', 'HV中位%']
    rows_mfe = []
    for sc in [1, 2, 3]:
        s = mfe_stats.get(sc, {})
        if not s:
            continue
        rows_mfe.append([
            SCENARIO_NAMES[sc], s['N'], s.get('hours_p50', '-'),
            s.get('mfe_6_p50', '-'), s.get('mfe_12_p50', '-'),
            s.get('mfe_24_p50', '-'), s.get('mfe_48_p50', '-'),
            s.get('mfe_48_p90', '-'), f"{s.get('mfe_48_burst3', 0)}%",
            f"{s.get('hv_p50', '-')}%",
        ])
    rpt.add_table(headers_mfe, rows_mfe, highlight_pnl_cols=[3, 4, 5, 6, 7])

    # ====== Part 2: Delta × T 对比矩阵 ======
    rpt.add_section('Part 2: Delta × 到期时间 对比矩阵',
                    '每场景的期权ROI统计（EV=胜率×盈亏比-败率）')

    for sc in [1, 2, 3]:
        sc_data = df_options[df_options['scenario'] == sc]
        if len(sc_data) == 0:
            continue

        # 期货基准
        futures_pnls = df_signals[df_signals['scenario'] == sc]['futures_pnl'].tolist()
        f_ev = calc_ev(futures_pnls)

        rpt.add_section(f'{SCENARIO_NAMES[sc]}',
                        f'期货基准: N={f_ev["N"]} EV={f_ev["EV"]} WR={f_ev["wr"]}% Σ={f_ev["sum_pnl"]}%')

        headers = ['Delta', 'T(天)', 'N', 'EV', 'WR%', 'PR',
                   'avg_ROI%', '中位ROI%', 'ROI>50%占比', 'avg亏损%', 'Σ ROI%']
        rows = []

        for t_days in T_DAYS:
            for delta in DELTA_LEVELS:
                sub = sc_data[(sc_data['delta'] == delta) & (sc_data['t_days'] == t_days)]
                if len(sub) == 0:
                    continue
                rois = sub['option_roi'].tolist()
                ev = calc_ev(rois)
                avg_roi = round(np.mean(rois), 1)
                med_roi = round(np.median(rois), 1)
                big_win = round((sub['option_roi'] > 50).mean() * 100, 1)
                losses = sub[sub['option_roi'] <= 0]['option_roi']
                avg_loss = round(losses.mean(), 1) if len(losses) > 0 else 0

                rows.append([f'Δ={delta}', t_days, ev['N'], ev['EV'], ev['wr'],
                            ev['pr'], avg_roi, med_roi, f'{big_win}%', avg_loss, ev['sum_pnl']])

        rpt.add_table(headers, rows, highlight_pnl_cols=[3, 6, 7, 10])

    # ====== Part 3: 最优组合推荐 ======
    rpt.add_section('Part 3: 每场景最优Delta+T推荐')

    for sc in [1, 2, 3]:
        sc_data = df_options[df_options['scenario'] == sc]
        if len(sc_data) == 0:
            continue

        best_ev = -999
        best_combo = None

        for t_days in T_DAYS:
            for delta in DELTA_LEVELS:
                sub = sc_data[(sc_data['delta'] == delta) & (sc_data['t_days'] == t_days)]
                if len(sub) < 10:
                    continue
                rois = sub['option_roi'].tolist()
                ev = calc_ev(rois)
                if ev['EV'] > best_ev:
                    best_ev = ev['EV']
                    best_combo = {
                        'delta': delta, 't_days': t_days,
                        'ev': ev, 'avg_roi': round(np.mean(rois), 1),
                        'med_roi': round(np.median(rois), 1),
                    }

        if best_combo:
            b = best_combo
            e = b['ev']
            rpt.add_text(
                f'{SCENARIO_NAMES[sc]}: '
                f'<b>Delta={b["delta"]}, T={b["t_days"]}天</b> | '
                f'EV={e["EV"]} WR={e["wr"]}% PR={e["pr"]} | '
                f'avg_ROI={b["avg_roi"]}% med_ROI={b["med_roi"]}% Σ={e["sum_pnl"]}%',
                color='#3fb950' if e['EV'] > 0 else '#f85149'
            )

    # ====== Part 4: 期权 vs 期货 同本金对比 ======
    rpt.add_section('Part 4: 期权 vs 期货收益对比（同本金1万元）',
                    '期货按10%保证金计算杠杆；期权以权利金为全部投入。两者统一到"投入1万元赚多少元"。')

    for sc in [1, 2, 3]:
        sc_data = df_options[df_options['scenario'] == sc]
        sc_signals = df_signals[df_signals['scenario'] == sc]
        if len(sc_data) == 0:
            continue

        # 找最优组合
        best_ev = -999
        best_delta = 0.5
        best_t = 15
        for t_days in T_DAYS:
            for delta in DELTA_LEVELS:
                sub = sc_data[(sc_data['delta'] == delta) & (sc_data['t_days'] == t_days)]
                if len(sub) < 10:
                    continue
                ev = calc_ev(sub['option_roi'].tolist())
                if ev['EV'] > best_ev:
                    best_ev = ev['EV']
                    best_delta = delta
                    best_t = t_days

        opt_sub = sc_data[(sc_data['delta'] == best_delta) & (sc_data['t_days'] == best_t)]

        # 同本金对比：
        # 期货：1万保证金 → 控制 1万/10% = 10万面值 → 赚 futures_pnl% × 10万
        # 期权：1万权利金 → 赚 option_roi% × 1万
        # 统一到"每投入1万元的收益"
        CAPITAL = 10000
        MARGIN_RATIO = 0.10  # 期货保证金比例

        # 期货：每投入1万保证金的收益（元）
        f_pnls_yuan = sc_signals['futures_pnl'] / 100 * (CAPITAL / MARGIN_RATIO)  # pnl% × 面值
        f_pnls_pct = f_pnls_yuan / CAPITAL * 100  # 转为本金收益率%

        # 期权：每投入1万权利金的收益（元）
        o_pnls_pct = opt_sub['option_roi']  # 已经是权利金收益率%

        f_ev = calc_ev(f_pnls_pct.tolist())
        o_ev = calc_ev(o_pnls_pct.tolist())

        # 计算最大单笔亏损
        f_max_loss = round(f_pnls_pct.min(), 1) if len(f_pnls_pct) > 0 else 0
        o_max_loss = round(o_pnls_pct.min(), 1) if len(o_pnls_pct) > 0 else 0

        headers_cmp = ['指标', f'期货(1万保证金)', f'期权Δ={best_delta},T={best_t}天(1万权利金)']
        rows_cmp = [
            ['样本数', f_ev['N'], o_ev['N']],
            ['EV', f_ev['EV'], o_ev['EV']],
            ['胜率%', f_ev['wr'], o_ev['wr']],
            ['盈亏比', f_ev['pr'], o_ev['pr']],
            ['平均本金收益%', f_ev['avg_pnl'], o_ev['avg_pnl']],
            ['累计本金收益%', f_ev['sum_pnl'], o_ev['sum_pnl']],
            ['最大单笔亏损%', f_max_loss, o_max_loss],
            ['最大亏损封顶', '无（可能穿仓）', '100%（权利金归零）'],
        ]
        rpt.add_section(f'{SCENARIO_NAMES[sc]}')
        rpt.add_table(headers_cmp, rows_cmp, highlight_pnl_cols=[1, 2])

    # ====== Part 5: Theta影响分析 ======
    rpt.add_section('Part 5: Theta衰减影响',
                    '不同Delta下平均每天Theta占权利金的比例')

    for sc in [1, 2, 3]:
        sc_data = df_options[df_options['scenario'] == sc]
        if len(sc_data) == 0:
            continue

        headers_th = ['Delta', 'T(天)', '权利金均值', 'Theta/天', 'Theta/权利金%',
                      '持仓中位(天)', 'Theta总损耗%']
        rows_th = []

        for t_days in T_DAYS:
            for delta in DELTA_LEVELS:
                sub = sc_data[(sc_data['delta'] == delta) & (sc_data['t_days'] == t_days)]
                if len(sub) < 5:
                    continue
                avg_prem = sub['entry_premium'].mean()
                avg_theta = sub['theta_daily'].mean()
                theta_pct = abs(avg_theta) / avg_prem * 100 if avg_prem > 0 else 0
                med_hold_days = sub['bars_held'].median() * 10 / 60 / 9.5  # 转交易日
                total_theta_pct = theta_pct * med_hold_days

                rows_th.append([f'Δ={delta}', t_days,
                               round(avg_prem, 2), round(avg_theta, 4),
                               f'{theta_pct:.2f}%', round(med_hold_days, 1),
                               f'{total_theta_pct:.1f}%'])

        if rows_th:
            rpt.add_section(f'{SCENARIO_NAMES[sc]}')
            rpt.add_table(headers_th, rows_th)

    # ====== Part 6: 品种流动性排名 ======
    rpt.add_section('Part 6: 品种期权流动性参考')

    try:
        import os
        probe_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'probe_results.json')
        with open(probe_path, 'r') as f:
            probe = json.load(f)
        results = probe.get('results', {})
        liquidity = []
        for sym_key, info in results.items():
            if info.get('has_options'):
                liquidity.append({
                    'name': info['name'],
                    'count': info['option_count'],
                    'main': info.get('main_contract', ''),
                    'group': '高波动' if sym_key in HIGH_VOL else '常规',
                })
        liquidity.sort(key=lambda x: x['count'], reverse=True)

        headers_liq = ['排名', '品种', '分组', '期权合约数', '主力合约']
        rows_liq = []
        for i, item in enumerate(liquidity[:20]):
            rows_liq.append([i + 1, item['name'], item['group'],
                            item['count'], item['main']])
        rpt.add_table(headers_liq, rows_liq)
        rpt.add_text('合约数越多通常流动性越好。实盘选择时优先成交量≥100手、持仓≥500手的合约。',
                     color='#8b949e')
    except Exception:
        rpt.add_text('probe_results.json 未找到，跳过流动性排名', color='#f85149')

    # ====== Part 7: 总结推荐规则 ======
    rpt.add_section('Part 7: 最终推荐规则总结')

    rpt.add_text('<b>期权选择流程：信号触发 → 确认场景 → 按规则选合约</b>')

    for sc in [1, 2, 3]:
        sc_data = df_options[df_options['scenario'] == sc]
        if len(sc_data) == 0:
            continue

        best_ev = -999
        best_delta = 0.5
        best_t = 15
        for t_days in T_DAYS:
            for delta in DELTA_LEVELS:
                sub = sc_data[(sc_data['delta'] == delta) & (sc_data['t_days'] == t_days)]
                if len(sub) < 10:
                    continue
                ev = calc_ev(sub['option_roi'].tolist())
                if ev['EV'] > best_ev:
                    best_ev = ev['EV']
                    best_delta = delta
                    best_t = t_days

        cp_desc = '看涨(Call)' if True else '看跌(Put)'  # 方向由信号决定
        rpt.add_text(
            f'<b>{SCENARIO_NAMES[sc]}</b><br>'
            f'&nbsp;&nbsp;Delta: {best_delta} | 到期: ≥{best_t}天 | '
            f'方向: 做多买Call，做空买Put<br>'
            f'&nbsp;&nbsp;选合约: 当月ATM附近，成交量最大，日成交≥100手<br>'
            f'&nbsp;&nbsp;出场: 跟随期货出场信号平仓期权'
        )

    rpt.add_text('')
    rpt.add_text(
        '<b>重要假设与局限</b>：<br>'
        '1. 用历史波动率(HV)近似隐含波动率(IV)，实际IV通常高于HV<br>'
        '2. 假设持仓期间IV不变，实际IV会随行情波动<br>'
        '3. 滑点按理论价5%扣除，流动性差的品种实际滑点可能更大<br>'
        '4. 未考虑Gamma效应的路径依赖（只看入场和出场两个点）<br>'
        '5. 真正的验证需要用实际期权历史价格数据',
        color='#8b949e'
    )

    return rpt


# ============================================================
#  主流程
# ============================================================

if __name__ == '__main__':
    # Step 1: 收集信号
    df_signals = collect_signals()
    for sc in [1, 2, 3]:
        n = len(df_signals[df_signals['scenario'] == sc])
        print(f"  场景{sc}: {n}笔")

    # Step 2: MFE统计
    print("\nMFE统计...")
    mfe_stats = mfe_summary(df_signals)
    for sc, s in mfe_stats.items():
        print(f"  场景{sc}: N={s['N']} MFE_8h_P50={s.get('mfe_48_p50','-')}% "
              f"爆发率={s.get('mfe_48_burst3',0)}% 持仓中位={s.get('hours_p50','-')}h "
              f"HV中位={s.get('hv_p50','-')}%")

    # Step 3: 期权P&L模拟
    print(f"\n期权模拟: {len(DELTA_LEVELS)}Delta × {len(T_DAYS)}T × {len(df_signals)}笔...")
    df_options = run_option_simulation(df_signals)
    print(f"  生成 {len(df_options)} 条模拟记录")

    # Step 4: 控制台快速摘要
    print("\n" + "=" * 80)
    print("快速摘要: 每场景最优Delta+T")
    print("=" * 80)

    for sc in [1, 2, 3]:
        sc_data = df_options[df_options['scenario'] == sc]
        if len(sc_data) == 0:
            continue
        best_ev = -999
        best_info = ""
        for t_days in T_DAYS:
            for delta in DELTA_LEVELS:
                sub = sc_data[(sc_data['delta'] == delta) & (sc_data['t_days'] == t_days)]
                if len(sub) < 10:
                    continue
                rois = sub['option_roi'].tolist()
                ev = calc_ev(rois)
                if ev['EV'] > best_ev:
                    best_ev = ev['EV']
                    best_info = (f"Δ={delta} T={t_days}天 | "
                                f"EV={ev['EV']} WR={ev['wr']}% PR={ev['pr']} "
                                f"avg={round(np.mean(rois),1)}% Σ={ev['sum_pnl']}%")
        print(f"  场景{sc}: {best_info}")

    # Step 5: HTML报告
    print("\n生成报告...")
    rpt = build_report(df_signals, df_options, mfe_stats)
    rpt.save('output/option_strategy.html')

    print("\n完成。")
