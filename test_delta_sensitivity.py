# -*- coding: utf-8 -*-
"""
Delta敏感性分析
===============
测试深度虚值(Δ=0.1)在不同滑点假设下的真实表现。

核心问题：Δ=0.1看起来MFE爆炸，但实际交易中：
1. 深度虚值买卖价差大（滑点远超5%）
2. IV微笑：深度虚值IV > ATM IV，实际权利金更贵
3. 权利金绝对值小，1跳波动就是几十%

本脚本用不同滑点模拟，看哪个Delta真正最优。
"""

import sys
import io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from data_loader import load_all, sym_name, tick_size
from signal_core import (SignalDetector, ExitTracker, SYMBOL_CONFIGS,
                         DEFAULT_STOP_TICKS, classify_scenario, SCENARIO_NAMES)
from stats_utils import calc_ev
from volatility import add_hv
from option_pricing import black76_price, black76_greeks, find_strike_by_delta, R_DEFAULT
from report_engine import Report

# ============================================================
#  配置
# ============================================================

DELTA_LEVELS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
T_DAYS = 15
BARS_PER_DAY = 57
TRADING_DAYS_YEAR = 245
MAX_HOLD_BARS = 144

# 滑点假设矩阵（按Delta调整）
# 深度虚值的买卖价差占比远大于平值
SLIPPAGE_BY_DELTA = {
    0.10: [5, 10, 15, 20, 25],   # 深度虚值：5%~25%
    0.15: [5, 10, 15, 20],
    0.20: [5, 10, 15],
    0.25: [5, 8, 12],
    0.30: [5, 8, 10],
    0.40: [5, 7],
    0.50: [5, 7],
}

# 止盈档位
TP_LEVELS = [50, 80, 100, 150, 200, 300]
BEST_TS = 36  # 6小时时间止损（之前验证的最优区间）


def collect_signals():
    print("加载数据...")
    all_data = load_all(period_min=10, days=170, last_days=None,
                        emas=(10, 20, 120), er_periods=(5, 20, 40), atr_period=14)
    print(f"品种数: {len(all_data)}")

    for sym_key, df in all_data.items():
        add_hv(df, windows=[20, 40])
        df['er5_delta_6'] = df['er_5'] - df['er_5'].shift(6)

    records = []
    LAST_DAYS = 120

    for sym_key, df in sorted(all_data.items()):
        n = len(df)
        signal_start = max(130, n - LAST_DAYS * BARS_PER_DAY)
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

            er5_d6 = row.get('er5_delta_6', 0)
            if pd.isna(er5_d6):
                er5_d6 = 0
            if scenario == 1 and er5_d6 <= -0.41:
                continue

            hv = row.get('hv_20', np.nan)
            if pd.isna(hv) or hv <= 0:
                hv = row.get('hv_40', np.nan)
            if pd.isna(hv) or hv <= 0:
                continue

            records.append({
                'symbol': sym_name(sym_key), 'sym_key': sym_key,
                'idx': i, 'df_key': sym_key,
                'direction': result.direction,
                'type': result.signal_type,
                'scenario': scenario,
                'entry_price': result.entry_price,
                'pullback_extreme': result.pullback_extreme,
                'hv': hv,
                'atr': atr,
            })

    print(f"信号数: {len(records)}笔")
    return records, all_data


def simulate_with_slippage(df, sig, delta, slippage_pct):
    """单笔信号在指定Delta和滑点下的完整模拟"""
    idx = sig['idx']
    n = len(df)
    entry_price = sig['entry_price']
    is_long = sig['direction'] == 'long'
    cp = 'call' if is_long else 'put'
    sigma = sig['hv']
    T_entry = T_DAYS / TRADING_DAYS_YEAR

    K = find_strike_by_delta(entry_price, T_entry, R_DEFAULT, sigma, delta, cp)
    entry_premium = black76_price(entry_price, K, T_entry, R_DEFAULT, sigma, cp)

    if entry_premium <= 0:
        return None

    entry_cost = entry_premium * (1 + slippage_pct / 100)

    # 逐根跟踪
    max_roi = -100
    tp_hits = {}
    roi_at_ts = None  # 时间止损点的ROI

    max_bars = min(MAX_HOLD_BARS, n - idx - 1)

    for j in range(1, max_bars + 1):
        bar = df.iloc[idx + j]

        T_now = max(T_entry - j / (BARS_PER_DAY * TRADING_DAYS_YEAR), 0.0001)

        if is_long:
            best_price = bar['high']
        else:
            best_price = bar['low']

        # 最有利时的期权价
        opt_best = black76_price(best_price, K, T_now, R_DEFAULT, sigma, cp)
        opt_best_rev = opt_best * (1 - slippage_pct / 100)
        roi_best = (opt_best_rev - entry_cost) / entry_cost * 100

        if roi_best > max_roi:
            max_roi = roi_best

        # 止盈触发
        for tp in TP_LEVELS:
            if tp not in tp_hits and roi_best >= tp:
                tp_hits[tp] = j

        # 收盘ROI（用于时间止损）
        opt_close = black76_price(bar['close'], K, T_now, R_DEFAULT, sigma, cp)
        opt_close_rev = opt_close * (1 - slippage_pct / 100)
        roi_close = (opt_close_rev - entry_cost) / entry_cost * 100

        if j == BEST_TS:
            roi_at_ts = roi_close

    # 如果不够K线
    if roi_at_ts is None:
        roi_at_ts = roi_close if max_bars > 0 else -slippage_pct * 2

    # 各止盈+时间止损组合的最终ROI
    results = {
        'scenario': sig['scenario'],
        'delta': delta,
        'slippage': slippage_pct,
        'max_roi': round(max_roi, 1),
        'entry_premium': round(entry_premium, 2),
        'entry_cost': round(entry_cost, 2),
    }

    for tp in TP_LEVELS:
        if tp in tp_hits and tp_hits[tp] <= BEST_TS:
            results[f'tp{tp}_roi'] = tp
        else:
            results[f'tp{tp}_roi'] = roi_at_ts

    return results


def main():
    signals, all_data = collect_signals()

    # 只测场景1和3（场景2已确认不适合期权）
    signals = [s for s in signals if s['scenario'] in [1, 3]]
    print(f"场景1+3信号数: {len(signals)}笔")

    all_records = []
    total = len(signals) * sum(len(v) for v in SLIPPAGE_BY_DELTA.values())
    done = 0

    for sig in signals:
        df = all_data[sig['df_key']]
        for delta in DELTA_LEVELS:
            for slip in SLIPPAGE_BY_DELTA[delta]:
                r = simulate_with_slippage(df, sig, delta, slip)
                if r:
                    all_records.append(r)
                done += 1
        if done % 2000 == 0:
            print(f"  进度: {done}/{total}...")

    df_all = pd.DataFrame(all_records)
    print(f"总记录: {len(df_all)}条")

    # ============================================================
    #  输出报告
    # ============================================================
    rpt = Report('Delta敏感性分析：深度虚值期权真的更好吗？')
    rpt.add_text('核心问题：Delta越低杠杆越高，但滑点和流动性也越差。综合考虑哪个Delta最优？')

    # Part 1: MFE对比（固定5%滑点，纯粹比较杠杆效应）
    rpt.add_section('Part 1: 各Delta的MFE分布（滑点=5%基准）',
                    '不考虑滑点差异，纯看杠杆效应')

    for sc in [1, 3]:
        headers = ['Delta', 'N', 'MFE中位%', 'MFE P75%', 'MFE P90%',
                   '≥50%', '≥100%', '≥200%', '权利金中位']
        rows = []
        for delta in DELTA_LEVELS:
            sub = df_all[(df_all['scenario'] == sc) & (df_all['delta'] == delta) & (df_all['slippage'] == 5)]
            if len(sub) == 0:
                continue
            mfe = sub['max_roi']
            rows.append([
                f'Δ={delta}', len(sub),
                round(mfe.median(), 1), round(mfe.quantile(0.75), 1), round(mfe.quantile(0.90), 1),
                f"{(mfe >= 50).mean() * 100:.1f}%",
                f"{(mfe >= 100).mean() * 100:.1f}%",
                f"{(mfe >= 200).mean() * 100:.1f}%",
                round(sub['entry_premium'].median(), 1),
            ])
        rpt.add_section(f'{SCENARIO_NAMES[sc]}')
        rpt.add_table(headers, rows, highlight_pnl_cols=[2, 3, 4])

    # Part 2: 核心——不同滑点下的EV（止盈+6h时间止损）
    rpt.add_section('Part 2: 不同滑点假设下的EV（止盈+6h时间止损）',
                    '深度虚值的滑点实际可能是15-25%，看看还能不能赚')

    for sc in [1, 3]:
        for tp in [100, 150, 200]:
            headers = ['Delta \\ 滑点'] + [f'{s}%' for s in [5, 8, 10, 12, 15, 20, 25]]
            rows = []
            for delta in DELTA_LEVELS:
                row = [f'Δ={delta}']
                for slip_test in [5, 8, 10, 12, 15, 20, 25]:
                    sub = df_all[(df_all['scenario'] == sc) &
                                 (df_all['delta'] == delta) &
                                 (df_all['slippage'] == slip_test)]
                    if len(sub) == 0:
                        row.append('-')
                        continue
                    pnls = sub[f'tp{tp}_roi'].tolist()
                    ev = calc_ev(pnls)
                    color_tag = '🟢' if ev['EV'] > 0 else '🔴'
                    row.append(f"{ev['EV']}")
                    # row.append(f"{color_tag} {ev['EV']}")
                row.append(row)
                rows.append(row)
            rpt.add_section(f'{SCENARIO_NAMES[sc]} | 止盈+{tp}%')
            rpt.add_table(headers, rows)

    # Part 3: 每个Delta的"滑点容忍度"
    rpt.add_section('Part 3: 滑点容忍度（EV从正变负的临界点）',
                    '各Delta最多能承受多少滑点仍然盈利？')

    for sc in [1, 3]:
        headers = ['Delta', '最优止盈%', '5%滑点EV', '盈亏平衡滑点', '权利金中位']
        rows = []
        for delta in DELTA_LEVELS:
            # 找最优TP（5%滑点下）
            best_tp = None
            best_ev = -999
            for tp in TP_LEVELS:
                sub = df_all[(df_all['scenario'] == sc) &
                             (df_all['delta'] == delta) &
                             (df_all['slippage'] == 5)]
                if len(sub) == 0:
                    continue
                ev = calc_ev(sub[f'tp{tp}_roi'].tolist())
                if ev['EV'] > best_ev:
                    best_ev = ev['EV']
                    best_tp = tp

            if best_tp is None:
                continue

            # 找盈亏平衡滑点
            breakeven_slip = '>25%'
            for slip_test in SLIPPAGE_BY_DELTA[delta]:
                sub = df_all[(df_all['scenario'] == sc) &
                             (df_all['delta'] == delta) &
                             (df_all['slippage'] == slip_test)]
                if len(sub) == 0:
                    continue
                ev = calc_ev(sub[f'tp{tp}_roi'].tolist())
                if ev['EV'] <= 0:
                    breakeven_slip = f'~{slip_test}%'
                    break

            prem = df_all[(df_all['scenario'] == sc) &
                          (df_all['delta'] == delta) &
                          (df_all['slippage'] == 5)]['entry_premium'].median()

            rows.append([
                f'Δ={delta}', f'+{best_tp}%',
                best_ev, breakeven_slip, round(prem, 1)
            ])

        rpt.add_section(f'{SCENARIO_NAMES[sc]}')
        rpt.add_table(headers, rows, highlight_pnl_cols=[2])

    # Part 4: 最终推荐
    rpt.add_section('Part 4: 综合推荐')
    rpt.add_text(
        '<b>选择依据</b>：不是EV最高的Delta，而是在<b>合理滑点假设</b>下EV仍然稳定为正的Delta。',
        color='#e3b341'
    )
    rpt.add_text(
        '深度虚值(Δ≤0.15)：理论EV最高，但权利金极低（几十点），'
        '实际买卖价差可能占权利金15-25%，且流动性差难以成交。',
        color='#8b949e'
    )
    rpt.add_text(
        '浅虚值(Δ=0.2~0.3)：EV略低于深度虚值，但滑点容忍度高得多，'
        '权利金适中（100-300点），成交量大，实际可执行性强。',
        color='#8b949e'
    )

    rpt.save('output/delta_sensitivity.html')
    print("\n报告已保存: output/delta_sensitivity.html")

    # 控制台摘要
    print("\n" + "=" * 80)
    print("核心结论摘要")
    print("=" * 80)
    for sc in [1, 3]:
        print(f"\n{SCENARIO_NAMES[sc]}:")
        for delta in DELTA_LEVELS:
            sub5 = df_all[(df_all['scenario'] == sc) & (df_all['delta'] == delta) & (df_all['slippage'] == 5)]
            if len(sub5) == 0:
                continue
            # 用止盈150%+6h时间止损
            ev5 = calc_ev(sub5['tp150_roi'].tolist())

            # 找最大测试滑点
            max_slip = max(SLIPPAGE_BY_DELTA[delta])
            sub_max = df_all[(df_all['scenario'] == sc) & (df_all['delta'] == delta) & (df_all['slippage'] == max_slip)]
            if len(sub_max) > 0:
                ev_max = calc_ev(sub_max['tp150_roi'].tolist())
                print(f"  Δ={delta}: 滑点5%→EV={ev5['EV']}  滑点{max_slip}%→EV={ev_max['EV']}  "
                      f"权利金中位={sub5['entry_premium'].median():.0f}点")
            else:
                print(f"  Δ={delta}: 滑点5%→EV={ev5['EV']}  权利金中位={sub5['entry_premium'].median():.0f}点")


if __name__ == '__main__':
    main()
