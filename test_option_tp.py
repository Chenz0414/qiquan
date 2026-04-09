# -*- coding: utf-8 -*-
"""
期权止盈策略研究
================
核心改进：计算期权MFE路径（逐根K线的期权理论价），
然后对比不同ROI%止盈档位的表现。

关键问题：期权涨到+50%/+100%/+200%/+300%时止盈 vs 跟期货止损，哪个更好？

输出: output/option_tp_strategy.html
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
                         SCENARIO_NAMES)
from stats_utils import calc_ev
from volatility import add_hv
from option_pricing import black76_price, black76_greeks, find_strike_by_delta, R_DEFAULT
from report_engine import Report


# ============================================================
#  配置
# ============================================================

DELTA_LEVELS = [0.10, 0.20, 0.30, 0.50]  # 深虚值到平值
T_DAYS_DEFAULT = 15
SLIPPAGE_PCT = 5.0
BARS_PER_DAY = 57
TRADING_DAYS_YEAR = 245
MAX_HOLD_BARS = 144  # 最长观察24小时

# 止盈档位（期权ROI%）
TP_LEVELS = [30, 50, 80, 100, 150, 200, 300]

# 时间止损：如果N根K线内没到止盈，按当时价格平仓
TIME_STOPS = [6, 12, 18, 24, 36, 48]  # 1h/2h/3h/4h/6h/8h


# ============================================================
#  数据收集（同 test_option_backtest.py）
# ============================================================

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
                'symbol': name, 'sym_key': sym_key,
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


# ============================================================
#  期权MFE路径计算
# ============================================================

def compute_option_mfe_path(df, sig, delta, t_days):
    """
    计算单笔信号的期权MFE路径。

    逐根K线计算期权理论价变化，记录：
    - 每根K线的期权ROI%
    - 最大期权ROI%（期权MFE）
    - 到达各止盈档位的根数（如果到达）
    """
    idx = sig['idx']
    n = len(df)
    entry_price = sig['entry_price']
    is_long = sig['direction'] == 'long'
    cp = 'call' if is_long else 'put'
    sigma = sig['hv']
    T_entry = t_days / TRADING_DAYS_YEAR

    # 入场期权
    K = find_strike_by_delta(entry_price, T_entry, R_DEFAULT, sigma, delta, cp)
    entry_premium = black76_price(entry_price, K, T_entry, R_DEFAULT, sigma, cp)

    if entry_premium <= 0:
        return None

    entry_cost = entry_premium * (1 + SLIPPAGE_PCT / 100)

    # 逐根跟踪
    max_roi = -100
    max_roi_bar = 0
    roi_path = []  # (bar_offset, roi%)

    # 止盈触发记录
    tp_hits = {}  # tp_level -> {'bar': offset, 'roi': roi}

    # 期货追踪止损（用于对比）
    tracker = ExitTracker(
        direction=sig['direction'], entry_price=entry_price,
        pullback_extreme=sig['pullback_extreme'],
        tick_size=tick_size(sig['sym_key']),
        stop_ticks=DEFAULT_STOP_TICKS,
    )
    futures_exit_bar = None
    futures_exit_roi = None  # 期货止损时点的期权ROI

    max_bars = min(MAX_HOLD_BARS, n - idx - 1)

    for j in range(1, max_bars + 1):
        bar = df.iloc[idx + j]
        prev_bar = df.iloc[idx + j - 1]

        if pd.isna(bar['ema10']):
            continue

        # 时间推移
        T_now = max(T_entry - j / (BARS_PER_DAY * TRADING_DAYS_YEAR), 0.0001)

        # 用bar的high/low计算期权峰值（更精确的MFE）
        if is_long:
            # 做多：high时期权最贵
            best_price_in_bar = bar['high']
            worst_price_in_bar = bar['low']
        else:
            # 做空：low时期权最贵
            best_price_in_bar = bar['low']
            worst_price_in_bar = bar['high']

        # 最有利时的期权价
        opt_best = black76_price(best_price_in_bar, K, T_now, R_DEFAULT, sigma, cp)
        opt_best_revenue = opt_best * (1 - SLIPPAGE_PCT / 100)
        roi_best = (opt_best_revenue - entry_cost) / entry_cost * 100

        # 收盘时的期权价（实际可执行）
        opt_close = black76_price(bar['close'], K, T_now, R_DEFAULT, sigma, cp)
        opt_close_revenue = opt_close * (1 - SLIPPAGE_PCT / 100)
        roi_close = (opt_close_revenue - entry_cost) / entry_cost * 100

        roi_path.append((j, round(roi_close, 2)))

        # 更新MFE
        if roi_best > max_roi:
            max_roi = roi_best
            max_roi_bar = j

        # 检查止盈触发（用盘中最优价）
        for tp in TP_LEVELS:
            if tp not in tp_hits and roi_best >= tp:
                tp_hits[tp] = {'bar': j, 'roi': round(roi_best, 1)}

        # 期货追踪止损
        if futures_exit_bar is None:
            exit_key = 'S2' if sig['scenario'] in [1, 2] else 'S5.1'
            exit_events, _ = tracker.process_bar(
                close=bar['close'], high=bar['high'], low=bar['low'],
                ema10=bar['ema10'], prev_close=prev_bar['close'],
                prev_high=prev_bar['high'], prev_low=prev_bar['low'],
            )
            for ev in exit_events:
                if ev.strategy == exit_key:
                    futures_exit_bar = j
                    # 期货止损时的期权价
                    F_exit = ev.exit_price
                    opt_at_exit = black76_price(F_exit, K, T_now, R_DEFAULT, sigma, cp)
                    opt_exit_rev = opt_at_exit * (1 - SLIPPAGE_PCT / 100)
                    futures_exit_roi = round((opt_exit_rev - entry_cost) / entry_cost * 100, 2)

    return {
        'scenario': sig['scenario'],
        'symbol': sig['symbol'],
        'direction': sig['direction'],
        'delta': delta,
        'entry_premium': round(entry_cost, 2),
        'K': K,
        'max_roi': round(max_roi, 1),
        'max_roi_bar': max_roi_bar,
        'max_roi_hours': round(max_roi_bar * 10 / 60, 1),
        'tp_hits': tp_hits,
        'futures_exit_bar': futures_exit_bar,
        'futures_exit_roi': futures_exit_roi,
        'roi_path': roi_path,
    }


# ============================================================
#  止盈策略模拟
# ============================================================

def simulate_tp_strategies(all_results):
    """
    对每笔信号模拟不同止盈策略的最终ROI。

    策略矩阵：
    - 固定ROI止盈（到达就走）
    - 固定ROI止盈 + 时间止损（N根内没到就按市价走）
    - 跟期货止损走（基线）
    """
    records = []

    for r in all_results:
        if r is None:
            continue

        base = {
            'scenario': r['scenario'], 'symbol': r['symbol'],
            'delta': r['delta'], 'max_roi': r['max_roi'],
            'max_roi_bar': r['max_roi_bar'],
            'max_roi_hours': r['max_roi_hours'],
            'entry_premium': r['entry_premium'],
        }

        # 策略0：跟期货止损
        base['futures_stop_roi'] = r['futures_exit_roi'] if r['futures_exit_roi'] is not None else 0
        base['futures_stop_bar'] = r['futures_exit_bar'] if r['futures_exit_bar'] is not None else MAX_HOLD_BARS

        # 各止盈档位
        for tp in TP_LEVELS:
            hit = r['tp_hits'].get(tp)
            if hit:
                base[f'tp{tp}_hit'] = 1
                base[f'tp{tp}_roi'] = tp  # 止盈就是目标ROI（扣完滑点后实际可能略高）
                base[f'tp{tp}_bar'] = hit['bar']
            else:
                base[f'tp{tp}_hit'] = 0
                base[f'tp{tp}_roi'] = 0  # 没到止盈，需要配合时间止损
                base[f'tp{tp}_bar'] = 0

        # 止盈+时间止损组合：到达止盈就走，超时按收盘走
        for tp in TP_LEVELS:
            for ts_bar in TIME_STOPS:
                hit = r['tp_hits'].get(tp)
                if hit and hit['bar'] <= ts_bar:
                    # 在时间止损前到达止盈
                    roi = tp
                else:
                    # 超时，按时间止损点的收盘价走
                    roi_at_ts = None
                    for bar_offset, bar_roi in r['roi_path']:
                        if bar_offset == ts_bar:
                            roi_at_ts = bar_roi
                            break
                    if roi_at_ts is None:
                        # 如果没有足够K线，用最后一个
                        if r['roi_path']:
                            roi_at_ts = r['roi_path'][-1][1]
                        else:
                            roi_at_ts = -SLIPPAGE_PCT * 2  # 完全亏损
                    roi = roi_at_ts
                base[f'tp{tp}_ts{ts_bar}_roi'] = roi

        records.append(base)

    return pd.DataFrame(records)


# ============================================================
#  HTML报告
# ============================================================

def build_report(df_tp):
    rpt = Report('期权止盈策略研究')
    rpt.add_text('核心问题：期权涨到+X%时止盈 vs 跟期货追踪止损，哪个更好？')

    # ====== Part 1: 期权MFE分布 ======
    rpt.add_section('Part 1: 期权MFE分布（期权最大ROI%）',
                    '信号后期权价格最高涨到多少？')

    for delta in DELTA_LEVELS:
        headers = ['场景', 'N', 'MFE中位%', 'MFE P75%', 'MFE P90%',
                   '到峰中位(h)', '≥50%占比', '≥100%占比', '≥200%占比', '≥300%占比']
        rows = []
        for sc in [1, 2, 3]:
            sub = df_tp[(df_tp['scenario'] == sc) & (df_tp['delta'] == delta)]
            if len(sub) == 0:
                continue
            mfe = sub['max_roi']
            rows.append([
                SCENARIO_NAMES[sc], len(sub),
                round(mfe.median(), 1), round(mfe.quantile(0.75), 1), round(mfe.quantile(0.90), 1),
                round(sub['max_roi_hours'].median(), 1) if 'max_roi_hours' in sub else '-',
                f"{(mfe >= 50).mean() * 100:.1f}%",
                f"{(mfe >= 100).mean() * 100:.1f}%",
                f"{(mfe >= 200).mean() * 100:.1f}%",
                f"{(mfe >= 300).mean() * 100:.1f}%",
            ])
        rpt.add_section(f'Delta={delta}')
        rpt.add_table(headers, rows, highlight_pnl_cols=[2, 3, 4])

    # ====== Part 2: 止盈档位触发率 ======
    rpt.add_section('Part 2: 各止盈档位触发率',
                    '在24小时观察窗口内，期权能涨到+X%的概率')

    for delta in DELTA_LEVELS:
        headers_tp = ['场景', 'N'] + [f'+{tp}%' for tp in TP_LEVELS]
        rows_tp = []
        for sc in [1, 2, 3]:
            sub = df_tp[(df_tp['scenario'] == sc) & (df_tp['delta'] == delta)]
            if len(sub) == 0:
                continue
            row = [SCENARIO_NAMES[sc], len(sub)]
            for tp in TP_LEVELS:
                hit_rate = sub[f'tp{tp}_hit'].mean() * 100
                row.append(f'{hit_rate:.1f}%')
            rows_tp.append(row)
        rpt.add_section(f'Delta={delta}')
        rpt.add_table(headers_tp, rows_tp)

    # ====== Part 3: 止盈 vs 跟期货止损 ======
    rpt.add_section('Part 3: 固定止盈 vs 跟期货追踪止损',
                    '纯止盈（到了就走，没到就持有到期货止损触发）')

    for delta in DELTA_LEVELS:
        for sc in [1, 2, 3]:
            sub = df_tp[(df_tp['scenario'] == sc) & (df_tp['delta'] == delta)]
            if len(sub) == 0:
                continue

            # 跟期货止损的EV
            base_pnls = sub['futures_stop_roi'].tolist()
            base_ev = calc_ev(base_pnls)

            headers = ['策略', 'N', 'EV', 'WR%', 'PR', 'avg_ROI%', 'Σ ROI%']
            rows = []
            rows.append(['跟期货止损', base_ev['N'], base_ev['EV'], base_ev['wr'],
                         base_ev['pr'], base_ev['avg_pnl'], base_ev['sum_pnl']])

            for tp in TP_LEVELS:
                # 到达止盈 → 赚tp%；没到 → 跟期货止损
                pnls = []
                for _, row in sub.iterrows():
                    if row[f'tp{tp}_hit'] == 1:
                        pnls.append(tp)
                    else:
                        pnls.append(row['futures_stop_roi'])
                ev = calc_ev(pnls)
                rows.append([f'止盈+{tp}%', ev['N'], ev['EV'], ev['wr'],
                            ev['pr'], ev['avg_pnl'], ev['sum_pnl']])

            rpt.add_section(f'{SCENARIO_NAMES[sc]} | Delta={delta}')
            rpt.add_table(headers, rows, highlight_pnl_cols=[2, 5, 6])

    # ====== Part 4: 止盈 + 时间止损（核心） ======
    rpt.add_section('Part 4: 止盈 + 时间止损组合（核心）',
                    '到达止盈就走；超过N小时没到就按市价平仓。找最优组合。')

    for delta in DELTA_LEVELS:
        for sc in [1, 2, 3]:
            sub = df_tp[(df_tp['scenario'] == sc) & (df_tp['delta'] == delta)]
            if len(sub) == 0:
                continue

            headers = ['止盈%\\时间止损', 'N'] + [f'{ts*10//60}h' for ts in TIME_STOPS] + ['无时限']
            rows = []

            for tp in TP_LEVELS:
                row = [f'+{tp}%', len(sub)]

                # 各时间止损
                for ts_bar in TIME_STOPS:
                    col = f'tp{tp}_ts{ts_bar}_roi'
                    pnls = sub[col].tolist()
                    ev = calc_ev(pnls)
                    cell = f"EV={ev['EV']}"
                    row.append(cell)

                # 无时限（止盈 or 跟期货止损）
                pnls_inf = []
                for _, r in sub.iterrows():
                    if r[f'tp{tp}_hit'] == 1:
                        pnls_inf.append(tp)
                    else:
                        pnls_inf.append(r['futures_stop_roi'])
                ev_inf = calc_ev(pnls_inf)
                row.append(f"EV={ev_inf['EV']}")

                rows.append(row)

            # 基线：跟期货止损
            base_ev = calc_ev(sub['futures_stop_roi'].tolist())
            rows.append(['跟期货止损', len(sub)] + [f"EV={base_ev['EV']}"] * (len(TIME_STOPS) + 1))

            rpt.add_section(f'{SCENARIO_NAMES[sc]} | Delta={delta}')
            rpt.add_table(headers, rows)

    # ====== Part 5: 最优组合推荐 ======
    rpt.add_section('Part 5: 最优组合推荐')

    for delta in DELTA_LEVELS:
        for sc in [1, 2, 3]:
            sub = df_tp[(df_tp['scenario'] == sc) & (df_tp['delta'] == delta)]
            if len(sub) == 0:
                continue

            best_ev = -999
            best_label = ""
            best_stats = None

            # 跟期货止损
            base_pnls = sub['futures_stop_roi'].tolist()
            base_ev_val = calc_ev(base_pnls)
            if base_ev_val['EV'] > best_ev:
                best_ev = base_ev_val['EV']
                best_label = "跟期货止损"
                best_stats = base_ev_val

            # 止盈+时间止损
            for tp in TP_LEVELS:
                for ts_bar in TIME_STOPS:
                    col = f'tp{tp}_ts{ts_bar}_roi'
                    pnls = sub[col].tolist()
                    ev = calc_ev(pnls)
                    if ev['EV'] > best_ev:
                        best_ev = ev['EV']
                        best_label = f"止盈+{tp}% / {ts_bar * 10 // 60}h时间止损"
                        best_stats = ev

                # 无时限
                pnls_inf = []
                for _, r in sub.iterrows():
                    if r[f'tp{tp}_hit'] == 1:
                        pnls_inf.append(tp)
                    else:
                        pnls_inf.append(r['futures_stop_roi'])
                ev_inf = calc_ev(pnls_inf)
                if ev_inf['EV'] > best_ev:
                    best_ev = ev_inf['EV']
                    best_label = f"止盈+{tp}% / 无时限"
                    best_stats = ev_inf

            s = best_stats
            color = '#3fb950' if best_ev > 0 else '#f85149'
            rpt.add_text(
                f'{SCENARIO_NAMES[sc]} | Δ={delta}: '
                f'<b>{best_label}</b> | '
                f'EV={s["EV"]} WR={s["wr"]}% PR={s["pr"]} avg={s["avg_pnl"]}% Σ={s["sum_pnl"]}%',
                color=color
            )

    # ====== Part 6: 到达峰值后的回撤 ======
    rpt.add_section('Part 6: 期权MFE到达后的行为',
                    '峰值多久到达？到达后还能持续多久？')

    for delta in DELTA_LEVELS:
        for sc in [1, 2, 3]:
            sub = df_tp[(df_tp['scenario'] == sc) & (df_tp['delta'] == delta)]
            if len(sub) == 0:
                continue

            # 到峰值时间分布
            peak_bars = sub['max_roi_bar']
            peak_hours = peak_bars * 10 / 60

            headers_peak = ['指标', 'P25', 'P50', 'P75', 'P90']
            rows_peak = [
                ['到峰时间(h)',
                 round(peak_hours.quantile(0.25), 1),
                 round(peak_hours.quantile(0.50), 1),
                 round(peak_hours.quantile(0.75), 1),
                 round(peak_hours.quantile(0.90), 1)],
                ['峰值ROI%',
                 round(sub['max_roi'].quantile(0.25), 1),
                 round(sub['max_roi'].quantile(0.50), 1),
                 round(sub['max_roi'].quantile(0.75), 1),
                 round(sub['max_roi'].quantile(0.90), 1)],
            ]

            rpt.add_section(f'{SCENARIO_NAMES[sc]} | Δ={delta}')
            rpt.add_table(headers_peak, rows_peak, highlight_pnl_cols=[1, 2, 3, 4])

    rpt.add_text('')
    rpt.add_text(
        '<b>模型假设</b>：HV近似IV，持仓期间IV不变，滑点5%。'
        '实际IV通常>HV，Gamma效应会更强，期权MFE可能更高。',
        color='#8b949e'
    )

    return rpt


# ============================================================
#  主流程
# ============================================================

if __name__ == '__main__':
    signals, all_data = collect_signals()

    print(f"\n计算期权MFE路径（{len(signals)}笔 × {len(DELTA_LEVELS)} Delta）...")

    all_results = []
    for i, sig in enumerate(signals):
        df = all_data[sig['df_key']]
        for delta in DELTA_LEVELS:
            r = compute_option_mfe_path(df, sig, delta, T_DAYS_DEFAULT)
            if r:
                r['delta'] = delta
                all_results.append(r)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(signals)}...")

    print(f"  有效结果: {len(all_results)}条")

    # 模拟止盈策略
    print("\n模拟止盈策略...")
    df_tp = simulate_tp_strategies(all_results)
    print(f"  生成 {len(df_tp)} 条记录")

    # 快速摘要
    print("\n" + "=" * 80)
    print("期权MFE摘要")
    print("=" * 80)
    for delta in DELTA_LEVELS:
        for sc in [1, 2, 3]:
            sub = df_tp[(df_tp['scenario'] == sc) & (df_tp['delta'] == delta)]
            if len(sub) == 0:
                continue
            mfe = sub['max_roi']
            base_roi = sub['futures_stop_roi']
            print(f"  场景{sc} Δ={delta}: N={len(sub)} "
                  f"MFE中位={mfe.median():.0f}% P90={mfe.quantile(0.9):.0f}% "
                  f"≥100%={((mfe >= 100).mean() * 100):.1f}% "
                  f"跟期货止损EV={calc_ev(base_roi.tolist())['EV']}")

    # 报告
    print("\n生成报告...")
    rpt = build_report(df_tp)
    rpt.save('output/option_tp_strategy.html')
    print("完成。")
