# -*- coding: utf-8 -*-
"""
S6稳健性验证
=============
1. 分时间窗口（30/60/90/120天）对比S6 vs S2
2. 逐品种对比S6 vs S2（场景1+2合并）
3. 场景1和场景2分别看
"""

import sys
import io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from data_loader import load_all, sym_name, tick_size, BARS_PER_DAY
from signal_core import (SignalDetector, ExitTracker, classify_scenario,
                         DEFAULT_STOP_TICKS, SCENARIO_NAMES)
from report_engine import Report
from stats_utils import calc_ev

TIME_WINDOWS = [30, 60, 90, 120]  # 天
FOCUS_STRATEGIES = ['S2', 'S6']  # 主要对比
EXTRA_STRATEGIES = ['S1.1', 'S2.1', 'S3.1', 'S5.1', 'S5.2', 'S6.1']


def collect_trades():
    """收集所有信号的出场数据"""
    all_data = load_all(period_min=10, days=170, last_days=None,
                        emas=(5, 10, 20, 120), er_periods=(20,), atr_period=14)
    print(f"加载品种数: {len(all_data)}")

    records = []

    for sym_key, df in sorted(all_data.items()):
        n = len(df)
        # 用170天全量检测信号，后面按时间窗口过滤
        signal_start = max(130, n - 120 * BARS_PER_DAY)
        ts = tick_size(sym_key)
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
                direction=result.direction,
                entry_price=result.entry_price,
                pullback_extreme=result.pullback_extreme,
                tick_size=ts,
                stop_ticks=DEFAULT_STOP_TICKS,
                ema5_strategies=True,
            )

            all_strats = FOCUS_STRATEGIES + EXTRA_STRATEGIES
            exit_results = {s: None for s in all_strats}

            for j in range(i + 1, n):
                bar = df.iloc[j]
                prev_bar = df.iloc[j - 1]
                if pd.isna(bar['ema10']):
                    continue
                ema5_val = bar.get('ema5', None)
                if pd.isna(ema5_val):
                    ema5_val = None

                if not tracker.all_done():
                    exit_events, _ = tracker.process_bar(
                        close=bar['close'], high=bar['high'], low=bar['low'],
                        ema10=bar['ema10'], prev_close=prev_bar['close'],
                        prev_high=prev_bar['high'], prev_low=prev_bar['low'],
                        ema5=ema5_val,
                    )
                    for ev in exit_events:
                        if ev.strategy in exit_results and exit_results[ev.strategy] is None:
                            exit_results[ev.strategy] = {
                                'pnl': ev.pnl_pct, 'bars': ev.bars_held, 'reason': ev.exit_reason,
                            }
                if tracker.all_done():
                    break

            last_bar = df.iloc[-1]
            forced = tracker.force_close(last_bar['close'])
            for ev in forced:
                if ev.strategy in exit_results and exit_results[ev.strategy] is None:
                    exit_results[ev.strategy] = {
                        'pnl': ev.pnl_pct, 'bars': ev.bars_held, 'reason': ev.exit_reason,
                    }

            # 计算信号距数据末尾的天数
            bars_from_end = n - 1 - i
            days_from_end = bars_from_end / BARS_PER_DAY

            rec = {
                'symbol': sym_name(sym_key),
                'sym_key': sym_key,
                'scenario': scenario,
                'type': result.signal_type,
                'days_from_end': days_from_end,
            }
            for s in all_strats:
                r = exit_results.get(s)
                if r:
                    rec[f'{s}_pnl'] = r['pnl']
                    rec[f'{s}_reason'] = r['reason']
                    rec[f'{s}_bars'] = r['bars']
                else:
                    rec[f'{s}_pnl'] = 0
                    rec[f'{s}_reason'] = 'missing'
                    rec[f'{s}_bars'] = 0
            records.append(rec)

    return records, len(all_data)


def run():
    print("=" * 70)
    print("S6稳健性验证 | S6 vs S2 分时间窗口 + 逐品种")
    print("=" * 70)

    records, n_symbols = collect_trades()
    if not records:
        print("无数据!")
        return

    print(f"总信号: {len(records)}笔")

    rpt = Report('S6稳健性验证 | S6 vs S2 分时间窗口 + 逐品种')

    # ========== 1. 分时间窗口 ==========
    rpt.add_section('1. 分时间窗口: S6 vs S2')

    for scenario in [1, 2, 'all_12']:
        if scenario == 'all_12':
            label = '场景1+2合并'
            sub_all = [r for r in records if r['scenario'] in (1, 2)]
        else:
            label = SCENARIO_NAMES.get(scenario, f'场景{scenario}')
            sub_all = [r for r in records if r['scenario'] == scenario]

        if not sub_all:
            continue

        rpt.add_text(f'**{label}**')
        headers = ['时间窗口', 'N',
                    'S2_EV', 'S2_WR%', 'S2_PR', 'S2_累计%', 'S2_均值%',
                    'S6_EV', 'S6_WR%', 'S6_PR', 'S6_累计%', 'S6_均值%',
                    'S6胜?']
        rows = []

        for window in TIME_WINDOWS:
            sub = [r for r in sub_all if r['days_from_end'] <= window]
            # 排除backtest_end
            valid_s2 = [r for r in sub if r['S2_reason'] != 'backtest_end']
            valid_s6 = [r for r in sub if r['S6_reason'] != 'backtest_end']

            ev_s2 = calc_ev([r['S2_pnl'] for r in valid_s2])
            ev_s6 = calc_ev([r['S6_pnl'] for r in valid_s6])

            s6_wins = 'YES' if ev_s6['sum_pnl'] > ev_s2['sum_pnl'] else 'NO'

            rows.append([
                f'{window}天', len(sub),
                f"{ev_s2['EV']:+.2f}", f"{ev_s2['wr']:.1f}", f"{ev_s2['pr']:.2f}",
                f"{ev_s2['sum_pnl']:+.1f}", f"{ev_s2['avg_pnl']:+.4f}",
                f"{ev_s6['EV']:+.2f}", f"{ev_s6['wr']:.1f}", f"{ev_s6['pr']:.2f}",
                f"{ev_s6['sum_pnl']:+.1f}", f"{ev_s6['avg_pnl']:+.4f}",
                s6_wins,
            ])

        rpt.add_table(headers, rows, highlight_pnl_cols=[2, 5, 6, 7, 10, 11])

    # ========== 2. 逐品种对比 ==========
    rpt.add_section('2. 逐品种: S6 vs S2 (场景1+2)')

    sub_12 = [r for r in records if r['scenario'] in (1, 2)]

    symbols = sorted(set(r['sym_key'] for r in sub_12))
    headers = ['品种', 'N',
               'S2_EV', 'S2_累计%',
               'S6_EV', 'S6_累计%',
               '差(S6-S2)', 'S6胜?']
    rows = []
    s6_win_count = 0
    s2_win_count = 0

    for sym_key in symbols:
        sym_sub = [r for r in sub_12 if r['sym_key'] == sym_key]
        valid_s2 = [r for r in sym_sub if r['S2_reason'] != 'backtest_end']
        valid_s6 = [r for r in sym_sub if r['S6_reason'] != 'backtest_end']

        if not valid_s2 and not valid_s6:
            continue

        ev_s2 = calc_ev([r['S2_pnl'] for r in valid_s2])
        ev_s6 = calc_ev([r['S6_pnl'] for r in valid_s6])

        diff = ev_s6['sum_pnl'] - ev_s2['sum_pnl']
        winner = 'S6' if diff > 0 else ('S2' if diff < 0 else '平')
        if diff > 0:
            s6_win_count += 1
        elif diff < 0:
            s2_win_count += 1

        rows.append([
            sym_name(sym_key), max(len(valid_s2), len(valid_s6)),
            f"{ev_s2['EV']:+.2f}", f"{ev_s2['sum_pnl']:+.1f}",
            f"{ev_s6['EV']:+.2f}", f"{ev_s6['sum_pnl']:+.1f}",
            f"{diff:+.2f}", winner,
        ])

    # 按差值排序
    rows.sort(key=lambda x: float(x[6]), reverse=True)

    rpt.add_table(headers, rows, highlight_pnl_cols=[2, 3, 4, 5, 6])

    total_sym = s6_win_count + s2_win_count
    rpt.add_text(f'**品种胜负: S6赢{s6_win_count}个, S2赢{s2_win_count}个 '
                 f'({s6_win_count}/{total_sym} = {s6_win_count/total_sym*100:.0f}%)**' if total_sym > 0 else '')

    # ========== 3. 场景1和场景2分别逐品种 ==========
    for scenario in [1, 2]:
        label = SCENARIO_NAMES.get(scenario, f'场景{scenario}')
        rpt.add_section(f'3-{scenario}. 逐品种: S6 vs S2 ({label})')

        sub = [r for r in records if r['scenario'] == scenario]
        symbols = sorted(set(r['sym_key'] for r in sub))

        headers = ['品种', 'N',
                   'S2_EV', 'S2_累计%',
                   'S6_EV', 'S6_累计%',
                   '差(S6-S2)', 'S6胜?']
        rows = []
        s6w = 0
        s2w = 0

        for sym_key in symbols:
            sym_sub = [r for r in sub if r['sym_key'] == sym_key]
            valid_s2 = [r for r in sym_sub if r['S2_reason'] != 'backtest_end']
            valid_s6 = [r for r in sym_sub if r['S6_reason'] != 'backtest_end']

            if not valid_s2 and not valid_s6:
                continue

            ev_s2 = calc_ev([r['S2_pnl'] for r in valid_s2])
            ev_s6 = calc_ev([r['S6_pnl'] for r in valid_s6])

            diff = ev_s6['sum_pnl'] - ev_s2['sum_pnl']
            winner = 'S6' if diff > 0 else ('S2' if diff < 0 else '平')
            if diff > 0:
                s6w += 1
            elif diff < 0:
                s2w += 1

            rows.append([
                sym_name(sym_key), max(len(valid_s2), len(valid_s6)),
                f"{ev_s2['EV']:+.2f}", f"{ev_s2['sum_pnl']:+.1f}",
                f"{ev_s6['EV']:+.2f}", f"{ev_s6['sum_pnl']:+.1f}",
                f"{diff:+.2f}", winner,
            ])

        rows.sort(key=lambda x: float(x[6]), reverse=True)
        rpt.add_table(headers, rows, highlight_pnl_cols=[2, 3, 4, 5, 6])

        total_s = s6w + s2w
        if total_s > 0:
            rpt.add_text(f'**品种胜负: S6赢{s6w}个, S2赢{s2w}个 '
                         f'({s6w}/{total_s} = {s6w/total_s*100:.0f}%)**')

    # ========== 4. 逐笔差异分布 ==========
    rpt.add_section('4. 逐笔PnL差异分布 (场景1+2)')

    sub_12_valid = [r for r in sub_12
                    if r['S2_reason'] != 'backtest_end' and r['S6_reason'] != 'backtest_end']

    diffs = [r['S6_pnl'] - r['S2_pnl'] for r in sub_12_valid]
    s6_better = sum(1 for d in diffs if d > 0)
    s2_better = sum(1 for d in diffs if d < 0)
    tie = sum(1 for d in diffs if d == 0)
    total = len(diffs)

    headers = ['指标', '值']
    rows = [
        ['逐笔S6>S2', f'{s6_better}笔 ({s6_better/total*100:.1f}%)'],
        ['逐笔S2>S6', f'{s2_better}笔 ({s2_better/total*100:.1f}%)'],
        ['持平', f'{tie}笔 ({tie/total*100:.1f}%)'],
        ['差值均值', f'{np.mean(diffs):+.4f}%'],
        ['差值中位数', f'{np.median(diffs):+.4f}%'],
        ['差值P25', f'{np.percentile(diffs, 25):+.4f}%'],
        ['差值P75', f'{np.percentile(diffs, 75):+.4f}%'],
    ]
    rpt.add_table(headers, rows)

    # ========== 5. S6出场更早还是更晚 ==========
    rpt.add_section('5. 持仓时间对比 (场景1+2)')

    s2_bars = [r['S2_bars'] for r in sub_12_valid]
    s6_bars = [r['S6_bars'] for r in sub_12_valid]
    bars_diff = [s6 - s2 for s6, s2 in zip(s6_bars, s2_bars)]

    s6_earlier = sum(1 for d in bars_diff if d < 0)
    s2_earlier = sum(1 for d in bars_diff if d > 0)

    headers = ['指标', 'S2', 'S6']
    rows = [
        ['平均持仓根', f'{np.mean(s2_bars):.1f}', f'{np.mean(s6_bars):.1f}'],
        ['中位持仓根', f'{np.median(s2_bars):.0f}', f'{np.median(s6_bars):.0f}'],
        ['S6更早出场', '', f'{s6_earlier}笔 ({s6_earlier/total*100:.1f}%)'],
        ['S2更早出场', f'{s2_earlier}笔 ({s2_earlier/total*100:.1f}%)', ''],
    ]
    rpt.add_table(headers, rows)

    # ========== 汇总 ==========
    rpt.add_section('汇总结论')
    rpt.add_text('S6 = S2的EMA5版本（回调检测用EMA5代替EMA10，更紧跟踪，盘中触损）')
    rpt.add_text(f'数据: 最近120天, {n_symbols}品种, 10min K线')
    rpt.add_text('验证维度: 时间窗口一致性 + 逐品种胜负 + 逐笔差异分布 + 持仓时间')

    output_path = 'output/s6_robustness.html'
    rpt.save(output_path)
    print(f"\nHTML报告已保存: {output_path}")


if __name__ == '__main__':
    run()
