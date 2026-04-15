# -*- coding: utf-8 -*-
"""
ABC场景出场对比：当前S6/S5.1 vs 阶梯R（预设I / 2R）
====================================================
最近30天，逐场景对比出场效果
"""
import sys, io
if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from data_loader import load_all, sym_name, tick_size, BARS_PER_DAY
from signal_core import (
    SignalDetector, ExitTracker, LadderRTracker,
    SYMBOL_CONFIGS, DEFAULT_STOP_TICKS,
    classify_scenario, SCENARIO_EXIT, SCENARIO_NAMES,
)
from stats_utils import calc_ev
from report_engine import Report

LAST_DAYS = 30
OUTPUT = f'output/report_abc_ladder_compare_{LAST_DAYS}d.html'


def run(all_data):
    records = []
    for sym_key, df in sorted(all_data.items()):
        n = len(df)
        signal_start = max(130, n - LAST_DAYS * BARS_PER_DAY)
        ts = tick_size(sym_key)
        name = sym_name(sym_key)

        detector = SignalDetector(signal_types='ABC')

        for i in range(2, n):
            row = df.iloc[i]
            if pd.isna(row.get('ema10')) or pd.isna(row.get('ema20')) or pd.isna(row.get('ema120')):
                continue
            if pd.isna(row.get('atr')) or row['atr'] <= 0:
                continue

            result = detector.process_bar(
                close=row['close'], high=row['high'], low=row['low'],
                ema10=row['ema10'], ema20=row['ema20'], ema120=row['ema120'],
            )
            if result is None or i < signal_start:
                continue

            er20 = row.get('er_20', 0) or 0
            atr = row['atr']
            dev_atr = abs(result.entry_price - row['ema10']) / atr if atr > 0 else 0
            scenario = classify_scenario(result.signal_type, er20, dev_atr)
            if scenario is None:
                continue

            # 初始止损（ABC和LadderR共用）
            tick = ts * DEFAULT_STOP_TICKS
            if result.direction == 'long':
                init_stop = result.pullback_extreme - tick
            else:
                init_stop = result.pullback_extreme + tick

            # === 当前出场：ExitTracker ===
            et = ExitTracker(
                direction=result.direction,
                entry_price=result.entry_price,
                pullback_extreme=result.pullback_extreme,
                tick_size=ts,
                stop_ticks=DEFAULT_STOP_TICKS,
                ema5_strategies=True,
            )

            # === LadderR 预设I ===
            lr_i = LadderRTracker(
                direction=result.direction,
                entry_price=result.entry_price,
                stop_price=init_stop,
                tick_size=ts,
                preset='I',
                max_window=99999,  # 禁用60根timeout，与S6公平对比
            )

            # === LadderR 预设2R ===
            lr_2r = LadderRTracker(
                direction=result.direction,
                entry_price=result.entry_price,
                stop_price=init_stop,
                tick_size=ts,
                preset='2R',
                max_window=99999,
            )

            # 模拟出场
            exit_current = None  # 当前策略(S6 or S5.1)
            exit_lr_i = None
            exit_lr_2r = None
            current_strat = SCENARIO_EXIT[scenario]  # 'S6' or 'S5.1'

            for j in range(i + 1, n):
                bar = df.iloc[j]
                prev_bar = df.iloc[j - 1]
                if pd.isna(bar.get('ema10')):
                    continue

                # ExitTracker
                if exit_current is None and not et.all_done():
                    ema5_val = bar.get('ema5', None)
                    if ema5_val is not None and pd.isna(ema5_val):
                        ema5_val = None
                    exit_events, _ = et.process_bar(
                        close=bar['close'], high=bar['high'], low=bar['low'],
                        ema10=bar['ema10'], prev_close=prev_bar['close'],
                        prev_high=prev_bar['high'], prev_low=prev_bar['low'],
                        ema5=ema5_val,
                    )
                    for ev in exit_events:
                        if ev.strategy == current_strat and exit_current is None:
                            exit_current = ev

                # LadderR I
                if exit_lr_i is None:
                    ev_i = lr_i.process_bar(close=bar['close'], high=bar['high'], low=bar['low'])
                    if ev_i:
                        exit_lr_i = ev_i

                # LadderR 2R
                if exit_lr_2r is None:
                    ev_2r = lr_2r.process_bar(close=bar['close'], high=bar['high'], low=bar['low'])
                    if ev_2r:
                        exit_lr_2r = ev_2r

                if exit_current and exit_lr_i and exit_lr_2r:
                    break

            # 强制平仓
            last_close = df.iloc[-1]['close']
            if exit_current is None:
                forced = et.force_close(last_close)
                for ev in forced:
                    if ev.strategy == current_strat:
                        exit_current = ev
            if exit_lr_i is None:
                exit_lr_i = lr_i.force_close(last_close)
            if exit_lr_2r is None:
                exit_lr_2r = lr_2r.force_close(last_close)

            records.append({
                'symbol': name,
                'sym_key': sym_key,
                'datetime': row['datetime'],
                'direction': result.direction,
                'type': result.signal_type,
                'scenario': scenario,
                'er20': round(er20, 3),
                'entry_price': result.entry_price,
                'stop_price': init_stop,
                # 当前出场
                'cur_pnl': round(exit_current.pnl_pct, 4) if exit_current else None,
                'cur_bars': exit_current.bars_held if exit_current else 0,
                'cur_reason': exit_current.exit_reason if exit_current else 'none',
                # LadderR I
                'lri_pnl': round(exit_lr_i.pnl_pct, 4) if exit_lr_i else None,
                'lri_bars': exit_lr_i.bars_held if exit_lr_i else 0,
                'lri_reason': exit_lr_i.exit_reason if exit_lr_i else 'none',
                # LadderR 2R
                'lr2r_pnl': round(exit_lr_2r.pnl_pct, 4) if exit_lr_2r else None,
                'lr2r_bars': exit_lr_2r.bars_held if exit_lr_2r else 0,
                'lr2r_reason': exit_lr_2r.exit_reason if exit_lr_2r else 'none',
            })

    return pd.DataFrame(records)


def ev_row(label, df, pnl_col):
    """生成一行EV统计"""
    valid = df[df[pnl_col.replace('pnl', 'reason')] != 'backtest_end']
    if len(valid) == 0:
        return [label, len(df), '-', '-', '-', '-', '-']
    ev = calc_ev(valid[pnl_col].dropna().tolist())
    return [
        label, ev['N'],
        f'{ev["EV"]:+.2f}',
        f'{ev["wr"]:.1f}%',
        f'{ev["pr"]:.2f}',
        f'{ev["sum_pnl"]:+.2f}%',
        f'{ev["avg_pnl"]:+.4f}%',
    ]


def dir_ev(df, pnl_col, reason_col, direction):
    valid = df[(df['direction'] == direction) & (df[reason_col] != 'backtest_end')]
    if len(valid) == 0:
        return '-', '-', '-'
    ev = calc_ev(valid[pnl_col].dropna().tolist())
    return f'{ev["EV"]:+.2f}', f'{ev["sum_pnl"]:+.2f}%', f'{ev["N"]}'


def main():
    print(f"{'='*60}")
    print(f"  ABC场景出场对比：当前 vs 阶梯R（最近{LAST_DAYS}天）")
    print(f"{'='*60}")

    all_data = load_all(
        period_min=10, days=170, last_days=None,
        emas=(5, 10, 20, 60, 120),
        er_periods=(5, 20, 40),
        atr_period=14, min_bars=200,
    )
    print(f"品种: {len(all_data)}")

    df = run(all_data)
    print(f"总信号: {len(df)}")

    rpt = Report(f'ABC出场对比：当前 vs 阶梯R（{LAST_DAYS}天）')

    # === 逐场景对比 ===
    for s in [1, 2, 3]:
        sub = df[df['scenario'] == s]
        if len(sub) == 0:
            continue

        cur_strat = SCENARIO_EXIT[s]
        n = len(sub)
        nl = len(sub[sub['direction'] == 'long'])
        ns = len(sub[sub['direction'] == 'short'])

        rpt.add_section(f'场景{s}: {SCENARIO_NAMES[s]}',
                        subtitle=f'N={n} (多{nl}/空{ns})')

        # EV对比表
        headers = ['出场策略', 'N', 'EV', '胜率', '盈亏比', '累计', '均值']
        rows = [
            ev_row(f'{cur_strat}（当前）', sub, 'cur_pnl'),
            ev_row('阶梯I (1→0,3→1,+2R)', sub, 'lri_pnl'),
            ev_row('阶梯2R (2→0,+2R)', sub, 'lr2r_pnl'),
        ]
        rpt.add_table(headers, rows, highlight_pnl_cols=[2, 5, 6])

        # 持仓时间对比
        cur_bars = sub['cur_bars'].mean()
        lri_bars = sub['lri_bars'].mean()
        lr2r_bars = sub['lr2r_bars'].mean()
        rpt.add_text(f'平均持仓: {cur_strat}={cur_bars:.0f}根 | '
                     f'阶梯I={lri_bars:.0f}根 | 阶梯2R={lr2r_bars:.0f}根')

        # 多空拆分
        headers2 = ['出场策略', '多EV', '多累计', '多N', '空EV', '空累计', '空N']
        rows2 = []
        for label, pnl_col, reason_col in [
            (f'{cur_strat}（当前）', 'cur_pnl', 'cur_reason'),
            ('阶梯I', 'lri_pnl', 'lri_reason'),
            ('阶梯2R', 'lr2r_pnl', 'lr2r_reason'),
        ]:
            l_ev, l_cum, l_n = dir_ev(sub, pnl_col, reason_col, 'long')
            s_ev, s_cum, s_n = dir_ev(sub, pnl_col, reason_col, 'short')
            rows2.append([label, l_ev, l_cum, l_n, s_ev, s_cum, s_n])
        rpt.add_table(headers2, rows2, highlight_pnl_cols=[1, 2, 4, 5])

        # 逐笔对比：当前 vs 阶梯I，谁赚得多
        both_valid = sub.dropna(subset=['cur_pnl', 'lri_pnl'])
        if len(both_valid) > 0:
            lri_better = (both_valid['lri_pnl'] > both_valid['cur_pnl']).sum()
            cur_better = (both_valid['cur_pnl'] > both_valid['lri_pnl']).sum()
            tie = (both_valid['cur_pnl'] == both_valid['lri_pnl']).sum()
            rpt.add_text(f'逐笔PK({cur_strat} vs 阶梯I): '
                         f'{cur_strat}赢 {cur_better}笔 | 阶梯I赢 {lri_better}笔 | 平 {tie}笔')

    # === 全场景汇总 ===
    rpt.add_section('全场景汇总')
    headers = ['出场策略', 'N', 'EV', '胜率', '盈亏比', '累计', '均值']
    rows = [
        ev_row('当前（S6/S5.1）', df, 'cur_pnl'),
        ev_row('阶梯I', df, 'lri_pnl'),
        ev_row('阶梯2R', df, 'lr2r_pnl'),
    ]
    rpt.add_table(headers, rows, highlight_pnl_cols=[2, 5, 6])

    rpt.save(OUTPUT)
    print(f"\n报告已保存: {OUTPUT}")

    # === 控制台输出 ===
    for s in [1, 2, 3]:
        sub = df[df['scenario'] == s]
        if len(sub) == 0:
            continue
        cur_strat = SCENARIO_EXIT[s]
        nl = len(sub[sub['direction'] == 'long'])
        ns = len(sub[sub['direction'] == 'short'])
        print(f'\n{"="*60}')
        print(f'场景{s} | N={len(sub)} (多{nl}/空{ns}) | 当前出场: {cur_strat}')
        print(f'{"="*60}')

        for label, pnl_col, reason_col, bars_col in [
            (f'{cur_strat:6s}（当前）', 'cur_pnl', 'cur_reason', 'cur_bars'),
            ('阶梯I         ', 'lri_pnl', 'lri_reason', 'lri_bars'),
            ('阶梯2R        ', 'lr2r_pnl', 'lr2r_reason', 'lr2r_bars'),
        ]:
            valid = sub[sub[reason_col] != 'backtest_end']
            ev = calc_ev(valid[pnl_col].dropna().tolist()) if len(valid) > 0 else {'EV':0,'wr':0,'pr':0,'sum_pnl':0,'N':0}
            avg_bars = sub[bars_col].mean()
            print(f'  {label} EV={ev["EV"]:+.2f} | 胜率={ev["wr"]:5.1f}% | '
                  f'盈亏比={ev["pr"]:.2f} | 累计={ev["sum_pnl"]:+7.2f}% | '
                  f'持仓={avg_bars:.0f}根')

        # 多空
        for label, pnl_col, reason_col in [
            (f'{cur_strat}', 'cur_pnl', 'cur_reason'),
            ('阶梯I', 'lri_pnl', 'lri_reason'),
        ]:
            for d, dl in [('long', '多'), ('short', '空')]:
                dsub = sub[(sub['direction'] == d) & (sub[reason_col] != 'backtest_end')]
                if len(dsub) == 0:
                    continue
                dev = calc_ev(dsub[pnl_col].dropna().tolist())
                print(f'    {label} {dl}: N={dev["N"]} EV={dev["EV"]:+.2f} 累计={dev["sum_pnl"]:+.2f}%')


if __name__ == '__main__':
    main()
