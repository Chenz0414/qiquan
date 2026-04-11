# -*- coding: utf-8 -*-
"""
新出场策略对比回测
==================
120天 × 32品种 × 3场景，8种出场策略并行对比。
新增: S2.1(S2收盘触损), S5.2(S5.1中S2换S2.1), S6(EMA5盘中), S6.1(EMA5收盘)
"""

import sys
import io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from data_loader import load_all, sym_name, tick_size
from signal_core import (SignalDetector, ExitTracker, classify_scenario,
                         DEFAULT_STOP_TICKS, SCENARIO_NAMES)
from chart_engine import render_chart
from report_engine import Report
from stats_utils import calc_ev

LAST_DAYS = 120
BARS_PER_DAY = 57

# 所有出场策略（前4个是已有的，后4个是新增的）
ALL_STRATEGIES = ['S1.1', 'S2', 'S2.1', 'S3.1', 'S5.1', 'S5.2', 'S6', 'S6.1']
# 每个场景当前最优出场
SCENARIO_BEST = {1: 'S2', 2: 'S2', 3: 'S5.1'}


def run():
    print("=" * 70)
    print("新出场策略对比回测 | 120天 × 32品种 × 3场景 × 8出场")
    print("=" * 70)

    # 加载数据（含EMA5）
    all_data = load_all(period_min=10, days=170, last_days=None,
                        emas=(5, 10, 20, 120), er_periods=(20,), atr_period=14)
    print(f"加载品种数: {len(all_data)}")

    records = []

    for sym_key, df in sorted(all_data.items()):
        n = len(df)
        signal_start = max(130, n - LAST_DAYS * BARS_PER_DAY)
        name = sym_name(sym_key)
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

            # 创建ExitTracker，启用EMA5策略
            tracker = ExitTracker(
                direction=result.direction,
                entry_price=result.entry_price,
                pullback_extreme=result.pullback_extreme,
                tick_size=ts,
                stop_ticks=DEFAULT_STOP_TICKS,
                ema5_strategies=True,
            )

            exit_results = {s: None for s in ALL_STRATEGIES}

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
                                'pnl': ev.pnl_pct,
                                'price': ev.exit_price,
                                'idx': j,
                                'bars': ev.bars_held,
                                'reason': ev.exit_reason,
                            }
                if tracker.all_done():
                    break

            # 强制平仓
            last_bar = df.iloc[-1]
            forced = tracker.force_close(last_bar['close'])
            for ev in forced:
                if ev.strategy in exit_results and exit_results[ev.strategy] is None:
                    exit_results[ev.strategy] = {
                        'pnl': ev.pnl_pct,
                        'price': ev.exit_price,
                        'idx': n - 1,
                        'bars': ev.bars_held,
                        'reason': ev.exit_reason,
                    }

            # 初始止损价
            if result.direction == 'long':
                stop_price = result.pullback_extreme - ts * DEFAULT_STOP_TICKS
            else:
                stop_price = result.pullback_extreme + ts * DEFAULT_STOP_TICKS

            rec = {
                'symbol': name,
                'sym_key': sym_key,
                'time': row['datetime'],
                'direction': result.direction,
                'type': result.signal_type,
                'er20': round(er20, 3),
                'dev_atr': round(deviation_atr, 3),
                'scenario': scenario,
                'df': df,
                'entry_idx': i,
                'stop_price': stop_price,
            }
            for s in ALL_STRATEGIES:
                r = exit_results.get(s)
                if r:
                    rec[f'{s}_pnl'] = round(r['pnl'], 4)
                    rec[f'{s}_reason'] = r['reason']
                    rec[f'{s}_bars'] = r['bars']
                    rec[f'{s}_idx'] = r['idx']
                    rec[f'{s}_price'] = r['price']
                else:
                    rec[f'{s}_pnl'] = 0
                    rec[f'{s}_reason'] = 'missing'
                    rec[f'{s}_bars'] = 0
                    rec[f'{s}_idx'] = None
                    rec[f'{s}_price'] = 0

            records.append(rec)

        sym_count = sum(1 for r in records if r['sym_key'] == sym_key)
        if sym_count > 0:
            print(f"  {name:>6}: {sym_count}笔")

    if not records:
        print("无命中场景信号!")
        return

    total = len(records)
    s1_n = sum(1 for r in records if r['scenario'] == 1)
    s2_n = sum(1 for r in records if r['scenario'] == 2)
    s3_n = sum(1 for r in records if r['scenario'] == 3)
    print(f"\n合计: {total}笔 (场景1={s1_n}, 场景2={s2_n}, 场景3={s3_n})")

    # ===== 生成HTML报告 =====
    rpt = Report('新出场策略对比回测 | 120天×32品种×3场景×8出场')

    # --- 总览表 ---
    rpt.add_section('总览: 各场景 × 各出场策略')

    for scenario in [1, 2, 3]:
        sub = [r for r in records if r['scenario'] == scenario]
        if not sub:
            continue

        best_old = SCENARIO_BEST[scenario]
        scenario_name = SCENARIO_NAMES.get(scenario, f'场景{scenario}')
        rpt.add_text(f'**{scenario_name}** (N={len(sub)}, 当前最优: {best_old})')

        headers = ['出场策略', 'N', 'EV', '胜率%', '盈亏比', '累计PnL%', '均值PnL%', '平均持仓根', '类型']
        rows = []

        for s in ALL_STRATEGIES:
            pnl_key = f'{s}_pnl'
            reason_key = f'{s}_reason'
            bars_key = f'{s}_bars'

            # 过滤backtest_end
            valid = [r for r in sub if r[reason_key] != 'backtest_end']
            pnls = [r[pnl_key] for r in valid]
            bars_list = [r[bars_key] for r in valid]

            ev = calc_ev(pnls)
            avg_bars = round(np.mean(bars_list), 1) if bars_list else 0

            is_new = s in ('S2.1', 'S5.2', 'S6', 'S6.1')
            tag = '新' if is_new else '旧'

            rows.append([
                s, ev['N'], f"{ev['EV']:+.2f}", f"{ev['wr']:.1f}",
                f"{ev['pr']:.2f}", f"{ev['sum_pnl']:+.2f}",
                f"{ev['avg_pnl']:+.4f}", f"{avg_bars:.1f}", tag,
            ])

        rpt.add_table(headers, rows, highlight_pnl_cols=[2, 5, 6])

    # --- 新旧直接对比 ---
    rpt.add_section('新旧对比: 关键出场策略')

    comparisons = [
        (1, [('S2', '旧:盘中触损'), ('S2.1', '新:收盘触损'), ('S6', '新:EMA5盘中'), ('S6.1', '新:EMA5收盘')]),
        (2, [('S2', '旧:盘中触损'), ('S2.1', '新:收盘触损'), ('S6', '新:EMA5盘中'), ('S6.1', '新:EMA5收盘')]),
        (3, [('S5.1', '旧:S3.1+S2'), ('S5.2', '新:S3.1+S2.1'), ('S2.1', '参考:纯S2.1'), ('S6.1', '参考:EMA5收盘')]),
    ]

    for scenario, pairs in comparisons:
        sub = [r for r in records if r['scenario'] == scenario]
        if not sub:
            continue

        scenario_name = SCENARIO_NAMES.get(scenario, f'场景{scenario}')
        rpt.add_text(f'**{scenario_name}**')

        headers = ['策略', '说明', 'N', 'EV', '胜率%', '盈亏比', '累计PnL%', '均值PnL%']
        rows = []
        for s, desc in pairs:
            valid = [r for r in sub if r[f'{s}_reason'] != 'backtest_end']
            pnls = [r[f'{s}_pnl'] for r in valid]
            ev = calc_ev(pnls)
            rows.append([
                s, desc, ev['N'], f"{ev['EV']:+.2f}", f"{ev['wr']:.1f}",
                f"{ev['pr']:.2f}", f"{ev['sum_pnl']:+.2f}", f"{ev['avg_pnl']:+.4f}",
            ])
        rpt.add_table(headers, rows, highlight_pnl_cols=[3, 6, 7])

    # --- K线图示例 ---
    rpt.add_section('K线图示例: 新旧出场差异')

    for scenario in [1, 2, 3]:
        sub = [r for r in records if r['scenario'] == scenario]
        if not sub:
            continue

        best_old = SCENARIO_BEST[scenario]
        # 新策略候选
        new_candidates = ['S2.1', 'S6', 'S6.1'] if scenario in (1, 2) else ['S5.2', 'S2.1', 'S6.1']
        best_new = new_candidates[0]

        # 按新旧PnL差排序，选3笔代表
        for r in sub:
            r['_diff'] = r[f'{best_new}_pnl'] - r[f'{best_old}_pnl']

        sorted_sub = sorted(sub, key=lambda x: x['_diff'], reverse=True)

        picks = []
        if len(sorted_sub) >= 3:
            picks.append(('新策略最优胜', sorted_sub[0]))
            picks.append(('中位案例', sorted_sub[len(sorted_sub) // 2]))
            picks.append(('旧策略最优胜', sorted_sub[-1]))
        else:
            for idx, r in enumerate(sorted_sub):
                picks.append((f'案例{idx+1}', r))

        for label, r in picks:
            scenario_name = f'场景{scenario}'
            title = (f"{scenario_name} {label} | {r['symbol']} {r['type']}类 "
                     f"{r['direction']} | {str(r['time'])[:16]}")

            # 收集出场点（新旧关键策略）
            show_strategies = [best_old] + new_candidates
            exits_to_draw = []
            for s in show_strategies:
                idx_key = f'{s}_idx'
                price_key = f'{s}_price'
                if r.get(idx_key) is not None and r[f'{s}_reason'] != 'backtest_end':
                    exits_to_draw.append({
                        'name': s,
                        'idx': r[idx_key],
                        'price': r[price_key],
                    })

            max_exit_idx = max((ex['idx'] for ex in exits_to_draw), default=r['entry_idx'])
            after_needed = max_exit_idx - r['entry_idx'] + 10
            after_bars = max(40, min(after_needed, 80))

            extra_info = {
                'ER20': f"{r['er20']:.2f}",
                '偏离': f"{r['dev_atr']:.2f}ATR",
            }
            for s in show_strategies:
                extra_info[s] = f"{r[f'{s}_pnl']:+.2f}%"

            rpt.add_text(f'**{title}**')
            chart_html = render_chart(
                df=r['df'],
                entry_idx=r['entry_idx'],
                direction=r['direction'],
                before_bars=30,
                after_bars=after_bars,
                exits=exits_to_draw,
                stop_price=r['stop_price'],
                title=title,
                extra_info=extra_info,
            )
            rpt.add_chart(chart_html)

    # --- 策略说明 ---
    rpt.add_section('策略定义')
    rpt.add_text('**已有策略:**')
    rpt.add_text('S1.1: 当根close创新高→止损=当根low-5跳, 收盘价触发')
    rpt.add_text('S2: 回调(跌破EMA10)完成(站回EMA10)→止损=回调最低-5跳, 盘中触发')
    rpt.add_text('S3.1: 当根close创新高→止损=前根low-5跳, 收盘价触发')
    rpt.add_text('S5.1: S3.1兜底(收盘触损)+S2接管(盘中触损)')
    rpt.add_text('')
    rpt.add_text('**新增策略:**')
    rpt.add_text('S2.1: 同S2, 止损触发改为收盘价穿止损 (出场价=close)')
    rpt.add_text('S5.2: S3.1兜底(收盘触损)+S2.1接管(收盘触损)')
    rpt.add_text('S6: 同S2, 回调检测用EMA5代替EMA10(更紧跟踪), 盘中触发')
    rpt.add_text('S6.1: 同S6, 止损触发改为收盘价穿止损 (出场价=close)')
    rpt.add_text('')
    rpt.add_text('**场景定义:**')
    rpt.add_text('场景1: A类+ER(20)>=0.5+偏离>=1.0ATR (当前最优: S2)')
    rpt.add_text('场景2: C类+偏离>=2.0ATR (当前最优: S2)')
    rpt.add_text('场景3: B类+ER(20)>=0.5+偏离0.1~0.3ATR (当前最优: S5.1)')
    rpt.add_text(f'数据范围: 最近{LAST_DAYS}天, 全{len(all_data)}品种, 10min K线')

    output_path = 'output/new_exits_comparison.html'
    rpt.save(output_path)
    print(f"\nHTML报告已保存: {output_path}")


if __name__ == '__main__':
    run()
