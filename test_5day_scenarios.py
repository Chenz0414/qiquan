# -*- coding: utf-8 -*-
"""
最近5个交易日 × 全品种：3个开仓场景命中统计 + 各出场策略收益 + K线图
"""

import sys
import io
import numpy as np
import pandas as pd

# 解决Windows控制台中文编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from data_loader import load_all, sym_name, tick_size
from signal_core import SignalDetector, ExitTracker, SYMBOL_CONFIGS, DEFAULT_STOP_TICKS
from chart_engine import render_chart, get_chart_js
from report_engine import Report

LAST_DAYS = 10
BARS_PER_DAY = 57  # 10min


def classify_scenario(sig_type, er20, deviation_atr):
    """判断信号属于哪个场景，返回场景编号或None"""
    # 全局过滤：ER>=0.7正期望消失，不开仓
    if er20 >= 0.7:
        return None
    if sig_type == 'A' and er20 >= 0.5 and deviation_atr >= 1.0:
        return 1
    if sig_type == 'C' and deviation_atr >= 2.0:
        return 2
    if sig_type == 'B' and er20 >= 0.5 and deviation_atr >= 0.1 and deviation_atr < 0.3:
        return 3
    return None


def run():
    print("=" * 70)
    print(f"最近{LAST_DAYS}个交易日 x 全品种 | 3个开仓场景命中 + K线图")
    print("=" * 70)

    # 加载全品种
    all_data = load_all(period_min=10, days=170, last_days=None,
                        emas=(10, 20, 120), er_periods=(20, 40), atr_period=14)
    print(f"加载品种数: {len(all_data)}")

    # 收集命中信号（带df引用和出场idx，用于画图）
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

            # ER和偏离度
            er20 = row.get('er_20', np.nan)
            er40 = row.get('er_40', np.nan)
            if pd.isna(er20):
                er20 = 0
            if pd.isna(er40):
                er40 = 0
            atr = row['atr']
            deviation_atr = abs(result.entry_price - row['ema10']) / atr if atr > 0 else 0

            scenario = classify_scenario(result.signal_type, er20, deviation_atr)
            if scenario is None:
                continue  # 只保留命中场景的

            # 模拟出场，记录出场idx用于画图
            tracker = ExitTracker(
                direction=result.direction,
                entry_price=result.entry_price,
                pullback_extreme=result.pullback_extreme,
                tick_size=ts,
                stop_ticks=DEFAULT_STOP_TICKS,
            )

            exit_results = {}
            for s in ['S1', 'S2', 'S3']:
                exit_results[s] = {'pnl': 0, 'price': 0, 'idx': None, 'reason': ''}

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
                        if exit_results[ev.strategy]['idx'] is None:
                            exit_results[ev.strategy] = {
                                'pnl': ev.pnl_pct,
                                'price': ev.exit_price,
                                'idx': j,
                                'reason': ev.exit_reason,
                            }
                if tracker.all_done():
                    break

            # 强制平仓
            last_bar = df.iloc[-1]
            forced = tracker.force_close(last_bar['close'])
            for ev in forced:
                if exit_results[ev.strategy]['idx'] is None:
                    exit_results[ev.strategy] = {
                        'pnl': ev.pnl_pct,
                        'price': ev.exit_price,
                        'idx': n - 1,
                        'reason': ev.exit_reason,
                    }

            s1 = exit_results['S1']
            s2 = exit_results['S2']
            s3 = exit_results['S3']
            s4_pnl = (s1['pnl'] + s2['pnl']) / 2

            # 初始止损价
            if result.direction == 'long':
                stop_price = result.pullback_extreme - ts * DEFAULT_STOP_TICKS
            else:
                stop_price = result.pullback_extreme + ts * DEFAULT_STOP_TICKS

            records.append({
                'symbol': name,
                'sym_key': sym_key,
                'time': row['datetime'],
                'direction': result.direction,
                'type': result.signal_type,
                'er20': round(er20, 3),
                'er40': round(er40, 3),
                'dev_atr': round(deviation_atr, 3),
                'scenario': scenario,
                'S1': round(s1['pnl'], 4),
                'S2': round(s2['pnl'], 4),
                'S3': round(s3['pnl'], 4),
                'S4': round(s4_pnl, 4),
                'S1_reason': s1['reason'],
                'S2_reason': s2['reason'],
                'S3_reason': s3['reason'],
                # 画图用
                'df': df,
                'entry_idx': i,
                'stop_price': stop_price,
                'exits': [
                    {'name': 'S1', 'idx': s1['idx'], 'price': s1['price']},
                    {'name': 'S2', 'idx': s2['idx'], 'price': s2['price']},
                    {'name': 'S3', 'idx': s3['idx'], 'price': s3['price']},
                ],
            })

    if not records:
        print("最近5天无命中场景的信号!")
        return

    print(f"命中场景信号: {len(records)}笔")

    # ===== 生成HTML报告 =====
    rpt = Report(f'最近{LAST_DAYS}个交易日 - 开仓场景命中 K线验证')

    # 汇总表
    rpt.add_section('汇总统计')
    headers = ['场景', '笔数', 'S1累计', 'S2累计', 'S3累计', 'S4累计',
               'S1均值', 'S2均值', 'S3均值', 'S4均值']
    rows = []
    for s in [1, 2, 3]:
        sub = [r for r in records if r['scenario'] == s]
        if not sub:
            rows.append([f'场景{s}', 0, '-', '-', '-', '-', '-', '-', '-', '-'])
            continue
        s1_sum = sum(r['S1'] for r in sub)
        s2_sum = sum(r['S2'] for r in sub)
        s3_sum = sum(r['S3'] for r in sub)
        s4_sum = sum(r['S4'] for r in sub)
        nn = len(sub)
        rows.append([
            f'场景{s}', nn,
            f'{s1_sum:+.2f}%', f'{s2_sum:+.2f}%', f'{s3_sum:+.2f}%', f'{s4_sum:+.2f}%',
            f'{s1_sum/nn:+.4f}%', f'{s2_sum/nn:+.4f}%', f'{s3_sum/nn:+.4f}%', f'{s4_sum/nn:+.4f}%',
        ])
    # 合计
    s1_total = sum(r['S1'] for r in records)
    s2_total = sum(r['S2'] for r in records)
    s3_total = sum(r['S3'] for r in records)
    s4_total = sum(r['S4'] for r in records)
    nn = len(records)
    rows.append([
        '合计', nn,
        f'{s1_total:+.2f}%', f'{s2_total:+.2f}%', f'{s3_total:+.2f}%', f'{s4_total:+.2f}%',
        f'{s1_total/nn:+.4f}%', f'{s2_total/nn:+.4f}%', f'{s3_total/nn:+.4f}%', f'{s4_total/nn:+.4f}%',
    ])
    rpt.add_table(headers, rows)

    # 明细表
    rpt.add_section('信号明细')
    detail_headers = ['场景', '品种', '类型', '方向', '时间', 'ER20', 'ER40',
                      '偏离ATR', 'S1', 'S2', 'S3', 'S4', '状态']
    detail_rows = []
    for r in sorted(records, key=lambda x: (x['scenario'], str(x['time']))):
        holding = r['S1_reason'] == 'backtest_end' or r['S2_reason'] == 'backtest_end'
        detail_rows.append([
            f"场景{r['scenario']}", r['symbol'], f"{r['type']}类",
            r['direction'], str(r['time'])[:16],
            f"{r['er20']:.2f}", f"{r['er40']:.2f}", f"{r['dev_atr']:.2f}",
            f"{r['S1']:+.2f}%", f"{r['S2']:+.2f}%", f"{r['S3']:+.2f}%", f"{r['S4']:+.2f}%",
            '持仓中' if holding else '已平仓',
        ])
    rpt.add_table(detail_headers, detail_rows, highlight_pnl_cols=[8, 9, 10, 11])

    # 逐笔K线图
    for idx, r in enumerate(sorted(records, key=lambda x: (x['scenario'], str(x['time'])))):
        scenario = r['scenario']
        holding = r['S1_reason'] == 'backtest_end' or r['S2_reason'] == 'backtest_end'

        title = (f"场景{scenario} | {r['symbol']} {r['type']}类 {r['direction']} | "
                 f"{str(r['time'])[:16]}")

        # 过滤掉持仓中的exit标记（idx=最后一根不画）
        exits_to_draw = []
        for ex in r['exits']:
            if ex['idx'] is not None and ex['idx'] < len(r['df']) - 1:
                exits_to_draw.append(ex)
            elif ex['idx'] is not None and not holding:
                exits_to_draw.append(ex)

        # 计算合理的after_bars（覆盖所有出场点）
        max_exit_idx = max((ex['idx'] for ex in r['exits'] if ex['idx'] is not None),
                           default=r['entry_idx'])
        after_needed = max_exit_idx - r['entry_idx'] + 10
        after_bars = max(40, min(after_needed, 80))

        extra_info = {
            'ER20': f"{r['er20']:.2f}",
            'ER40': f"{r['er40']:.2f}",
            '偏离': f"{r['dev_atr']:.2f}ATR",
            'S1': f"{r['S1']:+.2f}%",
            'S2': f"{r['S2']:+.2f}%",
            'S3': f"{r['S3']:+.2f}%",
            'S4': f"{r['S4']:+.2f}%",
        }
        if holding:
            extra_info['状态'] = '持仓中'

        rpt.add_section(title)
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

    # 场景定义说明
    rpt.add_section('场景定义')
    rpt.add_text('场景1: A类 + ER(20)>=0.5 + 偏离>=1.0ATR (强趋势远离EMA10影线弹回)')
    rpt.add_text('场景2: C类 + 偏离>=2.0ATR (长回调蓄力爆发，不要求ER)')
    rpt.add_text('场景3: B类 + ER(20)>=0.5 + 偏离<0.3ATR (强趋势小幅回踩)')
    rpt.add_text(f'数据范围: 最近{LAST_DAYS}个交易日, 全{len(all_data)}品种, 10min K线')

    output_path = 'output/5day_scenarios_kline.html'
    rpt.save(output_path)
    print(f"HTML报告已保存: {output_path}")


if __name__ == '__main__':
    run()
