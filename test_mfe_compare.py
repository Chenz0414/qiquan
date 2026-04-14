# -*- coding: utf-8 -*-
"""
MFE 天花板对比：ABC场景系统 vs Type1 α/β
==========================================
同数据(36品种×120天×10min)，两套入场逻辑分别跑 MFE，
对比入场质量差异。

用法:
    python test_mfe_compare.py
"""

import sys
import io
import os
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding != 'utf-8' and not sys.stdout.closed:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from data_loader import load_all, BARS_PER_DAY, HIGH_VOL
from signal_core import (SYMBOL_CONFIGS, SignalDetector, classify_scenario,
                         DEFAULT_STOP_TICKS)
from stats_utils import calc_ev
from report_engine import Report

# 复用 Type1 的信号检测和分级
from test_type1_exit import (detect_signals, run_single_symbol, classify_tier,
                             classify_tier_overlap, TIER_ORDER, TIER_NAMES)
from test_trend_pullback import track_mfe_mae

# ============ 参数 ============
BACKTEST_DAYS = 120
WARMUP_DAYS = 50
MFE_WINDOW = 60
MARGIN_RATE = 0.15
STOP_TICKS = 5


# ============ ABC 系统 MFE ============

def run_abc_system(all_data):
    """跑 ABC 信号检测 + MFE追踪，返回 DataFrame"""
    print("运行 ABC 场景系统...")
    records = []

    for sym_key in sorted(all_data.keys()):
        df = all_data[sym_key]
        cfg = SYMBOL_CONFIGS.get(sym_key, {})
        ts = cfg.get('tick_size', 1.0)
        name = cfg.get('name', sym_key)

        n = len(df)
        start = max(0, n - BACKTEST_DAYS * BARS_PER_DAY)

        detector = SignalDetector('ABC')

        for i in range(n):
            row = df.iloc[i]
            if pd.isna(row['ema10']) or pd.isna(row['ema20']) or pd.isna(row['ema120']):
                detector._prev_close = row['close']
                detector._prev_ema10 = row['ema10']
                detector._bar_index += 1
                continue

            signal = detector.process_bar(
                close=row['close'], high=row['high'], low=row['low'],
                ema10=row['ema10'], ema20=row['ema20'], ema120=row['ema120'],
            )

            if signal and i >= start:
                er20 = row.get('er_20', 0)
                if pd.isna(er20):
                    er20 = 0
                atr_val = row.get('atr', 1.0)
                if pd.isna(atr_val) or atr_val <= 0:
                    continue

                # 偏离度
                deviation = abs(signal.entry_price - row['ema10']) / atr_val

                # 场景分类
                scenario = classify_scenario(signal.signal_type, er20, deviation)

                # MFE 追踪（从信号K线的下一根开始）
                entry_price = signal.entry_price
                pullback_extreme = signal.pullback_extreme

                if signal.direction == 'long':
                    stop_price = pullback_extreme - ts * STOP_TICKS
                else:
                    stop_price = pullback_extreme + ts * STOP_TICKS

                # 用和 Type1 一样的 track_mfe_mae
                high_arr = df['high'].values
                low_arr = df['low'].values
                close_arr = df['close'].values

                mfe_data = track_mfe_mae(
                    high_arr, low_arr, close_arr,
                    i, entry_price, signal.direction,
                    stop_price, atr_val
                )

                stop_dist = abs(entry_price - stop_price)
                r_mult = mfe_data['mfe_raw'] / stop_dist if stop_dist > 0 else 0

                records.append({
                    'symbol': sym_key,
                    'name': name,
                    'direction': signal.direction,
                    'signal_type': signal.signal_type,
                    'scenario': scenario,
                    'er_20': round(er20, 4),
                    'deviation_atr': round(deviation, 4),
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'stop_dist_atr': round(stop_dist / atr_val, 4) if atr_val > 0 else 0,
                    'mfe_raw': mfe_data['mfe_raw'],
                    'mfe_atr': mfe_data['mfe_atr'],
                    'mfe_margin': mfe_data['mfe_margin'],
                    'mae_margin': mfe_data['mae_margin'],
                    'stop_loss_margin': mfe_data['stop_loss_margin'],
                    'mfe_bar': mfe_data['mfe_bar'],
                    'stop_hit': mfe_data['stop_hit'],
                    'r_multiple': round(r_mult, 4),
                    'system': 'ABC',
                })

    df_abc = pd.DataFrame(records)
    print(f"  ABC 总信号: {len(df_abc)}")
    if len(df_abc) > 0:
        for s in [1, 2, 3, None]:
            sub = df_abc[df_abc['scenario'] == s]
            tag = f'场景{s}' if s else '无场景'
            print(f"    {tag}: {len(sub)}")
    return df_abc


# ============ Type1 系统 MFE ============

def run_type1_system(all_data):
    """跑 Type1 信号检测 + MFE + 分级，返回 DataFrame"""
    print("运行 Type1 系统...")

    all_trades = []
    for sym_key in sorted(all_data.keys()):
        df = all_data[sym_key]
        n = len(df)
        start = max(0, n - BACKTEST_DAYS * BARS_PER_DAY)
        trades = run_single_symbol(sym_key, df)
        trades = [t for t in trades if t['signal_bar_idx'] >= start]
        all_trades.extend(trades)

    df_all = pd.DataFrame(all_trades)
    df_filled = df_all[df_all['outcome'] == 'filled'].copy()

    if len(df_filled) == 0:
        return df_filled

    # 滚动胜率
    stop_dist = (df_filled['entry_price'] - df_filled['stop_price']).abs()
    r_mult = (df_filled['mfe_raw'] / stop_dist).fillna(0)
    df_filled['_is_profit'] = (r_mult >= 1).astype(int)
    df_filled = df_filled.sort_values(['symbol', 'signal_datetime']).reset_index(drop=True)
    df_filled['recent_win_n'] = -1
    for _, grp in df_filled.groupby('symbol'):
        wins = grp['_is_profit'].values
        idxs = grp.index
        for j_pos, idx in enumerate(idxs):
            if j_pos == 0:
                df_filled.at[idx, 'recent_win_n'] = -1
            else:
                s = max(0, j_pos - 5)
                prev = wins[s:j_pos]
                df_filled.at[idx, 'recent_win_n'] = int(prev.sum())
    df_filled.drop(columns=['_is_profit'], inplace=True)

    # 分级（互斥 + 重叠）
    df_filled['tier'] = df_filled.apply(classify_tier, axis=1)
    for t in TIER_ORDER:
        df_filled[f'in_{t}'] = False
    overlap_tiers = df_filled.apply(classify_tier_overlap, axis=1)
    for idx, tiers in overlap_tiers.items():
        for t in tiers:
            if t != 'gamma':
                df_filled.at[idx, f'in_{t}'] = True

    # R倍数
    df_filled['r_multiple'] = r_mult.values
    df_filled['system'] = 'Type1'

    print(f"  Type1 成交: {len(df_filled)}")
    tier_counts = df_filled['tier'].value_counts()
    for t in TIER_ORDER + ['gamma']:
        print(f"    {t}: {tier_counts.get(t, 0)}")

    return df_filled


# ============ MFE 统计辅助 ============

def mfe_stats_row(label, sub):
    """返回一行 MFE 统计"""
    if len(sub) == 0:
        return None
    mfe = sub['mfe_margin'].values
    r = sub['r_multiple'].values if 'r_multiple' in sub.columns else np.zeros(len(sub))
    stop_hit = sub['stop_hit'].astype(bool).sum() if 'stop_hit' in sub.columns else 0

    wr_r1 = (r >= 1).sum() / len(sub) * 100  # R>=1 胜率（MFE天花板到过1R）
    avg_r = r.mean()
    r3_pct = (r >= 3).sum() / len(sub) * 100
    loss_pct = (r < 1).sum() / len(sub) * 100

    return [label, len(sub),
            f'{np.mean(mfe):.2f}%', f'{np.median(mfe):.2f}%',
            f'{wr_r1:.1f}%', f'{avg_r:.2f}', f'{r3_pct:.1f}%', f'{loss_pct:.1f}%',
            f'{stop_hit/len(sub)*100:.1f}%']


# ============ 报告 ============

def build_report(df_abc, df_t1):
    """生成 MFE 对比报告"""
    rpt = Report('MFE 天花板对比：ABC场景系统 vs Type1 α/β')

    th = ['分组', 'N', 'avgMFE%', 'medMFE%', '胜率(R≥1)', 'avgR', 'R3+%', '亏损层(R<1)', '止损率']

    # ===== Section 1: 全量对比 =====
    rpt.add_section('1. 全量对比')
    rpt.add_text('两套系统在同一批数据上(36品种×120天×10min)的 MFE 天花板对比。')
    rpt.add_text('胜率 = MFE天花板到过1R（即价格曾有利移动超过止损距离）。')

    rows = []
    # ABC 全量
    r = mfe_stats_row('ABC 全量', df_abc)
    if r: rows.append(r)

    # Type1 全量
    r = mfe_stats_row('Type1 全量', df_t1)
    if r: rows.append(r)

    # Type1 α+β
    df_ab = df_t1[df_t1['tier'] != 'gamma']
    r = mfe_stats_row('Type1 α+β', df_ab)
    if r: rows.append(r)

    # Type1 γ
    df_gamma = df_t1[df_t1['tier'] == 'gamma']
    r = mfe_stats_row('Type1 γ(不做)', df_gamma)
    if r: rows.append(r)

    rpt.add_table(th, rows, highlight_pnl_cols=[2])

    # ===== Section 2: ABC 场景 vs Type1 子场景 =====
    rpt.add_section('2. 场景级对比')
    rpt.add_text('ABC 的3个可操作场景 vs Type1 的5个子场景（重叠版）。')

    rows = []
    # ABC 场景 1/2/3
    for s in [1, 2, 3]:
        sub = df_abc[df_abc['scenario'] == s]
        from signal_core import SCENARIO_NAMES
        label = f'ABC {SCENARIO_NAMES.get(s, f"场景{s}")}'
        # 截断太长的label
        label = label[:30]
        r = mfe_stats_row(label, sub)
        if r: rows.append(r)

    # ABC 无场景
    sub_none = df_abc[df_abc['scenario'].isna()]
    r = mfe_stats_row('ABC 无场景(不做)', sub_none)
    if r: rows.append(r)

    # 分隔
    rows.append(['---', '---', '---', '---', '---', '---', '---', '---', '---'])

    # Type1 子场景（重叠版）
    for tier in TIER_ORDER:
        col = f'in_{tier}'
        if col in df_t1.columns:
            sub = df_t1[df_t1[col] == True]
        else:
            sub = df_t1[df_t1['tier'] == tier]
        r = mfe_stats_row(f'Type1 {TIER_NAMES[tier]}', sub)
        if r: rows.append(r)

    rpt.add_table(th, rows, highlight_pnl_cols=[2])

    # ===== Section 3: 止损距离对比 =====
    rpt.add_section('3. 止损距离对比')
    rpt.add_text('止损距离(ATR单位)直接影响R倍数和胜率。')

    stop_th = ['分组', 'N', 'avg止损/ATR', 'med止损/ATR', 'P25', 'P75']
    stop_rows = []

    for s in [1, 2, 3]:
        sub = df_abc[df_abc['scenario'] == s]
        if len(sub) > 0:
            sd = sub['stop_dist_atr'].values
            stop_rows.append([f'ABC 场景{s}', len(sub),
                             f'{np.mean(sd):.2f}', f'{np.median(sd):.2f}',
                             f'{np.percentile(sd, 25):.2f}', f'{np.percentile(sd, 75):.2f}'])

    stop_rows.append(['---', '---', '---', '---', '---', '---'])

    for tier in TIER_ORDER:
        col = f'in_{tier}'
        if col in df_t1.columns:
            sub = df_t1[df_t1[col] == True]
        else:
            sub = df_t1[df_t1['tier'] == tier]
        if len(sub) > 0 and 'stop_dist_atr' in sub.columns:
            sd = sub['stop_dist_atr'].dropna().values
            if len(sd) > 0:
                stop_rows.append([f'Type1 {TIER_NAMES[tier][:15]}', len(sub),
                                 f'{np.mean(sd):.2f}', f'{np.median(sd):.2f}',
                                 f'{np.percentile(sd, 25):.2f}', f'{np.percentile(sd, 75):.2f}'])

    rpt.add_table(stop_th, stop_rows)

    # ===== Section 4: R倍数分布对比 =====
    rpt.add_section('4. R倍数分布结构')
    rpt.add_text('R倍数 = MFE / 止损距离。反映入场质量的理论天花板。')

    r_th = ['分组', 'N', 'R<1(亏)', 'R1~2', 'R2~3', 'R3~5', 'R5+(爆发)', 'avgR']

    def r_dist_row(label, sub):
        if len(sub) == 0 or 'r_multiple' not in sub.columns:
            return None
        r = sub['r_multiple'].values
        n = len(r)
        bins = [
            (r < 1, 'R<1'),
            ((r >= 1) & (r < 2), 'R1~2'),
            ((r >= 2) & (r < 3), 'R2~3'),
            ((r >= 3) & (r < 5), 'R3~5'),
            (r >= 5, 'R5+'),
        ]
        return [label, n] + [f'{mask.sum()/n*100:.1f}%' for mask, _ in bins] + [f'{r.mean():.2f}']

    rows = []
    for s in [1, 2, 3]:
        sub = df_abc[df_abc['scenario'] == s]
        r = r_dist_row(f'ABC 场景{s}', sub)
        if r: rows.append(r)

    rows.append(['---'] * 8)

    for tier in TIER_ORDER:
        col = f'in_{tier}'
        sub = df_t1[df_t1[col] == True] if col in df_t1.columns else df_t1[df_t1['tier'] == tier]
        r = r_dist_row(f'Type1 {TIER_NAMES[tier][:15]}', sub)
        if r: rows.append(r)

    rpt.add_table(r_th, rows, highlight_pnl_cols=[6])

    # ===== Section 5: 入场方式差异说明 =====
    rpt.add_section('5. 入场方式差异')
    rpt.add_text('两套系统的入场逻辑和MFE计算方式差异：')
    rpt.add_table(
        ['维度', 'ABC 场景系统', 'Type1 影线触碰'],
        [
            ['趋势判断', 'EMA20 > EMA120', 'ER(20) > 0.3 + close vs EMA60'],
            ['信号触发', 'A:影线弹回 B:1~3根回调 C:≥4根回调', '影线碰EMA10弹回(close不破)'],
            ['入场价', '信号K线收盘价（立即入场）', '信号K线high+1tick（挂单，5根有效）'],
            ['止损价', '回调极值-5跳', '信号K线low-5跳'],
            ['止损距离', '回调极值到收盘价 + buffer', '整根K线range + buffer（通常更大）'],
            ['场景过滤', 'ER≥0.5 + 偏离度门槛', '因子组合制(止损/热手/ER40)'],
            ['MFE起算', '信号K线之后第1根', '成交K线之后第1根（挂单可能延迟1~5根）'],
        ]
    )

    return rpt


# ============ Main ============

def main():
    print("=" * 70)
    print("  MFE 天花板对比：ABC场景系统 vs Type1 α/β")
    print("=" * 70)

    # 加载数据（需要所有 EMA）
    all_data = load_all(
        period_min=10,
        days=BACKTEST_DAYS + WARMUP_DAYS,
        last_days=None,
        emas=(5, 10, 20, 60, 120),
        er_periods=(5, 20, 40),
        atr_period=14,
        min_bars=200
    )
    print(f"加载了 {len(all_data)} 个品种")

    # 跑两套系统
    df_abc = run_abc_system(all_data)
    df_t1 = run_type1_system(all_data)

    if len(df_abc) == 0 or len(df_t1) == 0:
        print("数据不足，退出")
        return

    # 报告
    print("\n生成报告...")
    rpt = build_report(df_abc, df_t1)
    out_path = os.path.join('output', 'mfe_compare_report.html')
    rpt.save(out_path)
    print(f"报告已保存: {out_path}")

    # Console 摘要
    print(f'\n{"="*70}')
    print('  MFE 天花板对比摘要')
    print(f'{"="*70}')

    print(f'\nABC 场景系统:')
    for s in [1, 2, 3]:
        sub = df_abc[df_abc['scenario'] == s]
        if len(sub) > 0:
            r = sub['r_multiple'].values
            from signal_core import SCENARIO_NAMES
            label = SCENARIO_NAMES.get(s, f'场景{s}')[:30]
            print(f'  {label}: N={len(sub)}, avgMFE={sub["mfe_margin"].mean():.2f}%, '
                  f'胜率(R≥1)={((r>=1).sum()/len(r)*100):.1f}%, avgR={r.mean():.2f}, '
                  f'止损/ATR={sub["stop_dist_atr"].mean():.2f}')

    print(f'\nType1 子场景（重叠版）:')
    for tier in TIER_ORDER:
        col = f'in_{tier}'
        sub = df_t1[df_t1[col] == True] if col in df_t1.columns else df_t1[df_t1['tier'] == tier]
        if len(sub) > 0:
            r = sub['r_multiple'].values
            print(f'  {TIER_NAMES[tier]}: N={len(sub)}, avgMFE={sub["mfe_margin"].mean():.2f}%, '
                  f'胜率(R≥1)={((r>=1).sum()/len(r)*100):.1f}%, avgR={r.mean():.2f}, '
                  f'止损/ATR={sub["stop_dist_atr"].dropna().mean():.2f}')


if __name__ == '__main__':
    main()
