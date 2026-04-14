# -*- coding: utf-8 -*-
"""
Type1 α/β 出场策略研究
=======================
对 Type1（影线触碰EMA10）的 5 个子场景分别测试 8 种出场策略，
找出每个子场景的最优出场。

子场景：
  α-1: stop<1.5ATR + 热手>=4
  α-2: stop<1.5ATR + ER40>=0.42
  α-3: 热手>=4 + ER40>=0.42
  β-1: stop<1.5ATR + 热手>=3（不含α）
  β-2: 热手>=4 + density>=1（不含α，无stop<1.5无ER40）

用法:
    python test_type1_exit.py
"""

import sys
import io
import os
import numpy as np
import pandas as pd
from datetime import datetime

if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding != 'utf-8' and not sys.stdout.closed:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from data_loader import load_all, BARS_PER_DAY
from signal_core import (SYMBOL_CONFIGS, ExitTracker, DEFAULT_STOP_TICKS)
from stats_utils import calc_ev
from chart_engine import render_chart
from report_engine import Report

# 复用 test_trend_pullback 的信号检测
from test_trend_pullback import detect_signals, track_mfe_mae, round_to_tick

# ============ 参数 ============
ER_THRESHOLD = 0.3
PENDING_EXPIRY = 5
STOP_TICKS = 5
MFE_WINDOW = 60
MARGIN_RATE = 0.15
BACKTEST_DAYS = 120
WARMUP_DAYS = 50
EXIT_WINDOW = 60          # 出场追踪窗口

ALL_STRATEGIES = ['S1.1', 'S2', 'S2.1', 'S3.1', 'S5.1', 'S5.2', 'S6', 'S6.1']

# 子场景定义
TIER_NAMES = {
    'alpha-1': 'α-1: stop<1.5 + 热手≥4',
    'alpha-2': 'α-2: stop<1.5 + ER40≥0.42',
    'alpha-3': 'α-3: 热手≥4 + ER40≥0.42',
    'beta-1':  'β-1: stop<1.5 + 热手≥3',
    'beta-2':  'β-2: 热手≥4 + density≥1',
}
TIER_ORDER = ['alpha-1', 'alpha-2', 'alpha-3', 'beta-1', 'beta-2']


# ============ 分级逻辑 ============

def classify_tier(row):
    """
    返回子场景标签。按优先级归入唯一场景：
    α-1 > α-2 > α-3 > β-1 > β-2 > gamma
    """
    stop_ok = row.get('stop_dist_atr', 99) < 1.5
    hot4 = row.get('recent_win_n', -1) >= 4
    hot3 = row.get('recent_win_n', -1) >= 3
    er40_ok = row.get('er_40', 0) >= 0.42
    density_ok = row.get('signal_density', 0) >= 1

    # α: 双核命中（按优先级）
    if stop_ok and hot4:
        return 'alpha-1'
    if stop_ok and er40_ok:
        return 'alpha-2'
    if hot4 and er40_ok:
        return 'alpha-3'
    # β
    if stop_ok and hot3:
        return 'beta-1'
    if hot4 and density_ok and not stop_ok and not er40_ok:
        return 'beta-2'
    return 'gamma'


def classify_tier_overlap(row):
    """
    返回该笔交易满足的所有子场景（允许重叠）。
    """
    stop_ok = row.get('stop_dist_atr', 99) < 1.5
    hot4 = row.get('recent_win_n', -1) >= 4
    hot3 = row.get('recent_win_n', -1) >= 3
    er40_ok = row.get('er_40', 0) >= 0.42
    density_ok = row.get('signal_density', 0) >= 1

    tiers = []
    # α
    if stop_ok and hot4:
        tiers.append('alpha-1')
    if stop_ok and er40_ok:
        tiers.append('alpha-2')
    if hot4 and er40_ok:
        tiers.append('alpha-3')
    # β（只在不满足任何α时才算β）
    if not tiers:
        if stop_ok and hot3:
            tiers.append('beta-1')
        if hot4 and density_ok and not stop_ok and not er40_ok:
            tiers.append('beta-2')
    if not tiers:
        tiers.append('gamma')
    return tiers


# ============ 数据加载 + 信号检测 ============

def load_and_detect():
    """加载数据，检测信号，计算滚动胜率，返回 (df_filled, all_data)"""
    print("加载数据（含EMA5）...")
    all_data = load_all(
        period_min=10,
        days=BACKTEST_DAYS + WARMUP_DAYS,
        last_days=None,
        emas=(5, 10, 20, 60, 120),
        er_periods=(5, 20, 40),
        atr_period=14,
        min_bars=200
    )
    print(f"  加载了 {len(all_data)} 个品种")

    all_trades = []
    for sym_key in sorted(all_data.keys()):
        df = all_data[sym_key]
        n = len(df)
        start = max(0, n - BACKTEST_DAYS * BARS_PER_DAY)
        trades = run_single_symbol(sym_key, df)
        trades = [t for t in trades if t['signal_bar_idx'] >= start]
        all_trades.extend(trades)

    df_all = pd.DataFrame(all_trades)
    if len(df_all) == 0:
        print("没有检测到任何信号！")
        return pd.DataFrame(), all_data

    df_filled = df_all[df_all['outcome'] == 'filled'].copy()

    # 计算滚动胜率（同 test_trend_pullback.py）
    if len(df_filled) > 0:
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

    # MFE天花板用于参考
    if len(df_filled) > 0:
        df_filled['pnl_margin'] = df_filled['mfe_margin']

    # 时间列
    if len(df_filled) > 0 and 'signal_datetime' in df_filled.columns:
        try:
            dt = pd.to_datetime(df_filled['signal_datetime'])
            df_filled['month'] = dt.dt.to_period('M').astype(str)
            mid = dt.min() + (dt.max() - dt.min()) / 2
            df_filled['half'] = np.where(dt <= mid, 1, 2)
        except Exception:
            df_filled['month'] = 'unknown'
            df_filled['half'] = 1

    return df_filled, all_data


def run_single_symbol(sym_key, df):
    """信号检测 + MFE追踪，复用 detect_signals"""
    cfg = SYMBOL_CONFIGS.get(sym_key, {})
    tick_size = cfg.get('tick_size', 1.0)
    name = cfg.get('name', sym_key)

    all_signals = detect_signals(df, tick_size)

    trades = []
    for sig in all_signals:
        rec = {
            'symbol': sym_key,
            'name': name,
            'direction': sig['direction'],
            'signal_bar_idx': sig['signal_bar_idx'],
            'signal_datetime': sig['signal_datetime'],
            'outcome': sig['outcome'],
            'er_20': sig['er_20'],
            'atr': sig['atr'],
            'stop_dist_atr': sig.get('stop_dist_atr'),
            'signal_density': sig.get('signal_density'),
            'er_40': sig.get('er_40'),
            'signal_hour': sig.get('signal_hour'),
            'weekday': sig.get('weekday'),
        }

        if sig['outcome'] == 'filled':
            rec.update({
                'entry_bar_idx': sig['entry_bar_idx'],
                'entry_datetime': sig['entry_datetime'],
                'entry_price': sig['price'],
                'stop_price': sig['stop_price'],
                'bars_to_fill': sig['bars_to_fill'],
                'mfe_raw': sig['mfe_raw'],
                'mae_raw': sig['mae_raw'],
                'mfe_atr': sig['mfe_atr'],
                'mae_atr': sig['mae_atr'],
                'mfe_margin': sig['mfe_margin'],
                'mae_margin': sig['mae_margin'],
                'stop_loss_margin': sig['stop_loss_margin'],
                'mfe_bar': sig['mfe_bar'],
                'mae_bar': sig['mae_bar'],
                'stop_hit': sig['stop_hit'],
                'stop_hit_bar': sig['stop_hit_bar'],
                'bars_tracked': sig['bars_tracked'],
                # 信号K线原始价（用于 pullback_extreme）
                'high_at_signal': sig['high_at_signal'],
                'low_at_signal': sig['low_at_signal'],
                'close_at_signal': sig['close_at_signal'],
            })

        trades.append(rec)

    return trades


# ============ 出场模拟 ============

def simulate_exits(df_filled, all_data, prefix='',
                   use_real_extreme=False, track_from_signal=False):
    """
    对每笔 α/β 信号跑 ExitTracker（8种策略并行）。

    use_real_extreme: 止损用实际回调极值（信号→成交之间最低/最高）
    track_from_signal: 出场追踪从 signal_bar+1 开始（而非 entry_bar+1）

    入场价始终 = 挂单成交价（high+1tick），不变。
    prefix: 列名前缀
    """
    parts = []
    if use_real_extreme:
        parts.append('回调极值止损')
    if track_from_signal:
        parts.append('信号K线起算')
    mode = ' + '.join(parts) if parts else '原始'
    print(f"模拟出场策略 [{mode}]...")

    # 初始化出场列
    for s in ALL_STRATEGIES:
        df_filled[f'{prefix}{s}_pnl'] = 0.0
        df_filled[f'{prefix}{s}_reason'] = 'missing'
        df_filled[f'{prefix}{s}_bars'] = 0
        df_filled[f'{prefix}{s}_idx'] = None
        df_filled[f'{prefix}{s}_price'] = 0.0

    processed = 0
    skipped = 0

    for row_idx in df_filled.index:
        row = df_filled.loc[row_idx]
        sym = row['symbol']
        tier = row.get('tier', 'gamma')

        if tier == 'gamma':
            skipped += 1
            continue

        if sym not in all_data:
            skipped += 1
            continue

        df_sym = all_data[sym]
        n = len(df_sym)
        direction = row['direction']
        entry_price = row['entry_price']
        entry_idx = int(row['entry_bar_idx'])
        signal_idx = int(row['signal_bar_idx'])
        tick_size = SYMBOL_CONFIGS.get(sym, {}).get('tick_size', 1.0)

        # 确定 pullback_extreme
        if use_real_extreme:
            # 实际回调极值：signal_bar 到 entry_bar（含两端）的最低/最高价
            if direction == 'long':
                pullback_extreme = df_sym['low'].iloc[signal_idx:entry_idx + 1].min()
            else:
                pullback_extreme = df_sym['high'].iloc[signal_idx:entry_idx + 1].max()
        else:
            # 原始：信号K线的 low / high
            if direction == 'long':
                pullback_extreme = row['low_at_signal']
            else:
                pullback_extreme = row['high_at_signal']

        # 创建 ExitTracker — 入场价不变，只改 pullback_extreme
        tracker = ExitTracker(
            direction=direction,
            entry_price=entry_price,
            pullback_extreme=pullback_extreme,
            tick_size=tick_size,
            stop_ticks=DEFAULT_STOP_TICKS,
            ema5_strategies=True,
        )

        exit_results = {s: None for s in ALL_STRATEGIES}

        # 逐根追踪
        track_start = signal_idx if track_from_signal else entry_idx
        end_idx = min(n, track_start + EXIT_WINDOW + 1)
        for j in range(track_start + 1, end_idx):
            bar = df_sym.iloc[j]
            prev_bar = df_sym.iloc[j - 1]

            if pd.isna(bar['ema10']):
                continue

            ema5_val = bar.get('ema5', None)
            if ema5_val is not None and pd.isna(ema5_val):
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
        last_bar = df_sym.iloc[min(end_idx - 1, n - 1)]
        forced = tracker.force_close(last_bar['close'])
        for ev in forced:
            if ev.strategy in exit_results and exit_results[ev.strategy] is None:
                exit_results[ev.strategy] = {
                    'pnl': ev.pnl_pct,
                    'price': ev.exit_price,
                    'idx': min(end_idx - 1, n - 1),
                    'bars': ev.bars_held,
                    'reason': ev.exit_reason,
                }

        # 写入 DataFrame
        for s in ALL_STRATEGIES:
            r = exit_results.get(s)
            if r:
                df_filled.at[row_idx, f'{prefix}{s}_pnl'] = round(r['pnl'], 4)
                df_filled.at[row_idx, f'{prefix}{s}_reason'] = r['reason']
                df_filled.at[row_idx, f'{prefix}{s}_bars'] = r['bars']
                df_filled.at[row_idx, f'{prefix}{s}_idx'] = r['idx']
                df_filled.at[row_idx, f'{prefix}{s}_price'] = r['price']

        processed += 1
        if processed % 500 == 0:
            print(f"  已处理 {processed} 笔...")

    print(f"  完成: {processed} 笔处理, {skipped} 笔跳过(γ)")
    return df_filled


# ============ 报告生成 ============

def ev_row(label, pnls, reasons=None):
    """生成一行 EV 统计"""
    if reasons is not None:
        # 排除 backtest_end
        mask = [r != 'backtest_end' for r in reasons]
        pnls = [p for p, m in zip(pnls, mask) if m]
    st = calc_ev(pnls)
    if st['N'] == 0:
        return None
    return [label, st['N'], f'{st["EV"]:.2f}', f'{st["wr"]:.1f}%',
            f'{st["pr"]:.2f}', f'{st["avg_pnl"]:.2f}%', f'{st["sum_pnl"]:.1f}%']


def build_report(df_filled, all_data):
    """生成完整 HTML 报告"""
    rpt = Report('Type1 α/β 出场策略研究')

    # 只看 α/β
    df_ab = df_filled[df_filled['tier'] != 'gamma'].copy()
    n_total = len(df_filled)
    n_ab = len(df_ab)

    # ===== Section 1: 概览 =====
    rpt.add_section('1. 概览')
    rpt.add_text(f'品种: {len(all_data)} | 回测天数: {BACKTEST_DAYS} | 周期: 10min')
    rpt.add_text(f'全量成交: {n_total} | α+β: {n_ab} | γ跳过: {n_total - n_ab}')
    rpt.add_text(f'出场策略: {", ".join(ALL_STRATEGIES)} | 追踪窗口: {EXIT_WINDOW}根')

    # 各子场景概览
    rows = []
    for tier in TIER_ORDER:
        sub = df_ab[df_ab['tier'] == tier]
        if len(sub) == 0:
            continue
        avg_mfe = sub['mfe_margin'].mean()
        avg_stop = sub['stop_dist_atr'].mean() if 'stop_dist_atr' in sub.columns else 0
        rows.append([TIER_NAMES[tier], len(sub), f'{avg_mfe:.2f}%', f'{avg_stop:.2f}'])
    rpt.add_table(['子场景', 'N', 'avgMFE%（天花板）', 'avg止损/ATR'], rows,
                  highlight_pnl_cols=[2])

    # ===== Section 2: 互斥分配 — 5组×8策略 =====
    rpt.add_section('2A. 出场策略对比（互斥分配）')
    rpt.add_text('每笔信号只归入一个子场景（按优先级）。EV = 胜率×盈亏比 - 败率，排除backtest_end。')

    th = ['策略', 'N', 'EV', '胜率', '盈亏比', 'avg PnL%', '累计%']

    for tier in TIER_ORDER:
        sub = df_ab[df_ab['tier'] == tier]
        if len(sub) < 10:
            rpt.add_text(f'{TIER_NAMES[tier]}: N={len(sub)} < 10，跳过', color='#FFA500')
            continue

        rpt.add_section(f'[互斥] {TIER_NAMES[tier]} (N={len(sub)})')
        rows = []
        best_ev = -999
        best_s = None
        for s in ALL_STRATEGIES:
            pnls = sub[f'{s}_pnl'].tolist()
            reasons = sub[f'{s}_reason'].tolist()
            r = ev_row(s, pnls, reasons)
            if r:
                rows.append(r)
                ev_val = float(r[2])
                if ev_val > best_ev:
                    best_ev = ev_val
                    best_s = s
        rpt.add_table(th, rows, highlight_pnl_cols=[5, 6])
        if best_s:
            valid = sub[sub[f'{best_s}_reason'] != 'backtest_end']
            avg_bars = valid[f'{best_s}_bars'].mean() if len(valid) > 0 else 0
            rpt.add_text(f'▸ 推荐: {best_s} (EV={best_ev:.2f}, 平均持仓{avg_bars:.0f}根≈{avg_bars*10:.0f}分钟)',
                        color='#27AE60')

    # ===== Section 2B: 重叠分配 — 5组×8策略 =====
    rpt.add_section('2B. 出场策略对比（重叠分配）')
    rpt.add_text('同一笔信号可同时属于多个α子场景（如同时满足α-1和α-2）。β仍互斥。')
    rpt.add_text('重叠版显示每个子场景的完整样本，不受优先级影响。', color='#2980B9')

    for tier in TIER_ORDER:
        col = f'in_{tier}'
        if col not in df_ab.columns:
            continue
        sub = df_ab[df_ab[col] == True]
        if len(sub) < 10:
            rpt.add_text(f'{TIER_NAMES[tier]}: N={len(sub)} < 10，跳过', color='#FFA500')
            continue

        rpt.add_section(f'[重叠] {TIER_NAMES[tier]} (N={len(sub)})')
        rows = []
        best_ev = -999
        best_s = None
        for s in ALL_STRATEGIES:
            pnls = sub[f'{s}_pnl'].tolist()
            reasons = sub[f'{s}_reason'].tolist()
            r = ev_row(s, pnls, reasons)
            if r:
                rows.append(r)
                ev_val = float(r[2])
                if ev_val > best_ev:
                    best_ev = ev_val
                    best_s = s
        rpt.add_table(th, rows, highlight_pnl_cols=[5, 6])
        if best_s:
            valid = sub[sub[f'{best_s}_reason'] != 'backtest_end']
            avg_bars = valid[f'{best_s}_bars'].mean() if len(valid) > 0 else 0
            rpt.add_text(f'▸ 推荐: {best_s} (EV={best_ev:.2f}, 平均持仓{avg_bars:.0f}根≈{avg_bars*10:.0f}分钟)',
                        color='#27AE60')

    # ===== 互斥 vs 重叠 对比表 =====
    rpt.add_section('2C. 互斥 vs 重叠 对比')
    rpt.add_text('同一子场景在两种分配方式下的最优出场对比：')
    cmp_rows = []
    for tier in TIER_ORDER:
        # 互斥
        sub_ex = df_ab[df_ab['tier'] == tier]
        best_ex, ev_ex = _find_best_strategy(sub_ex) if len(sub_ex) >= 10 else (None, 0)
        # 重叠
        col = f'in_{tier}'
        sub_ov = df_ab[df_ab[col] == True] if col in df_ab.columns else pd.DataFrame()
        best_ov, ev_ov = _find_best_strategy(sub_ov) if len(sub_ov) >= 10 else (None, 0)

        if best_ex or best_ov:
            n_ex = len(sub_ex)
            n_ov = len(sub_ov)
            cmp_rows.append([
                TIER_NAMES[tier],
                f'{n_ex}', f'{best_ex or "-"}', f'{ev_ex:.2f}' if best_ex else '-',
                f'{n_ov}', f'{best_ov or "-"}', f'{ev_ov:.2f}' if best_ov else '-',
                f'{n_ov - n_ex:+d}' if n_ov != n_ex else '=',
            ])
    rpt.add_table(
        ['子场景', 'N(互斥)', '最优(互斥)', 'EV(互斥)',
         'N(重叠)', '最优(重叠)', 'EV(重叠)', 'ΔN'],
        cmp_rows
    )

    # ===== Section 2D: 出场起算对比 =====
    has_sig = f'sig_S6_pnl' in df_ab.columns
    if has_sig:
        rpt.add_section('2D. 出场起算对比：成交K线 vs 信号K线')
        rpt.add_text('同一批信号，入场价(挂单high+1tick)和止损(信号K线low-5跳)不变，只改出场追踪起点：')
        rpt.add_text('原始: 从 entry_bar+1 开始追踪（挂单成交后）')
        rpt.add_text('信号起算: 从 signal_bar+1 开始追踪（提前1~5根）')
        rpt.add_text('信号起算让出场策略多看到入场前的K线，更早建立回调/新高状态。', color='#2980B9')

        cmp_th = ['子场景', 'N',
                  '原始最优', '原始EV', '原始胜率', '原始累计',
                  '信号起算最优', '信号EV', '信号胜率', '信号累计']
        cmp_rows = []
        for tier in TIER_ORDER:
            col = f'in_{tier}'
            sub = df_ab[df_ab[col] == True] if col in df_ab.columns else pd.DataFrame()
            if len(sub) < 10:
                continue

            best1, ev1 = _find_best_strategy(sub)
            best2, ev2 = _find_best_strategy_prefix(sub, prefix='sig_')

            def _ev_str(sub, strat, prefix=''):
                if not strat:
                    return '-', '-', '-'
                v = sub[sub[f'{prefix}{strat}_reason'] != 'backtest_end']
                st = calc_ev(v[f'{prefix}{strat}_pnl'].tolist())
                return f'{st["EV"]:.2f}', f'{st["wr"]:.1f}%', f'{st["sum_pnl"]:.1f}%'

            ev1_s, wr1, cum1 = _ev_str(sub, best1, '')
            ev2_s, wr2, cum2 = _ev_str(sub, best2, 'sig_')

            cmp_rows.append([TIER_NAMES[tier], len(sub),
                            best1 or '-', ev1_s, wr1, cum1,
                            best2 or '-', ev2_s, wr2, cum2])

        rpt.add_table(cmp_th, cmp_rows)

        # S6 专项对比
        rpt.add_text('▸ S6 专项对比（固定看S6，不选最优）：')
        s6_th = ['子场景', 'N', '原始S6 EV', '原始S6 WR', '原始S6累计',
                 '信号S6 EV', '信号S6 WR', '信号S6累计']
        s6_rows = []
        for tier in TIER_ORDER:
            col = f'in_{tier}'
            sub = df_ab[df_ab[col] == True] if col in df_ab.columns else pd.DataFrame()
            if len(sub) < 10:
                continue

            def _s6_stats(sub, prefix=''):
                v = sub[sub[f'{prefix}S6_reason'] != 'backtest_end']
                if len(v) == 0:
                    return '-', '-', '-'
                st = calc_ev(v[f'{prefix}S6_pnl'].tolist())
                return f'{st["EV"]:.2f}', f'{st["wr"]:.1f}%', f'{st["sum_pnl"]:.1f}%'

            ev1, wr1, cum1 = _s6_stats(sub, '')
            ev2, wr2, cum2 = _s6_stats(sub, 'sig_')
            s6_rows.append([TIER_NAMES[tier], len(sub), ev1, wr1, cum1, ev2, wr2, cum2])

        rpt.add_table(s6_th, s6_rows)

    # ===== Section 3: 每组最优出场详分析 =====
    rpt.add_section('3. 各子场景最优出场详分析')

    for tier in TIER_ORDER:
        sub = df_ab[df_ab['tier'] == tier]
        if len(sub) < 10:
            continue

        # 找最优策略
        best_s, best_ev = _find_best_strategy(sub)
        if best_s is None:
            continue

        rpt.add_section(f'{TIER_NAMES[tier]} → {best_s}')
        pnl_col = f'{best_s}_pnl'
        reason_col = f'{best_s}_reason'
        bars_col = f'{best_s}_bars'

        # 排除 backtest_end
        valid = sub[sub[reason_col] != 'backtest_end'].copy()
        if len(valid) < 5:
            rpt.add_text(f'有效交易仅 {len(valid)} 笔，不足分析', color='#FFA500')
            continue

        # 3a. 多空对比
        rpt.add_text('▸ 多空对比：')
        rows = []
        for d, tag in [('long', '多头'), ('short', '空头')]:
            d_sub = valid[valid['direction'] == d]
            r = ev_row(tag, d_sub[pnl_col].tolist())
            if r:
                rows.append(r)
        if rows:
            rpt.add_table(th, rows, highlight_pnl_cols=[5, 6])

        # 3b. 品种分布（Top 10 + Bottom 5）
        rpt.add_text('▸ 品种分布（按EV排序）：')
        sym_rows = []
        for sym, grp in valid.groupby('symbol'):
            pnls = grp[pnl_col].tolist()
            st = calc_ev(pnls)
            name = SYMBOL_CONFIGS.get(sym, {}).get('name', sym)
            if st['N'] >= 3:
                sym_rows.append([name, st['N'], f'{st["EV"]:.2f}',
                                f'{st["wr"]:.0f}%', f'{st["sum_pnl"]:.1f}%'])
        sym_rows.sort(key=lambda x: float(x[2]), reverse=True)
        if sym_rows:
            rpt.add_table(['品种', 'N', 'EV', '胜率', '累计%'], sym_rows,
                          highlight_pnl_cols=[4])

        # 3c. 时间稳定性
        rpt.add_text('▸ 前半 vs 后半：')
        rows = []
        for h, tag in [(1, '前60天'), (2, '后60天')]:
            h_sub = valid[valid['half'] == h]
            r = ev_row(tag, h_sub[pnl_col].tolist())
            if r:
                rows.append(r)
        if rows:
            rpt.add_table(th, rows, highlight_pnl_cols=[5, 6])

    # ===== Section 4: 稳健性 =====
    rpt.add_section('4. 稳健性检查')

    from data_loader import HIGH_VOL

    for tier in TIER_ORDER:
        sub = df_ab[df_ab['tier'] == tier]
        if len(sub) < 10:
            continue

        best_s, _ = _find_best_strategy(sub)
        if best_s is None:
            continue

        pnl_col = f'{best_s}_pnl'
        reason_col = f'{best_s}_reason'
        valid = sub[sub[reason_col] != 'backtest_end']

        rpt.add_section(f'{TIER_NAMES[tier]} → {best_s} 稳健性')

        # 高波动 vs 常规
        rpt.add_text('▸ 高波动 vs 常规品种：')
        hv = valid[valid['symbol'].isin(HIGH_VOL)]
        nv = valid[~valid['symbol'].isin(HIGH_VOL)]
        rows = []
        r = ev_row('高波动', hv[pnl_col].tolist())
        if r: rows.append(r)
        r = ev_row('常规', nv[pnl_col].tolist())
        if r: rows.append(r)
        if rows:
            rpt.add_table(th, rows, highlight_pnl_cols=[5, 6])

        # 月度分解
        if 'month' in valid.columns:
            rpt.add_text('▸ 月度分解：')
            rows = []
            for m, grp in valid.groupby('month', sort=True):
                r = ev_row(m, grp[pnl_col].tolist())
                if r: rows.append(r)
            if rows:
                rpt.add_table(th, rows, highlight_pnl_cols=[5, 6])

    # ===== Section 5: K线图验证 =====
    rpt.add_section('5. K线图验证')

    for tier in TIER_ORDER:
        sub = df_ab[df_ab['tier'] == tier]
        if len(sub) < 5:
            continue

        best_s, _ = _find_best_strategy(sub)
        if best_s is None:
            continue

        pnl_col = f'{best_s}_pnl'
        reason_col = f'{best_s}_reason'
        valid = sub[sub[reason_col] != 'backtest_end']
        if len(valid) < 3:
            continue

        rpt.add_section(f'{TIER_NAMES[tier]} → {best_s} K线图')

        # Best 3
        best = valid.nlargest(min(3, len(valid)), pnl_col)
        for _, row in best.iterrows():
            _render_chart(rpt, row, all_data, best_s, 'BEST')

        # Worst 3
        worst = valid.nsmallest(min(3, len(valid)), pnl_col)
        for _, row in worst.iterrows():
            _render_chart(rpt, row, all_data, best_s, 'WORST')

        # Random 3
        n_rand = min(3, len(valid))
        rand = valid.sample(n=n_rand, random_state=42)
        for _, row in rand.iterrows():
            _render_chart(rpt, row, all_data, best_s, 'RANDOM')

    # ===== Section 6: 最终推荐 =====
    rpt.add_section('6. 最终推荐汇总')
    summary_rows = []
    for tier in TIER_ORDER:
        sub = df_ab[df_ab['tier'] == tier]
        if len(sub) < 10:
            continue
        best_s, best_ev = _find_best_strategy(sub)
        if best_s is None:
            continue
        valid = sub[sub[f'{best_s}_reason'] != 'backtest_end']
        st = calc_ev(valid[f'{best_s}_pnl'].tolist())
        avg_bars = valid[f'{best_s}_bars'].mean()
        summary_rows.append([
            TIER_NAMES[tier], len(sub), best_s,
            f'{st["EV"]:.2f}', f'{st["wr"]:.1f}%', f'{st["pr"]:.2f}',
            f'{st["sum_pnl"]:.1f}%', f'{avg_bars:.0f}根',
            f'{sub["mfe_margin"].mean():.2f}%',
        ])
    rpt.add_table(
        ['子场景', 'N', '推荐出场', 'EV', '胜率', '盈亏比', '累计%', '平均持仓', 'MFE天花板'],
        summary_rows, highlight_pnl_cols=[6]
    )

    return rpt


def _find_best_strategy(sub):
    """在子集中找 EV 最高的出场策略"""
    best_s = None
    best_ev = -999
    for s in ALL_STRATEGIES:
        pnls = sub[f'{s}_pnl'].tolist()
        reasons = sub[f'{s}_reason'].tolist()
        filtered = [p for p, r in zip(pnls, reasons) if r != 'backtest_end']
        if len(filtered) < 5:
            continue
        st = calc_ev(filtered)
        if st['EV'] > best_ev:
            best_ev = st['EV']
            best_s = s
    return best_s, best_ev


def _find_best_strategy_prefix(sub, prefix=''):
    """在子集中找 EV 最高的出场策略（支持列名前缀）"""
    best_s = None
    best_ev = -999
    for s in ALL_STRATEGIES:
        pnl_col = f'{prefix}{s}_pnl'
        reason_col = f'{prefix}{s}_reason'
        if pnl_col not in sub.columns:
            continue
        pnls = sub[pnl_col].tolist()
        reasons = sub[reason_col].tolist()
        filtered = [p for p, r in zip(pnls, reasons) if r != 'backtest_end']
        if len(filtered) < 5:
            continue
        st = calc_ev(filtered)
        if st['EV'] > best_ev:
            best_ev = st['EV']
            best_s = s
    return best_s, best_ev


def _render_chart(rpt, row, all_data, strategy, tag):
    """渲染一张K线图，标注入场+最优策略出场"""
    sym = row['symbol']
    if sym not in all_data:
        return
    df = all_data[sym]
    entry_idx = int(row['entry_bar_idx'])
    direction = row['direction']
    name = row.get('name', sym)

    exits = []

    # MFE 标注
    mfe_bar_idx = entry_idx + int(row['mfe_bar'])
    if direction == 'long':
        mfe_price = float(row['entry_price'] + row['mfe_raw'])
    else:
        mfe_price = float(row['entry_price'] - row['mfe_raw'])
    exits.append({'name': 'MFE', 'idx': mfe_bar_idx, 'price': mfe_price, 'color': '#FFD700'})

    # 最优策略出场标注
    exit_idx = row.get(f'{strategy}_idx')
    exit_price = row.get(f'{strategy}_price', 0)
    if exit_idx is not None and not pd.isna(exit_idx):
        exits.append({
            'name': strategy, 'idx': int(exit_idx),
            'price': float(exit_price), 'color': '#FF4444'
        })

    pnl = row.get(f'{strategy}_pnl', 0)
    bars = row.get(f'{strategy}_bars', 0)
    reason = row.get(f'{strategy}_reason', '')

    extra = {
        'MFE': f'{row["mfe_margin"]:.1f}%',
        f'{strategy}': f'{pnl:.2f}% ({reason}, {bars}根)',
        'Tier': row.get('tier', '?'),
        'Tag': tag,
    }

    title = f'[{tag}] {name} {direction} | {strategy}={pnl:.2f}%'

    chart_html = render_chart(
        df, entry_idx, direction,
        before_bars=20, after_bars=65,
        exits=exits,
        stop_price=float(row['stop_price']),
        ema_cols=['ema5', 'ema10', 'ema60'],
        title=title,
        extra_info=extra,
    )
    rpt.add_chart(chart_html)


# ============ Console 输出 ============

def print_summary(df_filled):
    """打印 console 摘要"""
    df_ab = df_filled[df_filled['tier'] != 'gamma']
    print(f'\n{"="*70}')
    print(f'  Type1 α/β 出场策略研究')
    print(f'{"="*70}')
    print(f'全量成交: {len(df_filled)} | α+β: {len(df_ab)} | γ: {len(df_filled) - len(df_ab)}')

    # 重叠版，两种入场模式对比
    print(f'\n--- 重叠分配 | 信号K线止损 vs 实际回调极值止损 ---')
    print(f'{"子场景":<25} {"N":>5}  {"原始最优":>8} {"原始EV":>7} {"原始WR":>7} {"原始累计":>8}  '
          f'{"极值最优":>8} {"极值EV":>7} {"极值WR":>7} {"极值累计":>8}')
    print('-' * 120)

    for tier in TIER_ORDER:
        col = f'in_{tier}'
        if col not in df_ab.columns:
            continue
        sub = df_ab[df_ab[col] == True]
        if len(sub) < 5:
            continue

        # 原始模式
        best_s1, ev1 = _find_best_strategy_prefix(sub, prefix='')
        st1 = None
        if best_s1:
            valid1 = sub[sub[f'{best_s1}_reason'] != 'backtest_end']
            st1 = calc_ev(valid1[f'{best_s1}_pnl'].tolist())

        # ABC模式
        best_s2, ev2 = _find_best_strategy_prefix(sub, prefix='sig_')
        st2 = None
        if best_s2:
            valid2 = sub[sub[f'sig_{best_s2}_reason'] != 'backtest_end']
            st2 = calc_ev(valid2[f'sig_{best_s2}_pnl'].tolist())

        line = f'{TIER_NAMES[tier]:<25} {len(sub):>5}  '
        if st1:
            line += f'{best_s1:>8} {st1["EV"]:>7.2f} {st1["wr"]:>6.1f}% {st1["sum_pnl"]:>7.1f}%  '
        else:
            line += f'{"-":>8} {"-":>7} {"-":>7} {"-":>8}  '
        if st2:
            line += f'{best_s2:>8} {st2["EV"]:>7.2f} {st2["wr"]:>6.1f}% {st2["sum_pnl"]:>7.1f}%'
        else:
            line += f'{"-":>8} {"-":>7} {"-":>7} {"-":>8}'
        print(line)

    # S6 专项对比
    print(f'\n--- S6 专项对比（重叠分配） ---')
    print(f'{"子场景":<25} {"N":>5}  {"原始S6 EV":>10} {"原始S6 WR":>10} {"原始S6累计":>11}  '
          f'{"极值S6 EV":>10} {"极值S6 WR":>10} {"极值S6累计":>11}')
    print('-' * 110)
    for tier in TIER_ORDER:
        col = f'in_{tier}'
        if col not in df_ab.columns:
            continue
        sub = df_ab[df_ab[col] == True]
        if len(sub) < 5:
            continue

        # 原始S6
        v1 = sub[sub['S6_reason'] != 'backtest_end']
        st1 = calc_ev(v1['S6_pnl'].tolist()) if len(v1) > 0 else None

        # ABC S6
        v2 = sub[sub['sig_S6_reason'] != 'backtest_end']
        st2 = calc_ev(v2['sig_S6_pnl'].tolist()) if len(v2) > 0 else None

        line = f'{TIER_NAMES[tier]:<25} {len(sub):>5}  '
        if st1 and st1['N'] > 0:
            line += f'{st1["EV"]:>10.2f} {st1["wr"]:>9.1f}% {st1["sum_pnl"]:>10.1f}%  '
        else:
            line += f'{"-":>10} {"-":>10} {"-":>11}  '
        if st2 and st2['N'] > 0:
            line += f'{st2["EV"]:>10.2f} {st2["wr"]:>9.1f}% {st2["sum_pnl"]:>10.1f}%'
        else:
            line += f'{"-":>10} {"-":>10} {"-":>11}'
        print(line)


# ============ Main ============

def main():
    df_filled, all_data = load_and_detect()
    if len(df_filled) == 0:
        print("无成交信号，退出")
        return

    # 分级（互斥版 + 重叠版）
    print("分级 α/β/γ...")
    df_filled['tier'] = df_filled.apply(classify_tier, axis=1)
    tier_counts = df_filled['tier'].value_counts()
    print("  互斥分配：")
    for t in TIER_ORDER + ['gamma']:
        print(f"    {t}: {tier_counts.get(t, 0)}")

    # 重叠版：每个tier一个bool列
    for t in TIER_ORDER:
        df_filled[f'in_{t}'] = False
    overlap_tiers = df_filled.apply(classify_tier_overlap, axis=1)
    for idx, tiers in overlap_tiers.items():
        for t in tiers:
            if t != 'gamma':
                df_filled.at[idx, f'in_{t}'] = True
    print("  重叠统计：")
    for t in TIER_ORDER:
        print(f"    {t}: {df_filled[f'in_{t}'].sum()}")

    # 出场模拟：原始 vs 信号K线起算
    df_filled = simulate_exits(df_filled, all_data, prefix='')
    df_filled = simulate_exits(df_filled, all_data, prefix='sig_', track_from_signal=True)

    # 报告
    print("生成报告...")
    rpt = build_report(df_filled, all_data)
    out_path = os.path.join('output', 'type1_exit_report.html')
    rpt.save(out_path)
    print(f"报告已保存: {out_path}")

    # Console摘要
    print_summary(df_filled)


if __name__ == '__main__':
    main()
