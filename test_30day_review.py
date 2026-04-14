# -*- coding: utf-8 -*-
"""
最近30天全场景回测：ABC场景1/2/3 + Type1 α/β分层
================================================
输出：R倍数质量分层表 × 每个场景 + 汇总EV
"""
import sys, io, math
if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from data_loader import load_all, sym_name, tick_size, BARS_PER_DAY, get_start_idx
from signal_core import (
    SignalDetector, ExitTracker, Type1SignalDetector, LadderRTracker,
    SYMBOL_CONFIGS, DEFAULT_STOP_TICKS,
    classify_scenario, classify_type1_tier,
    SCENARIO_EXIT, SCENARIO_PNL_COL, SCENARIO_NAMES, TYPE1_TIER_NAMES,
)
from stats_utils import calc_ev
from report_engine import Report

LAST_DAYS = 30
OUTPUT = 'output/report_30day_review.html'


# ============================================================
#  MFE 计算（60根窗口内最大有利偏移）
# ============================================================

def calc_mfe(df, entry_idx, direction, entry_price, stop_price,
             window=60, exit_bar=None):
    """
    返回 (mfe_raw, mfe_pct, mfe_bar, max_r)
    exit_bar: 实际出场的bar偏移量（从entry算起），限制MFE只算存活期间
    """
    is_long = (direction == 'long')
    stop_dist = abs(entry_price - stop_price) or 0.0001
    if exit_bar is not None:
        end_idx = min(entry_idx + exit_bar + 1, len(df))
    else:
        end_idx = min(entry_idx + window, len(df))
    best = 0.0
    best_bar = 0
    for j in range(entry_idx + 1, end_idx):
        h = df.iloc[j]['high']
        l = df.iloc[j]['low']
        if is_long:
            fav = h - entry_price
        else:
            fav = entry_price - l
        if fav > best:
            best = fav
            best_bar = j - entry_idx
    mfe_pct = best / entry_price * 100
    max_r = best / stop_dist
    return best, mfe_pct, best_bar, max_r


# ============================================================
#  ABC 场景回测
# ============================================================

def run_abc(all_data):
    """返回 DataFrame，每行一笔ABC场景交易"""
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

            # 出场模拟
            tracker = ExitTracker(
                direction=result.direction,
                entry_price=result.entry_price,
                pullback_extreme=result.pullback_extreme,
                tick_size=ts,
                stop_ticks=DEFAULT_STOP_TICKS,
                ema5_strategies=True,
            )
            stop_price = tracker.s6_stop if scenario in (1, 2) else tracker.s51_stop

            exit_results = {}
            for j in range(i + 1, n):
                bar = df.iloc[j]
                prev_bar = df.iloc[j - 1]
                if pd.isna(bar.get('ema10')):
                    continue
                ema5_val = bar.get('ema5', None)
                if ema5_val is not None and pd.isna(ema5_val):
                    ema5_val = None

                exit_events, _ = tracker.process_bar(
                    close=bar['close'], high=bar['high'], low=bar['low'],
                    ema10=bar['ema10'], prev_close=prev_bar['close'],
                    prev_high=prev_bar['high'], prev_low=prev_bar['low'],
                    ema5=ema5_val,
                )
                for ev in exit_events:
                    if ev.strategy not in exit_results:
                        exit_results[ev.strategy] = ev
                if tracker.all_done():
                    break

            # 强制平仓
            forced = tracker.force_close(df.iloc[-1]['close'])
            for ev in forced:
                if ev.strategy not in exit_results:
                    exit_results[ev.strategy] = ev

            # 对应场景的出场策略
            strat = SCENARIO_EXIT[scenario]
            strat_key = {'S6': 'S6', 'S5.1': 'S5.1'}[strat]
            ev = exit_results.get(strat_key)

            # MFE
            mfe_raw, mfe_pct, mfe_bar, max_r = calc_mfe(
                df, i, result.direction, result.entry_price, stop_price)

            records.append({
                'symbol': name,
                'sym_key': sym_key,
                'datetime': row['datetime'],
                'direction': result.direction,
                'type': result.signal_type,
                'scenario': scenario,
                'er20': round(er20, 3),
                'dev_atr': round(dev_atr, 2),
                'entry_price': result.entry_price,
                'stop_price': stop_price,
                'mfe_raw': mfe_raw,
                'mfe_margin': round(mfe_pct, 4),
                'mfe_bar': mfe_bar,
                'max_r': round(max_r, 4),
                'pnl': round(ev.pnl_pct, 4) if ev else None,
                'exit_reason': ev.exit_reason if ev else 'none',
                'bars_held': ev.bars_held if ev else 0,
            })

    return pd.DataFrame(records)


# ============================================================
#  Type1 回测
# ============================================================

def run_type1(all_data):
    """返回 DataFrame，每行一笔Type1成交交易"""
    records = []
    for sym_key, df in sorted(all_data.items()):
        n = len(df)
        signal_start = max(130, n - LAST_DAYS * BARS_PER_DAY)
        ts = tick_size(sym_key)
        name = sym_name(sym_key)

        detector = Type1SignalDetector()
        active_trackers = []  # (LadderRTracker, meta_dict)

        for i in range(2, n):
            row = df.iloc[i]
            if pd.isna(row.get('ema10')) or pd.isna(row.get('atr')) or row['atr'] <= 0:
                continue
            ema60 = row.get('ema60', None)
            er20 = row.get('er_20', None)
            er40 = row.get('er_40', None)
            if ema60 is None or pd.isna(ema60):
                continue
            if er20 is None or pd.isna(er20):
                er20 = 0.0
            if er40 is None or pd.isna(er40):
                er40 = 0.0

            # 检查挂单成交
            fill_result = detector.check_fill(
                high=row['high'], low=row['low'], opn=row['open'])
            if fill_result and fill_result['status'] == 'filled':
                sig = fill_result['signal']
                tier, preset = classify_type1_tier(
                    sig.stop_dist_atr, sig.recent_win_n, sig.er_40, sig.signal_density)

                if preset is not None and sig.bar_index >= signal_start - 5:
                    # 创建阶梯追踪器
                    tracker = LadderRTracker(
                        direction=sig.direction,
                        entry_price=sig.pending_price,
                        stop_price=sig.stop_price,
                        tick_size=ts,
                        preset=preset,
                    )
                    meta = {
                        'symbol': name, 'sym_key': sym_key,
                        'datetime': df.iloc[sig.bar_index]['datetime'],
                        'fill_datetime': row['datetime'],
                        'direction': sig.direction,
                        'tier': tier, 'preset': preset,
                        'entry_price': sig.pending_price,
                        'stop_price': sig.stop_price,
                        'stop_dist_atr': sig.stop_dist_atr,
                        'er_40': sig.er_40,
                        'signal_density': sig.signal_density,
                        'recent_win_n': sig.recent_win_n,
                        'entry_idx': i,
                    }
                    active_trackers.append((tracker, meta))

            # 处理活跃追踪器
            still_active = []
            for tr, meta in active_trackers:
                ev = tr.process_bar(close=row['close'], high=row['high'], low=row['low'])
                if ev:
                    mfe_raw, mfe_pct, mfe_bar, max_r = calc_mfe(
                        df, meta['entry_idx'], meta['direction'],
                        meta['entry_price'], meta['stop_price'],
                        exit_bar=ev.bars_held - 1)
                    win = max_r >= 1.0
                    detector.record_trade_result(win)
                    records.append({
                        **meta,
                        'mfe_raw': mfe_raw,
                        'mfe_margin': round(mfe_pct, 4),
                        'mfe_bar': mfe_bar,
                        'max_r': round(max_r, 4),
                        'pnl': ev.pnl_pct,
                        'exit_reason': ev.exit_reason,
                        'bars_held': ev.bars_held,
                    })
                else:
                    still_active.append((tr, meta))
            active_trackers = still_active

            # 信号检测
            detector.process_bar(
                close=row['close'], high=row['high'], low=row['low'],
                opn=row['open'],
                ema10=row['ema10'], ema60=ema60,
                er20=er20, er40=er40,
                atr=row['atr'], tick_size=ts,
            )

        # 强制平仓
        if n > 0:
            last_close = df.iloc[-1]['close']
            for tr, meta in active_trackers:
                ev = tr.force_close(last_close)
                if ev:
                    mfe_raw, mfe_pct, mfe_bar, max_r = calc_mfe(
                        df, meta['entry_idx'], meta['direction'],
                        meta['entry_price'], meta['stop_price'],
                        exit_bar=ev.bars_held - 1)
                    records.append({
                        **meta,
                        'mfe_raw': mfe_raw,
                        'mfe_margin': round(mfe_pct, 4),
                        'mfe_bar': mfe_bar,
                        'max_r': round(max_r, 4),
                        'pnl': ev.pnl_pct,
                        'exit_reason': ev.exit_reason,
                        'bars_held': ev.bars_held,
                    })

    return pd.DataFrame(records)


# ============================================================
#  质量分层表
# ============================================================

def build_tier_table(rpt, df, title, section_title=None):
    """R倍数质量分层表"""
    if df is None or len(df) == 0:
        if section_title:
            rpt.add_section(section_title)
        rpt.add_text(f'{title}: 无数据')
        return

    stop_dist = (df['entry_price'] - df['stop_price']).abs()
    r_mult = df['mfe_raw'] / stop_dist.replace(0, np.nan)
    r_mult = r_mult.fillna(0)

    total_n = len(df)
    total_mfe_sum = df['mfe_margin'].sum()

    def tier_row(label, mask, bold=False):
        sub = df[mask]
        if len(sub) == 0:
            return [f'{"**" if bold else ""}{label}{"**" if bold else ""}',
                    0, '0.0%', '-', '-', '-', '-', '-']
        rs = r_mult[mask]
        mfe_contrib = sub['mfe_margin'].sum() / total_mfe_sum * 100 if total_mfe_sum > 0 else 0
        p = '**' if bold else ''
        return [
            f'{p}{label}{p}',
            len(sub),
            f'{len(sub)/total_n*100:.1f}%',
            f'{sub["mfe_margin"].mean():.2f}%',
            f'{sub["mfe_margin"].median():.2f}%',
            f'{rs.mean():.2f}',
            f'{mfe_contrib:.1f}%',
            f'{sub["mfe_bar"].mean():.0f}',
        ]

    tiers = [
        ('R=0 入场即反转',   r_mult == 0),
        ('R 0~0.5 未到半损',  (r_mult > 0) & (r_mult < 0.5)),
        ('R 0.5~1 未到保本',  (r_mult >= 0.5) & (r_mult < 1)),
        ('R 1~2 到过1:1',    (r_mult >= 1) & (r_mult < 2)),
        ('R 2~3 到过2:1',    (r_mult >= 2) & (r_mult < 3)),
        ('R 3+ 超过3:1',     r_mult >= 3),
    ]

    headers = ['层级', 'N', '占比', 'avg MFE%', 'med MFE%',
               'avg R', 'MFE贡献', 'avg MFE bar']
    rows = [tier_row(label, mask) for label, mask in tiers]

    # 汇总行
    rows.append(tier_row('亏损层 (R<1)', r_mult < 1, bold=True))
    rows.append(tier_row('盈利层 (R≥1)', r_mult >= 1, bold=True))
    rows.append(tier_row('合计', pd.Series(True, index=df.index), bold=True))

    if section_title:
        rpt.add_section(section_title)
    rpt.add_text(f'<b>{title}</b>  (N={total_n}，R=MFE/止损距离，60根窗口)')
    rpt.add_table(headers, rows, highlight_pnl_cols=[3, 4])

    # 胜率 + avgR
    wr = (r_mult >= 1).sum() / total_n * 100
    avg_r = r_mult.mean()
    rpt.add_text(f'胜率(R≥1): <b>{wr:.1f}%</b> | avgR: <b>{avg_r:.2f}</b> | '
                 f'avgMFE: {df["mfe_margin"].mean():.2f}%')


def build_ev_summary(rpt, df, pnl_col='pnl', label=''):
    """EV统计行"""
    valid = df.dropna(subset=[pnl_col])
    valid = valid[valid['exit_reason'] != 'backtest_end']
    if len(valid) == 0:
        rpt.add_text(f'{label} EV统计: 无有效数据（全部backtest_end）')
        return
    ev = calc_ev(valid[pnl_col].tolist())
    rpt.add_text(
        f'{label} 出场EV: <b>EV={ev["EV"]:+.2f}</b> | '
        f'N={ev["N"]} | 胜率={ev["wr"]:.1f}% | 盈亏比={ev["pr"]:.2f} | '
        f'累计={ev["sum_pnl"]:+.2f}% | 均值={ev["avg_pnl"]:+.4f}%'
    )


def build_direction_split(rpt, df, pnl_col='pnl'):
    """多空拆分EV"""
    for d, label in [('long', '做多'), ('short', '做空')]:
        sub = df[(df['direction'] == d) & (df['exit_reason'] != 'backtest_end')]
        if len(sub) == 0:
            continue
        ev = calc_ev(sub[pnl_col].tolist())
        r_mult = sub['mfe_raw'] / (sub['entry_price'] - sub['stop_price']).abs().replace(0, np.nan)
        r_mult = r_mult.fillna(0)
        wr = (r_mult >= 1).sum() / len(sub) * 100
        rpt.add_text(
            f'  {label}: N={ev["N"]} | EV={ev["EV"]:+.2f} | '
            f'胜率(R≥1)={wr:.1f}% | avgR={r_mult.mean():.2f} | 累计={ev["sum_pnl"]:+.2f}%'
        )


def build_symbol_breakdown(rpt, df, pnl_col='pnl'):
    """逐品种统计表"""
    if len(df) == 0:
        return
    stop_dist = (df['entry_price'] - df['stop_price']).abs()
    df = df.copy()
    df['r_mult'] = df['mfe_raw'] / stop_dist.replace(0, np.nan)
    df['r_mult'] = df['r_mult'].fillna(0)

    headers = ['品种', 'N', '胜率(R≥1)', 'avgR', 'avgMFE%', '累计PnL%', '多', '空']
    rows = []
    for sym, sub in df.groupby('symbol'):
        valid = sub[sub['exit_reason'] != 'backtest_end']
        n = len(sub)
        wr = (sub['r_mult'] >= 1).sum() / n * 100 if n > 0 else 0
        avg_r = sub['r_mult'].mean()
        avg_mfe = sub['mfe_margin'].mean()
        cum_pnl = valid[pnl_col].sum() if len(valid) > 0 else 0
        n_long = len(sub[sub['direction'] == 'long'])
        n_short = len(sub[sub['direction'] == 'short'])
        rows.append([
            sym, n, f'{wr:.0f}%', f'{avg_r:.2f}',
            f'{avg_mfe:.2f}%', f'{cum_pnl:+.2f}%',
            n_long, n_short,
        ])
    rows.sort(key=lambda x: float(x[5].replace('%', '').replace('+', '')), reverse=True)
    rpt.add_table(headers, rows, highlight_pnl_cols=[5])


# ============================================================
#  主函数
# ============================================================

def main():
    print(f"{'='*60}")
    print(f"  最近{LAST_DAYS}天全场景回测 (ABC + Type1)")
    print(f"{'='*60}")

    # 加载数据（用缓存，不裁剪，信号范围由signal_start控制）
    all_data = load_all(
        period_min=10, days=170, last_days=None,
        emas=(5, 10, 20, 60, 120),
        er_periods=(5, 20, 40),
        atr_period=14,
        min_bars=200,
    )
    print(f"加载品种数: {len(all_data)}")

    # --- ABC ---
    print("\n[ABC场景] 信号检测+出场模拟...")
    df_abc = run_abc(all_data)
    print(f"  ABC总信号: {len(df_abc)}笔")
    for s in [1, 2, 3]:
        sub = df_abc[df_abc['scenario'] == s]
        print(f"  场景{s}: {len(sub)}笔")

    # --- Type1 ---
    print("\n[Type1] 信号检测+挂单成交+出场模拟...")
    df_t1 = run_type1(all_data)
    print(f"  Type1总成交: {len(df_t1)}笔")
    if len(df_t1) > 0:
        for tier in ['alpha-1', 'alpha-2', 'alpha-3', 'beta-1', 'beta-2']:
            sub = df_t1[df_t1['tier'] == tier]
            if len(sub) > 0:
                print(f"  {TYPE1_TIER_NAMES[tier]}: {len(sub)}笔")

    # ========== 报告生成 ==========
    rpt = Report(f'最近{LAST_DAYS}天全场景回测')

    rpt.add_section('概览')
    rpt.add_text(f'回测范围: 最近{LAST_DAYS}个交易日 | 品种: {len(all_data)}个 | '
                 f'周期: 10min')

    # --- ABC 场景汇总 ---
    rpt.add_section('一、ABC场景汇总')
    summary_headers = ['场景', '笔数', '出场策略', '多/空', '累计PnL%', 'EV', '胜率', '盈亏比']
    summary_rows = []
    for s in [1, 2, 3]:
        sub = df_abc[df_abc['scenario'] == s]
        if len(sub) == 0:
            summary_rows.append([f'场景{s}', 0, '-', '-', '-', '-', '-', '-'])
            continue
        valid = sub[sub['exit_reason'] != 'backtest_end']
        ev = calc_ev(valid['pnl'].dropna().tolist()) if len(valid) > 0 else calc_ev([])
        n_l = len(sub[sub['direction'] == 'long'])
        n_s = len(sub[sub['direction'] == 'short'])
        summary_rows.append([
            SCENARIO_NAMES[s].split('→')[0].strip(),
            len(sub), SCENARIO_EXIT[s],
            f'{n_l}/{n_s}',
            f'{ev["sum_pnl"]:+.2f}%',
            f'{ev["EV"]:+.2f}',
            f'{ev["wr"]:.1f}%',
            f'{ev["pr"]:.2f}',
        ])
    rpt.add_table(summary_headers, summary_rows, highlight_pnl_cols=[4, 5])

    # --- 每个场景详细分层 ---
    for s in [1, 2, 3]:
        sub = df_abc[df_abc['scenario'] == s]
        title = SCENARIO_NAMES[s]
        build_tier_table(rpt, sub, f'质量分层', section_title=f'场景{s}: {title}')
        if len(sub) > 0:
            build_ev_summary(rpt, sub, label=f'场景{s} {SCENARIO_EXIT[s]}')
            build_direction_split(rpt, sub)
            build_symbol_breakdown(rpt, sub)

    # --- Type1 汇总 ---
    rpt.add_section('二、Type1场景汇总')
    if len(df_t1) == 0:
        rpt.add_text('Type1: 30天内无成交信号')
    else:
        t1_headers = ['子场景', '笔数', '出场预设', '多/空', '累计PnL%', 'EV', '胜率', '盈亏比']
        t1_rows = []
        for tier in ['alpha-1', 'alpha-2', 'alpha-3', 'beta-1', 'beta-2']:
            sub = df_t1[df_t1['tier'] == tier]
            if len(sub) == 0:
                continue
            valid = sub[sub['exit_reason'] != 'backtest_end']
            ev = calc_ev(valid['pnl'].dropna().tolist()) if len(valid) > 0 else calc_ev([])
            n_l = len(sub[sub['direction'] == 'long'])
            n_s = len(sub[sub['direction'] == 'short'])
            t1_rows.append([
                TYPE1_TIER_NAMES[tier],
                len(sub), sub.iloc[0]['preset'],
                f'{n_l}/{n_s}',
                f'{ev["sum_pnl"]:+.2f}%',
                f'{ev["EV"]:+.2f}',
                f'{ev["wr"]:.1f}%',
                f'{ev["pr"]:.2f}',
            ])
        # α合并行
        alpha_tiers = df_t1[df_t1['tier'].str.startswith('alpha')]
        if len(alpha_tiers) > 0:
            valid = alpha_tiers[alpha_tiers['exit_reason'] != 'backtest_end']
            ev = calc_ev(valid['pnl'].dropna().tolist()) if len(valid) > 0 else calc_ev([])
            n_l = len(alpha_tiers[alpha_tiers['direction'] == 'long'])
            n_s = len(alpha_tiers[alpha_tiers['direction'] == 'short'])
            t1_rows.append([
                '**α合计**', len(alpha_tiers), 'I',
                f'{n_l}/{n_s}',
                f'{ev["sum_pnl"]:+.2f}%',
                f'{ev["EV"]:+.2f}',
                f'{ev["wr"]:.1f}%',
                f'{ev["pr"]:.2f}',
            ])
        rpt.add_table(t1_headers, t1_rows, highlight_pnl_cols=[4, 5])

        # α合并分层表
        if len(alpha_tiers) > 0:
            build_tier_table(rpt, alpha_tiers, 'α合并质量分层',
                             section_title='Type1 α档（stop<1.5 + 热手/ER40组合）')
            build_ev_summary(rpt, alpha_tiers, label='α合并 LR_I')
            build_direction_split(rpt, alpha_tiers)
            build_symbol_breakdown(rpt, alpha_tiers)

        # β-1
        beta1 = df_t1[df_t1['tier'] == 'beta-1']
        if len(beta1) > 0:
            build_tier_table(rpt, beta1, 'β-1质量分层',
                             section_title='Type1 β-1（stop<1.5 + 热手≥3）')
            build_ev_summary(rpt, beta1, label='β-1 LR_I')
            build_direction_split(rpt, beta1)

        # β-2
        beta2 = df_t1[df_t1['tier'] == 'beta-2']
        if len(beta2) > 0:
            build_tier_table(rpt, beta2, 'β-2质量分层',
                             section_title='Type1 β-2（热手≥4 + density≥1）')
            build_ev_summary(rpt, beta2, label='β-2 LR_2R')
            build_direction_split(rpt, beta2)

    # --- 全局汇总 ---
    rpt.add_section('三、全系统汇总')
    total_abc = len(df_abc)
    total_t1 = len(df_t1[df_t1['tier'] != 'gamma']) if len(df_t1) > 0 else 0
    rpt.add_text(f'ABC场景信号: {total_abc}笔 | Type1(α+β): {total_t1}笔 | '
                 f'合计: {total_abc + total_t1}笔')

    if total_abc > 0:
        abc_valid = df_abc[df_abc['exit_reason'] != 'backtest_end']
        ev_abc = calc_ev(abc_valid['pnl'].dropna().tolist())
        rpt.add_text(f'ABC整体: EV={ev_abc["EV"]:+.2f} | 累计={ev_abc["sum_pnl"]:+.2f}%')
    if total_t1 > 0:
        t1_valid = df_t1[df_t1['exit_reason'] != 'backtest_end']
        ev_t1 = calc_ev(t1_valid['pnl'].dropna().tolist())
        rpt.add_text(f'Type1整体: EV={ev_t1["EV"]:+.2f} | 累计={ev_t1["sum_pnl"]:+.2f}%')

    # 保存
    rpt.save(OUTPUT)
    print(f"\n报告已保存: {OUTPUT}")
    print("用浏览器打开查看完整分层数据")


if __name__ == '__main__':
    main()
