# -*- coding: utf-8 -*-
"""
最近5个交易日真实期权回测
=========================
1. 找出最近5个交易日的信号（用缓存期货数据）
2. 用天勤拉真实期权K线数据
3. 用真实价格计算各止盈策略收益
4. 固定每笔2万人民币开仓

输出: output/real_option_5d.html
"""

import sys
import io
import os
import json
import time
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from data_loader import load_all, sym_name, tick_size, multiplier, add_indicators
from signal_core import (SignalDetector, ExitTracker, SYMBOL_CONFIGS,
                         DEFAULT_STOP_TICKS, classify_scenario, SCENARIO_NAMES)
from stats_utils import calc_ev
from volatility import add_hv
from option_pricing import black76_greeks, black76_price, find_strike_by_delta, R_DEFAULT
from report_engine import Report


# ============================================================
#  配置
# ============================================================

POSITION_SIZE = 20000        # 每笔固定2万元
LAST_DAYS = 5                # 最近5个交易日
BARS_PER_DAY = 57
TRADING_DAYS_YEAR = 245
MAX_HOLD_BARS = 144          # 最长观察24小时

# 止盈档位（期权ROI%）
TP_LEVELS = [50, 80, 100, 150, 200, 300]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
#  Step 1: 找信号
# ============================================================

def find_recent_signals():
    """在最近5个交易日内找出所有信号"""
    print("=" * 70)
    print("Step 1: 查找最近5个交易日信号")
    print("=" * 70)

    all_data = load_all(period_min=10, days=170, last_days=None,
                        emas=(10, 20, 120), er_periods=(5, 20, 40), atr_period=14)
    print(f"  加载品种数: {len(all_data)}")

    for sym_key, df in all_data.items():
        add_hv(df, windows=[20, 40])
        df['er5_delta_6'] = df['er_5'] - df['er_5'].shift(6)

    signals = []

    # 计算最近5个交易日的日期范围（按实际日期，不按K线数量）
    all_dates = set()
    for sym_key, df in all_data.items():
        if 'datetime' in df.columns:
            all_dates.update(df['datetime'].dt.date.unique())
    sorted_dates = sorted(all_dates)
    last_5_dates = set(sorted_dates[-LAST_DAYS:]) if len(sorted_dates) >= LAST_DAYS else set(sorted_dates)
    # 信号必须在这些日期内，且之后至少有6小时(36根)的数据才能跟踪
    MIN_REMAINING_BARS = 36  # 至少6小时后续数据
    print(f"  最近{LAST_DAYS}个交易日: {sorted(last_5_dates)}")

    for sym_key, df in sorted(all_data.items()):
        n = len(df)
        name = sym_name(sym_key)
        detector = SignalDetector(signal_types='ABC')

        for i in range(max(2, 130), n):
            row = df.iloc[i]
            if pd.isna(row.get('ema10')) or pd.isna(row.get('ema20')) or pd.isna(row.get('ema120')):
                continue
            if pd.isna(row.get('atr')) or row['atr'] <= 0:
                continue

            result = detector.process_bar(
                close=row['close'], high=row['high'], low=row['low'],
                ema10=row['ema10'], ema20=row['ema20'], ema120=row['ema120'],
            )
            if result is None:
                continue

            # 信号必须在最近5个交易日内
            if 'datetime' in df.columns:
                sig_date = row['datetime'].date()
                if sig_date not in last_5_dates:
                    continue

            # 信号后必须有足够的K线来跟踪（否则还在持仓中）
            remaining = n - i - 1
            if remaining < MIN_REMAINING_BARS:
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

            signal_time = row['datetime'] if 'datetime' in df.columns else None

            signals.append({
                'symbol': name,
                'sym_key': sym_key,
                'idx': i,
                'df_key': sym_key,
                'direction': result.direction,
                'type': result.signal_type,
                'scenario': scenario,
                'entry_price': result.entry_price,
                'pullback_extreme': result.pullback_extreme,
                'hv': hv,
                'atr': atr,
                'signal_time': signal_time,
                'er20': er20,
                'deviation_atr': round(deviation_atr, 2),
            })

    print(f"\n  最近{LAST_DAYS}天信号数: {len(signals)}笔")
    for s in signals:
        print(f"    {s['signal_time']} {s['symbol']:6s} {s['direction']:5s} "
              f"场景{s['scenario']} {s['type']}类 F={s['entry_price']}")

    return signals, all_data


# ============================================================
#  Step 2: 选期权合约 + 拉K线
# ============================================================

def parse_option_symbol(opt_sym):
    """解析期权合约名，提取行权价和call/put

    格式：
    - SHFE: SHFE.ag2606C8000, SHFE.ag2606P7000
    - CZCE: CZCE.TA605C4800, CZCE.SA605P1020
    - DCE:  DCE.lh2605-C-15200, DCE.jm2605-P-1100
    - GFEX: GFEX.lc2605-C-154000
    """
    # DCE/GFEX格式: xxx-C-12345 或 xxx-P-12345
    match_dash = re.search(r'-([CP])-(\d+)$', opt_sym)
    if match_dash:
        cp = 'call' if match_dash.group(1) == 'C' else 'put'
        strike = float(match_dash.group(2))
        return cp, strike

    # SHFE/CZCE格式: xxxC12345 或 xxxP12345
    name = opt_sym.split('.')[-1]
    match = re.search(r'([CP])(\d+)$', name)
    if match:
        cp = 'call' if match.group(1) == 'C' else 'put'
        strike = float(match.group(2))
        return cp, strike

    return None, None


def select_option_contracts(api, sym_key, direction, entry_price, hv):
    """
    选择最佳期权合约：
    1. 查询期权链
    2. 按理论Delta排序
    3. 返回多个候选（从低Delta到高Delta），后面拉K线时验证流动性
    """
    probe_path = os.path.join(SCRIPT_DIR, 'probe_results.json')
    with open(probe_path, 'r', encoding='utf-8') as f:
        probe = json.load(f)

    info = probe['results'].get(sym_key, {})
    main_contract = info.get('main_contract', '')
    if not main_contract or not info.get('has_options', False):
        return []

    try:
        opts = api.query_options(main_contract, expired=False)
        if not opts:
            return []
    except Exception as e:
        print(f"    查询期权失败: {e}")
        return []

    cp_want = 'call' if direction == 'long' else 'put'
    T_approx = 20 / TRADING_DAYS_YEAR

    candidates = []
    for opt_sym in opts:
        cp, strike = parse_option_symbol(opt_sym)
        if cp != cp_want:
            continue
        if strike <= 0:
            continue

        # 用Black-76算理论Delta
        greeks = black76_greeks(entry_price, strike, T_approx, R_DEFAULT, hv, cp)
        delta_abs = abs(greeks['delta'])
        theo_price = greeks['price']

        candidates.append({
            'symbol': opt_sym,
            'strike': strike,
            'delta': round(delta_abs, 3),
            'theo_price': round(theo_price, 2),
        })

    # 按Delta排序
    candidates.sort(key=lambda x: x['delta'])

    # 过滤：Delta 0.05~0.60，且理论价 > 3（避免废纸合约）
    return [c for c in candidates if 0.05 <= c['delta'] <= 0.60 and c['theo_price'] > 3]


def pull_option_klines(api, opt_symbol, bars=500):
    """拉取期权10分钟K线"""
    try:
        klines = api.get_kline_serial(opt_symbol, duration_seconds=600, data_length=bars)

        deadline = time.time() + 15
        while True:
            api.wait_update(deadline=time.time() + 3)
            if len(klines) > 0 and not pd.isna(klines.iloc[-1]['close']):
                break
            if time.time() > deadline:
                break

        df = klines.copy()
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ns')
        # 天勤返回UTC时间，统一加8小时转北京时间
        # （与data_loader.load_symbol的逻辑一致）
        df['datetime'] = df['datetime'] + pd.Timedelta(hours=8)

        cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'open_oi', 'close_oi']
        keep = [c for c in cols if c in df.columns]
        df = df[keep].copy()
        df = df.dropna(subset=['close']).reset_index(drop=True)
        return df

    except Exception as e:
        print(f"    拉取K线失败 {opt_symbol}: {e}")
        return None


def select_best_with_real_data(api, candidates, signal_time, kline_cache):
    """
    合约选择策略（用户指定）：
    - Delta 0.05~0.30 范围内
    - 选当天成交量最高的合约
    - 如果成交量差距在10%以内，选Delta最低的

    kline_cache: dict，缓存已拉过的期权K线
    """
    # 只看 Delta 0.05~0.30
    low_delta_cands = [c for c in candidates if 0.05 <= c['delta'] <= 0.30]
    if not low_delta_cands:
        # 放宽到0.40
        low_delta_cands = [c for c in candidates if 0.05 <= c['delta'] <= 0.40]
    if not low_delta_cands:
        low_delta_cands = candidates

    checked = []

    for cand in low_delta_cands:
        opt_sym = cand['symbol']

        # 拉K线（带缓存）
        if opt_sym not in kline_cache:
            opt_df = pull_option_klines(api, opt_sym, bars=500)
            kline_cache[opt_sym] = opt_df
        else:
            opt_df = kline_cache[opt_sym]

        if opt_df is None or len(opt_df) == 0:
            continue

        # 检查信号时刻附近是否有数据
        time_diff = (opt_df['datetime'] - signal_time).abs()
        min_diff = time_diff.min()
        if min_diff > pd.Timedelta(minutes=30):
            continue

        match_idx = time_diff.idxmin()
        match_bar = opt_df.iloc[match_idx]

        entry_price = match_bar['close']
        if entry_price <= 0:
            continue

        # 当天成交量
        signal_date = signal_time.date() if hasattr(signal_time, 'date') else None
        if signal_date:
            day_bars = opt_df[opt_df['datetime'].dt.date == signal_date]
        else:
            day_bars = opt_df
        daily_volume = day_bars['volume'].sum() if 'volume' in day_bars.columns else 0

        oi = match_bar.get('close_oi', match_bar.get('open_oi', 0))
        if pd.isna(oi):
            oi = 0

        info = {
            'symbol': opt_sym,
            'strike': cand['strike'],
            'delta': cand['delta'],
            'theo_price': cand['theo_price'],
            'opt_df': opt_df,
            'entry_price_real': entry_price,
            'daily_volume': daily_volume,
            'open_interest': oi,
        }
        checked.append(info)

    if not checked:
        return None

    # 选择逻辑：成交量最高的；如果差距10%以内，选Delta最低的
    # 1. 按成交量降序排列
    checked.sort(key=lambda x: -x['daily_volume'])
    max_vol = checked[0]['daily_volume']

    if max_vol <= 0:
        # 没有成交量数据，直接按Delta排序选最低的
        checked.sort(key=lambda x: x['delta'])
        return checked[0]

    # 2. 找出成交量在最高值90%以上的（差距10%以内）
    threshold = max_vol * 0.90
    top_group = [c for c in checked if c['daily_volume'] >= threshold]

    # 3. 在这个组里选Delta最低的
    top_group.sort(key=lambda x: x['delta'])
    return top_group[0]


# ============================================================
#  Step 3: 用真实价格计算收益
# ============================================================

def simulate_real_option(futures_df, opt_df, sig, opt_info, mult):
    """用真实期权价格模拟交易"""
    idx = sig['idx']
    n = len(futures_df)
    entry_time = sig['signal_time']
    is_long = sig['direction'] == 'long'

    # 在期权K线中找入场时刻
    time_diff = (opt_df['datetime'] - entry_time).abs()
    match_idx = time_diff.idxmin()
    if time_diff.iloc[match_idx] > pd.Timedelta(minutes=30):
        return None

    entry_bar = opt_df.iloc[match_idx]
    entry_price_opt = entry_bar['close']
    if entry_price_opt <= 0:
        return None

    # 计算手数（2万元 / 每手成本）
    cost_per_lot = entry_price_opt * mult
    if cost_per_lot <= 0:
        return None
    lots = max(1, int(POSITION_SIZE / cost_per_lot))
    actual_cost = lots * cost_per_lot

    # 逐根跟踪
    max_roi = -100
    max_roi_bar = 0
    tp_hits = {}
    roi_path = []

    # 期货追踪止损
    tracker = ExitTracker(
        direction=sig['direction'], entry_price=sig['entry_price'],
        pullback_extreme=sig['pullback_extreme'],
        tick_size=tick_size(sig['sym_key']),
        stop_ticks=DEFAULT_STOP_TICKS,
    )
    futures_exit_bar = None
    futures_exit_opt_roi = None

    max_bars = min(MAX_HOLD_BARS, n - idx - 1)

    for j in range(1, max_bars + 1):
        f_bar = futures_df.iloc[idx + j]
        if 'datetime' not in futures_df.columns:
            continue
        f_time = f_bar['datetime']
        prev_bar = futures_df.iloc[idx + j - 1]

        # 在期权K线中找对应时刻
        t_diff = (opt_df['datetime'] - f_time).abs()
        closest = t_diff.idxmin()
        if t_diff.iloc[closest] > pd.Timedelta(minutes=15):
            continue

        opt_bar = opt_df.iloc[closest]
        opt_high = opt_bar['high']
        opt_low = opt_bar['low']
        opt_close = opt_bar['close']

        if opt_close <= 0:
            continue

        # 无论call/put，期权价格上涨=有利
        roi_best = (opt_high - entry_price_opt) / entry_price_opt * 100
        roi_close = (opt_close - entry_price_opt) / entry_price_opt * 100

        roi_path.append((j, round(roi_close, 2)))

        if roi_best > max_roi:
            max_roi = roi_best
            max_roi_bar = j

        # 止盈触发
        for tp in TP_LEVELS:
            if tp not in tp_hits and roi_best >= tp:
                sell_price = entry_price_opt * (1 + tp / 100)
                tp_hits[tp] = {
                    'bar': j,
                    'pnl_yuan': round((sell_price - entry_price_opt) * lots * mult, 0),
                }

        # 期货追踪止损
        if futures_exit_bar is None and not pd.isna(f_bar.get('ema10', np.nan)):
            exit_key = 'S2' if sig['scenario'] in [1, 2] else 'S5.1'
            exit_events, _ = tracker.process_bar(
                close=f_bar['close'], high=f_bar['high'], low=f_bar['low'],
                ema10=f_bar['ema10'], prev_close=prev_bar['close'],
                prev_high=prev_bar['high'], prev_low=prev_bar['low'],
            )
            for ev in exit_events:
                if ev.strategy == exit_key:
                    futures_exit_bar = j
                    futures_exit_opt_roi = roi_close

    last_roi = roi_path[-1][1] if roi_path else -100

    # 判断是否仍在持仓
    still_open = (futures_exit_bar is None) and (max_bars < MAX_HOLD_BARS)

    result = {
        'symbol': sig['symbol'],
        'sym_key': sig['sym_key'],
        'direction': sig['direction'],
        'scenario': sig['scenario'],
        'signal_type': sig['type'],
        'signal_time': str(sig['signal_time']),
        'futures_price': sig['entry_price'],
        'option_contract': opt_info['symbol'],
        'option_strike': opt_info['strike'],
        'option_delta': opt_info['delta'],
        'option_entry_price': round(entry_price_opt, 2),
        'daily_volume': opt_info.get('daily_volume', 0),
        'open_interest': opt_info.get('open_interest', 0),
        'lots': lots,
        'actual_cost': round(actual_cost, 0),
        'multiplier': mult,
        'max_roi': round(max_roi, 1),
        'max_roi_bar': max_roi_bar,
        'max_roi_hours': round(max_roi_bar * 10 / 60, 1),
        'futures_exit_bar': futures_exit_bar,
        'futures_exit_roi': round(futures_exit_opt_roi, 1) if futures_exit_opt_roi is not None else None,
        'still_open': still_open,
        'bars_tracked': len(roi_path),
        'max_bars_available': max_bars,
    }

    # 各止盈策略
    for tp in TP_LEVELS:
        if tp in tp_hits:
            result[f'tp{tp}_hit'] = 1
            result[f'tp{tp}_roi'] = tp
            result[f'tp{tp}_pnl'] = tp_hits[tp]['pnl_yuan']
            result[f'tp{tp}_bar'] = tp_hits[tp]['bar']
        else:
            result[f'tp{tp}_hit'] = 0
            exit_roi = futures_exit_opt_roi if futures_exit_opt_roi is not None else last_roi
            result[f'tp{tp}_roi'] = round(exit_roi, 1)
            pnl = exit_roi / 100 * entry_price_opt * lots * mult
            result[f'tp{tp}_pnl'] = round(pnl, 0)
            result[f'tp{tp}_bar'] = futures_exit_bar if futures_exit_bar else max_bars

    # 跟期货止损
    if futures_exit_opt_roi is not None:
        result['futures_stop_roi'] = round(futures_exit_opt_roi, 1)
        result['futures_stop_pnl'] = round(futures_exit_opt_roi / 100 * entry_price_opt * lots * mult, 0)
    else:
        result['futures_stop_roi'] = round(last_roi, 1)
        result['futures_stop_pnl'] = round(last_roi / 100 * entry_price_opt * lots * mult, 0)

    return result


# ============================================================
#  报告
# ============================================================

def build_report(results):
    from chart_engine import render_chart, get_chart_js

    # 分离：已出场 vs 仍在持仓
    closed = [r for r in results if not r.get('still_open', False)]
    still_open = [r for r in results if r.get('still_open', False)]

    rpt = Report('最近5个交易日真实期权回测')
    rpt.add_text(f'每笔固定开仓 {POSITION_SIZE:,} 元 | 真实期权K线 | Delta 0.05~0.30 选成交量最高')

    if still_open:
        rpt.add_text(
            f'排除仍在持仓的 {len(still_open)} 笔: '
            + ', '.join(f"{r['symbol']}({r['signal_time'][:10]})" for r in still_open),
            color='#8b949e'
        )

    if not closed:
        rpt.add_text('没有已出场的交易', color='#f85149')
        return rpt

    # 注入K线JS
    rpt.add_html(f'<script>{get_chart_js()}</script>')

    # Part 1: 信号列表（只统计已出场的）
    rpt.add_section('Part 1: 信号明细（已出场）')

    headers = ['时间', '品种', '方向', '场景', '期货价',
               '期权合约', 'Δ', '行权价', '期权入场价', '手数', '投入(元)',
               'MFE%', '峰值(h)', '跟期货止损ROI%', '状态']
    rows = []
    for r in closed:
        opt_name = r['option_contract'].split('.')[-1]
        rows.append([
            r['signal_time'][:16], r['symbol'],
            '多' if r['direction'] == 'long' else '空',
            f"S{r['scenario']}", r['futures_price'],
            opt_name, r['option_delta'], r['option_strike'],
            r['option_entry_price'], r['lots'],
            f"{r['actual_cost']:,.0f}",
            f"{r['max_roi']}%", f"{r['max_roi_hours']}h",
            f"{r['futures_stop_roi']}%",
            '期货已止损' if r['futures_exit_bar'] else f"观察{r['bars_tracked']}根",
        ])
    rpt.add_table(headers, rows, highlight_pnl_cols=[11, 13])

    # Part 2: 各止盈策略汇总
    rpt.add_section('Part 2: 各止盈策略汇总（元）')

    headers2 = ['策略', '触发次数', '平均ROI%', '总盈亏(元)']
    rows2 = []

    fut_pnls = [r['futures_stop_pnl'] for r in closed]
    fut_rois = [r['futures_stop_roi'] for r in closed]
    rows2.append([
        '跟期货止损', '-',
        f"{np.mean(fut_rois):.1f}%",
        f"{sum(fut_pnls):,.0f}",
    ])

    for tp in TP_LEVELS:
        hits = sum(r[f'tp{tp}_hit'] for r in closed)
        rois = [r[f'tp{tp}_roi'] for r in closed]
        pnls = [r[f'tp{tp}_pnl'] for r in closed]
        rows2.append([
            f'止盈+{tp}%',
            f"{hits}/{len(closed)}",
            f"{np.mean(rois):.1f}%",
            f"{sum(pnls):,.0f}",
        ])

    rpt.add_table(headers2, rows2, highlight_pnl_cols=[2, 3])

    # Part 3: 逐笔盈亏（元）
    rpt.add_section('Part 3: 逐笔盈亏（元）')

    headers3 = ['时间', '品种', 'S', 'Δ', '投入']
    headers3 += ['跟期货'] + [f'+{tp}%' for tp in TP_LEVELS]
    rows3 = []

    for r in closed:
        row = [
            r['signal_time'][:16], r['symbol'],
            r['scenario'], r['option_delta'],
            f"{r['actual_cost']:,.0f}",
            f"{r['futures_stop_pnl']:,.0f}",
        ]
        for tp in TP_LEVELS:
            row.append(f"{r[f'tp{tp}_pnl']:,.0f}")
        rows3.append(row)

    total_row = ['合计', '', '', '', f"{sum(r['actual_cost'] for r in closed):,.0f}"]
    total_row.append(f"{sum(fut_pnls):,.0f}")
    for tp in TP_LEVELS:
        total_row.append(f"{sum(r[f'tp{tp}_pnl'] for r in closed):,.0f}")
    rows3.append(total_row)

    rpt.add_table(headers3, rows3, highlight_pnl_cols=list(range(5, 5 + 1 + len(TP_LEVELS))))

    # Part 4: 每笔K线图（期货+期权双图）
    rpt.add_section('Part 4: 逐笔K线验证')

    for r in closed:
        sig = r.get('_sig')
        futures_df = r.get('_futures_df')
        opt_df = r.get('_opt_df')

        if sig is None or futures_df is None:
            continue

        idx = sig['idx']
        direction = sig['direction']

        # 期货K线图
        exits = []
        if r['futures_exit_bar']:
            exit_idx = idx + r['futures_exit_bar']
            if exit_idx < len(futures_df):
                exits.append({
                    'name': 'S2' if r['scenario'] in [1, 2] else 'S5.1',
                    'idx': exit_idx,
                    'price': futures_df.iloc[exit_idx]['close'],
                })

        futures_chart = render_chart(
            df=futures_df, entry_idx=idx, direction=direction,
            exits=exits,
            before_bars=20, after_bars=60,
            ema_cols=['ema10', 'ema20', 'ema120'],
            title=f"期货: {r['symbol']} {direction} 场景{r['scenario']} {r['signal_time'][:16]}",
            extra_info={
                'F入场': f"{r['futures_price']}",
                '期权': r['option_contract'].split('.')[-1],
                'Δ': f"{r['option_delta']}",
                '期权MFE': f"{r['max_roi']}%",
            },
            width=900, height=320,
        )
        rpt.add_html(futures_chart)

        # 期权K线图
        if opt_df is not None and len(opt_df) > 0:
            # 在期权K线中找入场位置
            entry_time = sig['signal_time']
            t_diff = (opt_df['datetime'] - entry_time).abs()
            opt_entry_idx = t_diff.idxmin()

            opt_exits = []
            # MFE峰值标记
            if r['max_roi_bar'] > 0:
                peak_time_approx = entry_time + pd.Timedelta(minutes=r['max_roi_bar'] * 10)
                t_diff_peak = (opt_df['datetime'] - peak_time_approx).abs()
                peak_idx = t_diff_peak.idxmin()
                if t_diff_peak.iloc[peak_idx] < pd.Timedelta(minutes=30):
                    opt_exits.append({
                        'name': f'MFE+{r["max_roi"]}%',
                        'idx': peak_idx,
                        'price': opt_df.iloc[peak_idx]['high'],
                        'color': '#FF9800',
                    })

            opt_chart = render_chart(
                df=opt_df, entry_idx=opt_entry_idx, direction='long',  # 期权买方always long
                exits=opt_exits,
                before_bars=10, after_bars=60,
                ema_cols=[],
                title=f"期权: {r['option_contract'].split('.')[-1]} 入场价={r['option_entry_price']}",
                extra_info={
                    '入场价': f"{r['option_entry_price']}",
                    '手数': f"{r['lots']}",
                    '投入': f"{r['actual_cost']:,.0f}元",
                    'MFE': f"+{r['max_roi']}%",
                },
                width=900, height=320,
            )
            rpt.add_html(opt_chart)

        rpt.add_html('<hr style="border-color:#21262d; margin:20px 0;">')

    # Part 5: 结论
    rpt.add_section('Part 5: 结论')

    best_label = '跟期货止损'
    best_total = sum(fut_pnls)
    for tp in TP_LEVELS:
        total = sum(r[f'tp{tp}_pnl'] for r in closed)
        if total > best_total:
            best_total = total
            best_label = f'止盈+{tp}%'

    total_invested = sum(r['actual_cost'] for r in closed)
    roi_total = best_total / total_invested * 100 if total_invested > 0 else 0

    rpt.add_text(
        f'{len(closed)}笔已出场交易（排除{len(still_open)}笔持仓中），'
        f'总投入 {total_invested:,.0f}元<br>'
        f'最优策略: <b>{best_label}</b>，总盈亏 <b>{best_total:,.0f}元</b> '
        f'(总投入收益率 {roi_total:.1f}%)',
        color='#3fb950' if best_total > 0 else '#f85149'
    )

    return rpt


# ============================================================
#  主流程
# ============================================================

def main():
    from tqsdk import TqApi, TqAuth
    import config as cfg

    print("=" * 70)
    print("最近5个交易日真实期权回测")
    print(f"每笔固定开仓: {POSITION_SIZE:,}元")
    print("=" * 70)

    # Step 1: 找信号（用缓存数据，不需要连天勤）
    signals, all_data = find_recent_signals()

    if not signals:
        print("最近5天没有信号！")
        return

    # Step 2: 连天勤拉期权数据
    print("\n" + "=" * 70)
    print("Step 2: 连接天勤，拉取真实期权数据")
    print("=" * 70)

    api = TqApi(auth=TqAuth(cfg.TQ_ACCOUNT, cfg.TQ_PASSWORD))

    try:
        # 缓存
        option_chains_cache = {}  # 期权链（按品种+方向）
        kline_cache = {}          # 期权K线（按合约号）
        results = []

        for i, sig in enumerate(signals):
            sym_key = sig['sym_key']
            cache_key = f"{sym_key}_{sig['direction']}"
            print(f"\n[{i+1}/{len(signals)}] {sig['signal_time']} {sig['symbol']} "
                  f"{sig['direction']} 场景{sig['scenario']}")

            # 查期权链（按品种+方向缓存）
            if cache_key not in option_chains_cache:
                print(f"  查询{sig['symbol']}期权链（{sig['direction']}）...")
                candidates = select_option_contracts(
                    api, sym_key, sig['direction'], sig['entry_price'], sig['hv']
                )
                option_chains_cache[cache_key] = candidates
                print(f"  候选合约: {len(candidates)}个 "
                      f"(Delta {candidates[0]['delta']:.2f}~{candidates[-1]['delta']:.2f})"
                      if candidates else "  无候选合约")
            else:
                candidates = option_chains_cache[cache_key]

            if not candidates:
                print(f"  跳过（无期权）")
                continue

            # 从低Delta开始找有真实数据的合约
            print(f"  从低Delta开始逐个检查流动性...")
            best = select_best_with_real_data(api, candidates, sig['signal_time'], kline_cache)

            if best is None:
                print(f"  无可用合约，跳过")
                continue

            print(f"  选中: {best['symbol']} Δ={best['delta']} K={best['strike']} "
                  f"入场价={best['entry_price_real']} 日成交={best.get('daily_volume', '?')}")

            # 模拟交易
            futures_df = all_data[sig['df_key']]
            mult = multiplier(sym_key)
            trade_result = simulate_real_option(
                futures_df, best['opt_df'], sig, best, mult
            )

            if trade_result is not None:
                # 保存df引用用于画K线
                trade_result['_futures_df'] = futures_df
                trade_result['_opt_df'] = best['opt_df']
                trade_result['_sig'] = sig
                results.append(trade_result)
                status = '仍在持仓' if trade_result['still_open'] else '已出场'
                print(f"  ✓ MFE={trade_result['max_roi']}% "
                      f"跟期货止损={trade_result['futures_stop_roi']}% "
                      f"投入={trade_result['actual_cost']:,.0f}元 [{status}]")
            else:
                print(f"  时间不匹配，跳过")

            time.sleep(0.3)

    finally:
        api.close()

    print(f"\n有效交易: {len(results)}笔")

    if not results:
        print("没有有效交易结果！")
        return

    # Step 3: 报告
    print("\n" + "=" * 70)
    print("Step 3: 生成报告")
    print("=" * 70)

    rpt = build_report(results)
    rpt.save('output/real_option_5d.html')
    print("\n报告已保存: output/real_option_5d.html")

    # 控制台摘要
    print("\n" + "=" * 70)
    print("收益汇总")
    print("=" * 70)
    total_cost = sum(r['actual_cost'] for r in results)
    print(f"总投入: {total_cost:,.0f}元 ({len(results)}笔×约{POSITION_SIZE:,}元)")
    print(f"{'策略':20s} {'总盈亏':>10s} {'收益率':>8s}")
    print("-" * 45)

    fut_total = sum(r['futures_stop_pnl'] for r in results)
    print(f"{'跟期货止损':20s} {fut_total:>10,.0f}元 {fut_total/total_cost*100:>7.1f}%")
    for tp in TP_LEVELS:
        total = sum(r[f'tp{tp}_pnl'] for r in results)
        print(f"{'止盈+' + str(tp) + '%':20s} {total:>10,.0f}元 {total/total_cost*100:>7.1f}%")


if __name__ == '__main__':
    main()
