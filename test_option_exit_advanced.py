# -*- coding: utf-8 -*-
"""
期权出场策略优化研究
====================
在 test_option_tp.py 基础上扩展：
1. ER门槛参数化（0.4/0.45/0.5）— 先放大样本量优化出场，再收紧
2. 追踪止盈 — 期权ROI达触发值后，回撤到峰值×保留比例就出场
3. 分批出场 — 半仓低止盈锁利 + 半仓博更高收益
4. 场景化时间止损 — 不同场景不同最大持仓

输出: output/option_exit_advanced.html
"""

import sys
import io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from data_loader import load_all, sym_name, tick_size, HIGH_VOL
from signal_core import (SignalDetector, ExitTracker, SYMBOL_CONFIGS,
                         DEFAULT_STOP_TICKS, SCENARIO_NAMES)
from stats_utils import calc_ev
from volatility import add_hv
from option_pricing import black76_price, black76_greeks, find_strike_by_delta, R_DEFAULT
from report_engine import Report


# ============================================================
#  配置
# ============================================================

ER_THRESHOLDS = [0.4, 0.45, 0.5]

DELTA_LEVELS = [0.10, 0.20, 0.30]
T_DAYS_DEFAULT = 15
SLIPPAGE_PCT = 5.0
BARS_PER_DAY = 57
TRADING_DAYS_YEAR = 245
MAX_HOLD_BARS = 144  # 24小时

# 固定止盈档位
TP_LEVELS = [30, 50, 80, 100, 150, 200, 300]

# 时间止损（K线根数）
TIME_STOPS = [6, 12, 18, 24, 36, 48]  # 1h/2h/3h/4h/6h/8h

# 追踪止盈配置
TRAIL_TRIGGERS = [50, 80, 100, 150]       # 触发门槛 ROI%
TRAIL_PCTS = [0.50, 0.60, 0.70, 0.80]    # 保留比例（0.6 = 跌到峰值60%出场）

# 分批出场配置
PARTIAL_CONFIGS = [
    {'leg1_tp': 50,  'leg2_tp': 150, 'split': 0.5},
    {'leg1_tp': 50,  'leg2_tp': 200, 'split': 0.5},
    {'leg1_tp': 80,  'leg2_tp': 200, 'split': 0.5},
    {'leg1_tp': 80,  'leg2_tp': 300, 'split': 0.5},
    {'leg1_tp': 100, 'leg2_tp': 300, 'split': 0.5},
]

# 场景名（适配不同ER门槛）
def scenario_name(sc, threshold):
    if sc == 1:
        return f'场景1: A+ER≥{threshold}+偏离≥1ATR'
    elif sc == 2:
        return '场景2: C+偏离≥2ATR'
    elif sc == 3:
        return f'场景3: B+ER≥{threshold}+偏离0.1~0.3ATR'
    return f'场景{sc}'


# ============================================================
#  ER门槛参数化
# ============================================================

def classify_with_threshold(sig_type, er20, deviation_atr, threshold):
    """classify_scenario 但 ER 门槛可调"""
    if er20 >= 0.7:
        return None
    if sig_type == 'A' and er20 >= threshold and deviation_atr >= 1.0:
        return 1
    if sig_type == 'C' and deviation_atr >= 2.0:
        return 2
    if sig_type == 'B' and er20 >= threshold and 0.1 <= deviation_atr < 0.3:
        return 3
    return None


# ============================================================
#  期权MFE路径计算（从 test_option_tp.py 复制，避免import冲突）
# ============================================================

def compute_option_mfe_path(df, sig, delta, t_days):
    """计算单笔信号的期权MFE路径：逐根K线的期权ROI%"""
    idx = sig['idx']
    n = len(df)
    entry_price = sig['entry_price']
    is_long = sig['direction'] == 'long'
    cp = 'call' if is_long else 'put'
    sigma = sig['hv']
    T_entry = t_days / TRADING_DAYS_YEAR

    K = find_strike_by_delta(entry_price, T_entry, R_DEFAULT, sigma, delta, cp)
    entry_premium = black76_price(entry_price, K, T_entry, R_DEFAULT, sigma, cp)

    if entry_premium <= 0:
        return None

    entry_cost = entry_premium * (1 + SLIPPAGE_PCT / 100)

    # 入场Greeks
    entry_greeks = black76_greeks(entry_price, K, T_entry, R_DEFAULT, sigma, cp)
    entry_delta = abs(entry_greeks['delta'])
    entry_gamma = entry_greeks['gamma']
    entry_theta = entry_greeks['theta']
    entry_gt_ratio = (entry_gamma / abs(entry_theta)) if entry_theta != 0 else 999

    max_roi = -100
    max_roi_bar = 0
    roi_path = []
    price_path = []  # (bar, F_high, F_low, F_close, T_now)
    greeks_path = []  # (bar, delta, gamma, theta, gamma_theta_ratio, intrinsic_pct)
    tp_hits = {}

    tracker = ExitTracker(
        direction=sig['direction'], entry_price=entry_price,
        pullback_extreme=sig['pullback_extreme'],
        tick_size=tick_size(sig['sym_key']),
        stop_ticks=DEFAULT_STOP_TICKS,
    )
    futures_exit_bar = None
    futures_exit_roi = None

    # 信号K线50%止损
    sig_high = sig.get('sig_high', entry_price)
    sig_low = sig.get('sig_low', entry_price)
    sig_range = sig_high - sig_low
    candle_stops = {}  # pct -> stop_price
    for pct in [0.3, 0.4, 0.5, 0.6]:
        if is_long:
            candle_stops[pct] = sig_high - sig_range * pct
        else:
            candle_stops[pct] = sig_low + sig_range * pct
    candle_exit = {}  # pct -> {'bar': , 'roi': }

    max_bars = min(MAX_HOLD_BARS, n - idx - 1)

    for j in range(1, max_bars + 1):
        bar = df.iloc[idx + j]
        prev_bar = df.iloc[idx + j - 1]

        if pd.isna(bar['ema10']):
            continue

        T_now = max(T_entry - j / (BARS_PER_DAY * TRADING_DAYS_YEAR), 0.0001)
        price_path.append((j, bar['high'], bar['low'], bar['close'], T_now))

        if is_long:
            best_price_in_bar = bar['high']
        else:
            best_price_in_bar = bar['low']

        opt_best = black76_price(best_price_in_bar, K, T_now, R_DEFAULT, sigma, cp)
        opt_best_revenue = opt_best * (1 - SLIPPAGE_PCT / 100)
        roi_best = (opt_best_revenue - entry_cost) / entry_cost * 100

        opt_close = black76_price(bar['close'], K, T_now, R_DEFAULT, sigma, cp)
        opt_close_revenue = opt_close * (1 - SLIPPAGE_PCT / 100)
        roi_close = (opt_close_revenue - entry_cost) / entry_cost * 100

        roi_path.append((j, round(roi_close, 2)))

        # Greeks at close
        g = black76_greeks(bar['close'], K, T_now, R_DEFAULT, sigma, cp)
        cur_delta = abs(g['delta'])
        cur_gamma = g['gamma']
        cur_theta = g['theta']
        cur_gt = (cur_gamma / abs(cur_theta)) if cur_theta != 0 else 999
        # 内在价值占比
        if cp == 'call':
            intrinsic = max(bar['close'] - K, 0)
        else:
            intrinsic = max(K - bar['close'], 0)
        opt_total = g['price'] if g['price'] > 0 else 0.0001
        intrinsic_pct = intrinsic / opt_total * 100
        greeks_path.append((j, cur_delta, cur_gamma, cur_theta, cur_gt, round(intrinsic_pct, 1)))

        if roi_best > max_roi:
            max_roi = roi_best
            max_roi_bar = j

        for tp in TP_LEVELS:
            if tp not in tp_hits and roi_best >= tp:
                tp_hits[tp] = {'bar': j, 'roi': round(roi_best, 1)}

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
                    F_exit = ev.exit_price
                    opt_at_exit = black76_price(F_exit, K, T_now, R_DEFAULT, sigma, cp)
                    opt_exit_rev = opt_at_exit * (1 - SLIPPAGE_PCT / 100)
                    futures_exit_roi = round((opt_exit_rev - entry_cost) / entry_cost * 100, 2)

        # 信号K线比例止损检查
        for pct, stop_price in candle_stops.items():
            if pct in candle_exit:
                continue
            hit = False
            if is_long and bar['low'] <= stop_price:
                hit = True
                F_stop = stop_price
            elif not is_long and bar['high'] >= stop_price:
                hit = True
                F_stop = stop_price
            if hit:
                opt_stop = black76_price(F_stop, K, T_now, R_DEFAULT, sigma, cp)
                opt_stop_rev = opt_stop * (1 - SLIPPAGE_PCT / 100)
                candle_exit[pct] = {
                    'bar': j,
                    'roi': round((opt_stop_rev - entry_cost) / entry_cost * 100, 2)
                }

    return {
        'scenario': sig['scenario'],
        'symbol': sig['symbol'],
        'sym_key': sig['sym_key'],
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
        'price_path': price_path,
        'candle_exit': candle_exit,
        'greeks_path': greeks_path,
        'entry_delta': entry_delta,
        'entry_gamma': entry_gamma,
        'entry_theta': entry_theta,
        'entry_gt_ratio': entry_gt_ratio,
        'K': K,
        'cp': cp,
        'sigma': sigma,
    }


# ============================================================
#  信号收集（参数化ER门槛）
# ============================================================

def collect_signals(all_data, er_threshold=0.5, last_days=120):
    """收集信号，ER门槛可调"""
    records = []

    for sym_key, df in sorted(all_data.items()):
        n = len(df)
        signal_start = max(130, n - last_days * BARS_PER_DAY)
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

            scenario = classify_with_threshold(result.signal_type, er20, deviation_atr, er_threshold)
            if scenario is None:
                continue

            # 场景1过滤：er5_delta_6 <= -0.41 不开仓
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
                'sig_high': row['high'],
                'sig_low': row['low'],
                'hv': hv, 'atr': atr,
            })

    return records


# ============================================================
#  追踪止盈
# ============================================================

def simulate_trailing_stop(roi_path, trigger_pct, trail_pct, time_stop_bar=None):
    """
    追踪止盈策略。

    roi_path: [(bar_offset, roi%)] — 每根K线的收盘ROI
    trigger_pct: 触发追踪的ROI% (e.g. 80 = 涨到+80%后开始追踪)
    trail_pct: 保留比例 (e.g. 0.6 = 跌到峰值的60%就出场)
    time_stop_bar: 最大持仓K线数，超时按当时价出场（None=无限制）

    返回: (exit_roi%, exit_bar, exit_reason)
    """
    peak_roi = 0
    trailing_active = False
    last_roi = 0
    last_bar = 0

    for bar_offset, roi in roi_path:
        # 时间止损检查
        if time_stop_bar is not None and bar_offset > time_stop_bar:
            return (last_roi, last_bar, 'time_stop')

        last_roi = roi
        last_bar = bar_offset

        if roi > peak_roi:
            peak_roi = roi

        # 激活追踪
        if not trailing_active and peak_roi >= trigger_pct:
            trailing_active = True

        # 追踪触发出场
        if trailing_active and peak_roi > 0:
            trail_floor = peak_roi * trail_pct
            if roi <= trail_floor:
                return (roi, bar_offset, 'trail_stop')

    # 路径结束仍未出场
    return (last_roi, last_bar, 'path_end')


# ============================================================
#  分批出场
# ============================================================

def simulate_partial_exit(roi_path, tp_hits, leg1_tp, leg2_tp, time_stop_bar, split=0.5):
    """
    分批出场策略。

    leg1: split比例仓位在leg1_tp止盈
    leg2: (1-split)比例仓位在leg2_tp止盈
    两腿都有time_stop兜底。

    返回: blended_roi%
    """
    # Leg 1
    hit1 = tp_hits.get(leg1_tp)
    if hit1 and (time_stop_bar is None or hit1['bar'] <= time_stop_bar):
        roi_leg1 = leg1_tp
    else:
        roi_leg1 = _roi_at_bar(roi_path, time_stop_bar)

    # Leg 2
    hit2 = tp_hits.get(leg2_tp)
    if hit2 and (time_stop_bar is None or hit2['bar'] <= time_stop_bar):
        roi_leg2 = leg2_tp
    else:
        roi_leg2 = _roi_at_bar(roi_path, time_stop_bar)

    return split * roi_leg1 + (1 - split) * roi_leg2


def _roi_at_bar(roi_path, target_bar):
    """在roi_path中找到指定bar的ROI，找不到用最后一根"""
    if target_bar is None:
        return roi_path[-1][1] if roi_path else 0
    for bar_offset, roi in roi_path:
        if bar_offset == target_bar:
            return roi
        if bar_offset > target_bar:
            return roi  # 用超过的那根近似
    return roi_path[-1][1] if roi_path else 0


# ============================================================
#  新策略1: 时间递减止盈
# ============================================================

# 配置：(小时, 止盈目标ROI%)
DECAY_TP_CONFIGS = [
    # 名称, [(hour_limit, tp_target), ...]
    ('激进递减', [(1, 300), (2, 200), (4, 150), (6, 100), (8, 50)]),
    ('温和递减', [(2, 200), (4, 150), (6, 100), (8, 80)]),
    ('快速递减', [(1, 200), (2, 150), (3, 100), (4, 80), (6, 50)]),
]

def simulate_decay_tp(roi_path, config_tiers, time_stop_bar=None):
    """
    时间递减止盈：止盈目标随时间下降。
    config_tiers: [(hour_limit, tp_target), ...] 按小时排列
    每根K线检查当前所处时段，用对应的tp_target。
    """
    last_roi = 0
    last_bar = 0
    for bar_offset, roi in roi_path:
        if time_stop_bar is not None and bar_offset > time_stop_bar:
            return (last_roi, last_bar, 'time_stop')
        last_roi = roi
        last_bar = bar_offset

        hours = bar_offset * 10 / 60
        # 找当前时段的止盈目标
        current_tp = None
        for h_limit, tp_target in config_tiers:
            if hours <= h_limit:
                current_tp = tp_target
                break
        if current_tp is None:
            current_tp = config_tiers[-1][1]  # 超过最大时段用最后一档

        if roi >= current_tp:
            return (roi, bar_offset, f'decay_tp_{current_tp}')

    return (last_roi, last_bar, 'path_end')


# ============================================================
#  新策略2: Delta翻倍出场
# ============================================================

DELTA_MULT_LEVELS = [1.5, 2.0, 2.5, 3.0]  # Delta达到入场的X倍时出场

def simulate_delta_exit(greeks_path, roi_path, entry_delta, target_mult, time_stop_bar=None):
    """
    Delta翻倍出场：当前Delta达到入场Delta×目标倍数时出场。
    """
    target_delta = entry_delta * target_mult
    last_roi = 0
    last_bar = 0
    for (bar, cur_delta, _, _, _, _), (_, roi) in zip(greeks_path, roi_path):
        if time_stop_bar is not None and bar > time_stop_bar:
            return (last_roi, last_bar, 'time_stop')
        last_roi = roi
        last_bar = bar

        if cur_delta >= target_delta:
            return (roi, bar, 'delta_exit')

    return (last_roi, last_bar, 'path_end')


# ============================================================
#  新策略3: Gamma/Theta比值出场
# ============================================================

GT_DECAY_LEVELS = [0.30, 0.40, 0.50, 0.60]  # G/T降到入场时的X倍以下出场

def simulate_gt_exit(greeks_path, roi_path, entry_gt_ratio, decay_pct, time_stop_bar=None):
    """
    Gamma/Theta比值出场：性价比降到入场时的decay_pct以下出场。
    """
    if entry_gt_ratio <= 0 or entry_gt_ratio >= 999:
        # 入场时theta=0，无法计算，走完全程
        last = roi_path[-1] if roi_path else (0, 0)
        return (last[1], last[0], 'path_end')

    threshold = entry_gt_ratio * decay_pct
    last_roi = 0
    last_bar = 0
    # 跳过前3根K线（避免开盘波动造成的假信号）
    for (bar, _, _, _, cur_gt, _), (_, roi) in zip(greeks_path, roi_path):
        if time_stop_bar is not None and bar > time_stop_bar:
            return (last_roi, last_bar, 'time_stop')
        last_roi = roi
        last_bar = bar

        if bar >= 3 and cur_gt <= threshold:
            return (roi, bar, 'gt_exit')

    return (last_roi, last_bar, 'path_end')


# ============================================================
#  新策略4: K线结构追踪止盈
# ============================================================

CANDLE_TRAIL_ACTIVATIONS = [30, 50, 80, 100, 150, 200]  # ROI%触发门槛

def simulate_candle_trail(price_path, roi_path, activation_roi,
                          trail_mode, K, sigma, cp, entry_cost,
                          ts_val, is_long):
    """
    K线结构追踪止盈。

    1. 等待期权ROI达到activation_roi
    2. 激活后，每根期货创新高/低的K线更新追踪止损：
       - 'half': 止损 = 该K线high - (high-low)*0.5  (做多) / low + (high-low)*0.5 (做空)
       - 'low5':  止损 = 该K线low - 5跳 (做多) / high + 5跳 (做空)
    3. 期货盘中穿追踪止损 → 计算期权价出场

    返回: (exit_roi%, exit_bar, reason)
    """
    activated = False
    trail_stop = None
    peak_price = -1e18 if is_long else 1e18
    last_roi = 0
    last_bar = 0
    just_updated = False  # 刚设定/更新止盈位的那根K线不判断触碰

    for (bar, F_high, F_low, F_close, T_now), (_, roi) in zip(price_path, roi_path):
        last_roi = roi
        last_bar = bar

        # 步骤1：等ROI达标
        if not activated:
            if roi >= activation_roi:
                activated = True
                # 激活时以当前bar为第一根"新高K线"，设定止盈位
                if is_long:
                    peak_price = F_high
                    if trail_mode == 'half':
                        trail_stop = F_high - (F_high - F_low) * 0.5
                    else:  # low5
                        trail_stop = F_low - 5 * ts_val
                else:
                    peak_price = F_low
                    if trail_mode == 'half':
                        trail_stop = F_low + (F_high - F_low) * 0.5
                    else:
                        trail_stop = F_high + 5 * ts_val
                just_updated = True  # 这根K线不检查触碰
            continue

        # 步骤2：已激活，先检查是否触碰止盈（用上一根K线设定的止盈位）
        if is_long:
            if not just_updated and trail_stop is not None and F_low <= trail_stop:
                # 出场：用止损价计算期权价
                opt_exit = black76_price(trail_stop, K, T_now, R_DEFAULT, sigma, cp)
                opt_rev = opt_exit * (1 - SLIPPAGE_PCT / 100)
                exit_roi = (opt_rev - entry_cost) / entry_cost * 100
                return (round(exit_roi, 2), bar, 'candle_trail')

            # 步骤3：检查是否创新高，更新追踪止盈位
            just_updated = False
            if F_high > peak_price:
                peak_price = F_high
                if trail_mode == 'half':
                    trail_stop = F_high - (F_high - F_low) * 0.5
                else:
                    trail_stop = F_low - 5 * ts_val
                just_updated = True  # 创新高K线本身不触发

        else:
            if not just_updated and trail_stop is not None and F_high >= trail_stop:
                opt_exit = black76_price(trail_stop, K, T_now, R_DEFAULT, sigma, cp)
                opt_rev = opt_exit * (1 - SLIPPAGE_PCT / 100)
                exit_roi = (opt_rev - entry_cost) / entry_cost * 100
                return (round(exit_roi, 2), bar, 'candle_trail')

            just_updated = False
            if F_low < peak_price:
                peak_price = F_low
                if trail_mode == 'half':
                    trail_stop = F_low + (F_high - F_low) * 0.5
                else:
                    trail_stop = F_high + 5 * ts_val
                just_updated = True

    return (last_roi, last_bar, 'path_end')


# ============================================================
#  综合策略模拟
# ============================================================

def simulate_all_strategies(all_results):
    """
    对每笔信号模拟所有出场策略，返回DataFrame。
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

        roi_path = r['roi_path']
        tp_hits = r['tp_hits']

        # === 基线：跟期货止损 ===
        base['futures_stop_roi'] = r['futures_exit_roi'] if r['futures_exit_roi'] is not None else 0
        base['futures_stop_bar'] = r['futures_exit_bar'] if r['futures_exit_bar'] is not None else MAX_HOLD_BARS

        # === 固定止盈 ===
        for tp in TP_LEVELS:
            hit = tp_hits.get(tp)
            if hit:
                base[f'tp{tp}_hit'] = 1
                base[f'tp{tp}_roi'] = tp
                base[f'tp{tp}_bar'] = hit['bar']
            else:
                base[f'tp{tp}_hit'] = 0
                base[f'tp{tp}_roi'] = 0
                base[f'tp{tp}_bar'] = 0

        # === 固定止盈 + 时间止损 ===
        for tp in TP_LEVELS:
            for ts_bar in TIME_STOPS:
                hit = tp_hits.get(tp)
                if hit and hit['bar'] <= ts_bar:
                    roi = tp
                else:
                    roi = _roi_at_bar(roi_path, ts_bar)
                base[f'tp{tp}_ts{ts_bar}_roi'] = roi

        # === 追踪止盈 ===
        for trigger in TRAIL_TRIGGERS:
            for trail_pct in TRAIL_PCTS:
                key = f'trail_{trigger}_{int(trail_pct*100)}'
                exit_roi, exit_bar, reason = simulate_trailing_stop(
                    roi_path, trigger, trail_pct)
                base[f'{key}_roi'] = round(exit_roi, 2)
                base[f'{key}_bar'] = exit_bar
                base[f'{key}_reason'] = reason

        # === 追踪止盈 + 时间止损 ===
        for trigger in TRAIL_TRIGGERS:
            for trail_pct in TRAIL_PCTS:
                for ts_bar in TIME_STOPS:
                    key = f'trail_{trigger}_{int(trail_pct*100)}_ts{ts_bar}'
                    exit_roi, exit_bar, reason = simulate_trailing_stop(
                        roi_path, trigger, trail_pct, time_stop_bar=ts_bar)
                    base[f'{key}_roi'] = round(exit_roi, 2)

        # === 分批出场 ===
        for pc in PARTIAL_CONFIGS:
            for ts_bar in TIME_STOPS:
                key = f'partial_{pc["leg1_tp"]}_{pc["leg2_tp"]}_ts{ts_bar}'
                roi = simulate_partial_exit(
                    roi_path, tp_hits, pc['leg1_tp'], pc['leg2_tp'],
                    ts_bar, pc['split'])
                base[f'{key}_roi'] = round(roi, 2)

        greeks_path = r['greeks_path']
        entry_delta = r['entry_delta']
        entry_gt_ratio = r['entry_gt_ratio']
        base['entry_delta_val'] = entry_delta
        base['entry_gt_ratio'] = entry_gt_ratio

        # === 新策略1: 时间递减止盈 ===
        for cfg_name, cfg_tiers in DECAY_TP_CONFIGS:
            for ts_bar in TIME_STOPS:
                key = f'decay_{cfg_name}_ts{ts_bar}'
                exit_roi, exit_bar, reason = simulate_decay_tp(roi_path, cfg_tiers, ts_bar)
                base[f'{key}_roi'] = round(exit_roi, 2)
            # 无时限
            key_nots = f'decay_{cfg_name}'
            exit_roi, exit_bar, reason = simulate_decay_tp(roi_path, cfg_tiers)
            base[f'{key_nots}_roi'] = round(exit_roi, 2)
            base[f'{key_nots}_reason'] = reason

        # === 新策略2: Delta翻倍出场 ===
        for mult in DELTA_MULT_LEVELS:
            key = f'delta_x{int(mult*10)}'
            exit_roi, exit_bar, reason = simulate_delta_exit(
                greeks_path, roi_path, entry_delta, mult)
            base[f'{key}_roi'] = round(exit_roi, 2)
            base[f'{key}_bar'] = exit_bar
            base[f'{key}_reason'] = reason
            # + 时间止损
            for ts_bar in TIME_STOPS:
                key_ts = f'delta_x{int(mult*10)}_ts{ts_bar}'
                exit_roi_ts, _, _ = simulate_delta_exit(
                    greeks_path, roi_path, entry_delta, mult, ts_bar)
                base[f'{key_ts}_roi'] = round(exit_roi_ts, 2)

        # === 新策略3: Gamma/Theta比值出场 ===
        for decay in GT_DECAY_LEVELS:
            key = f'gt_{int(decay*100)}'
            exit_roi, exit_bar, reason = simulate_gt_exit(
                greeks_path, roi_path, entry_gt_ratio, decay)
            base[f'{key}_roi'] = round(exit_roi, 2)
            base[f'{key}_bar'] = exit_bar
            base[f'{key}_reason'] = reason
            # + 时间止损
            for ts_bar in TIME_STOPS:
                key_ts = f'gt_{int(decay*100)}_ts{ts_bar}'
                exit_roi_ts, _, _ = simulate_gt_exit(
                    greeks_path, roi_path, entry_gt_ratio, decay, ts_bar)
                base[f'{key_ts}_roi'] = round(exit_roi_ts, 2)

        # === 新策略4: 信号K线比例止损 ===
        candle_exit = r['candle_exit']
        CANDLE_STOP_PCTS = [0.3, 0.4, 0.5, 0.6]
        for pct in CANDLE_STOP_PCTS:
            ce = candle_exit.get(pct)
            if ce:
                base[f'candle{int(pct*100)}_stop_roi'] = ce['roi']
                base[f'candle{int(pct*100)}_stop_bar'] = ce['bar']
                base[f'candle{int(pct*100)}_hit'] = 1
            else:
                # 没触发止损 → 看期货追踪止损或持有到底
                base[f'candle{int(pct*100)}_stop_roi'] = roi_path[-1][1] if roi_path else 0
                base[f'candle{int(pct*100)}_stop_bar'] = roi_path[-1][0] if roi_path else 0
                base[f'candle{int(pct*100)}_hit'] = 0

        # 信号K线止损 + 固定止盈组合：到达止盈就走，触发K线止损就止损
        for pct in CANDLE_STOP_PCTS:
            ce = candle_exit.get(pct)
            for tp in TP_LEVELS:
                hit_tp = tp_hits.get(tp)
                key = f'candle{int(pct*100)}_tp{tp}'
                if hit_tp and ce:
                    # 两者都会触发，看谁先
                    if hit_tp['bar'] <= ce['bar']:
                        base[f'{key}_roi'] = tp  # 止盈先到
                    else:
                        base[f'{key}_roi'] = ce['roi']  # 止损先到
                elif hit_tp and not ce:
                    base[f'{key}_roi'] = tp  # 只有止盈触发
                elif ce and not hit_tp:
                    base[f'{key}_roi'] = ce['roi']  # 只有止损触发
                else:
                    base[f'{key}_roi'] = roi_path[-1][1] if roi_path else 0  # 都没触发

        # === 新策略5: K线结构追踪止盈 ===
        price_path = r['price_path']
        opt_K = r['K']
        opt_sigma = r['sigma']
        opt_cp = r['cp']
        is_long = r['direction'] == 'long'
        ts_val = tick_size(r.get('sym_key', ''))  # 需要从r获取

        for act_roi in CANDLE_TRAIL_ACTIVATIONS:
            for mode in ['half', 'low5']:
                key = f'ctrail_{mode}_{act_roi}'
                exit_roi, exit_bar, reason = simulate_candle_trail(
                    price_path, roi_path, act_roi, mode,
                    opt_K, opt_sigma, opt_cp, r['entry_premium'],
                    ts_val, is_long)
                base[f'{key}_roi'] = exit_roi
                base[f'{key}_bar'] = exit_bar
                base[f'{key}_reason'] = reason

        records.append(base)

    return pd.DataFrame(records)


# ============================================================
#  HTML 报告
# ============================================================

def build_report(results_by_threshold):
    rpt = Report('期权出场策略优化研究')
    rpt.add_text('ER门槛参数化 + 追踪止盈 + 分批出场 综合对比')

    # ====== Part 0: ER门槛对比概览 ======
    rpt.add_section('Part 0: ER门槛对比概览',
                    '降低门槛→更多信号→更大样本量')

    headers = ['ER门槛', '场景', 'N', 'Delta', 'MFE中位%', 'MFE P75%',
               '≥50%占比', '≥100%占比', '最优固定止盈EV']
    rows = []
    for th in ER_THRESHOLDS:
        df_tp = results_by_threshold[th]
        for sc in [1, 3]:
            for delta in DELTA_LEVELS:
                sub = df_tp[(df_tp['scenario'] == sc) & (df_tp['delta'] == delta)]
                if len(sub) == 0:
                    continue
                mfe = sub['max_roi']
                # 找最优固定止盈EV
                best_tp_ev = -999
                for tp in TP_LEVELS:
                    pnls = []
                    for _, row in sub.iterrows():
                        if row[f'tp{tp}_hit'] == 1:
                            pnls.append(tp)
                        else:
                            pnls.append(row['futures_stop_roi'])
                    ev = calc_ev(pnls)['EV']
                    if ev > best_tp_ev:
                        best_tp_ev = ev

                rows.append([
                    th, scenario_name(sc, th), len(sub), delta,
                    round(mfe.median(), 1), round(mfe.quantile(0.75), 1),
                    f"{(mfe >= 50).mean() * 100:.0f}%",
                    f"{(mfe >= 100).mean() * 100:.0f}%",
                    best_tp_ev,
                ])
    rpt.add_table(headers, rows, highlight_pnl_cols=[4, 5, 8])

    # 后续Part按最宽门槛（0.4）详细展开
    df_main = results_by_threshold[ER_THRESHOLDS[0]]
    th_main = ER_THRESHOLDS[0]

    # ====== Part 1: MFE分布 ======
    rpt.add_section(f'Part 1: 期权MFE分布 (ER≥{th_main})',
                    '信号后期权价格最高涨到多少？')

    for delta in DELTA_LEVELS:
        h = ['场景', 'N', 'MFE中位%', 'MFE P75%', 'MFE P90%',
             '到峰中位(h)', '≥50%', '≥100%', '≥200%']
        r_rows = []
        for sc in [1, 3]:
            sub = df_main[(df_main['scenario'] == sc) & (df_main['delta'] == delta)]
            if len(sub) == 0:
                continue
            mfe = sub['max_roi']
            r_rows.append([
                scenario_name(sc, th_main), len(sub),
                round(mfe.median(), 1), round(mfe.quantile(0.75), 1),
                round(mfe.quantile(0.90), 1),
                round(sub['max_roi_hours'].median(), 1),
                f"{(mfe >= 50).mean() * 100:.0f}%",
                f"{(mfe >= 100).mean() * 100:.0f}%",
                f"{(mfe >= 200).mean() * 100:.0f}%",
            ])
        rpt.add_section(f'Delta={delta}')
        rpt.add_table(h, r_rows, highlight_pnl_cols=[2, 3, 4])

    # ====== Part 2: 固定止盈触发率 ======
    rpt.add_section(f'Part 2: 止盈触发率 (ER≥{th_main})')

    for delta in DELTA_LEVELS:
        h = ['场景', 'N'] + [f'+{tp}%' for tp in TP_LEVELS]
        r_rows = []
        for sc in [1, 3]:
            sub = df_main[(df_main['scenario'] == sc) & (df_main['delta'] == delta)]
            if len(sub) == 0:
                continue
            row = [scenario_name(sc, th_main), len(sub)]
            for tp in TP_LEVELS:
                hit_rate = sub[f'tp{tp}_hit'].mean() * 100
                row.append(f'{hit_rate:.0f}%')
            r_rows.append(row)
        rpt.add_section(f'Delta={delta}')
        rpt.add_table(h, r_rows)

    # ====== Part 3: 固定止盈+时间止损（沿用原逻辑） ======
    rpt.add_section(f'Part 3: 固定止盈+时间止损 (ER≥{th_main})',
                    '到达止盈就走；超过N小时没到按市价平仓')

    for delta in DELTA_LEVELS:
        for sc in [1, 3]:
            sub = df_main[(df_main['scenario'] == sc) & (df_main['delta'] == delta)]
            if len(sub) == 0:
                continue

            h = ['止盈%\\时间止损', 'N'] + [f'{ts*10//60}h' for ts in TIME_STOPS] + ['无时限']
            r_rows = []

            for tp in TP_LEVELS:
                row = [f'+{tp}%', len(sub)]
                for ts_bar in TIME_STOPS:
                    col = f'tp{tp}_ts{ts_bar}_roi'
                    pnls = sub[col].tolist()
                    ev = calc_ev(pnls)
                    row.append(f"{ev['EV']}")
                # 无时限
                pnls_inf = []
                for _, r in sub.iterrows():
                    if r[f'tp{tp}_hit'] == 1:
                        pnls_inf.append(tp)
                    else:
                        pnls_inf.append(r['futures_stop_roi'])
                ev_inf = calc_ev(pnls_inf)
                row.append(f"{ev_inf['EV']}")
                r_rows.append(row)

            # 基线
            base_ev = calc_ev(sub['futures_stop_roi'].tolist())
            r_rows.append(['跟期货止损', len(sub)] + [f"{base_ev['EV']}"] * (len(TIME_STOPS) + 1))

            rpt.add_section(f'{scenario_name(sc, th_main)} | Δ={delta}')
            rpt.add_table(h, r_rows)

    # ====== Part 4: 追踪止盈（核心新增） ======
    rpt.add_section(f'Part 4: 追踪止盈 (ER≥{th_main})',
                    '期权ROI达到触发值后，回撤到峰值×保留比例就出场')

    for delta in DELTA_LEVELS:
        for sc in [1, 3]:
            sub = df_main[(df_main['scenario'] == sc) & (df_main['delta'] == delta)]
            if len(sub) == 0:
                continue

            h = ['触发%\\保留比例', 'N'] + [f'{int(p*100)}%' for p in TRAIL_PCTS]
            r_rows = []

            for trigger in TRAIL_TRIGGERS:
                row = [f'+{trigger}%触发', len(sub)]
                for trail_pct in TRAIL_PCTS:
                    key = f'trail_{trigger}_{int(trail_pct*100)}'
                    pnls = sub[f'{key}_roi'].tolist()
                    ev = calc_ev(pnls)
                    # 激活率
                    activated = (sub[f'{key}_reason'] != 'path_end').sum()
                    act_rate = activated / len(sub) * 100
                    row.append(f"EV={ev['EV']} ({act_rate:.0f}%触发)")
                r_rows.append(row)

            rpt.add_section(f'{scenario_name(sc, th_main)} | Δ={delta}')
            rpt.add_table(h, r_rows)

    # ====== Part 5: 追踪止盈+时间止损 ======
    rpt.add_section(f'Part 5: 追踪止盈+时间止损 (ER≥{th_main})',
                    '追踪止盈加时间兜底')

    for delta in DELTA_LEVELS:
        for sc in [1, 3]:
            sub = df_main[(df_main['scenario'] == sc) & (df_main['delta'] == delta)]
            if len(sub) == 0:
                continue

            # 只展示前几名追踪配置 × 时间止损
            best_trail_configs = []
            for trigger in TRAIL_TRIGGERS:
                for trail_pct in TRAIL_PCTS:
                    key = f'trail_{trigger}_{int(trail_pct*100)}'
                    pnls = sub[f'{key}_roi'].tolist()
                    ev = calc_ev(pnls)['EV']
                    best_trail_configs.append((trigger, trail_pct, ev))
            best_trail_configs.sort(key=lambda x: x[2], reverse=True)
            top3 = best_trail_configs[:3]

            h = ['追踪配置', 'N'] + [f'{ts*10//60}h' for ts in TIME_STOPS] + ['无时限']
            r_rows = []

            for trigger, trail_pct, base_ev in top3:
                row = [f'+{trigger}%触发/{int(trail_pct*100)}%保留', len(sub)]
                for ts_bar in TIME_STOPS:
                    key = f'trail_{trigger}_{int(trail_pct*100)}_ts{ts_bar}'
                    pnls = sub[f'{key}_roi'].tolist()
                    ev = calc_ev(pnls)
                    row.append(f"{ev['EV']}")
                # 无时限
                key_nots = f'trail_{trigger}_{int(trail_pct*100)}'
                ev_nots = calc_ev(sub[f'{key_nots}_roi'].tolist())
                row.append(f"{ev_nots['EV']}")
                r_rows.append(row)

            rpt.add_section(f'{scenario_name(sc, th_main)} | Δ={delta}')
            rpt.add_table(h, r_rows)

    # ====== Part 6: 分批出场 ======
    rpt.add_section(f'Part 6: 分批出场 (ER≥{th_main})',
                    '半仓低止盈锁利 + 半仓留高止盈博更大收益')

    for delta in DELTA_LEVELS:
        for sc in [1, 3]:
            sub = df_main[(df_main['scenario'] == sc) & (df_main['delta'] == delta)]
            if len(sub) == 0:
                continue

            h = ['分批配置', 'N'] + [f'{ts*10//60}h' for ts in TIME_STOPS]
            r_rows = []

            for pc in PARTIAL_CONFIGS:
                label = f'半仓+{pc["leg1_tp"]}% + 半仓+{pc["leg2_tp"]}%'
                row = [label, len(sub)]
                for ts_bar in TIME_STOPS:
                    key = f'partial_{pc["leg1_tp"]}_{pc["leg2_tp"]}_ts{ts_bar}'
                    pnls = sub[f'{key}_roi'].tolist()
                    ev = calc_ev(pnls)
                    row.append(f"{ev['EV']}")
                r_rows.append(row)

            rpt.add_section(f'{scenario_name(sc, th_main)} | Δ={delta}')
            rpt.add_table(h, r_rows)

    # ====== Part 6b: 时间递减止盈（新） ======
    rpt.add_section(f'Part 6b: 时间递减止盈 (ER≥{th_main})',
                    '止盈目标随时间递减：越等越降低期望，符合最优停止理论')

    for delta in DELTA_LEVELS:
        for sc in [1, 3]:
            sub = df_main[(df_main['scenario'] == sc) & (df_main['delta'] == delta)]
            if len(sub) == 0:
                continue

            h = ['递减配置', 'N'] + [f'{ts*10//60}h' for ts in TIME_STOPS] + ['无时限']
            r_rows = []
            for cfg_name, cfg_tiers in DECAY_TP_CONFIGS:
                desc = '/'.join([f'{t[0]}h→+{t[1]}%' for t in cfg_tiers])
                row = [f'{cfg_name}', len(sub)]
                for ts_bar in TIME_STOPS:
                    key = f'decay_{cfg_name}_ts{ts_bar}'
                    pnls = sub[f'{key}_roi'].tolist()
                    ev = calc_ev(pnls)
                    row.append(f"{ev['EV']}")
                # 无时限
                key_nots = f'decay_{cfg_name}'
                ev_nots = calc_ev(sub[f'{key_nots}_roi'].tolist())
                row.append(f"{ev_nots['EV']}")
                r_rows.append(row)

            # 显示配置详情
            rpt.add_section(f'{scenario_name(sc, th_main)} | Δ={delta}')
            for cfg_name, cfg_tiers in DECAY_TP_CONFIGS:
                desc = ' → '.join([f'{t[0]}h内+{t[1]}%' for t in cfg_tiers])
                rpt.add_text(f'{cfg_name}: {desc}', color='#8b949e')
            rpt.add_table(h, r_rows)

    # ====== Part 6c: Delta翻倍出场（新） ======
    rpt.add_section(f'Part 6c: Delta翻倍出场 (ER≥{th_main})',
                    '入场Delta×N倍时出场：从OTM加速区吃完Gamma收益后离场')

    for delta in DELTA_LEVELS:
        for sc in [1, 3]:
            sub = df_main[(df_main['scenario'] == sc) & (df_main['delta'] == delta)]
            if len(sub) == 0:
                continue

            avg_entry_delta = sub['entry_delta_val'].mean()

            h = ['Delta倍数', 'N', '目标Delta', '触发率'] + [f'{ts*10//60}h' for ts in TIME_STOPS] + ['无时限']
            r_rows = []
            for mult in DELTA_MULT_LEVELS:
                key = f'delta_x{int(mult*10)}'
                pnls = sub[f'{key}_roi'].tolist()
                ev_nots = calc_ev(pnls)
                triggered = (sub[f'{key}_reason'] == 'delta_exit').sum()
                trig_rate = triggered / len(sub) * 100

                row = [f'{mult}x', len(sub),
                       f'{avg_entry_delta*mult:.2f}',
                       f'{trig_rate:.0f}%']
                for ts_bar in TIME_STOPS:
                    key_ts = f'delta_x{int(mult*10)}_ts{ts_bar}'
                    ev = calc_ev(sub[f'{key_ts}_roi'].tolist())
                    row.append(f"{ev['EV']}")
                row.append(f"{ev_nots['EV']}")
                r_rows.append(row)

            rpt.add_section(f'{scenario_name(sc, th_main)} | Δ={delta} (入场Δ≈{avg_entry_delta:.3f})')
            rpt.add_table(h, r_rows)

    # ====== Part 6d: Gamma/Theta比值出场（新） ======
    rpt.add_section(f'Part 6d: Gamma/Theta比值出场 (ER≥{th_main})',
                    'G/T比=性价比指标，降到入场时X%以下出场')

    for delta in DELTA_LEVELS:
        for sc in [1, 3]:
            sub = df_main[(df_main['scenario'] == sc) & (df_main['delta'] == delta)]
            if len(sub) == 0:
                continue

            avg_gt = sub['entry_gt_ratio'].median()

            h = ['G/T衰减阈值', 'N', '触发率'] + [f'{ts*10//60}h' for ts in TIME_STOPS] + ['无时限']
            r_rows = []
            for decay in GT_DECAY_LEVELS:
                key = f'gt_{int(decay*100)}'
                pnls = sub[f'{key}_roi'].tolist()
                ev_nots = calc_ev(pnls)
                triggered = (sub[f'{key}_reason'] == 'gt_exit').sum()
                trig_rate = triggered / len(sub) * 100

                row = [f'降至{int(decay*100)}%', len(sub), f'{trig_rate:.0f}%']
                for ts_bar in TIME_STOPS:
                    key_ts = f'gt_{int(decay*100)}_ts{ts_bar}'
                    ev = calc_ev(sub[f'{key_ts}_roi'].tolist())
                    row.append(f"{ev['EV']}")
                row.append(f"{ev_nots['EV']}")
                r_rows.append(row)

            rpt.add_section(f'{scenario_name(sc, th_main)} | Δ={delta} (入场G/T中位={avg_gt:.1f})')
            rpt.add_table(h, r_rows)

    # ====== Part 6e: 信号K线比例止损（新） ======
    rpt.add_section(f'Part 6e: 信号K线比例止损 (ER≥{th_main})',
                    '创新高K线的整体高度×X%作为止损，配合固定止盈')

    CANDLE_STOP_PCTS = [0.3, 0.4, 0.5, 0.6]
    for delta in DELTA_LEVELS:
        for sc in [1, 3]:
            sub = df_main[(df_main['scenario'] == sc) & (df_main['delta'] == delta)]
            if len(sub) == 0:
                continue

            # 纯止损（无止盈），看止损本身的特性
            rpt.add_section(f'{scenario_name(sc, th_main)} | Δ={delta} — 纯止损特性')
            h = ['K线止损比例', 'N', '触发率', '触发时EV', '触发中位bar', '未触发时EV']
            r_rows = []
            for pct in CANDLE_STOP_PCTS:
                key = f'candle{int(pct*100)}'
                triggered = sub[f'{key}_hit'].sum()
                trig_rate = triggered / len(sub) * 100
                # 触发的那些
                sub_hit = sub[sub[f'{key}_hit'] == 1]
                sub_miss = sub[sub[f'{key}_hit'] == 0]
                ev_hit = calc_ev(sub_hit[f'{key}_stop_roi'].tolist()) if len(sub_hit) > 0 else {'EV': '-', 'N': 0}
                ev_miss = calc_ev(sub_miss[f'{key}_stop_roi'].tolist()) if len(sub_miss) > 0 else {'EV': '-', 'N': 0}
                med_bar = sub_hit[f'{key}_stop_bar'].median() if len(sub_hit) > 0 else '-'
                r_rows.append([
                    f'{int(pct*100)}%', len(sub), f'{trig_rate:.0f}%',
                    ev_hit['EV'], med_bar, ev_miss['EV']
                ])
            rpt.add_table(h, r_rows)

            # K线止损+固定止盈组合
            rpt.add_section(f'{scenario_name(sc, th_main)} | Δ={delta} — K线止损+固定止盈')
            h = ['止损\\止盈', 'N'] + [f'+{tp}%' for tp in TP_LEVELS]
            r_rows = []
            for pct in CANDLE_STOP_PCTS:
                row = [f'{int(pct*100)}%K线止损', len(sub)]
                for tp in TP_LEVELS:
                    key = f'candle{int(pct*100)}_tp{tp}'
                    pnls = sub[f'{key}_roi'].tolist()
                    ev = calc_ev(pnls)
                    row.append(f"{ev['EV']}")
                r_rows.append(row)
            # 对比：固定止盈/无时限（期货追踪止损兜底）
            row_base = ['期货追踪止损', len(sub)]
            for tp in TP_LEVELS:
                pnls_base = []
                for _, r in sub.iterrows():
                    if r[f'tp{tp}_hit'] == 1:
                        pnls_base.append(tp)
                    else:
                        pnls_base.append(r['futures_stop_roi'])
                ev = calc_ev(pnls_base)
                row_base.append(f"{ev['EV']}")
            r_rows.append(row_base)
            rpt.add_table(h, r_rows)

    # ====== Part 6f: K线结构追踪止盈（核心新增） ======
    rpt.add_section(f'Part 6f: K线结构追踪止盈 (ER≥{th_main})',
                    'ROI达标后，用创新高K线结构追踪止盈。half=K线高度50%回撤，low5=K线最低-5跳')

    for delta in DELTA_LEVELS:
        for sc in [1, 3]:
            sub = df_main[(df_main['scenario'] == sc) & (df_main['delta'] == delta)]
            if len(sub) == 0:
                continue

            for mode, mode_name in [('half', '高度50%回撤'), ('low5', '最低价-5跳')]:
                h = [f'触发ROI\\{mode_name}', 'N', 'EV', 'WR%', 'PR', '触发率', '出场中位bar', 'avg ROI%', 'Σ ROI%']
                r_rows = []
                for act in CANDLE_TRAIL_ACTIVATIONS:
                    key = f'ctrail_{mode}_{act}'
                    pnls = sub[f'{key}_roi'].tolist()
                    ev = calc_ev(pnls)
                    triggered = (sub[f'{key}_reason'] == 'candle_trail').sum()
                    trig_rate = triggered / len(sub) * 100
                    sub_trig = sub[sub[f'{key}_reason'] == 'candle_trail']
                    med_bar = round(sub_trig[f'{key}_bar'].median() * 10 / 60, 1) if len(sub_trig) > 0 else '-'
                    r_rows.append([
                        f'ROI≥{act}%', ev['N'], ev['EV'], ev['wr'], ev['pr'],
                        f'{trig_rate:.0f}%', f'{med_bar}h', ev['avg_pnl'], ev['sum_pnl']
                    ])

                # 对比基线
                base_ev = calc_ev(sub['futures_stop_roi'].tolist())
                r_rows.append([
                    '基线:跟期货止损', base_ev['N'], base_ev['EV'], base_ev['wr'],
                    base_ev['pr'], '-', '-', base_ev['avg_pnl'], base_ev['sum_pnl']
                ])

                rpt.add_section(f'{scenario_name(sc, th_main)} | Δ={delta} | {mode_name}')
                rpt.add_table(h, r_rows, highlight_pnl_cols=[2, 7, 8])

    # ====== Part 7: 总排名 ======
    rpt.add_section('Part 7: 全策略总排名',
                    '所有策略类型按EV排序，找最优出场')

    for delta in DELTA_LEVELS:
        for sc in [1, 3]:
            sub = df_main[(df_main['scenario'] == sc) & (df_main['delta'] == delta)]
            if len(sub) == 0:
                continue

            rankings = []

            # 基线：跟期货止损
            ev_base = calc_ev(sub['futures_stop_roi'].tolist())
            rankings.append(('跟期货止损', ev_base))

            # 固定止盈+时间止损
            for tp in TP_LEVELS:
                for ts_bar in TIME_STOPS:
                    col = f'tp{tp}_ts{ts_bar}_roi'
                    ev = calc_ev(sub[col].tolist())
                    rankings.append((f'固定+{tp}%/{ts_bar*10//60}h', ev))
                # 无时限
                pnls_inf = []
                for _, r in sub.iterrows():
                    if r[f'tp{tp}_hit'] == 1:
                        pnls_inf.append(tp)
                    else:
                        pnls_inf.append(r['futures_stop_roi'])
                ev_inf = calc_ev(pnls_inf)
                rankings.append((f'固定+{tp}%/无时限', ev_inf))

            # 追踪止盈
            for trigger in TRAIL_TRIGGERS:
                for trail_pct in TRAIL_PCTS:
                    key = f'trail_{trigger}_{int(trail_pct*100)}'
                    ev = calc_ev(sub[f'{key}_roi'].tolist())
                    rankings.append((f'追踪+{trigger}%/{int(trail_pct*100)}%保留', ev))

            # 追踪+时间止损
            for trigger in TRAIL_TRIGGERS:
                for trail_pct in TRAIL_PCTS:
                    for ts_bar in TIME_STOPS:
                        key = f'trail_{trigger}_{int(trail_pct*100)}_ts{ts_bar}'
                        ev = calc_ev(sub[f'{key}_roi'].tolist())
                        rankings.append((f'追踪+{trigger}%/{int(trail_pct*100)}%/{ts_bar*10//60}h', ev))

            # 分批出场
            for pc in PARTIAL_CONFIGS:
                for ts_bar in TIME_STOPS:
                    key = f'partial_{pc["leg1_tp"]}_{pc["leg2_tp"]}_ts{ts_bar}'
                    ev = calc_ev(sub[f'{key}_roi'].tolist())
                    rankings.append((f'分批{pc["leg1_tp"]}+{pc["leg2_tp"]}/{ts_bar*10//60}h', ev))

            # 时间递减止盈
            for cfg_name, cfg_tiers in DECAY_TP_CONFIGS:
                for ts_bar in TIME_STOPS:
                    key = f'decay_{cfg_name}_ts{ts_bar}'
                    ev = calc_ev(sub[f'{key}_roi'].tolist())
                    rankings.append((f'递减({cfg_name})/{ts_bar*10//60}h', ev))
                key_nots = f'decay_{cfg_name}'
                ev = calc_ev(sub[f'{key_nots}_roi'].tolist())
                rankings.append((f'递减({cfg_name})/无时限', ev))

            # Delta翻倍出场
            for mult in DELTA_MULT_LEVELS:
                key = f'delta_x{int(mult*10)}'
                ev = calc_ev(sub[f'{key}_roi'].tolist())
                rankings.append((f'Delta×{mult}/无时限', ev))
                for ts_bar in TIME_STOPS:
                    key_ts = f'delta_x{int(mult*10)}_ts{ts_bar}'
                    ev = calc_ev(sub[f'{key_ts}_roi'].tolist())
                    rankings.append((f'Delta×{mult}/{ts_bar*10//60}h', ev))

            # Gamma/Theta比值出场
            for decay in GT_DECAY_LEVELS:
                key = f'gt_{int(decay*100)}'
                ev = calc_ev(sub[f'{key}_roi'].tolist())
                rankings.append((f'G/T降至{int(decay*100)}%/无时限', ev))
                for ts_bar in TIME_STOPS:
                    key_ts = f'gt_{int(decay*100)}_ts{ts_bar}'
                    ev = calc_ev(sub[f'{key_ts}_roi'].tolist())
                    rankings.append((f'G/T降至{int(decay*100)}%/{ts_bar*10//60}h', ev))

            # 信号K线止损+固定止盈
            for pct in [0.3, 0.4, 0.5, 0.6]:
                for tp in TP_LEVELS:
                    key = f'candle{int(pct*100)}_tp{tp}'
                    ev = calc_ev(sub[f'{key}_roi'].tolist())
                    rankings.append((f'K线{int(pct*100)}%止损+{tp}%止盈', ev))

            # K线结构追踪止盈
            for act in CANDLE_TRAIL_ACTIVATIONS:
                for mode, mode_label in [('half', '半K线'), ('low5', '低-5跳')]:
                    key = f'ctrail_{mode}_{act}'
                    ev = calc_ev(sub[f'{key}_roi'].tolist())
                    rankings.append((f'{mode_label}追踪/ROI≥{act}%', ev))

            # 按EV排序取Top15
            rankings.sort(key=lambda x: x[1]['EV'], reverse=True)

            h = ['排名', '策略', 'N', 'EV', 'WR%', 'PR', 'avg ROI%', 'Σ ROI%']
            r_rows = []
            for rank, (name, ev) in enumerate(rankings[:15], 1):
                r_rows.append([
                    rank, name, ev['N'], ev['EV'], ev['wr'],
                    ev['pr'], ev['avg_pnl'], ev['sum_pnl']
                ])

            rpt.add_section(f'{scenario_name(sc, th_main)} | Δ={delta}')
            rpt.add_table(h, r_rows, highlight_pnl_cols=[3, 6, 7])

    # ====== Part 8: ER门槛收紧验证 ======
    rpt.add_section('Part 8: ER门槛收紧验证',
                    '同一策略在不同ER门槛下的表现——收紧后应更优')

    # 用Part7中各门槛的Top1策略做对比
    for delta in DELTA_LEVELS:
        for sc in [1, 3]:
            h = ['ER门槛', 'N', '最优策略', 'EV', 'WR%', 'PR', 'Σ ROI%']
            r_rows = []

            for th in ER_THRESHOLDS:
                df_tp = results_by_threshold[th]
                sub = df_tp[(df_tp['scenario'] == sc) & (df_tp['delta'] == delta)]
                if len(sub) == 0:
                    continue

                best_name = ''
                best_ev = {'EV': -999}

                # 扫描所有策略找最优
                for tp in TP_LEVELS:
                    for ts_bar in TIME_STOPS:
                        col = f'tp{tp}_ts{ts_bar}_roi'
                        ev = calc_ev(sub[col].tolist())
                        if ev['EV'] > best_ev['EV']:
                            best_ev = ev
                            best_name = f'固定+{tp}%/{ts_bar*10//60}h'

                for trigger in TRAIL_TRIGGERS:
                    for trail_pct in TRAIL_PCTS:
                        key = f'trail_{trigger}_{int(trail_pct*100)}'
                        ev = calc_ev(sub[f'{key}_roi'].tolist())
                        if ev['EV'] > best_ev['EV']:
                            best_ev = ev
                            best_name = f'追踪+{trigger}%/{int(trail_pct*100)}%'

                for trigger in TRAIL_TRIGGERS:
                    for trail_pct in TRAIL_PCTS:
                        for ts_bar in TIME_STOPS:
                            key = f'trail_{trigger}_{int(trail_pct*100)}_ts{ts_bar}'
                            ev = calc_ev(sub[f'{key}_roi'].tolist())
                            if ev['EV'] > best_ev['EV']:
                                best_ev = ev
                                best_name = f'追踪+{trigger}%/{int(trail_pct*100)}%/{ts_bar*10//60}h'

                for pc in PARTIAL_CONFIGS:
                    for ts_bar in TIME_STOPS:
                        key = f'partial_{pc["leg1_tp"]}_{pc["leg2_tp"]}_ts{ts_bar}'
                        ev = calc_ev(sub[f'{key}_roi'].tolist())
                        if ev['EV'] > best_ev['EV']:
                            best_ev = ev
                            best_name = f'分批{pc["leg1_tp"]}+{pc["leg2_tp"]}/{ts_bar*10//60}h'

                # 时间递减止盈
                for cfg_name, cfg_tiers in DECAY_TP_CONFIGS:
                    key_nots = f'decay_{cfg_name}'
                    ev = calc_ev(sub[f'{key_nots}_roi'].tolist())
                    if ev['EV'] > best_ev['EV']:
                        best_ev = ev
                        best_name = f'递减({cfg_name})'
                    for ts_bar in TIME_STOPS:
                        key = f'decay_{cfg_name}_ts{ts_bar}'
                        ev = calc_ev(sub[f'{key}_roi'].tolist())
                        if ev['EV'] > best_ev['EV']:
                            best_ev = ev
                            best_name = f'递减({cfg_name})/{ts_bar*10//60}h'

                # Delta翻倍
                for mult in DELTA_MULT_LEVELS:
                    key = f'delta_x{int(mult*10)}'
                    ev = calc_ev(sub[f'{key}_roi'].tolist())
                    if ev['EV'] > best_ev['EV']:
                        best_ev = ev
                        best_name = f'Delta×{mult}'
                    for ts_bar in TIME_STOPS:
                        key_ts = f'delta_x{int(mult*10)}_ts{ts_bar}'
                        ev = calc_ev(sub[f'{key_ts}_roi'].tolist())
                        if ev['EV'] > best_ev['EV']:
                            best_ev = ev
                            best_name = f'Delta×{mult}/{ts_bar*10//60}h'

                # G/T比值
                for decay in GT_DECAY_LEVELS:
                    key = f'gt_{int(decay*100)}'
                    ev = calc_ev(sub[f'{key}_roi'].tolist())
                    if ev['EV'] > best_ev['EV']:
                        best_ev = ev
                        best_name = f'G/T降至{int(decay*100)}%'
                    for ts_bar in TIME_STOPS:
                        key_ts = f'gt_{int(decay*100)}_ts{ts_bar}'
                        ev = calc_ev(sub[f'{key_ts}_roi'].tolist())
                        if ev['EV'] > best_ev['EV']:
                            best_ev = ev
                            best_name = f'G/T降至{int(decay*100)}%/{ts_bar*10//60}h'

                # K线止损+止盈
                for pct in [0.3, 0.4, 0.5, 0.6]:
                    for tp in TP_LEVELS:
                        key = f'candle{int(pct*100)}_tp{tp}'
                        ev = calc_ev(sub[f'{key}_roi'].tolist())
                        if ev['EV'] > best_ev['EV']:
                            best_ev = ev
                            best_name = f'K线{int(pct*100)}%止损+{tp}%止盈'

                # K线追踪止盈
                for act in CANDLE_TRAIL_ACTIVATIONS:
                    for mode, mode_label in [('half', '半K线'), ('low5', '低-5跳')]:
                        key = f'ctrail_{mode}_{act}'
                        ev = calc_ev(sub[f'{key}_roi'].tolist())
                        if ev['EV'] > best_ev['EV']:
                            best_ev = ev
                            best_name = f'{mode_label}追踪/ROI≥{act}%'

                r_rows.append([
                    th, best_ev['N'], best_name,
                    best_ev['EV'], best_ev['wr'], best_ev['pr'], best_ev['sum_pnl']
                ])

            if r_rows:
                rpt.add_section(f'{scenario_name(sc, th_main)} | Δ={delta}')
                rpt.add_table(h, r_rows, highlight_pnl_cols=[3, 6])

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
    print("=" * 70)
    print("期权出场策略优化研究")
    print("=" * 70)

    # 加载数据（一次加载，多次使用）
    print("\n加载数据...")
    all_data = load_all(period_min=10, days=170, last_days=None,
                        emas=(10, 20, 120), er_periods=(5, 20, 40), atr_period=14)
    print(f"品种数: {len(all_data)}")

    for sym_key, df in all_data.items():
        add_hv(df, windows=[20, 40])
        df['er5_delta_6'] = df['er_5'] - df['er_5'].shift(6)

    # 多门槛循环
    results_by_threshold = {}

    for th in ER_THRESHOLDS:
        print(f"\n{'='*50}")
        print(f"ER门槛 = {th}")
        print(f"{'='*50}")

        signals = collect_signals(all_data, er_threshold=th)
        print(f"  信号数: {len(signals)}笔")

        # 按场景统计
        from collections import Counter
        sc_counts = Counter(s['scenario'] for s in signals)
        for sc in sorted(sc_counts):
            print(f"    场景{sc}: {sc_counts[sc]}笔")

        # 计算期权MFE路径
        print(f"  计算期权MFE路径（{len(signals)}笔 × {len(DELTA_LEVELS)} Delta）...")
        all_results = []
        for i, sig in enumerate(signals):
            df = all_data[sig['df_key']]
            for delta in DELTA_LEVELS:
                r = compute_option_mfe_path(df, sig, delta, T_DAYS_DEFAULT)
                if r:
                    r['delta'] = delta
                    all_results.append(r)
            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/{len(signals)}...")

        print(f"  有效结果: {len(all_results)}条")

        # 模拟所有策略
        print("  模拟出场策略...")
        df_tp = simulate_all_strategies(all_results)
        print(f"  生成 {len(df_tp)} 条记录")

        results_by_threshold[th] = df_tp

    # 生成报告
    print("\n" + "=" * 70)
    print("生成报告...")
    rpt = build_report(results_by_threshold)
    rpt.save('output/option_exit_advanced.html')
    print("完成。")
