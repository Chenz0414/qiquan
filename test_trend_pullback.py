# -*- coding: utf-8 -*-
"""
强趋势回调 Type1（影线触碰EMA10）MFE 研究
==========================================
全新入场逻辑，独立于现有 ABC 信号系统。

规则：
  趋势：ER(20) > 0.3
  方向：close > EMA60 只做多，close < EMA60 只做空
  信号：影线碰EMA10弹回（close不破），prev_close未破EMA10
  入场：信号K线 high+1tick 挂单（多），5根内未触发则撤，跳空不入
  止损：信号K线 low - 5*tick
  追踪：入场后60根K线的MFE/MAE

用法:
    python test_trend_pullback.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

from data_loader import load_all, BARS_PER_DAY
from signal_core import SYMBOL_CONFIGS
from stats_utils import calc_ev, ev_line
from chart_engine import render_chart, get_chart_js
from report_engine import Report

# ============ 固定参数（不做优化） ============
ER_THRESHOLD = 0.3
PENDING_EXPIRY = 5       # 挂单有效K线数
STOP_TICKS = 5           # 止损缓冲跳数
MFE_WINDOW = 60          # MFE追踪窗口（根）
MARGIN_RATE = 0.15       # 保证金比例（固定15%）
BACKTEST_DAYS = 120
WARMUP_DAYS = 50
TRIM_PCTS = [3, 5, 10]


# ============ 信号检测 ============

def detect_signals(df, tick_size):
    """
    扫描单品种 DataFrame，检测 Type1 影线触碰信号。
    同时计算7个子测试所需的特征字段。

    返回: list[dict]  每个dict是一个信号（待挂单）
    """
    n = len(df)
    signals = []
    pending = None  # 当前挂单

    # 需要的列预取为 numpy array 提速
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    opn = df['open'].values
    ema10 = df['ema10'].values
    ema60 = df['ema60'].values
    er20 = df['er_20'].values
    atr = df['atr'].values

    # 新增ER列
    er5 = df['er_5'].values if 'er_5' in df.columns else np.full(n, np.nan)
    er40 = df['er_40'].values if 'er_40' in df.columns else np.full(n, np.nan)

    # 成交量
    volume = df['volume'].values if 'volume' in df.columns else np.full(n, np.nan)
    vol_ma20 = pd.Series(volume).rolling(20, min_periods=20).mean().values

    # ATR 20周期均值（用于ATR膨胀比计算）
    atr_ma20 = pd.Series(atr).rolling(20, min_periods=20).mean().values

    # RSI(14) 预计算
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss_s = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi_arr = (100 - 100 / (1 + gain / loss_s.replace(0, np.nan))).values

    # EMA10斜率预计算（5根变化/ATR）
    ema10_series = pd.Series(ema10)
    ema10_shift5 = ema10_series.shift(5).values

    # R²(30) 预计算
    r2_arr = np.full(n, np.nan)
    x_30 = np.arange(30)
    for k in range(30, n):
        y = close[k-30:k]
        if np.std(y) > 0:
            corr = np.corrcoef(x_30, y)[0, 1]
            r2_arr[k] = corr ** 2

    # ADX(14) 预计算
    adx_arr = np.full(n, np.nan)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr_arr = np.zeros(n)
    for k in range(1, n):
        h_diff = high[k] - high[k-1]
        l_diff = low[k-1] - low[k]
        plus_dm[k] = h_diff if (h_diff > l_diff and h_diff > 0) else 0
        minus_dm[k] = l_diff if (l_diff > h_diff and l_diff > 0) else 0
        tr_arr[k] = max(high[k] - low[k], abs(high[k] - close[k-1]), abs(low[k] - close[k-1]))
    atr14_s = pd.Series(tr_arr).ewm(span=14, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=14, adjust=False).mean().values / np.where(atr14_s > 0, atr14_s, 1) * 100
    minus_di = pd.Series(minus_dm).ewm(span=14, adjust=False).mean().values / np.where(atr14_s > 0, atr14_s, 1) * 100
    dx = np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, plus_di + minus_di, 1) * 100
    adx_arr = pd.Series(dx).ewm(span=14, adjust=False).mean().values

    # 有 datetime 列就用，没有就用 index
    has_dt = 'datetime' in df.columns
    datetimes = df['datetime'].values if has_dt else None
    # 提取小时用于尾盘判断
    if has_dt:
        hours = pd.to_datetime(df['datetime']).dt.hour.values
    else:
        hours = None

    # 信号密集度追踪：记录最近信号的bar index
    recent_signal_bars = []

    filled_trades = []

    # 测试5：回踩次数计数器
    touch_count_long = 0
    touch_count_short = 0
    prev_dir = 0  # 上一根bar的EMA60方向：1=多 -1=空 0=未知

    for i in range(1, n):
        # --- 1. 检查挂单状态 ---
        if pending is not None:
            # 过期检查
            if i > pending['expiry_idx']:
                pending['outcome'] = 'expired'
                signals.append(pending)
                pending = None
            else:
                triggered = False
                if pending['direction'] == 'long':
                    if opn[i] > pending['price']:
                        pending['outcome'] = 'gap_skip'
                        signals.append(pending)
                        pending = None
                    elif high[i] >= pending['price']:
                        triggered = True
                else:  # short
                    if opn[i] < pending['price']:
                        pending['outcome'] = 'gap_skip'
                        signals.append(pending)
                        pending = None
                    elif low[i] <= pending['price']:
                        triggered = True

                if triggered:
                    pending['outcome'] = 'filled'
                    pending['entry_bar_idx'] = i
                    pending['entry_datetime'] = datetimes[i] if has_dt else i
                    pending['bars_to_fill'] = i - pending['signal_bar_idx']
                    mfe_data = track_mfe_mae(
                        high, low, close, i,
                        pending['price'], pending['direction'],
                        pending['stop_price'], pending['atr']
                    )
                    pending.update(mfe_data)
                    signals.append(pending)
                    filled_trades.append(pending)
                    pending = None

        # --- 更新回踩计数：EMA60方向切换时重置 ---
        if not np.isnan(ema60[i]):
            curr_dir = 1 if close[i] > ema60[i] else (-1 if close[i] < ema60[i] else 0)
            if curr_dir != prev_dir and prev_dir != 0:
                # 方向切换，重置对应方向的计数
                if curr_dir == 1:
                    touch_count_long = 0
                elif curr_dir == -1:
                    touch_count_short = 0
            prev_dir = curr_dir

        # --- 2. 前置过滤 ---
        if np.isnan(er20[i]) or np.isnan(atr[i]) or np.isnan(ema10[i]) or np.isnan(ema60[i]):
            continue
        if er20[i] <= ER_THRESHOLD:
            continue

        # --- 3. 方向判断 ---
        direction = None
        if close[i] > ema60[i]:
            direction = 'long'
        elif close[i] < ema60[i]:
            direction = 'short'
        else:
            continue

        # --- 4. 信号检测 ---
        signal_found = False

        if direction == 'long':
            if (low[i] <= ema10[i] and close[i] > ema10[i]
                    and close[i-1] >= ema10[i-1]):
                pend_price = round_to_tick(high[i] + tick_size, tick_size)
                stop = round_to_tick(low[i] - STOP_TICKS * tick_size, tick_size)
                touch_count_long += 1
                signal_found = True
        else:  # short
            if (high[i] >= ema10[i] and close[i] < ema10[i]
                    and close[i-1] <= ema10[i-1]):
                pend_price = round_to_tick(low[i] - tick_size, tick_size)
                stop = round_to_tick(high[i] + STOP_TICKS * tick_size, tick_size)
                touch_count_short += 1
                signal_found = True

        if signal_found:
            # --- 计算子测试特征 ---
            bar_range = high[i] - low[i]
            body = abs(close[i] - opn[i])
            atr_i = atr[i]

            # 测试1：是否阳线（多头看阳线，空头看阴线）
            if direction == 'long':
                is_bullish = close[i] > opn[i]
            else:
                is_bullish = close[i] < opn[i]  # 空头"阳"=阴线=有利方向

            # 测试2：实体占比 = body / range（越小=影线越长）
            body_ratio = body / bar_range if bar_range > 0 else 1.0

            # 测试4：穿透EMA10深度（ATR单位）
            if direction == 'long':
                penetration = (ema10[i] - low[i]) / atr_i if atr_i > 0 else 0
            else:
                penetration = (high[i] - ema10[i]) / atr_i if atr_i > 0 else 0

            # 测试5：第几次回踩
            touch_seq = touch_count_long if direction == 'long' else touch_count_short

            # 测试6：信号前20根的趋势波幅 / ATR
            # 多头：前20根最高点 - 前20根最低点（不含信号K线自身）
            lookback = 20
            if i >= lookback + 1 and atr_i > 0:
                prev_high = high[i-lookback:i]   # 不含当根
                prev_low = low[i-lookback:i]
                momentum = (max(prev_high) - min(prev_low)) / atr_i
            else:
                momentum = 0.0

            # --- 新增反向过滤特征 ---

            # 测试8：止损距离 / ATR（影线深度代理）
            stop_distance = abs(pend_price - stop)
            stop_dist_atr = stop_distance / atr_i if atr_i > 0 else 0

            # 测试9：信号时段（小时）
            signal_hour = int(hours[i]) if hours is not None else -1

            # 测试10：ATR膨胀比 = ATR / ATR_MA20
            atr_ratio = atr[i] / atr_ma20[i] if (not np.isnan(atr_ma20[i]) and atr_ma20[i] > 0) else 1.0

            # 测试11：信号密集度（最近10根内出现过几次Type1信号）
            recent_signal_bars = [b for b in recent_signal_bars if b >= i - 10]
            signal_density = len(recent_signal_bars)
            recent_signal_bars.append(i)

            # 测试12：日内方向（close vs 当日开盘价的偏移/ATR）
            # 用当天第一根K线的open近似当日开盘价
            if has_dt:
                sig_dt = pd.Timestamp(datetimes[i])
                sig_date = sig_dt.date()
                # 向前找当天第一根bar的open
                day_open = opn[i]  # fallback
                for back in range(min(i, 300)):
                    check_dt = pd.Timestamp(datetimes[i - back])
                    if check_dt.date() != sig_date:
                        # 前一根是不同日期，当前back-1位置就是当天开始
                        if back > 0:
                            day_open = opn[i - back + 1]
                        break
                intraday_dir = (close[i] - day_open) / atr_i if atr_i > 0 else 0
            else:
                intraday_dir = 0.0

            # --- 第二轮特征 ---

            # EMA展开度 = (ema10 - ema60) / atr（顺势为正）
            ema_spread = (ema10[i] - ema60[i]) / atr_i if atr_i > 0 else 0
            if direction == 'short':
                ema_spread = -ema_spread

            # 前一根K线变动/ATR（正=顺势）
            prev_bar_move = 0.0
            if i >= 1 and atr_i > 0:
                if direction == 'long':
                    prev_bar_move = (close[i-1] - opn[i-1]) / atr_i
                else:
                    prev_bar_move = (opn[i-1] - close[i-1]) / atr_i

            # 星期几
            weekday = pd.Timestamp(datetimes[i]).weekday() if has_dt else -1

            # --- 第三轮特征（因子系统） ---

            # ER(5)近6根变化
            er5_chg6 = (er5[i] - er5[i-6]) if (i >= 6 and not np.isnan(er5[i]) and not np.isnan(er5[i-6])) else 0.0

            # ER(40)水平
            er_40_val = er40[i] if not np.isnan(er40[i]) else 0.0

            # ER(40)近12根变化
            er40_chg12 = (er40[i] - er40[i-12]) if (i >= 12 and not np.isnan(er40[i]) and not np.isnan(er40[i-12])) else 0.0

            # RSI(14)
            rsi_val = rsi_arr[i] if not np.isnan(rsi_arr[i]) else 50.0

            # 成交量比
            vol_ratio_val = volume[i] / vol_ma20[i] if (not np.isnan(vol_ma20[i]) and vol_ma20[i] > 0) else 1.0

            # ADX(14)
            adx_val = adx_arr[i] if not np.isnan(adx_arr[i]) else 0.0

            # R²(30)
            r2_val = r2_arr[i] if not np.isnan(r2_arr[i]) else 0.0

            # EMA10斜率 (5根变化/ATR)
            ma_slope_val = 0.0
            if not np.isnan(ema10_shift5[i]) and atr_i > 0:
                slope_raw = (ema10[i] - ema10_shift5[i]) / atr_i / 5
                ma_slope_val = slope_raw if direction == 'long' else -slope_raw

            # 回调前连续顺势K线数
            consec = 0
            for back in range(1, min(i, 50) + 1):
                idx_b = i - back
                if direction == 'long':
                    if close[idx_b] > opn[idx_b]:  # 阳线=顺势
                        consec += 1
                    else:
                        break
                else:
                    if close[idx_b] < opn[idx_b]:  # 阴线=顺势
                        consec += 1
                    else:
                        break
            consec_trend_bars = consec

            # 趋势年龄（从EMA60方向切换到现在的根数）
            trend_age_val = 0
            for back in range(1, min(i, 500) + 1):
                idx_b = i - back
                if np.isnan(ema60[idx_b]):
                    break
                if direction == 'long' and close[idx_b] < ema60[idx_b]:
                    trend_age_val = back
                    break
                elif direction == 'short' and close[idx_b] > ema60[idx_b]:
                    trend_age_val = back
                    break
            else:
                trend_age_val = min(i, 500)

            sig = {
                'signal_bar_idx': i,
                'signal_datetime': datetimes[i] if has_dt else i,
                'direction': direction,
                'price': pend_price,
                'stop_price': stop,
                'expiry_idx': i + PENDING_EXPIRY,
                'er_20': round(er20[i], 4),
                'atr': round(atr[i], 4),
                'ema10': round(ema10[i], 2),
                'ema60': round(ema60[i], 2),
                'close_at_signal': round(close[i], 2),
                'high_at_signal': round(high[i], 2),
                'low_at_signal': round(low[i], 2),
                # 子测试字段
                'is_bullish': is_bullish,
                'body_ratio': round(body_ratio, 4),
                'penetration_atr': round(penetration, 4),
                'touch_seq': touch_seq,
                'momentum_atr': round(momentum, 4),
                # 反向过滤特征
                'stop_dist_atr': round(stop_dist_atr, 4),
                'signal_hour': signal_hour,
                'atr_ratio': round(atr_ratio, 4),
                'signal_density': signal_density,
                'intraday_dir': round(intraday_dir, 4),
                # 第二轮特征
                'ema_spread': round(ema_spread, 4),
                'prev_bar_move': round(prev_bar_move, 4),
                'weekday': weekday,
                # 第三轮特征（因子系统）
                'er5_chg6': round(er5_chg6, 4),
                'er_40': round(er_40_val, 4),
                'er40_chg12': round(er40_chg12, 4),
                'rsi_14': round(rsi_val, 2),
                'vol_ratio': round(vol_ratio_val, 4),
                'adx_14': round(adx_val, 2),
                'r2_30': round(r2_val, 4),
                'ma_slope': round(ma_slope_val, 4),
                'consec_trend_bars': consec_trend_bars,
                'trend_age': trend_age_val,
                'outcome': None,
            }
            # 新信号替换旧pending
            if pending is not None:
                pending['outcome'] = 'replaced'
                signals.append(pending)
            pending = sig

    # 收尾：剩余pending
    if pending is not None:
        pending['outcome'] = 'expired'
        signals.append(pending)

    return signals


def track_mfe_mae(high, low, close, entry_idx, entry_price, direction,
                  stop_price, atr_val):
    """
    追踪入场后60根K线的MFE/MAE。
    止损触发后停止追踪（实际已出场）。
    MAE上限 = 止损距离（不可能亏更多）。

    参数都是 numpy array（high/low/close）或标量。
    返回 dict。
    """
    n = len(high)
    mfe_raw = 0.0
    mae_raw = 0.0
    mfe_bar = 0
    mae_bar = 0
    stop_hit = False
    stop_hit_bar = 0
    bars_tracked = 0

    # MAE上限 = 止损距离
    stop_distance = abs(entry_price - stop_price)

    for k in range(1, MFE_WINDOW + 1):
        idx = entry_idx + k
        if idx >= n:
            break
        bars_tracked = k

        if direction == 'long':
            # 止损bar也可能先涨后跌，先记录MFE再判止损
            favorable = high[idx] - entry_price
            if favorable > mfe_raw:
                mfe_raw = favorable
                mfe_bar = k
            if low[idx] <= stop_price:
                stop_hit = True
                stop_hit_bar = k
                mae_raw = stop_distance
                mae_bar = k
                break  # 出场，停止追踪
            adverse = min(entry_price - low[idx], stop_distance)
        else:
            favorable = entry_price - low[idx]
            if favorable > mfe_raw:
                mfe_raw = favorable
                mfe_bar = k
            if high[idx] >= stop_price:
                stop_hit = True
                stop_hit_bar = k
                mae_raw = stop_distance
                mae_bar = k
                break  # 出场，停止追踪
            adverse = min(high[idx] - entry_price, stop_distance)

        # MFE已在上面更新，这里只更新MAE
        if adverse > mae_raw:
            mae_raw = adverse
            mae_bar = k

    atr_safe = atr_val if atr_val > 0 else 1.0
    # 保证金收益率 = 价格变动% / 保证金比例
    mfe_margin = mfe_raw / entry_price / MARGIN_RATE * 100  # %
    mae_margin = mae_raw / entry_price / MARGIN_RATE * 100  # %
    stop_loss_margin = stop_distance / entry_price / MARGIN_RATE * 100  # %

    return {
        'mfe_raw': round(mfe_raw, 4),
        'mae_raw': round(mae_raw, 4),
        'mfe_atr': round(mfe_raw / atr_safe, 4),
        'mae_atr': round(mae_raw / atr_safe, 4),
        'mfe_pct': round(mfe_raw / entry_price * 100, 4),
        'mae_pct': round(mae_raw / entry_price * 100, 4),
        'mfe_margin': round(mfe_margin, 2),
        'mae_margin': round(mae_margin, 2),
        'stop_loss_margin': round(stop_loss_margin, 2),
        'mfe_bar': mfe_bar,
        'mae_bar': mae_bar,
        'stop_hit': stop_hit,
        'stop_hit_bar': stop_hit_bar if stop_hit else 0,
        'bars_tracked': bars_tracked,
    }


def round_to_tick(price, tick_size):
    """四舍五入到最近的 tick"""
    return round(round(price / tick_size) * tick_size, 10)


# ============ 单品种运行 ============

def run_single_symbol(sym_key, df):
    """对单品种运行信号检测 + MFE追踪，返回交易记录列表"""
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
            # 子测试字段（所有信号都记录，含未成交的）
            'is_bullish': sig.get('is_bullish'),
            'body_ratio': sig.get('body_ratio'),
            'penetration_atr': sig.get('penetration_atr'),
            'touch_seq': sig.get('touch_seq'),
            'momentum_atr': sig.get('momentum_atr'),
            'stop_dist_atr': sig.get('stop_dist_atr'),
            'signal_hour': sig.get('signal_hour'),
            'atr_ratio': sig.get('atr_ratio'),
            'signal_density': sig.get('signal_density'),
            'intraday_dir': sig.get('intraday_dir'),
            'ema_spread': sig.get('ema_spread'),
            'prev_bar_move': sig.get('prev_bar_move'),
            'weekday': sig.get('weekday'),
            # 第三轮因子
            'er5_chg6': sig.get('er5_chg6'),
            'er_40': sig.get('er_40'),
            'er40_chg12': sig.get('er40_chg12'),
            'rsi_14': sig.get('rsi_14'),
            'vol_ratio': sig.get('vol_ratio'),
            'adx_14': sig.get('adx_14'),
            'r2_30': sig.get('r2_30'),
            'ma_slope': sig.get('ma_slope'),
            'consec_trend_bars': sig.get('consec_trend_bars'),
            'trend_age': sig.get('trend_age'),
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
                'mfe_pct': sig['mfe_pct'],
                'mae_pct': sig['mae_pct'],
                'mfe_margin': sig['mfe_margin'],
                'mae_margin': sig['mae_margin'],
                'stop_loss_margin': sig['stop_loss_margin'],
                'mfe_bar': sig['mfe_bar'],
                'mae_bar': sig['mae_bar'],
                'stop_hit': sig['stop_hit'],
                'stop_hit_bar': sig['stop_hit_bar'],
                'bars_tracked': sig['bars_tracked'],
            })

        trades.append(rec)

    return trades


# ============ 全品种运行 ============

def run_all():
    """加载全部品种，运行回测，返回 (df_all, df_filled, all_data)"""
    print("加载数据（缓存）...")
    all_data = load_all(
        period_min=10,
        days=BACKTEST_DAYS + WARMUP_DAYS,
        last_days=None,
        emas=(10, 20, 60, 120),
        er_periods=(5, 20, 40),
        atr_period=14,
        min_bars=200
    )
    print(f"  加载了 {len(all_data)} 个品种")

    all_trades = []
    for sym_key in sorted(all_data.keys()):
        df = all_data[sym_key]
        # 信号只在最后 BACKTEST_DAYS 天内生成，前面是预热
        n = len(df)
        start = max(0, n - BACKTEST_DAYS * BARS_PER_DAY)
        # 但检测需要 prev bar，所以从 start 开始的子集要包含前面足够数据
        # 解决方案：整段扫描，但只保留 signal_bar_idx >= start 的信号
        trades = run_single_symbol(sym_key, df)
        # 过滤：只保留信号在回测区间内的
        trades = [t for t in trades if t['signal_bar_idx'] >= start]
        all_trades.extend(trades)

    df_all = pd.DataFrame(all_trades)
    if len(df_all) == 0:
        print("没有检测到任何信号！")
        return df_all, df_all, all_data

    df_filled = df_all[df_all['outcome'] == 'filled'].copy()

    # 无硬过滤：所有因子都进评分系统，不做后置过滤
    # touch>=6、滚动胜率等全部作为因子参与组合扫描

    # 计算"同品种最近5笔已完成信号的盈利层数量"（作为评分因子）
    if len(df_filled) > 0:
        stop_dist = (df_filled['entry_price'] - df_filled['stop_price']).abs()
        r_mult = (df_filled['mfe_raw'] / stop_dist).fillna(0)
        df_filled['_is_profit'] = (r_mult >= 1).astype(int)
        # 按品种+时间排序
        df_filled = df_filled.sort_values(['symbol', 'signal_datetime']).reset_index(drop=True)
        # 用index对齐，避免groupby顺序问题
        df_filled['recent_win_n'] = -1
        df_filled['recent_total_n'] = 0
        for _, grp in df_filled.groupby('symbol'):
            wins = grp['_is_profit'].values
            idxs = grp.index
            for j_pos, idx in enumerate(idxs):
                if j_pos == 0:
                    df_filled.at[idx, 'recent_win_n'] = -1
                    df_filled.at[idx, 'recent_total_n'] = 0
                else:
                    start = max(0, j_pos - 5)
                    prev = wins[start:j_pos]
                    df_filled.at[idx, 'recent_win_n'] = int(prev.sum())
                    df_filled.at[idx, 'recent_total_n'] = len(prev)
        df_filled.drop(columns=['_is_profit'], inplace=True)

    # PnL = MFE（理论天花板：假设每笔都在MFE完美出场）
    # 入场研究只看天花板，出场策略单独细化
    if len(df_filled) > 0:
        df_filled['pnl_margin'] = df_filled['mfe_margin']

    # 添加辅助列
    if len(df_filled) > 0 and 'signal_datetime' in df_filled.columns:
        try:
            dt = pd.to_datetime(df_filled['signal_datetime'])
            df_filled['month'] = dt.dt.to_period('M').astype(str)
            # 时间切分：前半 vs 后半
            mid = dt.min() + (dt.max() - dt.min()) / 2
            df_filled['half'] = np.where(dt <= mid, 1, 2)
        except Exception:
            df_filled['month'] = 'unknown'
            df_filled['half'] = 1

    return df_all, df_filled, all_data


# ============ 分析工具 ============

def dist_stats(values, label=''):
    """计算分布统计"""
    if len(values) == 0:
        return {'N': 0, 'avg': 0, 'med': 0, 'p25': 0, 'p75': 0, 'p90': 0, 'std': 0}
    v = np.array(values)
    return {
        'N': len(v),
        'avg': round(np.mean(v), 3),
        'med': round(np.median(v), 3),
        'p25': round(np.percentile(v, 25), 3),
        'p75': round(np.percentile(v, 75), 3),
        'p90': round(np.percentile(v, 90), 3),
        'std': round(np.std(v), 3),
    }


def ev_row(label, pnls):
    """生成一行MFE统计，返回列表（用于表格行）"""
    v = np.array(pnls)
    n = len(v)
    avg = np.mean(v) if n > 0 else 0
    med = np.median(v) if n > 0 else 0
    p75 = np.percentile(v, 75) if n > 0 else 0
    total = np.sum(v) if n > 0 else 0
    warn = ' *' if n < 30 else ''
    return [label, f'{n}{warn}', f'{avg:.2f}%', f'{med:.2f}%',
            f'{p75:.2f}%', f'{total:.1f}%']


def tier_summary_row(label, df_sub):
    """生成一行R倍数分布摘要，用于子测试对比。
    列：场景, N, avg MFE%, avg R, 亏损层占比(R<1), R3+占比, R3+MFE贡献
    """
    n = len(df_sub)
    if n == 0:
        return None
    stop_dist = (df_sub['entry_price'] - df_sub['stop_price']).abs()
    r_mult = df_sub['mfe_raw'] / stop_dist
    r_mult = r_mult.fillna(0)

    avg_mfe = df_sub['mfe_margin'].mean()
    avg_r = r_mult.mean()

    loss_pct = (r_mult < 1).sum() / n * 100
    r3_mask = r_mult >= 3
    r3_pct = r3_mask.sum() / n * 100
    total_mfe = df_sub['mfe_margin'].sum()
    r3_mfe_contrib = df_sub.loc[r3_mask, 'mfe_margin'].sum() / total_mfe * 100 if total_mfe > 0 else 0

    warn = ' *' if n < 30 else ''
    return [label, f'{n}{warn}', f'{avg_mfe:.2f}%', f'{avg_r:.2f}',
            f'{loss_pct:.1f}%', f'{r3_pct:.1f}%', f'{r3_mfe_contrib:.1f}%']


def build_quality_tiers_for_subset(rpt, df_sub, title):
    """为任意子集生成完整R倍数分层表（用于子测试详细对比）"""
    if len(df_sub) == 0:
        return
    stop_dist = (df_sub['entry_price'] - df_sub['stop_price']).abs()
    r_mult = df_sub['mfe_raw'] / stop_dist
    r_mult = r_mult.fillna(0)
    df_sub = df_sub.copy()
    df_sub['r_multiple'] = r_mult
    total_n = len(df_sub)
    total_mfe_sum = df_sub['mfe_margin'].sum()

    def _row(label, sub, bold=False):
        if len(sub) == 0:
            return None
        mfe_contrib = sub['mfe_margin'].sum() / total_mfe_sum * 100 if total_mfe_sum > 0 else 0
        r_vals = sub['r_multiple']
        p = '**' if bold else ''
        warn = ' *' if len(sub) < 30 else ''
        return [f'{p}{label}{p}', f'{len(sub)}{warn}', f'{len(sub)/total_n*100:.1f}%',
                f'{sub["mfe_margin"].mean():.2f}%', f'{sub["mfe_margin"].median():.2f}%',
                f'{r_vals.mean():.2f}', f'{mfe_contrib:.1f}%',
                f'{sub["mfe_bar"].mean():.0f}']

    tiers = [
        ('R=0 入场即反转', r_mult == 0),
        ('R 0~0.5 未到半损', (r_mult > 0) & (r_mult < 0.5)),
        ('R 0.5~1 未到保本', (r_mult >= 0.5) & (r_mult < 1)),
        ('R 1~2 到过1:1', (r_mult >= 1) & (r_mult < 2)),
        ('R 2~3 到过2:1', (r_mult >= 2) & (r_mult < 3)),
        ('R 3+ 超过3:1', r_mult >= 3),
    ]
    headers = ['层级', 'N', '占比', 'avg MFE%', 'med MFE%',
               'avg R', 'MFE贡献', 'avg MFE bar']
    rows = []
    for label, mask in tiers:
        sub = df_sub[mask].copy()
        sub['r_multiple'] = r_mult[mask]
        row = _row(label, sub)
        if row: rows.append(row)

    # 汇总行
    loss_sub = df_sub[r_mult < 1].copy(); loss_sub['r_multiple'] = r_mult[r_mult < 1]
    profit_sub = df_sub[r_mult >= 1].copy(); profit_sub['r_multiple'] = r_mult[r_mult >= 1]
    all_sub = df_sub.copy()
    r_loss = _row('亏损层 (R<1)', loss_sub, bold=True)
    r_profit = _row('盈利层 (R>=1)', profit_sub, bold=True)
    r_all = _row('合计', all_sub, bold=True)
    if r_loss: rows.append(r_loss)
    if r_profit: rows.append(r_profit)
    if r_all: rows.append(r_all)

    rpt.add_text(f'▸ {title} (N={total_n})')
    rpt.add_table(headers, rows, highlight_pnl_cols=[3])


def build_quality_tiers(rpt, df_filled):
    """交易质量分层表：按MFE/止损距离的R倍数分层"""
    # 计算R倍数
    stop_dist = (df_filled['entry_price'] - df_filled['stop_price']).abs()
    r_mult = df_filled['mfe_raw'] / stop_dist
    r_mult = r_mult.fillna(0)
    df_filled = df_filled.copy()
    df_filled['r_multiple'] = r_mult

    stopped = df_filled['stop_hit'].astype(bool)
    total_n = len(df_filled)
    total_mfe_sum = df_filled['mfe_margin'].sum()

    # 分层定义：(label, R条件mask, 止损条件mask or None)
    def tier_row(label, sub, total_n, total_mfe_sum, bold=False):
        """生成一行tier统计"""
        if len(sub) == 0:
            return None
        mfe_contrib = sub['mfe_margin'].sum() / total_mfe_sum * 100 if total_mfe_sum > 0 else 0
        r_vals = sub['r_multiple'] if 'r_multiple' in sub.columns else pd.Series([0])
        prefix = '**' if bold else ''
        suffix = '**' if bold else ''
        return [
            f'{prefix}{label}{suffix}',
            len(sub),
            f'{len(sub)/total_n*100:.1f}%',
            f'{sub["mfe_margin"].mean():.2f}%',
            f'{sub["mfe_margin"].median():.2f}%',
            f'{r_vals.mean():.2f}',
            f'{mfe_contrib:.1f}%',
            f'{sub["mfe_bar"].mean():.0f}',
        ]

    loss_tiers = [
        ('  R=0 入场即反转',   (r_mult == 0)),
        ('  R 0~0.5 未到半损',  (r_mult > 0) & (r_mult < 0.5)),
        ('  R 0.5~1 未到保本',  (r_mult >= 0.5) & (r_mult < 1)),
    ]
    profit_tiers = [
        ('  R 1~2 到过1:1',    (r_mult >= 1) & (r_mult < 2)),
        ('  R 2~3 到过2:1',    (r_mult >= 2) & (r_mult < 3)),
        ('  R 3+ 超过3:1',     (r_mult >= 3)),
    ]

    headers = ['层级', 'N', '占比', 'avg MFE%', 'med MFE%',
               'avg R', 'MFE贡献', 'avg MFE bar']
    rows = []

    # 亏损层明细
    for label, mask in loss_tiers:
        sub = df_filled[mask].copy()
        sub['r_multiple'] = r_mult[mask]
        row = tier_row(label, sub, total_n, total_mfe_sum)
        if row:
            rows.append(row)

    # 盈利层明细
    for label, mask in profit_tiers:
        sub = df_filled[mask].copy()
        sub['r_multiple'] = r_mult[mask]
        row = tier_row(label, sub, total_n, total_mfe_sum)
        if row:
            rows.append(row)

    # 底部：亏损层汇总 + 盈利层汇总 + 合计
    loss_mask = r_mult < 1
    loss_sub = df_filled[loss_mask].copy()
    loss_sub['r_multiple'] = r_mult[loss_mask]
    rows.append(tier_row('亏损层 (R<1)', loss_sub, total_n, total_mfe_sum, bold=True))

    profit_mask = r_mult >= 1
    profit_sub = df_filled[profit_mask].copy()
    profit_sub['r_multiple'] = r_mult[profit_mask]
    rows.append(tier_row('盈利层 (R>=1)', profit_sub, total_n, total_mfe_sum, bold=True))

    df_with_r = df_filled.copy()
    df_with_r['r_multiple'] = r_mult
    rows.append(tier_row('合计', df_with_r, total_n, total_mfe_sum, bold=True))

    rpt.add_section('交易质量分层（按最高R倍数归类）')
    rpt.add_text('R = MFE / 止损距离。每笔按60根内达到的最高R归入对应层级。')
    rpt.add_table(headers, rows, highlight_pnl_cols=[3])


def build_subtests(rpt, df_filled):
    """构建12个子测试的对比分析 — 用R倍数分布摘要"""
    base_row = tier_summary_row('Base（全部）', df_filled)
    rpt.add_section('子测试：R倍数分布对比')
    rpt.add_text('每行显示该子集的R倍数分布结构。重点看：亏损层占比↓、R3+占比↑、R3+MFE贡献↑')

    th = ['场景', 'N', 'avg MFE%', 'avg R', '亏损层(R<1)', 'R3+占比', 'R3+MFE贡献']

    # ===== 测试1：阳线 vs 阴线 =====
    rpt.add_section('测试1：信号K线方向（阳线 vs 阴线）')
    rpt.add_text('多头信号看阳线(close>open)，空头信号看阴线(close<open)')
    rows = [base_row]
    bull = df_filled[df_filled['is_bullish'] == True]
    bear = df_filled[df_filled['is_bullish'] == False]
    for label, sub in [('有利方向K线', bull), ('不利方向K线', bear)]:
        r = tier_summary_row(label, sub)
        if r: rows.append(r)
    rpt.add_table(th, rows, highlight_pnl_cols=[2])

    # ===== 测试2：实体占比 =====
    rpt.add_section('测试2：实体占比（body/range）')
    rpt.add_text('实体占比越小 = 影线越长')
    rows = [base_row]
    for thresh, label in [(0.5, '< 1/2（影线长）'), (1/3, '< 1/3（影线很长）'),
                          (0.5, '>= 1/2（实体大）')]:
        if label.startswith('<'):
            sub = df_filled[df_filled['body_ratio'] < thresh]
        else:
            sub = df_filled[df_filled['body_ratio'] >= thresh]
        r = tier_summary_row(label, sub)
        if r: rows.append(r)
    rpt.add_table(th, rows, highlight_pnl_cols=[2])

    # ===== 测试3：ER分档 =====
    rpt.add_section('测试3：ER(20) 分档')
    rpt.add_text('Base已过滤ER>0.3，这里看更高门槛和区间')
    rows = [base_row]
    for er_min in [0.4, 0.5, 0.6]:
        sub = df_filled[df_filled['er_20'] >= er_min]
        r = tier_summary_row(f'>={er_min}', sub)
        if r: rows.append(r)
    for er_lo, er_hi in [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.0)]:
        sub = df_filled[(df_filled['er_20'] >= er_lo) & (df_filled['er_20'] < er_hi)]
        r = tier_summary_row(f'{er_lo}~{er_hi}', sub)
        if r: rows.append(r)
    rpt.add_table(th, rows, highlight_pnl_cols=[2])

    # ===== 测试4：穿透EMA10深度 =====
    rpt.add_section('测试4：穿透EMA10深度（ATR单位）')
    rpt.add_text('影线扎入EMA10下方的深度')
    rows = [base_row]
    for lo, hi, label in [(0, 0.1, '0~0.1（刚碰）'),
                          (0.1, 0.3, '0.1~0.3（浅穿）'),
                          (0.3, 0.5, '0.3~0.5（中穿）'),
                          (0.5, 1.0, '0.5~1.0（深穿）'),
                          (1.0, 999, '>=1.0（很深）')]:
        sub = df_filled[(df_filled['penetration_atr'] >= lo) & (df_filled['penetration_atr'] < hi)]
        r = tier_summary_row(label, sub)
        if r: rows.append(r)
    rpt.add_table(th, rows, highlight_pnl_cols=[2])

    # ===== 测试5：第几次回踩 =====
    rpt.add_section('测试5：第几次回踩EMA10')
    rpt.add_text('趋势中第N次碰EMA10（touch>=6已硬过滤）')
    rows = [base_row]
    for seq in [1, 2, 3, 4, 5]:
        sub = df_filled[df_filled['touch_seq'] == seq]
        r = tier_summary_row(f'第{seq}次', sub)
        if r: rows.append(r)
    rpt.add_table(th, rows, highlight_pnl_cols=[2])

    # ===== 测试6：信号前涨幅 =====
    rpt.add_section('测试6：信号前20根涨幅（ATR单位）')
    rpt.add_text('回踩前的动能，数值越大=之前拉得越猛')
    rows = [base_row]
    pcts = df_filled['momentum_atr'].quantile([0.25, 0.5, 0.75]).values
    for lo, hi, label in [(0, pcts[0], f'P0~P25 (<{pcts[0]:.1f})'),
                          (pcts[0], pcts[1], f'P25~P50 ({pcts[0]:.1f}~{pcts[1]:.1f})'),
                          (pcts[1], pcts[2], f'P50~P75 ({pcts[1]:.1f}~{pcts[2]:.1f})'),
                          (pcts[2], 9999, f'P75+ (>={pcts[2]:.1f})')]:
        sub = df_filled[(df_filled['momentum_atr'] >= lo) & (df_filled['momentum_atr'] < hi)]
        r = tier_summary_row(label, sub)
        if r: rows.append(r)
    rpt.add_table(th, rows, highlight_pnl_cols=[2])

    # ===== 测试7：挂单成交速度 =====
    rpt.add_section('测试7：挂单成交速度（bars_to_fill）')
    rpt.add_text('1根就突破 vs 拖到4~5根，动能差异')
    rows = [base_row]
    for b in [1, 2, 3, 4, 5]:
        sub = df_filled[df_filled['bars_to_fill'] == b]
        r = tier_summary_row(f'{b}根成交', sub)
        if r: rows.append(r)
    rpt.add_table(th, rows, highlight_pnl_cols=[2])

    # ================================================================
    # 反向过滤子测试（测试8~12）
    # ================================================================

    # ===== 测试8：止损距离 / ATR =====
    rpt.add_section('测试8（反向）：止损距离（ATR单位）')
    rpt.add_text('止损距离 = |挂单价 - 止损价| / ATR。影线越长止损越远。')
    if 'stop_dist_atr' in df_filled.columns:
        rows = [base_row]
        for lo, hi, label in [(0, 0.5, '<0.5 ATR（紧止损）'),
                              (0.5, 1.0, '0.5~1.0 ATR'),
                              (1.0, 1.5, '1.0~1.5 ATR'),
                              (1.5, 2.0, '1.5~2.0 ATR'),
                              (2.0, 999, '>=2.0 ATR（远止损）')]:
            sub = df_filled[(df_filled['stop_dist_atr'] >= lo) & (df_filled['stop_dist_atr'] < hi)]
            r = tier_summary_row(label, sub)
            if r: rows.append(r)
        rpt.add_table(th, rows, highlight_pnl_cols=[2])
        q = df_filled['stop_dist_atr'].quantile([0.25, 0.5, 0.75, 0.9])
        rpt.add_text(f'止损距离分位: P25={q.iloc[0]:.2f} P50={q.iloc[1]:.2f} P75={q.iloc[2]:.2f} P90={q.iloc[3]:.2f}')

        # 详细分层对比：每个止损距离组 vs Base
        rpt.add_text('各止损距离组的完整R倍数分层：')
        build_quality_tiers_for_subset(rpt, df_filled, 'Base（全部）')
        for lo, hi, label in [(0.5, 1.0, '0.5~1.0 ATR'),
                              (1.0, 1.5, '1.0~1.5 ATR'),
                              (1.5, 2.0, '1.5~2.0 ATR'),
                              (2.0, 999, '>=2.0 ATR')]:
            sub = df_filled[(df_filled['stop_dist_atr'] >= lo) & (df_filled['stop_dist_atr'] < hi)]
            if len(sub) >= 20:
                build_quality_tiers_for_subset(rpt, sub, label)

    # ===== 测试9：信号时段 =====
    rpt.add_section('测试9（反向）：信号时段')
    rpt.add_text('按交易时段分组。重点看亏损层占比和R3+占比差异。')
    if 'signal_hour' in df_filled.columns and df_filled['signal_hour'].max() > 0:
        rows = [base_row]
        time_groups = [
            ('夜盘 21:00-23:59', [21, 22, 23]),
            ('夜盘 0:00-2:59', [0, 1, 2]),
            ('上午 9:00-11:29', [9, 10, 11]),
            ('下午 13:30-14:59', [13, 14]),
        ]
        for label, hour_list in time_groups:
            sub = df_filled[df_filled['signal_hour'].isin(hour_list)]
            r = tier_summary_row(label, sub)
            if r: rows.append(r)
        rpt.add_table(th, rows, highlight_pnl_cols=[2])

        # 按小时细分
        rpt.add_text('按小时细分：')
        hour_counts = df_filled['signal_hour'].value_counts().sort_index()
        rows2 = [base_row]
        for h in sorted(hour_counts.index):
            sub = df_filled[df_filled['signal_hour'] == h]
            if len(sub) >= 5:
                r = tier_summary_row(f'{h}:00', sub)
                if r: rows2.append(r)
        rpt.add_table(th, rows2, highlight_pnl_cols=[2])

    # ===== 测试10：ATR膨胀比 =====
    rpt.add_section('测试10（反向）：ATR膨胀比（ATR / ATR_MA20）')
    rpt.add_text('当前波动率相对近期均值。>1.5=波动剧增。')
    if 'atr_ratio' in df_filled.columns:
        rows = [base_row]
        for lo, hi, label in [(0, 0.8, '<0.8（低波）'),
                              (0.8, 1.0, '0.8~1.0（正常偏低）'),
                              (1.0, 1.2, '1.0~1.2（正常）'),
                              (1.2, 1.5, '1.2~1.5（偏高）'),
                              (1.5, 999, '>=1.5（高波）')]:
            sub = df_filled[(df_filled['atr_ratio'] >= lo) & (df_filled['atr_ratio'] < hi)]
            r = tier_summary_row(label, sub)
            if r: rows.append(r)
        rpt.add_table(th, rows, highlight_pnl_cols=[2])

    # ===== 测试11：信号密集度 =====
    rpt.add_section('测试11（反向）：信号密集度（最近10根内信号数）')
    rpt.add_text('密集=震荡，稀疏=干净回调。')
    if 'signal_density' in df_filled.columns:
        rows = [base_row]
        for d in sorted(df_filled['signal_density'].unique()):
            sub = df_filled[df_filled['signal_density'] == d]
            if len(sub) >= 5:
                label = f'密度={d}' if d < 3 else f'密度>={d}'
                r = tier_summary_row(label, sub)
                if r: rows.append(r)
        rpt.add_table(th, rows, highlight_pnl_cols=[2])
        # 二分
        sub_clean = df_filled[df_filled['signal_density'] == 0]
        sub_dense = df_filled[df_filled['signal_density'] >= 1]
        if len(sub_clean) > 0 and len(sub_dense) > 0:
            rows_bin = [base_row,
                        tier_summary_row('无前置(density=0)', sub_clean),
                        tier_summary_row('有前置(density>=1)', sub_dense)]
            rows_bin = [r for r in rows_bin if r]
            rpt.add_table(th, rows_bin, highlight_pnl_cols=[2])

    # ===== 测试12：日内方向 =====
    rpt.add_section('测试12（反向）：日内方向（(close-日开盘)/ATR）')
    rpt.add_text('做多+日内偏空=逆日内方向。看分布结构差异。')
    if 'intraday_dir' in df_filled.columns:
        rows = [base_row]
        for d_name, d_val in [('多头', 'long'), ('空头', 'short')]:
            sub_d = df_filled[df_filled['direction'] == d_val]
            if len(sub_d) < 10:
                continue
            if d_val == 'long':
                sub_follow = sub_d[sub_d['intraday_dir'] > 0]
                sub_against = sub_d[sub_d['intraday_dir'] <= 0]
            else:
                sub_follow = sub_d[sub_d['intraday_dir'] < 0]
                sub_against = sub_d[sub_d['intraday_dir'] >= 0]
            for label, sub in [(f'{d_name}+顺日内', sub_follow), (f'{d_name}+逆日内', sub_against)]:
                r = tier_summary_row(label, sub)
                if r: rows.append(r)
        rpt.add_table(th, rows, highlight_pnl_cols=[2])

        # 偏移强度分档
        rpt.add_text('日内偏移强度分档（不分多空）：')
        rows3 = [base_row]
        abs_dir = df_filled['intraday_dir'].abs()
        for lo, hi, label in [(0, 0.5, '|偏移|<0.5 ATR'),
                              (0.5, 1.0, '0.5~1.0 ATR'),
                              (1.0, 2.0, '1.0~2.0 ATR'),
                              (2.0, 999, '>=2.0 ATR')]:
            sub = df_filled[(abs_dir >= lo) & (abs_dir < hi)]
            r = tier_summary_row(label, sub)
            if r: rows3.append(r)
        rpt.add_table(th, rows3, highlight_pnl_cols=[2])

    # ================================================================
    # 第二轮新特征（测试13~16）
    # ================================================================

    # ===== 测试13：EMA展开度 =====
    if 'ema_spread' in df_filled.columns:
        rpt.add_section('测试13（新）：EMA展开度（(ema10-ema60)/ATR，顺势为正）')
        rpt.add_text('趋势展开程度。太小=EMA粘合趋势弱，太大=过度延伸。')
        rows = [base_row]
        pcts = df_filled['ema_spread'].quantile([0.25, 0.5, 0.75]).values
        for lo, hi, label in [(0, pcts[0], f'P0~P25 (<{pcts[0]:.1f})'),
                              (pcts[0], pcts[1], f'P25~P50 ({pcts[0]:.1f}~{pcts[1]:.1f})'),
                              (pcts[1], pcts[2], f'P50~P75 ({pcts[1]:.1f}~{pcts[2]:.1f})'),
                              (pcts[2], 9999, f'P75+ (>={pcts[2]:.1f})')]:
            sub = df_filled[(df_filled['ema_spread'] >= lo) & (df_filled['ema_spread'] < hi)]
            r = tier_summary_row(label, sub)
            if r: rows.append(r)
        rpt.add_table(th, rows, highlight_pnl_cols=[2])
        # 也看极端值
        rpt.add_text('极端值：')
        rows2 = [base_row]
        for lo, hi, label in [(0, 1, '<1 ATR（EMA粘合）'),
                              (1, 2, '1~2 ATR'),
                              (2, 3, '2~3 ATR'),
                              (3, 5, '3~5 ATR'),
                              (5, 999, '>=5 ATR（过度延伸）')]:
            sub = df_filled[(df_filled['ema_spread'] >= lo) & (df_filled['ema_spread'] < hi)]
            r = tier_summary_row(label, sub)
            if r: rows2.append(r)
        rpt.add_table(th, rows2, highlight_pnl_cols=[2])

    # ===== 测试14：回调前一根K线力度 =====
    if 'prev_bar_move' in df_filled.columns:
        rpt.add_section('测试14（新）：信号前一根K线变动（ATR单位，正=顺势）')
        rpt.add_text('前一根K线的变动：正=顺势方向，负=逆势（回调方向）。大负值=急跌回调。')
        rows = [base_row]
        pcts = df_filled['prev_bar_move'].quantile([0.25, 0.5, 0.75]).values
        for lo, hi, label in [(-999, pcts[0], f'P0~P25 (<{pcts[0]:.2f})'),
                              (pcts[0], pcts[1], f'P25~P50 ({pcts[0]:.2f}~{pcts[1]:.2f})'),
                              (pcts[1], pcts[2], f'P50~P75 ({pcts[1]:.2f}~{pcts[2]:.2f})'),
                              (pcts[2], 999, f'P75+ (>={pcts[2]:.2f})')]:
            sub = df_filled[(df_filled['prev_bar_move'] >= lo) & (df_filled['prev_bar_move'] < hi)]
            r = tier_summary_row(label, sub)
            if r: rows.append(r)
        rpt.add_table(th, rows, highlight_pnl_cols=[2])

    # ===== 测试15：星期几 =====
    if 'weekday' in df_filled.columns and df_filled['weekday'].max() >= 0:
        rpt.add_section('测试15（新）：星期几')
        rpt.add_text('周一=0 ... 周五=4。看哪天信号质量差。')
        rows = [base_row]
        day_names = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五'}
        for d in range(5):
            sub = df_filled[df_filled['weekday'] == d]
            r = tier_summary_row(day_names.get(d, str(d)), sub)
            if r: rows.append(r)
        rpt.add_table(th, rows, highlight_pnl_cols=[2])

    # ===== 第三轮新因子子测试 =====

    # ===== 测试16：ER(5)近6根变化 =====
    if 'er5_chg6' in df_filled.columns:
        rpt.add_section('测试16：ER(5)近6根变化（短期动量加速）')
        rpt.add_text('ER5变化：正=短期动量加速，负=减速。ABC系统中<=-0.41为危险，>=0.50为加仓。')
        rows = [base_row]
        for lo, hi, label in [(-999, -0.41, '<=-0.41（急减速）'),
                              (-0.41, -0.2, '-0.41~-0.2'),
                              (-0.2, 0, '-0.2~0'),
                              (0, 0.2, '0~0.2'),
                              (0.2, 0.5, '0.2~0.5'),
                              (0.5, 999, '>=0.50（急加速）')]:
            sub = df_filled[(df_filled['er5_chg6'] >= lo) & (df_filled['er5_chg6'] < hi)]
            r = tier_summary_row(label, sub)
            if r: rows.append(r)
        rpt.add_table(th, rows, highlight_pnl_cols=[2])

    # ===== 测试17：ER(40) =====
    if 'er_40' in df_filled.columns:
        rpt.add_section('测试17：ER(40)水平（中期趋势）')
        rpt.add_text('ER(40)>=0.42为ABC系统加仓级。')
        rows = [base_row]
        for lo, hi, label in [(0, 0.2, '<0.2'), (0.2, 0.3, '0.2~0.3'),
                              (0.3, 0.42, '0.3~0.42'), (0.42, 0.5, '0.42~0.5（加仓级）'),
                              (0.5, 999, '>=0.5（强趋势）')]:
            sub = df_filled[(df_filled['er_40'] >= lo) & (df_filled['er_40'] < hi)]
            r = tier_summary_row(label, sub)
            if r: rows.append(r)
        rpt.add_table(th, rows, highlight_pnl_cols=[2])

    # ===== 测试18：ER(40)近12根变化 =====
    if 'er40_chg12' in df_filled.columns:
        rpt.add_section('测试18：ER(40)近12根变化（中期趋势加速）')
        rpt.add_text('>=0.14为ABC系统加仓级。')
        rows = [base_row]
        for lo, hi, label in [(-999, -0.1, '<-0.1（趋势减弱）'),
                              (-0.1, 0, '-0.1~0'),
                              (0, 0.07, '0~0.07'),
                              (0.07, 0.14, '0.07~0.14'),
                              (0.14, 999, '>=0.14（加仓级）')]:
            sub = df_filled[(df_filled['er40_chg12'] >= lo) & (df_filled['er40_chg12'] < hi)]
            r = tier_summary_row(label, sub)
            if r: rows.append(r)
        rpt.add_table(th, rows, highlight_pnl_cols=[2])

    # ===== 测试19：RSI(14) =====
    if 'rsi_14' in df_filled.columns:
        rpt.add_section('测试19：RSI(14)')
        rpt.add_text('回调时RSI位置。<40=超卖区回踩，40~60=健康区，>70=超买区。')
        rows = [base_row]
        for lo, hi, label in [(0, 30, '<30（深度超卖）'), (30, 40, '30~40'),
                              (40, 50, '40~50'), (50, 60, '50~60'),
                              (60, 70, '60~70'), (70, 100, '>=70（超买）')]:
            sub = df_filled[(df_filled['rsi_14'] >= lo) & (df_filled['rsi_14'] < hi)]
            r = tier_summary_row(label, sub)
            if r: rows.append(r)
        rpt.add_table(th, rows, highlight_pnl_cols=[2])

    # ===== 测试20：成交量比 =====
    if 'vol_ratio' in df_filled.columns:
        rpt.add_section('测试20：成交量比（volume/vol_MA20）')
        rpt.add_text('当前成交量相对20根均量。ABC系统中样本外失效，R3+视角重测。')
        rows = [base_row]
        for lo, hi, label in [(0, 0.5, '<0.5（缩量）'), (0.5, 0.8, '0.5~0.8'),
                              (0.8, 1.2, '0.8~1.2（正常）'), (1.2, 2.0, '1.2~2.0（放量）'),
                              (2.0, 999, '>=2.0（巨量）')]:
            sub = df_filled[(df_filled['vol_ratio'] >= lo) & (df_filled['vol_ratio'] < hi)]
            r = tier_summary_row(label, sub)
            if r: rows.append(r)
        rpt.add_table(th, rows, highlight_pnl_cols=[2])

    # ===== 测试21：ADX(14) =====
    if 'adx_14' in df_filled.columns:
        rpt.add_section('测试21：ADX(14)（趋势强度）')
        rpt.add_text('ADX>25通常认为趋势明确。之前6品种测试4/6正向，R3+视角重测。')
        rows = [base_row]
        for lo, hi, label in [(0, 15, '<15（无趋势）'), (15, 25, '15~25'),
                              (25, 35, '25~35（趋势明确）'), (35, 50, '35~50（强趋势）'),
                              (50, 999, '>=50（极强）')]:
            sub = df_filled[(df_filled['adx_14'] >= lo) & (df_filled['adx_14'] < hi)]
            r = tier_summary_row(label, sub)
            if r: rows.append(r)
        rpt.add_table(th, rows, highlight_pnl_cols=[2])

    # ===== 测试22：R²(30) =====
    if 'r2_30' in df_filled.columns:
        rpt.add_section('测试22：R²(30)（趋势线性度）')
        rpt.add_text('最近30根close的线性拟合度。高=走势平滑，低=震荡。')
        rows = [base_row]
        for lo, hi, label in [(0, 0.3, '<0.3（震荡）'), (0.3, 0.5, '0.3~0.5'),
                              (0.5, 0.7, '0.5~0.7'), (0.7, 0.85, '0.7~0.85'),
                              (0.85, 1.01, '>=0.85（线性趋势）')]:
            sub = df_filled[(df_filled['r2_30'] >= lo) & (df_filled['r2_30'] < hi)]
            r = tier_summary_row(label, sub)
            if r: rows.append(r)
        rpt.add_table(th, rows, highlight_pnl_cols=[2])

    # ===== 测试23：EMA10斜率 =====
    if 'ma_slope' in df_filled.columns:
        rpt.add_section('测试23：EMA10斜率（5根变化/ATR，顺势为正）')
        rpt.add_text('EMA10的斜率。正=均线向上（多头），负=向下。')
        rows = [base_row]
        pcts = df_filled['ma_slope'].quantile([0.25, 0.5, 0.75]).values
        for lo, hi, label in [(-999, pcts[0], f'P0~P25 (<{pcts[0]:.3f})'),
                              (pcts[0], pcts[1], f'P25~P50'),
                              (pcts[1], pcts[2], f'P50~P75'),
                              (pcts[2], 999, f'P75+ (>={pcts[2]:.3f})')]:
            sub = df_filled[(df_filled['ma_slope'] >= lo) & (df_filled['ma_slope'] < hi)]
            r = tier_summary_row(label, sub)
            if r: rows.append(r)
        rpt.add_table(th, rows, highlight_pnl_cols=[2])

    # ===== 测试24：回调前连续顺势K线数 =====
    if 'consec_trend_bars' in df_filled.columns:
        rpt.add_section('测试24：回调前连续顺势K线数')
        rpt.add_text('回调前连续多少根阳线（多头）/阴线（空头）。越多=拉升越连贯。')
        rows = [base_row]
        for val in [0, 1, 2, 3, 4]:
            sub = df_filled[df_filled['consec_trend_bars'] == val]
            r = tier_summary_row(f'{val}根', sub)
            if r: rows.append(r)
        sub = df_filled[df_filled['consec_trend_bars'] >= 5]
        r = tier_summary_row('>=5根', sub)
        if r: rows.append(r)
        rpt.add_table(th, rows, highlight_pnl_cols=[2])

    # ===== 测试25：趋势年龄 =====
    if 'trend_age' in df_filled.columns:
        rpt.add_section('测试25：趋势年龄（根数）')
        rpt.add_text('从close上穿/下穿EMA60到现在的根数。越大=趋势越成熟。')
        rows = [base_row]
        for lo, hi, label in [(0, 20, '<20根（趋势初期）'),
                              (20, 50, '20~50根'),
                              (50, 100, '50~100根'),
                              (100, 200, '100~200根'),
                              (200, 9999, '>=200根（老趋势）')]:
            sub = df_filled[(df_filled['trend_age'] >= lo) & (df_filled['trend_age'] < hi)]
            r = tier_summary_row(label, sub)
            if r: rows.append(r)
        rpt.add_table(th, rows, highlight_pnl_cols=[2])


def build_signal_classification(rpt, df_filled):
    """因子组合触发制：穷举扫描 + 触发匹配验证"""
    stop_dist = (df_filled['entry_price'] - df_filled['stop_price']).abs()
    r_mult = (df_filled['mfe_raw'] / stop_dist).fillna(0)
    df = df_filled.copy()
    df['r_multiple'] = r_mult

    th = ['场景', 'N', 'avg MFE%', 'avg R', '亏损层(R<1)', 'R3+占比', 'R3+MFE贡献']
    base_row = tier_summary_row('Base（全部）', df)

    # 辅助：计算子集的R3+占比
    def r3_pct(sub):
        if len(sub) == 0:
            return 0
        sd = (sub['entry_price'] - sub['stop_price']).abs()
        rm = (sub['mfe_raw'] / sd).fillna(0)
        return (rm >= 3).sum() / len(sub) * 100

    # ============================================================
    # 因子组合穷举扫描
    # ============================================================
    rpt.add_section('因子组合触发系统：穷举扫描')
    rpt.add_text('扫描所有单因子和双因子组合，筛选R3+>25%的正向触发条件。')
    rpt.add_text('无硬过滤 — 命中任意正向组合→做，未命中→跳过。')

    # 构建因子条件池
    mom_p50 = df['momentum_atr'].quantile(0.5) if 'momentum_atr' in df.columns else 5.0
    mom_p75 = df['momentum_atr'].quantile(0.75) if 'momentum_atr' in df.columns else 6.5

    factor_conditions = []  # (name, category, mask)

    # A. 趋势强度
    factor_conditions.append(('ER20>=0.4', 'A', df['er_20'] >= 0.4))
    factor_conditions.append(('ER20>=0.5', 'A', df['er_20'] >= 0.5))
    if 'er_40' in df.columns:
        factor_conditions.append(('ER40>=0.3', 'A', df['er_40'] >= 0.3))
        factor_conditions.append(('ER40>=0.42', 'A', df['er_40'] >= 0.42))
    if 'ema_spread' in df.columns:
        factor_conditions.append(('EMA展开>=2', 'A', df['ema_spread'] >= 2))
        factor_conditions.append(('EMA展开<1', 'A', df['ema_spread'] < 1))
    if 'adx_14' in df.columns:
        factor_conditions.append(('ADX>=25', 'A', df['adx_14'] >= 25))
        factor_conditions.append(('ADX>=35', 'A', df['adx_14'] >= 35))

    # B. 趋势动量
    if 'er5_chg6' in df.columns:
        factor_conditions.append(('ER5变化>=0.3', 'B', df['er5_chg6'] >= 0.3))
        factor_conditions.append(('ER5变化>=0.5', 'B', df['er5_chg6'] >= 0.5))
        factor_conditions.append(('ER5变化<=-0.3', 'B', df['er5_chg6'] <= -0.3))
    if 'er40_chg12' in df.columns:
        factor_conditions.append(('ER40变化>=0.14', 'B', df['er40_chg12'] >= 0.14))
        factor_conditions.append(('ER40变化>=0.07', 'B', df['er40_chg12'] >= 0.07))
    factor_conditions.append(('动量>=P50', 'B', df['momentum_atr'] >= mom_p50))
    factor_conditions.append(('动量>=P75', 'B', df['momentum_atr'] >= mom_p75))
    if 'ma_slope' in df.columns:
        sl_p75 = df['ma_slope'].quantile(0.75)
        factor_conditions.append(('MA斜率>=P75', 'B', df['ma_slope'] >= sl_p75))

    # C. 回调质量
    factor_conditions.append(('stop<1.0ATR', 'C', df['stop_dist_atr'] < 1.0))
    factor_conditions.append(('stop<1.5ATR', 'C', df['stop_dist_atr'] < 1.5))
    factor_conditions.append(('stop>=2.0ATR', 'C', df['stop_dist_atr'] >= 2.0))
    factor_conditions.append(('density>=1', 'C', df['signal_density'] >= 1))
    factor_conditions.append(('density>=2', 'C', df['signal_density'] >= 2))
    factor_conditions.append(('density>=3', 'C', df['signal_density'] >= 3))
    factor_conditions.append(('touch>=2', 'C', df['touch_seq'] >= 2))
    factor_conditions.append(('touch>=4', 'C', df['touch_seq'] >= 4))
    factor_conditions.append(('touch<6', 'C', df['touch_seq'] < 6))
    if 'consec_trend_bars' in df.columns:
        factor_conditions.append(('连续顺势>=3', 'C', df['consec_trend_bars'] >= 3))
        factor_conditions.append(('连续顺势>=5', 'C', df['consec_trend_bars'] >= 5))

    # D. 波动率
    factor_conditions.append(('ATR比<0.8', 'D', df['atr_ratio'] < 0.8))
    if 'rsi_14' in df.columns:
        factor_conditions.append(('RSI 40~60', 'D', (df['rsi_14'] >= 40) & (df['rsi_14'] <= 60)))
        factor_conditions.append(('RSI>=50', 'D', df['rsi_14'] >= 50))
    if 'r2_30' in df.columns:
        factor_conditions.append(('R2>=0.5', 'D', df['r2_30'] >= 0.5))
        factor_conditions.append(('R2>=0.7', 'D', df['r2_30'] >= 0.7))

    # E. 成交量
    if 'vol_ratio' in df.columns:
        factor_conditions.append(('缩量<0.8', 'E', df['vol_ratio'] < 0.8))
        factor_conditions.append(('放量>=1.5', 'E', df['vol_ratio'] >= 1.5))

    # F. 时间/品种状态
    factor_conditions.append(('非周一', 'F', df['weekday'] != 0))
    factor_conditions.append(('周四', 'F', df['weekday'] == 3))
    factor_conditions.append(('非夜盘21-23', 'F', ~df['signal_hour'].isin([21, 22, 23])))
    if 'recent_win_n' in df.columns:
        factor_conditions.append(('热手>=3', 'F', df['recent_win_n'] >= 3))
        factor_conditions.append(('热手>=4', 'F', df['recent_win_n'] >= 4))
        factor_conditions.append(('热手=5', 'F', df['recent_win_n'] == 5))
    if 'trend_age' in df.columns:
        factor_conditions.append(('趋势>50根', 'F', df['trend_age'] >= 50))

    # ======== 单因子扫描 ========
    rpt.add_text(f'▸ 单因子扫描（共{len(factor_conditions)}个条件）：')
    single_results = []
    for name, cat, mask in factor_conditions:
        sub = df[mask]
        n = len(sub)
        if n < 10:
            continue
        r3 = r3_pct(sub)
        single_results.append((name, cat, n, r3, mask))

    # 按R3+排序显示
    single_results.sort(key=lambda x: -x[3])
    rows = [base_row]
    for name, cat, n, r3, mask in single_results:
        r = tier_summary_row(f'[{cat}] {name}', df[mask])
        if r: rows.append(r)
    rpt.add_table(th, rows, highlight_pnl_cols=[2])

    # ======== 双因子组合扫描 ========
    # 只组合R3+前15的因子，避免爆炸
    top_factors = [x for x in single_results if x[3] >= 20][:15]  # R3+>=20%的前15个
    rpt.add_text(f'▸ 双因子组合扫描（取R3+>=20%的前15个因子两两组合）：')

    combo_results = []
    for i_idx in range(len(top_factors)):
        for j_idx in range(i_idx + 1, len(top_factors)):
            n1, c1, _, r3_1, m1 = top_factors[i_idx]
            n2, c2, _, r3_2, m2 = top_factors[j_idx]
            # 跳过同类因子中明显包含关系的（如 ER20>=0.4 和 ER20>=0.5）
            combo_mask = m1 & m2
            sub = df[combo_mask]
            n = len(sub)
            if n < 20:
                continue
            r3 = r3_pct(sub)
            if r3 >= 25:  # 只保留R3+>25%的
                combo_results.append((f'{n1} + {n2}', n, r3, combo_mask))

    combo_results.sort(key=lambda x: -x[2])
    if combo_results:
        rows = [base_row]
        for name, n, r3, mask in combo_results[:30]:  # 最多显示30个
            r = tier_summary_row(name, df[mask])
            if r: rows.append(r)
        rpt.add_table(th, rows, highlight_pnl_cols=[2])
        rpt.add_text(f'共找到 {len(combo_results)} 个R3+>25%的双因子组合')
    else:
        rpt.add_text('未找到R3+>25%的双因子组合')

    # ======== 触发匹配验证 ========
    rpt.add_section('因子触发匹配验证')
    rpt.add_text('用所有R3+>25%的单因子和双因子组合作为触发条件。')
    rpt.add_text('信号命中任意一个→做，未命中任何→跳过。')

    # 收集所有R3+>25%的触发条件
    all_triggers = []
    for name, cat, n, r3, mask in single_results:
        if r3 >= 25 and n >= 30:
            all_triggers.append((f'[单]{name}', r3, mask))
    for name, n, r3, mask in combo_results:
        if n >= 20:
            all_triggers.append((f'[双]{name}', r3, mask))

    if all_triggers:
        # 计算每笔信号是否命中至少一个触发
        hit_any = pd.Series(False, index=df.index)
        best_r3 = pd.Series(0.0, index=df.index)
        for name, r3, mask in all_triggers:
            hit_any = hit_any | mask
            best_r3 = best_r3.where(~mask | (best_r3 >= r3), r3)

        df_hit = df[hit_any]
        df_miss = df[~hit_any]

        rpt.add_text(f'触发条件总数: {len(all_triggers)}')
        rpt.add_text(f'命中至少一个: {len(df_hit)} 笔 ({len(df_hit)/len(df)*100:.1f}%)')
        rpt.add_text(f'未命中任何: {len(df_miss)} 笔 ({len(df_miss)/len(df)*100:.1f}%)')

        rows = [
            base_row,
            tier_summary_row(f'命中组（{len(df_hit)}笔）', df_hit),
            tier_summary_row(f'未命中组（{len(df_miss)}笔）', df_miss),
        ]
        rows = [r for r in rows if r]
        rpt.add_table(th, rows, highlight_pnl_cols=[2])

        # 命中组按最高R3+分档
        rpt.add_text('▸ 命中组按最高匹配R3+分档：')
        df_hit_copy = df_hit.copy()
        df_hit_copy['best_r3'] = best_r3[hit_any].values
        rows = [tier_summary_row('命中组全部', df_hit)]
        for lo, hi, label in [(35, 100, 'R3+>=35%（博大）'),
                              (30, 35, 'R3+ 30~35%'),
                              (25, 30, 'R3+ 25~30%（标准）')]:
            sub = df_hit_copy[(df_hit_copy['best_r3'] >= lo) & (df_hit_copy['best_r3'] < hi)]
            r = tier_summary_row(label, sub)
            if r: rows.append(r)
        rpt.add_table(th, rows, highlight_pnl_cols=[2])

        # 详细分层对比
        build_quality_tiers_for_subset(rpt, df_hit, '命中组')
        build_quality_tiers_for_subset(rpt, df_miss, '未命中组')

        # 列出入围的触发条件清单
        rpt.add_text('▸ 入围触发条件清单（按R3+排序，前30）：')
        trigger_rows = []
        all_triggers.sort(key=lambda x: -x[1])
        for name, r3, mask in all_triggers[:30]:
            sub = df[mask]
            trigger_rows.append([name, len(sub), f'{r3:.1f}%'])
        rpt.add_table(['触发条件', 'N', 'R3+'], trigger_rows)

    # ============================================================
    # 多维度门槛分析
    # ============================================================
    rpt.add_section('多维度门槛分析：什么标准筛选触发条件？')
    rpt.add_text('一个因子/组合值得做，可以是因为R3+高、avgMFE高、亏损层低、或avgR高。')
    rpt.add_text('分别用不同维度做入围标准，看哪种效果最好。')

    # 收集所有单因子+双因子的统计
    all_candidates = []
    base_avg_mfe = df['mfe_margin'].mean()
    base_loss = (r_mult < 1).sum() / len(df) * 100
    base_avg_r = r_mult.mean()
    base_r3_val = r3_pct(df)

    for name, cat, n, r3_val, mask in single_results:
        if n >= 30:
            sub = df[mask]
            sd = (sub['entry_price'] - sub['stop_price']).abs()
            rm = (sub['mfe_raw'] / sd).fillna(0)
            all_candidates.append({
                'name': name, 'n': n, 'mask': mask,
                'r3': r3_val,
                'avg_mfe': sub['mfe_margin'].mean(),
                'loss_pct': (rm < 1).sum() / n * 100,
                'avg_r': rm.mean(),
            })
    for name, n, r3_val, mask in combo_results:
        if n >= 20:
            sub = df[mask]
            sd = (sub['entry_price'] - sub['stop_price']).abs()
            rm = (sub['mfe_raw'] / sd).fillna(0)
            all_candidates.append({
                'name': name, 'n': n, 'mask': mask,
                'r3': r3_val,
                'avg_mfe': sub['mfe_margin'].mean(),
                'loss_pct': (rm < 1).sum() / n * 100,
                'avg_r': rm.mean(),
            })

    rpt.add_text(f'Base: R3+={base_r3_val:.1f}% avgMFE={base_avg_mfe:.2f}% '
                 f'亏损层={base_loss:.1f}% avgR={base_avg_r:.2f}')
    rpt.add_text(f'候选因子/组合总数: {len(all_candidates)}')

    # 定义多种入围标准
    def eval_scheme(scheme_name, qualify_fn):
        """评估一种入围方案"""
        qualified = [c for c in all_candidates if qualify_fn(c)]
        if not qualified:
            return None
        hit = pd.Series(False, index=df.index)
        for c in qualified:
            hit = hit | c['mask']
        n_hit = hit.sum()
        n_miss = len(df) - n_hit
        if n_hit == 0 or n_miss == 0:
            return None
        r3_h = r3_pct(df[hit])
        r3_m = r3_pct(df[~hit])
        sd_h = (df[hit]['entry_price'] - df[hit]['stop_price']).abs()
        rm_h = (df[hit]['mfe_raw'] / sd_h).fillna(0)
        sd_m = (df[~hit]['entry_price'] - df[~hit]['stop_price']).abs()
        rm_m = (df[~hit]['mfe_raw'] / sd_m).fillna(0)
        return [
            scheme_name, len(qualified),
            f'{n_hit} ({n_hit/len(df)*100:.0f}%)',
            f'{n_miss} ({n_miss/len(df)*100:.0f}%)',
            f'{r3_h:.1f}%', f'{r3_m:.1f}%', f'{r3_h-r3_m:.1f}%',
            f'{(rm_h<1).sum()/n_hit*100:.1f}%', f'{(rm_m<1).sum()/n_miss*100:.1f}%',
            f'{df[hit]["mfe_margin"].mean():.2f}%', f'{df[~hit]["mfe_margin"].mean():.2f}%',
            f'{rm_h.mean():.2f}', f'{rm_m.mean():.2f}',
        ]

    schemes = [
        # 方案A：纯R3+门槛
        ('A1: R3+>=26%', lambda c: c['r3'] >= 26),
        ('A2: R3+>=28%', lambda c: c['r3'] >= 28),
        ('A3: R3+>=30%', lambda c: c['r3'] >= 30),
        ('A4: R3+>=32%', lambda c: c['r3'] >= 32),
        ('A5: R3+>=35%', lambda c: c['r3'] >= 35),
        # 方案B：纯亏损层门槛
        ('B1: 亏损层<48%', lambda c: c['loss_pct'] < 48),
        ('B2: 亏损层<46%', lambda c: c['loss_pct'] < 46),
        ('B3: 亏损层<44%', lambda c: c['loss_pct'] < 44),
        ('B4: 亏损层<42%', lambda c: c['loss_pct'] < 42),
        # 方案C：纯avgMFE门槛
        ('C1: avgMFE>=8%', lambda c: c['avg_mfe'] >= 8),
        ('C2: avgMFE>=9%', lambda c: c['avg_mfe'] >= 9),
        ('C3: avgMFE>=10%', lambda c: c['avg_mfe'] >= 10),
        ('C4: avgMFE>=12%', lambda c: c['avg_mfe'] >= 12),
        # 方案D：纯avgR门槛
        ('D1: avgR>=2.3', lambda c: c['avg_r'] >= 2.3),
        ('D2: avgR>=2.5', lambda c: c['avg_r'] >= 2.5),
        ('D3: avgR>=2.8', lambda c: c['avg_r'] >= 2.8),
        ('D4: avgR>=3.0', lambda c: c['avg_r'] >= 3.0),
        # 方案E：组合标准（满足任一即可）
        ('E1: R3+>=28 或 亏损<46', lambda c: c['r3'] >= 28 or c['loss_pct'] < 46),
        ('E2: R3+>=30 或 亏损<44', lambda c: c['r3'] >= 30 or c['loss_pct'] < 44),
        ('E3: R3+>=28 或 avgMFE>=9', lambda c: c['r3'] >= 28 or c['avg_mfe'] >= 9),
        ('E4: R3+>=30 或 亏损<46 或 avgR>=2.8', lambda c: c['r3'] >= 30 or c['loss_pct'] < 46 or c['avg_r'] >= 2.8),
        ('E5: avgR>=2.5 且 亏损<50', lambda c: c['avg_r'] >= 2.5 and c['loss_pct'] < 50),
    ]

    scheme_rows = []
    for name, fn in schemes:
        row = eval_scheme(name, fn)
        if row:
            scheme_rows.append(row)

    if scheme_rows:
        rpt.add_table(
            ['方案', '条件数', '命中笔数', '未命中笔数',
             '命中R3+', '未命中R3+', 'R3+差',
             '命中亏损层', '未命中亏损层',
             '命中avgMFE', '未命中avgMFE',
             '命中avgR', '未命中avgR'],
            scheme_rows
        )

    # 最优方案的详细分层对比
    rpt.add_text('▸ 各类最优方案详细对比：')
    best_schemes = [
        ('A3: R3+>=30%', lambda c: c['r3'] >= 30),
        ('B2: 亏损层<46%', lambda c: c['loss_pct'] < 46),
        ('C2: avgMFE>=9%', lambda c: c['avg_mfe'] >= 9),
        ('D2: avgR>=2.5', lambda c: c['avg_r'] >= 2.5),
        ('E4: 组合', lambda c: c['r3'] >= 30 or c['loss_pct'] < 46 or c['avg_r'] >= 2.8),
    ]
    for name, fn in best_schemes:
        qualified = [c for c in all_candidates if fn(c)]
        if not qualified:
            continue
        hit = pd.Series(False, index=df.index)
        for c in qualified:
            hit = hit | c['mask']
        if hit.sum() < 50:
            continue
        rows = [base_row,
                tier_summary_row(f'{name} 命中', df[hit]),
                tier_summary_row(f'{name} 未命中', df[~hit])]
        rows = [r for r in rows if r]
        rpt.add_table(th, rows, highlight_pnl_cols=[2])

    pass  # build_signal_classification end



def trim_analysis(values, pcts=TRIM_PCTS):
    """去极值分析：砍掉top N%后重算"""
    results = []
    v = np.sort(values)
    for pct in pcts:
        cut = max(1, int(len(v) * pct / 100))
        trimmed = v[:-cut]
        st = dist_stats(trimmed)
        st['trim_pct'] = pct
        st['cut_n'] = cut
        results.append(st)
    return results


# ============ 报告生成 ============

def build_report(df_all, df_filled, all_data):
    """生成完整HTML报告"""
    rpt = Report('强趋势回调 Type1（影线触碰）MFE 研究')

    n_signals = len(df_all)
    n_filled = len(df_filled)
    n_expired = len(df_all[df_all['outcome'] == 'expired'])
    n_gap = len(df_all[df_all['outcome'] == 'gap_skip'])
    n_replaced = len(df_all[df_all['outcome'] == 'replaced'])

    # ===== Section 1: 概览 =====
    rpt.add_section('1. 概览')
    rpt.add_text(f'品种: {len(all_data)} | 回测天数: {BACKTEST_DAYS} | 周期: 10min')
    rpt.add_text(f'参数: ER>{ER_THRESHOLD}, 挂单{PENDING_EXPIRY}根, 止损{STOP_TICKS}跳, MFE窗口{MFE_WINDOW}根, 保证金{MARGIN_RATE*100:.0f}%')
    rpt.add_text('无硬过滤 — 全因子组合触发制', color='#27AE60')
    rpt.add_text(f'总信号: {n_signals} | 成交: {n_filled} | '
                 f'过期: {n_expired} | 跳空跳过: {n_gap} | 被替换: {n_replaced}')

    fill_rate = n_filled / n_signals * 100 if n_signals > 0 else 0
    rpt.add_text(f'填单率: {fill_rate:.1f}%')

    if n_filled == 0:
        rpt.add_text('没有成交信号，无法生成分析。', color='red')
        return rpt

    # 按方向统计
    long_n = len(df_filled[df_filled['direction'] == 'long'])
    short_n = len(df_filled[df_filled['direction'] == 'short'])
    rpt.add_table(
        ['方向', '成交数', '占比'],
        [['多头', long_n, f'{long_n/n_filled*100:.1f}%'],
         ['空头', short_n, f'{short_n/n_filled*100:.1f}%']]
    )

    if n_filled < 30:
        rpt.add_text('⚠ 总样本不足30，以下结论需谨慎', color='#FFA500')

    # ===== EV 期望（MFE天花板） =====
    rpt.add_section('EV 期望（保证金%，每笔=MFE天花板）')
    rpt.add_text('PnL=MFE：假设每笔在最大有利点完美出场，衡量入场质量的理论天花板。出场策略另做。')
    pnl_all = df_filled['pnl_margin'].values
    mfe_atr_all = df_filled['mfe_atr'].values
    rpt.add_table(
        ['范围', 'N', 'avg MFE%', 'median%', 'P75%', 'avg ATR', 'median ATR', '累计%'],
        [['全部', n_filled, f'{np.mean(pnl_all):.2f}%', f'{np.median(pnl_all):.2f}%',
          f'{np.percentile(pnl_all, 75):.2f}%',
          f'{np.mean(mfe_atr_all):.2f}', f'{np.median(mfe_atr_all):.2f}',
          f'{np.sum(pnl_all):.1f}%']],
        highlight_pnl_cols=[2, 7]
    )

    # 按方向
    rows = []
    for d in ['long', 'short']:
        sub = df_filled[df_filled['direction'] == d]
        if len(sub) > 0:
            v = sub['pnl_margin'].values
            va = sub['mfe_atr'].values
            tag = '多头' if d == 'long' else '空头'
            rows.append([tag, len(sub), f'{np.mean(v):.2f}%', f'{np.median(v):.2f}%',
                        f'{np.percentile(v, 75):.2f}%',
                        f'{np.mean(va):.2f}', f'{np.median(va):.2f}',
                        f'{np.sum(v):.1f}%'])
    if rows:
        rpt.add_table(
            ['方向', 'N', 'avg MFE%', 'median%', 'P75%', 'avg ATR', 'median ATR', '累计%'],
            rows, highlight_pnl_cols=[2, 7]
        )

    # ===== Section 2: MFE 分布 =====
    rpt.add_section('2. MFE 分布')
    mfe_vals = df_filled['mfe_atr'].values
    mfe_margin_vals = df_filled['mfe_margin'].values
    st = dist_stats(mfe_vals)
    st_m = dist_stats(mfe_margin_vals)
    rpt.add_table(
        ['指标', 'avg', 'median', 'P25', 'P75', 'P90', 'std'],
        [['MFE(ATR)', st['avg'], st['med'], st['p25'], st['p75'], st['p90'], st['std']],
         ['MFE(保证金%)', st_m['avg'], st_m['med'], st_m['p25'], st_m['p75'], st_m['p90'], st_m['std']]]
    )

    # 按方向
    for d in ['long', 'short']:
        sub = df_filled[df_filled['direction'] == d]
        if len(sub) > 0:
            s = dist_stats(sub['mfe_atr'].values)
            sm = dist_stats(sub['mfe_margin'].values)
            tag = '多头' if d == 'long' else '空头'
            rpt.add_text(f'  {tag} (N={s["N"]}): ATR avg={s["avg"]} med={s["med"]} | 保证金% avg={sm["avg"]} med={sm["med"]}')

    # ===== Section 3: MAE 分布 =====
    rpt.add_section('3. MAE 分布（止损后停止追踪）')
    mae_vals = df_filled['mae_atr'].values
    mae_margin_vals = df_filled['mae_margin'].values
    st = dist_stats(mae_vals)
    st_m = dist_stats(mae_margin_vals)
    rpt.add_table(
        ['指标', 'avg', 'median', 'P25', 'P75', 'P90', 'std'],
        [['MAE(ATR)', st['avg'], st['med'], st['p25'], st['p75'], st['p90'], st['std']],
         ['MAE(保证金%)', st_m['avg'], st_m['med'], st_m['p25'], st_m['p75'], st_m['p90'], st_m['std']]]
    )

    # 止损金额参考
    sl_margin = df_filled['stop_loss_margin'].values
    st_sl = dist_stats(sl_margin)
    rpt.add_text(f'止损距离(保证金%): avg={st_sl["avg"]}% med={st_sl["med"]}%')

    for d in ['long', 'short']:
        sub = df_filled[df_filled['direction'] == d]
        if len(sub) > 0:
            s = dist_stats(sub['mae_atr'].values)
            sm = dist_stats(sub['mae_margin'].values)
            tag = '多头' if d == 'long' else '空头'
            rpt.add_text(f'  {tag} (N={s["N"]}): ATR avg={s["avg"]} med={s["med"]} | 保证金% avg={sm["avg"]} med={sm["med"]}')

    # ===== Section 4: MFE 到达时间 =====
    rpt.add_section('4. MFE/MAE 到达时间')
    mfe_bar_vals = df_filled['mfe_bar'].values
    mae_bar_vals = df_filled['mae_bar'].values
    st_mfe_bar = dist_stats(mfe_bar_vals)
    st_mae_bar = dist_stats(mae_bar_vals)
    rpt.add_table(
        ['指标', 'avg', 'median', 'P25', 'P75'],
        [['MFE峰值(bar)', st_mfe_bar['avg'], st_mfe_bar['med'], st_mfe_bar['p25'], st_mfe_bar['p75']],
         ['MAE谷值(bar)', st_mae_bar['avg'], st_mae_bar['med'], st_mae_bar['p25'], st_mae_bar['p75']]]
    )
    rpt.add_text(f'MFE通常在第{st_mfe_bar["med"]:.0f}根到达（约{st_mfe_bar["med"]*10:.0f}分钟）')

    # ===== Section 5: 止损命中率 =====
    rpt.add_section('5. 止损命中率')
    stop_mask = df_filled['stop_hit'].astype(bool)
    stop_total = stop_mask.sum()
    stop_rate = stop_total / n_filled * 100
    rpt.add_text(f'总体: {stop_total}/{n_filled} = {stop_rate:.1f}%')
    stopped = df_filled[stop_mask]
    if len(stopped) > 0:
        avg_stop_bar = stopped['stop_hit_bar'].mean()
        rpt.add_text(f'止损平均在第{avg_stop_bar:.1f}根触发（约{avg_stop_bar*10:.0f}分钟）')

    # 按方向
    rows = []
    for d in ['long', 'short']:
        sub = df_filled[df_filled['direction'] == d]
        if len(sub) > 0:
            tag = '多头' if d == 'long' else '空头'
            sr = sub['stop_hit'].astype(bool).sum() / len(sub) * 100
            rows.append([tag, len(sub), f'{sr:.1f}%'])
    if rows:
        rpt.add_table(['方向', 'N', '止损率'], rows)

    # 止损 vs 未止损的 MFE 对比
    not_stopped = df_filled[~stop_mask]
    if len(stopped) > 0 and len(not_stopped) > 0:
        rpt.add_text(f'止损组 MFE: ATR avg={stopped["mfe_atr"].mean():.3f}, 保证金% avg={stopped["mfe_margin"].mean():.2f}%')
        rpt.add_text(f'未止损组 MFE: ATR avg={not_stopped["mfe_atr"].mean():.3f}, 保证金% avg={not_stopped["mfe_margin"].mean():.2f}%')

    # ===== 交易质量分层 =====
    build_quality_tiers(rpt, df_filled)

    # ===== 子测试 =====
    build_subtests(rpt, df_filled)

    # ===== 信号分类分析 =====
    build_signal_classification(rpt, df_filled)

    # ===== Section 6: 稳健性检查 =====
    rpt.add_section('6. 稳健性检查')

    # 6a: 时间切分
    rpt.add_section('6a. 时间切分（前60天 vs 后60天）')
    rows = []
    for h in [1, 2]:
        sub = df_filled[df_filled['half'] == h]
        if len(sub) > 0:
            v = sub['pnl_margin'].values
            sr = sub['stop_hit'].astype(bool).sum() / len(sub) * 100
            label = '前半' if h == 1 else '后半'
            rows.append([label, len(sub), f'{np.mean(v):.2f}%', f'{np.median(v):.2f}%',
                        f'{np.percentile(v, 75):.2f}%', f'{sr:.1f}%'])
            if len(sub) < 30:
                rpt.add_text(f'⚠ {label}样本{len(sub)}笔 < 30', color='#FFA500')
    if rows:
        rpt.add_table(['时段', 'N', 'avg MFE%', 'median%', 'P75%', '止损率'], rows,
                     highlight_pnl_cols=[2])

    # 6b: 去极值
    rpt.add_section('6b. 去极值')
    full_st = dist_stats(mfe_vals)
    rows = [['不砍', full_st['N'], full_st['avg'], full_st['med'], '-']]
    for tr in trim_analysis(mfe_vals):
        rows.append([f'砍top {tr["trim_pct"]}%', tr['N'], tr['avg'], tr['med'],
                     f'砍了{tr["cut_n"]}笔'])
    rpt.add_table(['处理', 'N', 'MFE avg', 'MFE med', '说明'], rows)

    # 6c: 品种集中度
    rpt.add_section('6c. 品种集中度')
    sym_mfe = df_filled.groupby('symbol')['mfe_atr'].agg(['mean', 'count']).reset_index()
    sym_mfe = sym_mfe.sort_values('mean', ascending=False)
    top5_syms = sym_mfe.head(5)['symbol'].tolist()
    top5 = df_filled[df_filled['symbol'].isin(top5_syms)]
    rest = df_filled[~df_filled['symbol'].isin(top5_syms)]
    rows = []
    if len(top5) > 0:
        s = dist_stats(top5['mfe_atr'].values)
        rows.append(['Top5品种', s['N'], s['avg'], s['med']])
    if len(rest) > 0:
        s = dist_stats(rest['mfe_atr'].values)
        rows.append(['其余品种', s['N'], s['avg'], s['med']])
    if rows:
        rpt.add_table(['分组', 'N', 'MFE avg', 'MFE med'], rows)
    rpt.add_text(f'Top5品种: {", ".join([SYMBOL_CONFIGS.get(s, {}).get("name", s) for s in top5_syms])}')

    # 6d: 月度分解
    rpt.add_section('6d. 月度分解')
    if 'month' in df_filled.columns:
        rows = []
        for m, sub in df_filled.groupby('month', sort=True):
            v = sub['pnl_margin'].values
            sr = sub['stop_hit'].astype(bool).sum() / len(sub) * 100
            rows.append([m, len(sub), f'{np.mean(v):.2f}%', f'{np.median(v):.2f}%',
                        f'{np.percentile(v, 75):.2f}%', f'{sr:.1f}%'])
        if rows:
            rpt.add_table(['月份', 'N', 'avg MFE%', 'median%', 'P75%', '止损率'], rows,
                         highlight_pnl_cols=[2])

    # 6e: 多空对比
    rpt.add_section('6e. 多空对比')
    rows = []
    for d in ['long', 'short']:
        sub = df_filled[df_filled['direction'] == d]
        if len(sub) > 0:
            v = sub['pnl_margin'].values
            sr = sub['stop_hit'].astype(bool).sum() / len(sub) * 100
            tag = '多头' if d == 'long' else '空头'
            rows.append([tag, len(sub), f'{np.mean(v):.2f}%', f'{np.median(v):.2f}%',
                        f'{np.percentile(v, 75):.2f}%', f'{np.sum(v):.1f}%', f'{sr:.1f}%'])
    if rows:
        rpt.add_table(['方向', 'N', 'avg MFE%', 'median%', 'P75%', '累计%', '止损率'], rows,
                     highlight_pnl_cols=[2, 5])

    # 6f: 品种MFE分布
    rpt.add_section('6f. 品种MFE分布')
    rows = []
    sym_stats = []
    for sym, sub in df_filled.groupby('symbol'):
        v = sub['pnl_margin'].values
        name = SYMBOL_CONFIGS.get(sym, {}).get('name', sym)
        sym_stats.append({'name': name, 'N': len(v), 'avg': np.mean(v),
                         'med': np.median(v)})
    sym_stats.sort(key=lambda x: x['avg'], reverse=True)
    for s in sym_stats:
        warn = ' *' if s['N'] < 5 else ''
        rows.append([s['name'], f'{s["N"]}{warn}', f'{s["avg"]:.2f}%', f'{s["med"]:.2f}%'])
    if rows:
        rpt.add_table(['品种', 'N', 'avg MFE%', 'median%'], rows,
                     highlight_pnl_cols=[2])

    # 品种MFE概要
    avgs = [s['avg'] for s in sym_stats if s['N'] >= 5]
    if avgs:
        rpt.add_text(f'品种avg MFE（N>=5）: min={min(avgs):.2f}% max={max(avgs):.2f}% '
                     f'med={np.median(avgs):.2f}%')

    # ===== Section 7: K线图 =====
    rpt.add_section('7. 样本K线图')
    _add_charts(rpt, df_filled, all_data)

    return rpt


def _add_charts(rpt, df_filled, all_data):
    """添加K线图：最佳MFE / 最差MAE / 随机"""
    if len(df_filled) == 0:
        return

    # 最佳 MFE 10张
    rpt.add_section('7a. 最佳 MFE（Top 10）')
    best = df_filled.nlargest(min(10, len(df_filled)), 'mfe_atr')
    for _, row in best.iterrows():
        _render_one_chart(rpt, row, all_data)

    # 最差 MAE 10张
    rpt.add_section('7b. 最差 MAE（Top 10）')
    worst = df_filled.nlargest(min(10, len(df_filled)), 'mae_atr')
    for _, row in worst.iterrows():
        _render_one_chart(rpt, row, all_data)

    # 随机 10张
    rpt.add_section('7c. 随机样本（10张）')
    n_random = min(10, len(df_filled))
    random_sample = df_filled.sample(n=n_random, random_state=42)
    for _, row in random_sample.iterrows():
        _render_one_chart(rpt, row, all_data)


def _render_one_chart(rpt, row, all_data):
    """渲染一张K线图"""
    sym = row['symbol']
    if sym not in all_data:
        return
    df = all_data[sym]
    entry_idx = int(row['entry_bar_idx'])
    direction = row['direction']
    name = row.get('name', sym)

    # MFE 标注
    mfe_bar_idx = entry_idx + int(row['mfe_bar'])
    exits = [{'name': 'MFE', 'idx': mfe_bar_idx,
              'price': float(row['entry_price'] + row['mfe_raw']
                             if direction == 'long'
                             else row['entry_price'] - row['mfe_raw']),
              'color': '#FFD700'}]

    extra = {
        'MFE': f'{row["mfe_atr"]:.2f}ATR / {row["mfe_margin"]:.1f}%',
        'MAE': f'{row["mae_atr"]:.2f}ATR / {row["mae_margin"]:.1f}%',
        'ER': f'{row["er_20"]:.2f}',
        '止损': '触发' if row['stop_hit'] else '未触',
    }

    title = f'{name} {direction} | MFE={row["mfe_margin"]:.1f}%'

    chart_html = render_chart(
        df, entry_idx, direction,
        before_bars=20, after_bars=65,
        exits=exits,
        stop_price=float(row['stop_price']),
        ema_cols=['ema10', 'ema60'],
        title=title,
        extra_info=extra,
    )
    rpt.add_chart(chart_html)


# ============ Console 输出 ============

def print_summary(df_all, df_filled):
    """打印console摘要"""
    n_signals = len(df_all)
    n_filled = len(df_filled)
    n_expired = len(df_all[df_all['outcome'] == 'expired'])
    n_gap = len(df_all[df_all['outcome'] == 'gap_skip'])
    n_replaced = len(df_all[df_all['outcome'] == 'replaced'])

    print(f'\n{"="*60}')
    print(f'  强趋势回调 Type1（影线触碰）MFE 研究')
    print(f'{"="*60}')
    print(f'总信号: {n_signals} | 成交: {n_filled} | 过期: {n_expired} | '
          f'跳空: {n_gap} | 替换: {n_replaced}')

    if n_filled == 0:
        print('没有成交！')
        return

    long_n = len(df_filled[df_filled['direction'] == 'long'])
    short_n = len(df_filled[df_filled['direction'] == 'short'])
    print(f'多头: {long_n} | 空头: {short_n}')

    mfe = df_filled['mfe_atr']
    mae = df_filled['mae_atr']
    mfe_m = df_filled['mfe_margin']
    print(f'\nMFE (ATR): avg={mfe.mean():.3f}  med={mfe.median():.3f}  '
          f'P75={mfe.quantile(0.75):.3f}  P90={mfe.quantile(0.9):.3f}')
    print(f'MFE (保证金%): avg={mfe_m.mean():.2f}%  med={mfe_m.median():.2f}%  '
          f'P75={mfe_m.quantile(0.75):.2f}%  P90={mfe_m.quantile(0.9):.2f}%')
    print(f'MAE (ATR): avg={mae.mean():.3f}  med={mae.median():.3f}  '
          f'P75={mae.quantile(0.75):.3f}')
    print(f'MFE到达: avg={df_filled["mfe_bar"].mean():.0f} bars  '
          f'med={df_filled["mfe_bar"].median():.0f} bars')

    stop_rate = df_filled['stop_hit'].astype(bool).sum() / n_filled * 100
    print(f'止损命中: {stop_rate:.1f}%')

    # MFE天花板
    pnl = df_filled['pnl_margin'].values
    print(f'\nMFE天花板（保证金%）: avg={np.mean(pnl):.2f}% med={np.median(pnl):.2f}% '
          f'P75={np.percentile(pnl, 75):.2f}% cum={np.sum(pnl):.1f}%')

    # 时间切分
    for h in [1, 2]:
        sub = df_filled[df_filled['half'] == h]
        if len(sub) > 0:
            label = '前半' if h == 1 else '后半'
            print(f'  {label}: N={len(sub)} MFE avg={sub["mfe_atr"].mean():.3f}')

    print()


# ============ 主入口 ============

def main():
    df_all, df_filled, all_data = run_all()

    if len(df_filled) == 0:
        print("没有成交信号，结束。")
        return

    # Console 摘要
    print_summary(df_all, df_filled)

    # CSV
    os.makedirs('output', exist_ok=True)
    csv_path = 'output/trend_pullback_trades.csv'
    df_filled.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'CSV: {csv_path}')

    # HTML 报告
    rpt = build_report(df_all, df_filled, all_data)
    html_path = 'output/trend_pullback_report.html'
    rpt.save(html_path)
    print(f'HTML: {html_path}')


if __name__ == '__main__':
    main()
