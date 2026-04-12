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

    # ATR 20周期均值（用于ATR膨胀比计算）
    atr_ma20 = pd.Series(atr).rolling(20, min_periods=20).mean().values

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
        er_periods=(20,),
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

    # === 反向过滤：touch>=6 剔除 ===
    if len(df_filled) > 0 and 'touch_seq' in df_filled.columns:
        before_n = len(df_filled)
        df_filled = df_filled[df_filled['touch_seq'] < 6].copy()
        cut_n = before_n - len(df_filled)
        print(f'  反向过滤 touch>=6: 剔除 {cut_n} 笔, 剩余 {len(df_filled)} 笔')

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
    """构建7个子测试的对比分析"""
    base_pnl = df_filled['pnl_margin'].tolist()
    base_ev = calc_ev(base_pnl)

    base_avg = np.mean(base_pnl)
    rpt.add_section('子测试：场景过滤对比（vs Base avg={:.2f}%）'.format(base_avg))
    rpt.add_text(f'Base: N={len(base_pnl)} avg MFE={base_avg:.2f}% '
                 f'median={np.median(base_pnl):.2f}%')

    headers = ['场景', 'N', 'avg MFE', 'median', 'P75', '累计']

    # ===== 测试1：阳线 vs 阴线 =====
    rpt.add_section('测试1：信号K线方向（阳线 vs 阴线）')
    rpt.add_text('多头信号看阳线(close>open)，空头信号看阴线(close<open)')
    rows = [ev_row('Base（全部）', base_pnl)]
    bull = df_filled[df_filled['is_bullish'] == True]
    bear = df_filled[df_filled['is_bullish'] == False]
    if len(bull) > 0:
        rows.append(ev_row('有利方向K线', bull['pnl_margin'].tolist()))
    if len(bear) > 0:
        rows.append(ev_row('不利方向K线', bear['pnl_margin'].tolist()))
    rpt.add_table(headers, rows, highlight_pnl_cols=[2, 5])

    # ===== 测试2：实体占比 =====
    rpt.add_section('测试2：实体占比（body/range）')
    rpt.add_text('实体占比越小 = 影线越长。< 1/2 和 < 1/3 分别测试')
    rows = [ev_row('Base（全部）', base_pnl)]
    for thresh, label in [(0.5, '< 1/2（影线长）'), (1/3, '< 1/3（影线很长）'),
                          (0.5, '>= 1/2（实体大）')]:
        if label.startswith('<'):
            sub = df_filled[df_filled['body_ratio'] < thresh]
        else:
            sub = df_filled[df_filled['body_ratio'] >= thresh]
        if len(sub) > 0:
            rows.append(ev_row(label, sub['pnl_margin'].tolist()))
    rpt.add_table(headers, rows, highlight_pnl_cols=[2, 5])

    # ===== 测试3：ER分档 =====
    rpt.add_section('测试3：ER(20) 分档')
    rpt.add_text('Base已过滤ER>0.3，这里看更高门槛的效果')
    rows = [ev_row('Base (>0.3)', base_pnl)]
    for er_min in [0.4, 0.5, 0.6]:
        sub = df_filled[df_filled['er_20'] >= er_min]
        if len(sub) > 0:
            rows.append(ev_row(f'>={er_min}', sub['pnl_margin'].tolist()))
    # 也看区间
    for er_lo, er_hi in [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.0)]:
        sub = df_filled[(df_filled['er_20'] >= er_lo) & (df_filled['er_20'] < er_hi)]
        if len(sub) > 0:
            rows.append(ev_row(f'{er_lo}~{er_hi}', sub['pnl_margin'].tolist()))
    rpt.add_table(headers, rows, highlight_pnl_cols=[2, 5])

    # ===== 测试4：穿透EMA10深度 =====
    rpt.add_section('测试4：穿透EMA10深度（ATR单位）')
    rpt.add_text('影线扎入EMA10下方的深度，越深=回踩越深')
    rows = [ev_row('Base（全部）', base_pnl)]
    for lo, hi, label in [(0, 0.1, '0~0.1（刚碰）'),
                          (0.1, 0.3, '0.1~0.3（浅穿）'),
                          (0.3, 0.5, '0.3~0.5（中穿）'),
                          (0.5, 1.0, '0.5~1.0（深穿）'),
                          (1.0, 999, '>=1.0（很深）')]:
        sub = df_filled[(df_filled['penetration_atr'] >= lo) & (df_filled['penetration_atr'] < hi)]
        if len(sub) > 0:
            rows.append(ev_row(label, sub['pnl_margin'].tolist()))
    rpt.add_table(headers, rows, highlight_pnl_cols=[2, 5])

    # ===== 测试5：第几次回踩 =====
    rpt.add_section('测试5：第几次回踩EMA10')
    rpt.add_text('趋势中第N次碰EMA10，越后面趋势可能越衰竭')
    rows = [ev_row('Base（全部）', base_pnl)]
    for seq in [1, 2, 3, 4, 5]:
        sub = df_filled[df_filled['touch_seq'] == seq]
        if len(sub) > 0:
            rows.append(ev_row(f'第{seq}次', sub['pnl_margin'].tolist()))
    sub_6plus = df_filled[df_filled['touch_seq'] >= 6]
    if len(sub_6plus) > 0:
        rows.append(ev_row('>=6次', sub_6plus['pnl_margin'].tolist()))
    rpt.add_table(headers, rows, highlight_pnl_cols=[2, 5])

    # ===== 测试6：信号前涨幅 =====
    rpt.add_section('测试6：信号前20根涨幅（ATR单位）')
    rpt.add_text('回踩前的动能，数值越大=之前拉得越猛')
    rows = [ev_row('Base（全部）', base_pnl)]
    pcts = df_filled['momentum_atr'].quantile([0.25, 0.5, 0.75]).values
    for lo, hi, label in [(0, pcts[0], f'P0~P25 (<{pcts[0]:.1f})'),
                          (pcts[0], pcts[1], f'P25~P50 ({pcts[0]:.1f}~{pcts[1]:.1f})'),
                          (pcts[1], pcts[2], f'P50~P75 ({pcts[1]:.1f}~{pcts[2]:.1f})'),
                          (pcts[2], 9999, f'P75+ (>={pcts[2]:.1f})')]:
        sub = df_filled[(df_filled['momentum_atr'] >= lo) & (df_filled['momentum_atr'] < hi)]
        if len(sub) > 0:
            rows.append(ev_row(label, sub['pnl_margin'].tolist()))
    rpt.add_table(headers, rows, highlight_pnl_cols=[2, 5])

    # ===== 测试7：挂单成交速度 =====
    rpt.add_section('测试7：挂单成交速度（bars_to_fill）')
    rpt.add_text('1根就突破 vs 拖到4~5根，动能差异')
    rows = [ev_row('Base（全部）', base_pnl)]
    for b in [1, 2, 3, 4, 5]:
        sub = df_filled[df_filled['bars_to_fill'] == b]
        if len(sub) > 0:
            rows.append(ev_row(f'{b}根成交', sub['pnl_margin'].tolist()))
    rpt.add_table(headers, rows, highlight_pnl_cols=[2, 5])

    # ================================================================
    # 反向过滤子测试（测试8~12）
    # ================================================================

    # ===== 测试8：止损距离 / ATR =====
    rpt.add_section('测试8（反向）：止损距离（ATR单位）')
    rpt.add_text('止损距离 = |挂单价 - 止损价| / ATR。影线越长止损越远，风险敞口越大。')
    if 'stop_dist_atr' in df_filled.columns:
        rows = [ev_row('Base（全部）', base_pnl)]
        for lo, hi, label in [(0, 0.5, '<0.5 ATR（紧止损）'),
                              (0.5, 1.0, '0.5~1.0 ATR'),
                              (1.0, 1.5, '1.0~1.5 ATR'),
                              (1.5, 2.0, '1.5~2.0 ATR'),
                              (2.0, 999, '>=2.0 ATR（远止损）')]:
            sub = df_filled[(df_filled['stop_dist_atr'] >= lo) & (df_filled['stop_dist_atr'] < hi)]
            if len(sub) > 0:
                rows.append(ev_row(label, sub['pnl_margin'].tolist()))
        rpt.add_table(headers, rows, highlight_pnl_cols=[2, 5])
        # 补充分位数参考
        q = df_filled['stop_dist_atr'].quantile([0.25, 0.5, 0.75, 0.9])
        rpt.add_text(f'止损距离分位: P25={q.iloc[0]:.2f} P50={q.iloc[1]:.2f} P75={q.iloc[2]:.2f} P90={q.iloc[3]:.2f}')

    # ===== 测试9：信号时段 =====
    rpt.add_section('测试9（反向）：信号时段')
    rpt.add_text('按信号K线所在小时分档。关注尾盘（收盘前1小时）是否表现差。')
    if 'signal_hour' in df_filled.columns and df_filled['signal_hour'].max() > 0:
        rows = [ev_row('Base（全部）', base_pnl)]
        # 先按每个小时统计样本量，只显示有数据的时段
        hour_counts = df_filled['signal_hour'].value_counts().sort_index()
        # 按交易时段分组：夜盘(21-23, 0-2)、上午(9-11)、下午(13-15)
        time_groups = [
            ('夜盘 21:00-23:59', [21, 22, 23]),
            ('夜盘 0:00-2:59', [0, 1, 2]),
            ('上午 9:00-11:29', [9, 10, 11]),
            ('下午 13:30-14:59', [13, 14]),
        ]
        for label, hour_list in time_groups:
            sub = df_filled[df_filled['signal_hour'].isin(hour_list)]
            if len(sub) > 0:
                rows.append(ev_row(f'{label} (N={len(sub)})', sub['pnl_margin'].tolist()))
        # 也单独看每个小时
        rpt.add_table(headers, rows, highlight_pnl_cols=[2, 5])

        rpt.add_text('按小时细分：')
        rows2 = [ev_row('Base', base_pnl)]
        for h in sorted(hour_counts.index):
            sub = df_filled[df_filled['signal_hour'] == h]
            if len(sub) >= 5:
                rows2.append(ev_row(f'{h}:00', sub['pnl_margin'].tolist()))
        rpt.add_table(headers, rows2, highlight_pnl_cols=[2, 5])

    # ===== 测试10：ATR膨胀比 =====
    rpt.add_section('测试10（反向）：ATR膨胀比（ATR / ATR_MA20）')
    rpt.add_text('当前波动率相对近期均值。>1.5=波动剧增，可能方向不明。')
    if 'atr_ratio' in df_filled.columns:
        rows = [ev_row('Base（全部）', base_pnl)]
        for lo, hi, label in [(0, 0.8, '<0.8（低波）'),
                              (0.8, 1.0, '0.8~1.0（正常偏低）'),
                              (1.0, 1.2, '1.0~1.2（正常）'),
                              (1.2, 1.5, '1.2~1.5（偏高）'),
                              (1.5, 999, '>=1.5（高波）')]:
            sub = df_filled[(df_filled['atr_ratio'] >= lo) & (df_filled['atr_ratio'] < hi)]
            if len(sub) > 0:
                rows.append(ev_row(label, sub['pnl_margin'].tolist()))
        rpt.add_table(headers, rows, highlight_pnl_cols=[2, 5])

    # ===== 测试11：信号密集度 =====
    rpt.add_section('测试11（反向）：信号密集度（最近10根内信号数）')
    rpt.add_text('最近10根K线内出过几次Type1信号（含当前）。密集=震荡，稀疏=干净回调。')
    if 'signal_density' in df_filled.columns:
        rows = [ev_row('Base（全部）', base_pnl)]
        for d in sorted(df_filled['signal_density'].unique()):
            sub = df_filled[df_filled['signal_density'] == d]
            if len(sub) >= 5:
                label = f'密度={d}' if d < 3 else f'密度>={d}'
                rows.append(ev_row(label, sub['pnl_margin'].tolist()))
        # 也做二分：0 vs >=1
        sub_clean = df_filled[df_filled['signal_density'] == 0]
        sub_dense = df_filled[df_filled['signal_density'] >= 1]
        if len(sub_clean) > 0 and len(sub_dense) > 0:
            rpt.add_text(f'无前置信号(density=0): N={len(sub_clean)}')
            rpt.add_text(f'有前置信号(density>=1): N={len(sub_dense)}')
        rpt.add_table(headers, rows, highlight_pnl_cols=[2, 5])

    # ===== 测试12：日内方向 =====
    rpt.add_section('测试12（反向）：日内方向（(close-日开盘)/ATR）')
    rpt.add_text('信号时刻 close vs 当日开盘价，正=日内偏多，负=日内偏空。'
                 '做多信号+日内偏空=逆日内方向，可能表现差。')
    if 'intraday_dir' in df_filled.columns:
        rows = [ev_row('Base（全部）', base_pnl)]
        # 按方向分别看：做多时日内方向正负，做空时日内方向正负
        for d_name, d_val in [('多头', 'long'), ('空头', 'short')]:
            sub_d = df_filled[df_filled['direction'] == d_val]
            if len(sub_d) < 10:
                continue
            # 对做多：intraday_dir > 0 = 顺势，< 0 = 逆势
            # 对做空：intraday_dir < 0 = 顺势，> 0 = 逆势
            if d_val == 'long':
                sub_follow = sub_d[sub_d['intraday_dir'] > 0]
                sub_against = sub_d[sub_d['intraday_dir'] <= 0]
            else:
                sub_follow = sub_d[sub_d['intraday_dir'] < 0]
                sub_against = sub_d[sub_d['intraday_dir'] >= 0]
            if len(sub_follow) > 0:
                rows.append(ev_row(f'{d_name}+顺日内', sub_follow['pnl_margin'].tolist()))
            if len(sub_against) > 0:
                rows.append(ev_row(f'{d_name}+逆日内', sub_against['pnl_margin'].tolist()))
        rpt.add_table(headers, rows, highlight_pnl_cols=[2, 5])

        # 也按ATR单位分档看强度
        rpt.add_text('日内偏移强度分档（不分多空）：')
        rows3 = [ev_row('Base', base_pnl)]
        abs_dir = df_filled['intraday_dir'].abs()
        for lo, hi, label in [(0, 0.5, '|偏移|<0.5 ATR'),
                              (0.5, 1.0, '0.5~1.0 ATR'),
                              (1.0, 2.0, '1.0~2.0 ATR'),
                              (2.0, 999, '>=2.0 ATR')]:
            sub = df_filled[(abs_dir >= lo) & (abs_dir < hi)]
            if len(sub) > 0:
                rows3.append(ev_row(label, sub['pnl_margin'].tolist()))
        rpt.add_table(headers, rows3, highlight_pnl_cols=[2, 5])


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
    rpt.add_text('已启用反向过滤: touch>=6 剔除', color='#E67E22')
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
