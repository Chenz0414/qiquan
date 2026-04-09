# -*- coding: utf-8 -*-
"""
回测引擎：信号识别 + 3种出场方式并行模拟（S1/S2/S3）
用天勤TqSdk拉取历史K线，纯pandas离线计算
输出 trades.csv

趋势判断：EMA20 > EMA120
信号检测：B类回调(收盘跌破EMA10后收回)，回调>=4根
出场：S1当根新高追踪 + S2回调追踪 + S3前根新高追踪
"""

import os
import sys
import math
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from tqsdk import TqApi, TqAuth
import config as cfg
from signal_core import SignalDetector, ExitTracker


def fetch_klines(api, symbol, period_min, total_days):
    """拉取K线数据，返回DataFrame"""
    duration_seconds = period_min * 60
    bars_per_day = int(9.5 * 60 / period_min) + 5
    total_bars = min(bars_per_day * total_days, 8964)

    klines = api.get_kline_serial(symbol, duration_seconds=duration_seconds, data_length=total_bars)
    api.wait_update()

    df = klines.copy()
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ns')
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.dropna(subset=['close'])
    df = df.reset_index(drop=True)
    return df


def calc_indicators(df):
    """计算EMA和ATR"""
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema120'] = df['close'].ewm(span=120, adjust=False).mean()

    # ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=cfg.ATR_PERIOD).mean()

    # 趋势方向：EMA20 > EMA120 → 多头
    df['trend'] = 0
    df.loc[df['ema20'] > df['ema120'], 'trend'] = 1
    df.loc[df['ema20'] < df['ema120'], 'trend'] = -1

    return df


def detect_signals(df, period_min, all_klines_dict):
    """
    使用 SignalDetector 检测入场信号，返回信号列表。
    """
    signals = []
    n = len(df)
    if n < 130:  # EMA120需要足够预热
        return signals

    detector = SignalDetector(signal_types='ABC')

    start_idx = max(130, 2)  # EMA120预热

    for i in range(start_idx, n):
        row = df.iloc[i]

        if pd.isna(row['ema10']) or pd.isna(row['ema20']) or pd.isna(row['ema120']):
            continue
        if pd.isna(row['atr']) or row['atr'] <= 0:
            continue

        result = detector.process_bar(
            close=row['close'],
            high=row['high'],
            low=row['low'],
            ema10=row['ema10'],
            ema20=row['ema20'],
            ema120=row['ema120'],
        )

        if result is not None:
            sig = _build_signal(df, i, period_min, result, all_klines_dict)
            signals.append(sig)

    return signals


def _build_signal(df, idx, period_min, detected, all_klines_dict):
    """构建信号字典，包含课题字段"""
    row = df.iloc[idx]

    # 回调强度
    pullback_bars = detected.pullback_bars
    pullback_extreme = detected.pullback_extreme
    if pullback_bars > 0:
        pb_amplitude = abs(pullback_extreme - row['ema10']) / row['close'] * 100
        pullback_intensity = pb_amplitude / pullback_bars
    else:
        pullback_intensity = float('nan')

    # 距EMA120的ATR倍数
    dist_ema120 = abs(row['close'] - row['ema120']) / row['atr'] if row['atr'] > 0 else 0

    # 均线宽度
    ma_width_pct = abs(row['ema20'] - row['ema120']) / row['close'] * 100

    # 多周期共振
    tf_60min_aligned = None
    if period_min == 10:
        tf_60min_aligned = _check_alignment(all_klines_dict.get(60), row['datetime'], detected.direction)

    return {
        'period': f'{period_min}min',
        'direction': detected.direction,
        'entry_type': detected.signal_type,
        'entry_time': row['datetime'],
        'entry_price': detected.entry_price,
        'entry_idx': idx,
        'signal_seq': detected.signal_seq,
        'dist_ema120_atr': round(dist_ema120, 2),
        'pullback_bars': pullback_bars,
        'pullback_intensity': round(pullback_intensity, 4) if not math.isnan(pullback_intensity) else float('nan'),
        'tf_60min_aligned': tf_60min_aligned,
        'ma_width_pct': round(ma_width_pct, 4),
        'pullback_extreme': pullback_extreme,
    }


def _check_alignment(bigger_df, signal_time, direction):
    """检查大周期趋势是否与信号方向一致"""
    if bigger_df is None:
        return None
    mask = bigger_df['datetime'] <= signal_time
    if mask.sum() == 0:
        return None
    last_bar = bigger_df.loc[mask].iloc[-1]
    if pd.isna(last_bar.get('ema20')) or pd.isna(last_bar.get('ema120')):
        return None

    bigger_trend = 1 if last_bar['ema20'] > last_bar['ema120'] else -1
    expected = 1 if direction == 'long' else -1
    return bigger_trend == expected


def simulate_exits(df, signals, period_min):
    """
    对每个信号模拟3种出场方式（S1/S2/S3），全部使用ExitTracker。
    返回完整的交易记录列表。
    """
    trades = []
    n = len(df)

    for sig in signals:
        entry_idx = sig['entry_idx']
        entry_price = sig['entry_price']
        direction = sig['direction']
        pullback_extreme = sig['pullback_extreme']
        is_long = direction == 'long'

        trade = {
            'period': sig['period'],
            'direction': direction,
            'entry_type': sig['entry_type'],
            'entry_time': sig['entry_time'],
            'entry_price': entry_price,
            'signal_seq': sig['signal_seq'],
            'dist_ema120_atr': sig['dist_ema120_atr'],
            'pullback_bars': sig['pullback_bars'],
            'pullback_intensity': sig['pullback_intensity'],
            'tf_60min_aligned': sig['tf_60min_aligned'],
            'ma_width_pct': sig['ma_width_pct'],
        }

        # 使用 ExitTracker 处理 S1/S2/S3
        tracker = ExitTracker(
            direction=direction,
            entry_price=entry_price,
            pullback_extreme=pullback_extreme,
            tick_size=cfg.TICK_SIZE,
            stop_ticks=cfg.STOP_TICKS,
        )

        results = {'S1': None, 'S2': None, 'S3': None}

        for j in range(entry_idx + 1, n):
            bar = df.iloc[j]
            prev_bar = df.iloc[j - 1]

            if pd.isna(bar['ema10']):
                continue

            if not tracker.all_done():
                exit_events, _ = tracker.process_bar(
                    close=bar['close'],
                    high=bar['high'],
                    low=bar['low'],
                    ema10=bar['ema10'],
                    prev_close=prev_bar['close'],
                    prev_high=prev_bar['high'],
                    prev_low=prev_bar['low'],
                )
                for ev in exit_events:
                    if results[ev.strategy] is None:
                        results[ev.strategy] = {
                            'time': bar['datetime'],
                            'price': ev.exit_price,
                            'pnl': ev.pnl_pct,
                            'reason': ev.exit_reason,
                            'bars': ev.bars_held,
                        }

            if tracker.all_done():
                break

        # 回测结束强制平仓
        last_bar = df.iloc[-1]
        forced = tracker.force_close(last_bar['close'])
        for ev in forced:
            if results[ev.strategy] is None:
                results[ev.strategy] = {
                    'time': last_bar['datetime'],
                    'price': ev.exit_price,
                    'pnl': ev.pnl_pct,
                    'reason': ev.exit_reason,
                    'bars': ev.bars_held,
                }

        # 填充trade字典
        for s_name, prefix in [('S1', 'exit_s1'), ('S2', 'exit_s2'), ('S3', 'exit_s3')]:
            r = results[s_name]
            trade[f'{prefix}_time'] = r['time']
            trade[f'{prefix}_price'] = round(r['price'], 2)
            trade[f'{prefix}_pnl'] = round(r['pnl'], 4)
            trade[f'{prefix}_reason'] = r['reason']
            trade[f'{prefix}_bars'] = r['bars']

        trades.append(trade)

    return trades


def run():
    """主函数"""
    print("=" * 60)
    print(f"回测引擎启动 | 品种: {cfg.SYMBOL_NAME} | 周期: {cfg.PERIODS}")
    print(f"趋势: EMA20>EMA120 | 出场: S1/S2/S3")
    print(f"回测区间: {cfg.BACKTEST_DAYS}天 + {cfg.WARMUP_DAYS}天预热")
    print("=" * 60)

    api = TqApi(auth=TqAuth(cfg.TQ_ACCOUNT, cfg.TQ_PASSWORD))
    total_days = cfg.BACKTEST_DAYS + cfg.WARMUP_DAYS

    all_klines = {}
    print("\n[1/3] 拉取K线数据...")
    for p in cfg.PERIODS:
        print(f"  拉取 {p}min K线...", end=" ")
        df = fetch_klines(api, cfg.SYMBOL, p, total_days)
        df = calc_indicators(df)
        all_klines[p] = df
        print(f"共 {len(df)} 根, 时间范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

    ref_df = all_klines[cfg.PERIODS[-1]]
    warmup_bars_ref = int(cfg.WARMUP_DAYS * 9.5 * 60 / cfg.PERIODS[-1])
    if warmup_bars_ref < len(ref_df):
        buy_hold_start = ref_df.iloc[warmup_bars_ref]['close']
        buy_hold_end = ref_df.iloc[-1]['close']
        buy_hold_return = (buy_hold_end - buy_hold_start) / buy_hold_start * 100
        print(f"\n  买入持有基准: {buy_hold_start} -> {buy_hold_end} = {buy_hold_return:.2f}%")
    else:
        buy_hold_return = 0

    api.close()

    all_trades = []
    print("\n[2/3] 检测信号 & 模拟出场...")
    for p in cfg.PERIODS:
        print(f"  {p}min...", end=" ")
        signals = detect_signals(all_klines[p], p, all_klines)
        print(f"发现 {len(signals)} 个信号...", end=" ")
        if signals:
            trades = simulate_exits(all_klines[p], signals, p)
            all_trades.extend(trades)
            print(f"完成 {len(trades)} 笔交易")
        else:
            print("无信号")

    print(f"\n[3/3] 输出CSV...")
    os.makedirs(os.path.join(os.path.dirname(__file__), cfg.OUTPUT_DIR), exist_ok=True)
    csv_path = os.path.join(os.path.dirname(__file__), cfg.OUTPUT_DIR, cfg.CSV_FILENAME)

    if all_trades:
        df_trades = pd.DataFrame(all_trades)
        df_trades.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  已输出 {len(df_trades)} 笔交易到 {csv_path}")

        meta_path = os.path.join(os.path.dirname(__file__), cfg.OUTPUT_DIR, 'meta.csv')
        pd.DataFrame([{'buy_hold_return': round(buy_hold_return, 4),
                        'symbol': cfg.SYMBOL_NAME,
                        'backtest_days': cfg.BACKTEST_DAYS}]).to_csv(meta_path, index=False)

        print("\n" + "=" * 60)
        print("简要统计：")
        for p in cfg.PERIODS:
            subset = df_trades[df_trades['period'] == f'{p}min']
            if len(subset) > 0:
                print(f"  {p}min: {len(subset)}笔 | "
                      f"S1累计:{subset['exit_s1_pnl'].sum():.2f}% | "
                      f"S2累计:{subset['exit_s2_pnl'].sum():.2f}% | "
                      f"S3累计:{subset['exit_s3_pnl'].sum():.2f}%")
        print(f"  买入持有基准: {buy_hold_return:.2f}%")
    else:
        print("  未检测到任何信号！")

    print("\n回测完成。")


if __name__ == '__main__':
    run()
