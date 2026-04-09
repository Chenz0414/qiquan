# -*- coding: utf-8 -*-
"""
验证今日是否有信号：拉全品种10min K线，跑信号检测。
"""
import logging
from tqsdk import TqApi, TqAuth
from signal_core import (
    SignalDetector, SYMBOL_CONFIGS, DEFAULT_STOP_TICKS,
    classify_scenario, SCENARIO_EXIT,
)
from data_loader import add_indicators, sym_name

logging.basicConfig(level=logging.INFO, format='%(message)s')

api = TqApi(auth=TqAuth("bonjour0414", "zc950414"))

total = 0
for sym_key, cfg in SYMBOL_CONFIGS.items():
    tq_sym = f"KQ.m@{sym_key}"
    klines = api.get_kline_serial(tq_sym, 10 * 60, data_length=500)

api.wait_update()
print("数据就绪，开始回放...\n")

for sym_key, cfg in SYMBOL_CONFIGS.items():
    tq_sym = f"KQ.m@{sym_key}"
    serial = api.get_kline_serial(tq_sym, 10 * 60, data_length=500)

    import pandas as pd
    from datetime import datetime, timedelta

    df = serial.copy()
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ns')
    # 转北京时间
    df['datetime'] = df['datetime'] + timedelta(hours=8)
    df = df.dropna(subset=['close']).reset_index(drop=True)

    if len(df) < 150:
        continue

    df = add_indicators(df, emas=(10, 20, 120),
                        er_periods=(5, 20, 40), atr_period=14)
    if 'er_5' in df.columns:
        df['er5_delta_6'] = df['er_5'] - df['er_5'].shift(6)
    if 'er_40' in df.columns:
        df['er40_delta_12'] = df['er_40'] - df['er_40'].shift(12)

    name = sym_name(sym_key)
    detector = SignalDetector()

    # 预热到倒数50根
    warmup_end = max(130, len(df) - 50)
    for i in range(1, warmup_end):
        row = df.iloc[i]
        detector.process_bar(row['close'], row['high'], row['low'],
                             row['ema10'], row['ema20'], row['ema120'])

    # 回放最后50根，记录信号
    signals_found = []
    for i in range(warmup_end, len(df) - 1):
        row = df.iloc[i]
        signal = detector.process_bar(
            row['close'], row['high'], row['low'],
            row['ema10'], row['ema20'], row['ema120'])

        if signal is not None:
            er20 = float(row.get('er_20', 0) or 0)
            atr = float(row.get('atr', 0) or 0)
            if atr <= 0:
                continue
            dev = abs(signal.entry_price - row['ema10']) / atr
            scenario = classify_scenario(signal.signal_type, er20, dev)

            t = row['datetime'].strftime('%m-%d %H:%M')

            if scenario is not None:
                # 检查场景1的er5过滤
                er5d6 = float(row.get('er5_delta_6', 0) or 0)
                if scenario == 1 and er5d6 <= -0.41:
                    signals_found.append(
                        f"  {t} {signal.direction} {signal.signal_type} "
                        f"场景{scenario} ER={er20:.2f} dev={dev:.1f}ATR "
                        f"→ 被er5过滤(er5d6={er5d6:.2f})")
                else:
                    signals_found.append(
                        f"  {t} {signal.direction} {signal.signal_type} "
                        f"场景{scenario} ER={er20:.2f} dev={dev:.1f}ATR ✓ 有效信号")
                    total += 1
            else:
                signals_found.append(
                    f"  {t} {signal.direction} {signal.signal_type} "
                    f"ER={er20:.2f} dev={dev:.1f}ATR → 无场景(被过滤)")

    # 趋势状态
    last = df.iloc[-2]
    trend = "多" if last['ema20'] > last['ema120'] else "空"
    er20_val = last.get('er_20', 0) or 0

    if signals_found:
        print(f"【{name}】趋势={trend} ER20={er20_val:.2f}")
        for s in signals_found:
            print(s)
    else:
        # 也打印趋势状态供参考
        print(f"  {name}: 趋势={trend} ER20={er20_val:.2f} — 无信号")

print(f"\n{'='*50}")
print(f"总计有效信号: {total}")

api.close()
