# -*- coding: utf-8 -*-
"""
K线数据本地缓存
===============
拉取过的数据存到 data_cache/ 目录，下次直接读本地。
盘中数据有更新时间戳判断，盘后数据直接用缓存。

用法:
    from data_cache import get_klines

    api = TqApi(...)
    df = get_klines(api, "KQ.m@SHFE.ag", "SHFE.ag", period_min=10, days=170)
    api.close()
"""

import os
import time
import pandas as pd
import config as cfg

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# 缓存永久保留，不自动过期
# 用户可选择用缓存还是重新拉取
CACHE_TTL = float('inf')


def _cache_path(symbol_key, period_min, days):
    """缓存文件路径"""
    safe_name = symbol_key.replace(".", "_")
    return os.path.join(CACHE_DIR, f"{safe_name}_{period_min}min_{days}d.parquet")


def _is_cache_valid(path):
    """缓存是否存在"""
    return os.path.exists(path)


def get_klines(api, tq_symbol, symbol_key, period_min=10, days=170, force_refresh=False):
    """
    获取K线数据，优先用缓存。

    参数:
        api: TqApi 实例
        tq_symbol: 天勤合约代码, e.g. "KQ.m@SHFE.ag"
        symbol_key: 品种代码, e.g. "SHFE.ag"
        period_min: K线周期(分钟)
        days: 请求天数
        force_refresh: 强制刷新缓存

    返回:
        DataFrame: 天勤全部原始字段 + ma_fast(MA10) + ma_slow(MA20)
        columns=[datetime, open, high, low, close, volume, open_oi, close_oi, ma_fast, ma_slow]
    """
    path = _cache_path(symbol_key, period_min, days)

    # 尝试读缓存
    if not force_refresh and _is_cache_valid(path):
        try:
            df = pd.read_parquet(path)
            mtime = os.path.getmtime(path)
            age = time.time() - mtime
            if age < 3600:
                age_str = f"{age/60:.0f}分钟前"
            elif age < 86400:
                age_str = f"{age/3600:.1f}小时前"
            else:
                age_str = f"{age/86400:.1f}天前"
            print(f"  [缓存] 读取 {symbol_key} ({len(df)}根K线, {age_str}缓存)")
            return df
        except Exception:
            pass  # 缓存损坏，重新拉取

    # 从天勤拉取
    bars_per_day = int(9.5 * 60 / period_min) + 5
    total_bars = min(bars_per_day * days, 8964)

    print(f"  [天勤] 拉取 {symbol_key} ({tq_symbol}), {total_bars} bars ...")
    klines = api.get_kline_serial(tq_symbol, duration_seconds=period_min * 60, data_length=total_bars)

    deadline = time.time() + 60
    while True:
        api.wait_update(deadline=time.time() + 5)
        if len(klines) > 0 and not pd.isna(klines.iloc[-1]['close']):
            break
        if time.time() > deadline:
            print(f"  WARNING: wait_update timeout for {symbol_key}")
            break

    df = klines.copy()
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ns')

    # 保留天勤返回的全部有用字段
    raw_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'open_oi', 'close_oi']
    keep_cols = [c for c in raw_cols if c in df.columns]
    df = df[keep_cols].copy()
    df = df.dropna(subset=['close']).reset_index(drop=True)

    # 预算常用EMA
    df['ema10'] = df['close'].ewm(span=cfg.EMA_SIGNAL, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=cfg.EMA_TREND, adjust=False).mean()
    df['ema120'] = df['close'].ewm(span=cfg.EMA_LONG, adjust=False).mean()
    # 兼容旧代码
    df['ma_fast'] = df['ema10']
    df['ma_slow'] = df['ema20']

    # 保存缓存
    try:
        df.to_parquet(path, index=False)
        print(f"  [缓存] 已保存 {symbol_key} ({len(df)}根K线)")
    except Exception as e:
        print(f"  [缓存] 保存失败: {e}")

    return df


def add_ema(df, signal=None, trend=None, long=None):
    """为原始数据添加/覆盖EMA（用于需要非默认参数的场景）"""
    signal = signal or cfg.EMA_SIGNAL
    trend = trend or cfg.EMA_TREND
    long = long or cfg.EMA_LONG
    df = df.copy()
    df['ema10'] = df['close'].ewm(span=signal, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=trend, adjust=False).mean()
    df['ema120'] = df['close'].ewm(span=long, adjust=False).mean()
    df['ma_fast'] = df['ema10']
    df['ma_slow'] = df['ema20']
    return df


def list_cache():
    """列出所有缓存文件"""
    files = []
    for f in os.listdir(CACHE_DIR):
        if f.endswith('.parquet'):
            path = os.path.join(CACHE_DIR, f)
            mtime = os.path.getmtime(path)
            age_h = (time.time() - mtime) / 3600
            size_kb = os.path.getsize(path) / 1024
            valid = age_h < (CACHE_TTL / 3600)
            files.append({
                'file': f,
                'size': f"{size_kb:.0f}KB",
                'age': f"{age_h:.1f}h",
                'valid': valid,
            })
    return files


def clear_cache():
    """清空所有缓存"""
    count = 0
    for f in os.listdir(CACHE_DIR):
        if f.endswith('.parquet'):
            os.remove(os.path.join(CACHE_DIR, f))
            count += 1
    return count
