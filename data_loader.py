# -*- coding: utf-8 -*-
"""
数据加载 + 指标计算 通用模块
============================
替代各 test_*.py 中重复的 load_and_prepare() 逻辑。

用法:
    from data_loader import load_symbol, load_all, add_indicators

    df = load_symbol('SHFE.ag')
    data = load_all(last_days=120)
"""

import os
import numpy as np
import pandas as pd
from signal_core import SYMBOL_CONFIGS

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
BARS_PER_DAY = 57  # 10min: int(9.5*60/10)

# 高波动品种（研究中反复使用的分组）
HIGH_VOL = {"GFEX.lc", "DCE.jm", "SHFE.ag", "CZCE.FG", "CZCE.SA",
            "INE.sc", "CZCE.MA", "CZCE.TA", "DCE.eb", "DCE.lh"}


def _to_cache_key(sym_key: str) -> str:
    """SHFE.ag -> SHFE_ag"""
    return sym_key.replace(".", "_")


def _to_sym_key(cache_key: str) -> str:
    """SHFE_ag -> SHFE.ag（首个_替换为.）"""
    parts = cache_key.split("_", 1)
    return f"{parts[0]}.{parts[1]}" if len(parts) == 2 else cache_key


def add_indicators(df, emas=(10, 20, 120), er_periods=(20,), atr_period=14):
    """
    为 DataFrame 添加技术指标。

    参数:
      emas: EMA周期列表，如 (10, 20, 60, 120)
      er_periods: ER周期列表，如 (20,) 或 (20, 40)
      atr_period: ATR周期，0则不算

    返回:
      df（原地修改并返回）
    """
    for p in emas:
        col = f'ema{p}'
        df[col] = df['close'].ewm(span=p, adjust=False).mean()

    # 兼容旧代码别名
    if 10 in emas:
        df['ema10'] = df['ema10']  # 已存在
        df['ma_fast'] = df['ema10']
    if 20 in emas:
        df['ma_slow'] = df['ema20']

    for p in er_periods:
        net = (df['close'] - df['close'].shift(p)).abs()
        bar_sum = df['close'].diff().abs().rolling(p).sum()
        df[f'er_{p}'] = net / bar_sum.replace(0, np.nan)
        # 常用别名
        if p == 20:
            df['er_20'] = df['er_20']

    if atr_period > 0:
        prev_close = df['close'].shift(1)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                (df['high'] - prev_close).abs(),
                (df['low'] - prev_close).abs()
            )
        )
        df['atr'] = df['tr'].rolling(window=atr_period).mean()

    return df


def load_symbol(sym_key, period_min=10, days=170,
                emas=(10, 20, 120), er_periods=(20,), atr_period=14):
    """
    加载单个品种的缓存数据并计算指标。

    参数:
      sym_key: 品种代码，如 'SHFE.ag' 或 'SHFE_ag'（两种都行）
    返回:
      DataFrame 或 None（缓存不存在时）
    """
    if "." not in sym_key:
        sym_key = _to_sym_key(sym_key)
    cache_key = _to_cache_key(sym_key)
    path = os.path.join(CACHE_DIR, f"{cache_key}_{period_min}min_{days}d.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    # TqSdk存储UTC时间，转换为北京时间（+8h）
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        if df['datetime'].dt.tz is None:
            # 无时区信息，视为UTC，直接加8小时
            if len(df) > 0 and df['datetime'].iloc[0].hour < 9:
                df['datetime'] = df['datetime'] + pd.Timedelta(hours=8)
    return add_indicators(df, emas=emas, er_periods=er_periods, atr_period=atr_period)


def load_all(period_min=10, days=170, last_days=120,
             emas=(10, 20, 120), er_periods=(20,), atr_period=14,
             min_bars=200):
    """
    加载全部32品种，返回 dict。

    参数:
      last_days: 只保留最后N天的数据区间（None=不裁剪）
      min_bars: 裁剪后少于此数的品种跳过
    返回:
      {sym_key: df} 字典
    """
    result = {}
    for sym_key in sorted(SYMBOL_CONFIGS.keys()):
        df = load_symbol(sym_key, period_min, days, emas, er_periods, atr_period)
        if df is None:
            continue
        if last_days is not None:
            n = len(df)
            start = max(0, n - last_days * BARS_PER_DAY)
            df = df.iloc[start:].reset_index(drop=True)
        if len(df) < min_bars:
            continue
        result[sym_key] = df
    return result


def get_start_idx(df, last_days=120):
    """计算 start_idx（跳过预热期，只在最后N天范围内产生信号）"""
    return max(0, len(df) - last_days * BARS_PER_DAY)


# ============ 便捷查询 ============

def sym_name(sym_key):
    """SHFE.ag -> 白银"""
    if "." not in sym_key:
        sym_key = _to_sym_key(sym_key)
    cfg = SYMBOL_CONFIGS.get(sym_key)
    return cfg['name'] if cfg else sym_key


def tick_size(sym_key):
    if "." not in sym_key:
        sym_key = _to_sym_key(sym_key)
    return SYMBOL_CONFIGS[sym_key]['tick_size']


def multiplier(sym_key):
    if "." not in sym_key:
        sym_key = _to_sym_key(sym_key)
    return SYMBOL_CONFIGS[sym_key]['multiplier']


def sym_group(sym_key):
    """返回 'H'(高波动) 或 'N'(常规)"""
    if "." not in sym_key:
        sym_key = _to_sym_key(sym_key)
    return 'H' if sym_key in HIGH_VOL else 'N'
