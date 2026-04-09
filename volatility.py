# -*- coding: utf-8 -*-
"""
历史波动率计算模块
==================
从10分钟K线计算年化历史波动率（HV），作为期权定价的IV近似。

用法:
    from volatility import add_hv
    df = add_hv(df, windows=[20, 40, 60])
    # df 新增列: hv_20, hv_40, hv_60
"""

import numpy as np
import pandas as pd

# 年化因子：10分钟K线，每天57根，每年245个交易日
BARS_PER_DAY = 57
TRADING_DAYS = 245
ANNUAL_FACTOR = np.sqrt(BARS_PER_DAY * TRADING_DAYS)  # ≈ 118.2


def add_hv(df, windows=(20, 40, 60)):
    """
    为DataFrame添加年化历史波动率列。

    参数:
      df: 含 'close' 列的 DataFrame
      windows: 滚动窗口列表（K线根数）
    返回:
      df（原地修改并返回）
    """
    log_ret = np.log(df['close'] / df['close'].shift(1))

    for w in windows:
        df[f'hv_{w}'] = log_ret.rolling(window=w).std() * ANNUAL_FACTOR

    return df
