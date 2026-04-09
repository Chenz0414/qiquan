# -*- coding: utf-8 -*-
"""
回测系统配置文件
更换品种/周期/时间只需修改此文件
"""

# ============ 天勤账号 ============
TQ_ACCOUNT = "bonjour0414"
TQ_PASSWORD = "zc950414"

# ============ 品种（回测用，切换品种改这里） ============
SYMBOL = "KQ.m@DCE.lh"   # 生猪主力连续
SYMBOL_NAME = "生猪"
TICK_SIZE = 5            # 最小变动价位（元/吨）
STOP_TICKS = 5           # 止损缓冲跳数（120天32品种验证，5跳总累计+234%优于1跳+161%）
CONTRACT_MULTIPLIER = 16  # 合约乘数（吨）

# ============ 多品种配置（监控+回测共用，见 signal_core.py） ============
# 回测时如需切换品种，修改上方 SYMBOL/TICK_SIZE/CONTRACT_MULTIPLIER
# 实时监控使用 signal_core.SYMBOL_CONFIGS 自动匹配品种参数

# ============ 时间周期 ============
PERIODS = [2, 10, 60]    # 分钟级别K线周期列表

# ============ 回测时间 ============
BACKTEST_DAYS = 120      # 信号统计区间（天）
WARMUP_DAYS = 50         # MA预热期（天）

# ============ 均线参数（全部用EMA） ============
EMA_SIGNAL = 10          # 信号均线（回调穿越用）
EMA_TREND = 20           # 趋势快线
EMA_LONG = 120           # 趋势慢线
ATR_PERIOD = 14          # ATR周期

# 兼容旧代码（逐步移除）
MA_FAST = EMA_SIGNAL
MA_SLOW = EMA_TREND
MA_LONG = EMA_LONG

# ============ 输出路径 ============
OUTPUT_DIR = "output"
CSV_FILENAME = "trades.csv"
REPORT_FILENAME = "report.html"
