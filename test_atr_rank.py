# -*- coding: utf-8 -*-
"""
全品种ATR%排序 — 用于品种分类（高波动 vs 常规）
读取10min缓存数据，计算每个品种的ATR(14)/close均值
"""

import os
import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")

# 品种名称映射
SYMBOL_NAMES = {
    "SHFE_au": "黄金", "SHFE_ag": "白银", "INE_sc": "原油",
    "SHFE_cu": "铜", "SHFE_al": "铝", "SHFE_zn": "锌",
    "SHFE_ni": "镍", "INE_bc": "国际铜",
    "SHFE_rb": "螺纹钢", "SHFE_hc": "热卷", "DCE_i": "铁矿石",
    "DCE_jm": "焦煤", "DCE_j": "焦炭",
    "SHFE_bu": "沥青", "SHFE_ru": "橡胶", "SHFE_sp": "纸浆",
    "SHFE_fu": "燃料油", "INE_lu": "低硫燃油",
    "DCE_eg": "乙二醇", "DCE_pp": "聚丙烯", "DCE_l": "塑料",
    "DCE_pg": "LPG", "DCE_eb": "苯乙烯", "DCE_v": "PVC",
    "CZCE_TA": "PTA", "CZCE_MA": "甲醇", "CZCE_SA": "纯碱",
    "CZCE_FG": "玻璃",
    "DCE_m": "豆粕", "DCE_y": "豆油", "DCE_p": "棕榈油",
    "CZCE_SR": "白糖", "CZCE_CF": "棉花", "CZCE_RM": "菜粕",
    "GFEX_lc": "碳酸锂", "DCE_lh": "生猪",
}

ATR_PERIOD = 14

results = []

for fname in sorted(os.listdir(CACHE_DIR)):
    if not fname.endswith("_10min_170d.parquet"):
        continue

    symbol_key = fname.replace("_10min_170d.parquet", "")
    name = SYMBOL_NAMES.get(symbol_key, symbol_key)

    path = os.path.join(CACHE_DIR, fname)
    df = pd.read_parquet(path)

    # 计算ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=ATR_PERIOD).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100

    # 日均振幅%
    df['range_pct'] = (df['high'] - df['low']) / df['close'] * 100

    atr_pct_mean = df['atr_pct'].dropna().mean()
    range_pct_mean = df['range_pct'].dropna().mean()
    avg_price = df['close'].mean()
    n_bars = len(df)

    results.append({
        '品种': name,
        '代码': symbol_key,
        'ATR%均值': round(atr_pct_mean, 4),
        '振幅%均值': round(range_pct_mean, 4),
        '均价': round(avg_price, 1),
        'K线数': n_bars,
    })

# 按ATR%降序排列
results_df = pd.DataFrame(results).sort_values('ATR%均值', ascending=False).reset_index(drop=True)
results_df.index += 1

print("=" * 70)
print("全品种10min ATR%排序（降序）")
print("=" * 70)
print(results_df.to_string())
print(f"\n共 {len(results_df)} 个品种")

# 统计分位数
atr_values = results_df['ATR%均值']
print(f"\nATR%分布: min={atr_values.min():.4f}, median={atr_values.median():.4f}, max={atr_values.max():.4f}")
print(f"P25={atr_values.quantile(0.25):.4f}, P50={atr_values.quantile(0.50):.4f}, P75={atr_values.quantile(0.75):.4f}")
