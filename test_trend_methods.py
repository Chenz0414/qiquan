# -*- coding: utf-8 -*-
"""
趋势判断方法对比测试
====================
固定入场信号（A+B类回调MA10），固定出场（2%止损/5%止盈）
只变趋势判断方式，对比各方案的期望值。
SMA和EMA都测。32品种，120天。
"""

import os
import pandas as pd
import numpy as np
from datetime import timedelta
from signal_core import SYMBOL_CONFIGS

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STOP_LOSS_PCT = 2.0   # 固定止损%
TAKE_PROFIT_PCT = 5.0  # 固定止盈%
MA_SLOPE_LOOKBACK = 5  # 斜率回看根数


def get_all_10min_symbols():
    symbols = []
    for f in os.listdir(CACHE_DIR):
        if f.endswith('_10min_170d.parquet'):
            symbols.append(f.replace('_10min_170d.parquet', ''))
    return sorted(symbols)


def sym_key_to_config_key(sym_key):
    parts = sym_key.split('_', 1)
    return f"{parts[0]}.{parts[1]}" if len(parts) == 2 else sym_key


def get_sym_name(sym_key):
    ck = sym_key_to_config_key(sym_key)
    return SYMBOL_CONFIGS[ck]['name'] if ck in SYMBOL_CONFIGS else sym_key


def load_cached(symbol_key):
    path = os.path.join(CACHE_DIR, f"{symbol_key}_10min_170d.parquet")
    return pd.read_parquet(path) if os.path.exists(path) else None


def prepare_data(df):
    """计算所有需要的均线（SMA + EMA）"""
    for p in [10, 20, 60, 120]:
        df[f'sma_{p}'] = df['close'].rolling(p).mean()
        df[f'ema_{p}'] = df['close'].ewm(span=p, adjust=False).mean()
    return df


# ============================================================
# 趋势判断函数：返回 1=多头, -1=空头, 0=无趋势
# ============================================================

def trend_fast_slow(row, prefix, fast=10, slow=20):
    """快慢均线交叉"""
    f = row[f'{prefix}_{fast}']
    s = row[f'{prefix}_{slow}']
    if pd.isna(f) or pd.isna(s):
        return 0
    return 1 if f > s else (-1 if f < s else 0)


def trend_triple(row, prefix):
    """三线排列: 10 > 20 > 60"""
    m10 = row[f'{prefix}_10']
    m20 = row[f'{prefix}_20']
    m60 = row[f'{prefix}_60']
    if pd.isna(m10) or pd.isna(m20) or pd.isna(m60):
        return 0
    if m10 > m20 > m60:
        return 1
    elif m10 < m20 < m60:
        return -1
    return 0


def trend_close_vs_ma(row, prefix, period=60):
    """价格在均线哪一侧"""
    ma = row[f'{prefix}_{period}']
    if pd.isna(ma):
        return 0
    return 1 if row['close'] > ma else (-1 if row['close'] < ma else 0)


def trend_slope(df, i, prefix, period=20):
    """均线斜率：当前MA20 vs N根前的MA20"""
    if i < MA_SLOPE_LOOKBACK:
        return 0
    curr = df.iloc[i][f'{prefix}_{period}']
    prev = df.iloc[i - MA_SLOPE_LOOKBACK][f'{prefix}_{period}']
    if pd.isna(curr) or pd.isna(prev):
        return 0
    if curr > prev:
        return 1
    elif curr < prev:
        return -1
    return 0


# 三个方案对比（加MA10>MA20做基准）
TREND_METHODS = [
    ('MA10>MA20',     lambda df, i, pfx: trend_fast_slow(df.iloc[i], pfx, 10, 20)),
    ('三线排列',       lambda df, i, pfx: trend_triple(df.iloc[i], pfx)),
    ('MA20>MA120',    lambda df, i, pfx: trend_fast_slow(df.iloc[i], pfx, 20, 120)),
]


def detect_signals_with_trend(df, trend_fn, ma_prefix):
    """
    用指定的趋势判断函数检测A+B类信号。
    入场信号仍基于MA10（用对应的sma_10或ema_10）回调。
    """
    signals = []
    n = len(df)
    warmup = 130  # 足够MA120预热

    b_below_start = -1
    b_pullback_low = None
    b_pullback_high = None
    prev_trend = 0

    ma_fast_col = f'{ma_prefix}_10'

    for i in range(warmup, n):
        row = df.iloc[i]
        ma_f = row[ma_fast_col]
        if pd.isna(ma_f):
            continue

        close = row['close']
        high = row['high']
        low = row['low']

        trend = trend_fn(df, i, ma_prefix)

        if trend != prev_trend and trend != 0:
            b_below_start = -1
            b_pullback_low = None
            b_pullback_high = None
        prev_trend = trend
        if trend == 0:
            continue

        if i < 1:
            continue
        prev = df.iloc[i - 1]
        prev_close = prev['close']
        prev_ma_f = prev[ma_fast_col]
        if pd.isna(prev_ma_f):
            continue

        found = []

        if trend == 1:
            # A类
            if low <= ma_f and close > ma_f and prev_close > prev_ma_f:
                found.append(('A', 'long'))
            # B类
            if b_below_start == -1:
                if close < ma_f and prev_close >= prev_ma_f:
                    b_below_start = i
                    b_pullback_low = low
            else:
                b_pullback_low = min(b_pullback_low, low)
                if close > ma_f:
                    pb_bars = i - b_below_start
                    if pb_bars >= 4:
                        found.append(('B', 'long'))
                    b_below_start = -1
                    b_pullback_low = None

        elif trend == -1:
            # A类
            if high >= ma_f and close < ma_f and prev_close < prev_ma_f:
                found.append(('A', 'short'))
            # B类
            if b_below_start == -1:
                if close > ma_f and prev_close <= prev_ma_f:
                    b_below_start = i
                    b_pullback_high = high
            else:
                b_pullback_high = max(b_pullback_high, high)
                if close < ma_f:
                    pb_bars = i - b_below_start
                    if pb_bars >= 4:
                        found.append(('B', 'short'))
                    b_below_start = -1
                    b_pullback_high = None

        for sig_type, direction in found:
            signals.append({
                'type': sig_type,
                'direction': direction,
                'idx': i,
                'entry_price': close,
            })

    return signals


def simulate_fixed_exit(df, sig):
    """固定2%止损 / 5%止盈"""
    entry_idx = sig['idx']
    entry_price = sig['entry_price']
    is_long = sig['direction'] == 'long'
    n = len(df)

    sl_price = entry_price * (1 - STOP_LOSS_PCT / 100) if is_long else entry_price * (1 + STOP_LOSS_PCT / 100)
    tp_price = entry_price * (1 + TAKE_PROFIT_PCT / 100) if is_long else entry_price * (1 - TAKE_PROFIT_PCT / 100)

    for j in range(entry_idx + 1, n):
        bar = df.iloc[j]
        if is_long:
            if bar['low'] <= sl_price:
                return -STOP_LOSS_PCT
            if bar['high'] >= tp_price:
                return +TAKE_PROFIT_PCT
        else:
            if bar['high'] >= sl_price:
                return -STOP_LOSS_PCT
            if bar['low'] <= tp_price:
                return +TAKE_PROFIT_PCT

    # 未触发，按最后收盘价算
    last_close = df.iloc[-1]['close']
    if is_long:
        return (last_close - entry_price) / entry_price * 100
    else:
        return (entry_price - last_close) / entry_price * 100


def calc_ev(pnls):
    """计算期望 = 胜率×盈亏比 - (1-胜率)"""
    if len(pnls) == 0:
        return 0, 0, 0, 0
    pnls = np.array(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    n = len(pnls)
    wr = len(wins) / n
    avg_w = np.mean(wins) if len(wins) > 0 else 0
    avg_l = abs(np.mean(losses)) if len(losses) > 0 else 0.001
    ratio = avg_w / avg_l
    ev = wr * ratio - (1 - wr)
    return ev, wr, ratio, n


def main():
    all_symbols = get_all_10min_symbols()
    print("=" * 100)
    print(f"  趋势判断方法对比 | 全品种 | 120天 | 出场: {STOP_LOSS_PCT}%止损/{TAKE_PROFIT_PCT}%止盈")
    print("=" * 100)

    # 预加载所有数据
    sym_data = {}
    for sym_key in all_symbols:
        df = load_cached(sym_key)
        if df is None:
            continue
        df = prepare_data(df)
        cutoff = df['datetime'].iloc[-1] - timedelta(days=120)
        df_120 = df[df['datetime'] >= cutoff].copy().reset_index(drop=True)
        if len(df_120) < 200:
            continue
        sym_data[sym_key] = df_120

    print(f"  加载 {len(sym_data)} 品种\n")

    # 收集结果
    results = []

    for ma_type in ['ema']:
        ma_label = 'SMA' if ma_type == 'sma' else 'EMA'
        for method_name, trend_fn in TREND_METHODS:
            all_pnls_a = []
            all_pnls_b = []
            sym_details = {}  # 每品种明细

            for sym_key, df in sym_data.items():
                sigs = detect_signals_with_trend(df, trend_fn, ma_type)
                sym_pnls_a = []
                sym_pnls_b = []
                for s in sigs:
                    pnl = simulate_fixed_exit(df, s)
                    if s['type'] == 'A':
                        all_pnls_a.append(pnl)
                        sym_pnls_a.append(pnl)
                    else:
                        all_pnls_b.append(pnl)
                        sym_pnls_b.append(pnl)
                sym_details[sym_key] = {
                    'pnls_a': sym_pnls_a, 'pnls_b': sym_pnls_b,
                }

            ev_a, wr_a, ratio_a, n_a = calc_ev(all_pnls_a)
            ev_b, wr_b, ratio_b, n_b = calc_ev(all_pnls_b)
            ev_all, wr_all, ratio_all, n_all = calc_ev(all_pnls_a + all_pnls_b)

            results.append({
                'ma': ma_label, 'method': method_name,
                'n_a': n_a, 'ev_a': ev_a, 'wr_a': wr_a, 'ratio_a': ratio_a,
                'n_b': n_b, 'ev_b': ev_b, 'wr_b': wr_b, 'ratio_b': ratio_b,
                'n_all': n_all, 'ev_all': ev_all, 'wr_all': wr_all, 'ratio_all': ratio_all,
                'sym_details': sym_details,
            })

            print(f"\n  {ma_label} {method_name:<12} | "
                  f"A类: N={n_a:>5} 期望={ev_a:>+5.2f} 胜率={wr_a*100:>5.1f}% 盈亏比={ratio_a:>4.2f} | "
                  f"B类: N={n_b:>5} 期望={ev_b:>+5.2f} 胜率={wr_b*100:>5.1f}% 盈亏比={ratio_b:>4.2f} | "
                  f"全部: N={n_all:>5} 期望={ev_all:>+5.2f}")

        print()

    # ============================================================
    # 汇总对比（三方案）
    # ============================================================
    print("\n" + "=" * 100)
    print("  三方案全品种对比")
    print("=" * 100)
    for r in results:
        print(f"\n  【{r['method']}】")
        print(f"    全部: N={r['n_all']:>5} 期望={r['ev_all']:>+5.2f} 胜率={r['wr_all']*100:>5.1f}% 盈亏比={r['ratio_all']:>4.2f}")
        print(f"    A 类: N={r['n_a']:>5} 期望={r['ev_a']:>+5.2f} 胜率={r['wr_a']*100:>5.1f}% 盈亏比={r['ratio_a']:>4.2f}")
        print(f"    B 类: N={r['n_b']:>5} 期望={r['ev_b']:>+5.2f} 胜率={r['wr_b']*100:>5.1f}% 盈亏比={r['ratio_b']:>4.2f}")

    # ============================================================
    # 逐品种对比（三方案 A+B）
    # ============================================================
    r_by_method = {r['method']: r for r in results}
    method_names = [m[0] for m in TREND_METHODS]
    all_syms = sorted(set().union(*[r['sym_details'].keys() for r in results]))

    print("\n" + "=" * 120)
    print("  逐品种对比（A+B合计期望）")
    print("=" * 120)
    header = f"  {'品种':<10}"
    for mn in method_names:
        header += f" | {mn+' N':>8} {mn+' 期望':>8}"
    header += " | 最优"
    print(header)
    print(f"  {'-'*110}")

    method_win_count = {mn: 0 for mn in method_names}
    sym_rows = []

    for sym_key in all_syms:
        sym_name = get_sym_name(sym_key)
        row_data = []
        for mn in method_names:
            d = r_by_method[mn]['sym_details'].get(sym_key, {'pnls_a': [], 'pnls_b': []})
            pnls = d['pnls_a'] + d['pnls_b']
            ev, _, _, n = calc_ev(pnls)
            row_data.append((n, ev))
        best_ev = max(rd[1] for rd in row_data)
        best_method = method_names[[rd[1] for rd in row_data].index(best_ev)]
        method_win_count[best_method] += 1
        sym_rows.append((sym_name, row_data, best_method))

    sym_rows.sort(key=lambda x: max(rd[1] for rd in x[1]), reverse=True)
    for sym_name, row_data, best_method in sym_rows:
        line = f"  {sym_name:<10}"
        for n, ev in row_data:
            ev_s = f'{ev:>+6.2f}' if n > 0 else '   N/A'
            line += f" | {n:>8} {ev_s:>8}"
        line += f" | {best_method}"
        print(line)

    print(f"\n  品种胜负:", end='')
    for mn in method_names:
        print(f"  {mn}: {method_win_count[mn]}品种", end='')
    print()

    # ============================================================
    # 逐品种B类对比（三方案）
    # ============================================================
    print("\n" + "=" * 120)
    print("  逐品种对比（仅B类期望）")
    print("=" * 120)
    header_b = f"  {'品种':<10}"
    for mn in method_names:
        header_b += f" | {mn+' B_N':>8} {mn+' B期望':>8}"
    header_b += " | 最优"
    print(header_b)
    print(f"  {'-'*110}")

    method_win_b = {mn: 0 for mn in method_names}
    sym_rows_b = []

    for sym_key in all_syms:
        sym_name = get_sym_name(sym_key)
        row_data = []
        for mn in method_names:
            d = r_by_method[mn]['sym_details'].get(sym_key, {'pnls_a': [], 'pnls_b': []})
            ev, _, _, n = calc_ev(d['pnls_b'])
            row_data.append((n, ev))
        if all(rd[0] == 0 for rd in row_data):
            continue
        best_ev = max(rd[1] for rd in row_data)
        best_method = method_names[[rd[1] for rd in row_data].index(best_ev)]
        method_win_b[best_method] += 1
        sym_rows_b.append((sym_name, row_data, best_method))

    sym_rows_b.sort(key=lambda x: max(rd[1] for rd in x[1]), reverse=True)
    for sym_name, row_data, best_method in sym_rows_b:
        line = f"  {sym_name:<10}"
        for n, ev in row_data:
            ev_s = f'{ev:>+6.2f}' if n > 0 else '   N/A'
            line += f" | {n:>8} {ev_s:>8}"
        line += f" | {best_method}"
        print(line)

    print(f"\n  B类品种胜负:", end='')
    for mn in method_names:
        print(f"  {mn}: {method_win_b[mn]}品种", end='')
    print()

    # 保存CSV
    csv_results = [{k: v for k, v in r.items() if k != 'sym_details'} for r in results]
    csv_path = os.path.join(OUTPUT_DIR, 'trend_methods_full.csv')
    pd.DataFrame(csv_results).to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n数据: {csv_path}")


if __name__ == '__main__':
    main()
