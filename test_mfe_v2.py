# -*- coding: utf-8 -*-
"""
MFE v2：哪些信号特征能预判爆发性行情？
——在normal阶段的信号中，筛出最有爆发潜力的
——期权视角：MFE_8h > 3% 视为"爆发"，看什么因子能区分
"""

import os, sys, time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import timedelta
from tqsdk import TqApi, TqAuth
import config as cfg
from signal_core import SignalDetector, SYMBOL_CONFIGS
from data_cache import get_klines

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

SYMBOLS = {
    'SHFE.ag': {'tq': 'KQ.m@SHFE.ag', 'name': '白银'},
    'SHFE.au': {'tq': 'KQ.m@SHFE.au', 'name': '黄金'},
    'INE.sc':  {'tq': 'KQ.m@INE.sc',  'name': '原油'},
    'GFEX.lc': {'tq': 'KQ.m@GFEX.lc', 'name': '碳酸锂'},
    'DCE.lh':  {'tq': 'KQ.m@DCE.lh',  'name': '生猪'},
    'DCE.v':   {'tq': 'KQ.m@DCE.v',   'name': 'PVC'},
}

MFE_WINDOW = 48  # 8小时（48根10min）
BOOM_THRESHOLD = 3.0  # MFE > 3% 视为"爆发"


def enrich_signals(df, signals):
    """给每个信号加上各种特征，用于分析哪些因子能预判爆发"""
    atr_raw = df['high'].rolling(14).max() - df['low'].rolling(14).min()
    atr = atr_raw / 14
    df['atr'] = atr

    ma10 = df['ma_fast']
    ma20 = df['ma_slow']
    df['ma_slope'] = (ma10 - ma10.shift(5)) / atr
    df['ma_width'] = (ma10 - ma20) / atr
    df['ma60'] = df['close'].rolling(60).mean()

    # 波动率：近12根K线的振幅/ATR
    df['recent_vol'] = df['high'].rolling(12).max() - df['low'].rolling(12).min()
    df['vol_ratio'] = df['recent_vol'] / atr  # 近期波动/ATR

    # 成交量
    df['vol_ma'] = df['volume'].rolling(20).mean()

    for sig in signals:
        i = sig['idx']
        row = df.iloc[i]

        # 基础特征
        sig['ma_slope'] = abs(row['ma_slope']) if not pd.isna(row['ma_slope']) else 0
        sig['ma_width'] = abs(row['ma_width']) if not pd.isna(row['ma_width']) else 0
        sig['ma_slope_raw'] = row['ma_slope'] if not pd.isna(row['ma_slope']) else 0

        # 趋势年龄：MA10连续在MA20同侧多少根了
        trend_age = 0
        is_above = ma10.iloc[i] > ma20.iloc[i] if not pd.isna(ma10.iloc[i]) else True
        for j in range(i - 1, max(i - 200, 0), -1):
            if pd.isna(ma10.iloc[j]) or pd.isna(ma20.iloc[j]):
                break
            if (ma10.iloc[j] > ma20.iloc[j]) == is_above:
                trend_age += 1
            else:
                break
        sig['trend_age'] = trend_age

        # 距60MA的距离（%）
        ma60 = row['ma60'] if not pd.isna(row['ma60']) else row['close']
        sig['dist_ma60'] = abs(row['close'] - ma60) / ma60 * 100

        # 近期波动率
        sig['vol_ratio'] = row['vol_ratio'] if not pd.isna(row['vol_ratio']) else 1.0

        # 成交量比（当前bar成交量 / 20均量）
        vol_ma = row['vol_ma'] if not pd.isna(row['vol_ma']) and row['vol_ma'] > 0 else 1
        sig['volume_ratio'] = row['volume'] / vol_ma

        # 回调深度（回调最远价 vs 入场价，占ATR的比例）
        if not pd.isna(row['atr']) and row['atr'] > 0:
            sig['pb_depth_atr'] = abs(sig['entry_price'] - sig.get('pb_extreme', sig['entry_price'])) / row['atr']
        else:
            sig['pb_depth_atr'] = 0

        # MFE计算
        end_idx = min(i + MFE_WINDOW, len(df) - 1)
        if i + 1 <= end_idx:
            segment = df.iloc[i + 1: end_idx + 1]
            if sig['direction'] == 'short':
                mfe = (sig['entry_price'] - segment['low'].min()) / sig['entry_price'] * 100
            else:
                mfe = (segment['high'].max() - sig['entry_price']) / sig['entry_price'] * 100
            sig['mfe_8h'] = max(mfe, 0)
        else:
            sig['mfe_8h'] = 0

        sig['is_boom'] = sig['mfe_8h'] >= BOOM_THRESHOLD

    return signals


def analyze_factor(signals, factor_name, bins=3):
    """分析某个因子对爆发率的影响"""
    values = [s[factor_name] for s in signals]
    if not values or all(v == values[0] for v in values):
        return None

    # 按分位数分箱
    try:
        percentiles = np.percentile(values, [33, 67])
    except:
        return None

    groups = {'低': [], '中': [], '高': []}
    for s in signals:
        v = s[factor_name]
        if v <= percentiles[0]:
            groups['低'].append(s)
        elif v <= percentiles[1]:
            groups['中'].append(s)
        else:
            groups['高'].append(s)

    result = {}
    for label, group in groups.items():
        if not group:
            continue
        n = len(group)
        boom_n = sum(1 for s in group if s['is_boom'])
        avg_mfe = np.mean([s['mfe_8h'] for s in group])
        boom_rate = boom_n / n * 100
        result[label] = {
            'n': n, 'boom_n': boom_n, 'boom_rate': boom_rate,
            'avg_mfe': avg_mfe,
            'range': (min(s[factor_name] for s in group), max(s[factor_name] for s in group)),
        }
    return result


def main():
    api = TqApi(auth=TqAuth(cfg.TQ_ACCOUNT, cfg.TQ_PASSWORD))

    all_signals = []
    per_symbol = {}

    for sym_key, sym_info in SYMBOLS.items():
        print(f"\n加载 {sym_info['name']}...", end=' ')
        df = get_klines(api, sym_info['tq'], sym_key, period_min=10, days=170)

        cutoff = df['datetime'].iloc[-1] - timedelta(days=120)
        df = df[df['datetime'] >= cutoff].copy().reset_index(drop=True)

        detector = SignalDetector(min_pb_bars=2)
        warmup = cfg.MA_SLOW + 5
        signals = []
        for i in range(warmup, len(df)):
            row = df.iloc[i]
            if pd.isna(row['ma_fast']) or pd.isna(row['ma_slow']):
                continue
            result = detector.process_bar(
                close=row['close'], high=row['high'], low=row['low'],
                ma_fast=row['ma_fast'], ma_slow=row['ma_slow'],
            )
            if result is not None:
                signals.append({
                    'idx': i, 'time': row['datetime'],
                    'direction': result.direction,
                    'entry_price': result.entry_price,
                    'pb_bars': result.pullback_bars,
                    'pb_extreme': result.pullback_extreme,
                    'symbol': sym_key, 'name': sym_info['name'],
                    'with_trend': True,  # SignalDetector已经内置了MA方向过滤
                })

        signals = enrich_signals(df, signals)
        per_symbol[sym_key] = {'signals': signals, 'name': sym_info['name']}
        all_signals.extend(signals)
        boom_n = sum(1 for s in signals if s['is_boom'])
        print(f"{len(signals)}信号, {boom_n}个爆发(MFE>3%)")

    api.close()

    # ============================================================
    # 全品种因子分析
    # ============================================================
    factors = [
        ('ma_slope', 'MA斜率(绝对值)'),
        ('ma_width', 'MA间距(绝对值)'),
        ('trend_age', '趋势年龄(根数)'),
        ('dist_ma60', '距60MA距离(%)'),
        ('vol_ratio', '近期波动率'),
        ('volume_ratio', '成交量比'),
        ('pb_bars', '回调根数'),
    ]

    print(f"\n\n{'='*80}")
    print(f"  全品种因子分析：什么特征的信号更容易爆发？")
    print(f"  爆发定义：MFE_8h > {BOOM_THRESHOLD}%")
    print(f"  总信号: {len(all_signals)}, 爆发: {sum(1 for s in all_signals if s['is_boom'])}")
    print(f"{'='*80}")

    factor_scores = {}
    for factor_key, factor_label in factors:
        result = analyze_factor(all_signals, factor_key)
        if not result:
            continue

        print(f"\n  {factor_label}:")
        max_boom = max(r['boom_rate'] for r in result.values())
        min_boom = min(r['boom_rate'] for r in result.values())
        spread = max_boom - min_boom  # 区分度

        for label in ['低', '中', '高']:
            if label in result:
                r = result[label]
                bar = '█' * int(r['boom_rate'] / 2)
                star = ' ★' if r['boom_rate'] == max_boom and spread > 10 else ''
                print(f"    {label}: {r['n']:>3}信号 | 爆发{r['boom_n']:>3}个({r['boom_rate']:>5.1f}%) "
                      f"| 平均MFE={r['avg_mfe']:>5.2f}% | {bar}{star}")

        factor_scores[factor_key] = spread
        if spread > 10:
            print(f"    → 区分度: {spread:.1f}% ★★★ 强因子")
        elif spread > 5:
            print(f"    → 区分度: {spread:.1f}% ★★ 中等因子")
        else:
            print(f"    → 区分度: {spread:.1f}% ★ 弱因子")

    # ============================================================
    # 分品种看最强因子的效果
    # ============================================================
    # 找出最强的3个因子
    top_factors = sorted(factor_scores.items(), key=lambda x: -x[1])[:3]
    print(f"\n\n{'='*80}")
    print(f"  Top因子: {', '.join(f[0] for f in top_factors)}")
    print(f"{'='*80}")

    for sym_key, data in per_symbol.items():
        signals = data['signals']
        if len(signals) < 10:
            continue
        boom_n = sum(1 for s in signals if s['is_boom'])
        total_boom_rate = boom_n / len(signals) * 100 if signals else 0
        print(f"\n  {data['name']}: {len(signals)}信号, 基础爆发率={total_boom_rate:.1f}%")

        for factor_key, _ in top_factors:
            factor_label = dict(factors)[factor_key]
            result = analyze_factor(signals, factor_key)
            if not result:
                continue
            high = result.get('高', {})
            low = result.get('低', {})
            if high and low:
                improvement = high.get('boom_rate', 0) - low.get('boom_rate', 0)
                print(f"    {factor_label}: 低={low.get('boom_rate',0):.1f}% → 高={high.get('boom_rate',0):.1f}% (差{improvement:+.1f}%)")

    # ============================================================
    # 组合筛选器：用top因子的"高"值组合过滤
    # ============================================================
    print(f"\n\n{'='*80}")
    print(f"  组合筛选器测试")
    print(f"{'='*80}")

    # 计算全局各因子的P67阈值
    thresholds = {}
    for factor_key, _ in top_factors:
        vals = [s[factor_key] for s in all_signals]
        thresholds[factor_key] = np.percentile(vals, 67)
        print(f"  {factor_key} P67阈值: {thresholds[factor_key]:.2f}")

    for sym_key, data in per_symbol.items():
        signals = data['signals']
        if len(signals) < 5:
            continue

        # 方案A：任意1个top因子为"高"
        filterA = [s for s in signals
                   if any(s[fk] >= thresholds[fk] for fk, _ in top_factors)]
        # 方案B：至少2个top因子为"高"
        filterB = [s for s in signals
                   if sum(1 for fk, _ in top_factors if s[fk] >= thresholds[fk]) >= 2]

        base_boom = sum(1 for s in signals if s['is_boom'])
        base_rate = base_boom / len(signals) * 100 if signals else 0

        print(f"\n  {data['name']}:")
        print(f"    无过滤: {len(signals):>3}信号 → {base_boom:>2}爆发 ({base_rate:.1f}%) | 平均MFE={np.mean([s['mfe_8h'] for s in signals]):.2f}%")

        if filterA:
            boomA = sum(1 for s in filterA if s['is_boom'])
            rateA = boomA / len(filterA) * 100
            print(f"    方案A:  {len(filterA):>3}信号 → {boomA:>2}爆发 ({rateA:.1f}%) | 平均MFE={np.mean([s['mfe_8h'] for s in filterA]):.2f}%")

        if filterB:
            boomB = sum(1 for s in filterB if s['is_boom'])
            rateB = boomB / len(filterB) * 100
            print(f"    方案B:  {len(filterB):>3}信号 → {boomB:>2}爆发 ({rateB:.1f}%) | 平均MFE={np.mean([s['mfe_8h'] for s in filterB]):.2f}%")
        else:
            print(f"    方案B:  0信号")

    # ============================================================
    # 图表
    # ============================================================
    n_sym = len(per_symbol)
    fig, axes = plt.subplots(n_sym, 2, figsize=(18, 4.5 * n_sym))
    if n_sym == 1:
        axes = axes.reshape(1, -1)

    for row_idx, (sym_key, data) in enumerate(per_symbol.items()):
        signals = data['signals']
        if not signals:
            continue

        # 左图：MFE分布 - 爆发 vs 非爆发
        ax = axes[row_idx, 0]
        boom_mfe = [s['mfe_8h'] for s in signals if s['is_boom']]
        normal_mfe = [s['mfe_8h'] for s in signals if not s['is_boom']]
        if boom_mfe and normal_mfe:
            ax.hist([normal_mfe, boom_mfe], bins=20, stacked=True,
                    color=['#ccc', '#4CAF50'], label=['普通', f'爆发(>{BOOM_THRESHOLD}%)'],
                    edgecolor='white')
        elif normal_mfe:
            ax.hist(normal_mfe, bins=20, color='#ccc', label='普通', edgecolor='white')
        ax.axvline(x=BOOM_THRESHOLD, color='red', linestyle='--', alpha=0.7, label=f'爆发线({BOOM_THRESHOLD}%)')
        ax.set_title(f"{data['name']} — MFE_8h分布", fontweight='bold')
        ax.set_xlabel('MFE %')
        ax.legend()

        # 右图：最强因子 vs MFE散点图
        ax = axes[row_idx, 1]
        if top_factors:
            best_factor = top_factors[0][0]
            best_label = dict(factors)[best_factor]
            x = [s[best_factor] for s in signals]
            y = [s['mfe_8h'] for s in signals]
            colors = ['#4CAF50' if s['is_boom'] else '#999' for s in signals]
            sizes = [60 if s['is_boom'] else 20 for s in signals]
            ax.scatter(x, y, c=colors, s=sizes, alpha=0.6, edgecolors='white', linewidth=0.5)
            ax.axhline(y=BOOM_THRESHOLD, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel(best_label)
            ax.set_ylabel('MFE_8h %')
            ax.set_title(f"{data['name']} — {best_label} vs MFE", fontweight='bold')
            ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'output', 'mfe_factors.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图表: {out_path}")


if __name__ == '__main__':
    main()
