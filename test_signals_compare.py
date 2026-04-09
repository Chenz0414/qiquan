# -*- coding: utf-8 -*-
"""
多种爆发信号对比测试（期权视角）
——6种信号类型，统一用MFE衡量爆发力
——样本内(前60天)/样本外(后60天)验证
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

MFE_WINDOWS = [12, 24, 48, 72]  # 2h/4h/8h/12h
BOOM_PCT = 3.0


def prepare_df(df):
    """计算各种技术指标"""
    ma10 = df['ma_fast']
    ma20 = df['ma_slow']

    # ATR (真实ATR)
    tr = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': (df['high'] - df['close'].shift(1)).abs(),
        'lc': (df['low'] - df['close'].shift(1)).abs(),
    }).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # 均线
    df['ma60'] = df['close'].rolling(60).mean()

    # 布林带
    df['bb_mid'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * bb_std
    df['bb_lower'] = df['bb_mid'] - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100  # 百分比
    df['bb_width_pct'] = df['bb_width'].rolling(120).rank(pct=True)  # 120根内百分位

    # N根最高/最低
    df['high_20'] = df['high'].rolling(20).max()
    df['low_20'] = df['low'].rolling(20).min()
    df['high_40'] = df['high'].rolling(40).max()
    df['low_40'] = df['low'].rolling(40).min()

    # ATR变化率
    atr_slow = df['atr'].rolling(20).mean()
    df['atr_ratio'] = df['atr'] / atr_slow  # >1 = 波动扩张

    # 成交量
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma20']

    # 前10根振幅（压缩检测）
    df['range_10'] = df['high'].rolling(10).max() - df['low'].rolling(10).min()
    df['range_10_pct'] = df['range_10'] / df['close'] * 100
    df['range_10_rank'] = df['range_10_pct'].rolling(120).rank(pct=True)

    # MA斜率
    df['ma_slope'] = (ma10 - ma10.shift(5)) / df['atr']
    df['ma_width'] = (ma10 - ma20) / df['atr']

    # 距60MA
    df['dist_ma60'] = ((df['close'] - df['ma60']) / df['ma60'] * 100).abs()

    return df


def calc_mfe_all(df, idx, direction):
    """计算信号后各窗口MFE"""
    entry = df.iloc[idx]['close']
    results = {}
    for w in MFE_WINDOWS:
        end = min(idx + w, len(df) - 1)
        if idx + 1 > end:
            results[w] = 0
            continue
        seg = df.iloc[idx + 1: end + 1]
        if direction == 'short':
            mfe = (entry - seg['low'].min()) / entry * 100
        else:
            mfe = (seg['high'].max() - entry) / entry * 100
        results[w] = max(mfe, 0)
    return results


def detect_signals(df, warmup=65):
    """检测6种信号"""
    all_signals = {
        'squeeze': [],       # 波动率压缩突破
        'breakout_20': [],   # 20根新高/新低
        'breakout_40': [],   # 40根新高/新低
        'atr_expand': [],    # ATR突然放大
        'ma_align': [],      # 均线排列刚形成
        'vol_break': [],     # 缩量后放量突破
        'b_pullback': [],    # B类回调（现有）
    }

    # B类回调用SignalDetector
    detector = SignalDetector(min_pb_bars=2)

    # 状态追踪
    prev_ma_aligned = False  # 上一根是否均线排列
    squeeze_active = False   # 布林带是否处于压缩中
    low_vol_count = 0        # 连续缩量根数

    for i in range(warmup, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        if pd.isna(row['atr']) or pd.isna(row['ma60']) or row['atr'] == 0:
            continue

        close = row['close']
        ma10 = row['ma_fast']
        ma20 = row['ma_slow']
        ma60 = row['ma60']

        if pd.isna(ma10) or pd.isna(ma20) or pd.isna(ma60):
            continue

        # 判断趋势方向（用均线）
        trend_up = ma10 > ma20
        direction = 'long' if trend_up else 'short'

        # ------ 信号1: 波动率压缩突破 ------
        # 布林带宽度降到近120根的P20以下 = 压缩
        # 然后价格突破布林带 = 突破
        if not pd.isna(row['bb_width_pct']):
            was_squeeze = squeeze_active
            squeeze_active = row['bb_width_pct'] < 0.20  # 当前是压缩状态

            if was_squeeze and not squeeze_active:
                # 刚刚从压缩中走出来
                if close > row['bb_upper'] and not pd.isna(row['bb_upper']):
                    all_signals['squeeze'].append({
                        'idx': i, 'time': row['datetime'], 'direction': 'long',
                        'entry_price': close, 'type': 'squeeze',
                    })
                elif close < row['bb_lower'] and not pd.isna(row['bb_lower']):
                    all_signals['squeeze'].append({
                        'idx': i, 'time': row['datetime'], 'direction': 'short',
                        'entry_price': close, 'type': 'squeeze',
                    })

            # 补充：压缩状态中价格突破布林带（不等压缩结束）
            if squeeze_active:
                if close > row['bb_upper'] and prev['close'] <= prev.get('bb_upper', 9999999):
                    all_signals['squeeze'].append({
                        'idx': i, 'time': row['datetime'], 'direction': 'long',
                        'entry_price': close, 'type': 'squeeze',
                    })
                elif close < row['bb_lower'] and prev['close'] >= prev.get('bb_lower', 0):
                    all_signals['squeeze'].append({
                        'idx': i, 'time': row['datetime'], 'direction': 'short',
                        'entry_price': close, 'type': 'squeeze',
                    })

        # ------ 信号2: 20根新高/新低突破 ------
        if not pd.isna(row['high_20']) and not pd.isna(prev.get('high_20', np.nan)):
            prev_high20 = df.iloc[i-1]['high'].max() if i > 0 else 0
            # 收盘价创20根新高
            if close >= row['high_20'] and prev['close'] < df.iloc[max(0,i-1)]['high_20']:
                all_signals['breakout_20'].append({
                    'idx': i, 'time': row['datetime'], 'direction': 'long',
                    'entry_price': close, 'type': 'breakout_20',
                })
            # 收盘价创20根新低
            if close <= row['low_20'] and prev['close'] > df.iloc[max(0,i-1)]['low_20']:
                all_signals['breakout_20'].append({
                    'idx': i, 'time': row['datetime'], 'direction': 'short',
                    'entry_price': close, 'type': 'breakout_20',
                })

        # ------ 信号3: 40根新高/新低突破 ------
        if not pd.isna(row['high_40']):
            if close >= row['high_40'] and prev['close'] < df.iloc[max(0,i-1)]['high_40']:
                all_signals['breakout_40'].append({
                    'idx': i, 'time': row['datetime'], 'direction': 'long',
                    'entry_price': close, 'type': 'breakout_40',
                })
            if close <= row['low_40'] and prev['close'] > df.iloc[max(0,i-1)]['low_40']:
                all_signals['breakout_40'].append({
                    'idx': i, 'time': row['datetime'], 'direction': 'short',
                    'entry_price': close, 'type': 'breakout_40',
                })

        # ------ 信号4: ATR突然放大 ------
        if not pd.isna(row['atr_ratio']):
            prev_ratio = df.iloc[i-1]['atr_ratio'] if not pd.isna(df.iloc[i-1]['atr_ratio']) else 1
            # ATR比率从<1.2突然跳到>1.5（波动率突然扩张）
            if row['atr_ratio'] > 1.5 and prev_ratio < 1.3:
                all_signals['atr_expand'].append({
                    'idx': i, 'time': row['datetime'], 'direction': direction,
                    'entry_price': close, 'type': 'atr_expand',
                })

        # ------ 信号5: 均线排列刚形成 ------
        # 多头排列: ma10 > ma20 > ma60, 空头排列: ma10 < ma20 < ma60
        ma_aligned_up = ma10 > ma20 > ma60
        ma_aligned_down = ma10 < ma20 < ma60
        curr_aligned = ma_aligned_up or ma_aligned_down

        if curr_aligned and not prev_ma_aligned:
            # 刚刚形成排列
            d = 'long' if ma_aligned_up else 'short'
            all_signals['ma_align'].append({
                'idx': i, 'time': row['datetime'], 'direction': d,
                'entry_price': close, 'type': 'ma_align',
            })
        prev_ma_aligned = curr_aligned

        # ------ 信号6: 缩量后放量突破 ------
        if not pd.isna(row['vol_ratio']):
            if row['vol_ratio'] < 0.7:
                low_vol_count += 1
            else:
                if low_vol_count >= 3 and row['vol_ratio'] > 1.5:
                    # 连续3根以上缩量后突然放量
                    # 方向看当根K线
                    d = 'long' if close > row['open'] else 'short'
                    all_signals['vol_break'].append({
                        'idx': i, 'time': row['datetime'], 'direction': d,
                        'entry_price': close, 'type': 'vol_break',
                    })
                low_vol_count = 0

        # ------ 信号7: B类回调 ------
        result = detector.process_bar(
            close=close, high=row['high'], low=row['low'],
            ma_fast=ma10, ma_slow=ma20,
        )
        if result is not None:
            all_signals['b_pullback'].append({
                'idx': i, 'time': row['datetime'], 'direction': result.direction,
                'entry_price': result.entry_price, 'type': 'b_pullback',
                'pb_bars': result.pullback_bars,
            })

    return all_signals


def dedup_signals(signals, min_gap=6):
    """同类型信号去重：6根内只保留第一个"""
    if not signals:
        return signals
    deduped = [signals[0]]
    for s in signals[1:]:
        if s['idx'] - deduped[-1]['idx'] >= min_gap:
            deduped.append(s)
    return deduped


def main():
    api = TqApi(auth=TqAuth(cfg.TQ_ACCOUNT, cfg.TQ_PASSWORD))

    # 存储所有品种所有信号
    grand = {}  # type -> list of signals (across all symbols)
    type_names = {
        'squeeze': '布林压缩突破',
        'breakout_20': '20根新高低',
        'breakout_40': '40根新高低',
        'atr_expand': 'ATR放大',
        'ma_align': '均线排列',
        'vol_break': '缩量放量',
        'b_pullback': 'B类回调',
    }

    for sig_type in type_names:
        grand[sig_type] = {'train': [], 'test': [], 'all': []}

    per_symbol = {}

    for sym_key, sym_info in SYMBOLS.items():
        print(f"\n{'='*55}")
        print(f"  {sym_info['name']} ({sym_key})")
        print(f"{'='*55}")

        df = get_klines(api, sym_info['tq'], sym_key, period_min=10, days=170)
        cutoff120 = df['datetime'].iloc[-1] - timedelta(days=120)
        df = df[df['datetime'] >= cutoff120].copy().reset_index(drop=True)
        df = prepare_df(df)

        midpoint = df['datetime'].iloc[-1] - timedelta(days=60)

        # 检测信号
        all_sigs = detect_signals(df)

        sym_summary = {}

        for sig_type, sigs in all_sigs.items():
            # 去重
            sigs = dedup_signals(sigs)

            # 计算MFE + 附加特征
            for s in sigs:
                s['mfe'] = calc_mfe_all(df, s['idx'], s['direction'])
                s['mfe_8h'] = s['mfe'][48]
                s['is_boom'] = s['mfe_8h'] >= BOOM_PCT
                s['symbol'] = sym_key
                s['sym_name'] = sym_info['name']

                # 附加过滤因子
                row = df.iloc[s['idx']]
                s['dist_ma60'] = row['dist_ma60'] if not pd.isna(row['dist_ma60']) else 0
                s['bb_width_pct'] = row['bb_width_pct'] if not pd.isna(row['bb_width_pct']) else 0.5
                s['atr_ratio'] = row['atr_ratio'] if not pd.isna(row['atr_ratio']) else 1.0
                s['range_10_rank'] = row['range_10_rank'] if not pd.isna(row['range_10_rank']) else 0.5

            train = [s for s in sigs if s['time'] < midpoint]
            test = [s for s in sigs if s['time'] >= midpoint]

            grand[sig_type]['train'].extend(train)
            grand[sig_type]['test'].extend(test)
            grand[sig_type]['all'].extend(sigs)

            n_boom = sum(1 for s in sigs if s['is_boom'])
            sym_summary[sig_type] = {'total': len(sigs), 'boom': n_boom, 'train': len(train), 'test': len(test)}

        # 品种级汇总
        print(f"  {'信号类型':<14} {'总数':>5} {'爆发':>5} {'爆发率':>7} | {'MFE_2h':>7} {'MFE_4h':>7} {'MFE_8h':>7}")
        print(f"  {'-'*65}")
        for sig_type, stats in sym_summary.items():
            sigs = [s for s in all_sigs[sig_type] if s.get('mfe')]  # only enriched ones
            # 用去重后的
            sigs_d = dedup_signals(all_sigs[sig_type])
            sigs_d = [s for s in sigs_d if 'mfe' in s]
            if not sigs_d:
                print(f"  {type_names[sig_type]:<14} {0:>5}")
                continue
            n = len(sigs_d)
            boom = sum(1 for s in sigs_d if s['is_boom'])
            rate = boom / n * 100 if n > 0 else 0
            mfe2 = np.mean([s['mfe'][12] for s in sigs_d])
            mfe4 = np.mean([s['mfe'][24] for s in sigs_d])
            mfe8 = np.mean([s['mfe'][48] for s in sigs_d])
            print(f"  {type_names[sig_type]:<14} {n:>5} {boom:>5} {rate:>6.1f}% | {mfe2:>6.2f}% {mfe4:>6.2f}% {mfe8:>6.2f}%")

        per_symbol[sym_key] = sym_summary

    api.close()

    # ============================================================
    # 全品种汇总
    # ============================================================
    print(f"\n\n{'='*80}")
    print(f"  全品种汇总：各信号类型对比")
    print(f"{'='*80}")
    print(f"  {'信号类型':<14} {'总数':>5} {'爆发':>5} {'爆发率':>7} | {'MFE_2h':>7} {'MFE_4h':>7} {'MFE_8h':>7} {'MFE_12h':>8}")
    print(f"  {'-'*75}")

    type_ranking = []
    for sig_type in type_names:
        sigs = grand[sig_type]['all']
        if not sigs:
            continue
        n = len(sigs)
        boom = sum(1 for s in sigs if s['is_boom'])
        rate = boom / n * 100
        mfe2 = np.mean([s['mfe'][12] for s in sigs])
        mfe4 = np.mean([s['mfe'][24] for s in sigs])
        mfe8 = np.mean([s['mfe'][48] for s in sigs])
        mfe12 = np.mean([s['mfe'][72] for s in sigs])
        star = ' ★' if rate > 40 else (' ☆' if rate > 30 else '')
        print(f"  {type_names[sig_type]:<14} {n:>5} {boom:>5} {rate:>6.1f}%{star} | "
              f"{mfe2:>6.2f}% {mfe4:>6.2f}% {mfe8:>6.2f}% {mfe12:>7.2f}%")
        type_ranking.append((sig_type, rate, mfe8, n))

    type_ranking.sort(key=lambda x: -x[2])  # 按MFE_8h排序

    # ============================================================
    # 样本内/样本外验证
    # ============================================================
    print(f"\n\n{'='*80}")
    print(f"  样本内(前60天) vs 样本外(后60天)")
    print(f"{'='*80}")
    print(f"  {'信号类型':<14} | {'训练':>5} {'爆发率':>7} {'MFE_8h':>7} | {'测试':>5} {'爆发率':>7} {'MFE_8h':>7} | {'一致?':>5}")
    print(f"  {'-'*80}")

    for sig_type in type_names:
        train = grand[sig_type]['train']
        test = grand[sig_type]['test']
        if not train or not test:
            continue
        tr_rate = sum(1 for s in train if s['is_boom']) / len(train) * 100
        te_rate = sum(1 for s in test if s['is_boom']) / len(test) * 100
        tr_mfe = np.mean([s['mfe_8h'] for s in train])
        te_mfe = np.mean([s['mfe_8h'] for s in test])

        # 判断一致性
        if abs(tr_rate - te_rate) < 15 and tr_mfe > 0 and te_mfe > 0:
            verdict = '✓'
        elif te_rate > tr_rate * 0.5:
            verdict = '~'
        else:
            verdict = '✗'

        print(f"  {type_names[sig_type]:<14} | {len(train):>5} {tr_rate:>6.1f}% {tr_mfe:>6.2f}% | "
              f"{len(test):>5} {te_rate:>6.1f}% {te_mfe:>6.2f}% | {verdict:>5}")

    # ============================================================
    # 因子过滤：对每种信号加dist_ma60过滤
    # ============================================================
    print(f"\n\n{'='*80}")
    print(f"  因子增强：各信号 + 距60MA过滤")
    print(f"{'='*80}")

    for sig_type in type_names:
        test = grand[sig_type]['test']
        if len(test) < 10:
            continue

        # 用训练集定阈值
        train = grand[sig_type]['train']
        if len(train) < 5:
            continue
        threshold = np.percentile([s['dist_ma60'] for s in train], 60)

        te_pass = [s for s in test if s['dist_ma60'] >= threshold]
        te_fail = [s for s in test if s['dist_ma60'] < threshold]

        if not te_pass or not te_fail:
            continue

        base_rate = sum(1 for s in test if s['is_boom']) / len(test) * 100
        pass_rate = sum(1 for s in te_pass if s['is_boom']) / len(te_pass) * 100
        fail_rate = sum(1 for s in te_fail if s['is_boom']) / len(te_fail) * 100
        pass_mfe = np.mean([s['mfe_8h'] for s in te_pass])
        fail_mfe = np.mean([s['mfe_8h'] for s in te_fail])

        effective = '✓' if pass_rate - fail_rate > 10 else ('~' if pass_rate > fail_rate else '✗')
        print(f"\n  {type_names[sig_type]} (阈值={threshold:.2f}%):")
        print(f"    无过滤:   {len(test):>3}个 爆发率={base_rate:.1f}% MFE={np.mean([s['mfe_8h'] for s in test]):.2f}%")
        print(f"    达标:     {len(te_pass):>3}个 爆发率={pass_rate:.1f}% MFE={pass_mfe:.2f}%")
        print(f"    不达标:   {len(te_fail):>3}个 爆发率={fail_rate:.1f}% MFE={fail_mfe:.2f}%")
        print(f"    过滤效果: {effective}")

    # ============================================================
    # 信号组合：不同信号共振
    # ============================================================
    print(f"\n\n{'='*80}")
    print(f"  信号共振：同一时间窗口内多种信号重叠")
    print(f"{'='*80}")

    # 检查：某个信号出现前后3根K线内，是否有其他类型信号也出现
    for primary_type in ['squeeze', 'breakout_40', 'b_pullback']:
        primary = grand[primary_type]['test']
        if not primary:
            continue

        with_resonance = []
        without_resonance = []

        other_types = [t for t in type_names if t != primary_type]
        for s in primary:
            # 检查前后3根内有没有其他信号
            has_other = False
            for ot in other_types:
                for os_ in grand[ot]['test']:
                    if os_['symbol'] == s['symbol'] and abs(os_['idx'] - s['idx']) <= 3:
                        if os_['direction'] == s['direction']:  # 同方向才算共振
                            has_other = True
                            break
                if has_other:
                    break

            if has_other:
                with_resonance.append(s)
            else:
                without_resonance.append(s)

        if with_resonance and without_resonance:
            r_rate = sum(1 for s in with_resonance if s['is_boom']) / len(with_resonance) * 100
            nr_rate = sum(1 for s in without_resonance if s['is_boom']) / len(without_resonance) * 100
            r_mfe = np.mean([s['mfe_8h'] for s in with_resonance])
            nr_mfe = np.mean([s['mfe_8h'] for s in without_resonance])
            print(f"\n  {type_names[primary_type]}:")
            print(f"    有共振: {len(with_resonance):>3}个 爆发率={r_rate:.1f}% MFE={r_mfe:.2f}%")
            print(f"    无共振: {len(without_resonance):>3}个 爆发率={nr_rate:.1f}% MFE={nr_mfe:.2f}%")

    # ============================================================
    # 图表
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 图1: 各信号类型MFE_8h对比（箱线图）
    ax = axes[0, 0]
    box_data = []
    box_labels = []
    for sig_type, cn_name in type_names.items():
        sigs = grand[sig_type]['all']
        if len(sigs) >= 5:
            box_data.append([s['mfe_8h'] for s in sigs])
            box_labels.append(f"{cn_name}\n(N={len(sigs)})")
    if box_data:
        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, showmeans=True,
                        showfliers=False)
        colors = ['#4CAF50', '#2196F3', '#1565C0', '#FF9800', '#9C27B0', '#f44336', '#607D8B']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
    ax.set_title('各信号类型 MFE_8h 分布', fontweight='bold')
    ax.set_ylabel('MFE %')
    ax.grid(True, alpha=0.3)

    # 图2: 爆发率对比（柱状图）
    ax = axes[0, 1]
    names = []
    train_rates = []
    test_rates = []
    for sig_type in type_names:
        train = grand[sig_type]['train']
        test = grand[sig_type]['test']
        if not train or not test:
            continue
        names.append(type_names[sig_type])
        train_rates.append(sum(1 for s in train if s['is_boom']) / len(train) * 100)
        test_rates.append(sum(1 for s in test if s['is_boom']) / len(test) * 100)

    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, train_rates, w, label='训练(前60天)', color='#90CAF9')
    ax.bar(x + w/2, test_rates, w, label='测试(后60天)', color='#4CAF50')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=15)
    ax.set_ylabel('爆发率 %')
    ax.set_title('训练 vs 测试 爆发率', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 图3: 信号数量 vs 平均MFE（气泡图）
    ax = axes[1, 0]
    for sig_type in type_names:
        sigs = grand[sig_type]['all']
        if not sigs:
            continue
        n = len(sigs)
        avg_mfe = np.mean([s['mfe_8h'] for s in sigs])
        boom_rate = sum(1 for s in sigs if s['is_boom']) / n * 100
        ax.scatter(n, avg_mfe, s=boom_rate * 5, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.annotate(type_names[sig_type], (n, avg_mfe), fontsize=8,
                   textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel('信号数量')
    ax.set_ylabel('平均 MFE_8h %')
    ax.set_title('信号数量 vs 质量（气泡大小=爆发率）', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 图4: 各品种各信号爆发率热力图
    ax = axes[1, 1]
    sym_names = [SYMBOLS[k]['name'] for k in SYMBOLS]
    sig_names = [type_names[t] for t in type_names]
    heat_data = np.zeros((len(SYMBOLS), len(type_names)))
    for i, sym_key in enumerate(SYMBOLS):
        for j, sig_type in enumerate(type_names):
            sigs = [s for s in grand[sig_type]['all'] if s['symbol'] == sym_key]
            if sigs:
                heat_data[i, j] = sum(1 for s in sigs if s['is_boom']) / len(sigs) * 100
            else:
                heat_data[i, j] = np.nan

    im = ax.imshow(heat_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=60)
    ax.set_xticks(range(len(sig_names)))
    ax.set_xticklabels(sig_names, fontsize=7, rotation=30, ha='right')
    ax.set_yticks(range(len(sym_names)))
    ax.set_yticklabels(sym_names, fontsize=9)
    for i in range(len(sym_names)):
        for j in range(len(sig_names)):
            val = heat_data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.0f}%', ha='center', va='center', fontsize=7,
                       color='white' if val > 40 else 'black')
    ax.set_title('各品种×各信号 爆发率(%)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='爆发率%')

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'output', 'signals_compare.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图表: {out_path}")


if __name__ == '__main__':
    main()
