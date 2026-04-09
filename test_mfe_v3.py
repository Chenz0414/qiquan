# -*- coding: utf-8 -*-
"""
MFE v3：样本内/样本外验证
——前60天定规则，后60天验证
——核心因子：距60MA距离
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

MFE_WINDOW = 48  # 8小时
BOOM_PCT = 3.0   # 爆发定义


def get_signals_with_features(df):
    """检测信号并附加特征"""
    atr_raw = df['high'].rolling(14).max() - df['low'].rolling(14).min()
    atr = atr_raw / 14
    df['atr'] = atr
    df['ma60'] = df['close'].rolling(60).mean()
    df['vol_ma'] = df['volume'].rolling(20).mean()

    ma10 = df['ma_fast']
    ma20 = df['ma_slow']
    df['ma_slope'] = (ma10 - ma10.shift(5)) / atr
    df['ma_width'] = (ma10 - ma20) / atr

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
        if result is None:
            continue

        # 距60MA距离
        ma60 = row['ma60'] if not pd.isna(row['ma60']) else row['close']
        dist_ma60 = abs(row['close'] - ma60) / ma60 * 100

        # MA斜率
        ma_slope = abs(row['ma_slope']) if not pd.isna(row['ma_slope']) else 0

        # 成交量比
        vol_ma = row['vol_ma'] if not pd.isna(row['vol_ma']) and row['vol_ma'] > 0 else 1
        volume_ratio = row['volume'] / vol_ma

        # 趋势年龄
        trend_age = 0
        is_above = ma10.iloc[i] > ma20.iloc[i] if not pd.isna(ma10.iloc[i]) else True
        for j in range(i - 1, max(i - 200, 0), -1):
            if pd.isna(ma10.iloc[j]) or pd.isna(ma20.iloc[j]):
                break
            if (ma10.iloc[j] > ma20.iloc[j]) == is_above:
                trend_age += 1
            else:
                break

        # MFE
        end_idx = min(i + MFE_WINDOW, len(df) - 1)
        if i + 1 <= end_idx:
            seg = df.iloc[i + 1: end_idx + 1]
            if result.direction == 'short':
                mfe = (result.entry_price - seg['low'].min()) / result.entry_price * 100
            else:
                mfe = (seg['high'].max() - result.entry_price) / result.entry_price * 100
            mfe = max(mfe, 0)
        else:
            mfe = 0

        signals.append({
            'idx': i, 'time': row['datetime'],
            'direction': result.direction,
            'entry_price': result.entry_price,
            'pb_bars': result.pullback_bars,
            'dist_ma60': dist_ma60,
            'ma_slope': ma_slope,
            'volume_ratio': volume_ratio,
            'trend_age': trend_age,
            'mfe_8h': mfe,
            'is_boom': mfe >= BOOM_PCT,
        })

    return signals


def print_split_test(label, train_sigs, test_sigs, factor, threshold):
    """打印样本内外的对比"""
    # 样本内
    tr_pass = [s for s in train_sigs if s[factor] >= threshold]
    tr_fail = [s for s in train_sigs if s[factor] < threshold]
    tr_pass_boom = sum(1 for s in tr_pass if s['is_boom']) if tr_pass else 0
    tr_fail_boom = sum(1 for s in tr_fail if s['is_boom']) if tr_fail else 0
    tr_pass_rate = tr_pass_boom / len(tr_pass) * 100 if tr_pass else 0
    tr_fail_rate = tr_fail_boom / len(tr_fail) * 100 if tr_fail else 0
    tr_pass_mfe = np.mean([s['mfe_8h'] for s in tr_pass]) if tr_pass else 0
    tr_fail_mfe = np.mean([s['mfe_8h'] for s in tr_fail]) if tr_fail else 0

    # 样本外
    te_pass = [s for s in test_sigs if s[factor] >= threshold]
    te_fail = [s for s in test_sigs if s[factor] < threshold]
    te_pass_boom = sum(1 for s in te_pass if s['is_boom']) if te_pass else 0
    te_fail_boom = sum(1 for s in te_fail if s['is_boom']) if te_fail else 0
    te_pass_rate = te_pass_boom / len(te_pass) * 100 if te_pass else 0
    te_fail_rate = te_fail_boom / len(te_fail) * 100 if te_fail else 0
    te_pass_mfe = np.mean([s['mfe_8h'] for s in te_pass]) if te_pass else 0
    te_fail_mfe = np.mean([s['mfe_8h'] for s in te_fail]) if te_fail else 0

    print(f"\n  {label} (阈值: {factor} >= {threshold:.2f})")
    print(f"  {'':>12} {'样本内(前60天)':>25}  {'样本外(后60天)':>25}")
    print(f"  {'':>12} {'信号':>5} {'爆发':>5} {'爆发率':>7} {'MFE均':>7}  {'信号':>5} {'爆发':>5} {'爆发率':>7} {'MFE均':>7}")
    print(f"  {'达标':>12} {len(tr_pass):>5} {tr_pass_boom:>5} {tr_pass_rate:>6.1f}% {tr_pass_mfe:>6.2f}%  {len(te_pass):>5} {te_pass_boom:>5} {te_pass_rate:>6.1f}% {te_pass_mfe:>6.2f}%")
    print(f"  {'不达标':>12} {len(tr_fail):>5} {tr_fail_boom:>5} {tr_fail_rate:>6.1f}% {tr_fail_mfe:>6.2f}%  {len(te_fail):>5} {te_fail_boom:>5} {te_fail_rate:>6.1f}% {te_fail_mfe:>6.2f}%")

    # 判断是否样本外依然有效
    train_diff = tr_pass_rate - tr_fail_rate
    test_diff = te_pass_rate - te_fail_rate
    if test_diff > 5 and train_diff > 5:
        verdict = "✓ 样本外有效"
    elif test_diff > 0:
        verdict = "~ 方向正确但弱"
    else:
        verdict = "✗ 样本外失效"
    print(f"  {'判定':>12} 样本内差={train_diff:+.1f}%  样本外差={test_diff:+.1f}%  → {verdict}")
    return train_diff, test_diff


def main():
    api = TqApi(auth=TqAuth(cfg.TQ_ACCOUNT, cfg.TQ_PASSWORD))

    all_train = []
    all_test = []
    per_symbol = {}

    for sym_key, sym_info in SYMBOLS.items():
        print(f"加载 {sym_info['name']}...", end=' ')
        df = get_klines(api, sym_info['tq'], sym_key, period_min=10, days=170)

        # 120天窗口
        cutoff120 = df['datetime'].iloc[-1] - timedelta(days=120)
        df = df[df['datetime'] >= cutoff120].copy().reset_index(drop=True)

        signals = get_signals_with_features(df)

        # 60/60分割
        midpoint = df['datetime'].iloc[-1] - timedelta(days=60)
        train = [s for s in signals if s['time'] < midpoint]
        test = [s for s in signals if s['time'] >= midpoint]

        per_symbol[sym_key] = {
            'name': sym_info['name'], 'train': train, 'test': test, 'all': signals,
        }
        all_train.extend(train)
        all_test.extend(test)
        print(f"共{len(signals)}信号 (训练{len(train)} / 测试{len(test)})")

    api.close()

    # ============================================================
    # 1. 全品种汇总：样本内定阈值，样本外验证
    # ============================================================
    print(f"\n\n{'='*80}")
    print(f"  样本内/样本外验证")
    print(f"  训练集: 前60天 ({len(all_train)}信号)")
    print(f"  测试集: 后60天 ({len(all_test)}信号)")
    print(f"{'='*80}")

    # 用训练集的P50/P67/P75定阈值
    factors_to_test = [
        ('dist_ma60', '距60MA距离'),
        ('ma_slope', 'MA斜率'),
        ('volume_ratio', '成交量比(反向)'),
        ('trend_age', '趋势年龄'),
    ]

    results_summary = []

    for factor, label in factors_to_test:
        vals = [s[factor] for s in all_train]
        if not vals:
            continue

        print(f"\n{'─'*70}")
        print(f"  因子: {label}")
        print(f"{'─'*70}")

        for pct_label, pct in [('P50', 50), ('P67', 67), ('P75', 75)]:
            threshold = np.percentile(vals, pct)
            tr_diff, te_diff = print_split_test(
                f"{label}({pct_label})", all_train, all_test, factor, threshold)
            results_summary.append({
                'factor': label, 'pct': pct_label, 'threshold': threshold,
                'train_diff': tr_diff, 'test_diff': te_diff,
            })

    # ============================================================
    # 2. 分品种验证最强因子
    # ============================================================
    # 用全局训练集的P67作为dist_ma60的阈值
    dist_threshold = np.percentile([s['dist_ma60'] for s in all_train], 67)

    print(f"\n\n{'='*80}")
    print(f"  分品种验证：距60MA距离 >= {dist_threshold:.2f}%")
    print(f"{'='*80}")

    sym_results = []

    for sym_key, data in per_symbol.items():
        train = data['train']
        test = data['test']
        name = data['name']

        if len(test) < 5:
            print(f"\n  {name}: 测试集样本不足({len(test)}个), 跳过")
            continue

        tr_pass = [s for s in train if s['dist_ma60'] >= dist_threshold]
        tr_fail = [s for s in train if s['dist_ma60'] < dist_threshold]
        te_pass = [s for s in test if s['dist_ma60'] >= dist_threshold]
        te_fail = [s for s in test if s['dist_ma60'] < dist_threshold]

        print(f"\n  {name}:")
        print(f"    训练: 达标{len(tr_pass)}个 爆发{sum(1 for s in tr_pass if s['is_boom'])}个"
              f"({sum(1 for s in tr_pass if s['is_boom'])/len(tr_pass)*100:.0f}%) | "
              f"不达标{len(tr_fail)}个 爆发{sum(1 for s in tr_fail if s['is_boom'])}个"
              f"({sum(1 for s in tr_fail if s['is_boom'])/len(tr_fail)*100:.0f}%)" if tr_pass and tr_fail else "")

        if te_pass and te_fail:
            pass_rate = sum(1 for s in te_pass if s['is_boom']) / len(te_pass) * 100
            fail_rate = sum(1 for s in te_fail if s['is_boom']) / len(te_fail) * 100
            pass_mfe = np.mean([s['mfe_8h'] for s in te_pass])
            fail_mfe = np.mean([s['mfe_8h'] for s in te_fail])
            diff = pass_rate - fail_rate
            print(f"    测试: 达标{len(te_pass)}个 爆发率={pass_rate:.1f}% MFE={pass_mfe:.2f}% | "
                  f"不达标{len(te_fail)}个 爆发率={fail_rate:.1f}% MFE={fail_mfe:.2f}%")
            valid = "✓" if diff > 5 else ("~" if diff > 0 else "✗")
            print(f"    → 达标组比不达标组高 {diff:+.1f}%  {valid}")
            sym_results.append({'name': name, 'diff': diff, 'pass_rate': pass_rate,
                               'fail_rate': fail_rate, 'pass_mfe': pass_mfe, 'fail_mfe': fail_mfe,
                               'n_test': len(te_pass) + len(te_fail)})
        elif te_pass:
            pass_rate = sum(1 for s in te_pass if s['is_boom']) / len(te_pass) * 100
            print(f"    测试: 达标{len(te_pass)}个 爆发率={pass_rate:.1f}% (无不达标组对照)")
        elif te_fail:
            fail_rate = sum(1 for s in te_fail if s['is_boom']) / len(te_fail) * 100
            print(f"    测试: 不达标{len(te_fail)}个 爆发率={fail_rate:.1f}% (无达标组)")

    # ============================================================
    # 3. 多因子组合的样本外测试
    # ============================================================
    # 成交量是反向因子（低成交量好），用 < P33
    vol_threshold = np.percentile([s['volume_ratio'] for s in all_train], 33)

    print(f"\n\n{'='*80}")
    print(f"  多因子组合样本外测试")
    print(f"  规则：距60MA >= {dist_threshold:.2f}% 且 成交量比 < {vol_threshold:.2f}")
    print(f"{'='*80}")

    combo_train = [s for s in all_train if s['dist_ma60'] >= dist_threshold and s['volume_ratio'] < vol_threshold]
    combo_test = [s for s in all_test if s['dist_ma60'] >= dist_threshold and s['volume_ratio'] < vol_threshold]
    rest_train = [s for s in all_train if s not in combo_train]
    rest_test = [s for s in all_test if s not in combo_test]

    def boom_stats(sigs):
        if not sigs:
            return 0, 0, 0
        n = len(sigs)
        b = sum(1 for s in sigs if s['is_boom'])
        mfe = np.mean([s['mfe_8h'] for s in sigs])
        return n, b/n*100, mfe

    n, br, mfe = boom_stats(combo_train)
    print(f"  训练-通过: {n:>3}信号 爆发率={br:.1f}% MFE={mfe:.2f}%")
    n, br, mfe = boom_stats(rest_train)
    print(f"  训练-其余: {n:>3}信号 爆发率={br:.1f}% MFE={mfe:.2f}%")
    n, br, mfe = boom_stats(combo_test)
    print(f"  测试-通过: {n:>3}信号 爆发率={br:.1f}% MFE={mfe:.2f}%")
    n, br, mfe = boom_stats(rest_test)
    print(f"  测试-其余: {n:>3}信号 爆发率={br:.1f}% MFE={mfe:.2f}%")

    # ============================================================
    # 4. 不同阈值的稳健性（距60MA）
    # ============================================================
    print(f"\n\n{'='*80}")
    print(f"  稳健性检查：距60MA不同阈值的样本外表现")
    print(f"{'='*80}")
    print(f"  {'阈值':>8} {'测试-达标':>10} {'爆发率':>8} {'MFE均值':>8} | {'测试-不达标':>12} {'爆发率':>8} {'MFE均值':>8}")

    for pct in [40, 50, 60, 67, 75, 80]:
        thr = np.percentile([s['dist_ma60'] for s in all_train], pct)
        te_pass = [s for s in all_test if s['dist_ma60'] >= thr]
        te_fail = [s for s in all_test if s['dist_ma60'] < thr]
        if te_pass and te_fail:
            pr = sum(1 for s in te_pass if s['is_boom']) / len(te_pass) * 100
            fr = sum(1 for s in te_fail if s['is_boom']) / len(te_fail) * 100
            pm = np.mean([s['mfe_8h'] for s in te_pass])
            fm = np.mean([s['mfe_8h'] for s in te_fail])
            marker = ' ★' if pr - fr > 10 else ''
            print(f"  P{pct} {thr:>5.2f}% {len(te_pass):>8}个 {pr:>7.1f}% {pm:>7.2f}% | "
                  f"{len(te_fail):>10}个 {fr:>7.1f}% {fm:>7.2f}%{marker}")

    # ============================================================
    # 图表
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for idx, (sym_key, data) in enumerate(per_symbol.items()):
        ax = axes[idx // 3, idx % 3]
        train = data['train']
        test = data['test']
        name = data['name']

        # 训练集：灰色
        tr_x = [s['dist_ma60'] for s in train]
        tr_y = [s['mfe_8h'] for s in train]
        ax.scatter(tr_x, tr_y, c='#ccc', s=15, alpha=0.5, label='训练(前60天)')

        # 测试集：按爆发着色
        for s in test:
            color = '#4CAF50' if s['is_boom'] else '#ff9800'
            ax.scatter(s['dist_ma60'], s['mfe_8h'], c=color, s=30, alpha=0.7,
                      edgecolors='white', linewidth=0.3)

        # 阈值线
        ax.axvline(x=dist_threshold, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(y=BOOM_PCT, color='blue', linestyle='--', alpha=0.3, linewidth=1)

        # 标注
        te_pass = [s for s in test if s['dist_ma60'] >= dist_threshold]
        te_fail = [s for s in test if s['dist_ma60'] < dist_threshold]
        if te_pass:
            pr = sum(1 for s in te_pass if s['is_boom']) / len(te_pass) * 100
            ax.text(0.98, 0.98, f'达标: {pr:.0f}%爆发\n(N={len(te_pass)})',
                   transform=ax.transAxes, ha='right', va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='#4CAF5030'))
        if te_fail:
            fr = sum(1 for s in te_fail if s['is_boom']) / len(te_fail) * 100
            ax.text(0.02, 0.98, f'不达标: {fr:.0f}%爆发\n(N={len(te_fail)})',
                   transform=ax.transAxes, ha='left', va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='#ff980030'))

        ax.set_title(f'{name} (测试集=后60天)', fontweight='bold')
        ax.set_xlabel('距60MA距离 (%)')
        ax.set_ylabel('MFE_8h (%)')
        ax.grid(True, alpha=0.2)

    plt.suptitle('样本外验证：距60MA距离 vs MFE（红线=阈值，绿=爆发，橙=未爆发）',
                fontweight='bold', fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'output', 'mfe_oos_test.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图表: {out_path}")

    # ============================================================
    # 最终结论
    # ============================================================
    print(f"\n\n{'='*80}")
    print(f"  最终结论")
    print(f"{'='*80}")
    valid_count = sum(1 for r in sym_results if r['diff'] > 5)
    weak_count = sum(1 for r in sym_results if 0 < r['diff'] <= 5)
    fail_count = sum(1 for r in sym_results if r['diff'] <= 0)
    print(f"  6个品种样本外验证结果:")
    print(f"    有效(差>5%): {valid_count}个")
    print(f"    方向正确:    {weak_count}个")
    print(f"    失效:        {fail_count}个")
    if sym_results:
        avg_diff = np.mean([r['diff'] for r in sym_results])
        print(f"    平均提升: {avg_diff:+.1f}%")


if __name__ == '__main__':
    main()
