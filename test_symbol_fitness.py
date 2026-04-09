# -*- coding: utf-8 -*-
"""
品种适配度研究：稳定性 + 滚动预测 + 截面ER排名
================================================
Q1: 品种在策略下的表现是一直差/好，还是周期性波动？（4×30天窗口）
Q2: 用过去7天表现排名能否预测未来7天？（滚动walk-forward）
Q3: 信号触发时品种的ER(20)截面排名是否有预测力？
"""

import sys
import io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from data_loader import load_all, sym_name, tick_size, BARS_PER_DAY
from signal_core import SignalDetector, ExitTracker, SYMBOL_CONFIGS, DEFAULT_STOP_TICKS
from stats_utils import calc_ev
from chart_engine import render_chart, get_chart_js
from report_engine import Report

# ============ 常量 ============
LAST_DAYS = 120
WINDOW_DAYS = 30
N_WINDOWS = LAST_DAYS // WINDOW_DAYS  # 4
BLOCK_DAYS = 7
BLOCK_BARS = BLOCK_DAYS * BARS_PER_DAY  # 399

SCENARIO_EXIT = {1: 'S2', 2: 'S2', 3: 'S5.1'}


def classify_scenario(sig_type, er20, deviation_atr):
    """判断信号属于哪个场景，返回场景编号或None"""
    if er20 >= 0.7:
        return None
    if sig_type == 'A' and er20 >= 0.5 and deviation_atr >= 1.0:
        return 1
    if sig_type == 'C' and deviation_atr >= 2.0:
        return 2
    if sig_type == 'B' and er20 >= 0.5 and deviation_atr >= 0.1 and deviation_atr < 0.3:
        return 3
    return None


# ============ 1. 信号收集 ============

def build_er40_breadth(all_data):
    """
    预计算ER40变化(12根)广度面板：每个bar_offset有多少品种ER40Δ12>=0.14
    用于反向过滤：广度=0时不开仓
    """
    sym_er40 = {}
    for sym, df in all_data.items():
        n = len(df)
        analysis_start = max(0, n - LAST_DAYS * BARS_PER_DAY)
        col = 'er_40'
        if col in df.columns:
            sym_er40[sym] = pd.Series(
                df[col].iloc[analysis_start:].values,
                index=range(n - analysis_start))
    er40_panel = pd.DataFrame(sym_er40)
    er40_delta12 = er40_panel.diff(12)
    # 每个bar_offset：有多少品种ER40Δ12>=0.14
    breadth = (er40_delta12 >= 0.14).sum(axis=1)
    return breadth  # Series, index=bar_offset, value=count


def collect_all_trades(all_data):
    """对全品种跑信号检测+出场，返回交易列表"""
    all_trades = []

    # 预算ER40广度（反向过滤用）
    er40_breadth = build_er40_breadth(all_data)

    for sym_key, df in sorted(all_data.items()):
        n = len(df)
        analysis_start = max(0, n - LAST_DAYS * BARS_PER_DAY)
        ts = tick_size(sym_key)
        name = sym_name(sym_key)

        detector = SignalDetector(signal_types='ABC')

        for i in range(max(2, 130), n):
            row = df.iloc[i]
            if pd.isna(row['ema10']) or pd.isna(row['ema20']) or pd.isna(row['ema120']):
                continue
            if pd.isna(row['atr']) or row['atr'] <= 0:
                continue

            result = detector.process_bar(
                close=row['close'], high=row['high'], low=row['low'],
                ema10=row['ema10'], ema20=row['ema20'], ema120=row['ema120'],
            )

            if result is None or i < analysis_start:
                continue

            er20 = row.get('er_20', np.nan)
            if pd.isna(er20):
                er20 = 0
            atr = row['atr']
            deviation_atr = abs(result.entry_price - row['ema10']) / atr if atr > 0 else 0

            scenario = classify_scenario(result.signal_type, er20, deviation_atr)
            if scenario is None:
                continue

            # ER40广度反向过滤：0个品种ER40Δ12>=0.14时不开仓
            offset = i - analysis_start
            er40_b = int(er40_breadth.get(offset, 0)) if offset >= 0 else 0

            # 跑出场
            tracker = ExitTracker(
                direction=result.direction,
                entry_price=result.entry_price,
                pullback_extreme=result.pullback_extreme,
                tick_size=ts,
                stop_ticks=DEFAULT_STOP_TICKS,
            )

            target_strategy = SCENARIO_EXIT[scenario]
            target_pnl = None
            target_reason = 'backtest_end'
            exit_bar_idx = n - 1
            exit_price = df.iloc[-1]['close']

            for j in range(i + 1, n):
                bar = df.iloc[j]
                prev = df.iloc[j - 1]
                if pd.isna(bar['ema10']):
                    continue
                exits, _ = tracker.process_bar(
                    close=bar['close'], high=bar['high'], low=bar['low'],
                    ema10=bar['ema10'], prev_close=prev['close'],
                    prev_high=prev['high'], prev_low=prev['low'],
                )
                for ev in exits:
                    if ev.strategy == target_strategy:
                        target_pnl = ev.pnl_pct
                        target_reason = ev.exit_reason
                        exit_bar_idx = j
                        exit_price = ev.exit_price
                        break
                if target_pnl is not None:
                    break
                if tracker.all_done():
                    break

            # 如果目标策略没触发，force_close
            if target_pnl is None:
                forced = tracker.force_close(df.iloc[-1]['close'])
                for ev in forced:
                    if ev.strategy == target_strategy:
                        target_pnl = ev.pnl_pct
                        target_reason = ev.exit_reason
                        break
            if target_pnl is None:
                target_pnl = 0

            # 初始止损价
            if result.direction == 'long':
                stop_price = result.pullback_extreme - ts * DEFAULT_STOP_TICKS
            else:
                stop_price = result.pullback_extreme + ts * DEFAULT_STOP_TICKS

            all_trades.append({
                'sym_key': sym_key,
                'name': name,
                'bar_idx': i,
                'datetime': row['datetime'],
                'scenario': scenario,
                'type': result.signal_type,
                'direction': result.direction,
                'er_20': round(er20, 3),
                'deviation_atr': round(deviation_atr, 3),
                'pnl': round(target_pnl, 4),
                'exit_reason': target_reason,
                'entry_price': result.entry_price,
                'exit_price': exit_price,
                'exit_bar_idx': exit_bar_idx,
                'stop_price': stop_price,
                'pullback_extreme': result.pullback_extreme,
                'er40_breadth': er40_b,
                'filtered_er40b': er40_b == 0,  # True=被反向过滤掉
            })

        print(f"  {name}({sym_key}): {sum(1 for t in all_trades if t['sym_key']==sym_key)}笔")

    return all_trades


# ============ 2. Q1：品种时间稳定性 ============

def analyze_q1(all_trades, all_data):
    """4×30天窗口分析品种稳定性"""
    # 为每笔交易计算所在窗口
    results = {}  # {sym_key: {window_idx: [pnl, ...]}}

    for t in all_trades:
        if t['exit_reason'] == 'backtest_end':
            continue
        sym = t['sym_key']
        df = all_data[sym]
        n = len(df)
        analysis_start = max(0, n - LAST_DAYS * BARS_PER_DAY)
        window_bars = WINDOW_DAYS * BARS_PER_DAY
        w_idx = min((t['bar_idx'] - analysis_start) // window_bars, N_WINDOWS - 1)
        if w_idx < 0:
            continue
        results.setdefault(sym, {}).setdefault(w_idx, []).append(t['pnl'])

    # 计算每个(品种,窗口)的统计
    # 注意：用 sum_pnl 和 avg_pnl 作为主指标，而非 EV
    # EV 在全胜(WR=100%)时因 avg_loss=0.001 回退值会膨胀到数千，不可用于小样本
    heatmap = {}  # {sym_key: [stats_w0, stats_w1, stats_w2, stats_w3]}
    for sym in sorted(SYMBOL_CONFIGS.keys()):
        row = []
        for w in range(N_WINDOWS):
            pnls = results.get(sym, {}).get(w, [])
            if len(pnls) >= 1:
                st = calc_ev(pnls)
                row.append(st)
            else:
                row.append(None)
        heatmap[sym] = row

    # 稳定性分类 — 用 sum_pnl（累计盈亏%）判定，不用 EV
    # 理由：sum_pnl 不依赖 avg_loss 分母，小样本下依然有意义
    stability = {}
    for sym, row in heatmap.items():
        cells = [(r['sum_pnl'], r['N']) for r in row if r is not None and r['N'] >= 2]
        if len(cells) < 2:
            stability[sym] = '样本不足'
            continue
        pos_count = sum(1 for s, _ in cells if s > 0)
        neg_count = sum(1 for s, _ in cells if s <= 0)
        total_pnl = sum(s for s, _ in cells)
        total_n = sum(n for _, n in cells)
        avg_pnl_per_trade = total_pnl / total_n if total_n > 0 else 0
        if pos_count >= len(cells) * 0.75 and avg_pnl_per_trade > 0:
            stability[sym] = '稳定正向'
        elif neg_count >= len(cells) * 0.75:
            stability[sym] = '稳定负向'
        else:
            stability[sym] = '波动型'

    # 窗口间秩相关 — 用 avg_pnl 排名，不用 EV
    correlations = []
    for w in range(N_WINDOWS - 1):
        syms_both = []
        pnl_w1 = []
        pnl_w2 = []
        for sym in sorted(SYMBOL_CONFIGS.keys()):
            r1 = heatmap.get(sym, [None]*N_WINDOWS)[w]
            r2 = heatmap.get(sym, [None]*N_WINDOWS)[w+1]
            if r1 is not None and r2 is not None and r1['N'] >= 2 and r2['N'] >= 2:
                syms_both.append(sym)
                pnl_w1.append(r1['avg_pnl'])
                pnl_w2.append(r2['avg_pnl'])
        if len(syms_both) >= 5:
            s1 = pd.Series(pnl_w1).rank()
            s2 = pd.Series(pnl_w2).rank()
            corr = s1.corr(s2)
            correlations.append((w, w+1, round(corr, 3), len(syms_both)))
        else:
            correlations.append((w, w+1, None, len(syms_both)))

    return heatmap, stability, correlations


# ============ 3. Q2：滚动7天Walk-Forward ============

def analyze_q2(all_trades, all_data):
    """7天滚动walk-forward：过去7天表现排名 → 预测未来7天"""
    # 按品种按block分桶
    sym_block_pnl = {}  # {sym_key: {block_idx: [pnl, ...]}}

    for t in all_trades:
        if t['exit_reason'] == 'backtest_end':
            continue
        sym = t['sym_key']
        df = all_data[sym]
        n = len(df)
        analysis_start = max(0, n - LAST_DAYS * BARS_PER_DAY)
        b_idx = (t['bar_idx'] - analysis_start) // BLOCK_BARS
        n_blocks = (LAST_DAYS * BARS_PER_DAY) // BLOCK_BARS
        b_idx = min(b_idx, n_blocks - 1)
        if b_idx < 0:
            continue
        sym_block_pnl.setdefault(sym, {}).setdefault(b_idx, []).append(t['pnl'])

    n_blocks = (LAST_DAYS * BARS_PER_DAY) // BLOCK_BARS  # 17
    all_syms = sorted(set(t['sym_key'] for t in all_trades))

    walk_forward = []
    for step in range(1, n_blocks):
        # 过去一个block的品种表现
        past_scores = {}
        for sym in all_syms:
            pnls = sym_block_pnl.get(sym, {}).get(step - 1, [])
            past_scores[sym] = sum(pnls) if pnls else 0

        # 未来一个block的品种表现
        next_scores = {}
        for sym in all_syms:
            pnls = sym_block_pnl.get(sym, {}).get(step, [])
            next_scores[sym] = sum(pnls) if pnls else 0

        # 排名取Top10和Bottom10
        ranked = sorted(all_syms, key=lambda s: past_scores[s], reverse=True)
        top10 = ranked[:10]
        bot10 = ranked[-10:]

        top10_past = sum(past_scores[s] for s in top10)
        top10_next = sum(next_scores[s] for s in top10)
        top10_next_n = sum(len(sym_block_pnl.get(s, {}).get(step, [])) for s in top10)
        bot10_past = sum(past_scores[s] for s in bot10)
        bot10_next = sum(next_scores[s] for s in bot10)
        bot10_next_n = sum(len(sym_block_pnl.get(s, {}).get(step, [])) for s in bot10)

        walk_forward.append({
            'step': step,
            'top10_past': round(top10_past, 2),
            'top10_next': round(top10_next, 2),
            'top10_next_n': top10_next_n,
            'bot10_past': round(bot10_past, 2),
            'bot10_next': round(bot10_next, 2),
            'bot10_next_n': bot10_next_n,
            'spread': round(top10_next - bot10_next, 2),
            'top10_syms': top10[:3],  # 记前3名方便查看
        })

    return walk_forward


# ============ 4. Q3：截面ER(20)排名 ============

def analyze_q3(all_trades, all_data):
    """信号触发时品种的ER(20)截面排名"""
    # 构建ER面板：用bar_idx对齐（所有品种取相同的最后120天）
    # 简化方案：对每笔交易，计算信号时刻所有品种的ER(20)，得到该品种的排名
    # 这里用预计算方式：构建 {sym: {bar_offset: er_20}} 的映射

    # 先统一 analysis_start 后的 bar offset
    sym_er = {}  # {sym_key: pd.Series(er_20, index=bar_offset)}
    for sym, df in all_data.items():
        n = len(df)
        analysis_start = max(0, n - LAST_DAYS * BARS_PER_DAY)
        total_analysis_bars = n - analysis_start
        offsets = list(range(total_analysis_bars))
        ers = df['er_20'].iloc[analysis_start:].values
        sym_er[sym] = pd.Series(ers, index=offsets)

    er_panel = pd.DataFrame(sym_er)
    er_rank = er_panel.rank(axis=1, ascending=False, method='min')

    # 为每笔交易标注截面ER排名
    trades_with_rank = []
    for t in all_trades:
        if t['exit_reason'] == 'backtest_end':
            continue
        sym = t['sym_key']
        df = all_data[sym]
        n = len(df)
        analysis_start = max(0, n - LAST_DAYS * BARS_PER_DAY)
        offset = t['bar_idx'] - analysis_start
        if offset < 0 or offset >= len(er_rank):
            continue
        rank_val = er_rank.iloc[offset].get(sym, np.nan)
        if pd.isna(rank_val):
            continue
        t_copy = dict(t)
        t_copy['er_rank'] = int(rank_val)
        trades_with_rank.append(t_copy)

    # 分桶分析
    n_syms = len(all_data)
    # Top10 vs Bottom10
    top_threshold = 10
    bot_threshold = n_syms - 9  # rank >= 23 for 32 symbols

    buckets_coarse = {
        f'Top10 (rank<=10)': [t for t in trades_with_rank if t['er_rank'] <= top_threshold],
        f'中间 (11~{bot_threshold-1})': [t for t in trades_with_rank
                                          if top_threshold < t['er_rank'] < bot_threshold],
        f'Bottom10 (rank>={bot_threshold})': [t for t in trades_with_rank
                                               if t['er_rank'] >= bot_threshold],
    }

    # 6档细分
    step = max(1, n_syms // 6)
    buckets_fine = {}
    for i in range(6):
        lo = i * step + 1
        hi = min((i + 1) * step, n_syms)
        if i == 5:
            hi = n_syms
        label = f'rank {lo}~{hi}'
        buckets_fine[label] = [t for t in trades_with_rank if lo <= t['er_rank'] <= hi]

    # 按场景拆分
    scenario_buckets = {}
    for sc in [1, 2, 3]:
        sc_trades = [t for t in trades_with_rank if t['scenario'] == sc]
        sc_top = [t for t in sc_trades if t['er_rank'] <= top_threshold]
        sc_bot = [t for t in sc_trades if t['er_rank'] >= bot_threshold]
        scenario_buckets[sc] = {'top': sc_top, 'bot': sc_bot, 'all': sc_trades}

    return buckets_coarse, buckets_fine, scenario_buckets, trades_with_rank


# ============ 5. Q4：市场同步性 — 品种变化还是市场环境变化？ ============

def analyze_q4(all_trades, all_data):
    """
    核心问题：品种表现的波动是品种自身属性，还是市场整体环境在切换？
    如果是后者，品种筛选不如市场状态识别有用。

    方法：
    1. 按7天block构建 品种×时间 PnL矩阵
    2. 计算每个block的"市场因子"（全品种平均PnL）
    3. 看各品种是否跟市场因子同步（正PnL品种比例 vs 市场方向）
    4. 计算市场因子对品种表现的解释力
    5. 寻找可预测市场因子的先行指标（全品种平均ER、趋势品种比例等）
    """
    # --- 构建 品种×block PnL 矩阵 ---
    n_blocks = (LAST_DAYS * BARS_PER_DAY) // BLOCK_BARS

    # 每笔交易分配到block
    sym_block_pnl = {}
    for t in all_trades:
        if t['exit_reason'] == 'backtest_end':
            continue
        sym = t['sym_key']
        df = all_data[sym]
        n = len(df)
        analysis_start = max(0, n - LAST_DAYS * BARS_PER_DAY)
        b_idx = (t['bar_idx'] - analysis_start) // BLOCK_BARS
        b_idx = min(b_idx, n_blocks - 1)
        if b_idx < 0:
            continue
        sym_block_pnl.setdefault(sym, {}).setdefault(b_idx, []).append(t['pnl'])

    all_syms = sorted(set(t['sym_key'] for t in all_trades if t['exit_reason'] != 'backtest_end'))

    # --- 1. 市场因子：每个block的全品种平均PnL ---
    block_stats = []
    for b in range(n_blocks):
        sym_pnls = {}
        for sym in all_syms:
            pnls = sym_block_pnl.get(sym, {}).get(b, [])
            if pnls:
                sym_pnls[sym] = sum(pnls)

        if not sym_pnls:
            block_stats.append({
                'block': b, 'market_pnl': 0, 'n_syms_with_trades': 0,
                'n_positive': 0, 'n_negative': 0, 'sync_rate': 0,
                'market_dir': 'N/A',
            })
            continue

        market_pnl = np.mean(list(sym_pnls.values()))
        n_positive = sum(1 for v in sym_pnls.values() if v > 0)
        n_negative = sum(1 for v in sym_pnls.values() if v <= 0)
        n_total = len(sym_pnls)

        # 同步率：与市场方向一致的品种比例
        if market_pnl > 0:
            sync_rate = n_positive / n_total
            market_dir = '多'
        elif market_pnl < 0:
            sync_rate = n_negative / n_total
            market_dir = '空'
        else:
            sync_rate = 0
            market_dir = '平'

        block_stats.append({
            'block': b,
            'market_pnl': round(market_pnl, 3),
            'n_syms_with_trades': n_total,
            'n_positive': n_positive,
            'n_negative': n_negative,
            'sync_rate': round(sync_rate, 2),
            'market_dir': market_dir,
        })

    # --- 2. 好市场 vs 坏市场下的整体策略表现 ---
    good_blocks = [bs['block'] for bs in block_stats if bs['market_pnl'] > 0]
    bad_blocks = [bs['block'] for bs in block_stats if bs['market_pnl'] <= 0 and bs['n_syms_with_trades'] > 0]

    good_pnls = []
    bad_pnls = []
    for t in all_trades:
        if t['exit_reason'] == 'backtest_end':
            continue
        sym = t['sym_key']
        df = all_data[sym]
        n = len(df)
        analysis_start = max(0, n - LAST_DAYS * BARS_PER_DAY)
        b_idx = min((t['bar_idx'] - analysis_start) // BLOCK_BARS, n_blocks - 1)
        if b_idx in good_blocks:
            good_pnls.append(t['pnl'])
        elif b_idx in bad_blocks:
            bad_pnls.append(t['pnl'])

    # --- 3. 市场状态先行指标 ---
    # 在每个block开始时计算：全品种平均ER(20)、趋势品种比例
    block_indicators = []
    for b in range(n_blocks):
        er_values = []
        trend_count = 0
        total_count = 0
        for sym, df in all_data.items():
            n = len(df)
            analysis_start = max(0, n - LAST_DAYS * BARS_PER_DAY)
            bar_idx = analysis_start + b * BLOCK_BARS
            if bar_idx >= n:
                continue
            row = df.iloc[bar_idx]
            er = row.get('er_20', np.nan)
            if not pd.isna(er):
                er_values.append(er)
            ema20 = row.get('ema20', np.nan)
            ema120 = row.get('ema120', np.nan)
            if not pd.isna(ema20) and not pd.isna(ema120):
                total_count += 1
                if abs(ema20 - ema120) / ema120 > 0.005:  # 有明确趋势
                    trend_count += 1

        avg_er = np.mean(er_values) if er_values else 0
        trend_pct = trend_count / total_count if total_count > 0 else 0
        block_indicators.append({
            'block': b,
            'avg_er': round(avg_er, 3),
            'trend_pct': round(trend_pct, 2),
            'n_syms': len(er_values),
        })

    # --- 4. 先行指标 vs 下一期市场因子的相关性 ---
    # indicator[b] 预测 market_pnl[b] (同期，因为indicator是block开头的快照)
    indicator_corrs = {}
    for col in ['avg_er', 'trend_pct']:
        x_vals = []
        y_vals = []
        for b in range(n_blocks):
            ind = block_indicators[b]
            bs = block_stats[b]
            if bs['n_syms_with_trades'] > 0:
                x_vals.append(ind[col])
                y_vals.append(bs['market_pnl'])
        if len(x_vals) >= 5:
            sx = pd.Series(x_vals).rank()
            sy = pd.Series(y_vals).rank()
            corr = sx.corr(sy)
            indicator_corrs[col] = round(corr, 3)

    # --- 5. 品种间 PnL 相关性矩阵（衡量同步程度） ---
    # 用30天窗口而非7天，增加每格样本量
    sym_window_pnl = {}
    for t in all_trades:
        if t['exit_reason'] == 'backtest_end':
            continue
        sym = t['sym_key']
        df = all_data[sym]
        n = len(df)
        analysis_start = max(0, n - LAST_DAYS * BARS_PER_DAY)
        window_bars = WINDOW_DAYS * BARS_PER_DAY
        w_idx = min((t['bar_idx'] - analysis_start) // window_bars, N_WINDOWS - 1)
        if w_idx < 0:
            continue
        sym_window_pnl.setdefault(sym, {}).setdefault(w_idx, []).append(t['pnl'])

    # 构建 sym × window 矩阵（avg_pnl）
    matrix_syms = [s for s in all_syms
                   if sum(1 for w in range(N_WINDOWS) if sym_window_pnl.get(s, {}).get(w)) >= 3]
    if len(matrix_syms) >= 5:
        mat = pd.DataFrame(index=matrix_syms, columns=range(N_WINDOWS), dtype=float)
        for sym in matrix_syms:
            for w in range(N_WINDOWS):
                pnls = sym_window_pnl.get(sym, {}).get(w, [])
                mat.loc[sym, w] = np.mean(pnls) if pnls else 0
        # 品种间相关系数矩阵
        corr_matrix = mat.T.corr()
        # 取上三角的平均值（排除对角线）
        upper = []
        for i in range(len(matrix_syms)):
            for j in range(i + 1, len(matrix_syms)):
                val = corr_matrix.iloc[i, j]
                if not pd.isna(val):
                    upper.append(val)
        avg_cross_corr = np.mean(upper) if upper else 0
    else:
        avg_cross_corr = None
        matrix_syms = []

    return block_stats, good_pnls, bad_pnls, block_indicators, indicator_corrs, avg_cross_corr


# ============ 5b. Q5：实时市场状态检测 ============

def analyze_q5(all_trades, all_data):
    """
    核心问题：能否用已产生信号的滚动表现实时检测"当前市场状态差"？

    方法：
    1. 按时间排序全部469笔信号（跨品种）
    2. 对每笔信号，计算"过去K笔信号的滚动胜率和avg_pnl"
    3. 测试规则：如果滚动胜率<阈值 → 暂停交易 → 看跳过的信号是否确实差
    4. 同时测试一个更实际的检测器：最近K笔的累计PnL连续下降
    """
    # 按datetime排序全部有效信号
    valid = [t for t in all_trades if t['exit_reason'] != 'backtest_end']
    valid.sort(key=lambda t: t['datetime'])

    n = len(valid)
    pnl_seq = [t['pnl'] for t in valid]
    win_seq = [1 if t['pnl'] > 0 else 0 for t in valid]

    # --- 1. 不同窗口K的滚动指标 ---
    K_VALUES = [5, 10, 15, 20]
    rolling_data = {}  # {K: [{idx, rolling_wr, rolling_avg, next_pnl, next_win}, ...]}

    for K in K_VALUES:
        rows = []
        for i in range(K, n):
            window_wins = sum(win_seq[i-K:i])
            window_pnls = pnl_seq[i-K:i]
            rolling_wr = window_wins / K
            rolling_avg = np.mean(window_pnls)
            rows.append({
                'idx': i,
                'rolling_wr': rolling_wr,
                'rolling_avg': rolling_avg,
                'pnl': pnl_seq[i],
                'win': win_seq[i],
                'datetime': valid[i]['datetime'],
            })
        rolling_data[K] = rows

    # --- 2. 暂停交易规则测试 ---
    # 规则：当滚动胜率 < threshold 时，跳过下一笔（认为市场差）
    # 回测：比较"全做" vs "跳过低胜率期"
    WR_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40]
    pause_results = {}  # {(K, threshold): {traded_pnls, skipped_pnls, ...}}

    for K in K_VALUES:
        rows = rolling_data[K]
        for thr in WR_THRESHOLDS:
            traded_pnls = []
            skipped_pnls = []
            for r in rows:
                if r['rolling_wr'] >= thr:
                    traded_pnls.append(r['pnl'])
                else:
                    skipped_pnls.append(r['pnl'])

            pause_results[(K, thr)] = {
                'traded': traded_pnls,
                'skipped': skipped_pnls,
            }

    # --- 3. 滚动avg_pnl作为检测器 ---
    # 规则：当滚动avg_pnl < 0 时暂停
    avgpnl_results = {}
    for K in K_VALUES:
        rows = rolling_data[K]
        traded = [r['pnl'] for r in rows if r['rolling_avg'] >= 0]
        skipped = [r['pnl'] for r in rows if r['rolling_avg'] < 0]
        avgpnl_results[K] = {'traded': traded, 'skipped': skipped}

    # --- 4. 连续亏损检测 ---
    # 检测最近K笔中连续亏损>=M笔的情况
    CONSEC_M = [3, 4, 5]
    consec_results = {}
    for M in CONSEC_M:
        traded = []
        skipped = []
        for i in range(M, n):
            recent = win_seq[i-M:i]
            if sum(recent) == 0:  # 最近M笔全亏
                skipped.append(pnl_seq[i])
            else:
                traded.append(pnl_seq[i])
        consec_results[M] = {'traded': traded, 'skipped': skipped}

    # --- 5. 坏市场block中信号的特征分析 ---
    # 看坏block的信号在触发前有什么可观测特征
    n_blocks = (LAST_DAYS * BARS_PER_DAY) // BLOCK_BARS
    sym_block_pnl = {}
    for t in valid:
        sym = t['sym_key']
        df = all_data[sym]
        nn = len(df)
        analysis_start = max(0, nn - LAST_DAYS * BARS_PER_DAY)
        b_idx = min((t['bar_idx'] - analysis_start) // BLOCK_BARS, n_blocks - 1)
        sym_block_pnl.setdefault(b_idx, []).append(t)

    # 计算每个block的市场PnL
    block_market = {}
    for b, trades in sym_block_pnl.items():
        block_market[b] = np.mean([t['pnl'] for t in trades])

    # 信号特征对比：好block vs 坏block
    good_trades = [t for b, trades in sym_block_pnl.items()
                   for t in trades if block_market.get(b, 0) > 0]
    bad_trades = [t for b, trades in sym_block_pnl.items()
                  for t in trades if block_market.get(b, 0) <= 0]

    feature_compare = {}
    for label, trades_list in [('好市场', good_trades), ('坏市场', bad_trades)]:
        if not trades_list:
            continue
        feature_compare[label] = {
            'N': len(trades_list),
            'avg_er20': round(np.mean([t['er_20'] for t in trades_list]), 3),
            'avg_dev': round(np.mean([t['deviation_atr'] for t in trades_list]), 3),
            'sc1_pct': round(sum(1 for t in trades_list if t['scenario'] == 1) / len(trades_list) * 100, 1),
            'sc2_pct': round(sum(1 for t in trades_list if t['scenario'] == 2) / len(trades_list) * 100, 1),
            'sc3_pct': round(sum(1 for t in trades_list if t['scenario'] == 3) / len(trades_list) * 100, 1),
            'long_pct': round(sum(1 for t in trades_list if t['direction'] == 'long') / len(trades_list) * 100, 1),
            'avg_pnl': round(np.mean([t['pnl'] for t in trades_list]), 3),
            'wr': round(sum(1 for t in trades_list if t['pnl'] > 0) / len(trades_list) * 100, 1),
        }

    return rolling_data, pause_results, avgpnl_results, consec_results, feature_compare


# ============ 5c. Q6：市场广度 — ER广度 + 活跃持仓数 ============

def analyze_q6(all_trades, all_data):
    """
    市场广度：信号触发时全市场有多少品种处于"趋势就绪"状态？
    - ER广度：32品种中ER(20)>=0.5的个数
    - ER动态广度：多少品种的ER5变化(6根)>=0.50、ER40变化(12根)>=0.14
    - 活跃持仓：当前有多少其他品种正在持仓中
    """
    valid = [t for t in all_trades if t['exit_reason'] != 'backtest_end']

    # --- 1. 构建ER面板 + ER变化面板 ---
    er_panels = {}  # {period: DataFrame(bar_offset × sym)}
    for period in [5, 20, 40]:
        sym_er = {}
        for sym, df in all_data.items():
            n = len(df)
            analysis_start = max(0, n - LAST_DAYS * BARS_PER_DAY)
            col = f'er_{period}'
            if col in df.columns:
                sym_er[sym] = pd.Series(
                    df[col].iloc[analysis_start:].values,
                    index=range(n - analysis_start))
        er_panels[period] = pd.DataFrame(sym_er)

    # ER变化面板：各周期ER的短期变化
    er5_delta6 = er_panels[5].diff(6)     # ER5 6根变化
    er20_delta6 = er_panels[20].diff(6)   # ER20 6根变化
    er20_delta12 = er_panels[20].diff(12) # ER20 12根变化
    er40_delta12 = er_panels[40].diff(12) # ER40 12根变化

    # --- 2. 为每笔信号计算广度指标 ---
    results = []
    for t in valid:
        sym = t['sym_key']
        df = all_data[sym]
        n = len(df)
        analysis_start = max(0, n - LAST_DAYS * BARS_PER_DAY)
        offset = t['bar_idx'] - analysis_start
        if offset < 0:
            continue

        # ER广度：多少品种ER>=0.5
        breadth = {}
        for period in [5, 20, 40]:
            panel = er_panels[period]
            if offset < len(panel):
                row = panel.iloc[offset]
                breadth[f'er{period}_breadth'] = int((row >= 0.5).sum())
                breadth[f'er{period}_breadth_03'] = int((row >= 0.3).sum())
            else:
                breadth[f'er{period}_breadth'] = 0
                breadth[f'er{period}_breadth_03'] = 0

        # ER动态广度：多少品种ER在加速
        # ER变化广度：多少品种达到特定门槛
        # 门槛来自已验证的加仓规则：ER5Δ6>=0.50(场景1)、ER40Δ12>=0.14(场景2)
        # ER20补测多个门槛
        delta_configs = [
            ('er5d6', er5_delta6, [0.50, 0.25]),
            ('er20d6', er20_delta6, [0.30, 0.20, 0.10]),
            ('er20d12', er20_delta12, [0.20, 0.10]),
            ('er40d12', er40_delta12, [0.14, 0.07]),
        ]
        for key, panel, thresholds in delta_configs:
            if offset < len(panel):
                row_d = panel.iloc[offset]
                for thr in thresholds:
                    thr_key = f'{key}_ge_{str(thr).replace(".", "")}'
                    breadth[thr_key] = int((row_d >= thr).sum())
            else:
                for thr in thresholds:
                    thr_key = f'{key}_ge_{str(thr).replace(".", "")}'
                    breadth[thr_key] = 0

        # 活跃持仓数
        active = 0
        for other in valid:
            if other is t:
                continue
            o_sym = other['sym_key']
            o_df = all_data[o_sym]
            o_n = len(o_df)
            o_start = max(0, o_n - LAST_DAYS * BARS_PER_DAY)
            o_entry_offset = other['bar_idx'] - o_start
            o_exit_offset = other['exit_bar_idx'] - o_start
            if o_entry_offset <= offset < o_exit_offset:
                active += 1

        results.append({
            **t,
            **breadth,
            'active_positions': active,
        })

    return results


# ============ 6. K线图案例 ============

def find_chart_cases(heatmap, all_trades, all_data):
    """找崩塌型和逆袭型品种的K线案例"""
    charts = []

    # 崩塌型：前2窗口EV>0, 后2窗口EV<0
    # 逆袭型：前2窗口EV<0, 后2窗口EV>0
    crash_candidates = []
    surge_candidates = []

    for sym, row in heatmap.items():
        early = [r for r in row[:2] if r is not None and r['N'] >= 2]
        late = [r for r in row[2:] if r is not None and r['N'] >= 2]
        if not early or not late:
            continue
        early_ev = np.mean([r['EV'] for r in early])
        late_ev = np.mean([r['EV'] for r in late])

        if early_ev > 0.3 and late_ev < -0.1:
            crash_candidates.append((sym, early_ev, late_ev, early_ev - late_ev))
        if early_ev < -0.1 and late_ev > 0.3:
            surge_candidates.append((sym, early_ev, late_ev, late_ev - early_ev))

    crash_candidates.sort(key=lambda x: x[3], reverse=True)
    surge_candidates.sort(key=lambda x: x[3], reverse=True)

    def pick_signals(sym_key, window_range, label_prefix):
        """从指定窗口范围选最赚和最亏的信号"""
        df = all_data.get(sym_key)
        if df is None:
            return []
        n = len(df)
        analysis_start = max(0, n - LAST_DAYS * BARS_PER_DAY)
        window_bars = WINDOW_DAYS * BARS_PER_DAY

        sym_trades = [t for t in all_trades if t['sym_key'] == sym_key
                      and t['exit_reason'] != 'backtest_end']
        windowed = []
        for t in sym_trades:
            w = min((t['bar_idx'] - analysis_start) // window_bars, N_WINDOWS - 1)
            if w in window_range:
                windowed.append(t)
        if not windowed:
            return []

        windowed.sort(key=lambda t: t['pnl'], reverse=True)
        picks = []
        if windowed:
            picks.append(windowed[0])  # 最赚
        if len(windowed) >= 2:
            picks.append(windowed[-1])  # 最亏

        result = []
        for t in picks:
            w = min((t['bar_idx'] - analysis_start) // window_bars, N_WINDOWS - 1)
            exits_draw = [{'name': SCENARIO_EXIT[t['scenario']],
                           'idx': t['exit_bar_idx'],
                           'price': t['exit_price']}]
            chart_html = render_chart(
                df=df, entry_idx=t['bar_idx'], direction=t['direction'],
                before_bars=30, after_bars=60,
                exits=exits_draw,
                stop_price=t['stop_price'],
                title=f"{label_prefix} | {sym_name(sym_key)} 场景{t['scenario']} | W{w+1} | {t['direction']}",
                extra_info={
                    'ER20': f"{t['er_20']:.2f}",
                    'Dev': f"{t['deviation_atr']:.2f}ATR",
                    'PnL': f"{t['pnl']:+.2f}%",
                },
            )
            result.append(chart_html)
        return result

    # 最多各取1个品种
    if crash_candidates:
        sym = crash_candidates[0][0]
        charts.extend(pick_signals(sym, range(0, 2), f'崩塌型-好期(W1-2) {sym_name(sym)}'))
        charts.extend(pick_signals(sym, range(2, 4), f'崩塌型-差期(W3-4) {sym_name(sym)}'))

    if surge_candidates:
        sym = surge_candidates[0][0]
        charts.extend(pick_signals(sym, range(0, 2), f'逆袭型-差期(W1-2) {sym_name(sym)}'))
        charts.extend(pick_signals(sym, range(2, 4), f'逆袭型-好期(W3-4) {sym_name(sym)}'))

    return charts


# ============ 6. HTML报告 ============

def heatmap_cell(st):
    """生成热力图单元格HTML — 用 sum_pnl 和 avg_pnl 展示，避免 EV 在全胜时的虚高"""
    if st is None:
        return '<td style="color:#484f58">N=0</td>'
    n = st.get('N', 0)
    sum_pnl = st.get('sum_pnl', 0)
    avg_pnl = st.get('avg_pnl', 0)
    wr = st.get('wr', 0)
    if n < 2:
        color = '#484f58'
    elif avg_pnl > 0.5:
        color = '#3fb950'
    elif avg_pnl > 0:
        color = '#7ee787'
    elif avg_pnl > -0.5:
        color = '#f0883e'
    else:
        color = '#f85149'
    return (f'<td style="color:{color};font-weight:bold">'
            f'{sum_pnl:+.1f}%<br>'
            f'<span style="font-size:10px;color:#8b949e">'
            f'N={n} WR={wr}% avg={avg_pnl:+.2f}%</span></td>')


def build_report(heatmap, stability, correlations, walk_forward,
                 q3_coarse, q3_fine, q3_scenario, chart_htmls, all_trades,
                 q4_data=None, q5_data=None, q6_data=None, all_data=None):
    rpt = Report('品种适配度研究：稳定性 + 滚动预测 + 截面ER排名')

    # === Q1 热力图 ===
    rpt.add_section('一、品种时间稳定性（4×30天窗口）',
                     '每品种每窗口的累计PnL% | 排除backtest_end | 颜色按avg_pnl编码')

    # 用 add_html 输出热力图
    html_rows = []
    html_rows.append('<table>')
    html_rows.append('<tr><th>品种</th><th>W1 (1-30d)</th><th>W2 (31-60d)</th>'
                     '<th>W3 (61-90d)</th><th>W4 (91-120d)</th><th>120d累计</th>'
                     '<th>120d avg</th><th>分类</th></tr>')

    # 按120天累计PnL排序
    def sym_total_pnl(sym):
        row = heatmap[sym]
        return sum(r['sum_pnl'] for r in row if r is not None)
    sorted_syms = sorted(heatmap.keys(), key=sym_total_pnl, reverse=True)

    for sym in sorted_syms:
        row = heatmap[sym]
        total_pnl = sum(r['sum_pnl'] for r in row if r is not None)
        total_n = sum(r['N'] for r in row if r is not None)
        avg_pnl = total_pnl / total_n if total_n > 0 else 0
        stab = stability.get(sym, '?')

        stab_color = '#8b949e'
        if stab == '稳定正向':
            stab_color = '#3fb950'
        elif stab == '稳定负向':
            stab_color = '#f85149'
        elif stab == '波动型':
            stab_color = '#f0883e'

        html_rows.append(f'<tr><td>{sym_name(sym)}</td>')
        for w in range(N_WINDOWS):
            html_rows.append(heatmap_cell(row[w]))
        total_color = '#3fb950' if total_pnl > 0 else '#f85149'
        avg_color = '#3fb950' if avg_pnl > 0 else '#f85149'
        html_rows.append(f'<td style="color:{total_color};font-weight:bold">{total_pnl:+.1f}%</td>')
        html_rows.append(f'<td style="color:{avg_color}">{avg_pnl:+.2f}%</td>')
        html_rows.append(f'<td style="color:{stab_color}">{stab}</td></tr>')

    html_rows.append('</table>')
    rpt.add_html('\n'.join(html_rows))

    # 稳定性汇总
    rpt.add_section('稳定性分类汇总')
    cat_counts = {}
    for cat in stability.values():
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    for cat in ['稳定正向', '波动型', '稳定负向', '样本不足']:
        if cat in cat_counts:
            rpt.add_text(f'{cat}: {cat_counts[cat]}个品种')

    # 窗口间秩相关
    rpt.add_section('窗口间排名相关性（Spearman秩相关，基于avg_pnl排名）')
    rpt.add_text('注：使用 avg_pnl（每笔均值）做排名，避免 EV 在全胜/全负时的极端失真')
    corr_headers = ['窗口对', '相关系数', '品种数', '解读']
    corr_rows = []
    for w1, w2, corr, n_sym in correlations:
        if corr is not None:
            interp = '正相关→品种表现有延续性' if corr > 0.3 else (
                '负相关→品种表现会反转' if corr < -0.3 else '弱相关→品种表现不可预测')
            corr_rows.append([f'W{w1+1}→W{w2+1}', f'{corr:+.3f}', n_sym, interp])
        else:
            corr_rows.append([f'W{w1+1}→W{w2+1}', '数据不足', n_sym, '-'])
    rpt.add_table(corr_headers, corr_rows, highlight_pnl_cols=[1])

    if any(c[2] is not None for c in correlations):
        valid_corrs = [c[2] for c in correlations if c[2] is not None]
        avg_corr = np.mean(valid_corrs)
        rpt.add_text(f'平均相关系数: {avg_corr:+.3f}', color='#58a6ff')
        if avg_corr > 0.3:
            rpt.add_text('结论：品种表现有一定延续性，过去好的品种未来可能继续好', color='#3fb950')
        elif avg_corr < -0.3:
            rpt.add_text('结论：品种表现有反转倾向，过去好的品种未来反而差', color='#f85149')
        else:
            rpt.add_text('结论：品种表现窗口间关联弱，过去表现对未来预测力有限', color='#f0883e')

    # === Q2 Walk-Forward ===
    rpt.add_section('二、滚动7天Walk-Forward',
                     '过去7天PnL排名Top10 vs Bottom10 → 看下一个7天表现')

    wf_headers = ['步骤', 'Top10过去', 'Top10未来', 'Top10 N',
                  'Bot10过去', 'Bot10未来', 'Bot10 N', '利差']
    wf_rows = []
    for w in walk_forward:
        wf_rows.append([
            f'B{w["step"]}',
            f'{w["top10_past"]:+.2f}%', f'{w["top10_next"]:+.2f}%', w['top10_next_n'],
            f'{w["bot10_past"]:+.2f}%', f'{w["bot10_next"]:+.2f}%', w['bot10_next_n'],
            f'{w["spread"]:+.2f}%',
        ])
    rpt.add_table(wf_headers, wf_rows, highlight_pnl_cols=[2, 5, 7])

    # 汇总 + 统计显著性
    if walk_forward:
        total_top = sum(w['top10_next'] for w in walk_forward)
        total_bot = sum(w['bot10_next'] for w in walk_forward)
        n_steps = len(walk_forward)
        hits = sum(1 for w in walk_forward if w['spread'] > 0)
        hit_rate = hits / n_steps

        # 二项分布检验 P(X>=hits | n=n_steps, p=0.5)
        from math import comb
        p_value = sum(comb(n_steps, k) for k in range(hits, n_steps + 1)) / (2 ** n_steps)

        rpt.add_text(f'Top10累计未来PnL: {total_top:+.2f}% | Bottom10累计: {total_bot:+.2f}%')
        rpt.add_text(f'Top10胜出步数: {hits}/{n_steps} ({hit_rate:.0%})')
        rpt.add_text(f'二项检验 p-value: {p_value:.3f}（p<0.05才统计显著）',
                     color='#3fb950' if p_value < 0.05 else '#f0883e')
        rpt.add_text(f'Top10每笔均PnL: {total_top/max(1,sum(w["top10_next_n"] for w in walk_forward)):+.3f}% | '
                     f'Bot10每笔均PnL: {total_bot/max(1,sum(w["bot10_next_n"] for w in walk_forward)):+.3f}%')

        if p_value < 0.05 and total_top > total_bot:
            rpt.add_text('结论：过去7天排名有统计显著的前瞻性', color='#3fb950')
        elif p_value < 0.10 and total_top > total_bot:
            rpt.add_text('结论：过去7天排名有一定前瞻性，但未达5%显著水平（仅边际显著）', color='#f0883e')
        elif hit_rate <= 0.4:
            rpt.add_text('结论：过去7天排名无前瞻性', color='#f85149')
        else:
            rpt.add_text('结论：方向正确但统计不显著，需更长数据验证', color='#f0883e')

    # === Q3 截面ER排名 ===
    rpt.add_section('三、截面ER(20)排名',
                     '信号触发时，品种ER(20)在全品种中的排名 → 是否影响信号质量')

    rpt.add_text('⚠ 逻辑提醒：场景1/3的入场条件本身就要求ER>=0.5，'
                 '高ER品种天然排名靠前。下面的Top10 vs Bottom10本质上在比较'
                 '"有ER门槛的场景1/3" vs "无ER门槛的场景2"，'
                 '而非纯粹测试截面排名的增量价值。', color='#f0883e')

    q3_headers = ['分组', 'N', 'EV', 'WR%', 'PR', '累计PnL', '场景分布']
    q3_rows = []
    for label, trades in q3_coarse.items():
        pnls = [t['pnl'] for t in trades]
        st = calc_ev(pnls)
        # 显示场景分布以暴露 tautology
        sc_dist = {}
        for t in trades:
            sc_dist[t['scenario']] = sc_dist.get(t['scenario'], 0) + 1
        dist_str = ' '.join(f'S{k}:{v}' for k, v in sorted(sc_dist.items()))
        q3_rows.append([label, st['N'], st['EV'], st['wr'], st['pr'],
                        f"{st['sum_pnl']:+.2f}%", dist_str])
    rpt.add_table(q3_headers, q3_rows, highlight_pnl_cols=[2, 5])

    # 细分
    rpt.add_section('截面ER排名-6档细分')
    q3f_rows = []
    for label, trades in q3_fine.items():
        pnls = [t['pnl'] for t in trades]
        st = calc_ev(pnls)
        sc_dist = {}
        for t in trades:
            sc_dist[t['scenario']] = sc_dist.get(t['scenario'], 0) + 1
        dist_str = ' '.join(f'S{k}:{v}' for k, v in sorted(sc_dist.items()))
        q3f_rows.append([label, st['N'], st['EV'], st['wr'], st['pr'],
                         f"{st['sum_pnl']:+.2f}%", dist_str])
    rpt.add_table(q3_headers, q3f_rows, highlight_pnl_cols=[2, 5])

    # 场景内 ER 排名分析（控制场景后的纯净测试）
    rpt.add_section('截面ER排名 × 场景（控制场景后的纯净测试）')
    rpt.add_text('下面按场景拆分看ER排名效果。'
                 '场景2最干净（无ER入场门槛），可直接测试ER排名的独立价值。'
                 '场景1/3测试的是"在ER>=0.5基础上，更高排名是否有增量"。',
                 color='#58a6ff')

    q3s_headers = ['分组', 'N', 'EV', 'WR%', 'PR', '累计PnL', 'avgER20']
    for sc in [1, 2, 3]:
        data = q3_scenario.get(sc, {})
        sc_label = {1: '场景1(A类,ER>=0.5)', 2: '场景2(C类,无ER门槛)',
                    3: '场景3(B类,ER>=0.5)'}[sc]
        top_trades = data.get('top', [])
        bot_trades = data.get('bot', [])
        all_trades_sc = data.get('all', [])

        rows_sc = []
        for label, trades in [(f'{sc_label} Top10', top_trades),
                               (f'{sc_label} Bot10', bot_trades),
                               (f'{sc_label} 全部', all_trades_sc)]:
            pnls = [t['pnl'] for t in trades]
            st = calc_ev(pnls)
            avg_er = np.mean([t['er_20'] for t in trades]) if trades else 0
            rows_sc.append([label, st['N'], st['EV'], st['wr'], st['pr'],
                           f"{st['sum_pnl']:+.2f}%", f'{avg_er:.3f}'])
        rpt.add_table(q3s_headers, rows_sc, highlight_pnl_cols=[2, 5])

    # 结论
    # 场景2的对比最干净
    sc2 = q3_scenario.get(2, {})
    sc2_top = calc_ev([t['pnl'] for t in sc2.get('top', [])])
    sc2_bot = calc_ev([t['pnl'] for t in sc2.get('bot', [])])
    sc2_all = calc_ev([t['pnl'] for t in sc2.get('all', [])])

    rpt.add_section('Q3 小结')
    rpt.add_text(f'场景2（最干净的测试）：Top10 EV={sc2_top["EV"]:+.2f}(N={sc2_top["N"]}) vs '
                 f'Bot10 EV={sc2_bot["EV"]:+.2f}(N={sc2_bot["N"]}) vs '
                 f'全部 EV={sc2_all["EV"]:+.2f}(N={sc2_all["N"]})')
    if sc2_top['N'] >= 10 and sc2_bot['N'] >= 5:
        if sc2_top['EV'] > sc2_bot['EV'] + 0.2:
            rpt.add_text('场景2中ER排名高→信号更好，ER截面排名有独立过滤价值', color='#3fb950')
        elif sc2_bot['EV'] > sc2_top['EV'] + 0.2:
            rpt.add_text('场景2中ER排名高→信号反而差，ER截面排名不宜用于过滤', color='#f85149')
        else:
            rpt.add_text('场景2中ER排名区分力不明显', color='#f0883e')
    else:
        rpt.add_text('场景2中某组样本不足，结论暂不可靠', color='#f0883e')

    # === Q4 市场同步性 ===
    if q4_data:
        block_stats, good_pnls, bad_pnls, block_indicators, indicator_corrs, avg_cross_corr = q4_data

        rpt.add_section('四、市场同步性：品种在变化 还是 市场环境在变化？',
                         '如果所有品种同涨同跌，品种筛选不如市场状态识别')

        # 4a. 品种间 avg_pnl 截面相关系数
        rpt.add_section('4a. 品种间表现相关性')
        if avg_cross_corr is not None:
            rpt.add_text(f'品种间 avg_pnl 的平均截面相关系数: {avg_cross_corr:+.3f}')
            if avg_cross_corr > 0.3:
                rpt.add_text('品种间高度同步 → 市场环境是主要驱动因素，品种筛选价值有限', color='#f85149')
            elif avg_cross_corr > 0.1:
                rpt.add_text('品种间存在一定同步性 → 市场环境有影响，但品种差异也存在', color='#f0883e')
            elif avg_cross_corr > -0.1:
                rpt.add_text('品种间几乎独立 → 品种差异是主要因素，品种筛选可能有价值', color='#3fb950')
            else:
                rpt.add_text('品种间负相关 → 品种表现有分化，存在品种轮动机会', color='#58a6ff')
        else:
            rpt.add_text('品种数不足，无法计算截面相关')

        # 4b. 7天block市场因子详情
        rpt.add_section('4b. 7天窗口市场因子',
                         '每个7天block：全品种平均PnL、正/负品种数、同步率')
        bf_headers = ['Block', '市场PnL', '有信号品种', '正向', '负向',
                      '同步率', '方向', 'avgER', '趋势%']
        bf_rows = []
        for bs, bi in zip(block_stats, block_indicators):
            if bs['n_syms_with_trades'] == 0:
                continue
            bf_rows.append([
                f'B{bs["block"]}',
                f'{bs["market_pnl"]:+.3f}%',
                bs['n_syms_with_trades'],
                bs['n_positive'],
                bs['n_negative'],
                f'{bs["sync_rate"]:.0%}',
                bs['market_dir'],
                f'{bi["avg_er"]:.3f}',
                f'{bi["trend_pct"]:.0%}',
            ])
        rpt.add_table(bf_headers, bf_rows, highlight_pnl_cols=[1])

        # 同步率统计
        valid_blocks = [bs for bs in block_stats if bs['n_syms_with_trades'] >= 5]
        if valid_blocks:
            avg_sync = np.mean([bs['sync_rate'] for bs in valid_blocks])
            high_sync = sum(1 for bs in valid_blocks if bs['sync_rate'] >= 0.6)
            rpt.add_text(f'平均同步率: {avg_sync:.0%} | '
                         f'高同步(>=60%)的block: {high_sync}/{len(valid_blocks)}')
            if avg_sync >= 0.6:
                rpt.add_text('多数block中品种方向一致 → 市场整体环境是主因', color='#f85149')
            else:
                rpt.add_text('品种在同一时段内方向分化明显 → 品种级差异存在', color='#3fb950')

        # 4c. 好市场 vs 坏市场
        rpt.add_section('4c. 好市场期 vs 坏市场期',
                         '"好市场"=该block全品种平均PnL>0, "坏市场"=<=0')
        good_st = calc_ev(good_pnls)
        bad_st = calc_ev(bad_pnls)
        mkt_headers = ['市场状态', 'N', 'EV', 'WR%', 'PR', '累计PnL', 'avg_pnl']
        mkt_rows = [
            ['好市场期', good_st['N'], good_st['EV'], good_st['wr'], good_st['pr'],
             f"{good_st['sum_pnl']:+.2f}%", f"{good_st['avg_pnl']:+.3f}%"],
            ['坏市场期', bad_st['N'], bad_st['EV'], bad_st['wr'], bad_st['pr'],
             f"{bad_st['sum_pnl']:+.2f}%", f"{bad_st['avg_pnl']:+.3f}%"],
        ]
        rpt.add_table(mkt_headers, mkt_rows, highlight_pnl_cols=[2, 5])

        ev_diff = good_st['EV'] - bad_st['EV']
        rpt.add_text(f'好市场EV - 坏市场EV = {ev_diff:+.2f}')
        if ev_diff > 0.5:
            rpt.add_text('市场状态对策略表现影响巨大 → 识别市场状态比品种筛选更重要',
                         color='#f85149')
        elif ev_diff > 0.2:
            rpt.add_text('市场状态有一定影响，但品种差异也显著', color='#f0883e')
        else:
            rpt.add_text('市场状态影响不大，品种自身差异更主要', color='#3fb950')

        # 4d. 先行指标预测力
        rpt.add_section('4d. 市场状态先行指标',
                         'block开始时的全品种avgER和趋势比例 → 是否能预测该block表现')
        ind_headers = ['指标', 'vs当期市场PnL秩相关', '解读']
        ind_rows = []
        for col, label in [('avg_er', '全品种平均ER(20)'), ('trend_pct', '趋势品种比例')]:
            corr = indicator_corrs.get(col)
            if corr is not None:
                interp = '正向预测' if corr > 0.3 else ('反向预测' if corr < -0.3 else '预测力弱')
                ind_rows.append([label, f'{corr:+.3f}', interp])
            else:
                ind_rows.append([label, '数据不足', '-'])
        rpt.add_table(ind_headers, ind_rows, highlight_pnl_cols=[1])

        # Q4总结论
        rpt.add_section('Q4 综合结论')
        # 综合判断
        is_market_driven = False
        reasons = []
        if avg_cross_corr is not None and avg_cross_corr > 0.2:
            is_market_driven = True
            reasons.append(f'品种间相关系数{avg_cross_corr:+.3f}偏高')
        if valid_blocks and avg_sync >= 0.6:
            is_market_driven = True
            reasons.append(f'平均同步率{avg_sync:.0%}偏高')
        if ev_diff > 0.5:
            is_market_driven = True
            reasons.append(f'好/坏市场EV差{ev_diff:+.2f}很大')

        if is_market_driven:
            rpt.add_text(f'证据指向市场驱动：{", ".join(reasons)}', color='#f85149')
            rpt.add_text('建议：优先研究市场状态识别（何时开/关策略），品种筛选为辅助')
        else:
            reasons_against = []
            if avg_cross_corr is not None and avg_cross_corr <= 0.2:
                reasons_against.append(f'品种间相关系数{avg_cross_corr:+.3f}不高')
            if valid_blocks and avg_sync < 0.6:
                reasons_against.append(f'平均同步率{avg_sync:.0%}不高')
            rpt.add_text(f'证据指向品种差异为主：{", ".join(reasons_against)}', color='#3fb950')
            rpt.add_text('建议：品种筛选有价值，继续研究品种级特征')

    # === Q5 实时市场状态检测 ===
    if q5_data:
        rolling_data, pause_results, avgpnl_results, consec_results, feature_compare = q5_data

        rpt.add_section('五、实时市场状态检测',
                         '能否用最近N笔信号的滚动表现，实时判断"当前市场状态差"？')

        # 5a. 好/坏市场信号特征对比
        rpt.add_section('5a. 好市场 vs 坏市场信号的特征对比',
                         '坏市场期的信号在触发时有没有可观测的不同？')
        if feature_compare:
            feat_headers = ['特征', '好市场', '坏市场', '差异']
            feat_rows = []
            good_f = feature_compare.get('好市场', {})
            bad_f = feature_compare.get('坏市场', {})
            if good_f and bad_f:
                for label, key, fmt in [
                    ('笔数', 'N', '{}'),
                    ('胜率%', 'wr', '{:.1f}%'),
                    ('平均PnL%', 'avg_pnl', '{:+.3f}%'),
                    ('平均ER(20)', 'avg_er20', '{:.3f}'),
                    ('平均偏离ATR', 'avg_dev', '{:.3f}'),
                    ('场景1占比%', 'sc1_pct', '{:.1f}%'),
                    ('场景2占比%', 'sc2_pct', '{:.1f}%'),
                    ('场景3占比%', 'sc3_pct', '{:.1f}%'),
                    ('做多占比%', 'long_pct', '{:.1f}%'),
                ]:
                    gv = good_f.get(key, 0)
                    bv = bad_f.get(key, 0)
                    diff = bv - gv if isinstance(gv, (int, float)) else ''
                    diff_str = f'{diff:+.1f}' if isinstance(diff, (int, float)) else ''
                    feat_rows.append([label, fmt.format(gv), fmt.format(bv), diff_str])
                rpt.add_table(feat_headers, feat_rows, highlight_pnl_cols=[3])

        # 5b. 暂停规则回测：滚动胜率
        rpt.add_section('5b. 滚动胜率暂停规则',
                         '规则：如果最近K笔的胜率 < 阈值 → 跳过当前信号。测试各K和阈值组合')
        rpt.add_text('目标：被跳过的信号（skipped）应该明显比交易的（traded）差。如果skipped的avg_pnl<0，说明规则能识别差环境。')

        # 找最优组合
        best_combo = None
        best_improvement = -999
        wr_headers = ['K', '阈值', '交易N', '交易avgPnl', '交易WR%',
                      '跳过N', '跳过avgPnl', '跳过WR%', '过滤有效?']
        wr_rows = []
        for K in [10, 15, 20]:
            for thr in [0.25, 0.30, 0.35, 0.40]:
                res = pause_results.get((K, thr), {})
                traded = res.get('traded', [])
                skipped = res.get('skipped', [])
                if not traded or not skipped:
                    continue
                t_avg = np.mean(traded)
                s_avg = np.mean(skipped)
                t_wr = sum(1 for p in traded if p > 0) / len(traded) * 100
                s_wr = sum(1 for p in skipped if p > 0) / len(skipped) * 100
                improvement = t_avg - s_avg
                effective = '是' if s_avg < 0 and t_avg > s_avg else '否'
                wr_rows.append([
                    K, f'{thr:.0%}', len(traded), f'{t_avg:+.3f}%', f'{t_wr:.1f}%',
                    len(skipped), f'{s_avg:+.3f}%', f'{s_wr:.1f}%', effective,
                ])
                if improvement > best_improvement and s_avg < 0 and len(skipped) >= 5:
                    best_improvement = improvement
                    best_combo = (K, thr, len(traded), t_avg, len(skipped), s_avg)

        rpt.add_table(wr_headers, wr_rows, highlight_pnl_cols=[3, 6])

        if best_combo:
            K, thr, tn, tavg, sn, savg = best_combo
            rpt.add_text(f'最优组合: K={K}, 阈值={thr:.0%} → '
                         f'交易{tn}笔 avg={tavg:+.3f}% vs 跳过{sn}笔 avg={savg:+.3f}%',
                         color='#58a6ff')

        # 5c. 滚动avgPnl暂停规则
        rpt.add_section('5c. 滚动平均PnL暂停规则',
                         '规则：如果最近K笔的avg_pnl < 0 → 跳过')
        ap_headers = ['K', '交易N', '交易avgPnl', '交易WR%', '交易cumPnl',
                      '跳过N', '跳过avgPnl', '跳过WR%', '跳过cumPnl', '过滤有效?']
        ap_rows = []
        for K in [5, 10, 15, 20]:
            res = avgpnl_results.get(K, {})
            traded = res.get('traded', [])
            skipped = res.get('skipped', [])
            if not traded:
                continue
            t_avg = np.mean(traded)
            t_wr = sum(1 for p in traded if p > 0) / len(traded) * 100
            t_cum = sum(traded)
            s_avg = np.mean(skipped) if skipped else 0
            s_wr = sum(1 for p in skipped if p > 0) / len(skipped) * 100 if skipped else 0
            s_cum = sum(skipped)
            effective = '是' if skipped and s_avg < 0 and t_avg > s_avg else '否'
            ap_rows.append([
                K, len(traded), f'{t_avg:+.3f}%', f'{t_wr:.1f}%', f'{t_cum:+.2f}%',
                len(skipped), f'{s_avg:+.3f}%', f'{s_wr:.1f}%', f'{s_cum:+.2f}%', effective,
            ])
        rpt.add_table(ap_headers, ap_rows, highlight_pnl_cols=[2, 4, 6, 8])

        # 5d. 连续亏损检测
        rpt.add_section('5d. 连续亏损检测',
                         '规则：如果最近M笔全亏 → 跳过下一笔')
        cc_headers = ['连亏M笔', '交易N', '交易avgPnl', '交易WR%',
                      '跳过N', '跳过avgPnl', '跳过WR%', '过滤有效?']
        cc_rows = []
        for M in [3, 4, 5]:
            res = consec_results.get(M, {})
            traded = res.get('traded', [])
            skipped = res.get('skipped', [])
            if not traded:
                continue
            t_avg = np.mean(traded)
            t_wr = sum(1 for p in traded if p > 0) / len(traded) * 100
            s_avg = np.mean(skipped) if skipped else 0
            s_wr = sum(1 for p in skipped if p > 0) / len(skipped) * 100 if skipped else 0
            effective = '是' if skipped and s_avg < 0 else '否'
            cc_rows.append([
                f'连亏{M}笔', len(traded), f'{t_avg:+.3f}%', f'{t_wr:.1f}%',
                len(skipped), f'{s_avg:+.3f}%', f'{s_wr:.1f}%', effective,
            ])
        rpt.add_table(cc_headers, cc_rows, highlight_pnl_cols=[2, 5])

        # Q5 综合结论
        rpt.add_section('Q5 综合结论')
        # 检查各方法是否有效
        effective_methods = []

        if best_combo:
            K, thr, tn, tavg, sn, savg = best_combo
            if savg < -0.1 and sn >= 10:
                effective_methods.append(f'滚动胜率(K={K},thr={thr:.0%}): 跳过{sn}笔avg={savg:+.3f}%')

        for K in [10, 15, 20]:
            res = avgpnl_results.get(K, {})
            skipped = res.get('skipped', [])
            if skipped and np.mean(skipped) < -0.1 and len(skipped) >= 10:
                effective_methods.append(
                    f'滚动avgPnl<0(K={K}): 跳过{len(skipped)}笔avg={np.mean(skipped):+.3f}%')

        for M in [3, 4, 5]:
            res = consec_results.get(M, {})
            skipped = res.get('skipped', [])
            if skipped and np.mean(skipped) < -0.1 and len(skipped) >= 5:
                effective_methods.append(
                    f'连亏{M}笔: 跳过{len(skipped)}笔avg={np.mean(skipped):+.3f}%')

        if effective_methods:
            rpt.add_text('以下检测方法有效（跳过的信号确实亏损）：', color='#3fb950')
            for m in effective_methods:
                rpt.add_text(f'  • {m}', color='#3fb950')
            rpt.add_text('可在监控推送中加入"近期信号健康度"指标，作为仓位调整参考')
        else:
            rpt.add_text('所有检测方法的过滤效果都不明显 → 市场差期的信号特征和正常期无显著差异，'
                         '难以实时检测', color='#f0883e')

    # === Q6 市场广度 ===
    if q6_data:
        rpt.add_section('六、市场广度：ER广度 + 活跃持仓数',
                         '信号触发时全市场有多少品种处于趋势就绪？有多少品种正在持仓？')

        rpt.add_text('核心逻辑：如果很多品种都处于高ER状态、很多品种正在持仓 → 好市场。'
                     '如果只有1-2个品种发信号、其他全部止损出场 → 坏市场。')

        # 6a. ER(20)广度分桶
        rpt.add_section('6a. ER(20)广度：信号触发时有多少品种ER(20)>=0.5')
        # 分桶：0-5, 6-10, 11-15, 16-20, 21+
        breadth_buckets = [
            ('0~5个', 0, 5),
            ('6~10个', 6, 10),
            ('11~15个', 11, 15),
            ('16~20个', 16, 20),
            ('21+个', 21, 99),
        ]
        b_headers = ['ER广度', 'N', 'EV', 'WR%', 'PR', '累计PnL', 'avgPnl']
        b_rows = []
        for label, lo, hi in breadth_buckets:
            trades = [t for t in q6_data if lo <= t['er20_breadth'] <= hi]
            pnls = [t['pnl'] for t in trades]
            st = calc_ev(pnls)
            b_rows.append([label, st['N'], st['EV'], st['wr'], st['pr'],
                          f"{st['sum_pnl']:+.2f}%", f"{st['avg_pnl']:+.3f}%"])
        rpt.add_table(b_headers, b_rows, highlight_pnl_cols=[2, 5])

        # 统计分布
        breadths = [t['er20_breadth'] for t in q6_data]
        rpt.add_text(f'ER(20)广度分布: 最小={min(breadths)}, 最大={max(breadths)}, '
                     f'中位={np.median(breadths):.0f}, 均值={np.mean(breadths):.1f}')

        # 6b. ER(5)广度（短期动量广度）
        rpt.add_section('6b. ER(5)广度：短期动量广度')
        b5_buckets = [
            ('0~3个', 0, 3),
            ('4~6个', 4, 6),
            ('7~9个', 7, 9),
            ('10~12个', 10, 12),
            ('13+个', 13, 99),
        ]
        b5_rows = []
        for label, lo, hi in b5_buckets:
            trades = [t for t in q6_data if lo <= t['er5_breadth'] <= hi]
            pnls = [t['pnl'] for t in trades]
            st = calc_ev(pnls)
            b5_rows.append([label, st['N'], st['EV'], st['wr'], st['pr'],
                           f"{st['sum_pnl']:+.2f}%", f"{st['avg_pnl']:+.3f}%"])
        rpt.add_table(b_headers, b5_rows, highlight_pnl_cols=[2, 5])

        # 6c. 活跃持仓数
        rpt.add_section('6c. 活跃持仓数：信号触发时有多少其他品种正在持仓')
        act_buckets = [
            ('0个', 0, 0),
            ('1~2个', 1, 2),
            ('3~5个', 3, 5),
            ('6~10个', 6, 10),
            ('11+个', 11, 999),
        ]
        act_headers = ['活跃持仓', 'N', 'EV', 'WR%', 'PR', '累计PnL', 'avgPnl']
        act_rows = []
        for label, lo, hi in act_buckets:
            trades = [t for t in q6_data if lo <= t['active_positions'] <= hi]
            pnls = [t['pnl'] for t in trades]
            st = calc_ev(pnls)
            act_rows.append([label, st['N'], st['EV'], st['wr'], st['pr'],
                            f"{st['sum_pnl']:+.2f}%", f"{st['avg_pnl']:+.3f}%"])
        rpt.add_table(act_headers, act_rows, highlight_pnl_cols=[2, 5])

        actives = [t['active_positions'] for t in q6_data]
        rpt.add_text(f'活跃持仓分布: 最小={min(actives)}, 最大={max(actives)}, '
                     f'中位={np.median(actives):.0f}, 均值={np.mean(actives):.1f}')

        # 6d. ER变化广度：多少品种达到"加仓级别"的ER变化
        rpt.add_section('6d. ER变化广度（门槛法）：多少品种达到强变化',
                         '用已验证门槛：ER5Δ6>=0.50 / ER40Δ12>=0.14 | ER20补测多个门槛')
        rpt.add_text('目标：找到"0个品种达标"时的信号表现，做反向过滤（全市场无强动量→不开仓）')

        # 统一分桶
        small_buckets = [('0个', 0, 0), ('1~2个', 1, 2), ('3~5个', 3, 5),
                         ('6~10个', 6, 10), ('11+个', 11, 99)]
        d_headers = ['达标品种数', 'N', 'EV', 'WR%', 'PR', '累计PnL', 'avgPnl']

        # 所有指标×门槛
        all_indicators = [
            ('er5d6_ge_050', 'ER5变化(6根) >= 0.50（场景1加仓门槛）'),
            ('er5d6_ge_025', 'ER5变化(6根) >= 0.25（宽松门槛）'),
            ('er20d6_ge_030', 'ER20变化(6根) >= 0.30'),
            ('er20d6_ge_020', 'ER20变化(6根) >= 0.20'),
            ('er20d6_ge_010', 'ER20变化(6根) >= 0.10'),
            ('er20d12_ge_020', 'ER20变化(12根) >= 0.20'),
            ('er20d12_ge_010', 'ER20变化(12根) >= 0.10'),
            ('er40d12_ge_014', 'ER40变化(12根) >= 0.14（场景2加仓门槛）'),
            ('er40d12_ge_007', 'ER40变化(12根) >= 0.07（半门槛）'),
        ]

        for col, desc in all_indicators:
            # 先看0个档有多少笔，太少就跳过详细表
            zero_trades = [t for t in q6_data if t.get(col, 0) == 0]
            if not zero_trades:
                continue
            rpt.add_text(f'{desc}：')
            d_rows = []
            for label, lo, hi in small_buckets:
                trades = [t for t in q6_data if lo <= t.get(col, 0) <= hi]
                if not trades:
                    continue
                pnls = [t['pnl'] for t in trades]
                st = calc_ev(pnls)
                d_rows.append([label, st['N'], st['EV'], st['wr'], st['pr'],
                              f"{st['sum_pnl']:+.2f}%", f"{st['avg_pnl']:+.3f}%"])
            rpt.add_table(d_headers, d_rows, highlight_pnl_cols=[2, 5])

        # 横向汇总：各指标的"0个达标"档表现
        rpt.add_section('横向汇总：各指标"0个品种达标"时的信号表现')
        rpt.add_text('下表只看"0个品种达到门槛"的信号 vs 其余信号，横向对比哪个指标过滤效果最好')
        zero_headers = ['指标', '0个达标 N', '0个达标 EV', '0个达标 avgPnl',
                        '>=1个达标 N', '>=1个达标 EV', '>=1个达标 avgPnl', 'EV差']
        zero_rows = []
        for col, desc in all_indicators:
            zero = [t['pnl'] for t in q6_data if t.get(col, 0) == 0]
            nonzero = [t['pnl'] for t in q6_data if t.get(col, 0) >= 1]
            if not zero:
                continue
            st0 = calc_ev(zero)
            st1 = calc_ev(nonzero)
            diff = st1['EV'] - st0['EV']
            # 简化描述
            short_desc = desc.split('（')[0] if '（' in desc else desc
            zero_rows.append([short_desc,
                             st0['N'], st0['EV'], f"{st0['avg_pnl']:+.3f}%",
                             st1['N'], st1['EV'], f"{st1['avg_pnl']:+.3f}%",
                             f'{diff:+.2f}'])
        rpt.add_table(zero_headers, zero_rows, highlight_pnl_cols=[2, 5, 7])

        # 6e. 活跃持仓=0 深入验证
        rpt.add_section('6e. 活跃持仓=0 深入验证',
                         '60笔EV=1.75是否稳健？看分布、集中度、去极值后效果')

        zero_trades = [t for t in q6_data if t['active_positions'] == 0]
        if zero_trades:
            zero_pnls = sorted([t['pnl'] for t in zero_trades])
            n_zero = len(zero_trades)

            # PnL分布
            rpt.add_text(f'样本: {n_zero}笔')
            rpt.add_text(f'PnL分布: min={min(zero_pnls):+.2f}% | P25={np.percentile(zero_pnls,25):+.2f}% | '
                         f'P50={np.percentile(zero_pnls,50):+.2f}% | P75={np.percentile(zero_pnls,75):+.2f}% | '
                         f'max={max(zero_pnls):+.2f}%')

            # 去掉最大2笔后
            if n_zero > 5:
                trimmed = sorted(zero_pnls)[:-2]
                st_trim = calc_ev(trimmed)
                rpt.add_text(f'去掉最大2笔后: N={st_trim["N"]} EV={st_trim["EV"]} '
                             f'WR={st_trim["wr"]}% avgPnl={st_trim["avg_pnl"]:+.3f}%')

            # 去掉最大5笔后
            if n_zero > 10:
                trimmed5 = sorted(zero_pnls)[:-5]
                st_trim5 = calc_ev(trimmed5)
                rpt.add_text(f'去掉最大5笔后: N={st_trim5["N"]} EV={st_trim5["EV"]} '
                             f'WR={st_trim5["wr"]}% avgPnl={st_trim5["avg_pnl"]:+.3f}%')

            # 按窗口分布
            window_bars = WINDOW_DAYS * BARS_PER_DAY
            w_counts = {w: [] for w in range(N_WINDOWS)}
            for t in zero_trades:
                df = all_data[t['sym_key']]
                nn = len(df)
                a_start = max(0, nn - LAST_DAYS * BARS_PER_DAY)
                w = min((t['bar_idx'] - a_start) // window_bars, N_WINDOWS - 1)
                if w >= 0:
                    w_counts[w].append(t['pnl'])

            rpt.add_text('按30天窗口分布:')
            w_headers = ['窗口', 'N', 'EV', 'WR%', '累计PnL', 'avgPnl']
            w_rows = []
            for w in range(N_WINDOWS):
                pnls = w_counts[w]
                st = calc_ev(pnls)
                w_rows.append([f'W{w+1}', st['N'], st['EV'], st['wr'],
                              f"{st['sum_pnl']:+.2f}%", f"{st['avg_pnl']:+.3f}%"])
            rpt.add_table(w_headers, w_rows, highlight_pnl_cols=[2, 4])

            # 按场景分布
            rpt.add_text('按场景分布:')
            sc_headers = ['场景', 'N', 'EV', 'WR%', '累计PnL', 'avgPnl']
            sc_rows = []
            for sc in [1, 2, 3]:
                pnls = [t['pnl'] for t in zero_trades if t['scenario'] == sc]
                st = calc_ev(pnls)
                sc_rows.append([f'场景{sc}', st['N'], st['EV'], st['wr'],
                               f"{st['sum_pnl']:+.2f}%", f"{st['avg_pnl']:+.3f}%"])
            rpt.add_table(sc_headers, sc_rows, highlight_pnl_cols=[2, 4])

            # 品种分布（top5）
            sym_counts = {}
            for t in zero_trades:
                sym_counts[t['name']] = sym_counts.get(t['name'], 0) + 1
            top_syms = sorted(sym_counts.items(), key=lambda x: x[1], reverse=True)[:8]
            rpt.add_text(f'品种分布(前8): {", ".join(f"{s}({n})" for s, n in top_syms)}')

            # 对比：活跃持仓1~2 vs 3~5 的同样深入数据
            rpt.add_section('对照组：活跃持仓1~2个 和 3~5个')
            for lo, hi, label in [(1, 2, '1~2个'), (3, 5, '3~5个')]:
                grp = [t for t in q6_data if lo <= t['active_positions'] <= hi]
                if not grp:
                    continue
                grp_pnls = [t['pnl'] for t in grp]
                st = calc_ev(grp_pnls)
                rpt.add_text(f'活跃持仓{label}: N={st["N"]} EV={st["EV"]} WR={st["wr"]}% '
                             f'avgPnl={st["avg_pnl"]:+.3f}%')
                rpt.add_text(f'  PnL分布: P25={np.percentile(grp_pnls,25):+.2f}% | '
                             f'P50={np.percentile(grp_pnls,50):+.2f}% | '
                             f'P75={np.percentile(grp_pnls,75):+.2f}%')

        # Q6 综合结论
        rpt.add_section('Q6 综合结论')

    # === K线图案例 ===
    if chart_htmls:
        rpt.add_section('七、K线案例：崩塌型 vs 逆袭型',
                         '前期好后期差 vs 前期差后期好的品种，各展示好/差期各1笔')
        for ch in chart_htmls:
            rpt.add_chart(ch)
    else:
        rpt.add_section('七、K线案例')
        rpt.add_text('未找到符合条件的崩塌型/逆袭型品种案例')

    return rpt


# ============ 主流程 ============

def run():
    print("=" * 70)
    print("品种适配度研究：稳定性 + 滚动预测 + 截面ER排名")
    print("=" * 70)

    # 加载数据
    print("\n[1/5] 加载全品种数据...")
    all_data = load_all(period_min=10, days=170, last_days=None,
                        emas=(10, 20, 120), er_periods=(5, 20, 40), atr_period=14)
    print(f"加载品种数: {len(all_data)}")

    # 收集信号
    print("\n[2/5] 收集全品种信号...")
    all_trades = collect_all_trades(all_data)
    valid = [t for t in all_trades if t['exit_reason'] != 'backtest_end']
    print(f"合格信号总数: {len(all_trades)} (有效出场: {len(valid)})")
    for sc in [1, 2, 3]:
        sc_n = sum(1 for t in valid if t['scenario'] == sc)
        print(f"  场景{sc}: {sc_n}笔")

    # ER40广度反向过滤效果
    filtered = [t for t in valid if t.get('filtered_er40b')]
    passed = [t for t in valid if not t.get('filtered_er40b')]
    if filtered:
        st_f = calc_ev([t['pnl'] for t in filtered])
        st_p = calc_ev([t['pnl'] for t in passed])
        print(f"\n  ER40广度过滤: 过滤掉{len(filtered)}笔(EV={st_f['EV']}) | "
              f"保留{len(passed)}笔(EV={st_p['EV']})")

    # Q1分析
    print("\n[3/5] Q1: 品种时间稳定性分析...")
    heatmap, stability, correlations = analyze_q1(all_trades, all_data)
    for cat in ['稳定正向', '波动型', '稳定负向', '样本不足']:
        cnt = sum(1 for v in stability.values() if v == cat)
        if cnt:
            print(f"  {cat}: {cnt}个品种")

    # Q2分析
    print("\n[4/5] Q2: 滚动7天Walk-Forward...")
    walk_forward = analyze_q2(all_trades, all_data)
    if walk_forward:
        hit = sum(1 for w in walk_forward if w['spread'] > 0)
        print(f"  Top10胜出: {hit}/{len(walk_forward)} 步")

    # Q3分析
    print("\n[4/5] Q3: 截面ER(20)排名...")
    q3_coarse, q3_fine, q3_scenario, _ = analyze_q3(all_trades, all_data)
    for label, trades in q3_coarse.items():
        st = calc_ev([t['pnl'] for t in trades])
        print(f"  {label}: N={st['N']} EV={st['EV']}")

    # Q4分析
    print("\n[5/7] Q4: 市场同步性分析...")
    q4_data = analyze_q4(all_trades, all_data)
    block_stats, good_pnls, bad_pnls, block_indicators, indicator_corrs, avg_cross_corr = q4_data
    print(f"  好市场期信号: {len(good_pnls)}笔 | 坏市场期: {len(bad_pnls)}笔")
    if avg_cross_corr is not None:
        print(f"  品种间截面相关: {avg_cross_corr:+.3f}")
    for col, label in [('avg_er', 'avgER'), ('trend_pct', '趋势%')]:
        if col in indicator_corrs:
            print(f"  {label} vs 市场PnL: {indicator_corrs[col]:+.3f}")

    # Q5分析
    print("\n[6/8] Q5: 实时市场状态检测...")
    q5_data = analyze_q5(all_trades, all_data)
    rolling_data, pause_results, avgpnl_results, consec_results, feature_compare = q5_data
    good_f = feature_compare.get('好市场', {})
    bad_f = feature_compare.get('坏市场', {})
    if good_f and bad_f:
        print(f"  好市场: N={good_f['N']} WR={good_f['wr']}% avgPnl={good_f['avg_pnl']:+.3f}%")
        print(f"  坏市场: N={bad_f['N']} WR={bad_f['wr']}% avgPnl={bad_f['avg_pnl']:+.3f}%")

    # Q6分析
    print("\n[7/9] Q6: 市场广度分析...")
    q6_data = analyze_q6(all_trades, all_data)
    if q6_data:
        breadths = [t['er20_breadth'] for t in q6_data]
        actives = [t['active_positions'] for t in q6_data]
        print(f"  ER(20)广度: 中位={np.median(breadths):.0f}, 均值={np.mean(breadths):.1f}")
        print(f"  活跃持仓: 中位={np.median(actives):.0f}, 均值={np.mean(actives):.1f}")

    # K线案例
    print("\n[8/9] 生成K线案例...")
    chart_htmls = find_chart_cases(heatmap, all_trades, all_data)
    print(f"  K线图: {len(chart_htmls)}张")

    # 生成报告
    print("\n[9/9] 生成报告...")
    rpt = build_report(heatmap, stability, correlations, walk_forward,
                       q3_coarse, q3_fine, q3_scenario, chart_htmls, all_trades,
                       q4_data=q4_data, q5_data=q5_data, q6_data=q6_data,
                       all_data=all_data)
    rpt.save('output/symbol_fitness.html')
    print("\n完成!")


if __name__ == '__main__':
    run()
