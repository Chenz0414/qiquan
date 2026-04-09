# -*- coding: utf-8 -*-
"""
趋势判断方法 × 出场策略 全因子交叉验证
=======================================
3种趋势方法 × 5种出场 × ABC信号 × ER过滤
32品种 120天，Bootstrap CI + 时间窗口 + 品种组 + LOSO鲁棒性
"""

import os
import random
import base64
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import timedelta
from signal_core import SYMBOL_CONFIGS, ExitTracker

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STOP_LOSS_PCT = 2.0
TAKE_PROFIT_PCT = 5.0
STOP_TICKS = 5
MIN_PB_BARS_BC = 4   # B类1~3根, C类>=4根 (原代码B类即>=4根)
WARMUP = 130
LAST_DAYS = 120
BARS_PER_DAY = 57

HIGH_VOL_KEYS = {"GFEX.lc", "DCE.jm", "SHFE.ag", "CZCE.FG", "CZCE.SA",
                 "INE.sc", "CZCE.MA", "CZCE.TA", "DCE.eb", "DCE.lh"}

EXIT_NAMES = ['Fixed', 'S1', 'S2', 'S3', 'S4']

MA_COLORS = {
    'ema_10': ('#FF9800', 'EMA10'),
    'ema_20': ('#2196F3', 'EMA20'),
    'ema_60': ('#E91E63', 'EMA60'),
    'ema_120': ('#9C27B0', 'EMA120'),
}


# ============================================================
# 工具函数
# ============================================================

def sym_key_to_config_key(sym_key):
    parts = sym_key.split('_', 1)
    return f"{parts[0]}.{parts[1]}" if len(parts) == 2 else sym_key


def get_sym_name(sym_key):
    ck = sym_key_to_config_key(sym_key)
    return SYMBOL_CONFIGS[ck]['name'] if ck in SYMBOL_CONFIGS else sym_key


def get_tick_size(sym_key):
    ck = sym_key_to_config_key(sym_key)
    return SYMBOL_CONFIGS[ck]['tick_size'] if ck in SYMBOL_CONFIGS else 1


def load_all_symbols():
    """加载全部10min品种数据"""
    sym_data = {}
    for f in sorted(os.listdir(CACHE_DIR)):
        if not f.endswith('_10min_170d.parquet'):
            continue
        sym_key = f.replace('_10min_170d.parquet', '')
        df = pd.read_parquet(os.path.join(CACHE_DIR, f))
        # 计算EMA
        for p in [10, 20, 60, 120]:
            df[f'ema_{p}'] = df['close'].ewm(span=p, adjust=False).mean()
        # 计算ER(20)
        net = (df['close'] - df['close'].shift(20)).abs()
        bar_sum = df['close'].diff().abs().rolling(20).sum()
        df['er_20'] = net / bar_sum.replace(0, np.nan)
        # 截取120天
        cutoff = df['datetime'].iloc[-1] - timedelta(days=LAST_DAYS)
        df_cut = df[df['datetime'] >= cutoff].copy().reset_index(drop=True)
        if len(df_cut) < 200:
            continue
        sym_data[sym_key] = df_cut
    return sym_data


# ============================================================
# 趋势判断函数
# ============================================================

def trend_ma10_ma20(df, i):
    row = df.iloc[i]
    m10, m20 = row['ema_10'], row['ema_20']
    if pd.isna(m10) or pd.isna(m20):
        return 0
    return 1 if m10 > m20 else (-1 if m10 < m20 else 0)


def trend_triple(df, i):
    row = df.iloc[i]
    m10, m20, m60 = row['ema_10'], row['ema_20'], row['ema_60']
    if pd.isna(m10) or pd.isna(m20) or pd.isna(m60):
        return 0
    if m10 > m20 > m60:
        return 1
    elif m10 < m20 < m60:
        return -1
    return 0


def trend_ma20_ma120(df, i):
    row = df.iloc[i]
    m20, m120 = row['ema_20'], row['ema_120']
    if pd.isna(m20) or pd.isna(m120):
        return 0
    return 1 if m20 > m120 else (-1 if m20 < m120 else 0)


TREND_METHODS = [
    ('MA10>MA20', trend_ma10_ma20),
    ('三线排列', trend_triple),
    ('MA20>MA120', trend_ma20_ma120),
]


# ============================================================
# 信号检测（每个趋势方法独立运行自己的状态机）
# ============================================================

def detect_signals_for_method(df, trend_fn):
    """
    用指定趋势函数独立驱动状态机检测A/B/C类信号。
    每个趋势方法有自己的回调追踪状态。
    """
    signals = []
    n = len(df)

    b_below_start = -1
    b_pb_low = None
    b_pb_high = None
    prev_trend = 0
    prev_close = None
    prev_ema10 = None

    for i in range(1, n):
        row = df.iloc[i]
        close, high, low = row['close'], row['high'], row['low']
        ema10 = row['ema_10']

        if pd.isna(ema10) or pd.isna(row['ema_120']):
            prev_close, prev_ema10 = close, ema10
            continue

        if prev_close is None:
            prev_close, prev_ema10 = close, ema10
            continue

        trend = trend_fn(df, i)

        if trend != prev_trend and trend != 0:
            b_below_start = -1
            b_pb_low = b_pb_high = None
        prev_trend = trend

        if trend == 0 or i < WARMUP:
            prev_close, prev_ema10 = close, ema10
            continue

        er_20 = row['er_20'] if not pd.isna(row['er_20']) else np.nan
        found = []

        if trend == 1:
            if low <= ema10 and close > ema10 and prev_close > prev_ema10:
                if b_below_start == -1:
                    found.append(('A', 'long', low, 0))
            if b_below_start == -1:
                if close < ema10 and prev_close >= prev_ema10:
                    b_below_start = i
                    b_pb_low = low
            else:
                b_pb_low = min(b_pb_low, low)
                if close > ema10:
                    pb_bars = i - b_below_start
                    if pb_bars >= 1:
                        sig_type = 'B' if pb_bars <= 3 else 'C'
                        found.append((sig_type, 'long', b_pb_low, pb_bars))
                    b_below_start = -1
                    b_pb_low = None

        elif trend == -1:
            if high >= ema10 and close < ema10 and prev_close < prev_ema10:
                if b_below_start == -1:
                    found.append(('A', 'short', high, 0))
            if b_below_start == -1:
                if close > ema10 and prev_close <= prev_ema10:
                    b_below_start = i
                    b_pb_high = high
            else:
                b_pb_high = max(b_pb_high, high)
                if close < ema10:
                    pb_bars = i - b_below_start
                    if pb_bars >= 1:
                        sig_type = 'B' if pb_bars <= 3 else 'C'
                        found.append((sig_type, 'short', b_pb_high, pb_bars))
                    b_below_start = -1
                    b_pb_high = None

        for sig_type, direction, pb_extreme, pb_bars in found:
            signals.append({
                'idx': i,
                'type': sig_type,
                'direction': direction,
                'entry_price': close,
                'pullback_extreme': pb_extreme,
                'pullback_bars': pb_bars,
                'er_20': er_20,
                'time': row['datetime'],
            })

        prev_close, prev_ema10 = close, ema10

    return signals


def detect_all_signals(df):
    """
    分别用3种趋势方法独立检测信号，然后合并。
    用(idx, direction)作为信号身份标识匹配重叠信号。
    """
    method_signals = {}
    for m_name, m_fn in TREND_METHODS:
        sigs = detect_signals_for_method(df, m_fn)
        method_signals[m_name] = sigs

    # 建立每个方法的信号索引: (idx, direction) -> signal
    method_index = {}
    for m_name in method_signals:
        idx_map = {}
        for sig in method_signals[m_name]:
            key = (sig['idx'], sig['direction'])
            idx_map[key] = sig
        method_index[m_name] = idx_map

    # 收集所有唯一的(idx, direction)
    all_keys = set()
    for m_name in method_signals:
        for sig in method_signals[m_name]:
            all_keys.add((sig['idx'], sig['direction']))

    # 合并：使用最早检测到该信号的方法的数据
    merged = []
    for key in sorted(all_keys):
        sig_data = None
        for m_name, _ in TREND_METHODS:
            if key in method_index[m_name]:
                sig_data = method_index[m_name][key].copy()
                break
        if sig_data is None:
            continue

        # 标记每个方法是否接纳此信号
        for m_name, _ in TREND_METHODS:
            sig_data[f'admitted_{m_name}'] = (key in method_index[m_name])

        merged.append(sig_data)

    return merged


# ============================================================
# 出场模拟
# ============================================================

def simulate_fixed_exit(df, sig):
    """固定2%SL/5%TP"""
    entry_idx = sig['idx']
    entry_price = sig['entry_price']
    is_long = sig['direction'] == 'long'
    n = len(df)

    sl = entry_price * (1 - STOP_LOSS_PCT / 100) if is_long else entry_price * (1 + STOP_LOSS_PCT / 100)
    tp = entry_price * (1 + TAKE_PROFIT_PCT / 100) if is_long else entry_price * (1 - TAKE_PROFIT_PCT / 100)

    for j in range(entry_idx + 1, n):
        bar = df.iloc[j]
        if is_long:
            if bar['low'] <= sl:
                return -STOP_LOSS_PCT, j - entry_idx, 'stop'
            if bar['high'] >= tp:
                return +TAKE_PROFIT_PCT, j - entry_idx, 'tp'
        else:
            if bar['high'] >= sl:
                return -STOP_LOSS_PCT, j - entry_idx, 'stop'
            if bar['low'] <= tp:
                return +TAKE_PROFIT_PCT, j - entry_idx, 'tp'

    last_close = df.iloc[-1]['close']
    pnl = ((last_close - entry_price) / entry_price * 100) if is_long else ((entry_price - last_close) / entry_price * 100)
    return pnl, n - 1 - entry_idx, 'backtest_end'


def simulate_trailing_exits(df, sig, tick_size):
    """S1/S2/S3追踪出场"""
    entry_idx = sig['idx']
    entry_price = sig['entry_price']
    is_long = sig['direction'] == 'long'
    n = len(df)

    tracker = ExitTracker(
        direction=sig['direction'],
        entry_price=entry_price,
        pullback_extreme=sig['pullback_extreme'],
        tick_size=tick_size,
        stop_ticks=STOP_TICKS,
    )

    results = {'S1': None, 'S2': None, 'S3': None}
    mfe = 0.0
    mae = 0.0

    for j in range(entry_idx + 1, n):
        bar = df.iloc[j]
        prev_bar = df.iloc[j - 1]
        if pd.isna(bar['ema_10']):
            continue

        # MFE/MAE
        if is_long:
            fav = (bar['high'] - entry_price) / entry_price * 100
            adv = (entry_price - bar['low']) / entry_price * 100
        else:
            fav = (entry_price - bar['low']) / entry_price * 100
            adv = (bar['high'] - entry_price) / entry_price * 100
        mfe = max(mfe, fav)
        mae = max(mae, adv)

        if not tracker.all_done():
            exit_events, _ = tracker.process_bar(
                close=bar['close'], high=bar['high'], low=bar['low'],
                ema10=bar['ema_10'], prev_close=prev_bar['close'],
                prev_high=prev_bar['high'], prev_low=prev_bar['low'],
            )
            for ev in exit_events:
                if results[ev.strategy] is None:
                    results[ev.strategy] = {
                        'pnl': ev.pnl_pct, 'bars': ev.bars_held, 'reason': ev.exit_reason,
                    }

        if tracker.all_done():
            break

    # 强制平仓
    forced = tracker.force_close(df.iloc[-1]['close'])
    for ev in forced:
        if results[ev.strategy] is None:
            results[ev.strategy] = {
                'pnl': ev.pnl_pct, 'bars': ev.bars_held, 'reason': ev.exit_reason,
            }

    return results, mfe, mae


def simulate_all_exits(df, sig, tick_size):
    """模拟全部5种出场"""
    # 固定出场
    fixed_pnl, fixed_bars, fixed_reason = simulate_fixed_exit(df, sig)

    # 追踪出场
    trailing, mfe, mae = simulate_trailing_exits(df, sig, tick_size)

    s1 = trailing['S1']
    s2 = trailing['S2']
    s3 = trailing['S3']

    # S4 = 半仓S1 + 半仓S2
    s4_pnl = (s1['pnl'] + s2['pnl']) / 2
    s4_bars = max(s1['bars'], s2['bars'])

    return {
        'Fixed_pnl': round(fixed_pnl, 4), 'Fixed_bars': fixed_bars, 'Fixed_reason': fixed_reason,
        'S1_pnl': round(s1['pnl'], 4), 'S1_bars': s1['bars'], 'S1_reason': s1['reason'],
        'S2_pnl': round(s2['pnl'], 4), 'S2_bars': s2['bars'], 'S2_reason': s2['reason'],
        'S3_pnl': round(s3['pnl'], 4), 'S3_bars': s3['bars'], 'S3_reason': s3['reason'],
        'S4_pnl': round(s4_pnl, 4), 'S4_bars': s4_bars,
        'mfe': round(mfe, 4), 'mae': round(mae, 4),
    }


# ============================================================
# 统计指标
# ============================================================

def calc_ev(pnls):
    """返回 (EV, 胜率, 盈亏比, N)"""
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
    return round(ev, 4), round(wr, 4), round(ratio, 4), n


def calc_group_stats(df_sub, exit_name):
    """计算一个分组在某出场下的统计"""
    pnl_col = f'{exit_name}_pnl'
    bars_col = f'{exit_name}_bars'
    if len(df_sub) == 0:
        return {'N': 0, 'EV': 0, 'wr': 0, 'pr': 0, 'sum': 0, 'bars': 0, 'burst': 0, 'capture': 0}
    pnls = df_sub[pnl_col].values
    ev, wr, pr, n = calc_ev(pnls)
    avg_bars = df_sub[bars_col].mean()
    burst = (df_sub['mfe'] >= 3.0).mean() if 'mfe' in df_sub.columns else 0
    # MFE捕获率
    valid_mfe = df_sub[df_sub['mfe'] > 0]
    capture = (valid_mfe[pnl_col] / valid_mfe['mfe']).mean() if len(valid_mfe) > 0 else 0
    return {
        'N': n, 'EV': ev, 'wr': round(wr * 100, 1), 'pr': pr,
        'sum': round(pnls.sum(), 2), 'bars': round(avg_bars, 1),
        'burst': round(burst * 100, 1), 'capture': round(capture, 2),
    }


def bootstrap_ev_diff(pnls_a, pnls_b, n_boot=10000):
    """Bootstrap两组PnL的EV差异置信区间"""
    a = np.array(pnls_a)
    b = np.array(pnls_b)
    diffs = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sa = a[rng.integers(0, len(a), len(a))]
        sb = b[rng.integers(0, len(b), len(b))]
        ev_a = calc_ev(sa)[0]
        ev_b = calc_ev(sb)[0]
        diffs.append(ev_a - ev_b)
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    mean_diff = np.mean(diffs)
    significant = (lo > 0) or (hi < 0)
    return round(mean_diff, 4), round(lo, 4), round(hi, 4), significant


# ============================================================
# K线图绘制
# ============================================================

def draw_trade_chart(df, sig, exit_fixed, exit_s2, sym_name):
    """画K线图，同时标注固定出场和S2出场"""
    entry_idx = sig['idx']
    is_long = sig['direction'] == 'long'

    fixed_bars = exit_fixed['Fixed_bars']
    s2_bars = exit_s2['S2_bars']
    max_exit_bar = max(fixed_bars, s2_bars)

    start = max(0, entry_idx - 25)
    end = min(len(df) - 1, entry_idx + max_exit_bar + 8)
    segment = df.iloc[start:end + 1].copy().reset_index(drop=True)
    entry_pos = entry_idx - start

    fig, ax = plt.subplots(figsize=(16, 7))

    # K线
    for idx in range(len(segment)):
        row = segment.iloc[idx]
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        color = '#ef5350' if c >= o else '#26a69a'
        ax.plot([idx, idx], [l, h], color=color, linewidth=0.8)
        body_bottom = min(o, c)
        body_height = max(abs(c - o), (h - l) * 0.01)
        rect = plt.Rectangle((idx - 0.35, body_bottom), 0.7, body_height,
                              facecolor=color, edgecolor=color, linewidth=0.5)
        ax.add_patch(rect)

    # EMA
    xs = np.arange(len(segment))
    for ma_col in ['ema_10', 'ema_20', 'ema_60', 'ema_120']:
        if ma_col in segment.columns:
            vals = segment[ma_col].values
            valid = ~np.isnan(vals)
            clr, label = MA_COLORS[ma_col]
            lw = 1.8 if ma_col in ['ema_20', 'ema_120'] else 1.0
            style = '--' if ma_col == 'ema_10' else '-'
            ax.plot(xs[valid], vals[valid], color=clr, linewidth=lw,
                    linestyle=style, alpha=0.85, label=label)

    # 入场
    marker = '^' if is_long else 'v'
    ax.scatter(entry_pos, sig['entry_price'], marker=marker, s=200,
               color='#FF1744', zorder=10, edgecolors='black', linewidths=0.8)
    ax.annotate(f'入场 {sig["entry_price"]:.1f}',
                (entry_pos, sig['entry_price']),
                textcoords="offset points", xytext=(-40, 20 if is_long else -25),
                fontsize=9, fontweight='bold', color='#FF1744',
                arrowprops=dict(arrowstyle='->', color='#FF1744', lw=1.2))

    # 固定出场标注
    fixed_exit_pos = entry_pos + fixed_bars
    if fixed_exit_pos < len(segment):
        fixed_pnl = exit_fixed['Fixed_pnl']
        fc = '#4CAF50' if fixed_pnl > 0 else '#f44336'
        ax.scatter(fixed_exit_pos, segment.iloc[fixed_exit_pos]['close'], marker='x',
                   s=150, color=fc, zorder=10, linewidths=2)
        ax.annotate(f'Fixed {fixed_pnl:+.2f}%',
                    (fixed_exit_pos, segment.iloc[fixed_exit_pos]['close']),
                    textcoords="offset points", xytext=(10, -25),
                    fontsize=8, color=fc, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=fc, alpha=0.8))

    # S2出场标注
    s2_exit_pos = entry_pos + s2_bars
    if s2_exit_pos < len(segment):
        s2_pnl = exit_s2['S2_pnl']
        sc = '#2196F3' if s2_pnl > 0 else '#FF5722'
        ax.scatter(s2_exit_pos, segment.iloc[s2_exit_pos]['close'], marker='D',
                   s=100, color=sc, zorder=10, linewidths=1.5)
        ax.annotate(f'S2 {s2_pnl:+.2f}%',
                    (s2_exit_pos, segment.iloc[s2_exit_pos]['close']),
                    textcoords="offset points", xytext=(10, 15),
                    fontsize=8, color=sc, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=sc, alpha=0.8))

    # X轴
    tick_step = max(1, len(segment) // 10)
    tick_positions = list(range(0, len(segment), tick_step))
    tick_labels = [segment.iloc[tp]['datetime'].strftime('%m/%d %H:%M')
                   if hasattr(segment.iloc[tp]['datetime'], 'strftime')
                   else str(segment.iloc[tp]['datetime'])[:16]
                   for tp in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30, fontsize=7)

    dir_str = '做多' if is_long else '做空'
    ax.set_title(
        f"{sym_name} | 被过滤信号 | {sig['type']}类{dir_str} | "
        f"ER={sig['er_20']:.3f} | Fixed:{exit_fixed['Fixed_pnl']:+.2f}% S2:{exit_s2['S2_pnl']:+.2f}%",
        fontsize=11, fontweight='bold', color='#ff6b6b')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ============================================================
# HTML报告生成
# ============================================================

def ev_color(ev):
    if ev >= 0.15:
        return '#4CAF50'
    elif ev >= 0.05:
        return '#FFC107'
    elif ev >= 0:
        return '#FF9800'
    return '#f44336'


def ev_cell(ev, n):
    if n < 50:
        return f'<td style="color:#666;font-style:italic" title="N={n}<50 样本不足">{ev:+.2f}*</td>'
    return f'<td style="color:{ev_color(ev)};font-weight:bold">{ev:+.2f}</td>'


def generate_html(df_all, matrix_stats, filtered_stats, bootstrap_results,
                  window_stats, group_stats, loso_stats, chart_images):
    """生成HTML报告"""

    # --- 判断结论 ---
    best_method_counts = {}
    for exit_name in EXIT_NAMES:
        best_ev = -999
        best_m = ''
        for row in matrix_stats:
            m_name, e_name, sig_label, stats, er_filtered = row
            if e_name == exit_name and sig_label == '全部' and not er_filtered:
                if m_name not in best_method_counts:
                    best_method_counts[m_name] = 0
                if stats['EV'] > best_ev:
                    best_ev = stats['EV']
                    best_m = m_name
        if best_m:
            best_method_counts[best_m] = best_method_counts.get(best_m, 0) + 1

    html = [f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>趋势判断 × 出场策略 全因子验证</title>
<style>
body {{ background: #1a1a2e; color: #e0e0e0; font-family: 'Microsoft YaHei', sans-serif; margin: 20px; }}
h1 {{ color: #ff6b6b; text-align: center; border-bottom: 2px solid #ff6b6b; padding-bottom: 10px; }}
h2 {{ color: #ffd700; margin-top: 30px; border-left: 4px solid #ffd700; padding-left: 12px; }}
h3 {{ color: #00d4ff; margin-top: 20px; }}
.summary {{ background: #16213e; border-radius: 10px; padding: 15px 20px; margin: 15px 0;
           border: 1px solid #0f3460; font-size: 14px; line-height: 1.8; }}
.summary b {{ color: #00d4ff; }}
table {{ border-collapse: collapse; margin: 15px 0; font-size: 13px; width: auto; }}
th {{ background: #0f3460; padding: 8px 12px; text-align: center; white-space: nowrap; }}
td {{ padding: 6px 12px; text-align: center; border-bottom: 1px solid #1a1a3e; white-space: nowrap; }}
tr:hover {{ background: #1a2a4e; }}
.best {{ background: #1a3a2e !important; }}
.trade-card {{ background: #16213e; border-radius: 10px; padding: 15px; margin: 15px 0;
              border: 1px solid #0f3460; }}
.trade-card img {{ width: 100%; border-radius: 6px; }}
.info-tag {{ display: inline-block; background: #0f3460; padding: 3px 10px; border-radius: 12px;
            font-size: 12px; margin: 2px 4px; }}
.win {{ color: #4CAF50; font-weight: bold; }}
.loss {{ color: #f44336; font-weight: bold; }}
.sig {{ color: #4CAF50; }} .insig {{ color: #666; }}
</style></head><body>
<h1>趋势判断方法 × 出场策略 全因子验证</h1>
<div class="summary">
    <b>测试规模:</b> {len(df_all)}笔信号 | 32品种 | 120天 | 10min<br>
    <b>趋势方法:</b> MA10>MA20 / 三线排列(10>20>60) / MA20>MA120<br>
    <b>出场策略:</b> 固定(2%SL/5%TP) / S1当根新高 / S2回调追踪 / S3前根新高 / S4(½S1+½S2)<br>
    <b>信号类型:</b> A(影线弹回) / B(1~3根回调) / C(≥4根回调)
</div>
"""]

    # === 1. 主对比表 ===
    html.append('<h2>1. 主对比表：趋势方法 × 出场策略（全部信号 + ER过滤）</h2>')

    for er_label, er_filter in [('全部信号', False), ('ER(20)≥0.5', True)]:
        html.append(f'<h3>{er_label}</h3>')
        html.append('<table><tr><th>趋势方法</th>')
        for e in EXIT_NAMES:
            html.append(f'<th colspan="4">{e}</th>')
        html.append('</tr><tr><th></th>')
        for _ in EXIT_NAMES:
            html.append('<th>EV</th><th>N</th><th>胜率</th><th>盈亏比</th>')
        html.append('</tr>')

        for m_name in ['MA10>MA20', '三线排列', 'MA20>MA120']:
            html.append(f'<tr><td><b>{m_name}</b></td>')
            for exit_name in EXIT_NAMES:
                # 找到对应统计
                stats = None
                for row in matrix_stats:
                    if row[0] == m_name and row[1] == exit_name and row[2] == '全部' and row[4] == er_filter:
                        stats = row[3]
                        break
                if stats:
                    html.append(ev_cell(stats['EV'], stats['N']))
                    html.append(f'<td>{stats["N"]}</td>')
                    html.append(f'<td>{stats["wr"]:.1f}%</td>')
                    html.append(f'<td>{stats["pr"]:.2f}</td>')
                else:
                    html.append('<td>-</td><td>-</td><td>-</td><td>-</td>')
            html.append('</tr>')
        html.append('</table>')

    # === 2. 信号类型分表 ===
    html.append('<h2>2. 按信号类型分解（无ER过滤）</h2>')
    for sig_type in ['A', 'B', 'C']:
        html.append(f'<h3>{sig_type}类信号</h3>')
        html.append('<table><tr><th>趋势方法</th>')
        for e in EXIT_NAMES:
            html.append(f'<th>{e} EV</th><th>N</th>')
        html.append('</tr>')
        for m_name in ['MA10>MA20', '三线排列', 'MA20>MA120']:
            html.append(f'<tr><td><b>{m_name}</b></td>')
            for exit_name in EXIT_NAMES:
                stats = None
                for row in matrix_stats:
                    if row[0] == m_name and row[1] == exit_name and row[2] == sig_type and not row[4]:
                        stats = row[3]
                        break
                if stats:
                    html.append(ev_cell(stats['EV'], stats['N']))
                    html.append(f'<td>{stats["N"]}</td>')
                else:
                    html.append('<td>-</td><td>-</td>')
            html.append('</tr>')
        html.append('</table>')

    # === 3. 被过滤信号分析 ===
    html.append('<h2>3. 被过滤信号分析（MA10>MA20接纳 但 MA20>MA120拒绝）</h2>')
    html.append('<table><tr><th>分类</th>')
    for e in EXIT_NAMES:
        html.append(f'<th>{e} EV</th><th>N</th>')
    html.append('</tr>')
    for label, stats_row in filtered_stats:
        html.append(f'<tr><td><b>{label}</b></td>')
        for exit_name in EXIT_NAMES:
            st = stats_row.get(exit_name)
            if st and st['N'] > 0:
                html.append(ev_cell(st['EV'], st['N']))
                html.append(f'<td>{st["N"]}</td>')
            else:
                html.append('<td>-</td><td>-</td>')
        html.append('</tr>')
    html.append('</table>')

    # === 4. Bootstrap CI ===
    html.append('<h2>4. Bootstrap置信区间（MA20>MA120 vs MA10>MA20 EV差）</h2>')
    html.append('<table><tr><th>出场</th><th>信号</th><th>EV差均值</th><th>95% CI下界</th><th>95% CI上界</th><th>显著</th></tr>')
    for row in bootstrap_results:
        exit_name, sig_type, mean_d, lo, hi, sig_flag = row
        sig_cls = 'sig' if sig_flag else 'insig'
        sig_text = '✓显著' if sig_flag else '✗不显著'
        html.append(f'<tr><td>{exit_name}</td><td>{sig_type}</td>')
        html.append(f'<td style="color:{ev_color(mean_d)}">{mean_d:+.4f}</td>')
        html.append(f'<td>{lo:+.4f}</td><td>{hi:+.4f}</td>')
        html.append(f'<td class="{sig_cls}">{sig_text}</td></tr>')
    html.append('</table>')

    # === 5. 时间窗口稳定性 ===
    html.append('<h2>5. 时间窗口稳定性（4×30天）</h2>')
    for exit_name in ['Fixed', 'S2', 'S4']:
        html.append(f'<h3>{exit_name}出场</h3>')
        html.append('<table><tr><th>趋势方法</th><th>W1 EV</th><th>W2 EV</th><th>W3 EV</th><th>W4 EV</th><th>胜窗口</th></tr>')
        for m_name in ['MA10>MA20', '三线排列', 'MA20>MA120']:
            html.append(f'<tr><td><b>{m_name}</b></td>')
            win_count = 0
            for w in range(4):
                st = None
                for row in window_stats:
                    if row[0] == m_name and row[1] == exit_name and row[2] == w:
                        st = row[3]
                        break
                if st:
                    html.append(ev_cell(st['EV'], st['N']))
                    if st['EV'] > 0:
                        win_count += 1
                else:
                    html.append('<td>-</td>')
            html.append(f'<td>{win_count}/4</td></tr>')
        html.append('</table>')

    # === 6. 品种组稳定性 ===
    html.append('<h2>6. 品种组稳定性（高波动 vs 常规）</h2>')
    html.append('<table><tr><th>趋势方法</th><th>组</th>')
    for e in ['Fixed', 'S1', 'S2', 'S3', 'S4']:
        html.append(f'<th>{e} EV</th>')
    html.append('</tr>')
    for m_name in ['MA10>MA20', '三线排列', 'MA20>MA120']:
        for grp in ['高波动', '常规']:
            html.append(f'<tr><td><b>{m_name}</b></td><td>{grp}</td>')
            for exit_name in EXIT_NAMES:
                st = None
                for row in group_stats:
                    if row[0] == m_name and row[1] == exit_name and row[2] == grp:
                        st = row[3]
                        break
                if st:
                    html.append(ev_cell(st['EV'], st['N']))
                else:
                    html.append('<td>-</td>')
            html.append('</tr>')
    html.append('</table>')

    # === 7. LOSO ===
    html.append('<h2>7. LOSO鲁棒性（剔除单品种后EV范围）</h2>')
    html.append('<table><tr><th>趋势方法</th><th>出场</th><th>EV最低</th><th>EV最高</th><th>排名翻转?</th></tr>')
    for row in loso_stats:
        m_name, exit_name, ev_min, ev_max, flipped = row
        flip_cls = 'loss' if flipped else 'win'
        flip_text = '是!' if flipped else '否'
        html.append(f'<tr><td><b>{m_name}</b></td><td>{exit_name}</td>')
        html.append(f'<td>{ev_min:+.4f}</td><td>{ev_max:+.4f}</td>')
        html.append(f'<td class="{flip_cls}">{flip_text}</td></tr>')
    html.append('</table>')

    # === 8. K线图 ===
    if chart_images:
        html.append('<h2>8. 被过滤信号K线验证（随机10笔）</h2>')
        for info, img_b64 in chart_images:
            pnl_f = info.get('Fixed_pnl', 0)
            pnl_s2 = info.get('S2_pnl', 0)
            html.append(f"""<div class="trade-card">
<div>
    <span class="info-tag">{info['sym_name']}</span>
    <span class="info-tag">{info['type']}类 {info['direction']}</span>
    <span class="info-tag">ER={info['er_20']:.3f}</span>
    <span class="info-tag {'win' if pnl_f > 0 else 'loss'}">Fixed {pnl_f:+.2f}%</span>
    <span class="info-tag {'win' if pnl_s2 > 0 else 'loss'}">S2 {pnl_s2:+.2f}%</span>
</div>
<img src="data:image/png;base64,{img_b64}">
</div>""")

    html.append('</body></html>')
    return '\n'.join(html)


# ============================================================
# 主流程
# ============================================================

def main():
    random.seed(77)
    np.random.seed(42)

    print("=" * 100)
    print("  趋势判断方法 × 出场策略 全因子验证 | 32品种 | 120天")
    print("=" * 100)

    # Step 1: 加载数据
    print("\n  [1/6] 加载数据...")
    sym_data = load_all_symbols()
    print(f"  加载 {len(sym_data)} 品种")

    # Step 2: 检测信号 + 模拟出场
    print("\n  [2/6] 检测信号 + 模拟出场...")
    all_trades = []

    for sym_key, df in sorted(sym_data.items()):
        tick_size = get_tick_size(sym_key)
        ck = sym_key_to_config_key(sym_key)
        sym_name = get_sym_name(sym_key)
        is_high_vol = ck in HIGH_VOL_KEYS

        signals = detect_all_signals(df)

        for sig in signals:
            exits = simulate_all_exits(df, sig, tick_size)
            trade = {
                'symbol': sym_key,
                'sym_name': sym_name,
                'group': '高波动' if is_high_vol else '常规',
                'type': sig['type'],
                'direction': sig['direction'],
                'idx': sig['idx'],
                'entry_price': sig['entry_price'],
                'pullback_extreme': sig['pullback_extreme'],
                'pullback_bars': sig['pullback_bars'],
                'er_20': sig['er_20'],
                'time': sig['time'],
            }
            # 趋势方法接纳标记
            for m_name, _ in TREND_METHODS:
                trade[f'admitted_{m_name}'] = sig[f'admitted_{m_name}']
            trade.update(exits)
            all_trades.append(trade)

        n_sigs = len(signals)
        if n_sigs > 0:
            print(f"    {sym_name:>6}: {n_sigs:>4}笔")

    df_all = pd.DataFrame(all_trades)
    print(f"\n  总信号: {len(df_all)}笔 (A={len(df_all[df_all['type']=='A'])}, "
          f"B={len(df_all[df_all['type']=='B'])}, C={len(df_all[df_all['type']=='C'])})")

    # Step 3: 计算矩阵统计
    print("\n  [3/6] 计算统计矩阵...")
    matrix_stats = []  # [(method, exit, sig_type, stats_dict, er_filtered)]

    for m_name, _ in TREND_METHODS:
        admitted = df_all[df_all[f'admitted_{m_name}'] == True]

        for er_filtered in [False, True]:
            if er_filtered:
                subset = admitted[admitted['er_20'] >= 0.5]
            else:
                subset = admitted

            for sig_label in ['全部', 'A', 'B', 'C']:
                if sig_label == '全部':
                    sig_sub = subset
                else:
                    sig_sub = subset[subset['type'] == sig_label]

                for exit_name in EXIT_NAMES:
                    stats = calc_group_stats(sig_sub, exit_name)
                    matrix_stats.append((m_name, exit_name, sig_label, stats, er_filtered))

    # 打印主要结果
    print(f"\n  {'方法':<12} | {'出场':<6} | {'N':>5} | {'EV':>6} | {'胜率':>5} | {'盈亏比':>6}")
    print(f"  {'-'*60}")
    for m_name, exit_name, sig_label, stats, er_filtered in matrix_stats:
        if sig_label == '全部' and not er_filtered:
            print(f"  {m_name:<12} | {exit_name:<6} | {stats['N']:>5} | {stats['EV']:>+6.2f} | "
                  f"{stats['wr']:>5.1f}% | {stats['pr']:>5.2f}")

    # Step 4: 被过滤信号分析
    print("\n  [4/6] 被过滤信号分析...")
    filtered_stats = []

    # 被MA20>MA120过滤但MA10>MA20接纳的信号
    mask_filtered = (df_all['admitted_MA10>MA20'] == True) & (df_all['admitted_MA20>MA120'] == False)
    df_filtered = df_all[mask_filtered]
    # 被MA20>MA120接纳的信号
    df_admitted = df_all[df_all['admitted_MA20>MA120'] == True]

    for label, subset in [('被过滤信号(全部)', df_filtered),
                           ('被接纳信号(全部)', df_admitted),
                           ('被过滤-A类', df_filtered[df_filtered['type'] == 'A']),
                           ('被过滤-B类', df_filtered[df_filtered['type'] == 'B']),
                           ('被过滤-C类', df_filtered[df_filtered['type'] == 'C'])]:
        row_stats = {}
        for exit_name in EXIT_NAMES:
            row_stats[exit_name] = calc_group_stats(subset, exit_name)
        filtered_stats.append((label, row_stats))

    for label, stats_row in filtered_stats:
        ev_fixed = stats_row['Fixed']['EV']
        ev_s2 = stats_row['S2']['EV']
        n = stats_row['Fixed']['N']
        print(f"    {label}: N={n:>5} Fixed_EV={ev_fixed:+.2f} S2_EV={ev_s2:+.2f}")

    # Step 5: Bootstrap CI
    print("\n  [5/6] Bootstrap置信区间...")
    bootstrap_results = []

    df_m20 = df_all[df_all['admitted_MA20>MA120'] == True]
    df_m10 = df_all[df_all['admitted_MA10>MA20'] == True]

    for exit_name in EXIT_NAMES:
        pnl_col = f'{exit_name}_pnl'
        for sig_label in ['全部', 'C']:
            if sig_label == '全部':
                pnls_20 = df_m20[pnl_col].values
                pnls_10 = df_m10[pnl_col].values
            else:
                pnls_20 = df_m20[df_m20['type'] == sig_label][pnl_col].values
                pnls_10 = df_m10[df_m10['type'] == sig_label][pnl_col].values

            if len(pnls_20) < 30 or len(pnls_10) < 30:
                continue

            mean_d, lo, hi, sig_flag = bootstrap_ev_diff(pnls_20, pnls_10)
            bootstrap_results.append((exit_name, sig_label, mean_d, lo, hi, sig_flag))
            sig_mark = '***' if sig_flag else ''
            print(f"    {exit_name} {sig_label}: diff={mean_d:+.4f} CI=[{lo:+.4f}, {hi:+.4f}] {sig_mark}")

    # Step 6: 鲁棒性检验
    print("\n  [6/6] 鲁棒性检验...")

    # 6a: 时间窗口
    window_stats = []
    dates = df_all['time'].sort_values()
    if len(dates) > 0:
        date_min = dates.iloc[0]
        date_max = dates.iloc[-1]
        window_size = (date_max - date_min) / 4

        for m_name, _ in TREND_METHODS:
            admitted = df_all[df_all[f'admitted_{m_name}'] == True]
            for w in range(4):
                w_start = date_min + window_size * w
                w_end = date_min + window_size * (w + 1)
                w_sub = admitted[(admitted['time'] >= w_start) & (admitted['time'] < w_end)]
                for exit_name in EXIT_NAMES:
                    stats = calc_group_stats(w_sub, exit_name)
                    window_stats.append((m_name, exit_name, w, stats))

    # 6b: 品种组
    group_stats_list = []
    for m_name, _ in TREND_METHODS:
        admitted = df_all[df_all[f'admitted_{m_name}'] == True]
        for grp in ['高波动', '常规']:
            g_sub = admitted[admitted['group'] == grp]
            for exit_name in EXIT_NAMES:
                stats = calc_group_stats(g_sub, exit_name)
                group_stats_list.append((m_name, exit_name, grp, stats))

    # 6c: LOSO
    loso_stats = []
    all_syms = df_all['symbol'].unique()
    for exit_name in ['Fixed', 'S2', 'S4']:
        for m_name in ['MA10>MA20', 'MA20>MA120']:
            admitted = df_all[df_all[f'admitted_{m_name}'] == True]
            pnl_col = f'{exit_name}_pnl'
            full_ev = calc_ev(admitted[pnl_col].values)[0]

            evs = []
            for sym in all_syms:
                sub = admitted[admitted['symbol'] != sym]
                ev = calc_ev(sub[pnl_col].values)[0]
                evs.append(ev)

            ev_min, ev_max = min(evs), max(evs)
            # 检查是否有翻转：MA20>MA120是否在某个LOSO下低于MA10>MA20的full_ev
            other_m = 'MA10>MA20' if m_name == 'MA20>MA120' else 'MA20>MA120'
            other_admitted = df_all[df_all[f'admitted_{other_m}'] == True]
            other_ev = calc_ev(other_admitted[pnl_col].values)[0]
            flipped = (m_name == 'MA20>MA120' and ev_min < other_ev) or \
                      (m_name == 'MA10>MA20' and ev_max > full_ev)

            loso_stats.append((m_name, exit_name, ev_min, ev_max, flipped))

    print(f"    LOSO完成: {len(loso_stats)}组")

    # K线图：被过滤信号随机抽10笔
    print("\n  生成K线验证图...")
    chart_images = []
    if len(df_filtered) > 0:
        sample_indices = df_filtered.sample(min(10, len(df_filtered)), random_state=77).index
        for idx in sample_indices:
            trade = df_all.loc[idx]
            sym_key = trade['symbol']
            df_sym = sym_data.get(sym_key)
            if df_sym is None:
                continue

            sig_dict = {
                'idx': int(trade['idx']),
                'type': trade['type'],
                'direction': trade['direction'],
                'entry_price': trade['entry_price'],
                'pullback_extreme': trade['pullback_extreme'],
                'er_20': trade['er_20'] if not pd.isna(trade['er_20']) else 0,
                'time': trade['time'],
            }
            exit_fixed = {'Fixed_pnl': trade['Fixed_pnl'], 'Fixed_bars': trade['Fixed_bars']}
            exit_s2 = {'S2_pnl': trade['S2_pnl'], 'S2_bars': trade['S2_bars']}

            try:
                img = draw_trade_chart(df_sym, sig_dict, exit_fixed, exit_s2, trade['sym_name'])
                info = {
                    'sym_name': trade['sym_name'], 'type': trade['type'],
                    'direction': trade['direction'], 'er_20': sig_dict['er_20'],
                    'Fixed_pnl': trade['Fixed_pnl'], 'S2_pnl': trade['S2_pnl'],
                }
                chart_images.append((info, img))
            except Exception as e:
                print(f"    图表错误 {sym_key}: {e}")

    # 生成HTML
    print("\n  生成HTML报告...")
    html = generate_html(df_all, matrix_stats, filtered_stats, bootstrap_results,
                         window_stats, group_stats_list, loso_stats, chart_images)

    html_path = os.path.join(OUTPUT_DIR, 'trend_exit_matrix.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  HTML: {html_path}")

    # 导出CSV
    csv_path = os.path.join(OUTPUT_DIR, 'trend_exit_matrix_trades.csv')
    export_cols = ['symbol', 'sym_name', 'group', 'type', 'direction', 'time',
                   'entry_price', 'pullback_extreme', 'pullback_bars', 'er_20',
                   'admitted_MA10>MA20', 'admitted_三线排列', 'admitted_MA20>MA120',
                   'Fixed_pnl', 'Fixed_bars', 'S1_pnl', 'S1_bars',
                   'S2_pnl', 'S2_bars', 'S3_pnl', 'S3_bars', 'S4_pnl', 'S4_bars',
                   'mfe', 'mae']
    df_all[[c for c in export_cols if c in df_all.columns]].to_csv(
        csv_path, index=False, encoding='utf-8-sig')
    print(f"  CSV: {csv_path}")

    # 汇总CSV
    summary_rows = []
    for m_name, exit_name, sig_label, stats, er_filtered in matrix_stats:
        summary_rows.append({
            'method': m_name, 'exit': exit_name, 'signal': sig_label,
            'er_filter': 'ER>=0.5' if er_filtered else 'all',
            **stats,
        })
    summary_path = os.path.join(OUTPUT_DIR, 'trend_exit_matrix_summary.csv')
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"  Summary: {summary_path}")

    # 回归验证
    print("\n  回归验证（固定出场 vs 已知结果）:")
    for m_name, exit_name, sig_label, stats, er_filtered in matrix_stats:
        if exit_name == 'Fixed' and sig_label == '全部' and not er_filtered:
            print(f"    {m_name}: EV={stats['EV']:+.2f} N={stats['N']}")

    print("\n  完成！")


if __name__ == '__main__':
    main()
