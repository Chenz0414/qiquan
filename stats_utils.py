# -*- coding: utf-8 -*-
"""
统计工具模块
============
替代各 test_*.py 中重复的 calc_ev() 逻辑。

用法:
    from stats_utils import calc_ev, calc_ev_df, group_ev
"""

import numpy as np
import pandas as pd


def calc_ev(pnls):
    """
    计算策略期望值（归一化EV）。

    EV = 胜率 × 盈亏比 - 败率
    含义：以平均亏损为1个单位时，每笔交易的期望收益单位数。
    EV > 0 = 正期望策略。

    参数:
      pnls: list[float] 或 np.array，每笔交易的盈亏百分比
    返回:
      dict: {N, EV, wr, pr, avg_w, avg_l, sum_pnl, avg_pnl}
    """
    if not pnls or len(pnls) == 0:
        return {'N': 0, 'EV': 0, 'wr': 0, 'pr': 0,
                'avg_w': 0, 'avg_l': 0, 'sum_pnl': 0, 'avg_pnl': 0}

    pnls = list(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    n = len(pnls)
    wr = len(wins) / n
    avg_w = np.mean(wins) if wins else 0
    avg_l = abs(np.mean(losses)) if losses else 0.001
    pr = avg_w / avg_l if avg_l > 0 else 0
    ev = wr * pr - (1 - wr)

    return {
        'N': n,
        'EV': round(ev, 2),
        'wr': round(wr * 100, 1),
        'pr': round(pr, 2),
        'avg_w': round(avg_w, 4),
        'avg_l': round(avg_l, 4),
        'sum_pnl': round(sum(pnls), 2),
        'avg_pnl': round(np.mean(pnls), 4),
    }


def calc_ev_df(df, pnl_col='s2_pnl', reason_col=None, exclude_backtest_end=True):
    """
    从 DataFrame 计算EV，自动过滤 backtest_end。

    参数:
      df: 交易记录 DataFrame
      pnl_col: PnL列名
      reason_col: exit_reason列名（如 's2_reason'）。
                  None时自动推断：pnl_col的前缀+'_reason'
      exclude_backtest_end: 是否排除 backtest_end 交易
    """
    if exclude_backtest_end:
        if reason_col is None:
            # s2_pnl -> s2_reason; exit_s2_pnl -> exit_s2_reason
            prefix = pnl_col.rsplit('_pnl', 1)[0]
            reason_col = f'{prefix}_reason'
        if reason_col in df.columns:
            df = df[df[reason_col] != 'backtest_end']
    return calc_ev(df[pnl_col].tolist())


def group_ev(df, group_col, pnl_col='s2_pnl', reason_col=None,
             exclude_backtest_end=True, sort_by='EV', ascending=False):
    """
    按某列分组计算EV。

    用法:
      results = group_ev(df_all, 'symbol', pnl_col='s2_pnl')
      # returns: [{'group': '白银', 'N': 45, 'EV': 1.23, ...}, ...]
    """
    rows = []
    for name, sub in df.groupby(group_col, sort=False):
        st = calc_ev_df(sub, pnl_col, reason_col, exclude_backtest_end)
        st['group'] = name
        rows.append(st)
    if sort_by and rows:
        rows.sort(key=lambda x: x.get(sort_by, 0), reverse=not ascending)
    return rows


def ev_line(label, pnls):
    """生成一行统计摘要文本"""
    st = calc_ev(pnls)
    return (f"{label}: N={st['N']} EV={st['EV']} "
            f"WR={st['wr']}% PR={st['pr']} "
            f"sum={st['sum_pnl']} avg={st['avg_pnl']}")


def filter_by_range(df, **kwargs):
    """
    按条件过滤 DataFrame。

    用法:
      filtered = filter_by_range(df, er_20_min=0.5, er_20_max=0.7, type='B')

    规则:
      field=value → 精确匹配
      field_min=value → >= 过滤
      field_max=value → < 过滤
      field_in=[values] → isin 过滤
    """
    mask = pd.Series(True, index=df.index)
    for key, val in kwargs.items():
        if key.endswith('_min'):
            col = key[:-4]
            mask &= df[col] >= val
        elif key.endswith('_max'):
            col = key[:-4]
            mask &= df[col] < val
        elif key.endswith('_in'):
            col = key[:-3]
            mask &= df[col].isin(val)
        else:
            mask &= df[key] == val
    return df[mask]
