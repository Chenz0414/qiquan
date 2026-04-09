# -*- coding: utf-8 -*-
"""
Black-76 期货期权定价模型
=========================
期货期权标准定价，不需要处理股息率和借贷成本。

用法:
    from option_pricing import black76_price, black76_greeks, find_strike_by_delta

    price = black76_price(F=7000, K=7000, T=30/245, r=0.02, sigma=0.25, cp='call')
    greeks = black76_greeks(F=7000, K=7000, T=30/245, r=0.02, sigma=0.25, cp='call')
    K = find_strike_by_delta(F=7000, T=30/245, r=0.02, sigma=0.25, target_delta=0.5, cp='call')
"""

import numpy as np
from scipy.stats import norm

R_DEFAULT = 0.02  # 无风险利率


def black76_price(F, K, T, r, sigma, cp='call'):
    """
    Black-76期权理论价格。

    参数:
      F: 期货价格
      K: 行权价
      T: 到期时间（年），如 30/245 = 约30个交易日
      r: 无风险利率
      sigma: 波动率（年化）
      cp: 'call' 或 'put'
    返回:
      期权理论价格
    """
    if T <= 0 or sigma <= 0:
        # 到期时
        if cp == 'call':
            return max(F - K, 0)
        else:
            return max(K - F, 0)

    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    discount = np.exp(-r * T)

    if cp == 'call':
        return discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def black76_greeks(F, K, T, r, sigma, cp='call'):
    """
    计算Greeks。

    返回:
      dict: {delta, gamma, theta, vega, price}
      theta: 每交易日的时间衰减（负值）
      vega: 波动率变动1%（绝对值）时的价格变化
    """
    if T <= 0 or sigma <= 0:
        price = max(F - K, 0) if cp == 'call' else max(K - F, 0)
        delta = 1.0 if (cp == 'call' and F > K) else (-1.0 if (cp == 'put' and F < K) else 0.0)
        return {'delta': delta, 'gamma': 0, 'theta': 0, 'vega': 0, 'price': price}

    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    discount = np.exp(-r * T)
    n_d1 = norm.pdf(d1)

    if cp == 'call':
        price = discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
        delta = discount * norm.cdf(d1)
    else:
        price = discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
        delta = -discount * norm.cdf(-d1)

    gamma = discount * n_d1 / (F * sigma * np.sqrt(T))

    # theta: 每交易日（除以245）
    theta_annual = (-F * discount * n_d1 * sigma / (2 * np.sqrt(T))
                    - r * price)
    theta_daily = theta_annual / 245

    # vega: 波动率变动0.01（1个百分点）时的价格变化
    vega = F * discount * n_d1 * np.sqrt(T) * 0.01

    return {
        'delta': round(delta, 4),
        'gamma': round(gamma, 6),
        'theta': round(theta_daily, 4),
        'vega': round(vega, 4),
        'price': round(price, 4),
    }


def find_strike_by_delta(F, T, r, sigma, target_delta, cp='call'):
    """
    根据目标Delta反算行权价（二分法）。

    参数:
      target_delta: 目标Delta绝对值，如0.5=平值, 0.3=虚值
      cp: 'call' 或 'put'
    返回:
      行权价K
    """
    if T <= 0 or sigma <= 0:
        return F  # fallback

    # 搜索范围：期货价格的50%~200%
    lo, hi = F * 0.5, F * 2.0

    for _ in range(100):
        mid = (lo + hi) / 2
        g = black76_greeks(F, mid, T, r, sigma, cp)
        d = abs(g['delta'])

        if abs(d - target_delta) < 0.001:
            return round(mid, 2)

        if cp == 'call':
            # call delta随K增大而减小
            if d > target_delta:
                lo = mid
            else:
                hi = mid
        else:
            # put |delta|随K减小而减小
            if d > target_delta:
                hi = mid
            else:
                lo = mid

    return round((lo + hi) / 2, 2)


def option_pnl(F_entry, F_exit, T_entry, bars_held, sigma, target_delta,
               cp='call', r=R_DEFAULT, slippage_pct=5.0):
    """
    计算单笔期权交易的收益率。

    参数:
      F_entry: 入场期货价格
      F_exit: 出场期货价格
      T_entry: 入场时到期时间（年）
      bars_held: 持仓K线根数（10分钟）
      sigma: 波动率（年化HV）
      target_delta: 目标Delta绝对值
      cp: 'call' 或 'put'
      r: 无风险利率
      slippage_pct: 滑点百分比（买卖价差，按理论价扣除）
    返回:
      dict: {roi_pct, entry_premium, exit_premium, K, delta_entry, theta_daily, ...}
    """
    # 反算行权价
    K = find_strike_by_delta(F_entry, T_entry, r, sigma, target_delta, cp)

    # 入场期权价
    entry_greeks = black76_greeks(F_entry, K, T_entry, r, sigma, cp)
    entry_premium = entry_greeks['price']

    if entry_premium <= 0:
        return None  # 无效定价

    # 出场时的剩余到期时间
    # 10分钟一根，57根/天，245天/年
    time_elapsed = bars_held / (57 * 245)
    T_exit = max(T_entry - time_elapsed, 0)

    # 出场期权价
    exit_premium = black76_price(F_exit, K, T_exit, r, sigma, cp)

    # 扣除滑点（买入时多付，卖出时少收）
    entry_cost = entry_premium * (1 + slippage_pct / 100)
    exit_revenue = exit_premium * (1 - slippage_pct / 100)

    # 收益率
    roi = (exit_revenue - entry_cost) / entry_cost * 100

    return {
        'roi_pct': round(roi, 2),
        'entry_premium': round(entry_premium, 4),
        'exit_premium': round(exit_premium, 4),
        'entry_cost': round(entry_cost, 4),
        'K': K,
        'delta_entry': entry_greeks['delta'],
        'gamma_entry': entry_greeks['gamma'],
        'theta_daily': entry_greeks['theta'],
        'vega_entry': entry_greeks['vega'],
        'T_entry_days': round(T_entry * 245, 1),
        'T_exit_days': round(T_exit * 245, 1),
        'bars_held': bars_held,
    }
