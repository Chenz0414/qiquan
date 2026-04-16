# -*- coding: utf-8 -*-
"""
合约代码解析
============
统一把用户/系统各种格式的合约代码解析成结构化字段：

  - "rb2510"                 → future
  - "SHFE.rb2510"            → future（带交易所）
  - "SHFE.rb"                → future 主连（无月份）
  - "SHFE.rb2510-C-3500"     → option call
  - "rb2510-P-3400"          → option put

同时提供：
  - SYMBOL_CONFIGS 反查得到 sym_key / exchange
  - 品种 → 板块映射（贵金属/有色/黑色/能化/农产品/其他）
"""

import re
from typing import Optional, Dict

from signal_core import SYMBOL_CONFIGS


# ============================================================
#  品种 → 板块映射
# ============================================================

SECTOR_MAP: Dict[str, str] = {
    # 贵金属
    "au": "贵金属", "ag": "贵金属",
    # 有色
    "cu": "有色", "al": "有色", "zn": "有色", "ni": "有色", "bc": "有色",
    # 黑色
    "rb": "黑色", "hc": "黑色", "i": "黑色", "jm": "黑色", "j": "黑色",
    # 能化
    "bu": "能化", "ru": "能化", "sp": "能化", "fu": "能化", "lu": "能化",
    "eg": "能化", "pp": "能化", "l": "能化", "pg": "能化", "eb": "能化",
    "TA": "能化", "MA": "能化", "SA": "能化", "FG": "能化",
    "sc": "能化",
    # 农产品
    "m": "农产品", "y": "农产品", "p": "农产品",
    "SR": "农产品", "CF": "农产品", "RM": "农产品",
    "lh": "农产品",
    # 其他
    "lc": "其他", "v": "其他",
}


# 品种 → 交易所（从 SYMBOL_CONFIGS 反查）
def _build_symbol_to_exchange() -> Dict[str, str]:
    m = {}
    for sym_key in SYMBOL_CONFIGS:
        # 'SHFE.rb' → underlying='rb', exchange='SHFE'
        if "." in sym_key:
            ex, under = sym_key.split(".", 1)
            m[under] = ex
    return m


SYMBOL_TO_EXCHANGE: Dict[str, str] = _build_symbol_to_exchange()


# ============================================================
#  解析函数
# ============================================================

# 期权后缀：-C-3500 / -P-3400 / C-3500 / P3400（有些交易所老格式）
_OPTION_SUFFIX = re.compile(r"[-_]?(C|P)[-_]?(\d+(?:\.\d+)?)$", re.IGNORECASE)

# 合约前缀："rb" / "TA" / "MA" （字母部分）
_UNDER_PREFIX = re.compile(r"^([A-Za-z]+)")


def parse_contract(code: str) -> Optional[dict]:
    """
    解析合约代码，返回 dict 或 None。

    返回结构：
      - future: {kind, code, sym_key, exchange, underlying, month}
      - option: {kind, code, underlying_sym_key, underlying_contract,
                 strike, cp, exchange, underlying}
    """
    if not code:
        return None

    raw = code.strip()

    # 1) 剥离交易所前缀
    exchange = None
    body = raw
    if "." in body:
        a, b = body.split(".", 1)
        if a.isupper() and len(a) <= 6:
            exchange = a
            body = b

    # 2) 判断是否期权（末尾 -C-xxx / -P-xxx）
    opt_m = _OPTION_SUFFIX.search(body)
    is_option = opt_m is not None
    under_body = body[:opt_m.start()] if is_option else body

    # 3) 拆出标的品种（字母）+ 月份（数字）
    u_m = _UNDER_PREFIX.match(under_body)
    if not u_m:
        return None
    underlying = u_m.group(1)
    month = under_body[u_m.end():]

    # 4) 回填交易所
    if exchange is None:
        # 规则：CZCE 品种名大写（TA/MA/SA/FG/SR/CF/RM）
        key = underlying.upper() if underlying.upper() in (
            "TA", "MA", "SA", "FG", "SR", "CF", "RM"
        ) else underlying.lower()
        exchange = SYMBOL_TO_EXCHANGE.get(key)

    # 5) 归一化 sym_key
    under_key_local = (underlying.upper() if underlying.upper() in (
        "TA", "MA", "SA", "FG", "SR", "CF", "RM") else underlying.lower())
    sym_key = f"{exchange}.{under_key_local}" if exchange else None

    if is_option:
        cp = opt_m.group(1).upper()
        strike = float(opt_m.group(2))
        # 标的合约："rb2510" → "SHFE.rb2510"
        under_contract = (f"{exchange}.{under_key_local}{month}"
                          if exchange else f"{under_key_local}{month}")
        return {
            "kind": "option",
            "code": raw,
            "exchange": exchange,
            "underlying": under_key_local,
            "underlying_sym_key": sym_key,
            "underlying_contract": under_contract,
            "month": month or None,
            "strike": strike,
            "cp": cp,
        }

    return {
        "kind": "future",
        "code": raw,
        "exchange": exchange,
        "underlying": under_key_local,
        "sym_key": sym_key,
        "month": month or None,
    }


def get_sector(underlying_or_sym_key: str) -> Optional[str]:
    """按 underlying 或 sym_key 返回板块名"""
    if not underlying_or_sym_key:
        return None
    key = underlying_or_sym_key.strip()
    if "." in key:
        key = key.split(".", 1)[1]
    # CZCE 大写保持，其他小写
    key_norm = (key.upper() if key.upper() in (
        "TA", "MA", "SA", "FG", "SR", "CF", "RM") else key.lower())
    return SECTOR_MAP.get(key_norm)


def get_sym_meta(sym_key: str) -> dict:
    """返回品种元信息 {tick_size, multiplier, name, sector}"""
    meta = dict(SYMBOL_CONFIGS.get(sym_key, {}))
    meta["sector"] = get_sector(sym_key)
    return meta
