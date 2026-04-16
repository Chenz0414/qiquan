# -*- coding: utf-8 -*-
"""
板块集中度暴露检测
==================
聚合"当前持仓 + 候选池"按板块、方向分桶，当某板块同向 ≥ N 品种时产出警告。

用途：§8 (plan) — 避免"黑色系同时触发 3 个多头 ⇒ 其实是一笔交易"的集中风险。
"""

from typing import Dict, Iterable, List
from contract_parser import get_sector


DEFAULT_WARN_THRESHOLD = 3


def _empty_slot() -> dict:
    return {
        "long_count": 0, "short_count": 0,
        "long_symbols": [], "short_symbols": [],
        "long_candidates": [], "short_candidates": [],
    }


def compute_sector_exposure(open_signals: Iterable[dict],
                            candidate_pool) -> Dict[str, dict]:
    """
    合并持仓 + 候选池按板块聚合。

    参数:
      open_signals   — signal_db.get_open_signals() 返回的 list[dict]
      candidate_pool — dashboard.candidate_pool（dict）或其 .values()（iterable）

    返回:
      {sector: {long_count, short_count,
                long_symbols, short_symbols,
                long_candidates, short_candidates}}
    """
    out: Dict[str, dict] = {}

    for s in open_signals or []:
        sec = s.get("sector") or get_sector(s.get("sym_key") or "")
        if not sec:
            continue
        slot = out.setdefault(sec, _empty_slot())
        direction = s.get("direction")
        sym = s.get("sym_key")
        if direction == "long" and sym:
            slot["long_count"] += 1
            if sym not in slot["long_symbols"]:
                slot["long_symbols"].append(sym)
        elif direction == "short" and sym:
            slot["short_count"] += 1
            if sym not in slot["short_symbols"]:
                slot["short_symbols"].append(sym)

    cands = candidate_pool.values() if isinstance(candidate_pool, dict) \
        else (candidate_pool or [])
    for c in cands:
        sec = c.get("sector") or get_sector(c.get("sym_key") or "")
        if not sec:
            continue
        slot = out.setdefault(sec, _empty_slot())
        bias = c.get("direction_bias")
        sym = c.get("sym_key")
        if bias == "long" and sym:
            if sym not in slot["long_candidates"]:
                slot["long_candidates"].append(sym)
        elif bias == "short" and sym:
            if sym not in slot["short_candidates"]:
                slot["short_candidates"].append(sym)

    return out


def detect_sector_warnings(exposure: Dict[str, dict],
                           min_same_dir: int = DEFAULT_WARN_THRESHOLD
                           ) -> List[dict]:
    """
    把集中度 >= 阈值的板块打成警告清单。
    count 同时统计已持仓 + 候选。
    """
    out: List[dict] = []
    for sec, slot in exposure.items():
        long_total = slot["long_count"] + len(slot["long_candidates"])
        short_total = slot["short_count"] + len(slot["short_candidates"])
        if long_total >= min_same_dir:
            out.append({
                "sector": sec, "direction": "long",
                "count": long_total,
                "position_symbols": list(slot["long_symbols"]),
                "candidate_symbols": list(slot["long_candidates"]),
            })
        if short_total >= min_same_dir:
            out.append({
                "sector": sec, "direction": "short",
                "count": short_total,
                "position_symbols": list(slot["short_symbols"]),
                "candidate_symbols": list(slot["short_candidates"]),
            })
    return out
