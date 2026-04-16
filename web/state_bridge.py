# -*- coding: utf-8 -*-
"""
线程安全的共享状态
==================
MonitorEngine(主线程) 写入 → FastAPI(后台线程) 读取 + SSE推送
"""

import asyncio
import threading
import json
import logging
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


class DashboardState:
    def __init__(self):
        self._lock = threading.Lock()
        self._subscribers: list[asyncio.Queue] = []

        # 引擎状态（引擎写入，Web读取）
        self.engine_status: str = "starting"
        self.connection_time: str = None
        self.last_bar_time: dict = {}        # {sym_key: iso_time}
        self.active_positions: dict = {}     # {sym_key: position_data}
        self.detector_trends: dict = {}      # {sym_key: trend_dir}
        self.last_error: str = None

        # 事件缓冲（最近200条）
        self._events = deque(maxlen=200)

        # ================================================================
        #  Phase A3 新增：决策支持字段
        # ================================================================

        # 拒绝流水（observer emit → 最近500条）
        self.reject_stream: deque = deque(maxlen=500)

        # 每品种状态快照（合约查询/候选池的数据底）
        #   sym_key -> {pullback_stage, pullback_bar_count, candidate_scenario,
        #               last_ema_snapshot, last_price, last_er20, last_atr, ...}
        self.symbol_states: dict = {}

        # 今日规则计数（入场/拒绝，每分钟增量推送）
        #   rule_key -> {hits, open, wins, losses, rejects}
        self.rule_today_counters: dict = {}

        # 候选信号池（distance_ticks 排序，2Hz 增量推送）
        #   sym_key -> candidate dict（含 kind/trigger_price/distance_ticks/rule_120d/option_alt）
        self.candidate_pool: dict = {}

        # 市场热力图（32 品种快照，每 bar 刷新一次）
        self.symbol_heatmap: dict = {}

        # 持仓实时 MFE/MAE/R（1Hz 聚合推送）
        self.portfolio_live: dict = {}

        # 板块集中度暴露
        #   {sector: {long_count, short_count, long_symbols, short_symbols}}
        self.sector_exposure: dict = {}

        # 期权报价快照（1Hz）与期权链（每日）
        self.option_quotes_snapshot: dict = {}
        self.option_chain: dict = {}

        # 手动干预状态（持久化到 monitor_settings.json 侧）
        self.paused_rules: dict = {}         # rule_key -> until_iso
        self.silenced_symbols: dict = {}     # sym_key -> until_iso

        # 规则漂移告警（当前最严重级别）
        self.drift_state: dict = {}          # rule_key -> {severity, z_score, ...}

    # ================================================================
    #  引擎调用（主线程，加锁写入）
    # ================================================================
    def update_position(self, sym_key: str, data: dict):
        with self._lock:
            self.active_positions[sym_key] = data

    def remove_position(self, sym_key: str):
        with self._lock:
            self.active_positions.pop(sym_key, None)

    def update_engine_status(self, status: str):
        with self._lock:
            self.engine_status = status
            if status == "running" and not self.connection_time:
                self.connection_time = datetime.now().isoformat()

    def update_bar_time(self, sym_key: str, time_str: str):
        with self._lock:
            self.last_bar_time[sym_key] = time_str

    def update_trends(self, trends: dict):
        with self._lock:
            self.detector_trends.update(trends)

    def push_event(self, event_type: str, data: dict):
        """推送事件给所有 SSE 订阅者"""
        event = {
            "type": event_type,
            "data": data,
            "time": datetime.now().isoformat(),
        }

        with self._lock:
            self._events.append(event)
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass  # 慢客户端丢弃旧事件

    # ================================================================
    #  Phase A3 新增写入器（引擎主线程调用）
    # ================================================================

    def push_reject(self, sym_key: str, stage: str, reason: str,
                    context: dict = None):
        """拒绝流水入队 + SSE 广播"""
        entry = {
            "sym_key": sym_key, "stage": stage, "reason": reason,
            "context": context or {},
            "time": datetime.now().isoformat(),
        }
        with self._lock:
            self.reject_stream.append(entry)
        # 降级：SSE 事件走推送通道
        self.push_event("reject", entry)

    def update_symbol_state(self, sym_key: str, patch: dict):
        with self._lock:
            cur = self.symbol_states.setdefault(sym_key, {})
            cur.update(patch)

    def bump_rule_counter(self, rule_key: str, field: str, delta: int = 1):
        with self._lock:
            counters = self.rule_today_counters.setdefault(
                rule_key, {"hits": 0, "open": 0, "wins": 0,
                           "losses": 0, "rejects": 0})
            counters[field] = counters.get(field, 0) + delta

    def set_candidate(self, sym_key: str, candidate: dict = None):
        """candidate=None 则移除"""
        with self._lock:
            if candidate is None:
                self.candidate_pool.pop(sym_key, None)
            else:
                self.candidate_pool[sym_key] = candidate

    def update_heatmap_cell(self, sym_key: str, cell: dict):
        with self._lock:
            self.symbol_heatmap[sym_key] = cell

    def update_portfolio_live(self, signal_id, live: dict):
        with self._lock:
            self.portfolio_live[str(signal_id)] = live

    def remove_portfolio_live(self, signal_id):
        with self._lock:
            self.portfolio_live.pop(str(signal_id), None)

    def update_sector_exposure(self, exposure: dict):
        with self._lock:
            self.sector_exposure = dict(exposure)

    def update_option_quotes(self, quotes_patch: dict):
        with self._lock:
            self.option_quotes_snapshot.update(quotes_patch)

    def set_option_chain(self, underlying: str, chain: dict):
        with self._lock:
            self.option_chain[underlying] = chain

    def set_pause_rule(self, rule_key: str, until_iso: str = None):
        with self._lock:
            if until_iso is None:
                self.paused_rules.pop(rule_key, None)
            else:
                self.paused_rules[rule_key] = until_iso

    def set_silence_symbol(self, sym_key: str, until_iso: str = None):
        with self._lock:
            if until_iso is None:
                self.silenced_symbols.pop(sym_key, None)
            else:
                self.silenced_symbols[sym_key] = until_iso

    def update_drift_state(self, drift_patch: dict):
        """drift_patch: {rule_key -> {severity, z_score, ...}}"""
        with self._lock:
            self.drift_state.update(drift_patch)

    # ================================================================
    #  Web调用（FastAPI线程）
    # ================================================================
    def snapshot(self) -> dict:
        """返回全量快照"""
        with self._lock:
            return {
                "engine_status": self.engine_status,
                "connection_time": self.connection_time,
                "active_positions": dict(self.active_positions),
                "last_bar_time": dict(self.last_bar_time),
                "detector_trends": dict(self.detector_trends),
                "last_error": self.last_error,
                # --- Phase A3 ---
                "symbol_states": dict(self.symbol_states),
                "rule_today_counters": dict(self.rule_today_counters),
                "candidate_pool": dict(self.candidate_pool),
                "symbol_heatmap": dict(self.symbol_heatmap),
                "portfolio_live": dict(self.portfolio_live),
                "sector_exposure": dict(self.sector_exposure),
                "option_quotes_snapshot": dict(self.option_quotes_snapshot),
                "paused_rules": dict(self.paused_rules),
                "silenced_symbols": dict(self.silenced_symbols),
                "drift_state": dict(self.drift_state),
                "reject_stream": list(self.reject_stream)[-50:],  # 最近50条
            }

    def get_candidates(self, min_distance_ticks: int = None,
                       kind: str = None) -> list:
        with self._lock:
            items = list(self.candidate_pool.values())
        if min_distance_ticks is not None:
            items = [c for c in items
                     if c.get("distance_ticks") is not None
                     and c["distance_ticks"] <= min_distance_ticks]
        if kind and kind != "all":
            items = [c for c in items if c.get("kind") == kind]
        items.sort(key=lambda c: c.get("distance_ticks")
                   if c.get("distance_ticks") is not None else 9999)
        return items

    def get_rejects(self, limit: int = 200) -> list:
        with self._lock:
            return list(self.reject_stream)[-limit:]

    def get_heatmap(self) -> list:
        with self._lock:
            return list(self.symbol_heatmap.values())

    def get_symbol_state(self, sym_key: str) -> dict:
        with self._lock:
            return dict(self.symbol_states.get(sym_key, {}))

    def get_portfolio_live(self) -> dict:
        with self._lock:
            return dict(self.portfolio_live)

    def get_drift(self) -> dict:
        with self._lock:
            return dict(self.drift_state)

    async def subscribe(self) -> asyncio.Queue:
        """SSE 客户端订阅事件流"""
        q = asyncio.Queue(maxsize=100)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass
