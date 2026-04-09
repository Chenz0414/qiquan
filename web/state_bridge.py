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
            }

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
