# -*- coding: utf-8 -*-
"""
API 路由 + SSE 端点
"""

import asyncio
import json
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/")
    async def index(request: Request):
        """仪表盘主页"""
        return request.app.state.templates.TemplateResponse(
            request=request, name="index.html")

    @router.get("/api/snapshot")
    async def snapshot(request: Request):
        """全量状态快照"""
        dashboard = request.app.state.dashboard
        db = request.app.state.db

        snap = dashboard.snapshot()
        snap["daily_stats"] = db.get_daily_stats()
        return snap

    @router.get("/api/signals")
    async def signals(request: Request, status: str = "all", limit: int = 50):
        """信号历史"""
        db = request.app.state.db
        if status == "open":
            return db.get_open_signals()
        return db.get_recent_signals(limit=limit)

    @router.get("/api/events")
    async def sse_events(request: Request):
        """SSE 实时事件流"""
        dashboard = request.app.state.dashboard
        queue = await dashboard.subscribe()

        async def event_stream():
            try:
                # 先发一个心跳确认连接
                yield f"event: connected\ndata: {{}}\n\n"
                while True:
                    try:
                        event = await asyncio.wait_for(
                            queue.get(), timeout=30)
                        data = json.dumps(event["data"], ensure_ascii=False)
                        yield f"event: {event['type']}\ndata: {data}\n\n"
                    except asyncio.TimeoutError:
                        # 30秒无事件，发心跳保活
                        yield f"event: heartbeat\ndata: {{}}\n\n"
            except asyncio.CancelledError:
                pass
            finally:
                dashboard.unsubscribe(queue)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return router
