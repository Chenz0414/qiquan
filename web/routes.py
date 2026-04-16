# -*- coding: utf-8 -*-
"""
API 路由 + SSE 端点
===================
Phase A4 扩展：
  - /api/contract/inspect
  - /api/rules/catalog|summary|card/:rule_key|card/:rule_key/triggers
  - /api/candidates
  - /api/rejects + /api/rejects/aggregate
  - /api/market/heatmap
  - /api/sector/exposure
  - /api/positions/live
  - /api/drift
"""

import asyncio
import json
from datetime import datetime
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

from rules_catalog import RULES, GLOBAL_FILTERS, REJECT_STAGES, REJECT_REASONS


# ============================================================
#  规则 <-> 信号过滤映射（根据 signals 表的 scenario 列反查）
# ============================================================

RULE_KEY_TO_SCENARIO = {
    "scenario_1": 1,
    "scenario_2": 2,
    "scenario_3": 3,
}


def _rule_triggers(db, rule_key: str, limit: int = 20,
                   status: str = "all") -> list:
    """按规则 key 从 signals 表取最近触发"""
    scenario = RULE_KEY_TO_SCENARIO.get(rule_key)
    if scenario is None:
        # Type1 等非场景规则：Phase A4 暂返空数组（引擎未标 rule_key）
        return []
    sql = "SELECT * FROM signals WHERE scenario = ?"
    params = [scenario]
    if status in ("open", "closed"):
        sql += " AND status = ?"
        params.append(status)
    sql += " ORDER BY entry_time DESC LIMIT ?"
    params.append(limit)
    rows = db._conn.execute(sql, params).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["stop_history"] = json.loads(d.get("stop_history") or "[]")
        except json.JSONDecodeError:
            d["stop_history"] = []
        out.append(d)
    return out


def _rule_card(dashboard, db, rule_key: str) -> dict:
    """聚合规则卡片数据"""
    rule = RULES.get(rule_key)
    if not rule:
        raise HTTPException(404, f"unknown rule_key: {rule_key}")

    today_counters = dashboard.rule_today_counters.get(rule_key, {})
    stats_120 = db.get_rule_stats(rule_key, 120)
    stats_7 = db.get_rule_stats(rule_key, 7)
    drift = db.get_latest_drift(rule_key)

    return {
        "rule_key": rule_key,
        "label": rule["label"],
        "family": rule["family"],
        "description": rule["description"],
        "conditions": rule["conditions"],
        "exit_strategy": rule.get("exit_strategy"),
        "rationale": rule.get("rationale"),
        "disabled": rule.get("disabled", False),
        "stats": {
            "today": today_counters,
            "window_7d": stats_7,
            "window_120d": stats_120,
        },
        "drift": drift[0] if drift else None,
        "recent_triggers_20": _rule_triggers(db, rule_key, 20, "all"),
    }


# ============================================================
#  合约查询（Phase A4 薄版：用 dashboard_state + db 拼装）
# ============================================================

def _normalize_code(code: str) -> str:
    """rb2510 / SHFE.rb2510 / SHFE.rb → 归一化 sym_key 候选"""
    if not code:
        return ""
    c = code.strip()
    # 去掉交易所前缀
    if "." in c:
        c = c.split(".", 1)[1]
    # 去掉月份数字（rb2510 → rb）
    import re
    m = re.match(r"^([A-Za-z]+)", c)
    return m.group(1).lower() if m else c.lower()


def _resolve_sym_key(dashboard, code: str) -> str:
    """尝试把用户输入解析为 dashboard 里的 sym_key"""
    norm = _normalize_code(code)
    snap_states = dashboard.symbol_states or {}
    # 精确匹配（sym_key 可能是 'SHFE.rb' 这种带交易所的）
    for k in snap_states.keys():
        if k.lower() == code.lower():
            return k
        if k.split(".")[-1].lower() == norm:
            return k
    # 也查 heatmap / positions / trends
    for src in (dashboard.symbol_heatmap, dashboard.active_positions,
                dashboard.detector_trends):
        for k in (src or {}).keys():
            if k.split(".")[-1].lower() == norm:
                return k
    return norm  # 兜底返回归一化值


def _contract_inspect(dashboard, db, code: str) -> dict:
    sym_key = _resolve_sym_key(dashboard, code)
    sym_state = dashboard.symbol_states.get(sym_key, {})
    heat = dashboard.symbol_heatmap.get(sym_key, {})
    position = dashboard.active_positions.get(sym_key)

    last_price = (sym_state.get("last_price")
                  or heat.get("last_price"))

    panel = {
        "ema5": sym_state.get("ema5"),
        "ema10": sym_state.get("ema10") or sym_state.get("last_ema_snapshot", {}).get("ema10"),
        "ema20": sym_state.get("ema20") or sym_state.get("last_ema_snapshot", {}).get("ema20"),
        "ema60": sym_state.get("ema60"),
        "ema120": sym_state.get("ema120") or sym_state.get("last_ema_snapshot", {}).get("ema120"),
        "er20": sym_state.get("last_er20") or heat.get("er20"),
        "atr": sym_state.get("last_atr"),
        "deviation_atr": sym_state.get("deviation_atr") or heat.get("deviation_atr"),
        "trend_dir": dashboard.detector_trends.get(sym_key)
                     if sym_key in dashboard.detector_trends else heat.get("trend_dir"),
        "bar_time": dashboard.last_bar_time.get(sym_key),
    }

    current_state = {
        "has_position": position is not None,
        "position_id": position.get("signal_id") if position else None,
        "pullback_stage": sym_state.get("pullback_stage"),
        "pullback_bar_count": sym_state.get("pullback_bar_count", 0),
        "candidate_scenario": sym_state.get("candidate_scenario"),
    }

    # 120d 规则胜率（按当前 candidate_scenario 取）
    rule_stats_120d = {}
    if current_state["candidate_scenario"]:
        rk = f"scenario_{current_state['candidate_scenario']}"
        rule_stats_120d = db.get_rule_stats(rk, 120)

    return {
        "kind": "future",
        "code": code,
        "sym_key": sym_key,
        "sym_name": (position or heat).get("sym_name") if (position or heat) else None,
        "last_price": last_price,
        "panel": panel,
        "current_state": current_state,
        "rule_stats_120d": rule_stats_120d,
        "options_suggestion": None,        # Phase B 填充
        "sector_correlation_warning": None,
        "rule_drift_warning": (db.get_latest_drift(
            f"scenario_{current_state['candidate_scenario']}"
        )[0:1] if current_state["candidate_scenario"] else [None])[0],
    }


# ============================================================
#  Router
# ============================================================

def create_router() -> APIRouter:
    router = APIRouter()

    # ----------------------------------------------------------
    #  兼容老端点
    # ----------------------------------------------------------

    @router.get("/")
    async def index(request: Request):
        return request.app.state.templates.TemplateResponse(
            request=request, name="index.html")

    @router.get("/api/snapshot")
    async def snapshot(request: Request):
        dashboard = request.app.state.dashboard
        db = request.app.state.db
        snap = dashboard.snapshot()
        snap["daily_stats"] = db.get_daily_stats()
        return snap

    @router.get("/api/signals")
    async def signals(request: Request, status: str = "all", limit: int = 50):
        db = request.app.state.db
        if status == "open":
            return db.get_open_signals()
        return db.get_recent_signals(limit=limit)

    # ----------------------------------------------------------
    #  合约查询（§5）
    # ----------------------------------------------------------

    @router.get("/api/contract/inspect")
    async def contract_inspect(request: Request, code: str):
        if not code:
            raise HTTPException(400, "missing code")
        dashboard = request.app.state.dashboard
        db = request.app.state.db
        return _contract_inspect(dashboard, db, code)

    # ----------------------------------------------------------
    #  规则（§6 / §17）
    # ----------------------------------------------------------

    @router.get("/api/rules/catalog")
    async def rules_catalog_ep():
        return {
            "rules": RULES,
            "global_filters": GLOBAL_FILTERS,
            "reject_stages": REJECT_STAGES,
            "reject_reasons": REJECT_REASONS,
        }

    @router.get("/api/rules/summary")
    async def rules_summary(request: Request, window: str = "7d"):
        dashboard = request.app.state.dashboard
        db = request.app.state.db
        win_days = 7 if window == "7d" else 120
        out = []
        for rk, meta in RULES.items():
            today = dashboard.rule_today_counters.get(rk, {})
            stats = db.get_rule_stats(rk, win_days)
            drift = db.get_latest_drift(rk)
            out.append({
                "rule_key": rk,
                "label": meta["label"],
                "family": meta["family"],
                "disabled": meta.get("disabled", False),
                "today": today,
                "stats": stats,
                "drift": drift[0] if drift else None,
            })
        return {"window": window, "rules": out}

    @router.get("/api/rules/card/{rule_key}")
    async def rule_card(request: Request, rule_key: str):
        dashboard = request.app.state.dashboard
        db = request.app.state.db
        return _rule_card(dashboard, db, rule_key)

    @router.get("/api/rules/card/{rule_key}/triggers")
    async def rule_card_triggers(request: Request, rule_key: str,
                                 limit: int = 50, status: str = "all"):
        db = request.app.state.db
        return {
            "rule_key": rule_key,
            "status": status,
            "items": _rule_triggers(db, rule_key, limit, status),
        }

    # ----------------------------------------------------------
    #  候选池（§3）
    # ----------------------------------------------------------

    @router.get("/api/candidates")
    async def candidates(request: Request,
                         min_distance_ticks: int = None,
                         kind: str = "all"):
        dashboard = request.app.state.dashboard
        items = dashboard.get_candidates(min_distance_ticks, kind)
        return {"count": len(items), "items": items}

    # ----------------------------------------------------------
    #  拒绝流水（§12）
    # ----------------------------------------------------------

    @router.get("/api/rejects")
    async def rejects(request: Request, stage: str = None,
                      sym_key: str = None, limit: int = 200,
                      source: str = "db"):
        """source=db 取库；source=stream 取内存 deque"""
        dashboard = request.app.state.dashboard
        db = request.app.state.db
        if source == "stream":
            items = dashboard.get_rejects(limit=limit)
            if stage:
                items = [x for x in items if x.get("stage") == stage]
            if sym_key:
                items = [x for x in items if x.get("sym_key") == sym_key]
            return {"source": "stream", "count": len(items), "items": items}
        items = db.get_rejects(stage=stage, sym_key=sym_key, limit=limit)
        return {"source": "db", "count": len(items), "items": items}

    @router.get("/api/rejects/aggregate")
    async def rejects_aggregate(request: Request, window: str = "today"):
        db = request.app.state.db
        agg = db.aggregate_rejects(window=window)
        return {"window": window, "items": agg}

    # ----------------------------------------------------------
    #  市场热力图 + 板块（§4 / §8）
    # ----------------------------------------------------------

    @router.get("/api/market/heatmap")
    async def market_heatmap(request: Request):
        dashboard = request.app.state.dashboard
        items = dashboard.get_heatmap()
        return {"count": len(items), "items": items}

    @router.get("/api/sector/exposure")
    async def sector_exposure(request: Request):
        dashboard = request.app.state.dashboard
        return dashboard.sector_exposure or {}

    # ----------------------------------------------------------
    #  持仓实时（§9）
    # ----------------------------------------------------------

    @router.get("/api/positions/live")
    async def positions_live(request: Request):
        dashboard = request.app.state.dashboard
        db = request.app.state.db
        live = dashboard.get_portfolio_live()
        return {
            "open_signals": db.get_open_signals(),
            "live": live,
        }

    # ----------------------------------------------------------
    #  规则漂移（§7）
    # ----------------------------------------------------------

    @router.get("/api/drift")
    async def drift(request: Request, rule_key: str = None):
        db = request.app.state.db
        rows = db.get_latest_drift(rule_key=rule_key)
        return {"count": len(rows), "items": rows}

    # ----------------------------------------------------------
    #  时钟校准（§19）
    # ----------------------------------------------------------

    @router.get("/api/time")
    async def server_time():
        return {"server_time": datetime.now().isoformat()}

    # ----------------------------------------------------------
    #  SSE（兼容保留）
    # ----------------------------------------------------------

    @router.get("/api/events")
    async def sse_events(request: Request):
        dashboard = request.app.state.dashboard
        queue = await dashboard.subscribe()

        async def event_stream():
            try:
                yield f"event: connected\ndata: {{}}\n\n"
                while True:
                    try:
                        event = await asyncio.wait_for(
                            queue.get(), timeout=30)
                        data = json.dumps(event["data"], ensure_ascii=False)
                        yield f"event: {event['type']}\ndata: {data}\n\n"
                    except asyncio.TimeoutError:
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
