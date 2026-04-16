# -*- coding: utf-8 -*-
"""Phase B5 SSE 节流 / 去重 / 事件发送 冒烟测试"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web"))

from web.state_bridge import DashboardState
from web.correlation_service import (
    compute_sector_exposure, detect_sector_warnings,
)


async def _drain(q: asyncio.Queue, max_wait: float = 0.01) -> list:
    """非阻塞抽干队列"""
    out = []
    try:
        while True:
            out.append(q.get_nowait())
    except asyncio.QueueEmpty:
        pass
    return out


async def _test_reject_dedup():
    ds = DashboardState()
    q = await ds.subscribe()

    # 同 key 连 3 次 → 只出一条 SSE
    for _ in range(3):
        ds.push_reject("SHFE.rb", "er20_filter",
                       "er20_too_low", {"er20": 0.3})
    events = await _drain(q)
    rejects = [e for e in events if e["type"] == "reject"]
    assert len(rejects) == 1, f"去重失败: {len(rejects)}"

    # deque 记录全量（没被去重）
    assert len(ds.reject_stream) == 3, len(ds.reject_stream)

    # 不同 reason → 不去重
    ds.push_reject("SHFE.rb", "er20_filter",
                   "deviation_out_of_band", {})
    events2 = await _drain(q)
    assert any(e["type"] == "reject" for e in events2), "不同key应发"
    print("  [OK] push_reject 30s 同键去重 + DB 全记")


async def _test_candidate_sse():
    ds = DashboardState()
    q = await ds.subscribe()

    cand = {"sym_key": "SHFE.rb", "kind": "A_long",
            "trigger_price": 3450, "distance_ticks": 2}
    ds.set_candidate("SHFE.rb", cand)
    events = await _drain(q)
    updates = [e for e in events if e["type"] == "candidate_update"]
    assert len(updates) == 1
    assert updates[0]["data"]["upserts"][0]["sym_key"] == "SHFE.rb"

    # 200ms 内重复 upsert 被节流
    ds.set_candidate("SHFE.rb", cand)
    events2 = await _drain(q)
    assert not any(e["type"] == "candidate_update" for e in events2), \
        "节流内应无事件"

    # 移除 → 立即发
    ds.set_candidate("SHFE.rb", None)
    events3 = await _drain(q)
    removes = [e for e in events3
               if e["type"] == "candidate_update"
               and e["data"]["removes"]]
    assert len(removes) == 1
    print("  [OK] set_candidate upsert 200ms 节流 + remove 即发")


async def _test_heatmap_portfolio():
    ds = DashboardState()
    q = await ds.subscribe()

    ds.update_heatmap_cell("SHFE.rb", {"sym_key": "SHFE.rb",
                                        "trend_dir": 1, "er20": 0.6})
    events = await _drain(q)
    hms = [e for e in events if e["type"] == "heatmap_delta"]
    assert len(hms) == 1

    ds.update_portfolio_live(1001, {"current_r": 0.8, "mfe_r": 1.2})
    ds.update_portfolio_live(1001, {"current_r": 0.85, "mfe_r": 1.25})
    events2 = await _drain(q)
    pls = [e for e in events2 if e["type"] == "position_live"]
    assert len(pls) == 1, f"1s 节流应只发1次, got {len(pls)}"

    ds.remove_portfolio_live(1001)
    events3 = await _drain(q)
    rms = [e for e in events3
           if e["type"] == "position_live"
           and e["data"].get("removed")]
    assert len(rms) == 1
    print("  [OK] heatmap_delta + position_live 节流/removal 正确")


async def _test_sector_exposure_and_warning():
    ds = DashboardState()
    q = await ds.subscribe()

    # 黑色系 3 个多头持仓 → 应触发 warning
    open_signals = [
        {"sym_key": "SHFE.rb", "direction": "long"},
        {"sym_key": "SHFE.hc", "direction": "long"},
        {"sym_key": "DCE.i",   "direction": "long"},
    ]
    cand_pool = {}
    exposure = compute_sector_exposure(open_signals, cand_pool)
    warnings = detect_sector_warnings(exposure, min_same_dir=3)
    assert any(w["sector"] == "黑色" and w["direction"] == "long"
               for w in warnings), "黑色系3多头应报警"

    ds.update_sector_exposure(exposure, warnings=warnings)
    events = await _drain(q)
    sws = [e for e in events if e["type"] == "sector_warning"]
    assert len(sws) >= 1, "应至少1条 sector_warning"

    # 60s 内同一 (sector, direction) 不重发
    ds.update_sector_exposure(exposure, warnings=warnings)
    events2 = await _drain(q)
    sws2 = [e for e in events2 if e["type"] == "sector_warning"]
    assert len(sws2) == 0, f"60s 内重发了 {len(sws2)} 条"
    print("  [OK] sector_exposure + sector_warning 60s 去重")


async def _test_drift_alert():
    ds = DashboardState()
    q = await ds.subscribe()

    # normal → 不发
    ds.update_drift_state({"scenario_1": {
        "severity": "normal", "z_score": 0.3}})
    events = await _drain(q)
    assert not any(e["type"] == "drift_alert" for e in events)

    # normal → warn 发
    ds.update_drift_state({"scenario_1": {
        "severity": "warn", "z_score": -1.5}})
    events2 = await _drain(q)
    alerts = [e for e in events2 if e["type"] == "drift_alert"]
    assert len(alerts) == 1

    # warn → warn 不发
    ds.update_drift_state({"scenario_1": {
        "severity": "warn", "z_score": -1.7}})
    events3 = await _drain(q)
    assert not any(e["type"] == "drift_alert" for e in events3)

    # warn → alert 发
    ds.update_drift_state({"scenario_1": {
        "severity": "alert", "z_score": -2.5}})
    events4 = await _drain(q)
    alerts2 = [e for e in events4 if e["type"] == "drift_alert"]
    assert len(alerts2) == 1
    print("  [OK] drift_alert 仅 severity 上升时发")


async def main():
    await _test_reject_dedup()
    await _test_candidate_sse()
    await _test_heatmap_portfolio()
    await _test_sector_exposure_and_warning()
    await _test_drift_alert()
    print("\n[ALL PASS]")


if __name__ == "__main__":
    asyncio.run(main())
