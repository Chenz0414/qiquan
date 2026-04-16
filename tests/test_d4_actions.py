# -*- coding: utf-8 -*-
"""D4 一键操作端点测试"""
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from web.app import create_app
from web.state_bridge import DashboardState
from signal_db import SignalDB


def _make_client():
    ds = DashboardState()
    db = SignalDB(":memory:")
    app = create_app(ds, db)
    return TestClient(app), ds, db


def test_silence_and_unsilence():
    client, ds, db = _make_client()
    r = client.post("/api/actions/silence",
                    json={"sym_key": "SHFE.rb", "minutes": 30})
    assert r.status_code == 200
    assert "SHFE.rb" in ds.silenced_symbols
    r = client.post("/api/actions/silence",
                    json={"sym_key": "SHFE.rb", "minutes": 0})
    assert r.status_code == 200
    assert "SHFE.rb" not in ds.silenced_symbols


def test_pause_rule_and_clear():
    client, ds, db = _make_client()
    r = client.post("/api/actions/pause_rule",
                    json={"rule_key": "scenario_1",
                          "until": "2026-04-18T00:00:00"})
    assert r.status_code == 200
    assert ds.paused_rules.get("scenario_1") == "2026-04-18T00:00:00"
    r = client.post("/api/actions/pause_rule",
                    json={"rule_key": "scenario_1", "until": "clear"})
    assert r.status_code == 200
    assert "scenario_1" not in ds.paused_rules


def test_note_add_and_list():
    client, ds, db = _make_client()
    r = client.post("/api/actions/note",
                    json={"text": "观察价位", "sym_key": "rb"})
    assert r.status_code == 200
    nid = r.json()["note_id"]
    assert nid > 0
    r = client.get("/api/actions/notes")
    assert r.status_code == 200
    items = r.json()["items"]
    assert any(n["id"] == nid for n in items)


def test_manual_close_flow():
    client, ds, db = _make_client()
    sid = db.record_entry(
        sym_key="SHFE.rb", sym_name="螺纹钢", direction="long",
        signal_type="A", pullback_bars=0, scenario=1,
        entry_price=3400.0, initial_stop=3380.0,
        entry_time=datetime.now().isoformat(),
        er20=0.6, deviation_atr=1.2,
        position_multiplier=1, exit_strategy="S6")
    r = client.post("/api/actions/manual_close",
                    json={"signal_id": sid, "price": 3450.0,
                          "reason": "target"})
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    # 再平一次必失败
    r = client.post("/api/actions/manual_close",
                    json={"signal_id": sid, "price": 3455.0})
    assert r.status_code == 400


def test_manual_close_missing_signal():
    client, ds, db = _make_client()
    r = client.post("/api/actions/manual_close",
                    json={"signal_id": 9999, "price": 100.0})
    assert r.status_code == 404


def test_action_state():
    client, ds, db = _make_client()
    client.post("/api/actions/silence",
                json={"sym_key": "DCE.i", "minutes": 10})
    r = client.get("/api/actions/state")
    assert r.status_code == 200
    data = r.json()
    assert "DCE.i" in data["silenced_symbols"]


if __name__ == "__main__":
    test_silence_and_unsilence(); print("ok silence")
    test_pause_rule_and_clear(); print("ok pause")
    test_note_add_and_list(); print("ok note")
    test_manual_close_flow(); print("ok manual_close")
    test_manual_close_missing_signal(); print("ok manual_close 404")
    test_action_state(); print("ok state")
    print("All D4 tests passed.")
