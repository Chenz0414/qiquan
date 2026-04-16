# -*- coding: utf-8 -*-
"""
Phase C 前端 dev server
=======================
不连 TqSdk，构造一份假数据塞进 DashboardState + SQLite，
让前端骨架可以在浏览器里跑。

用法:
  python -m web.dev_server
浏览器:
  http://127.0.0.1:8765/
"""

import sys
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn

from web.app import create_app
from web.state_bridge import DashboardState
from signal_db import SignalDB


def _seed_fake_data(ds: DashboardState, db: SignalDB):
    """塞一些假数据让界面不空。"""
    now = datetime.now().isoformat()
    ds.update_engine_status("running")

    # 假候选池
    ds.set_candidate("SHFE.rb", {
        "sym_key": "SHFE.rb", "sym_name": "螺纹钢",
        "kind": "A_long", "direction_bias": "long",
        "trigger_price": 3450.0, "current_price": 3448.0,
        "distance_ticks": 2,
        "er20": 0.62, "deviation_atr": 1.1,
    })
    ds.set_candidate("DCE.i", {
        "sym_key": "DCE.i", "sym_name": "铁矿石",
        "kind": "BC_short", "direction_bias": "short",
        "trigger_price": 720.0, "current_price": 724.5,
        "distance_ticks": -5,
        "er20": 0.55, "deviation_atr": 1.4,
    })
    ds.set_candidate("INE.sc", {
        "sym_key": "INE.sc", "sym_name": "原油",
        "kind": "type1_pending", "direction_bias": "long",
        "trigger_price": 642.0, "current_price": 641.8,
        "distance_ticks": 1,
        "er20": 0.48, "deviation_atr": 0.9,
    })

    # 热力图
    for sk, name, sec, td, er in [
        ("SHFE.rb", "螺纹钢", "黑色", 1, 0.62),
        ("SHFE.hc", "热卷", "黑色", 1, 0.58),
        ("DCE.i", "铁矿石", "黑色", -1, 0.55),
        ("SHFE.ag", "白银", "贵金属", -1, 0.02),
        ("SHFE.au", "黄金", "贵金属", 1, 0.45),
    ]:
        ds.update_heatmap_cell(sk, {
            "sym_key": sk, "sym_name": name, "sector": sec,
            "trend_dir": td, "er20": er, "deviation_atr": 0.8,
            "last_price": 3450.0, "sparkline_20": [3440+i for i in range(20)],
            "has_candidate": True, "has_position": False,
            "bar_time": now,
        })

    # 假持仓
    sig_id = db.record_entry(
        sym_key="CZCE.SR", sym_name="白糖", direction="long",
        signal_type="A", pullback_bars=0, scenario=1,
        entry_price=5328.0, initial_stop=5309.0,
        entry_time=now, er20=0.62, deviation_atr=1.6,
        position_multiplier=1, exit_strategy="S6",
    )
    ds.update_position("CZCE.SR", {
        "signal_id": sig_id, "sym_key": "CZCE.SR", "sym_name": "白糖",
        "direction": "long", "entry_price": 5328.0,
        "initial_stop": 5309.0, "current_stop": 5315.0,
        "exit_strategy": "S6", "bars_held": 8,
    })
    ds.update_portfolio_live(sig_id, {
        "current_r": 0.85, "mfe_r": 1.2, "mae_r": -0.3,
        "current_stop": 5315.0, "bars_held": 8,
    })

    # 假拒绝流水
    for i, (sk, stage, reason) in enumerate([
        ("SHFE.ni", "er20_filter", "er20_too_low"),
        ("DCE.p", "deviation_filter", "deviation_out_of_band"),
        ("SHFE.al", "global_filter", "chop_mode_active"),
        ("CZCE.MA", "near_miss", "ER差0.02即触发场景3"),
    ]):
        ds.push_reject(sk, stage, reason,
                       {"er20": 0.3 + i*0.05, "idx": i})

    # 板块暴露
    from web.correlation_service import (
        compute_sector_exposure, detect_sector_warnings)
    open_sigs = db.get_open_signals()
    exp = compute_sector_exposure(open_sigs, ds.candidate_pool)
    warnings = detect_sector_warnings(exp)
    ds.update_sector_exposure(exp, warnings=warnings)


def _sse_pump(ds: DashboardState):
    """后台线程：每 3 秒发一条 reject + 更新候选，模拟事件流。"""
    i = 0
    while True:
        time.sleep(3)
        i += 1
        ds.push_reject(
            "SHFE.rb", "er20_filter", "er20_too_low",
            {"er20": 0.3, "loop": i}
        )
        # 改一下候选距离
        if "SHFE.rb" in ds.candidate_pool:
            c = dict(ds.candidate_pool["SHFE.rb"])
            c["distance_ticks"] = (i % 5) - 1
            c["current_price"] = 3448.0 + (i % 5)
            ds.set_candidate("SHFE.rb", c)


def main():
    db = SignalDB(":memory:")
    ds = DashboardState()
    _seed_fake_data(ds, db)

    t = threading.Thread(target=_sse_pump, args=(ds,), daemon=True)
    t.start()

    app = create_app(ds, db)
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")


if __name__ == "__main__":
    main()
