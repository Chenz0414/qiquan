# -*- coding: utf-8 -*-
"""
Phase D3 沙盒回测服务
=====================
轻量 what-if 分析：基于 signal_db.signals 表历史数据，按新过滤阈值重算
命中数、胜率、avg_r、MFE 分位。不做真正的 bar 重放，只做"若过滤器不同
会保留哪些信号"的对比。

真正的多因子回测仍在本地 backtest_engine.py 跑；Web 沙盒面向
"调 er/deviation 阈值"这种快速假设验证。
"""
from __future__ import annotations

import math
import uuid
import threading
from datetime import datetime, timedelta
from typing import Optional


# 内存 job 存储（单机单进程）
_JOBS: dict = {}
_JOBS_LOCK = threading.Lock()


def _percentile(xs: list, p: float):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return float(xs[int(k)])
    return float(xs[lo] + (xs[hi] - xs[lo]) * (k - lo))


def _r_mult(row) -> Optional[float]:
    try:
        risk = abs((row["entry_price"] or 0) - (row["initial_stop"] or 0))
        if risk <= 0:
            return None
        move = (row["exit_price"] or 0) - (row["entry_price"] or 0)
        if row["direction"] == "short":
            move = -move
        return move / risk
    except Exception:
        return None


def _summarize(rows: list) -> dict:
    closed = [r for r in rows if r["status"] == "closed"
              and r["pnl_pct"] is not None]
    wins = sum(1 for r in closed if (r["pnl_pct"] or 0) > 0)
    r_mults = [r for r in (_r_mult(r) for r in closed) if r is not None]
    avg_r = sum(r_mults) / len(r_mults) if r_mults else None
    return {
        "hits": len(rows),
        "closed": len(closed),
        "wins": wins,
        "losses": len(closed) - wins,
        "win_rate": (wins / len(closed)) if closed else None,
        "avg_pnl_pct": ((sum(r["pnl_pct"] for r in closed) / len(closed))
                        if closed else None),
        "avg_r": avg_r,
        "mfe_p50": _percentile(r_mults, 0.5) if r_mults else None,
        "mfe_p90": _percentile(r_mults, 0.9) if r_mults else None,
    }


def run_sandbox(db, cfg: dict) -> dict:
    """
    cfg 字段:
      scenario: int|None          - 按 scenario 过滤（None=all）
      direction: "long"|"short"|None
      er_min: float               - 保留 er20 >= 此值
      er_max: float               - 保留 er20 <= 此值
      deviation_min: float        - 保留 |deviation_atr| >= 此值
      deviation_max: float
      days: int                   - 回看天数
    返回:
      {baseline: {...}, sandbox: {...}, delta: {...}}
    """
    days = int(cfg.get("days") or 120)
    since = (datetime.now() - timedelta(days=days)).isoformat()

    sql = "SELECT * FROM signals WHERE entry_time >= ?"
    params = [since]
    if cfg.get("scenario"):
        sql += " AND scenario = ?"
        params.append(int(cfg["scenario"]))
    if cfg.get("direction") in ("long", "short"):
        sql += " AND direction = ?"
        params.append(cfg["direction"])
    rows = [dict(r) for r in db._conn.execute(sql, params).fetchall()]
    baseline = _summarize(rows)

    er_min = cfg.get("er_min")
    er_max = cfg.get("er_max")
    dev_min = cfg.get("deviation_min")
    dev_max = cfg.get("deviation_max")

    def _keep(r):
        er = r.get("er20")
        dev = abs(r.get("deviation_atr") or 0)
        if er_min is not None and (er is None or er < er_min):
            return False
        if er_max is not None and (er is None or er > er_max):
            return False
        if dev_min is not None and dev < dev_min:
            return False
        if dev_max is not None and dev > dev_max:
            return False
        return True

    kept = [r for r in rows if _keep(r)]
    sandbox = _summarize(kept)

    def _delta(a, b):
        if a is None or b is None:
            return None
        return a - b

    delta = {
        "hits": sandbox["hits"] - baseline["hits"],
        "win_rate": _delta(sandbox["win_rate"], baseline["win_rate"]),
        "avg_r": _delta(sandbox["avg_r"], baseline["avg_r"]),
        "mfe_p50": _delta(sandbox["mfe_p50"], baseline["mfe_p50"]),
    }

    return {"baseline": baseline, "sandbox": sandbox,
            "delta": delta, "config": cfg}


def submit_job(db, cfg: dict) -> str:
    """同步执行并返回 job_id。沙盒只查 SQLite，<100ms。"""
    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "id": job_id, "status": "running",
            "created_at": datetime.now().isoformat(),
            "config": cfg, "result": None, "error": None,
        }
    try:
        result = run_sandbox(db, cfg)
        with _JOBS_LOCK:
            _JOBS[job_id].update({
                "status": "done", "result": result,
                "finished_at": datetime.now().isoformat(),
            })
    except Exception as e:
        with _JOBS_LOCK:
            _JOBS[job_id].update({
                "status": "error", "error": str(e),
                "finished_at": datetime.now().isoformat(),
            })
    return job_id


def get_job(job_id: str) -> Optional[dict]:
    with _JOBS_LOCK:
        return dict(_JOBS[job_id]) if job_id in _JOBS else None


def list_jobs(limit: int = 20) -> list:
    with _JOBS_LOCK:
        items = list(_JOBS.values())
    items.sort(key=lambda j: j["created_at"], reverse=True)
    return items[:limit]
