# -*- coding: utf-8 -*-
"""
Phase D1 cron 脚本：每日 16:30 计算规则统计 + 漂移
=================================================
针对 signal_db 内的 signals 表按 scenario 聚合：
  - 7d / 120d 两个窗口
  - hit_count / win_count / avg_pnl_pct / avg_r / mfe 分位
  - 漂移：7d vs 120d 胜率 z-score

使用:
  python -m scripts.compute_daily_rule_stats [--db PATH]
  crontab: 30 16 * * 1-5 cd /home/ubuntu/qiquan && .../python -m scripts.compute_daily_rule_stats
"""
import sys
import os
import math
import argparse
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signal_db import SignalDB
from rules_catalog import RULES

# 规则 key <-> signals.scenario 的映射（与 routes.py 保持一致）
RULE_TO_SCENARIO = {
    "scenario_1": 1,
    "scenario_2": 2,
    "scenario_3": 3,
}

WINDOWS = [7, 120]


def _fetch_signals(conn, scenario: int, since_iso: str):
    return conn.execute("""
        SELECT * FROM signals
        WHERE scenario = ? AND entry_time >= ?
    """, (scenario, since_iso)).fetchall()


def _percentile(xs: list, p: float):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return float(xs[int(k)])
    return float(xs[lo] + (xs[hi] - xs[lo]) * (k - lo))


def _compute_window(conn, rule_key: str, scenario: int,
                    window_days: int, today: datetime) -> dict:
    since = (today - timedelta(days=window_days)).isoformat()
    rows = _fetch_signals(conn, scenario, since)
    hit_count = len(rows)
    open_count = sum(1 for r in rows if r["status"] == "open")
    closed = [r for r in rows if r["status"] == "closed"
              and r["pnl_pct"] is not None]
    win_count = sum(1 for r in closed if (r["pnl_pct"] or 0) > 0)
    avg_pnl = (sum(r["pnl_pct"] for r in closed) / len(closed)
               if closed else None)

    # avg_r 基于 entry vs initial_stop 定义 1R
    r_mults = []
    for r in closed:
        try:
            risk = abs((r["entry_price"] or 0) - (r["initial_stop"] or 0))
            if risk <= 0:
                continue
            move = (r["exit_price"] or 0) - (r["entry_price"] or 0)
            if r["direction"] == "short":
                move = -move
            r_mults.append(move / risk)
        except Exception:
            continue
    avg_r = sum(r_mults) / len(r_mults) if r_mults else None
    mfe_p50 = _percentile(r_mults, 0.5) if r_mults else None
    mfe_p90 = _percentile(r_mults, 0.9) if r_mults else None

    return {
        "rule_key": rule_key,
        "window_days": window_days,
        "hit_count": hit_count,
        "open_count": open_count,
        "win_count": win_count,
        "avg_pnl_pct": avg_pnl,
        "avg_r": avg_r,
        "mfe_p50": mfe_p50,
        "mfe_p90": mfe_p90,
        "win_rate": (win_count / (hit_count - open_count)
                     if (hit_count - open_count) > 0 else None),
    }


def _compute_drift(recent: dict, hist: dict):
    r_wr = recent.get("win_rate")
    h_wr = hist.get("win_rate")
    if r_wr is None or h_wr is None:
        return None
    # 近似 std：二项 std = sqrt(p*(1-p)/n)
    n = hist["hit_count"] - hist["open_count"]
    if n <= 0:
        return None
    std = math.sqrt(h_wr * (1 - h_wr) / n) if 0 < h_wr < 1 else 0.05
    z = (r_wr - h_wr) / (std or 0.05)
    if abs(z) < 1:
        sev = "normal"
    elif abs(z) < 2:
        sev = "warn"
    else:
        sev = "alert"
    return {
        "recent_win_rate": r_wr,
        "historical_win_rate": h_wr,
        "z_score": z,
        "severity": sev,
    }


def run(db_path: str):
    db = SignalDB(db_path)
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_dt = datetime.now()

    summary = []
    for rule_key in RULES.keys():
        scenario = RULE_TO_SCENARIO.get(rule_key)
        if scenario is None:
            # 非 scenario 规则（如 type1_*）目前无法按 signals.scenario 映射
            continue
        stats_by_window = {}
        for w in WINDOWS:
            s = _compute_window(db._conn, rule_key, scenario, w, today_dt)
            db.upsert_rule_stats(
                date=today_str, rule_key=rule_key, window_days=w,
                hit_count=s["hit_count"], open_count=s["open_count"],
                win_count=s["win_count"], avg_pnl_pct=s["avg_pnl_pct"],
                avg_r=s["avg_r"], mfe_p50=s["mfe_p50"],
                mfe_p90=s["mfe_p90"])
            stats_by_window[w] = s

        drift = _compute_drift(stats_by_window[7], stats_by_window[120])
        if drift:
            db.upsert_rule_drift(
                date=today_str, rule_key=rule_key,
                recent_win_rate=drift["recent_win_rate"],
                historical_win_rate=drift["historical_win_rate"],
                z_score=drift["z_score"], severity=drift["severity"])
            summary.append((rule_key, drift))

    print(f"[daily_rule_stats] {today_str} 完成 · {len(summary)} 条漂移记录")
    for rk, d in summary:
        print(f"  {rk}: recent={d['recent_win_rate']:.2f} "
              f"hist={d['historical_win_rate']:.2f} "
              f"z={d['z_score']:+.2f} sev={d['severity']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="state/signals.db")
    args = ap.parse_args()
    run(args.db)


if __name__ == "__main__":
    main()
