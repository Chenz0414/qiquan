# -*- coding: utf-8 -*-
"""
Phase D2 cron 脚本：每日 15:15 生成日报并推送 PushPlus
======================================================
聚合 signal_db + rule_drift + rejected_signals，写入 daily_reports 表
并（可选）通过 PushPlus 推送简化摘要。

使用:
  python -m scripts.generate_daily_report [--db PATH] [--push]
  crontab: 15 15 * * 1-5 cd /home/ubuntu/qiquan && .../python -m scripts.generate_daily_report --push
"""
import sys
import os
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signal_db import SignalDB


def _today_signals(conn, date_str: str):
    return conn.execute("""
        SELECT * FROM signals
        WHERE date(entry_time) = ?
    """, (date_str,)).fetchall()


def _today_rejects(conn, date_str: str):
    return conn.execute("""
        SELECT reason, COUNT(*) AS cnt FROM rejected_signals
        WHERE date(created_at) = ?
        GROUP BY reason ORDER BY cnt DESC
    """, (date_str,)).fetchall()


def _open_positions(conn):
    return conn.execute(
        "SELECT * FROM signals WHERE status='open' ORDER BY entry_time DESC"
    ).fetchall()


def _drift_alerts(conn, date_str: str):
    return conn.execute("""
        SELECT * FROM rule_drift
        WHERE date=? AND severity IN ('warn','alert')
        ORDER BY severity DESC
    """, (date_str,)).fetchall()


def _signal_r_mult(row):
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


def build_report(db: SignalDB, date_str: str) -> tuple[str, dict]:
    conn = db._conn
    signals = _today_signals(conn, date_str)
    closed_today = [s for s in signals if s["status"] == "closed"]
    open_today = [s for s in signals if s["status"] == "open"]
    wins = [s for s in closed_today if (s["pnl_pct"] or 0) > 0]
    losses = [s for s in closed_today if (s["pnl_pct"] or 0) <= 0]
    total_pnl = sum((s["pnl_pct"] or 0) for s in closed_today)

    open_all = _open_positions(conn)
    rejects = _today_rejects(conn, date_str)
    drifts = _drift_alerts(conn, date_str)

    # 按场景分组
    by_scenario = {}
    for s in signals:
        sc = f"场景{s['scenario']}"
        g = by_scenario.setdefault(sc, {"hit": 0, "win": 0,
                                        "lose": 0, "open": 0})
        g["hit"] += 1
        if s["status"] == "open":
            g["open"] += 1
        elif (s["pnl_pct"] or 0) > 0:
            g["win"] += 1
        else:
            g["lose"] += 1

    lines = [f"# {date_str} 交易日报", ""]
    lines.append("## 今日触发")
    if not signals:
        lines.append("- 无")
    for sc, g in sorted(by_scenario.items()):
        lines.append(
            f"- {sc}: {g['hit']} 笔 (胜 {g['win']} / 负 {g['lose']} / 持仓 {g['open']})")
    lines.append(f"- 今日合计: {len(signals)} 笔入场, 已平 {len(closed_today)} 笔, "
                 f"胜率 {(len(wins)/len(closed_today)*100) if closed_today else 0:.0f}%, "
                 f"净盈亏 {total_pnl:+.2f}%")
    lines.append("")

    lines.append("## 持仓快照")
    if not open_all:
        lines.append("- 无持仓")
    for p in open_all[:20]:
        lines.append(
            f"- {p['sym_name']} {p['direction']} @{p['entry_price']} "
            f"止损 {p['initial_stop']} 场景{p['scenario']}")
    lines.append("")

    lines.append("## 规则漂移")
    if not drifts:
        lines.append("- 无告警")
    for d in drifts:
        lines.append(
            f"- {d['severity'].upper()} {d['rule_key']}: "
            f"近7天胜率 {d['recent_win_rate']*100:.0f}% vs 历史 "
            f"{d['historical_win_rate']*100:.0f}% z={d['z_score']:+.2f}")
    lines.append("")

    lines.append("## 拒绝分析")
    total_rej = sum(r["cnt"] for r in rejects)
    lines.append(f"- 今日拒绝 {total_rej} 次")
    for r in rejects[:8]:
        lines.append(f"  - {r['reason']}: {r['cnt']}")
    lines.append("")

    content = "\n".join(lines)
    stats = {
        "signals_today": len(signals),
        "closed_today": len(closed_today),
        "wins": len(wins), "losses": len(losses),
        "total_pnl_pct": total_pnl,
        "open_count": len(open_all),
        "drift_warn_count": len(drifts),
        "reject_count": total_rej,
    }
    return content, stats


def push_pushplus(title: str, content_html: str, token: str) -> bool:
    try:
        import requests
    except ImportError:
        print("[push] requests 未安装，跳过")
        return False
    try:
        resp = requests.post(
            "http://www.pushplus.plus/send",
            json={"token": token, "title": title,
                  "content": content_html, "template": "markdown"},
            timeout=10)
        data = resp.json()
        ok = data.get("code") == 200
        print(f"[push] {'成功' if ok else '失败'}: {data}")
        return ok
    except Exception as e:
        print(f"[push] 异常: {e}")
        return False


def _load_push_token() -> str:
    # 从 monitor_settings.json 读 pushplus_token
    try:
        with open("monitor_settings.json", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("pushplus_token") or ""
    except Exception:
        return os.environ.get("PUSHPLUS_TOKEN", "")


def run(db_path: str, do_push: bool = False, date_str: str = None):
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    db = SignalDB(db_path)
    content, stats = build_report(db, date_str)
    db.save_daily_report(date_str, content, stats)
    print(f"[daily_report] {date_str} 已写入 daily_reports 表")
    print("-" * 60)
    print(content)
    print("-" * 60)

    if do_push:
        token = _load_push_token()
        if not token:
            print("[push] 未找到 pushplus_token，跳过")
            return
        push_pushplus(f"交易日报 {date_str}", content, token)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="state/signals.db")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD，默认今天")
    args = ap.parse_args()
    run(args.db, do_push=args.push, date_str=args.date)


if __name__ == "__main__":
    main()
