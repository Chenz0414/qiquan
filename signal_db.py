# -*- coding: utf-8 -*-
"""
SQLite 信号历史持久化
=====================
记录每个信号的完整生命周期：入场 → 止损变更 → 平仓
"""

import os
import json
import sqlite3
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SignalDB:
    def __init__(self, db_path: str = "state/signals.db"):
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL 模式：读写并发 + 崩溃恢复
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.DatabaseError:
            logger.warning("无法切换 WAL，继续使用默认模式")
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS signals (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                sym_key         TEXT NOT NULL,
                sym_name        TEXT NOT NULL,
                direction       TEXT NOT NULL,
                signal_type     TEXT NOT NULL,
                scenario        INTEGER NOT NULL,
                entry_price     REAL NOT NULL,
                initial_stop    REAL NOT NULL,
                pullback_bars   INTEGER NOT NULL,
                deviation_atr   REAL,
                er20            REAL,
                position_multiplier INTEGER DEFAULT 1,
                exit_strategy   TEXT NOT NULL,
                entry_time      TEXT NOT NULL,
                exit_price      REAL,
                exit_reason     TEXT,
                pnl_pct         REAL,
                bars_held       INTEGER,
                exit_time       TEXT,
                stop_history    TEXT DEFAULT '[]',
                status          TEXT DEFAULT 'open'
            );
            CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);
            CREATE INDEX IF NOT EXISTS idx_signals_entry_time ON signals(entry_time);

            -- 拒绝流水（每根 bar 的拒绝原因落盘，供 near_miss / 聚合分析）
            CREATE TABLE IF NOT EXISTS rejected_signals (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                sym_key         TEXT NOT NULL,
                bar_time        TEXT NOT NULL,
                stage           TEXT NOT NULL,
                reason          TEXT NOT NULL,
                signal_type     TEXT,
                direction       TEXT,
                er20            REAL,
                deviation_atr   REAL,
                context_json    TEXT DEFAULT '{}',
                created_at      TEXT NOT NULL DEFAULT (datetime('now','localtime'))
            );
            CREATE INDEX IF NOT EXISTS idx_rejects_sym_time ON rejected_signals(sym_key, bar_time);
            CREATE INDEX IF NOT EXISTS idx_rejects_reason ON rejected_signals(reason);
            CREATE INDEX IF NOT EXISTS idx_rejects_created ON rejected_signals(created_at);

            -- 期权订阅（当日快照，供灰度追踪）
            CREATE TABLE IF NOT EXISTS option_subscriptions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                date            TEXT NOT NULL,
                underlying      TEXT NOT NULL,
                option_code     TEXT NOT NULL,
                strike          REAL,
                cp              TEXT,
                delta_est       REAL,
                expiry          TEXT,
                UNIQUE(date, option_code)
            );
            CREATE INDEX IF NOT EXISTS idx_opt_sub_date ON option_subscriptions(date);

            -- 每日规则统计（daily cron 预计算，提供快速胜率查询）
            CREATE TABLE IF NOT EXISTS daily_rule_stats (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                date            TEXT NOT NULL,
                rule_key        TEXT NOT NULL,
                window_days     INTEGER NOT NULL,
                hit_count       INTEGER DEFAULT 0,
                open_count      INTEGER DEFAULT 0,
                win_count       INTEGER DEFAULT 0,
                avg_pnl_pct     REAL,
                avg_r           REAL,
                mfe_p50         REAL,
                mfe_p90         REAL,
                UNIQUE(date, rule_key, window_days)
            );
            CREATE INDEX IF NOT EXISTS idx_drs_rule ON daily_rule_stats(rule_key);

            -- 规则漂移（每日计算 7d vs 120d 胜率 z-score）
            CREATE TABLE IF NOT EXISTS rule_drift (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                date                  TEXT NOT NULL,
                rule_key              TEXT NOT NULL,
                recent_win_rate       REAL,
                historical_win_rate   REAL,
                z_score               REAL,
                severity              TEXT,  -- normal/warn/alert
                UNIQUE(date, rule_key)
            );
            CREATE INDEX IF NOT EXISTS idx_drift_rule ON rule_drift(rule_key);

            -- 临时关注列表（用户手动添加的合约）
            CREATE TABLE IF NOT EXISTS watchlist (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                code            TEXT NOT NULL UNIQUE,
                added_at        TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                alert_price     REAL,
                alert_direction TEXT,
                expired_at      TEXT
            );

            -- 用户笔记 / 手动操作留痕
            CREATE TABLE IF NOT EXISTS manual_notes (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id       INTEGER,
                sym_key         TEXT,
                created_at      TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                text            TEXT NOT NULL,
                action_type     TEXT    -- note/manual_close/silence/pause_rule
            );
            CREATE INDEX IF NOT EXISTS idx_notes_signal ON manual_notes(signal_id);

            -- 每日收盘日报归档
            CREATE TABLE IF NOT EXISTS daily_reports (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                date              TEXT NOT NULL UNIQUE,
                content_markdown  TEXT NOT NULL,
                stats_json        TEXT DEFAULT '{}',
                generated_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self._conn.commit()

    def record_entry(self, sym_key: str, sym_name: str, direction: str,
                     signal_type: str, scenario: int, entry_price: float,
                     initial_stop: float, pullback_bars: int,
                     deviation_atr: float, er20: float,
                     position_multiplier: int, exit_strategy: str,
                     entry_time: str) -> int:
        """插入新信号，返回 signal_id"""
        cur = self._conn.execute("""
            INSERT INTO signals (sym_key, sym_name, direction, signal_type,
                scenario, entry_price, initial_stop, pullback_bars,
                deviation_atr, er20, position_multiplier, exit_strategy,
                entry_time, stop_history, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '[]', 'open')
        """, (sym_key, sym_name, direction, signal_type,
              scenario, entry_price, initial_stop, pullback_bars,
              deviation_atr, er20, position_multiplier, exit_strategy,
              entry_time))
        self._conn.commit()
        logger.debug(f"信号入库: id={cur.lastrowid} {sym_key} {direction}")
        return cur.lastrowid

    def record_stop_update(self, signal_id: int, old_stop: float,
                           new_stop: float, strategy: str, time: str):
        """追加止损变更到 stop_history"""
        row = self._conn.execute(
            "SELECT stop_history FROM signals WHERE id = ?",
            (signal_id,)).fetchone()
        if not row:
            return
        history = json.loads(row['stop_history'])
        history.append({
            'time': time, 'old': old_stop,
            'new': new_stop, 'strategy': strategy,
        })
        self._conn.execute(
            "UPDATE signals SET stop_history = ? WHERE id = ?",
            (json.dumps(history), signal_id))
        self._conn.commit()

    def record_exit(self, signal_id: int, exit_price: float,
                    exit_reason: str, pnl_pct: float,
                    bars_held: int, exit_time: str):
        """记录平仓"""
        self._conn.execute("""
            UPDATE signals SET exit_price = ?, exit_reason = ?,
                pnl_pct = ?, bars_held = ?, exit_time = ?, status = 'closed'
            WHERE id = ?
        """, (exit_price, exit_reason, pnl_pct, bars_held,
              exit_time, signal_id))
        self._conn.commit()
        logger.debug(f"信号平仓: id={signal_id} pnl={pnl_pct:+.2f}%")

    def get_open_signals(self) -> list[dict]:
        """获取所有持仓中的信号"""
        rows = self._conn.execute(
            "SELECT * FROM signals WHERE status = 'open' ORDER BY entry_time DESC"
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_recent_signals(self, limit: int = 50) -> list[dict]:
        """获取最近的信号（含已平仓）"""
        rows = self._conn.execute(
            "SELECT * FROM signals ORDER BY entry_time DESC LIMIT ?",
            (limit,)).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_daily_stats(self) -> dict:
        """今日统计"""
        today = datetime.now().strftime('%Y-%m-%d')
        row = self._conn.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) as open_count,
                   SUM(CASE WHEN status='closed' THEN 1 ELSE 0 END) as closed_count,
                   SUM(CASE WHEN status='closed' AND pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN status='closed' THEN pnl_pct ELSE 0 END) as total_pnl
            FROM signals WHERE entry_time >= ?
        """, (today,)).fetchone()
        return dict(row) if row else {}

    def _row_to_dict(self, row) -> dict:
        d = dict(row)
        d['stop_history'] = json.loads(d.get('stop_history', '[]'))
        return d

    # ================================================================
    #  rejected_signals DAO
    # ================================================================

    def record_reject(self, sym_key: str, bar_time: str, stage: str,
                      reason: str, signal_type: str = None,
                      direction: str = None, er20: float = None,
                      deviation_atr: float = None, context: dict = None) -> int:
        cur = self._conn.execute("""
            INSERT INTO rejected_signals (sym_key, bar_time, stage, reason,
                signal_type, direction, er20, deviation_atr, context_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (sym_key, bar_time, stage, reason, signal_type, direction,
              er20, deviation_atr, json.dumps(context or {}, ensure_ascii=False)))
        self._conn.commit()
        return cur.lastrowid

    def get_rejects(self, stage: str = None, sym_key: str = None,
                    limit: int = 200) -> list:
        where = []
        params = []
        if stage:
            where.append("stage = ?")
            params.append(stage)
        if sym_key:
            where.append("sym_key = ?")
            params.append(sym_key)
        sql = "SELECT * FROM rejected_signals"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["context"] = json.loads(d.pop("context_json", "{}"))
            except json.JSONDecodeError:
                d["context"] = {}
            out.append(d)
        return out

    def aggregate_rejects(self, window: str = "today") -> list:
        """按 reason 聚合，返回 [{reason, count}, ...]"""
        if window == "today":
            cutoff = datetime.now().strftime("%Y-%m-%d")
            rows = self._conn.execute("""
                SELECT reason, COUNT(*) as count FROM rejected_signals
                WHERE created_at >= ?
                GROUP BY reason ORDER BY count DESC
            """, (cutoff,)).fetchall()
        else:
            rows = self._conn.execute("""
                SELECT reason, COUNT(*) as count FROM rejected_signals
                GROUP BY reason ORDER BY count DESC
            """).fetchall()
        return [dict(r) for r in rows]

    # ================================================================
    #  option_subscriptions DAO
    # ================================================================

    def upsert_option_subscription(self, date: str, underlying: str,
                                   option_code: str, strike: float = None,
                                   cp: str = None, delta_est: float = None,
                                   expiry: str = None):
        self._conn.execute("""
            INSERT OR REPLACE INTO option_subscriptions
                (date, underlying, option_code, strike, cp, delta_est, expiry)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (date, underlying, option_code, strike, cp, delta_est, expiry))
        self._conn.commit()

    def get_option_subscriptions(self, date: str = None) -> list:
        if date:
            rows = self._conn.execute(
                "SELECT * FROM option_subscriptions WHERE date = ?",
                (date,)).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM option_subscriptions ORDER BY date DESC LIMIT 500"
            ).fetchall()
        return [dict(r) for r in rows]

    # ================================================================
    #  daily_rule_stats DAO
    # ================================================================

    def upsert_daily_rule_stats(self, date: str, rule_key: str,
                                window_days: int, stats: dict):
        self._conn.execute("""
            INSERT OR REPLACE INTO daily_rule_stats
                (date, rule_key, window_days, hit_count, open_count, win_count,
                 avg_pnl_pct, avg_r, mfe_p50, mfe_p90)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (date, rule_key, window_days,
              stats.get("hit_count", 0), stats.get("open_count", 0),
              stats.get("win_count", 0), stats.get("avg_pnl_pct"),
              stats.get("avg_r"), stats.get("mfe_p50"), stats.get("mfe_p90")))
        self._conn.commit()

    def upsert_rule_stats(self, date: str, rule_key: str,
                          window_days: int, hit_count: int,
                          open_count: int, win_count: int,
                          avg_pnl_pct: float = None,
                          avg_r: float = None,
                          mfe_p50: float = None,
                          mfe_p90: float = None):
        self._conn.execute("""
            INSERT OR REPLACE INTO daily_rule_stats
              (date, rule_key, window_days, hit_count, open_count,
               win_count, avg_pnl_pct, avg_r, mfe_p50, mfe_p90)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (date, rule_key, window_days, hit_count, open_count,
              win_count, avg_pnl_pct, avg_r, mfe_p50, mfe_p90))
        self._conn.commit()

    def get_rule_stats(self, rule_key: str, window_days: int = 120,
                       date: str = None) -> dict:
        if date is None:
            row = self._conn.execute("""
                SELECT * FROM daily_rule_stats
                WHERE rule_key = ? AND window_days = ?
                ORDER BY date DESC LIMIT 1
            """, (rule_key, window_days)).fetchone()
        else:
            row = self._conn.execute("""
                SELECT * FROM daily_rule_stats
                WHERE rule_key = ? AND window_days = ? AND date = ?
            """, (rule_key, window_days, date)).fetchone()
        return dict(row) if row else {}

    # ================================================================
    #  rule_drift DAO
    # ================================================================

    def upsert_rule_drift(self, date: str, rule_key: str,
                          recent_win_rate: float,
                          historical_win_rate: float,
                          z_score: float, severity: str):
        self._conn.execute("""
            INSERT OR REPLACE INTO rule_drift
                (date, rule_key, recent_win_rate, historical_win_rate,
                 z_score, severity)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (date, rule_key, recent_win_rate, historical_win_rate,
              z_score, severity))
        self._conn.commit()

    def get_latest_drift(self, rule_key: str = None) -> list:
        if rule_key:
            rows = self._conn.execute("""
                SELECT * FROM rule_drift WHERE rule_key = ?
                ORDER BY date DESC LIMIT 1
            """, (rule_key,)).fetchall()
        else:
            rows = self._conn.execute("""
                SELECT r.* FROM rule_drift r
                INNER JOIN (
                  SELECT rule_key, MAX(date) AS maxdate
                  FROM rule_drift GROUP BY rule_key
                ) m ON r.rule_key = m.rule_key AND r.date = m.maxdate
            """).fetchall()
        return [dict(r) for r in rows]

    # ================================================================
    #  watchlist DAO
    # ================================================================

    def add_watchlist(self, code: str, alert_price: float = None,
                      alert_direction: str = None,
                      expired_at: str = None) -> int:
        cur = self._conn.execute("""
            INSERT OR REPLACE INTO watchlist
                (code, alert_price, alert_direction, expired_at)
            VALUES (?, ?, ?, ?)
        """, (code, alert_price, alert_direction, expired_at))
        self._conn.commit()
        return cur.lastrowid

    def remove_watchlist(self, code_or_id) -> int:
        if isinstance(code_or_id, int):
            cur = self._conn.execute("DELETE FROM watchlist WHERE id = ?",
                                     (code_or_id,))
        else:
            cur = self._conn.execute("DELETE FROM watchlist WHERE code = ?",
                                     (code_or_id,))
        self._conn.commit()
        return cur.rowcount

    def get_watchlist(self) -> list:
        rows = self._conn.execute(
            "SELECT * FROM watchlist ORDER BY added_at DESC").fetchall()
        return [dict(r) for r in rows]

    # ================================================================
    #  manual_notes DAO
    # ================================================================

    def add_note(self, text: str, signal_id: int = None,
                 sym_key: str = None, action_type: str = "note") -> int:
        cur = self._conn.execute("""
            INSERT INTO manual_notes (signal_id, sym_key, text, action_type)
            VALUES (?, ?, ?, ?)
        """, (signal_id, sym_key, text, action_type))
        self._conn.commit()
        return cur.lastrowid

    def get_notes(self, signal_id: int = None, limit: int = 50) -> list:
        if signal_id:
            rows = self._conn.execute("""
                SELECT * FROM manual_notes WHERE signal_id = ?
                ORDER BY created_at DESC LIMIT ?
            """, (signal_id, limit)).fetchall()
        else:
            rows = self._conn.execute("""
                SELECT * FROM manual_notes ORDER BY created_at DESC LIMIT ?
            """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ================================================================
    #  daily_reports DAO
    # ================================================================

    def save_daily_report(self, date: str, content_markdown: str,
                          stats: dict = None) -> int:
        cur = self._conn.execute("""
            INSERT OR REPLACE INTO daily_reports
                (date, content_markdown, stats_json)
            VALUES (?, ?, ?)
        """, (date, content_markdown, json.dumps(stats or {}, ensure_ascii=False)))
        self._conn.commit()
        return cur.lastrowid

    def get_daily_report(self, date: str = None) -> dict:
        if date:
            row = self._conn.execute(
                "SELECT * FROM daily_reports WHERE date = ?",
                (date,)).fetchone()
        else:
            row = self._conn.execute(
                "SELECT * FROM daily_reports ORDER BY date DESC LIMIT 1"
            ).fetchone()
        if not row:
            return {}
        d = dict(row)
        try:
            d["stats"] = json.loads(d.pop("stats_json", "{}"))
        except json.JSONDecodeError:
            d["stats"] = {}
        return d

    def close(self):
        self._conn.close()
