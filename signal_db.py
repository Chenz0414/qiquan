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

    def close(self):
        self._conn.close()
