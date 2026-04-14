# -*- coding: utf-8 -*-
"""
状态持久化管理
==============
保存/恢复 SignalDetector + ExitTracker 状态，支持断线重连。
原子写入 + 备份文件防损坏。
"""

import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class StateManager:
    VERSION = 1

    def __init__(self, state_file: str):
        self.state_file = state_file
        self._backup_file = state_file + '.bak'

    def save(self, detectors: dict, trackers: dict,
             tracker_meta: dict, bar_counts: dict,
             t1_detectors: dict = None, t1_trackers: dict = None,
             t1_tracker_meta: dict = None):
        """
        保存全部状态到 JSON。
        原子写入：先写 .tmp 再 rename。

        参数:
          detectors: {sym_key: SignalDetector}
          trackers: {sym_key: ExitTracker}
          tracker_meta: {sym_key: {scenario, entry_time, ...}}
          bar_counts: {sym_key: int}
          t1_detectors: {sym_key: Type1SignalDetector}
          t1_trackers: {sym_key: LadderRTracker}
          t1_tracker_meta: {sym_key: {...}}
        """
        state = {
            "version": self.VERSION,
            "saved_at": datetime.now().isoformat(),
            "detectors": {k: v.to_dict() for k, v in detectors.items()},
            "active_positions": {},
            "bar_counts": bar_counts,
        }
        for sym_key, tracker in trackers.items():
            state["active_positions"][sym_key] = {
                "tracker": tracker.to_dict(),
                "meta": tracker_meta.get(sym_key, {}),
            }

        # Type1 状态
        if t1_detectors:
            state["t1_detectors"] = {k: v.to_dict() for k, v in t1_detectors.items()}
        if t1_trackers:
            state["t1_positions"] = {}
            for sym_key, tracker in t1_trackers.items():
                state["t1_positions"][sym_key] = {
                    "tracker": tracker.to_dict(),
                    "meta": (t1_tracker_meta or {}).get(sym_key, {}),
                }

        # 确保目录存在
        os.makedirs(os.path.dirname(self.state_file) or '.', exist_ok=True)

        # 备份旧文件
        if os.path.exists(self.state_file):
            try:
                if os.path.exists(self._backup_file):
                    os.remove(self._backup_file)
                os.replace(self.state_file, self._backup_file)
            except Exception as e:
                logger.warning(f"备份状态文件失败: {e}")

        # 原子写入
        tmp_file = self.state_file + '.tmp'
        try:
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            os.replace(tmp_file, self.state_file)
            logger.debug(f"状态已保存: {len(detectors)}个检测器, "
                        f"{len(trackers)}个活跃持仓")
        except Exception as e:
            logger.error(f"保存状态失败: {e}")
            # 清理临时文件
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except:
                    pass

    def load(self) -> dict | None:
        """
        加载状态，主文件损坏时回退到 .bak。

        返回:
          state dict 或 None（文件不存在或全部损坏）
        """
        for path in [self.state_file, self._backup_file]:
            if not os.path.exists(path):
                continue
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                if state.get("version") != self.VERSION:
                    logger.warning(f"状态文件版本不匹配: {state.get('version')} != {self.VERSION}")
                    continue
                logger.info(f"状态已恢复 (from {os.path.basename(path)}), "
                           f"保存时间: {state.get('saved_at')}")
                return state
            except Exception as e:
                logger.warning(f"加载 {path} 失败: {e}")
                continue

        logger.info("无可用状态文件，将冷启动")
        return None

    def clear(self):
        """删除状态文件（强制冷启动）"""
        for path in [self.state_file, self._backup_file]:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"已删除: {path}")
