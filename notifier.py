# -*- coding: utf-8 -*-
"""
微信推送通知（PushPlus）
========================
title = 列表里直接看到，够下单操作
content = 点开详情看到，补充信息
"""

import time
import logging
import requests
from signal_core import SYMBOL_CONFIGS, SCENARIO_NAMES

logger = logging.getLogger(__name__)


class PushPlusNotifier:
    API_URL = "http://www.pushplus.plus/send"

    def __init__(self, token: str,
                 stop_cooldown: int = 300, dry_run: bool = False):
        self.token = token
        self.stop_cooldown = stop_cooldown
        self.dry_run = dry_run
        # 止损推送限流：{sym_key: last_send_timestamp}
        self._stop_last_sent: dict[str, float] = {}

    def _send(self, title: str, content: str = ""):
        """发送消息到 PushPlus"""
        if self.dry_run:
            logger.info(f"[DRY-RUN] title: {title}")
            if content:
                logger.debug(f"[DRY-RUN] content: {content}")
            return True

        if not self.token:
            logger.warning("PushPlus token 未配置，跳过推送")
            return False

        payload = {
            "token": self.token,
            "title": title,
            "content": content or title,
            "template": "html",
        }

        for attempt in range(2):
            try:
                resp = requests.post(self.API_URL, json=payload, timeout=10)
                data = resp.json()
                if data.get("code") == 200:
                    logger.debug(f"推送成功: {title}")
                    return True
                logger.warning(f"推送失败: {data}")
            except Exception as e:
                logger.warning(f"推送异常 (attempt {attempt+1}): {e}")
            if attempt == 0:
                time.sleep(1)
        return False

    def _sym_label(self, sym_key: str) -> str:
        """SHFE.ag -> 白银"""
        cfg = SYMBOL_CONFIGS.get(sym_key)
        return cfg['name'] if cfg else sym_key

    def _dir_label(self, direction: str) -> str:
        return "多" if direction == "long" else "空"

    def _dir_verb(self, direction: str) -> str:
        return "做多" if direction == "long" else "做空"

    # ================================================================
    #  1. 新信号通知
    # ================================================================
    def notify_new_signal(self, sym_key: str, direction: str,
                          signal_type: str, pullback_bars: int,
                          entry_price: float, initial_stop: float,
                          scenario: int, deviation_atr: float,
                          er20: float, position_multiplier: int,
                          exit_strategy: str,
                          er40: float = None, er5_delta_6: float = None,
                          er40_delta_12: float = None,
                          ema10: float = None, ema20: float = None,
                          ema120: float = None):
        """
        title(工单名称): 【开仓】空PVC C4根 2x
        content第一行(设备名称): 入5407 止5473 ER0.55 S2出 2.3ATR
        """
        name = self._sym_label(sym_key)
        dl = self._dir_label(direction)
        pb = f"{signal_type}{pullback_bars}根" if pullback_bars > 0 else signal_type

        title = f"【开仓】{dl}{name} {pb} {position_multiplier}x"

        summary_line = (f"入{entry_price:.0f} 止{initial_stop:.0f} "
                        f"ER{er20:.2f} {exit_strategy}出 {deviation_atr:.1f}ATR")

        # 详情 HTML
        lines = [
            summary_line,
            f"<hr>",
            f"<h3>{dl}{name} ({sym_key})</h3>",
            f"<p>{SCENARIO_NAMES.get(scenario, f'场景{scenario}')}</p>",
            f"<p>入场价: <b>{entry_price}</b> | 初始止损: <b>{initial_stop}</b></p>",
            f"<p>偏离: {deviation_atr:.2f}ATR | ER(20): {er20:.2f}",
        ]
        if er40 is not None:
            lines[-1] += f" | ER(40): {er40:.2f}"
        lines[-1] += "</p>"

        if ema10 is not None:
            lines.append(f"<p>EMA10: {ema10:.1f} / EMA20: {ema20:.1f} / EMA120: {ema120:.1f}</p>")

        # 仓位加码原因
        reasons = []
        if position_multiplier >= 2:
            if scenario == 1:
                if er40 is not None and er40 >= 0.42:
                    reasons.append(f"ER(40)={er40:.2f}>=0.42")
                if er5_delta_6 is not None and er5_delta_6 >= 0.50:
                    reasons.append(f"ER5变化(6根)={er5_delta_6:.2f}>=0.50")
            elif scenario == 2:
                if er40_delta_12 is not None and er40_delta_12 >= 0.14:
                    reasons.append(f"ER40变化(12根)={er40_delta_12:.2f}>=0.14")
        if reasons:
            lines.append(f"<p>仓位: {position_multiplier}x ({', '.join(reasons)})</p>")
        else:
            lines.append(f"<p>仓位: {position_multiplier}x</p>")

        content = "\n".join(lines)
        return self._send(title, content)

    # ================================================================
    #  2. 止损挪动通知
    # ================================================================
    def notify_stop_moved(self, sym_key: str, direction: str,
                          strategy: str, old_stop: float, new_stop: float,
                          current_price: float):
        """
        title(工单名称): PVC空 止损↓ 5473→5420
        content第一行(设备名称): 现价5398 策略S2
        """
        now = time.time()
        last = self._stop_last_sent.get(sym_key, 0)
        if now - last < self.stop_cooldown:
            logger.debug(f"止损推送限流: {sym_key} ({now - last:.0f}s < {self.stop_cooldown}s)")
            return False

        name = self._sym_label(sym_key)
        dl = self._dir_label(direction)

        if direction == "long":
            arrow = "↑" if new_stop > old_stop else "↓"
        else:
            arrow = "↓" if new_stop < old_stop else "↑"

        title = f"【止损】{name}{dl} {arrow}{old_stop:.0f}→{new_stop:.0f}"

        content = (f"现价{current_price:.0f} 策略{strategy}\n<hr>"
                   f"<p>品种: {name} ({sym_key}) {dl}</p>"
                   f"<p>策略: {strategy}</p>"
                   f"<p>止损: {old_stop} → <b>{new_stop}</b></p>"
                   f"<p>当前价: {current_price}</p>")

        self._stop_last_sent[sym_key] = now
        return self._send(title, content)

    # ================================================================
    #  3. 平仓通知
    # ================================================================
    def notify_position_closed(self, sym_key: str, direction: str,
                               entry_price: float, exit_price: float,
                               pnl_pct: float, exit_strategy: str,
                               exit_reason: str, bars_held: int,
                               scenario: int):
        """
        title(工单名称): PVC空 平仓+2.3% S2
        content第一行(设备名称): 入5407出5281 持仓15根~150分钟
        """
        name = self._sym_label(sym_key)
        dl = self._dir_label(direction)
        sign = "+" if pnl_pct >= 0 else ""

        title = f"【平仓】{name}{dl} {sign}{pnl_pct:.1f}% {exit_strategy}"

        summary_line = f"入{entry_price:.0f}出{exit_price:.0f} 持仓{bars_held}根~{bars_held*10}分钟"

        content = (f"{summary_line}\n<hr>"
                   f"<h3>{name}{dl} 平仓 {sign}{pnl_pct:.1f}%</h3>"
                   f"<p>场景: {scenario} | 策略: {exit_strategy}</p>"
                   f"<p>入场: {entry_price} → 出场: <b>{exit_price}</b></p>"
                   f"<p>持仓: {bars_held}根K线 (~{bars_held*10}分钟)</p>"
                   f"<p>出场原因: {exit_reason}</p>")

        # 平仓后清除该品种的止损限流记录
        self._stop_last_sent.pop(sym_key, None)
        return self._send(title, content)

    # ================================================================
    #  4. 系统事件通知
    # ================================================================
    def notify_system_event(self, event: str, details: str = ""):
        """
        title: 监控启动 32品种 | 断线重连成功
        """
        return self._send(event, details)
