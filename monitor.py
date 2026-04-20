# -*- coding: utf-8 -*-
"""
核心监测引擎
============
TqSdk 实时订阅 → 信号检测 → 出场追踪 → 微信推送
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime

from tqsdk import TqApi, TqAuth

# 为 "行情监控" 项目提供实时 parquet（独立 latest/ 子目录，不覆盖 170d 历史）
_LATEST_RAW_COLS = ("datetime", "open", "high", "low", "close", "volume", "open_oi", "close_oi")
_LATEST_PERF_THRESHOLD_MS = 1000     # 单次 to_parquet 超过此值打 WARN
_LATEST_LOG_EVERY = 100              # 每 N 次成功写盘打一次 INFO 汇总（避免日志刷屏）
_latest_write_counter = {"n": 0, "max_ms": 0.0}

from signal_core import (
    SignalDetector, ExitTracker, Signal,
    SYMBOL_CONFIGS, DEFAULT_STOP_TICKS,
    classify_scenario, SCENARIO_EXIT, SCENARIO_NAMES,
    Type1SignalDetector, Type1Signal, LadderRTracker,
    classify_type1_tier, TYPE1_TIER_NAMES,
    RejectObserver,
)


class _DBRejectObserver(RejectObserver):
    """把拒绝原因落到 signal_db.rejected_signals + DashboardState.reject_stream。"""

    def __init__(self, signal_db=None, dashboard_state=None):
        self.signal_db = signal_db
        self.dashboard_state = dashboard_state

    def emit(self, sym_key, stage, reason, context=None):
        ctx = context or {}
        try:
            if self.signal_db:
                self.signal_db.record_reject(
                    sym_key=sym_key,
                    bar_time=ctx.get("bar_time") or datetime.now().isoformat(),
                    stage=stage, reason=reason,
                    signal_type=ctx.get("signal_type"),
                    direction=ctx.get("direction"),
                    er20=ctx.get("er20"),
                    deviation_atr=ctx.get("deviation_atr"),
                    context=ctx,
                )
        except Exception:
            logger.exception("record_reject 失败")
        try:
            if self.dashboard_state:
                self.dashboard_state.push_event("reject", {
                    "sym_key": sym_key, "stage": stage,
                    "reason": reason, "context": ctx,
                })
        except Exception:
            pass
from data_loader import add_indicators, sym_name
from monitor_config import MonitorConfig
from notifier import PushPlusNotifier
from state_manager import StateManager
from contract_parser import get_sym_meta
try:
    from web.correlation_service import (
        compute_sector_exposure, detect_sector_warnings,
    )
    _HAS_CORRELATION = True
except Exception:
    _HAS_CORRELATION = False


# 出场策略 → ExitTracker 止损属性名
_EXIT_STOP_ATTR = {
    "S1.1": "s11_stop", "S2": "s2_stop", "S2.1": "s21_stop",
    "S3.1": "s31_stop", "S5.1": "s51_stop", "S5.2": "s52_stop",
    "S6": "s6_stop", "S6.1": "s61_stop",
}

logger = logging.getLogger(__name__)


class MonitorEngine:
    """实时监测引擎：订阅K线 → 检测信号 → 追踪出场 → 推送通知"""

    def __init__(self, config: MonitorConfig, dry_run: bool = False,
                 dashboard_state=None, signal_db=None):
        self.config = config
        self.dry_run = dry_run
        self.api = None
        self._running = False
        self.dashboard_state = dashboard_state
        self.signal_db = signal_db

        self.notifier = PushPlusNotifier(
            token=config.pushplus_token,
            stop_cooldown=config.stop_update_cooldown,
            dry_run=dry_run,
        )
        self.state_mgr = StateManager(config.state_file)

        # 每品种状态 — ABC系统
        self.detectors: dict[str, SignalDetector] = {}
        self.trackers: dict[str, ExitTracker] = {}
        self.tracker_meta: dict[str, dict] = {}

        # 每品种状态 — Type1系统（与ABC并行）
        self.t1_detectors: dict[str, Type1SignalDetector] = {}
        self.t1_trackers: dict[str, LadderRTracker] = {}
        self.t1_tracker_meta: dict[str, dict] = {}

        self.kline_serials: dict[str, object] = {}
        self.quotes: dict[str, object] = {}  # 主连quote（含underlying_symbol）
        self.bar_counts: dict[str, int] = {}
        # 每品种最后处理的K线时间戳（用于检测新K线）
        self.last_bar_dt: dict[str, pd.Timestamp] = {}

        # 定时保存
        self._last_save = 0
        self._save_interval = 300  # 5分钟

        # 拒绝埋点观察者（落到 DB + 推到仪表盘）
        self.reject_observer = _DBRejectObserver(
            signal_db=self.signal_db,
            dashboard_state=self.dashboard_state,
        )

    # ================================================================
    #  启动 & 关闭
    # ================================================================
    def start(self):
        """主入口：连接 → 订阅 → 恢复状态 → 主循环"""
        logger.info(f"监控引擎启动: {len(self.config.symbols)}个品种, "
                    f"dry_run={self.dry_run}")
        self._running = True

        try:
            self._connect()
            self._subscribe_all()
            self._restore_state()
            self._warmup_all()

            # 发送启动通知
            active = len(self.trackers)
            msg = f"监控启动 {len(self.config.symbols)}品种"
            if active > 0:
                msg += f" {active}个活跃持仓"
            self.notifier.notify_system_event(msg)

            self._run_loop()
        except KeyboardInterrupt:
            logger.info("收到中断信号")
        except Exception as e:
            logger.exception(f"引擎异常退出: {e}")
            self.notifier.notify_system_event(f"错误: {e}")
        finally:
            self.shutdown()

    def shutdown(self):
        """优雅关闭：保存状态 + 关闭API"""
        self._running = False
        logger.info("正在关闭...")
        self._save_state()
        if self.api:
            try:
                self.api.close()
            except:
                pass
            self.api = None
        logger.info("监控引擎已关闭")

    # ================================================================
    #  连接 & 订阅
    # ================================================================
    def _connect(self):
        """创建 TqApi 连接"""
        logger.info("连接天勤...")
        self.api = TqApi(auth=TqAuth(
            self.config.tq_account, self.config.tq_password
        ))
        logger.info("天勤连接成功")

    def _subscribe_all(self):
        """订阅全部品种的10分钟K线 + 主连quote（用于获取主力合约代码）"""
        for sym_key in self.config.symbols:
            tq_symbol = f"KQ.m@{sym_key}"
            try:
                serial = self.api.get_kline_serial(
                    tq_symbol,
                    duration_seconds=600,  # 10分钟
                    data_length=2000,      # 约35天，足够EMA120预热
                )
                self.kline_serials[sym_key] = serial
                # 订阅主连quote，underlying_symbol字段是当前主力合约代码
                self.quotes[sym_key] = self.api.get_quote(tq_symbol)
                logger.debug(f"已订阅: {sym_key}")
            except Exception as e:
                logger.error(f"订阅失败 {sym_key}: {e}")

        # 等待初始数据加载
        self.api.wait_update()
        logger.info(f"已订阅 {len(self.kline_serials)} 个品种")

    def _get_dominant_info(self, sym_key: str) -> tuple:
        """获取主力合约代码和实时价格，返回 (合约简称, 实时价) 或 (None, None)"""
        quote = self.quotes.get(sym_key)
        if quote is None:
            return None, None
        try:
            underlying = quote.underlying_symbol  # 如 'SHFE.ag2506'
            last_price = quote.last_price
            if underlying and last_price == last_price:  # NaN check
                # 取简称: 'SHFE.ag2506' -> 'ag2506'
                short_name = underlying.split('.', 1)[-1] if '.' in underlying else underlying
                return short_name, last_price
        except Exception:
            pass
        return None, None

    # ================================================================
    #  状态恢复 & 预热
    # ================================================================
    def _restore_state(self):
        """从JSON恢复检测器和追踪器状态"""
        state = self.state_mgr.load()
        if state is None:
            # 冷启动：为每个品种创建新检测器
            for sym_key in self.config.symbols:
                self.detectors[sym_key] = SignalDetector(signal_types='ABC')
                self.detectors[sym_key].sym_key = sym_key
                self.detectors[sym_key].reject_observer = self.reject_observer
                self.t1_detectors[sym_key] = Type1SignalDetector()
            return

        # 恢复ABC检测器
        for sym_key, det_dict in state.get("detectors", {}).items():
            if sym_key in self.config.symbols:
                self.detectors[sym_key] = SignalDetector.from_dict(det_dict)

        # 恢复ABC活跃持仓
        for sym_key, pos in state.get("active_positions", {}).items():
            if sym_key in self.config.symbols:
                self.trackers[sym_key] = ExitTracker.from_dict(pos["tracker"])
                meta = pos["meta"]
                # 老状态可能没有 MFE/MAE 累加器，按 entry_price 补齐
                entry_p = float(meta.get('entry_price', 0) or 0)
                if 'running_max_high' not in meta and entry_p:
                    meta['running_max_high'] = entry_p
                if 'running_min_low' not in meta and entry_p:
                    meta['running_min_low'] = entry_p
                # initial_stop 若缺失，从 tracker 推回
                if 'initial_stop' not in meta:
                    tk = self.trackers[sym_key]
                    meta['initial_stop'] = getattr(tk, 's6_stop', None) or \
                        getattr(tk, 's2_stop', None) or entry_p
                self.tracker_meta[sym_key] = meta

        # 恢复Type1检测器
        for sym_key, det_dict in state.get("t1_detectors", {}).items():
            if sym_key in self.config.symbols:
                self.t1_detectors[sym_key] = Type1SignalDetector.from_dict(det_dict)

        # 恢复Type1活跃持仓
        for sym_key, pos in state.get("t1_positions", {}).items():
            if sym_key in self.config.symbols:
                self.t1_trackers[sym_key] = LadderRTracker.from_dict(pos["tracker"])
                self.t1_tracker_meta[sym_key] = pos["meta"]

        # 恢复进度
        self.bar_counts = state.get("bar_counts", {})

        # 补充未恢复的品种
        for sym_key in self.config.symbols:
            if sym_key not in self.detectors:
                self.detectors[sym_key] = SignalDetector(signal_types='ABC')
            if sym_key not in self.t1_detectors:
                self.t1_detectors[sym_key] = Type1SignalDetector()

        # 给所有 ABC 检测器注入拒绝观察者 + sym_key
        for sym_key, det in self.detectors.items():
            det.sym_key = sym_key
            det.reject_observer = self.reject_observer

        t1_active = len(self.t1_trackers)
        logger.info(f"状态恢复: {len(self.detectors)}个检测器, "
                    f"{len(self.trackers)}个ABC持仓, {t1_active}个Type1持仓")

    def _warmup_all(self):
        """
        冷启动时，用历史K线喂入detector到当前位置。
        热恢复时，只处理断线期间新增的K线。
        """
        for sym_key in self.config.symbols:
            if sym_key not in self.kline_serials:
                continue
            df = self._get_dataframe(sym_key)
            if df is None or len(df) < 200:
                logger.warning(f"{sym_key}: 数据不足，跳过预热")
                continue

            prev_count = self.bar_counts.get(sym_key, 0)
            n = len(df) - 1  # 最后一行是未完成的当前K线

            if prev_count >= n:
                continue  # 已经是最新的

            if prev_count == 0:
                # 冷启动：只预热最后500根（detector状态几十根就稳定）
                warmup_start = max(0, n - 500)
                logger.info(f"{sym_key}: 冷启动预热 {n - warmup_start} 根K线（跳过前{warmup_start}根）")
                detector = self.detectors[sym_key]
                t1_det = self.t1_detectors.get(sym_key)
                cfg_sym = SYMBOL_CONFIGS[sym_key]
                ts = cfg_sym['tick_size']
                for i in range(warmup_start, n):
                    row = df.iloc[i]
                    detector.process_bar(
                        row['close'], row['high'], row['low'],
                        row['ema10'], row['ema20'], row['ema120'],
                    )
                    if t1_det:
                        t1_det.process_bar(
                            close=row['close'], high=row['high'],
                            low=row['low'], opn=float(row.get('open', 0) or 0),
                            ema10=row['ema10'],
                            ema60=float(row.get('ema60', 0) or 0),
                            er20=float(row.get('er_20', 0) or 0),
                            er40=float(row.get('er_40', 0) or 0),
                            atr=float(row.get('atr', 0) or 0),
                            tick_size=ts,
                        )
                        t1_det.pending = None
                self.bar_counts[sym_key] = n
            else:
                # 热恢复：只处理新增K线（断线期间的）
                gap = n - prev_count
                logger.info(f"{sym_key}: 热恢复，处理 {gap} 根新K线...")
                for i in range(prev_count, n):
                    row = df.iloc[i]
                    prev_row = df.iloc[i - 1] if i > 0 else None
                    self._process_bar(sym_key, df, i)
                self.bar_counts[sym_key] = n

            # 记录最后已完成K线的时间戳
            self.last_bar_dt[sym_key] = df.iloc[n - 1]['datetime']

        self._save_state()
        logger.info("预热完成")

    # ================================================================
    #  主循环
    # ================================================================
    def _run_loop(self):
        """核心事件循环"""
        logger.info("进入主循环...")
        if self.dashboard_state:
            self.dashboard_state.update_engine_status("running")

        while self._running:
            try:
                deadline = time.time() + 30
                self.api.wait_update(deadline=deadline)
            except Exception as e:
                logger.error(f"wait_update 异常: {e}")
                if self._running:
                    self._reconnect()
                continue

            # 检查每个品种是否有新K线
            for sym_key in list(self.kline_serials.keys()):
                serial = self.kline_serials[sym_key]
                try:
                    if self.api.is_changing(serial):
                        self._process_new_bars(sym_key)
                except Exception as e:
                    logger.error(f"处理 {sym_key} 异常: {e}", exc_info=True)

            # 更新仪表盘趋势数据
            if self.dashboard_state:
                trends = {k: d.trend_dir for k, d in self.detectors.items()}
                self.dashboard_state.update_trends(trends)

            # 板块集中度聚合（10s 节流）
            now = time.time()
            if (self.dashboard_state and _HAS_CORRELATION and self.signal_db
                    and (now - getattr(self, '_last_sector_calc', 0)) > 10):
                try:
                    open_sigs = self.signal_db.get_open_signals()
                    exposure = compute_sector_exposure(
                        open_sigs, self.dashboard_state.candidate_pool
                    )
                    warnings = detect_sector_warnings(exposure)
                    self.dashboard_state.update_sector_exposure(
                        exposure, warnings=warnings
                    )
                except Exception as e:
                    logger.warning(f"sector_exposure 计算失败: {e}")
                self._last_sector_calc = now

            # 定时保存状态
            if now - self._last_save > self._save_interval:
                self._save_state()
                self._last_save = now

    # ================================================================
    #  K线处理
    # ================================================================
    def _get_dataframe(self, sym_key: str) -> pd.DataFrame:
        """将 TqSdk kline serial 转为带指标的 DataFrame"""
        serial = self.kline_serials[sym_key]
        df = serial.copy()
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ns')
        df = df.dropna(subset=['close']).reset_index(drop=True)

        if len(df) < 130:
            return None

        # 计算指标（复用 data_loader）
        df = add_indicators(df, emas=(5, 10, 20, 60, 120),
                            er_periods=(5, 20, 40), atr_period=14)

        # 仓位计算需要的变化量
        if 'er_5' in df.columns:
            df['er5_delta_6'] = df['er_5'] - df['er_5'].shift(6)
        if 'er_40' in df.columns:
            df['er40_delta_12'] = df['er_40'] - df['er_40'].shift(12)

        return df

    def _process_new_bars(self, sym_key: str):
        """检查并处理新完成的K线。

        优化：先用原始时间戳判断是否有新bar，只在有新bar时才算指标。
        避免每个tick都重算全部EMA/ER/ATR。
        """
        # 快速检查：有没有新完成的K线（不算指标）
        serial = self.kline_serials[sym_key]
        raw_dt = pd.to_datetime(serial['datetime'], unit='ns')
        # serial最后一行是未完成K线，倒数第二行是最后已完成K线
        n_raw = len(serial.dropna(subset=['close']))
        if n_raw < 2:
            return
        last_completed_dt = raw_dt.iloc[n_raw - 2]
        prev_dt = self.last_bar_dt.get(sym_key)

        if prev_dt is not None and last_completed_dt <= prev_dt:
            return  # 没有新bar，跳过（这里省掉了95%的无用计算）

        # 确认有新bar，才算指标
        df = self._get_dataframe(sym_key)
        if df is None:
            return

        n = len(df) - 1
        if n < 1:
            return

        last_completed_dt = df.iloc[n - 1]['datetime']
        prev_dt = self.last_bar_dt.get(sym_key)

        if prev_dt is not None and last_completed_dt <= prev_dt:
            return  # 没有新完成的K线

        if prev_dt is None:
            # 首次：只处理最后一根已完成K线
            self._process_bar(sym_key, df, n - 1)
        else:
            # 找到上次处理位置之后的所有新K线
            for i in range(n - 1, -1, -1):
                if df.iloc[i]['datetime'] <= prev_dt:
                    # 从 i+1 到 n-1 是新完成的K线
                    for j in range(i + 1, n):
                        self._process_bar(sym_key, df, j)
                    break

        self.last_bar_dt[sym_key] = last_completed_dt
        self.bar_counts[sym_key] = n

        # ======================================================
        # 为"行情监控"项目回写最新序列到 latest/ 子目录
        # 原 170d 历史缓存不动；latest 只有当前 serial 长度（约 35 天）
        # 行情监控的 watchdog/轮询会识别 mtime 变化触发阶段事件检测
        # ------------------------------------------------------
        # 只做附带动作，任何异常都吞掉不能影响 qiquan 主循环
        # ======================================================
        try:
            self._write_latest_parquet(sym_key, df)
        except Exception as e:
            logger.warning(f"[latest-parquet] {sym_key} 写盘失败: {type(e).__name__}: {e}")

    def _write_latest_parquet(self, sym_key: str, df: pd.DataFrame):
        """把当前 serial 的原始 OHLCV 写到 CACHE_DIR/latest/{safe}_{period}min.parquet。

        约定：
          - 只写 RAW_COLS 8 列（去掉 qiquan 内部算的 EMA/ER/ATR 指标，节省 IO）
          - datetime 保持 naive UTC（qiquan 内部就是 UTC；行情监控 load_latest 会再 +8h 转 CST）
          - 非原子覆盖写（pandas 默认 to_parquet）——读端要 try/except 容忍半截文件
        """
        from data_cache import CACHE_DIR  # qiquan/data_cache.py 定义
        latest_dir = os.path.join(CACHE_DIR, "latest")
        os.makedirs(latest_dir, exist_ok=True)

        safe = sym_key.replace(".", "_")
        # 假设周期 10min（qiquan 主路径就是 10min；如果扩展多周期后改这里）
        path = os.path.join(latest_dir, f"{safe}_10min.parquet")

        cols = [c for c in _LATEST_RAW_COLS if c in df.columns]
        raw = df[cols].copy()

        t0 = time.perf_counter()
        raw.to_parquet(path, index=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        _latest_write_counter["n"] += 1
        if elapsed_ms > _latest_write_counter["max_ms"]:
            _latest_write_counter["max_ms"] = elapsed_ms

        if elapsed_ms > _LATEST_PERF_THRESHOLD_MS:
            logger.warning(
                f"[latest-parquet] {sym_key} to_parquet 耗时 {elapsed_ms:.1f}ms "
                f"(> {_LATEST_PERF_THRESHOLD_MS}ms 阈值)"
            )

        if _latest_write_counter["n"] % _LATEST_LOG_EVERY == 0:
            logger.info(
                f"[latest-parquet] 累计写盘 {_latest_write_counter['n']} 次, "
                f"max={_latest_write_counter['max_ms']:.1f}ms"
            )

    def _process_bar(self, sym_key: str, df: pd.DataFrame, idx: int):
        """处理单根已完成的K线"""
        row = df.iloc[idx]
        prev_row = df.iloc[idx - 1] if idx > 0 else None

        close = row['close']
        high = row['high']
        low = row['low']
        ema5 = row.get('ema5')
        ema10 = row['ema10']
        ema20 = row['ema20']
        ema120 = row['ema120']

        # 1. 信号检测
        detector = self.detectors[sym_key]
        signal = detector.process_bar(close, high, low, ema10, ema20, ema120)

        # 2. 出场追踪（如果有活跃持仓）
        if sym_key in self.trackers:
            tracker = self.trackers[sym_key]
            meta = self.tracker_meta[sym_key]
            exit_strategy = meta['exit_strategy']

            # MFE/MAE 累加（在 tracker 更新前拿到最新 high/low）
            try:
                cur_max = meta.get('running_max_high', meta['entry_price'])
                cur_min = meta.get('running_min_low', meta['entry_price'])
                meta['running_max_high'] = max(cur_max, float(high))
                meta['running_min_low'] = min(cur_min, float(low))
            except Exception:
                pass

            prev_close = prev_row['close'] if prev_row is not None else close
            prev_high = prev_row['high'] if prev_row is not None else high
            prev_low = prev_row['low'] if prev_row is not None else low

            exit_events, stop_updates = tracker.process_bar(
                close=close, high=high, low=low, ema10=ema10,
                prev_close=prev_close, prev_high=prev_high, prev_low=prev_low,
                ema5=ema5,
            )

            # 止损挪动通知
            for su in stop_updates:
                if su.strategy == exit_strategy or (
                    exit_strategy == 'S5.1' and su.strategy in ('S2', 'S3.1')
                ):
                    dom_c, dom_p = self._get_dominant_info(sym_key)
                    self.notifier.notify_stop_moved(
                        sym_key=sym_key,
                        direction=tracker.direction,
                        strategy=su.strategy,
                        old_stop=su.old_stop,
                        new_stop=su.new_stop,
                        current_price=close,
                        dominant_contract=dom_c,
                        dominant_price=dom_p,
                    )
                    now_str = datetime.now().isoformat()
                    if self.signal_db and 'signal_id' in meta:
                        self.signal_db.record_stop_update(
                            meta['signal_id'], su.old_stop,
                            su.new_stop, su.strategy, now_str)
                    if self.dashboard_state:
                        self.dashboard_state.push_event('stop_update', {
                            'sym_key': sym_key,
                            'old_stop': su.old_stop, 'new_stop': su.new_stop,
                            'strategy': su.strategy, 'current_price': close,
                        })

            # 平仓通知
            for ev in exit_events:
                if ev.strategy == exit_strategy:
                    dom_c, dom_p = self._get_dominant_info(sym_key)
                    self.notifier.notify_position_closed(
                        sym_key=sym_key,
                        direction=tracker.direction,
                        entry_price=meta['entry_price'],
                        exit_price=ev.exit_price,
                        pnl_pct=ev.pnl_pct,
                        exit_strategy=exit_strategy,
                        exit_reason=ev.exit_reason,
                        bars_held=ev.bars_held,
                        scenario=meta['scenario'],
                        dominant_contract=dom_c,
                        dominant_price=dom_p,
                    )
                    now_str = datetime.now().isoformat()
                    # 计算 MFE/MAE（R 单位）
                    _mfe_r = _mae_r = None
                    try:
                        entry = float(meta['entry_price'])
                        init_stop = float(meta.get('initial_stop', 0) or 0)
                        dist = abs(entry - init_stop)
                        if dist > 0:
                            mh = float(meta.get('running_max_high', entry))
                            ml = float(meta.get('running_min_low', entry))
                            if tracker.direction == 'long':
                                _mfe_r = (mh - entry) / dist
                                _mae_r = (ml - entry) / dist
                            else:
                                _mfe_r = (entry - ml) / dist
                                _mae_r = (entry - mh) / dist
                    except Exception:
                        _mfe_r = _mae_r = None
                    if self.signal_db and 'signal_id' in meta:
                        self.signal_db.record_exit(
                            meta['signal_id'], ev.exit_price,
                            ev.exit_reason, ev.pnl_pct,
                            ev.bars_held, now_str,
                            mfe_r=_mfe_r, mae_r=_mae_r)
                    if self.dashboard_state:
                        self.dashboard_state.remove_position(sym_key)
                        if 'signal_id' in meta:
                            self.dashboard_state.remove_portfolio_live(
                                meta['signal_id'])
                        self.dashboard_state.push_event('position_closed', {
                            'sym_key': sym_key, 'pnl_pct': ev.pnl_pct,
                            'exit_price': ev.exit_price,
                        })
                    # 移除已平仓的持仓
                    del self.trackers[sym_key]
                    del self.tracker_meta[sym_key]
                    self._save_state()
                    logger.info(f"{sym_key} 平仓: {exit_strategy} "
                               f"pnl={ev.pnl_pct:+.2f}%")
                    break

        # 3. 新信号处理（仅在无活跃持仓时）
        if signal is not None and sym_key not in self.trackers:
            self._on_new_signal(sym_key, signal, row)

        # ============ Type1 系统（独立 try/except，不影响ABC） ============
        try:
            t1_det = self.t1_detectors.get(sym_key)
            if t1_det is None:
                return

            cfg_sym = SYMBOL_CONFIGS[sym_key]
            ts = cfg_sym['tick_size']
            er40_val = float(row.get('er_40', 0) or 0)

            # 4. Type1 挂单成交检查
            if t1_det.pending is not None and sym_key not in self.t1_trackers:
                fill_result = t1_det.check_fill(
                    high=high, low=low, opn=float(row.get('open', 0) or 0))
                if fill_result:
                    if fill_result['status'] == 'filled':
                        self._on_type1_fill(sym_key, fill_result['signal'], row)
                    elif fill_result['status'] in ('expired', 'gap_skip'):
                        logger.debug(f"{sym_key} Type1挂单{fill_result['status']}")

            # 5. Type1 出场追踪
            if sym_key in self.t1_trackers:
                ev = self.t1_trackers[sym_key].process_bar(close, high, low)
                if ev:
                    self._on_type1_exit(sym_key, ev)

            # 6. Type1 信号检测（始终喂数据保持状态，有持仓时忽略信号）
            t1_signal = t1_det.process_bar(
                close=close, high=high, low=low,
                opn=float(row.get('open', 0) or 0),
                ema10=ema10, ema60=float(row.get('ema60', 0) or 0),
                er20=float(row.get('er_20', 0) or 0),
                er40=er40_val,
                atr=float(row.get('atr', 0) or 0),
                tick_size=ts,
            )
            if t1_signal and sym_key not in self.t1_trackers:
                self._on_type1_signal(sym_key, t1_signal, row)
            elif sym_key in self.t1_trackers:
                # 有持仓时清除pending，不开新单
                t1_det.pending = None
        except Exception as e:
            logger.error(f"{sym_key} Type1处理异常: {e}", exc_info=True)

        # ============ Phase B2: 仪表盘状态回写 ============
        try:
            self._update_dashboard_bar(sym_key, df, idx)
        except Exception as e:
            logger.warning(f"{sym_key} dashboard 状态回写失败: {e}")

    # ================================================================
    #  Phase B2: Dashboard 状态回写
    # ================================================================
    def _update_dashboard_bar(self, sym_key: str, df: pd.DataFrame, idx: int):
        """每根已完成 K 线后把 symbol_states / heatmap / candidate_pool /
        portfolio_live 同步到 DashboardState。

        所有写入都是幂等 dict.update，不改引擎语义；异常时仅告警不崩溃。
        """
        if not self.dashboard_state:
            return

        row = df.iloc[idx]
        close = float(row['close'])
        ema5 = float(row.get('ema5', 0) or 0)
        ema10 = float(row['ema10'])
        ema20 = float(row['ema20'])
        ema60 = float(row.get('ema60', 0) or 0)
        ema120 = float(row['ema120'])
        er20 = float(row.get('er_20', 0) or 0)
        er40 = float(row.get('er_40', 0) or 0)
        atr = float(row.get('atr', 0) or 0)
        deviation_atr = abs(close - ema10) / atr if atr > 0 else 0.0
        bar_time = str(row['datetime'])

        det = self.detectors.get(sym_key)
        if det is None:
            return
        meta = get_sym_meta(sym_key)
        has_pos = sym_key in self.trackers

        # 1) symbol_states — 合约查询的数据底
        self.dashboard_state.update_symbol_state(sym_key, {
            "sym_name": meta.get("name"),
            "sector": meta.get("sector"),
            "last_price": close,
            "last_er20": er20,
            "last_er40": er40,
            "last_atr": atr,
            "deviation_atr": deviation_atr,
            "ema5": ema5, "ema10": ema10, "ema20": ema20,
            "ema60": ema60, "ema120": ema120,
            "last_ema_snapshot": det.last_ema_snapshot,
            "pullback_stage": det.pullback_stage,
            "pullback_bar_count": det.pullback_bar_count,
            "last_pullback_extreme": det.last_pullback_extreme,
            "candidate_scenario": det.candidate_scenario,
            "trend_dir": det.trend_dir,
            "has_position": has_pos,
            "bar_time": bar_time,
        })
        self.dashboard_state.update_bar_time(sym_key, bar_time)

        # 2) symbol_heatmap cell
        start = max(0, idx - 19)
        sparkline = [float(c) for c in df['close'].iloc[start:idx + 1].tolist()]
        self.dashboard_state.update_heatmap_cell(sym_key, {
            "sym_key": sym_key,
            "sym_name": meta.get("name"),
            "sector": meta.get("sector"),
            "trend_dir": det.trend_dir,
            "er20": er20,
            "deviation_atr": deviation_atr,
            "last_price": close,
            "sparkline_20": sparkline,
            "has_candidate": det.candidate_scenario is not None,
            "has_position": has_pos,
            "bar_time": bar_time,
        })

        # 3) candidate_pool — peek 下一 bar/当 bar 可入场的预触发
        tick = meta.get("tick_size", 1) or 1
        try:
            cand = det.peek_candidate(close)
        except Exception:
            cand = None
        if cand:
            trigger = float(cand.get("trigger_price") or close)
            # distance_ticks：负值 = 已突破，需回调
            diff_ticks = int(round((trigger - close) / tick)) if tick else None
            if det.trend_dir == 1:
                # 多头：trigger <= close（回到 EMA10）→ distance 负
                distance = diff_ticks
            else:
                distance = -diff_ticks if diff_ticks is not None else None
            self.dashboard_state.set_candidate(sym_key, {
                "sym_key": sym_key,
                "sym_name": meta.get("name"),
                "sector": meta.get("sector"),
                "kind": cand.get("kind"),
                "sig_type": cand.get("sig_type"),
                "trigger_price": trigger,
                "current_price": close,
                "distance_ticks": abs(diff_ticks) if diff_ticks is not None else None,
                "direction_bias": "long" if det.trend_dir == 1 else "short",
                "candidate_scenario": det.candidate_scenario,
                "pullback_bars": cand.get("pullback_bars"),
                "pullback_extreme": cand.get("pullback_extreme"),
                "er20": er20, "deviation_atr": deviation_atr,
                "bar_time": bar_time,
            })
        else:
            if sym_key in self.dashboard_state.candidate_pool:
                self.dashboard_state.set_candidate(sym_key, None)

        # 4) portfolio_live — 当前持仓的 R 倍数 / 止损
        if has_pos:
            tr = self.trackers[sym_key]
            tmeta = self.tracker_meta.get(sym_key, {})
            exit_strategy = tmeta.get('exit_strategy', 'S6')
            signal_id = tmeta.get('signal_id')
            try:
                cur_r = tr.current_r(close, exit_strategy=exit_strategy)
            except Exception:
                cur_r = None
            stop_attr = _EXIT_STOP_ATTR.get(exit_strategy, "s6_stop")
            current_stop = getattr(tr, stop_attr, None)
            if signal_id is not None:
                self.dashboard_state.update_portfolio_live(signal_id, {
                    "signal_id": signal_id,
                    "sym_key": sym_key,
                    "sym_name": meta.get("name"),
                    "direction": tr.direction,
                    "entry_price": tmeta.get('entry_price'),
                    "initial_stop": tmeta.get('initial_stop'),
                    "current_price": close,
                    "current_stop": current_stop,
                    "current_r": cur_r,
                    "exit_strategy": exit_strategy,
                    "bar_time": bar_time,
                })

    # ================================================================
    #  信号处理
    # ================================================================
    def _on_new_signal(self, sym_key: str, signal: Signal, row):
        """处理新信号：场景分类 → 仓位计算 → 创建追踪器 → 推送"""
        er20 = float(row.get('er_20', 0) or 0)
        atr = float(row.get('atr', 0) or 0)

        if atr <= 0:
            return

        deviation_atr = abs(signal.entry_price - row['ema10']) / atr

        # 场景分类（带方向过滤：场景2空头已停做）
        scenario = classify_scenario(signal.signal_type, er20, deviation_atr,
                                     direction=signal.direction)
        if scenario is None:
            return

        # ER(20) 硬性过滤（classify_scenario 已包含 >=0.7 过滤）
        # 场景1和3需要 ER(20)>=0.5，classify_scenario 已处理

        # 仓位计算
        position_multiplier = 1
        er40 = float(row.get('er_40', 0) or 0)
        er5_delta_6 = float(row.get('er5_delta_6', 0) or 0)
        er40_delta_12 = float(row.get('er40_delta_12', 0) or 0)

        if scenario == 1:
            # 场景1过滤：ER5变化(6根) <= -0.41 → 跳过
            if er5_delta_6 <= -0.41:
                logger.debug(f"{sym_key} 场景1信号跳过: er5_delta_6={er5_delta_6:.2f}")
                return
            # 加仓：ER(40)>=0.42 或 ER5变化(6根)>=0.50
            if er40 >= 0.42 or er5_delta_6 >= 0.50:
                position_multiplier = 2
        elif scenario == 2:
            # 加仓：ER40变化(12根)>=0.14
            if er40_delta_12 >= 0.14:
                position_multiplier = 2
        # 场景3：固定1x

        # 出场策略
        exit_strategy = SCENARIO_EXIT[scenario]

        # 创建 ExitTracker
        cfg_sym = SYMBOL_CONFIGS[sym_key]
        tracker = ExitTracker(
            direction=signal.direction,
            entry_price=signal.entry_price,
            pullback_extreme=signal.pullback_extreme,
            tick_size=cfg_sym['tick_size'],
            stop_ticks=DEFAULT_STOP_TICKS,
            ema5_strategies=(exit_strategy in ('S6', 'S6.1')),
        )
        # 初始止损价（先算出来，用于 MFE/MAE 的 R 单位）
        _tick_mfe = cfg_sym['tick_size'] * DEFAULT_STOP_TICKS
        if signal.direction == 'long':
            _init_stop_for_meta = signal.pullback_extreme - _tick_mfe
        else:
            _init_stop_for_meta = signal.pullback_extreme + _tick_mfe

        self.trackers[sym_key] = tracker
        self.tracker_meta[sym_key] = {
            'scenario': scenario,
            'signal_type': signal.signal_type,
            'entry_time': datetime.now().isoformat(),
            'position_multiplier': position_multiplier,
            'exit_strategy': exit_strategy,
            'entry_price': signal.entry_price,
            'initial_stop': _init_stop_for_meta,
            # MFE/MAE 累加器：入场以 entry_price 为基线
            'running_max_high': float(signal.entry_price),
            'running_min_low': float(signal.entry_price),
        }

        # 初始止损价
        tick = cfg_sym['tick_size'] * DEFAULT_STOP_TICKS
        if signal.direction == 'long':
            initial_stop = signal.pullback_extreme - tick
        else:
            initial_stop = signal.pullback_extreme + tick

        # 推送通知
        dom_c, dom_p = self._get_dominant_info(sym_key)
        self.notifier.notify_new_signal(
            sym_key=sym_key,
            direction=signal.direction,
            signal_type=signal.signal_type,
            pullback_bars=signal.pullback_bars,
            entry_price=signal.entry_price,
            initial_stop=initial_stop,
            scenario=scenario,
            deviation_atr=deviation_atr,
            er20=er20,
            position_multiplier=position_multiplier,
            exit_strategy=exit_strategy,
            er40=er40,
            er5_delta_6=er5_delta_6,
            er40_delta_12=er40_delta_12,
            ema10=float(row['ema10']),
            ema20=float(row['ema20']),
            ema120=float(row['ema120']),
            dominant_contract=dom_c,
            dominant_price=dom_p,
        )

        # 记录到数据库
        if self.signal_db:
            signal_id = self.signal_db.record_entry(
                sym_key=sym_key, sym_name=sym_name(sym_key),
                direction=signal.direction,
                signal_type=signal.signal_type,
                scenario=scenario, entry_price=signal.entry_price,
                initial_stop=initial_stop,
                pullback_bars=signal.pullback_bars,
                deviation_atr=deviation_atr, er20=er20,
                position_multiplier=position_multiplier,
                exit_strategy=exit_strategy,
                entry_time=self.tracker_meta[sym_key]['entry_time'],
            )
            self.tracker_meta[sym_key]['signal_id'] = signal_id

        # 推送到仪表盘
        if self.dashboard_state:
            self.dashboard_state.update_position(sym_key, {
                'sym_name': sym_name(sym_key),
                'direction': signal.direction,
                'entry_price': signal.entry_price,
                'current_stop': initial_stop,
                'exit_strategy': exit_strategy,
                'mode': 'S3.1兜底' if exit_strategy == 'S5.1' else '正常',
                'scenario': scenario,
                'position_multiplier': position_multiplier,
                'bars_held': 0,
            })
            self.dashboard_state.push_event('new_signal', {
                'sym_key': sym_key, 'sym_name': sym_name(sym_key),
                'direction': signal.direction,
                'signal_type': signal.signal_type,
                'scenario': scenario, 'entry_price': signal.entry_price,
            })

        self._save_state()
        logger.info(f"新信号: {sym_key} {signal.direction} {signal.signal_type} "
                    f"场景{scenario} 入{signal.entry_price} 止{initial_stop} "
                    f"{position_multiplier}x {exit_strategy}")

    # ================================================================
    #  Type1 信号处理
    # ================================================================

    def _on_type1_signal(self, sym_key: str, signal: Type1Signal, row):
        """Type1 新信号：分级 → 推送（挂单等成交）"""
        tier, preset = classify_type1_tier(
            signal.stop_dist_atr, signal.recent_win_n,
            signal.er_40, signal.signal_density)

        if preset is None:
            # γ，不做
            logger.debug(f"{sym_key} Type1信号γ跳过: "
                        f"stop_atr={signal.stop_dist_atr} hot={signal.recent_win_n} "
                        f"er40={signal.er_40} density={signal.signal_density}")
            # 清除pending防止成交
            self.t1_detectors[sym_key].pending = None
            return

        # 推送挂单信号
        tier_name = TYPE1_TIER_NAMES.get(tier, tier)
        dom_c, dom_p = self._get_dominant_info(sym_key)
        self.notifier.notify_type1_signal(
            sym_key=sym_key,
            direction=signal.direction,
            pending_price=signal.pending_price,
            stop_price=signal.stop_price,
            tier=tier_name,
            preset=preset,
            stop_dist_atr=signal.stop_dist_atr,
            er_40=signal.er_40,
            recent_win_n=signal.recent_win_n,
            dominant_contract=dom_c,
            dominant_price=dom_p,
        )

        logger.info(f"Type1信号: {sym_key} {signal.direction} {tier_name} "
                    f"挂单{signal.pending_price} 止损{signal.stop_price}")

    def _on_type1_fill(self, sym_key: str, signal: Type1Signal, row):
        """Type1 挂单成交：创建 LadderRTracker"""
        tier, preset = classify_type1_tier(
            signal.stop_dist_atr, signal.recent_win_n,
            signal.er_40, signal.signal_density)

        if preset is None:
            return

        cfg_sym = SYMBOL_CONFIGS[sym_key]
        tracker = LadderRTracker(
            direction=signal.direction,
            entry_price=signal.pending_price,
            stop_price=signal.stop_price,
            tick_size=cfg_sym['tick_size'],
            preset=preset,
        )

        self.t1_trackers[sym_key] = tracker
        self.t1_tracker_meta[sym_key] = {
            'tier': tier,
            'preset': preset,
            'entry_time': datetime.now().isoformat(),
            'entry_price': signal.pending_price,
            'stop_price': signal.stop_price,
            'direction': signal.direction,
        }

        tier_name = TYPE1_TIER_NAMES.get(tier, tier)
        dom_c, dom_p = self._get_dominant_info(sym_key)
        self.notifier.notify_type1_fill(
            sym_key=sym_key,
            direction=signal.direction,
            entry_price=signal.pending_price,
            stop_price=signal.stop_price,
            tier=tier_name,
            preset=preset,
            dominant_contract=dom_c,
            dominant_price=dom_p,
        )

        if self.dashboard_state:
            self.dashboard_state.update_position(f't1_{sym_key}', {
                'sym_name': f'[T1] {sym_name(sym_key)}',
                'direction': signal.direction,
                'entry_price': signal.pending_price,
                'current_stop': signal.stop_price,
                'exit_strategy': f'LR_{preset}',
                'mode': tier_name,
                'scenario': f'T1-{tier}',
                'position_multiplier': 1,
                'bars_held': 0,
            })

        self._save_state()
        logger.info(f"Type1成交: {sym_key} {signal.direction} {tier_name} "
                    f"入{signal.pending_price} LR_{preset}")

    def _on_type1_exit(self, sym_key: str, ev):
        """Type1 出场"""
        meta = self.t1_tracker_meta.get(sym_key, {})
        tier = meta.get('tier', '?')
        tier_name = TYPE1_TIER_NAMES.get(tier, tier)

        # 记录结果用于滚动胜率
        t1_det = self.t1_detectors.get(sym_key)
        if t1_det:
            win = ev.pnl_pct > 0
            t1_det.record_trade_result(win)

        dom_c, dom_p = self._get_dominant_info(sym_key)
        self.notifier.notify_type1_exit(
            sym_key=sym_key,
            direction=meta.get('direction', '?'),
            entry_price=meta.get('entry_price', 0),
            exit_price=ev.exit_price,
            pnl_pct=ev.pnl_pct,
            exit_reason=ev.exit_reason,
            bars_held=ev.bars_held,
            tier=tier_name,
            preset=meta.get('preset', '?'),
            dominant_contract=dom_c,
            dominant_price=dom_p,
        )

        if self.dashboard_state:
            self.dashboard_state.remove_position(f't1_{sym_key}')

        self.t1_trackers.pop(sym_key, None)
        self.t1_tracker_meta.pop(sym_key, None)
        self._save_state()
        logger.info(f"Type1平仓: {sym_key} {tier_name} pnl={ev.pnl_pct:+.2f}% "
                    f"{ev.exit_reason} {ev.bars_held}根")

    # ================================================================
    #  重连
    # ================================================================
    def _reconnect(self):
        """断线重连（指数退避）"""
        if self.dashboard_state:
            self.dashboard_state.update_engine_status("reconnecting")
        self._save_state()
        if self.api:
            try:
                self.api.close()
            except:
                pass
            self.api = None

        for attempt in range(self.config.max_reconnect_attempts):
            delay = min(
                self.config.reconnect_delay_base * (2 ** attempt),
                300,
            )
            logger.info(f"重连中... {delay}s后重试 (第{attempt+1}次)")
            time.sleep(delay)

            try:
                self._connect()
                self._subscribe_all()
                # 热恢复：处理断线期间的新K线
                self._warmup_all()
                self.notifier.notify_system_event(
                    f"重连成功 (第{attempt+1}次)")
                return
            except Exception as e:
                logger.error(f"重连失败: {e}")

        self.notifier.notify_system_event(
            f"重连失败 已达最大重试次数({self.config.max_reconnect_attempts})")
        self._running = False

    # ================================================================
    #  状态保存
    # ================================================================
    def _save_state(self):
        """保存全部状态到 JSON"""
        try:
            self.state_mgr.save(
                detectors=self.detectors,
                trackers=self.trackers,
                tracker_meta=self.tracker_meta,
                bar_counts=self.bar_counts,
                t1_detectors=self.t1_detectors,
                t1_trackers=self.t1_trackers,
                t1_tracker_meta=self.t1_tracker_meta,
            )
            self._last_save = time.time()
        except Exception as e:
            logger.error(f"保存状态失败: {e}")
