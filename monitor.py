# -*- coding: utf-8 -*-
"""
核心监测引擎
============
TqSdk 实时订阅 → 信号检测 → 出场追踪 → 微信推送
"""

import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime

from tqsdk import TqApi, TqAuth

from signal_core import (
    SignalDetector, ExitTracker, Signal,
    SYMBOL_CONFIGS, DEFAULT_STOP_TICKS,
    classify_scenario, SCENARIO_EXIT, SCENARIO_NAMES,
)
from data_loader import add_indicators, sym_name
from monitor_config import MonitorConfig
from notifier import PushPlusNotifier
from state_manager import StateManager

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

        # 每品种状态
        self.detectors: dict[str, SignalDetector] = {}
        self.trackers: dict[str, ExitTracker] = {}
        self.tracker_meta: dict[str, dict] = {}
        self.kline_serials: dict[str, object] = {}
        self.bar_counts: dict[str, int] = {}
        # 每品种最后处理的K线时间戳（用于检测新K线）
        self.last_bar_dt: dict[str, pd.Timestamp] = {}

        # 定时保存
        self._last_save = 0
        self._save_interval = 300  # 5分钟

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
        """订阅全部品种的10分钟K线"""
        for sym_key in self.config.symbols:
            tq_symbol = f"KQ.m@{sym_key}"
            try:
                serial = self.api.get_kline_serial(
                    tq_symbol,
                    duration_seconds=600,  # 10分钟
                    data_length=8964,      # 最大历史K线数
                )
                self.kline_serials[sym_key] = serial
                logger.debug(f"已订阅: {sym_key}")
            except Exception as e:
                logger.error(f"订阅失败 {sym_key}: {e}")

        # 等待初始数据加载
        self.api.wait_update()
        logger.info(f"已订阅 {len(self.kline_serials)} 个品种")

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
            return

        # 恢复检测器
        for sym_key, det_dict in state.get("detectors", {}).items():
            if sym_key in self.config.symbols:
                self.detectors[sym_key] = SignalDetector.from_dict(det_dict)

        # 恢复活跃持仓
        for sym_key, pos in state.get("active_positions", {}).items():
            if sym_key in self.config.symbols:
                self.trackers[sym_key] = ExitTracker.from_dict(pos["tracker"])
                self.tracker_meta[sym_key] = pos["meta"]

        # 恢复进度
        self.bar_counts = state.get("bar_counts", {})

        # 补充未恢复的品种
        for sym_key in self.config.symbols:
            if sym_key not in self.detectors:
                self.detectors[sym_key] = SignalDetector(signal_types='ABC')

        logger.info(f"状态恢复: {len(self.detectors)}个检测器, "
                    f"{len(self.trackers)}个活跃持仓")

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
                # 冷启动：喂入全部历史
                logger.info(f"{sym_key}: 冷启动预热 {n} 根K线...")
                detector = self.detectors[sym_key]
                for i in range(n):
                    row = df.iloc[i]
                    detector.process_bar(
                        row['close'], row['high'], row['low'],
                        row['ema10'], row['ema20'], row['ema120'],
                    )
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

            # 定时保存状态
            now = time.time()
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
        df = add_indicators(df, emas=(10, 20, 120),
                            er_periods=(5, 20, 40), atr_period=14)

        # 仓位计算需要的变化量
        if 'er_5' in df.columns:
            df['er5_delta_6'] = df['er_5'] - df['er_5'].shift(6)
        if 'er_40' in df.columns:
            df['er40_delta_12'] = df['er_40'] - df['er_40'].shift(12)

        return df

    def _process_new_bars(self, sym_key: str):
        """检查并处理新完成的K线。

        TqSdk的kline_serial是固定长度滑动窗口(data_length=8964)，
        新K线完成后总行数不变，不能用len(df)判断。
        改用最后已完成K线的时间戳来检测新K线。
        """
        df = self._get_dataframe(sym_key)
        if df is None:
            return

        n = len(df) - 1  # 最后一行是未完成K线（当前K线）
        if n < 1:
            return

        # 最后一根已完成K线的时间戳
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

    def _process_bar(self, sym_key: str, df: pd.DataFrame, idx: int):
        """处理单根已完成的K线"""
        row = df.iloc[idx]
        prev_row = df.iloc[idx - 1] if idx > 0 else None

        close = row['close']
        high = row['high']
        low = row['low']
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

            prev_close = prev_row['close'] if prev_row is not None else close
            prev_high = prev_row['high'] if prev_row is not None else high
            prev_low = prev_row['low'] if prev_row is not None else low

            exit_events, stop_updates = tracker.process_bar(
                close=close, high=high, low=low, ema10=ema10,
                prev_close=prev_close, prev_high=prev_high, prev_low=prev_low,
            )

            # 止损挪动通知
            for su in stop_updates:
                if su.strategy == exit_strategy or (
                    exit_strategy == 'S5.1' and su.strategy in ('S2', 'S3.1')
                ):
                    self.notifier.notify_stop_moved(
                        sym_key=sym_key,
                        direction=tracker.direction,
                        strategy=su.strategy,
                        old_stop=su.old_stop,
                        new_stop=su.new_stop,
                        current_price=close,
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
                    )
                    now_str = datetime.now().isoformat()
                    if self.signal_db and 'signal_id' in meta:
                        self.signal_db.record_exit(
                            meta['signal_id'], ev.exit_price,
                            ev.exit_reason, ev.pnl_pct,
                            ev.bars_held, now_str)
                    if self.dashboard_state:
                        self.dashboard_state.remove_position(sym_key)
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

        # 场景分类
        scenario = classify_scenario(signal.signal_type, er20, deviation_atr)
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
        )
        self.trackers[sym_key] = tracker
        self.tracker_meta[sym_key] = {
            'scenario': scenario,
            'signal_type': signal.signal_type,
            'entry_time': datetime.now().isoformat(),
            'position_multiplier': position_multiplier,
            'exit_strategy': exit_strategy,
            'entry_price': signal.entry_price,
        }

        # 初始止损价
        tick = cfg_sym['tick_size'] * DEFAULT_STOP_TICKS
        if signal.direction == 'long':
            initial_stop = signal.pullback_extreme - tick
        else:
            initial_stop = signal.pullback_extreme + tick

        # 推送通知
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
            )
            self._last_save = time.time()
        except Exception as e:
            logger.error(f"保存状态失败: {e}")
