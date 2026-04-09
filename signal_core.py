# -*- coding: utf-8 -*-
"""
信号检测 + 出场追踪 共享核心模块
=================================
回测引擎和实时监控共用此模块，保证信号完全一致。

策略：10min B类回调 >= 4根K线收回EMA10入场
趋势：EMA20 > EMA120（多头），EMA20 < EMA120（空头）
出场：S1 当根新高追踪 + S2 回调追踪 + S3 前根新高追踪
"""

from dataclasses import dataclass, field
from typing import Optional, List


# ============================================================
#  品种配置（tick_size 因品种不同）
# ============================================================

SYMBOL_CONFIGS = {
    # 贵金属 & 能化（已回测验证）
    "SHFE.au": {"tick_size": 0.02, "multiplier": 1000, "name": "黄金"},
    "SHFE.ag": {"tick_size": 1.0,  "multiplier": 15,   "name": "白银"},
    "INE.sc":  {"tick_size": 0.1,  "multiplier": 1000, "name": "原油"},
    # 有色金属
    "SHFE.cu": {"tick_size": 10,   "multiplier": 5,    "name": "铜"},
    "SHFE.al": {"tick_size": 5,    "multiplier": 5,    "name": "铝"},
    "SHFE.zn": {"tick_size": 5,    "multiplier": 5,    "name": "锌"},
    "SHFE.ni": {"tick_size": 10,   "multiplier": 1,    "name": "镍"},
    "INE.bc":  {"tick_size": 10,   "multiplier": 5,    "name": "国际铜"},
    # 黑色系
    "SHFE.rb": {"tick_size": 1,    "multiplier": 10,   "name": "螺纹钢"},
    "SHFE.hc": {"tick_size": 1,    "multiplier": 10,   "name": "热卷"},
    "DCE.i":   {"tick_size": 0.5,  "multiplier": 100,  "name": "铁矿石"},
    "DCE.jm":  {"tick_size": 0.5,  "multiplier": 60,   "name": "焦煤"},
    "DCE.j":   {"tick_size": 0.5,  "multiplier": 100,  "name": "焦炭"},
    # 能化
    "SHFE.bu": {"tick_size": 1,    "multiplier": 10,   "name": "沥青"},
    "SHFE.ru": {"tick_size": 5,    "multiplier": 10,   "name": "橡胶"},
    "SHFE.sp": {"tick_size": 2,    "multiplier": 10,   "name": "纸浆"},
    "SHFE.fu": {"tick_size": 1,    "multiplier": 10,   "name": "燃料油"},
    "INE.lu":  {"tick_size": 1,    "multiplier": 10,   "name": "低硫燃油"},
    "DCE.eg":  {"tick_size": 1,    "multiplier": 10,   "name": "乙二醇"},
    "DCE.pp":  {"tick_size": 1,    "multiplier": 5,    "name": "聚丙烯"},
    "DCE.l":   {"tick_size": 1,    "multiplier": 5,    "name": "塑料"},
    "DCE.pg":  {"tick_size": 1,    "multiplier": 20,   "name": "LPG"},
    "DCE.eb":  {"tick_size": 1,    "multiplier": 5,    "name": "苯乙烯"},
    "CZCE.TA": {"tick_size": 2,    "multiplier": 5,    "name": "PTA"},
    "CZCE.MA": {"tick_size": 1,    "multiplier": 10,   "name": "甲醇"},
    "CZCE.SA": {"tick_size": 1,    "multiplier": 20,   "name": "纯碱"},
    "CZCE.FG": {"tick_size": 1,    "multiplier": 20,   "name": "玻璃"},
    # 农产品
    "DCE.m":   {"tick_size": 1,    "multiplier": 10,   "name": "豆粕"},
    "DCE.y":   {"tick_size": 2,    "multiplier": 10,   "name": "豆油"},
    "DCE.p":   {"tick_size": 2,    "multiplier": 10,   "name": "棕榈油"},
    "CZCE.SR": {"tick_size": 1,    "multiplier": 10,   "name": "白糖"},
    "CZCE.CF": {"tick_size": 5,    "multiplier": 5,    "name": "棉花"},
    "CZCE.RM": {"tick_size": 1,    "multiplier": 10,   "name": "菜粕"},
    # 其他
    "GFEX.lc": {"tick_size": 50,   "multiplier": 1,    "name": "碳酸锂"},
    "DCE.lh":  {"tick_size": 5,    "multiplier": 16,   "name": "生猪"},
    "DCE.v":   {"tick_size": 5,    "multiplier": 5,    "name": "PVC"},
}

# 默认止损跳数（120天32品种验证，5跳最优）
DEFAULT_STOP_TICKS = 5


# ============================================================
#  数据结构
# ============================================================

@dataclass
class Signal:
    """入场信号"""
    direction: str          # "long" or "short"
    signal_type: str        # "A" 影线弹回 / "B" 1~3根回调 / "C" >=4根回调
    entry_price: float      # 入场价（收盘价）
    pullback_extreme: float # 回调极值（多头=最低价，空头=最高价）
    pullback_bars: int      # 回调K线数（A类=0）
    signal_seq: int         # 本轮趋势第几个信号
    bar_index: int          # 触发K线的index


@dataclass
class ExitEvent:
    """出场事件"""
    strategy: str           # "S1", "S2", "S3"
    exit_price: float
    exit_reason: str        # "stop" or "backtest_end"
    bars_held: int
    pnl_pct: float          # 盈亏百分比


@dataclass
class StopUpdate:
    """止损更新事件"""
    strategy: str           # "S1", "S2", "S3"
    old_stop: float
    new_stop: float


# ============================================================
#  SignalDetector — 逐K线信号检测状态机
# ============================================================

class SignalDetector:
    """
    A/B/C 三类信号检测状态机，逐K线调用process_bar()。

    趋势判断：ema20 > ema120 → 多头，反之空头。
    信号分类：
      A类：影线碰EMA10弹回（close不跌破，单根）
      B类：实体跌破EMA10后1~3根收回
      C类：实体跌破EMA10后>=4根收回
    """

    # B/C分界（回调根数 < bc_boundary → B类，>= bc_boundary → C类）
    BC_BOUNDARY = 4

    def __init__(self, signal_types: str = 'ABC'):
        """
        参数:
          signal_types: 要检测的信号类型，如 'ABC'、'C'、'BC' 等
        """
        self.signal_types = signal_types.upper()

        # 状态变量
        self.trend_dir: int = 0           # 当前趋势方向 1=多头 -1=空头 0=未知
        self.signal_count: int = 0        # 本轮趋势信号计数
        self.below_ma_start: int = -1     # B/C类回调开始的bar_index
        self.pullback_low: Optional[float] = None   # 多头回调期间最低价
        self.pullback_high: Optional[float] = None   # 空头回调期间最高价

        # 前一根K线数据
        self._prev_close: Optional[float] = None
        self._prev_ema10: Optional[float] = None

        # K线计数器
        self._bar_index: int = -1

    def reset(self):
        """完全重置状态"""
        self.__init__(signal_types=self.signal_types)

    def to_dict(self) -> dict:
        """序列化为dict（用于监控状态持久化）"""
        return {
            'signal_types': self.signal_types,
            'trend_dir': self.trend_dir,
            'signal_count': self.signal_count,
            'below_ma_start': self.below_ma_start,
            'pullback_low': self.pullback_low,
            'pullback_high': self.pullback_high,
            '_prev_close': self._prev_close,
            '_prev_ema10': self._prev_ema10,
            '_bar_index': self._bar_index,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'SignalDetector':
        """从dict恢复（断线重连时使用）"""
        det = cls.__new__(cls)
        det.signal_types = d['signal_types']
        det.trend_dir = d['trend_dir']
        det.signal_count = d['signal_count']
        det.below_ma_start = d['below_ma_start']
        det.pullback_low = d['pullback_low']
        det.pullback_high = d['pullback_high']
        det._prev_close = d['_prev_close']
        det._prev_ema10 = d['_prev_ema10']
        det._bar_index = d['_bar_index']
        return det

    def process_bar(self, close: float, high: float, low: float,
                    ema10: float, ema20: float, ema120: float) -> Optional[Signal]:
        """
        处理一根已收盘K线，返回信号或None。

        参数:
          close, high, low: 当根K线价格
          ema10: EMA(10)，用于信号检测（回调穿越）
          ema20: EMA(20)，用于趋势判断
          ema120: EMA(120)，用于趋势判断
        """
        self._bar_index += 1
        signal = None

        # 第一根K线只记录prev，不检测
        if self._prev_close is None:
            self._prev_close = close
            self._prev_ema10 = ema10
            return None

        # 趋势方向：EMA20 > EMA120
        if ema20 > ema120:
            curr_trend = 1
        elif ema20 < ema120:
            curr_trend = -1
        else:
            curr_trend = 0

        # 趋势翻转：重置计数器
        if curr_trend != self.trend_dir and curr_trend != 0:
            self.trend_dir = curr_trend
            self.signal_count = 0
            self.below_ma_start = -1
            self.pullback_low = None
            self.pullback_high = None

        if self.trend_dir == 0:
            self._prev_close = close
            self._prev_ema10 = ema10
            return None

        # ===== 多头信号检测 =====
        if self.trend_dir == 1:
            # A类：影线碰EMA10弹回（无正在进行的回调时）
            if 'A' in self.signal_types and self.below_ma_start == -1:
                if (low <= ema10 and close > ema10
                        and self._prev_close > self._prev_ema10):
                    self.signal_count += 1
                    signal = Signal(
                        direction='long',
                        signal_type='A',
                        entry_price=close,
                        pullback_extreme=low,
                        pullback_bars=0,
                        signal_seq=self.signal_count,
                        bar_index=self._bar_index,
                    )

            # B/C类：实体跌破EMA10后收回
            if self.below_ma_start == -1:
                if close < ema10 and self._prev_close >= self._prev_ema10:
                    self.below_ma_start = self._bar_index
                    self.pullback_low = low
            else:
                self.pullback_low = min(self.pullback_low, low)
                if close > ema10:
                    pb_bars = self._bar_index - self.below_ma_start
                    if pb_bars >= 1:
                        sig_type = 'B' if pb_bars < self.BC_BOUNDARY else 'C'
                        if sig_type in self.signal_types:
                            self.signal_count += 1
                            signal = Signal(
                                direction='long',
                                signal_type=sig_type,
                                entry_price=close,
                                pullback_extreme=self.pullback_low,
                                pullback_bars=pb_bars,
                                signal_seq=self.signal_count,
                                bar_index=self._bar_index,
                            )
                    self.below_ma_start = -1
                    self.pullback_low = None

        # ===== 空头信号检测 =====
        elif self.trend_dir == -1:
            # A类：影线碰EMA10弹回（无正在进行的回调时）
            if 'A' in self.signal_types and self.below_ma_start == -1:
                if (high >= ema10 and close < ema10
                        and self._prev_close < self._prev_ema10):
                    self.signal_count += 1
                    signal = Signal(
                        direction='short',
                        signal_type='A',
                        entry_price=close,
                        pullback_extreme=high,
                        pullback_bars=0,
                        signal_seq=self.signal_count,
                        bar_index=self._bar_index,
                    )

            # B/C类：实体跌破EMA10后收回
            if self.below_ma_start == -1:
                if close > ema10 and self._prev_close <= self._prev_ema10:
                    self.below_ma_start = self._bar_index
                    self.pullback_high = high
            else:
                self.pullback_high = max(self.pullback_high, high)
                if close < ema10:
                    pb_bars = self._bar_index - self.below_ma_start
                    if pb_bars >= 1:
                        sig_type = 'B' if pb_bars < self.BC_BOUNDARY else 'C'
                        if sig_type in self.signal_types:
                            self.signal_count += 1
                            signal = Signal(
                                direction='short',
                                signal_type=sig_type,
                                entry_price=close,
                                pullback_extreme=self.pullback_high,
                                pullback_bars=pb_bars,
                                signal_seq=self.signal_count,
                                bar_index=self._bar_index,
                            )
                    self.below_ma_start = -1
                    self.pullback_high = None

        # 更新prev
        self._prev_close = close
        self._prev_ema10 = ema10
        return signal


# ============================================================
#  ExitTracker — 逐K线出场追踪状态机（S1/S2/S3）
# ============================================================

class ExitTracker:
    """
    追踪单个持仓的出场方式：
      S1.1: 当根新高追踪止损，收盘价触发（close创新高时，止损更新为当根low - N跳）
      S2: 回调追踪止损，盘中触发（回调完成站回EMA10时，止损更新为回调最低点 - N跳）
      S3.1: 前根新高追踪止损，收盘价触发（close创新高时，止损更新为前根low - N跳）
      S5.1: S3.1兜底(收盘触损)+S2接管(盘中触损)

    初始止损统一为：回调极值 - N跳（N = stop_ticks）
    """

    def __init__(self, direction: str, entry_price: float,
                 pullback_extreme: float, tick_size: float,
                 stop_ticks: int = DEFAULT_STOP_TICKS):
        self.direction = direction
        self.entry_price = entry_price
        self.is_long = (direction == 'long')
        self.tick_size = tick_size
        self.stop_ticks = stop_ticks
        self.tick = tick_size * stop_ticks

        # 初始止损统一：回调极值 - N跳
        if self.is_long:
            init_stop = pullback_extreme - self.tick
        else:
            init_stop = pullback_extreme + self.tick

        # S1.1 当根新高追踪止损，收盘价触发
        self.s11_stop = init_stop
        self.s11_done = False
        self.s11_bars = 0

        # S2 回调追踪止损，盘中触发
        self.s2_stop = init_stop
        self.s2_done = False
        self.s2_bars = 0
        self.s2_state = 'normal'  # 'normal' or 'pullback'
        self.s2_tracking_extreme = None

        # S3.1 前根新高追踪止损，收盘价触发
        self.s31_stop = init_stop
        self.s31_done = False
        self.s31_bars = 0

        # S5.1 S3.1兜底(收盘触损)+S2接管(盘中触损)
        self.s51_stop = init_stop
        self.s51_done = False
        self.s51_bars = 0
        self.s51_mode = 's3.1'  # 's3.1' or 's2'
        self.s51_s2_state = 'normal'
        self.s51_s2_tracking_extreme = None

        # 上一次的止损值（用于检测止损移动）
        self._prev_s2_stop = self.s2_stop
        self._prev_s51_stop = self.s51_stop

    def all_done(self) -> bool:
        return self.s11_done and self.s2_done and self.s31_done and self.s51_done

    def process_bar(self, close: float, high: float, low: float,
                    ema10: float, prev_close: float,
                    prev_high: float = None, prev_low: float = None
                    ) -> tuple[List[ExitEvent], List[StopUpdate]]:
        """
        处理一根已收盘K线。

        参数:
          close, high, low: 当根K线
          ema10: 当根EMA10（S2回调判断用）
          prev_close: 前根收盘价（新高判断用）
          prev_high, prev_low: 前根高低点（S3用）

        返回:
          (exit_events, stop_updates)
        """
        exits: List[ExitEvent] = []
        stop_updates: List[StopUpdate] = []

        self._prev_s2_stop = self.s2_stop
        self._prev_s51_stop = self.s51_stop

        # ===== S1.1 当根新高追踪止损，收盘价触发 =====
        if not self.s11_done:
            self.s11_bars += 1

            if self.is_long and close <= self.s11_stop:
                self.s11_done = True
                pnl = (close - self.entry_price) / self.entry_price * 100
                exits.append(ExitEvent(
                    strategy='S1.1', exit_price=close,
                    exit_reason='stop', bars_held=self.s11_bars, pnl_pct=pnl,
                ))
            elif not self.is_long and close >= self.s11_stop:
                self.s11_done = True
                pnl = (self.entry_price - close) / self.entry_price * 100
                exits.append(ExitEvent(
                    strategy='S1.1', exit_price=close,
                    exit_reason='stop', bars_held=self.s11_bars, pnl_pct=pnl,
                ))
            else:
                # 追踪逻辑与S1完全一样
                if self.is_long and close > prev_close:
                    candidate = low - self.tick
                    self.s11_stop = max(self.s11_stop, candidate)
                elif not self.is_long and close < prev_close:
                    candidate = high + self.tick
                    self.s11_stop = min(self.s11_stop, candidate)

        # ===== S2 回调追踪止损 =====
        if not self.s2_done:
            self.s2_bars += 1

            if self.is_long and low <= self.s2_stop:
                self.s2_done = True
                pnl = (self.s2_stop - self.entry_price) / self.entry_price * 100
                exits.append(ExitEvent(
                    strategy='S2', exit_price=self.s2_stop,
                    exit_reason='stop', bars_held=self.s2_bars, pnl_pct=pnl,
                ))
            elif not self.is_long and high >= self.s2_stop:
                self.s2_done = True
                pnl = (self.entry_price - self.s2_stop) / self.entry_price * 100
                exits.append(ExitEvent(
                    strategy='S2', exit_price=self.s2_stop,
                    exit_reason='stop', bars_held=self.s2_bars, pnl_pct=pnl,
                ))
            else:
                # 状态机：回调完成时更新止损为回调最低点 - N跳
                if self.is_long:
                    if self.s2_state == 'normal':
                        if close < ema10:
                            self.s2_state = 'pullback'
                            self.s2_tracking_extreme = low
                    elif self.s2_state == 'pullback':
                        self.s2_tracking_extreme = min(self.s2_tracking_extreme, low)
                        if close > ema10:
                            candidate = self.s2_tracking_extreme - self.tick
                            self.s2_stop = max(self.s2_stop, candidate)
                            self.s2_state = 'normal'
                            self.s2_tracking_extreme = None
                else:  # short
                    if self.s2_state == 'normal':
                        if close > ema10:
                            self.s2_state = 'pullback'
                            self.s2_tracking_extreme = high
                    elif self.s2_state == 'pullback':
                        self.s2_tracking_extreme = max(self.s2_tracking_extreme, high)
                        if close < ema10:
                            candidate = self.s2_tracking_extreme + self.tick
                            self.s2_stop = min(self.s2_stop, candidate)
                            self.s2_state = 'normal'
                            self.s2_tracking_extreme = None

        # ===== S3.1 前根新高追踪止损，收盘价触发 =====
        if not self.s31_done:
            self.s31_bars += 1

            if self.is_long and close <= self.s31_stop:
                self.s31_done = True
                pnl = (close - self.entry_price) / self.entry_price * 100
                exits.append(ExitEvent(
                    strategy='S3.1', exit_price=close,
                    exit_reason='stop', bars_held=self.s31_bars, pnl_pct=pnl,
                ))
            elif not self.is_long and close >= self.s31_stop:
                self.s31_done = True
                pnl = (self.entry_price - close) / self.entry_price * 100
                exits.append(ExitEvent(
                    strategy='S3.1', exit_price=close,
                    exit_reason='stop', bars_held=self.s31_bars, pnl_pct=pnl,
                ))
            else:
                if prev_low is not None:
                    if self.is_long and close > prev_close:
                        candidate = prev_low - self.tick
                        self.s31_stop = max(self.s31_stop, candidate)
                    elif not self.is_long and close < prev_close:
                        candidate = prev_high + self.tick
                        self.s31_stop = min(self.s31_stop, candidate)

        # ===== S5.1 S3.1兜底(收盘触损) + S2接管(盘中触损) =====
        if not self.s51_done:
            self.s51_bars += 1

            # 止损触发：s3.1模式用close，s2模式用high/low
            hit_stop = False
            if self.s51_mode == 's3.1':
                if self.is_long and close <= self.s51_stop:
                    hit_stop = True
                elif not self.is_long and close >= self.s51_stop:
                    hit_stop = True
            else:  # s2 mode
                if self.is_long and low <= self.s51_stop:
                    hit_stop = True
                elif not self.is_long and high >= self.s51_stop:
                    hit_stop = True

            if hit_stop:
                self.s51_done = True
                if self.s51_mode == 's3.1':
                    # 收盘触损，出场价=close
                    exit_price = close
                else:
                    # 盘中触损，出场价=stop
                    exit_price = self.s51_stop
                if self.is_long:
                    pnl = (exit_price - self.entry_price) / self.entry_price * 100
                else:
                    pnl = (self.entry_price - exit_price) / self.entry_price * 100
                exits.append(ExitEvent(
                    strategy='S5.1', exit_price=exit_price,
                    exit_reason='stop', bars_held=self.s51_bars, pnl_pct=pnl,
                ))
            else:
                # S2回调追踪（始终在跑）
                s2_updated = False
                if self.is_long:
                    if self.s51_s2_state == 'normal':
                        if close < ema10:
                            self.s51_s2_state = 'pullback'
                            self.s51_s2_tracking_extreme = low
                    elif self.s51_s2_state == 'pullback':
                        self.s51_s2_tracking_extreme = min(self.s51_s2_tracking_extreme, low)
                        if close > ema10:
                            candidate = self.s51_s2_tracking_extreme - self.tick
                            if candidate > self.s51_stop:
                                self.s51_stop = candidate
                                s2_updated = True
                            self.s51_s2_state = 'normal'
                            self.s51_s2_tracking_extreme = None
                else:
                    if self.s51_s2_state == 'normal':
                        if close > ema10:
                            self.s51_s2_state = 'pullback'
                            self.s51_s2_tracking_extreme = high
                    elif self.s51_s2_state == 'pullback':
                        self.s51_s2_tracking_extreme = max(self.s51_s2_tracking_extreme, high)
                        if close < ema10:
                            candidate = self.s51_s2_tracking_extreme + self.tick
                            if candidate < self.s51_stop:
                                self.s51_stop = candidate
                                s2_updated = True
                            self.s51_s2_state = 'normal'
                            self.s51_s2_tracking_extreme = None

                if s2_updated:
                    self.s51_mode = 's2'

                # S3.1追踪（仅在s3.1模式下更新止损）
                if self.s51_mode == 's3.1':
                    if prev_low is not None:
                        if self.is_long and close > prev_close:
                            candidate = prev_low - self.tick
                            self.s51_stop = max(self.s51_stop, candidate)
                        elif not self.is_long and close < prev_close:
                            candidate = prev_high + self.tick
                            self.s51_stop = min(self.s51_stop, candidate)

        # 检测止损移动
        if not self.s2_done and self.s2_stop != self._prev_s2_stop:
            stop_updates.append(StopUpdate(
                strategy='S2', old_stop=self._prev_s2_stop, new_stop=self.s2_stop,
            ))
        if not self.s51_done and self.s51_stop != self._prev_s51_stop:
            stop_updates.append(StopUpdate(
                strategy='S5.1', old_stop=self._prev_s51_stop, new_stop=self.s51_stop,
            ))

        return exits, stop_updates

    def force_close(self, close_price: float) -> List[ExitEvent]:
        """强制平仓（回测结束或监控关闭时）"""
        exits = []
        for strategy, done, bars in [
            ('S1.1', self.s11_done, self.s11_bars),
            ('S2', self.s2_done, self.s2_bars),
            ('S3.1', self.s31_done, self.s31_bars),
            ('S5.1', self.s51_done, self.s51_bars),
        ]:
            if not done:
                if self.is_long:
                    pnl = (close_price - self.entry_price) / self.entry_price * 100
                else:
                    pnl = (self.entry_price - close_price) / self.entry_price * 100
                exits.append(ExitEvent(
                    strategy=strategy, exit_price=close_price,
                    exit_reason='backtest_end', bars_held=bars, pnl_pct=pnl,
                ))
        self.s11_done = True
        self.s2_done = True
        self.s31_done = True
        self.s51_done = True
        return exits

    def to_dict(self) -> dict:
        """序列化为dict（用于positions.json持久化）"""
        return {
            'direction': self.direction,
            'entry_price': self.entry_price,
            'is_long': self.is_long,
            'tick_size': self.tick_size,
            'stop_ticks': self.stop_ticks,
            'tick': self.tick,
            's11_stop': self.s11_stop, 's11_done': self.s11_done, 's11_bars': self.s11_bars,
            's2_stop': self.s2_stop, 's2_done': self.s2_done, 's2_bars': self.s2_bars,
            's2_state': self.s2_state, 's2_tracking_extreme': self.s2_tracking_extreme,
            's31_stop': self.s31_stop, 's31_done': self.s31_done, 's31_bars': self.s31_bars,
            's51_stop': self.s51_stop, 's51_done': self.s51_done, 's51_bars': self.s51_bars,
            's51_mode': self.s51_mode,
            's51_s2_state': self.s51_s2_state, 's51_s2_tracking_extreme': self.s51_s2_tracking_extreme,
            '_prev_s51_stop': self._prev_s51_stop,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'ExitTracker':
        """从dict恢复（断线重连时使用）"""
        tracker = cls.__new__(cls)
        tracker.direction = d['direction']
        tracker.entry_price = d['entry_price']
        tracker.is_long = d['is_long']
        tracker.tick_size = d['tick_size']
        tracker.stop_ticks = d.get('stop_ticks', DEFAULT_STOP_TICKS)
        tracker.tick = d['tick']
        tracker.s11_stop = d['s11_stop']
        tracker.s11_done = d['s11_done']
        tracker.s11_bars = d['s11_bars']
        tracker.s2_stop = d['s2_stop']
        tracker.s2_done = d['s2_done']
        tracker.s2_bars = d['s2_bars']
        tracker.s2_state = d['s2_state']
        tracker.s2_tracking_extreme = d['s2_tracking_extreme']
        tracker.s31_stop = d['s31_stop']
        tracker.s31_done = d['s31_done']
        tracker.s31_bars = d['s31_bars']
        tracker.s51_stop = d['s51_stop']
        tracker.s51_done = d['s51_done']
        tracker.s51_bars = d['s51_bars']
        tracker.s51_mode = d['s51_mode']
        tracker.s51_s2_state = d['s51_s2_state']
        tracker.s51_s2_tracking_extreme = d['s51_s2_tracking_extreme']
        tracker._prev_s2_stop = d['s2_stop']
        tracker._prev_s51_stop = d.get('_prev_s51_stop', d['s51_stop'])
        return tracker


# ============================================================
#  场景分类（全局共享）
# ============================================================

def classify_scenario(sig_type: str, er20: float, deviation_atr: float):
    """
    判断信号属于哪个可操作场景，返回场景编号 1/2/3 或 None。

    场景1: A类 + ER(20)>=0.5 + 偏离>=1.0ATR  → 用S2出场
    场景2: C类 + 偏离>=2.0ATR                 → 用S2出场
    场景3: B类 + ER(20)>=0.5 + 偏离0.1~0.3ATR → 用S5.1出场

    全局过滤: ER(20)>=0.7 正期望消失，不开仓。
    """
    if er20 >= 0.7:
        return None
    if sig_type == 'A' and er20 >= 0.5 and deviation_atr >= 1.0:
        return 1
    if sig_type == 'C' and deviation_atr >= 2.0:
        return 2
    if sig_type == 'B' and er20 >= 0.5 and 0.1 <= deviation_atr < 0.3:
        return 3
    return None


# 场景对应的出场策略和名称
SCENARIO_EXIT = {1: 'S2', 2: 'S2', 3: 'S5.1'}
SCENARIO_PNL_COL = {1: 's2_pnl', 2: 's2_pnl', 3: 's51_pnl'}
SCENARIO_REASON_COL = {1: 's2_reason', 2: 's2_reason', 3: 's51_reason'}
SCENARIO_NAMES = {
    1: '场景1: A类+ER≥0.5+偏离≥1.0ATR → S2',
    2: '场景2: C类+偏离≥2.0ATR → S2',
    3: '场景3: B类+ER≥0.5+偏离0.1~0.3ATR → S5.1',
}
