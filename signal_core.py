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
      S2.1: 同S2，止损触发改为收盘价穿止损
      S3.1: 前根新高追踪止损，收盘价触发（close创新高时，止损更新为前根low - N跳）
      S5.1: S3.1兜底(收盘触损)+S2接管(盘中触损)
      S5.2: S3.1兜底(收盘触损)+S2.1接管(收盘触损)
      S6: 同S2，回调检测用EMA5代替EMA10（更紧跟踪），盘中触发
      S6.1: 同S6，止损触发改为收盘价穿止损

    初始止损统一为：回调极值 - N跳（N = stop_ticks）
    """

    def __init__(self, direction: str, entry_price: float,
                 pullback_extreme: float, tick_size: float,
                 stop_ticks: int = DEFAULT_STOP_TICKS,
                 ema5_strategies: bool = False):
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

        # S2.1 同S2，收盘价触发
        self.s21_stop = init_stop
        self.s21_done = False
        self.s21_bars = 0
        self.s21_state = 'normal'
        self.s21_tracking_extreme = None

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

        # S5.2 S3.1兜底(收盘触损)+S2.1接管(收盘触损)
        self.s52_stop = init_stop
        self.s52_done = False
        self.s52_bars = 0
        self.s52_mode = 's3.1'  # 's3.1' or 's2.1'
        self.s52_s21_state = 'normal'
        self.s52_s21_tracking_extreme = None

        # S6 EMA5回调追踪止损，盘中触发
        self.s6_stop = init_stop
        self.s6_done = not ema5_strategies  # 未启用时直接标完成
        self.s6_bars = 0
        self.s6_state = 'normal'
        self.s6_tracking_extreme = None

        # S6.1 EMA5回调追踪止损，收盘价触发
        self.s61_stop = init_stop
        self.s61_done = not ema5_strategies
        self.s61_bars = 0
        self.s61_state = 'normal'
        self.s61_tracking_extreme = None

        # 上一次的止损值（用于检测止损移动）
        self._prev_s2_stop = self.s2_stop
        self._prev_s21_stop = self.s21_stop
        self._prev_s51_stop = self.s51_stop
        self._prev_s52_stop = self.s52_stop
        self._prev_s6_stop = self.s6_stop
        self._prev_s61_stop = self.s61_stop

    def all_done(self) -> bool:
        return (self.s11_done and self.s2_done and self.s21_done
                and self.s31_done and self.s51_done and self.s52_done
                and self.s6_done and self.s61_done)

    def process_bar(self, close: float, high: float, low: float,
                    ema10: float, prev_close: float,
                    prev_high: float = None, prev_low: float = None,
                    ema5: float = None,
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
        self._prev_s21_stop = self.s21_stop
        self._prev_s51_stop = self.s51_stop
        self._prev_s52_stop = self.s52_stop
        self._prev_s6_stop = self.s6_stop
        self._prev_s61_stop = self.s61_stop

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

        # ===== S2.1 同S2回调追踪，收盘价触发 =====
        if not self.s21_done:
            self.s21_bars += 1

            if self.is_long and close <= self.s21_stop:
                self.s21_done = True
                pnl = (close - self.entry_price) / self.entry_price * 100
                exits.append(ExitEvent(
                    strategy='S2.1', exit_price=close,
                    exit_reason='stop', bars_held=self.s21_bars, pnl_pct=pnl,
                ))
            elif not self.is_long and close >= self.s21_stop:
                self.s21_done = True
                pnl = (self.entry_price - close) / self.entry_price * 100
                exits.append(ExitEvent(
                    strategy='S2.1', exit_price=close,
                    exit_reason='stop', bars_held=self.s21_bars, pnl_pct=pnl,
                ))
            else:
                # 状态机与S2完全一致，用ema10
                if self.is_long:
                    if self.s21_state == 'normal':
                        if close < ema10:
                            self.s21_state = 'pullback'
                            self.s21_tracking_extreme = low
                    elif self.s21_state == 'pullback':
                        self.s21_tracking_extreme = min(self.s21_tracking_extreme, low)
                        if close > ema10:
                            candidate = self.s21_tracking_extreme - self.tick
                            self.s21_stop = max(self.s21_stop, candidate)
                            self.s21_state = 'normal'
                            self.s21_tracking_extreme = None
                else:  # short
                    if self.s21_state == 'normal':
                        if close > ema10:
                            self.s21_state = 'pullback'
                            self.s21_tracking_extreme = high
                    elif self.s21_state == 'pullback':
                        self.s21_tracking_extreme = max(self.s21_tracking_extreme, high)
                        if close < ema10:
                            candidate = self.s21_tracking_extreme + self.tick
                            self.s21_stop = min(self.s21_stop, candidate)
                            self.s21_state = 'normal'
                            self.s21_tracking_extreme = None

        # ===== S5.2 S3.1兜底(收盘触损) + S2.1接管(收盘触损) =====
        if not self.s52_done:
            self.s52_bars += 1

            # 两种模式都用收盘价触发
            hit_stop = False
            if self.is_long and close <= self.s52_stop:
                hit_stop = True
            elif not self.is_long and close >= self.s52_stop:
                hit_stop = True

            if hit_stop:
                self.s52_done = True
                exit_price = close
                if self.is_long:
                    pnl = (exit_price - self.entry_price) / self.entry_price * 100
                else:
                    pnl = (self.entry_price - exit_price) / self.entry_price * 100
                exits.append(ExitEvent(
                    strategy='S5.2', exit_price=exit_price,
                    exit_reason='stop', bars_held=self.s52_bars, pnl_pct=pnl,
                ))
            else:
                # S2.1回调追踪（始终在跑，用ema10）
                s21_updated = False
                if self.is_long:
                    if self.s52_s21_state == 'normal':
                        if close < ema10:
                            self.s52_s21_state = 'pullback'
                            self.s52_s21_tracking_extreme = low
                    elif self.s52_s21_state == 'pullback':
                        self.s52_s21_tracking_extreme = min(self.s52_s21_tracking_extreme, low)
                        if close > ema10:
                            candidate = self.s52_s21_tracking_extreme - self.tick
                            if candidate > self.s52_stop:
                                self.s52_stop = candidate
                                s21_updated = True
                            self.s52_s21_state = 'normal'
                            self.s52_s21_tracking_extreme = None
                else:
                    if self.s52_s21_state == 'normal':
                        if close > ema10:
                            self.s52_s21_state = 'pullback'
                            self.s52_s21_tracking_extreme = high
                    elif self.s52_s21_state == 'pullback':
                        self.s52_s21_tracking_extreme = max(self.s52_s21_tracking_extreme, high)
                        if close < ema10:
                            candidate = self.s52_s21_tracking_extreme + self.tick
                            if candidate < self.s52_stop:
                                self.s52_stop = candidate
                                s21_updated = True
                            self.s52_s21_state = 'normal'
                            self.s52_s21_tracking_extreme = None

                if s21_updated:
                    self.s52_mode = 's2.1'

                # S3.1追踪（仅在s3.1模式下更新止损）
                if self.s52_mode == 's3.1':
                    if prev_low is not None:
                        if self.is_long and close > prev_close:
                            candidate = prev_low - self.tick
                            self.s52_stop = max(self.s52_stop, candidate)
                        elif not self.is_long and close < prev_close:
                            candidate = prev_high + self.tick
                            self.s52_stop = min(self.s52_stop, candidate)

        # ===== S6 EMA5回调追踪止损，盘中触发 =====
        if not self.s6_done and ema5 is not None:
            self.s6_bars += 1

            if self.is_long and low <= self.s6_stop:
                self.s6_done = True
                pnl = (self.s6_stop - self.entry_price) / self.entry_price * 100
                exits.append(ExitEvent(
                    strategy='S6', exit_price=self.s6_stop,
                    exit_reason='stop', bars_held=self.s6_bars, pnl_pct=pnl,
                ))
            elif not self.is_long and high >= self.s6_stop:
                self.s6_done = True
                pnl = (self.entry_price - self.s6_stop) / self.entry_price * 100
                exits.append(ExitEvent(
                    strategy='S6', exit_price=self.s6_stop,
                    exit_reason='stop', bars_held=self.s6_bars, pnl_pct=pnl,
                ))
            else:
                # 状态机用ema5
                if self.is_long:
                    if self.s6_state == 'normal':
                        if close < ema5:
                            self.s6_state = 'pullback'
                            self.s6_tracking_extreme = low
                    elif self.s6_state == 'pullback':
                        self.s6_tracking_extreme = min(self.s6_tracking_extreme, low)
                        if close > ema5:
                            candidate = self.s6_tracking_extreme - self.tick
                            self.s6_stop = max(self.s6_stop, candidate)
                            self.s6_state = 'normal'
                            self.s6_tracking_extreme = None
                else:  # short
                    if self.s6_state == 'normal':
                        if close > ema5:
                            self.s6_state = 'pullback'
                            self.s6_tracking_extreme = high
                    elif self.s6_state == 'pullback':
                        self.s6_tracking_extreme = max(self.s6_tracking_extreme, high)
                        if close < ema5:
                            candidate = self.s6_tracking_extreme + self.tick
                            self.s6_stop = min(self.s6_stop, candidate)
                            self.s6_state = 'normal'
                            self.s6_tracking_extreme = None

        # ===== S6.1 EMA5回调追踪止损，收盘价触发 =====
        if not self.s61_done and ema5 is not None:
            self.s61_bars += 1

            if self.is_long and close <= self.s61_stop:
                self.s61_done = True
                pnl = (close - self.entry_price) / self.entry_price * 100
                exits.append(ExitEvent(
                    strategy='S6.1', exit_price=close,
                    exit_reason='stop', bars_held=self.s61_bars, pnl_pct=pnl,
                ))
            elif not self.is_long and close >= self.s61_stop:
                self.s61_done = True
                pnl = (self.entry_price - close) / self.entry_price * 100
                exits.append(ExitEvent(
                    strategy='S6.1', exit_price=close,
                    exit_reason='stop', bars_held=self.s61_bars, pnl_pct=pnl,
                ))
            else:
                # 状态机用ema5
                if self.is_long:
                    if self.s61_state == 'normal':
                        if close < ema5:
                            self.s61_state = 'pullback'
                            self.s61_tracking_extreme = low
                    elif self.s61_state == 'pullback':
                        self.s61_tracking_extreme = min(self.s61_tracking_extreme, low)
                        if close > ema5:
                            candidate = self.s61_tracking_extreme - self.tick
                            self.s61_stop = max(self.s61_stop, candidate)
                            self.s61_state = 'normal'
                            self.s61_tracking_extreme = None
                else:  # short
                    if self.s61_state == 'normal':
                        if close > ema5:
                            self.s61_state = 'pullback'
                            self.s61_tracking_extreme = high
                    elif self.s61_state == 'pullback':
                        self.s61_tracking_extreme = max(self.s61_tracking_extreme, high)
                        if close < ema5:
                            candidate = self.s61_tracking_extreme + self.tick
                            self.s61_stop = min(self.s61_stop, candidate)
                            self.s61_state = 'normal'
                            self.s61_tracking_extreme = None

        # 检测止损移动
        if not self.s2_done and self.s2_stop != self._prev_s2_stop:
            stop_updates.append(StopUpdate(
                strategy='S2', old_stop=self._prev_s2_stop, new_stop=self.s2_stop,
            ))
        if not self.s21_done and self.s21_stop != self._prev_s21_stop:
            stop_updates.append(StopUpdate(
                strategy='S2.1', old_stop=self._prev_s21_stop, new_stop=self.s21_stop,
            ))
        if not self.s51_done and self.s51_stop != self._prev_s51_stop:
            stop_updates.append(StopUpdate(
                strategy='S5.1', old_stop=self._prev_s51_stop, new_stop=self.s51_stop,
            ))
        if not self.s52_done and self.s52_stop != self._prev_s52_stop:
            stop_updates.append(StopUpdate(
                strategy='S5.2', old_stop=self._prev_s52_stop, new_stop=self.s52_stop,
            ))
        if not self.s6_done and self.s6_stop != self._prev_s6_stop:
            stop_updates.append(StopUpdate(
                strategy='S6', old_stop=self._prev_s6_stop, new_stop=self.s6_stop,
            ))
        if not self.s61_done and self.s61_stop != self._prev_s61_stop:
            stop_updates.append(StopUpdate(
                strategy='S6.1', old_stop=self._prev_s61_stop, new_stop=self.s61_stop,
            ))

        return exits, stop_updates

    def force_close(self, close_price: float) -> List[ExitEvent]:
        """强制平仓（回测结束或监控关闭时）"""
        exits = []
        for strategy, done, bars in [
            ('S1.1', self.s11_done, self.s11_bars),
            ('S2', self.s2_done, self.s2_bars),
            ('S2.1', self.s21_done, self.s21_bars),
            ('S3.1', self.s31_done, self.s31_bars),
            ('S5.1', self.s51_done, self.s51_bars),
            ('S5.2', self.s52_done, self.s52_bars),
            ('S6', self.s6_done, self.s6_bars),
            ('S6.1', self.s61_done, self.s61_bars),
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
        self.s21_done = True
        self.s31_done = True
        self.s51_done = True
        self.s52_done = True
        self.s6_done = True
        self.s61_done = True
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
            's21_stop': self.s21_stop, 's21_done': self.s21_done, 's21_bars': self.s21_bars,
            's21_state': self.s21_state, 's21_tracking_extreme': self.s21_tracking_extreme,
            's31_stop': self.s31_stop, 's31_done': self.s31_done, 's31_bars': self.s31_bars,
            's51_stop': self.s51_stop, 's51_done': self.s51_done, 's51_bars': self.s51_bars,
            's51_mode': self.s51_mode,
            's51_s2_state': self.s51_s2_state, 's51_s2_tracking_extreme': self.s51_s2_tracking_extreme,
            's52_stop': self.s52_stop, 's52_done': self.s52_done, 's52_bars': self.s52_bars,
            's52_mode': self.s52_mode,
            's52_s21_state': self.s52_s21_state, 's52_s21_tracking_extreme': self.s52_s21_tracking_extreme,
            's6_stop': self.s6_stop, 's6_done': self.s6_done, 's6_bars': self.s6_bars,
            's6_state': self.s6_state, 's6_tracking_extreme': self.s6_tracking_extreme,
            's61_stop': self.s61_stop, 's61_done': self.s61_done, 's61_bars': self.s61_bars,
            's61_state': self.s61_state, 's61_tracking_extreme': self.s61_tracking_extreme,
            '_prev_s51_stop': self._prev_s51_stop,
            '_prev_s52_stop': self._prev_s52_stop,
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
        tracker.s21_stop = d.get('s21_stop', d['s2_stop'])
        tracker.s21_done = d.get('s21_done', True)
        tracker.s21_bars = d.get('s21_bars', 0)
        tracker.s21_state = d.get('s21_state', 'normal')
        tracker.s21_tracking_extreme = d.get('s21_tracking_extreme')
        tracker.s31_stop = d['s31_stop']
        tracker.s31_done = d['s31_done']
        tracker.s31_bars = d['s31_bars']
        tracker.s51_stop = d['s51_stop']
        tracker.s51_done = d['s51_done']
        tracker.s51_bars = d['s51_bars']
        tracker.s51_mode = d['s51_mode']
        tracker.s51_s2_state = d['s51_s2_state']
        tracker.s51_s2_tracking_extreme = d['s51_s2_tracking_extreme']
        tracker.s52_stop = d.get('s52_stop', d['s51_stop'])
        tracker.s52_done = d.get('s52_done', True)
        tracker.s52_bars = d.get('s52_bars', 0)
        tracker.s52_mode = d.get('s52_mode', 's3.1')
        tracker.s52_s21_state = d.get('s52_s21_state', 'normal')
        tracker.s52_s21_tracking_extreme = d.get('s52_s21_tracking_extreme')
        tracker.s6_stop = d.get('s6_stop', d['s2_stop'])
        tracker.s6_done = d.get('s6_done', True)
        tracker.s6_bars = d.get('s6_bars', 0)
        tracker.s6_state = d.get('s6_state', 'normal')
        tracker.s6_tracking_extreme = d.get('s6_tracking_extreme')
        tracker.s61_stop = d.get('s61_stop', d['s2_stop'])
        tracker.s61_done = d.get('s61_done', True)
        tracker.s61_bars = d.get('s61_bars', 0)
        tracker.s61_state = d.get('s61_state', 'normal')
        tracker.s61_tracking_extreme = d.get('s61_tracking_extreme')
        tracker._prev_s2_stop = d['s2_stop']
        tracker._prev_s21_stop = d.get('s21_stop', d['s2_stop'])
        tracker._prev_s51_stop = d.get('_prev_s51_stop', d['s51_stop'])
        tracker._prev_s52_stop = d.get('_prev_s52_stop', d.get('s52_stop', d['s51_stop']))
        tracker._prev_s6_stop = d.get('s6_stop', d['s2_stop'])
        tracker._prev_s61_stop = d.get('s61_stop', d['s2_stop'])
        return tracker


# ============================================================
#  Type1 信号检测 — 影线触碰EMA10（与ABC并行运行）
# ============================================================

@dataclass
class Type1Signal:
    """Type1 挂单信号"""
    direction: str          # 'long' / 'short'
    pending_price: float    # 挂单价（high+1tick 多，low-1tick 空）
    stop_price: float       # 止损价（low-5tick 多，high+5tick 空）
    expiry_bars: int        # 挂单有效根数（默认5）
    bar_index: int          # 信号K线index
    # 分级所需因子
    stop_dist_atr: float    # 止损距离/ATR
    er_40: float            # ER(40)
    signal_density: int     # 最近10根内信号数
    recent_win_n: int       # 最近5笔滚动胜率


class Type1SignalDetector:
    """
    Type1 影线触碰EMA10信号检测状态机。
    逐K线调用 process_bar()，管理挂单状态。

    趋势：ER(20) > 0.3
    方向：close > EMA60 做多，close < EMA60 做空
    信号：影线碰EMA10弹回（close不破），prev_close未破EMA10
    入场：信号K线 high+1tick 挂单（多），5根内未触发则撤，跳空不入
    止损：信号K线 low - 5*tick
    """

    ER_THRESHOLD = 0.3
    PENDING_EXPIRY = 5
    STOP_TICKS = DEFAULT_STOP_TICKS  # 5

    def __init__(self):
        self._bar_index = -1
        self._prev_close = None
        self._prev_ema10 = None
        self._prev_ema60_dir = 0  # 1=多 -1=空 0=未知

        # 回踩计数（EMA60方向切换时重置）
        self._touch_count_long = 0
        self._touch_count_short = 0

        # 信号密集度追踪
        self._recent_signal_bars = []

        # 滚动胜率（最近5笔的MFE>=止损距离视为赢）
        self._recent_results = []  # list of bool (True=win)

        # 挂单状态
        self.pending = None  # dict or None

    def process_bar(self, close: float, high: float, low: float, opn: float,
                    ema10: float, ema60: float, er20: float, er40: float,
                    atr: float, tick_size: float) -> Optional[Type1Signal]:
        """
        处理一根已收盘K线。

        返回:
          Type1Signal（新挂单信号）或 None

        挂单成交/过期/跳空由调用方（monitor.py）在下一根bar处理。
        """
        self._bar_index += 1

        if self._prev_close is None:
            self._prev_close = close
            self._prev_ema10 = ema10
            return None

        if np.isnan(er20) or np.isnan(atr) or np.isnan(ema10) or np.isnan(ema60) or atr <= 0:
            self._prev_close = close
            self._prev_ema10 = ema10
            return None

        # EMA60方向切换 → 重置回踩计数
        curr_dir = 1 if close > ema60 else (-1 if close < ema60 else 0)
        if curr_dir != self._prev_ema60_dir and self._prev_ema60_dir != 0:
            if curr_dir == 1:
                self._touch_count_long = 0
            elif curr_dir == -1:
                self._touch_count_short = 0
        self._prev_ema60_dir = curr_dir

        # 趋势过滤
        if er20 <= self.ER_THRESHOLD:
            self._prev_close = close
            self._prev_ema10 = ema10
            return None

        # 方向判断
        direction = None
        if close > ema60:
            direction = 'long'
        elif close < ema60:
            direction = 'short'

        if direction is None:
            self._prev_close = close
            self._prev_ema10 = ema10
            return None

        # 信号检测
        signal = None
        tick = tick_size

        if direction == 'long':
            if (low <= ema10 and close > ema10
                    and self._prev_close >= self._prev_ema10):
                pending_price = _round_to_tick(high + tick, tick_size)
                stop_price = _round_to_tick(low - self.STOP_TICKS * tick, tick_size)
                self._touch_count_long += 1

                # 因子计算
                stop_dist_atr = abs(pending_price - stop_price) / atr
                er_40_val = er40 if not np.isnan(er40) else 0.0

                # 信号密集度
                self._recent_signal_bars = [b for b in self._recent_signal_bars
                                            if b >= self._bar_index - 10]
                density = len(self._recent_signal_bars)
                self._recent_signal_bars.append(self._bar_index)

                # 滚动胜率
                recent_win_n = sum(self._recent_results[-5:]) if self._recent_results else -1

                signal = Type1Signal(
                    direction='long',
                    pending_price=pending_price,
                    stop_price=stop_price,
                    expiry_bars=self.PENDING_EXPIRY,
                    bar_index=self._bar_index,
                    stop_dist_atr=round(stop_dist_atr, 4),
                    er_40=round(er_40_val, 4),
                    signal_density=density,
                    recent_win_n=recent_win_n,
                )

        elif direction == 'short':
            if (high >= ema10 and close < ema10
                    and self._prev_close <= self._prev_ema10):
                pending_price = _round_to_tick(low - tick, tick_size)
                stop_price = _round_to_tick(high + self.STOP_TICKS * tick, tick_size)
                self._touch_count_short += 1

                stop_dist_atr = abs(pending_price - stop_price) / atr
                er_40_val = er40 if not np.isnan(er40) else 0.0

                self._recent_signal_bars = [b for b in self._recent_signal_bars
                                            if b >= self._bar_index - 10]
                density = len(self._recent_signal_bars)
                self._recent_signal_bars.append(self._bar_index)

                recent_win_n = sum(self._recent_results[-5:]) if self._recent_results else -1

                signal = Type1Signal(
                    direction='short',
                    pending_price=pending_price,
                    stop_price=stop_price,
                    expiry_bars=self.PENDING_EXPIRY,
                    bar_index=self._bar_index,
                    stop_dist_atr=round(stop_dist_atr, 4),
                    er_40=round(er_40_val, 4),
                    signal_density=density,
                    recent_win_n=recent_win_n,
                )

        # 新信号替换旧挂单
        if signal is not None:
            self.pending = {
                'signal': signal,
                'expiry_bar': self._bar_index + self.PENDING_EXPIRY,
            }

        self._prev_close = close
        self._prev_ema10 = ema10
        return signal

    def check_fill(self, high: float, low: float, opn: float) -> Optional[dict]:
        """
        检查挂单是否成交。每根新bar调用。

        返回:
          {'status': 'filled', 'signal': Type1Signal} — 成交
          {'status': 'gap_skip'} — 跳空跳过
          {'status': 'expired'} — 过期
          None — 继续等待
        """
        if self.pending is None:
            return None

        sig = self.pending['signal']

        # 过期
        # 注意：monitor.py 中 check_fill 在 process_bar 之前调用，
        # _bar_index 落后1根，所以用 >= 补偿，确保挂单只有效5根
        if self._bar_index >= self.pending['expiry_bar']:
            self.pending = None
            return {'status': 'expired'}

        # 跳空检查
        if sig.direction == 'long':
            if opn > sig.pending_price:
                self.pending = None
                return {'status': 'gap_skip'}
            if high >= sig.pending_price:
                self.pending = None
                return {'status': 'filled', 'signal': sig}
        else:
            if opn < sig.pending_price:
                self.pending = None
                return {'status': 'gap_skip'}
            if low <= sig.pending_price:
                self.pending = None
                return {'status': 'filled', 'signal': sig}

        return None

    def record_trade_result(self, win: bool):
        """记录一笔交易结果（用于滚动胜率计算）"""
        self._recent_results.append(win)
        # 只保留最近20笔
        if len(self._recent_results) > 20:
            self._recent_results = self._recent_results[-20:]

    def to_dict(self) -> dict:
        return {
            '_bar_index': self._bar_index,
            '_prev_close': self._prev_close,
            '_prev_ema10': self._prev_ema10,
            '_prev_ema60_dir': self._prev_ema60_dir,
            '_touch_count_long': self._touch_count_long,
            '_touch_count_short': self._touch_count_short,
            '_recent_signal_bars': self._recent_signal_bars,
            '_recent_results': self._recent_results,
            'pending': {
                'signal': {
                    'direction': self.pending['signal'].direction,
                    'pending_price': self.pending['signal'].pending_price,
                    'stop_price': self.pending['signal'].stop_price,
                    'expiry_bars': self.pending['signal'].expiry_bars,
                    'bar_index': self.pending['signal'].bar_index,
                    'stop_dist_atr': self.pending['signal'].stop_dist_atr,
                    'er_40': self.pending['signal'].er_40,
                    'signal_density': self.pending['signal'].signal_density,
                    'recent_win_n': self.pending['signal'].recent_win_n,
                },
                'expiry_bar': self.pending['expiry_bar'],
            } if self.pending else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Type1SignalDetector':
        det = cls()
        det._bar_index = d.get('_bar_index', -1)
        det._prev_close = d.get('_prev_close')
        det._prev_ema10 = d.get('_prev_ema10')
        det._prev_ema60_dir = d.get('_prev_ema60_dir', 0)
        det._touch_count_long = d.get('_touch_count_long', 0)
        det._touch_count_short = d.get('_touch_count_short', 0)
        det._recent_signal_bars = d.get('_recent_signal_bars', [])
        det._recent_results = d.get('_recent_results', [])
        p = d.get('pending')
        if p and p.get('signal'):
            s = p['signal']
            det.pending = {
                'signal': Type1Signal(**s),
                'expiry_bar': p['expiry_bar'],
            }
        return det


def _round_to_tick(price: float, tick_size: float) -> float:
    """四舍五入到最近的 tick"""
    return round(round(price / tick_size) * tick_size, 10)


def classify_type1_tier(stop_dist_atr: float, recent_win_n: int,
                        er_40: float, signal_density: int) -> tuple:
    """
    Type1 子场景分级。

    返回: (tier_name, ladder_preset)
      tier_name: 'alpha-1'...'gamma'
      ladder_preset: 'I' / '2R' / None(γ不做)
    """
    stop_ok = stop_dist_atr < 1.5
    hot4 = recent_win_n >= 4
    hot3 = recent_win_n >= 3
    er40_ok = er_40 >= 0.42
    density_ok = signal_density >= 1

    # α: 双核命中
    if stop_ok and hot4:
        return ('alpha-1', 'I')
    if stop_ok and er40_ok:
        return ('alpha-2', 'I')
    if hot4 and er40_ok:
        return ('alpha-3', 'I')
    # β
    if stop_ok and hot3:
        return ('beta-1', 'I')
    if hot4 and density_ok and not stop_ok and not er40_ok:
        return ('beta-2', '2R')
    # γ
    return ('gamma', None)


TYPE1_TIER_NAMES = {
    'alpha-1': 'α-1: stop<1.5+热手≥4',
    'alpha-2': 'α-2: stop<1.5+ER40≥0.42',
    'alpha-3': 'α-3: 热手≥4+ER40≥0.42',
    'beta-1': 'β-1: stop<1.5+热手≥3',
    'beta-2': 'β-2: 热手≥4+density≥1',
    'gamma': 'γ不做',
}


# ============================================================
#  LadderRTracker — 阶梯R倍数移动止损（Type1专用）
# ============================================================

class LadderRTracker:
    """
    阶梯R倍数移动止损。

    两种预设模式：
      'I':  1→0, 3→1, 之后每+2R（5→3, 7→5, ...）—— α/β-1最优
      '2R': 均匀2R步长（2→0, 4→2, 6→4, ...）—— β-2最优

    也可自定义 steps 列表。

    逻辑：
      - 价格到达 trigger_r 时，止损移至 move_to_r
      - 盘中触损出场
      - 60根窗口超时按收盘价结算
    """

    # 预设方案
    PRESETS = {
        'I': [(1, 0), (3, 1)],    # 之后自动续 +2R
        '2R': [(2, 0)],           # 之后自动续 +2R
    }

    def __init__(self, direction: str, entry_price: float,
                 stop_price: float, tick_size: float,
                 preset: str = 'I', max_window: int = 60):
        self.direction = direction
        self.is_long = (direction == 'long')
        self.entry_price = entry_price
        self.stop_price = stop_price  # 初始止损
        self.tick_size = tick_size
        self.stop_dist = abs(entry_price - stop_price)
        self.max_window = max_window
        self.preset = preset

        # 构建阶梯表（预生成足够多的台阶）
        base_steps = self.PRESETS.get(preset, [(1, 0), (3, 1)])
        self.steps = list(base_steps)
        # 自动续接 +2R 直到 30R
        if self.steps:
            last_trigger = self.steps[-1][0]
            last_move = self.steps[-1][1]
            step_size = last_trigger - last_move  # 间距
            while last_trigger < 30:
                last_trigger += step_size
                last_move += step_size
                self.steps.append((last_trigger, last_move))

        # 状态
        self.curr_stop = stop_price
        self.curr_stop_r = -1.0
        self.next_step_idx = 0
        self.max_r_reached = 0.0
        self.bars = 0
        self.done = False
        self.strategy_name = f'LR_{preset}'

    def process_bar(self, close: float, high: float, low: float) -> Optional[ExitEvent]:
        """
        处理一根已收盘K线。返回 ExitEvent 或 None。
        只需要 close/high/low，不依赖均线。
        """
        if self.done:
            return None

        self.bars += 1

        # 1. 检查止损（盘中触发）
        if self.is_long:
            if low <= self.curr_stop:
                self.done = True
                pnl = (self.curr_stop - self.entry_price) / self.entry_price * 100
                reason = 'stop' if self.curr_stop_r < 0 else 'trail_stop'
                return ExitEvent(
                    strategy=self.strategy_name,
                    exit_price=self.curr_stop,
                    exit_reason=reason,
                    bars_held=self.bars,
                    pnl_pct=round(pnl, 4),
                )
            bar_r = (high - self.entry_price) / self.stop_dist
        else:
            if high >= self.curr_stop:
                self.done = True
                pnl = (self.entry_price - self.curr_stop) / self.entry_price * 100
                reason = 'stop' if self.curr_stop_r < 0 else 'trail_stop'
                return ExitEvent(
                    strategy=self.strategy_name,
                    exit_price=self.curr_stop,
                    exit_reason=reason,
                    bars_held=self.bars,
                    pnl_pct=round(pnl, 4),
                )
            bar_r = (self.entry_price - low) / self.stop_dist

        # 2. 更新最大R
        if bar_r > self.max_r_reached:
            self.max_r_reached = bar_r

        # 3. 阶梯升级
        while self.next_step_idx < len(self.steps):
            trigger_r, move_to_r = self.steps[self.next_step_idx]
            if self.max_r_reached >= trigger_r:
                if self.is_long:
                    new_stop = self.entry_price + move_to_r * self.stop_dist
                else:
                    new_stop = self.entry_price - move_to_r * self.stop_dist
                if (self.is_long and new_stop > self.curr_stop) or \
                   (not self.is_long and new_stop < self.curr_stop):
                    self.curr_stop = new_stop
                    self.curr_stop_r = move_to_r
                self.next_step_idx += 1
            else:
                break

        # 4. 窗口超时
        if self.bars >= self.max_window:
            self.done = True
            pnl = ((close - self.entry_price) if self.is_long
                   else (self.entry_price - close)) / self.entry_price * 100
            return ExitEvent(
                strategy=self.strategy_name,
                exit_price=close,
                exit_reason='timeout',
                bars_held=self.bars,
                pnl_pct=round(pnl, 4),
            )

        return None

    def force_close(self, close_price: float) -> Optional[ExitEvent]:
        """强制平仓（回测结束时调用）"""
        if self.done:
            return None
        self.done = True
        pnl = ((close_price - self.entry_price) if self.is_long
               else (self.entry_price - close_price)) / self.entry_price * 100
        return ExitEvent(
            strategy=self.strategy_name,
            exit_price=close_price,
            exit_reason='backtest_end',
            bars_held=self.bars,
            pnl_pct=round(pnl, 4),
        )

    def to_dict(self) -> dict:
        """序列化（监控状态持久化用）"""
        return {
            'direction': self.direction,
            'entry_price': self.entry_price,
            'stop_price': self.stop_price,
            'tick_size': self.tick_size,
            'preset': self.preset,
            'max_window': self.max_window,
            'curr_stop': self.curr_stop,
            'curr_stop_r': self.curr_stop_r,
            'next_step_idx': self.next_step_idx,
            'max_r_reached': self.max_r_reached,
            'bars': self.bars,
            'done': self.done,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'LadderRTracker':
        """从dict恢复"""
        tracker = cls(
            direction=d['direction'],
            entry_price=d['entry_price'],
            stop_price=d['stop_price'],
            tick_size=d['tick_size'],
            preset=d.get('preset', 'I'),
            max_window=d.get('max_window', 60),
        )
        tracker.curr_stop = d['curr_stop']
        tracker.curr_stop_r = d.get('curr_stop_r', -1.0)
        tracker.next_step_idx = d.get('next_step_idx', 0)
        tracker.max_r_reached = d.get('max_r_reached', 0.0)
        tracker.bars = d.get('bars', 0)
        tracker.done = d.get('done', False)
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


# 场景对应的出场策略和名称（2026-04-11更新：场景1/2从S2升级为S6）
SCENARIO_EXIT = {1: 'S6', 2: 'S6', 3: 'S5.1'}
SCENARIO_PNL_COL = {1: 's6_pnl', 2: 's6_pnl', 3: 's51_pnl'}
SCENARIO_REASON_COL = {1: 's6_reason', 2: 's6_reason', 3: 's51_reason'}
SCENARIO_NAMES = {
    1: '场景1: A类+ER≥0.5+偏离≥1.0ATR → S6',
    2: '场景2: C类+偏离≥2.0ATR → S6',
    3: '场景3: B类+ER≥0.5+偏离0.1~0.3ATR → S5.1',
}
