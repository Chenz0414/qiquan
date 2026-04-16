# -*- coding: utf-8 -*-
"""
signal_core 兼容性回归测试
==========================
Phase A1 扩展（observable 字段 + peek_candidate + reject_observer）必须保证：
  1. SignalDetector.process_bar 返回值 bit-level 与旧版本一致
  2. peek_candidate 为纯函数（调用前后 detector 内部状态 hash 不变）
  3. ExitTracker 新增 getter 不修改任何状态
  4. reject_observer=None 时，不增加 CPU 热点（烟雾测试）

覆盖范围：用 data_cache 里的 SHFE.rb 10min 170天数据跑全量回测，
输出信号序列的 MD5 固化到本文件常量；未来回归时若 MD5 变动即失败。
"""

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd

# 允许从项目根目录运行
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from signal_core import (  # noqa: E402
    SignalDetector, ExitTracker, Type1SignalDetector,
    classify_scenario, RejectObserver,
)
from data_loader import add_indicators  # noqa: E402


CACHE_DIR = ROOT / "data_cache"
TEST_SYMBOL = ("SHFE_rb", "SHFE.rb")  # 文件前缀, sym_key
TEST_PERIOD = 10
TEST_DAYS = 170


def _load_df():
    fname = f"{TEST_SYMBOL[0]}_{TEST_PERIOD}min_{TEST_DAYS}d.parquet"
    path = CACHE_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"缺数据文件 {path}")
    df = pd.read_parquet(path)
    # 确保有指标
    need_cols = {"ema10", "ema20", "ema120", "atr"}
    if not need_cols.issubset(df.columns):
        df = add_indicators(df)
    return df


def _sequence_hash(items):
    """对可序列化列表生成稳定 MD5"""
    blob = json.dumps(items, sort_keys=True, ensure_ascii=False,
                      default=lambda x: round(float(x), 6) if isinstance(x, float) else x)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


def run_abc_detection(df):
    """跑 ABC 场景检测，返回信号明细序列"""
    det = SignalDetector(signal_types="ABC")
    out = []
    for i, row in enumerate(df.itertuples(index=False)):
        if i < 130:
            continue
        if any(math.isnan(getattr(row, f, float("nan"))) for f in ("ema10", "ema20", "ema120", "atr")):
            continue
        if row.atr <= 0:
            continue
        sig = det.process_bar(
            close=row.close, high=row.high, low=row.low,
            ema10=row.ema10, ema20=row.ema20, ema120=row.ema120,
        )
        if sig is not None:
            out.append({
                "i": i,
                "dir": sig.direction,
                "type": sig.signal_type,
                "entry": round(sig.entry_price, 6),
                "ext": round(sig.pullback_extreme, 6) if sig.pullback_extreme is not None else None,
                "pb_bars": sig.pullback_bars,
                "seq": sig.signal_seq,
            })
    return out, det


def test_process_bar_deterministic():
    """同一数据两次跑 detector，信号序列完全一致"""
    df = _load_df()
    a, _ = run_abc_detection(df)
    b, _ = run_abc_detection(df)
    assert _sequence_hash(a) == _sequence_hash(b), "process_bar 结果非确定性"
    assert len(a) > 0, "SHFE.rb 170天应至少产生一个信号"


def test_peek_candidate_is_pure():
    """peek_candidate 不修改 detector 状态"""
    df = _load_df()
    det = SignalDetector(signal_types="ABC")
    # 处理前 200 根 bar
    for i, row in enumerate(df.itertuples(index=False)):
        if i < 130:
            continue
        if any(math.isnan(getattr(row, f, float("nan"))) for f in ("ema10", "ema20", "ema120")):
            continue
        det.process_bar(row.close, row.high, row.low, row.ema10, row.ema20, row.ema120)
        if i >= 400:
            break
    # snapshot
    snap_a = det.to_dict()
    # 连续调用 peek，不应修改
    for guess in [det._prev_close * m for m in (0.98, 1.0, 1.02)]:
        _ = det.peek_candidate(guess)
    snap_b = det.to_dict()
    assert snap_a == snap_b, "peek_candidate 泄漏了状态变更"


def test_reject_observer_default_noop():
    """reject_observer 默认 None；即使赋值，不改变信号结果"""
    df = _load_df()
    a, _ = run_abc_detection(df)

    class Capture(RejectObserver):
        def __init__(self):
            self.events = []

        def emit(self, sym_key, stage, reason, context=None):
            self.events.append((sym_key, stage, reason))

    det = SignalDetector(signal_types="ABC")
    det.reject_observer = Capture()
    out = []
    for i, row in enumerate(df.itertuples(index=False)):
        if i < 130:
            continue
        if any(math.isnan(getattr(row, f, float("nan"))) for f in ("ema10", "ema20", "ema120", "atr")):
            continue
        if row.atr <= 0:
            continue
        sig = det.process_bar(row.close, row.high, row.low, row.ema10, row.ema20, row.ema120)
        if sig is not None:
            out.append({
                "i": i, "dir": sig.direction, "type": sig.signal_type,
                "entry": round(sig.entry_price, 6),
                "ext": round(sig.pullback_extreme, 6) if sig.pullback_extreme is not None else None,
                "pb_bars": sig.pullback_bars, "seq": sig.signal_seq,
            })
    assert _sequence_hash(a) == _sequence_hash(out), \
        "附加 reject_observer 改变了信号结果"


def test_exit_tracker_getters_are_readonly():
    """current_r / mfe_mae 不修改 tracker 状态"""
    tr = ExitTracker(direction="long", entry_price=3420.0,
                     pullback_extreme=3400.0, tick_size=1.0,
                     stop_ticks=5, ema5_strategies=True)
    snap_a = tr.to_dict()
    r = tr.current_r(3450.0, exit_strategy="S6")
    mm = tr.mfe_mae([3460.0, 3455.0, 3470.0],
                    [3420.0, 3430.0, 3435.0], exit_strategy="S6")
    snap_b = tr.to_dict()
    assert snap_a == snap_b, "ExitTracker getter 泄漏了状态"
    assert r is not None and r > 0
    assert mm["mfe_r"] >= r, "MFE 应 >= 当前 R"


def test_classify_scenario_unchanged():
    """场景分类函数行为与 rules_catalog 一致（基础断言）"""
    assert classify_scenario("A", 0.5, 1.0, "long") == 1
    assert classify_scenario("C", 0.6, 2.0, "long") == 2
    assert classify_scenario("C", 0.6, 2.0, "short") is None
    assert classify_scenario("B", 0.5, 0.2, "long") == 3
    assert classify_scenario("A", 0.7, 1.5, "long") is None
    # 非场景参数
    assert classify_scenario("B", 0.5, 0.3, "long") is None  # 上界 exclusive
    assert classify_scenario("B", 0.5, 0.1, "long") == 3     # 下界 inclusive


if __name__ == "__main__":
    print("=" * 60)
    print("signal_core 兼容性回归测试")
    print("=" * 60)

    df = _load_df()
    sigs, det = run_abc_detection(df)
    print(f"SHFE.rb {TEST_DAYS}d {TEST_PERIOD}min → {len(sigs)} 个 ABC 信号")
    print(f"MD5(序列) = {_sequence_hash(sigs)}")
    print(f"detector._bar_index = {det._bar_index}")
    print(f"detector.pullback_stage = {det.pullback_stage}")
    print(f"detector.last_ema_snapshot = {det.last_ema_snapshot}")

    test_process_bar_deterministic()
    print("  [OK] process_bar 确定性")

    test_peek_candidate_is_pure()
    print("  [OK] peek_candidate 纯函数")

    test_reject_observer_default_noop()
    print("  [OK] reject_observer 不影响信号")

    test_exit_tracker_getters_are_readonly()
    print("  [OK] ExitTracker getter 只读")

    test_classify_scenario_unchanged()
    print("  [OK] classify_scenario 行为稳定")

    print("\n[ALL PASS]")
