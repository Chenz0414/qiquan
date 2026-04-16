# -*- coding: utf-8 -*-
"""
规则定义单一真相源
==================
所有场景/Type1分级/全局过滤的结构化定义都集中在这里。

前端 tooltip、规则卡片、统计 key、文档都从这里读取，避免散落字符串。
monitor 启动时 assert 本模块的阈值与 signal_core 中的常量一致。
"""

from typing import Dict, List, Optional

# ============================================================
#  规则定义
# ============================================================

RULES: Dict[str, dict] = {
    # ---- ABC 场景 ----
    "scenario_1": {
        "label": "场景1",
        "family": "ABC",
        "description": "A类 + ER(20)≥0.5 + 偏离≥1.0ATR → S6",
        "conditions": [
            {"key": "signal_type", "op": "==", "value": "A"},
            {"key": "er20", "op": "<", "value": 0.7},
            {"key": "er20", "op": ">=", "value": 0.5},
            {"key": "deviation_atr", "op": ">=", "value": 1.0},
        ],
        "direction_filter": None,          # 多空均做
        "exit_strategy": "S6",
        "rationale": "影线触碰 + 强趋势 + 显著偏离回调 → 用 EMA5 紧追踪给予较长止盈空间",
        "introduced_commit": "master",
    },
    "scenario_2": {
        "label": "场景2",
        "family": "ABC",
        "description": "C类 + 偏离≥2.0ATR → S6（仅做多）",
        "conditions": [
            {"key": "signal_type", "op": "==", "value": "C"},
            {"key": "er20", "op": "<", "value": 0.7},
            {"key": "deviation_atr", "op": ">=", "value": 2.0},
        ],
        "direction_filter": "long",        # 空头120天EV≈0，停做
        "exit_strategy": "S6",
        "rationale": "深度回调 + 极端偏离 → S6 紧追踪；空头样本120天EV=0故停做",
        "introduced_commit": "e6bfd88",
    },
    "scenario_3": {
        "label": "场景3",
        "family": "ABC",
        "description": "B类 + ER(20)≥0.5 + 偏离0.1~0.3ATR → S5.1",
        "conditions": [
            {"key": "signal_type", "op": "==", "value": "B"},
            {"key": "er20", "op": "<", "value": 0.7},
            {"key": "er20", "op": ">=", "value": 0.5},
            {"key": "deviation_atr", "op": ">=", "value": 0.1},
            {"key": "deviation_atr", "op": "<", "value": 0.3},
        ],
        "direction_filter": None,
        "exit_strategy": "S5.1",
        "rationale": "轻度回调 + 强趋势 → S3.1 兜底 + S2 盘中接管",
        "introduced_commit": "master",
    },
    # ---- Type1 分级 ----
    "type1_alpha_1": {
        "label": "α-1",
        "family": "Type1",
        "description": "stop<1.5ATR + 热手≥4",
        "conditions": [
            {"key": "er20", "op": ">", "value": 0.3},
            {"key": "stop_dist_atr", "op": "<", "value": 1.5},
            {"key": "recent_win_n", "op": ">=", "value": 4},
        ],
        "exit_strategy": "LR_I",
        "rationale": "止损紧 + 滚动连赢 → 1→0, 3→1 阶梯止损",
        "introduced_commit": "485bd10",
    },
    "type1_alpha_2": {
        "label": "α-2",
        "family": "Type1",
        "description": "stop<1.5ATR + ER40≥0.42",
        "conditions": [
            {"key": "er20", "op": ">", "value": 0.3},
            {"key": "stop_dist_atr", "op": "<", "value": 1.5},
            {"key": "er_40", "op": ">=", "value": 0.42},
        ],
        "exit_strategy": "LR_I",
        "rationale": "止损紧 + 长周期趋势强 → 阶梯 I",
        "introduced_commit": "485bd10",
    },
    "type1_alpha_3": {
        "label": "α-3",
        "family": "Type1",
        "description": "热手≥4 + ER40≥0.42",
        "conditions": [
            {"key": "er20", "op": ">", "value": 0.3},
            {"key": "recent_win_n", "op": ">=", "value": 4},
            {"key": "er_40", "op": ">=", "value": 0.42},
        ],
        "exit_strategy": "LR_I",
        "rationale": "连赢 + 长周期趋势强 → 阶梯 I",
        "introduced_commit": "485bd10",
    },
    "type1_beta_1": {
        "label": "β-1",
        "family": "Type1",
        "description": "stop<1.5ATR + 热手≥3",
        "conditions": [
            {"key": "er20", "op": ">", "value": 0.3},
            {"key": "stop_dist_atr", "op": "<", "value": 1.5},
            {"key": "recent_win_n", "op": ">=", "value": 3},
        ],
        "exit_strategy": "LR_I",
        "rationale": "次高质量配置，阶梯 I",
        "introduced_commit": "485bd10",
    },
    "type1_beta_2": {
        "label": "β-2",
        "family": "Type1",
        "description": "热手≥4 + density≥1（已停做）",
        "conditions": [
            {"key": "er20", "op": ">", "value": 0.3},
            {"key": "recent_win_n", "op": ">=", "value": 4},
            {"key": "signal_density", "op": ">=", "value": 1},
            {"key": "stop_dist_atr", "op": ">=", "value": 1.5},
            {"key": "er_40", "op": "<", "value": 0.42},
        ],
        "exit_strategy": None,
        "rationale": "120天N=29 EV=-0.55 累计-10.4%，多空均负；保留标签但不触发",
        "introduced_commit": "485bd10",
        "disabled": True,
        "disabled_since": "2026-04-15",
    },
    "type1_gamma": {
        "label": "γ",
        "family": "Type1",
        "description": "未满足 α/β 任一组合（不做）",
        "conditions": [],
        "exit_strategy": None,
        "rationale": "过滤档",
        "introduced_commit": "485bd10",
        "disabled": True,
    },
}


# ============================================================
#  全局过滤
# ============================================================

GLOBAL_FILTERS: List[dict] = [
    {
        "key": "er20_too_high",
        "label": "ER(20) ≥ 0.7",
        "description": "ER≥0.7 正期望消失，全场景禁入",
        "rationale": "趋势已极端，进场接刀",
        "applies_to": ["scenario_1", "scenario_2", "scenario_3"],
    },
    {
        "key": "scenario_2_short_disabled",
        "label": "场景2空头停做",
        "description": "场景2空头120天样本EV≈0",
        "rationale": "样本不显著，资金利用率低",
        "applies_to": ["scenario_2"],
        "disabled_since": "2026-04-15",
    },
]


# ============================================================
#  拒绝原因枚举（reject_observer 使用的 stage/reason 规范）
# ============================================================

REJECT_STAGES = {
    "pre_trend": "趋势判定阶段",
    "pre_signal": "信号分类阶段",
    "scenario_classify": "场景匹配阶段",
    "global_filter": "全局过滤阶段",
    "direction_filter": "方向过滤阶段",
    "type1_tier": "Type1 分级阶段",
    "cooldown": "推送冷却阶段",
}

REJECT_REASONS = {
    # pre_signal
    "no_pullback": "无回调发生",
    # scenario_classify
    "not_a_b_c_pattern": "不满足A/B/C任一场景阈值",
    "er20_too_low": "ER(20) 低于场景门槛",
    "er20_too_high": "ER(20) ≥ 0.7 禁入",
    "deviation_out_of_band": "偏离度不在场景区间",
    "scenario_2_short_filtered": "场景2空头停做",
    # type1
    "type1_er20_too_low": "Type1 要求 ER(20)>0.3",
    "type1_no_trend": "Type1 close 未明显突破 EMA60",
    "type1_gamma": "Type1 分级为 γ（不做）",
    "type1_beta_2_disabled": "Type1 β-2 已停做",
    # near_miss 特殊
    "near_miss_er20": "差一点 ER 达标",
    "near_miss_deviation": "差一点偏离度达标",
    "near_miss_stop_dist": "差一点 stop_dist 达标",
}


# ============================================================
#  工具函数
# ============================================================

def get_rule(rule_key: str) -> Optional[dict]:
    """按 key 取规则定义，找不到返回 None"""
    return RULES.get(rule_key)


def rule_label(rule_key: str) -> str:
    """取规则短名；未知 key 返回原 key"""
    r = RULES.get(rule_key)
    return r["label"] if r else rule_key


def active_rules(family: Optional[str] = None) -> List[str]:
    """返回当前启用的规则 key 列表"""
    return [
        k for k, v in RULES.items()
        if not v.get("disabled", False)
        and (family is None or v.get("family") == family)
    ]


def describe_filter(filter_key: str) -> Optional[dict]:
    for f in GLOBAL_FILTERS:
        if f["key"] == filter_key:
            return f
    return None


# ============================================================
#  与 signal_core 的一致性校验
# ============================================================

def assert_catalog_consistency():
    """
    启动时调用。校验 rules_catalog 的阈值与 signal_core 实际代码一致。

    发现漂移时抛 AssertionError，迫使开发者同步修改两处。
    """
    # 延迟导入避免循环
    from signal_core import classify_scenario, Type1SignalDetector, SCENARIO_EXIT

    # --- 场景1: A + ER=0.5 + 偏离=1.0 应命中 ---
    assert classify_scenario("A", 0.5, 1.0, "long") == 1, \
        "rules_catalog.scenario_1 阈值与 signal_core.classify_scenario 不一致"
    assert classify_scenario("A", 0.49, 1.0, "long") is None, "ER 门槛漂移"
    assert classify_scenario("A", 0.5, 0.99, "long") is None, "偏离门槛漂移"

    # --- 场景2: C + 偏离=2.0 应命中多头，空头应返回 None ---
    assert classify_scenario("C", 0.6, 2.0, "long") == 2
    assert classify_scenario("C", 0.6, 2.0, "short") is None, "scenario_2 空头过滤漂移"

    # --- 场景3: B + ER=0.5 + 偏离=0.2 应命中 ---
    assert classify_scenario("B", 0.5, 0.2, "long") == 3
    assert classify_scenario("B", 0.5, 0.3, "long") is None, "场景3上界漂移"

    # --- ER≥0.7 全局禁 ---
    assert classify_scenario("A", 0.7, 1.5, "long") is None, "ER≥0.7 全局过滤漂移"

    # --- Type1 ER 门槛 ---
    assert Type1SignalDetector.ER_THRESHOLD == 0.3, \
        "rules_catalog.type1_* ER 门槛与 signal_core 不一致"

    # --- 出场映射 ---
    assert SCENARIO_EXIT == {1: "S6", 2: "S6", 3: "S5.1"}, \
        "SCENARIO_EXIT 映射漂移"

    return True
