# -*- coding: utf-8 -*-
"""
监控系统配置
============
支持 JSON 文件加载 + 环境变量覆盖。
"""

import os
import json
from dataclasses import dataclass, field
from signal_core import SYMBOL_CONFIGS


@dataclass
class MonitorConfig:
    # 天勤账号
    tq_account: str = ""
    tq_password: str = ""

    # PushPlus
    pushplus_token: str = ""        # 用户token

    # 监控品种（空=全部32个）
    symbols: list = field(default_factory=list)

    # 状态持久化
    state_file: str = "state/monitor_state.json"

    # 日志
    log_file: str = "logs/monitor.log"
    log_level: str = "INFO"

    # 通知限流
    stop_update_cooldown: int = 300  # 止损推送冷却秒数（每品种）

    # 重连
    max_reconnect_attempts: int = 10
    reconnect_delay_base: int = 5    # 指数退避基数（秒）

    def __post_init__(self):
        if not self.symbols:
            self.symbols = list(SYMBOL_CONFIGS.keys())

    @classmethod
    def from_file(cls, path: str) -> 'MonitorConfig':
        """从 JSON 文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> 'MonitorConfig':
        """从环境变量加载（云端部署用）"""
        cfg = cls()
        cfg.tq_account = os.environ.get('TQ_ACCOUNT', cfg.tq_account)
        cfg.tq_password = os.environ.get('TQ_PASSWORD', cfg.tq_password)
        cfg.pushplus_token = os.environ.get('PUSHPLUS_TOKEN', cfg.pushplus_token)
        syms = os.environ.get('MONITOR_SYMBOLS')
        if syms:
            cfg.symbols = [s.strip() for s in syms.split(',')]
        return cfg

    @classmethod
    def load(cls, config_path: str = None) -> 'MonitorConfig':
        """加载配置：文件优先，环境变量覆盖"""
        if config_path and os.path.exists(config_path):
            cfg = cls.from_file(config_path)
        else:
            cfg = cls()

        # 环境变量覆盖（优先级最高）
        if os.environ.get('TQ_ACCOUNT'):
            cfg.tq_account = os.environ['TQ_ACCOUNT']
        if os.environ.get('TQ_PASSWORD'):
            cfg.tq_password = os.environ['TQ_PASSWORD']
        if os.environ.get('PUSHPLUS_TOKEN'):
            cfg.pushplus_token = os.environ['PUSHPLUS_TOKEN']

        return cfg
