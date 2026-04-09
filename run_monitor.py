# -*- coding: utf-8 -*-
"""
监控系统入口
============
用法:
    python run_monitor.py --config monitor_settings.json
    python run_monitor.py --dry-run
    python run_monitor.py --symbols SHFE.ag SHFE.au
    python run_monitor.py --reset-state
    python run_monitor.py --no-web
"""

import os
import sys
import atexit
import signal
import argparse
import logging
import threading
from logging.handlers import RotatingFileHandler

from monitor_config import MonitorConfig
from monitor import MonitorEngine


def setup_logging(log_file: str, level: str = "INFO"):
    """配置日志：文件(rotating) + 控制台"""
    os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # 文件日志（10MB x 5个文件）
    fh = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # 控制台日志
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # 抑制 TqSdk 的 DEBUG 日志洪泛（每秒数十条 websocket 消息会淹没应用日志）
    # TqSdk 会在 TqApi() 初始化时强制设置子logger级别为 DEBUG，
    # 所以在 handler 上也设置级别作为最终防线
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    for name in ('TqApi', 'TqApi.TqConnect', 'tqsdk'):
        logging.getLogger(name).setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description='期货信号监控系统')
    parser.add_argument('--config', default='monitor_settings.json',
                        help='配置文件路径 (默认: monitor_settings.json)')
    parser.add_argument('--symbols', nargs='*',
                        help='覆盖监控品种列表 (例: SHFE.ag SHFE.au)')
    parser.add_argument('--reset-state', action='store_true',
                        help='清除保存的状态，全新冷启动')
    parser.add_argument('--dry-run', action='store_true',
                        help='干跑模式：只记日志不推送微信')
    parser.add_argument('--log-level', default=None,
                        help='日志级别 (DEBUG/INFO/WARNING)')
    parser.add_argument('--no-web', action='store_true',
                        help='禁用Web仪表盘')
    parser.add_argument('--web-port', type=int, default=8080,
                        help='Web端口 (默认: 8080)')
    args = parser.parse_args()

    # 加载配置
    config = MonitorConfig.load(args.config)
    if args.symbols:
        config.symbols = args.symbols
    if args.log_level:
        config.log_level = args.log_level

    # 设置日志
    setup_logging(config.log_file, config.log_level)
    logger = logging.getLogger(__name__)

    # Web仪表盘
    dashboard_state = None
    signal_db = None

    if not args.no_web:
        from web.state_bridge import DashboardState
        from signal_db import SignalDB
        from web.app import create_app
        import uvicorn

        dashboard_state = DashboardState()
        db_path = config.state_file.replace('monitor_state.json', 'signals.db')
        signal_db = SignalDB(db_path)

        app = create_app(dashboard_state, signal_db)

        web_thread = threading.Thread(
            target=uvicorn.run,
            args=(app,),
            kwargs={
                "host": "0.0.0.0",
                "port": args.web_port,
                "log_level": "warning",
            },
            daemon=True,
        )
        web_thread.start()
        logger.info(f"Web仪表盘已启动: http://0.0.0.0:{args.web_port}")

    # 创建引擎
    engine = MonitorEngine(
        config, dry_run=args.dry_run,
        dashboard_state=dashboard_state,
        signal_db=signal_db,
    )

    # 清除状态
    if args.reset_state:
        engine.state_mgr.clear()
        logger.info("状态已清除，将冷启动")

    # 优雅关闭
    def shutdown_handler(*_args):
        logger.info("收到关闭信号...")
        engine.shutdown()
        if signal_db:
            signal_db.close()
        sys.exit(0)

    # Windows 没有 SIGTERM，用 atexit 兜底
    try:
        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)
    except (OSError, AttributeError):
        pass
    atexit.register(engine.shutdown)

    # 启动
    logger.info("=" * 50)
    logger.info("期货信号监控系统")
    logger.info(f"品种: {len(config.symbols)}个")
    logger.info(f"模式: {'干跑' if args.dry_run else '实时推送'}")
    if not args.no_web:
        logger.info(f"Web: http://0.0.0.0:{args.web_port}")
    logger.info("=" * 50)

    engine.start()


if __name__ == '__main__':
    main()
